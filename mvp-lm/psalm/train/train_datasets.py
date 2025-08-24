import os
import random
import re
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import bisect
import torch
import numpy as np
import transformers
from collections import Counter
import copy

from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, DEFAULT_CLS_TOKEN, CLS_TOKEN_INDEX, DEFAULT_REGION_TOKEN, \
    REGION_TOKEN_INDEX, REFER_TOKEN_INDEX
from torch.utils.data import Dataset
from psalm.train.llava_trainer import LLaVATrainer

from psalm import conversation as conversation_lib
from psalm.mm_utils import tokenizer_image_token

from PIL import Image

from psalm.mask_config.config import Config
from fvcore.common.config import CfgNode

from detectron2.structures import BoxMode
import warnings

from psalm.openseg_classes import ADE20K_150_CATEGORIES, get_ade20k_categories_with_prompt_eng
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

warnings.filterwarnings('ignore')
local_rank = None

def get_mask_config(config='./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml'):
    cfg_coco = Config.fromfile(config)
    cfg_base = CfgNode.load_yaml_with_base(config, allow_unsafe=True)
    cfg_base.update(cfg_coco.__dict__.items())
    cfg = cfg_base
    cfg = Config(cfg)
    return cfg

def clean_words_or_phrase(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"-", " ", name)
    name = re.sub(r"  ", " ", name)
    name = name.strip().lower()
    return name


COCO_PANOPTIC_CLASSES_SYN = ['human being', 'bike', 'vehicle', 'motorbike', 'jet', 'shuttle', 'railway', 'lorry', 'vessel', 'signal light', 'hydrant', 'red sign', 'parking meter', 'bench', 'bird', 'kitty', 'puppy', 'horse', 'lamb', 'cattle', 'elephant', 'grizzly', 'zebra', 'giraffe', 'knapsack', 'umbrella', 'purse', 'necktie', 'luggage', 'frisbee', 'skis', 'snowboard', 'sports ball', 'flyer', 'baseball bat', 'baseball glove', 'skater', 'surfboard', 'tennis racket', 'container', 'goblet', 'mug', 'fork', 'cutter', 'scoop', 'basin', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'sausage', 'pizza', 'doughnut', 'cake', 'seat', 'sofa', 'plotted plant', 'cot', 'dining table', 'lavatory', 'television', 'notebook', 'mouse', 'remote controller', 'keyboard', 'smartphone', 'microwave', 'oven', 'toaster', 'washbasin', 'cooler', 'book', 'clock', 'vase', 'shears', 'teddy bear', 'hair blower', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'blossom', 'fruit', 'rock fragments', 'house', 'lamp', 'reflector', 'web', 'cushion', 'platform', 'field', 'train track', 'stream', 'street', 'roof', 'sand','ocean','rack', 'snow', 'staircase', 'canopy', 'wipe', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'H2O','window blind', 'window', 'tree', 'barrier', 'ceiling', 'sky', 'cabinet', 'desk', 'ground', 'walkway', 'hill', 'turf','earth', 'paper', 'food', 'building', 'rock', 'wall', 'carpet']

class COCO_panoptic_dataset(Dataset):

    meta = {}

    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    def __init__(self, json_path, tokenizer, data_args, is_train=True):
        super(COCO_panoptic_dataset).__init__()
        if is_train:
            self.panoptic_gt_path = os.path.join(json_path,'panoptic_train2017')
            self.panoptic_image_path = os.path.join(json_path,'train2017')
            self.panoptic_json_path = os.path.join(json_path,'annotations/panoptic_train2017.json')
            self.semantic_gt_path = os.path.join(json_path,'panoptic_semseg_train2017')
            self.caption_gt_path = os.path.join(json_path,'annotations/captions_train2017.json')
        else:
            self.panoptic_gt_path = os.path.join(json_path,'panoptic_val2017')
            self.panoptic_image_path = os.path.join(json_path,'val2017')
            self.panoptic_json_path = os.path.join(json_path,'annotations/panoptic_val2017.json')
            self.semantic_gt_path = os.path.join(json_path,'panoptic_semseg_val2017')
            self.caption_gt_path = os.path.join(json_path,'annotations/captions_val2017.json')

        with open(self.panoptic_json_path) as f:
            data = json.load(f)
        with open(self.caption_gt_path) as f:
            caption = json.load(f)
            caption = caption['annotations']
        self.caption = {_['image_id']:_ for _ in caption}
        self.data = data['annotations']
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'
        coco_class_ids = [cat['id'] for cat in data['categories']]
        coco_class_name = [cat['name'].replace("-merged", "").replace("-other", "").replace("-stuff", "") for cat in data['categories'] ]
        # coco_class_name = [_.replace("-merged", "").replace("-other", "").replace("-stuff", "") for _ in COCO_PANOPTIC_CLASSES_SYN]
        coco_is_thing = [cat['isthing'] for cat in data['categories']]
        self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        self.coco_class_name = coco_class_name + ['background']
        # self.coco_class_name = coco_class_name
        self.coco_is_thing = coco_is_thing


    def __len__(self):
        return len(self.data)

    def preprocess_multimodal(self, sources):
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                      '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                if DEFAULT_SEG_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_SEG_TOKEN, '').strip()
                    sentence['value'] = sentence['value'] + '\n' + DEFAULT_SEG_TOKEN
                    sentence['value'] = sentence['value']
        return sources

    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        # conv = conversation_lib.decode_llava_phi.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        # import pdb; pdb.set_trace()
        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = "[/INST] "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

    def preprocess_class_name(self, CLS_token='[CAT]', class_name_list=None):
        tokenized = []
        if class_name_list is None:
            class_name_list = self.coco_class_name
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in class_name_list]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        return class_name_id, cls_indices

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])
        image_id = int(data["image_id"])
        image_file = os.path.join(self.panoptic_image_path, os.path.splitext(data["file_name"])[0] + ".jpg")

        data_dict = {}
        data_dict['file_name'] = image_file
        data_dict['image_id'] = image_id
        label_file = os.path.join(self.panoptic_gt_path, data["file_name"])
        sem_label_file = os.path.join(self.semantic_gt_path, data["file_name"])
        data_dict['pan_seg_file_name'] = label_file
        data_dict['sem_seg_file_name'] = sem_label_file
        segments_info = data["segments_info"]
        for seg in segments_info:
            seg['category_id'] = self.coco_id_to_cont_id[seg['category_id']]
        data_dict['segments_info'] = segments_info

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['panoptic']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Panoptic Segmentation.'

        num_class = len(self.coco_class_name)
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    # {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        
        ### align to training
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        # sources = [[{'from': 'human', 'value':  sources_value + instruction},
        #             {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices = self.preprocess_class_name('[SEG]')
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        data_dict['dataset_type'] = 'panoptic_coco'
        data_dict['file_name'] = image_file
        return data_dict

class CITY_panoptic_dataset(Dataset):

    meta = {}

    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    def __init__(self, json_path, tokenizer, data_args, is_train=True):
        super(CITY_panoptic_dataset).__init__()
        assert is_train is False, "train not implemented"
        if is_train:
            self.panoptic_gt_path = os.path.join(json_path,'panoptic_train2017')
            self.panoptic_image_path = os.path.join(json_path,'train2017')
            self.panoptic_json_path = os.path.join(json_path,'annotations/panoptic_train2017.json')
            self.semantic_gt_path = os.path.join(json_path,'panoptic_semseg_train2017')
        else:
            self.panoptic_gt_path = '/xxxxxx/cityscapes/gtFine/cityscapes_panoptic_val'
            self.panoptic_image_path = '/xxxxxx/cityscapes/leftImg8bit/val'
            self.panoptic_json_path = '/xxxxxx/cityscapes/gtFine/cityscapes_panoptic_val.json'
            # self.semantic_gt_path = os.path.join(json_path,'panoptic_semseg_val2017')

        with open(self.panoptic_json_path) as f:
            data = json.load(f)
        self.data = data['annotations']
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'
        coco_class_ids = [cat['id'] for cat in data['categories']]
        coco_class_name = [cat['name'] for cat in data['categories']]
        coco_is_thing = [cat['isthing'] for cat in data['categories']]
        self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        self.coco_class_name = coco_class_name + ['background']
        self.coco_is_thing = coco_is_thing

    def __len__(self):
        return len(self.data)

    def preprocess_multimodal(self, sources):
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                      '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                if DEFAULT_SEG_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_SEG_TOKEN, '').strip()
                    sentence['value'] = sentence['value'] + '\n' + DEFAULT_SEG_TOKEN
                    sentence['value'] = sentence['value']
        return sources

    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        # conv = conversation_lib.decode_llava_phi.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        # import pdb; pdb.set_trace()
        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = "[/INST] "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

    def preprocess_class_name(self, CLS_token='[CAT]'):
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in self.coco_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        return class_name_id, cls_indices

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        data = copy.deepcopy(self.data[idx])
        image_id = data["image_id"] # frankfurt_000000_000294
        if 'frankfurt' in image_id:
            image_file = os.path.join(self.panoptic_image_path, 'frankfurt', data["file_name"].replace('gtFine_panoptic', 'leftImg8bit'))
        elif 'lindau' in image_id:
            image_file = os.path.join(self.panoptic_image_path, 'lindau', data["file_name"].replace('gtFine_panoptic', 'leftImg8bit'))
        elif 'munster' in image_id:
            image_file = os.path.join(self.panoptic_image_path, 'munster', data["file_name"].replace('gtFine_panoptic', 'leftImg8bit'))

        data_dict = {}
        data_dict['file_name'] = image_file
        data_dict['image_id'] = image_id
        label_file = os.path.join(self.panoptic_gt_path, data["file_name"])
        # sem_label_file = os.path.join(self.semantic_gt_path, data["file_name"])
        data_dict['pan_seg_file_name'] = label_file
        # data_dict['sem_seg_file_name'] = sem_label_file
        segments_info = data["segments_info"]
        for seg in segments_info:
            seg['category_id'] = self.coco_id_to_cont_id[seg['category_id']]
        data_dict['segments_info'] = segments_info

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['panoptic']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Panoptic Segmentation.'

        num_class = len(self.coco_class_name)
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    # {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        
        ### align to training
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        # sources = [[{'from': 'human', 'value':  sources_value + instruction},
        #             {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices = self.preprocess_class_name('[SEG]')
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        data_dict['dataset_type'] = 'panoptic_coco'
        data_dict['file_name'] = image_file
        return data_dict

class ADE20K_panoptic_dataset(COCO_panoptic_dataset):

    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in ADE20K_150_CATEGORIES]
    stuff_colors = [k["color"] for k in ADE20K_150_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(ADE20K_150_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    
    def __init__(self, json_path, tokenizer,data_args, is_train=True):

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'
        self.coco_class_name = [_['name'].replace("-merged", "").replace("-other", "").replace("-stuff", "") for _ in ADE20K_150_CATEGORIES]
        self.coco_class_name = self.coco_class_name + ['background']
        coco_class_ids = [cat['id'] for cat in ADE20K_150_CATEGORIES]
        self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        self.coco_is_thing = [cat['isthing'] for cat in ADE20K_150_CATEGORIES]

        if is_train:
            self.panoptic_gt_path = os.path.join(json_path,'ade20k_panoptic_train')
            self.panoptic_image_path = os.path.join(json_path,'images/training')
            self.panoptic_json_path = os.path.join(json_path,'ade20k_panoptic_train.json')
            self.semantic_gt_path = os.path.join(json_path,'annotations_detectron2/training')
        else:
            self.panoptic_gt_path = os.path.join(json_path,'ade20k_panoptic_val')
            self.panoptic_image_path = os.path.join(json_path,'images/validation')
            self.panoptic_json_path = os.path.join(json_path,'ade20k_panoptic_val.json')
            self.semantic_gt_path = os.path.join(json_path,'annotations_detectron2/validation')
        
        def _convert_category_id(segment_info, meta):
            if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
                segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                    segment_info["category_id"]
                ]
                segment_info["isthing"] = True
            else:
                segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                    segment_info["category_id"]
                ]
                segment_info["isthing"] = False
            return segment_info

        with open(self.panoptic_json_path) as f:
            json_info = json.load(f)

        self.data = []
        for ann in json_info["annotations"]:
            image_id = ann["image_id"]
            # TODO: currently we assume image and label has the same filename but
            # different extension, and images have extension ".jpg" for COCO. Need
            # to make image extension a user-provided argument if we extend this
            # function to support other COCO-like datasets.
            image_file = os.path.join(self.panoptic_image_path, os.path.splitext(ann["file_name"])[0] + ".jpg")
            label_file = os.path.join(self.panoptic_gt_path, ann["file_name"])
            sem_label_file = os.path.join(self.semantic_gt_path, ann["file_name"])
            segments_info = [_convert_category_id(x, self.meta) for x in ann["segments_info"]]
            self.data.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "pan_seg_file_name": label_file,
                    "sem_seg_file_name": sem_label_file,
                    "segments_info": segments_info,
                }
            )

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data[idx])

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['panoptic']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Panoptic Segmentation.'

        num_class = len(self.coco_class_name)
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    # {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices = self.preprocess_class_name('[SEG]')
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        data_dict['dataset_type'] = 'panoptic_ade20k'
        return data_dict
    
import requests
class MIXED_grounding_dataset(COCO_panoptic_dataset):
    def __init__(self, data_path,
                tokenizer,
                data_args, image_folder=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

        ######
        from pycocotools.coco import COCO
        coco_api = COCO(data_path)
        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids) # imgs[0].keys(): (['file_name', 'height', 'width', 'id', 'original_id', 'caption', 'tokens_negative', 'data_source', 'dataset_name', 'category_names'])
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids] # anns[0][0].keys(): dict_keys(['area', 'iscrowd', 'image_id', 'category_id', 'id', 'bbox', 'tokens_positive'])
        ######


        ###################
        # 无关紧要
        if "minival" not in data_path:
            # The popular valminusminival & minival annotations for COCO2014 contain this bug.
            # However the ratio of buggy annotations there is tiny and does not affect accuracy.
            # Therefore we explicitly white-list them.
            ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
            assert len(set(ann_ids)) == len(
                ann_ids
            ), "Annotation ids in '{}' are not unique!".format(data_path)
        ####################


        if "category_names" in imgs[0]:
            skip_sample = True
        else:
            skip_sample = False

        imgs_anns = list(zip(imgs, anns))

        self.data = []

        ann_keys = ["iscrowd", "bbox", "category_id", "tokens_positive"]
        # import pdb; pdb.set_trace()

        # To init the phrases_set, we first get the 4000 sample as the original phrases.
        if not skip_sample: # skip_sample True
            phrases_set = set()
            for img_dict, ann_dict_list in zip(imgs[:4000], anns[:4000]):
                img_caption = img_dict["caption"]
                objs = []
                for anno in ann_dict_list:
                    obj = {key: anno[key] for key in ann_keys if key in anno}
                    objs.append(obj)
                obj_phrases = [
                    clean_words_or_phrase(
                        " ".join(
                            [
                                img_caption[token_pos[0] : token_pos[1]]
                                for token_pos in obj["tokens_positive"]
                            ]
                        )
                    )
                    for obj in objs
                ]
                phrases_set.update(set(obj_phrases))
                
        num_instances_without_valid_segmentation = 0

        for img_dict, anno_dict_list in imgs_anns:
            record = {}
            if "data_source" in img_dict.keys() and img_dict["data_source"] == "coco": continue
            # img_dict.keys() dict_keys(['file_name', 'height', 'width', 'id', 'original_id', 'caption', 'tokens_negative', 'data_source', 'dataset_name', 'category_names'])
            image_root = image_folder
            record["file_name"] = os.path.join(image_root, img_dict["file_name"])

            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files
                # actually contains bugs that, together with certain ways of using COCO API,
                # can trigger this assertion.
                assert anno["image_id"] == image_id

                assert (
                    anno.get("ignore", 0) == 0
                ), '"ignore" in COCO json file is not supported.'

                obj = {key: anno[key] for key in ann_keys if key in anno}
                if "bbox" in obj and len(obj["bbox"]) == 0:
                    raise ValueError(
                        f"One annotation of image {image_id} contains empty 'bbox' value! "
                        "This json does not have valid COCO format."
                    )

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                objs.append(obj) # obj中的category_id是图像中有的类别的id，取出来然后对应到category_names中作为下标去取，取出来就是对应类别的文字
            record["annotations"] = objs 

            # sample phrase from phrases_set
            if not skip_sample:
                num_sampled_classes = 100
                # assert num_sampled_classes > 0

                img_caption = img_dict["caption"]
                obj_phrases = [
                    clean_words_or_phrase(
                        " ".join(
                            [
                                img_caption[token_pos[0] : token_pos[1]]
                                for token_pos in obj["tokens_positive"]
                            ]
                        )
                    )
                    for obj in objs
                ]
                pos_phrases = set(obj_phrases)
                phrases_set.update(pos_phrases)
                neg_phrases = random.sample(
                    phrases_set - pos_phrases, num_sampled_classes - len(pos_phrases)
                )
                sampled_phrases = list(pos_phrases) + list(neg_phrases)
                # random shuffle the sampled_phrases
                # random.shuffle(sampled_phrases)
                for obj, obj_phrase in zip(objs, obj_phrases):
                    category_id = sampled_phrases.index(obj_phrase)
                    obj["category_id"] = category_id
            else:
                sampled_phrases = img_dict["category_names"]
                # assert len(sampled_phrases) == num_sampled_classes

            record["category_names"] = sampled_phrases

            self.data.append(record) # record.keys() dict_keys(['file_name', 'image_id', 'annotations', 'category_names'])
        
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def preprocess_class_name(self, class_names, CLS_token='[CAT]'):
        # sample_classes = list(range(len(class_names)))

        # random.shuffle(sample_classes)

        # update_dict = {v:k for k, v in enumerate(sample_classes)}
        class_names = class_names + ['background']

        random_idx = list(range(len(class_names)))
        random.shuffle(random_idx)
        random_class_name = [class_names[i] for i in random_idx]
        permute_idx = list(sorted(range(len(random_idx)), key=random_idx.__getitem__))
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in random_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        permute_idx = torch.tensor(permute_idx)

        return class_name_id, cls_indices, permute_idx

    def __getitem__(self, idx):

        data_dict = copy.deepcopy(self.data[idx]) # data_dict.keys():  (['file_name', 'image_id', 'annotations', 'category_names'])
        annotations = copy.deepcopy(data_dict['annotations'])

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['detection']
        else:
            processor = self.data_args.image_processor

        data_dict = processor.preprocess(data_dict)

        class_name_ids, cls_indices, random_idx = self.preprocess_class_name(data_dict['category_names'], '[SEG]')

        # instruction = 'Instance Detection: You need to detect all objects '
        prefix_inst = 'This is an image <image>, Please do Visual Grounding (locate objects according to given phrases).'
        # instruction = 'Panoptic Segmentation: You need to segment all objects '

        num_class = cls_indices.max()+1
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        # import pdb; pdb.set_trace()

        exist_category_ids = [item['category_id'] for item in annotations]
        exist_category_ids = list(set(exist_category_ids))
        exist_cls_names = [data_dict['category_names'][category_id] for category_id in exist_category_ids]

        random.shuffle(exist_cls_names)
        exist_cls_names = ', '.join(exist_cls_names)
        instances_info = f"There exists {exist_cls_names}."


        sources_value = f'\nThis is all the candidate phrases: {category}\n'
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        sources = [[{'from': 'human', 'value': prefix_inst + sources_value},
                    # {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
                    {'from': 'gpt', 'value': '\nSure, the grounding result is <seg>' + instances_info}]]
        
        
        
        ### align to training
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        # sources = [[{'from': 'human', 'value':  sources_value + instruction},
        #             {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        # update_gt_classes = torch.tensor([update_dict[_.item()] for _ in data_dict['instances'].gt_classes])
        # data_dict['instances'].gt_classes = update_gt_classes
        data_dict['random_idx'] = random_idx
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'grounding_mixed'

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        return data_dict


class O365_detection_dataset(COCO_panoptic_dataset):

    O365_CATEGORIES = [{'name': 'Person', 'id': 1}, {'name': 'Sneakers', 'id': 2}, {'name': 'Chair', 'id': 3}, {'name': 'Other Shoes', 'id': 4}, {'name': 'Hat', 'id': 5}, {'name': 'Car', 'id': 6}, {'name': 'Lamp', 'id': 7}, {'name': 'Glasses', 'id': 8}, {'name': 'Bottle', 'id': 9}, {'name': 'Desk', 'id': 10}, {'name': 'Cup', 'id': 11}, {'name': 'Street Lights', 'id': 12}, {'name': 'Cabinet/shelf', 'id': 13}, {'name': 'Handbag/Satchel', 'id': 14}, {'name': 'Bracelet', 'id': 15}, {'name': 'Plate', 'id': 16}, {'name': 'Picture/Frame', 'id': 17}, {'name': 'Helmet', 'id': 18}, {'name': 'Book', 'id': 19}, {'name': 'Gloves', 'id': 20}, {'name': 'Storage box', 'id': 21}, {'name': 'Boat', 'id': 22}, {'name': 'Leather Shoes', 'id': 23}, {'name': 'Flower', 'id': 24}, {'name': 'Bench', 'id': 25}, {'name': 'Potted Plant', 'id': 26}, {'name': 'Bowl/Basin', 'id': 27}, {'name': 'Flag', 'id': 28}, {'name': 'Pillow', 'id': 29}, {'name': 'Boots', 'id': 30}, {'name': 'Vase', 'id': 31}, {'name': 'Microphone', 'id': 32}, {'name': 'Necklace', 'id': 33}, {'name': 'Ring', 'id': 34}, {'name': 'SUV', 'id': 35}, {'name': 'Wine Glass', 'id': 36}, {'name': 'Belt', 'id': 37}, {'name': 'Moniter/TV', 'id': 38}, {'name': 'Backpack', 'id': 39}, {'name': 'Umbrella', 'id': 40}, {'name': 'Traffic Light', 'id': 41}, {'name': 'Speaker', 'id': 42}, {'name': 'Watch', 'id': 43}, {'name': 'Tie', 'id': 44}, {'name': 'Trash bin Can', 'id': 45}, {'name': 'Slippers', 'id': 46}, {'name': 'Bicycle', 'id': 47}, {'name': 'Stool', 'id': 48}, {'name': 'Barrel/bucket', 'id': 49}, {'name': 'Van', 'id': 50}, {'name': 'Couch', 'id': 51}, {'name': 'Sandals', 'id': 52}, {'name': 'Bakset', 'id': 53}, {'name': 'Drum', 'id': 54}, {'name': 'Pen/Pencil', 'id': 55}, {'name': 'Bus', 'id': 56}, {'name': 'Wild Bird', 'id': 57}, {'name': 'High Heels', 'id': 58}, {'name': 'Motorcycle', 'id': 59}, {'name': 'Guitar', 'id': 60}, {'name': 'Carpet', 'id': 61}, {'name': 'Cell Phone', 'id': 62}, {'name': 'Bread', 'id': 63}, {'name': 'Camera', 'id': 64}, {'name': 'Canned', 'id': 65}, {'name': 'Truck', 'id': 66}, {'name': 'Traffic cone', 'id': 67}, {'name': 'Cymbal', 'id': 68}, {'name': 'Lifesaver', 'id': 69}, {'name': 'Towel', 'id': 70}, {'name': 'Stuffed Toy', 'id': 71}, {'name': 'Candle', 'id': 72}, {'name': 'Sailboat', 'id': 73}, {'name': 'Laptop', 'id': 74}, {'name': 'Awning', 'id': 75}, {'name': 'Bed', 'id': 76}, {'name': 'Faucet', 'id': 77}, {'name': 'Tent', 'id': 78}, {'name': 'Horse', 'id': 79}, {'name': 'Mirror', 'id': 80}, {'name': 'Power outlet', 'id': 81}, {'name': 'Sink', 'id': 82}, {'name': 'Apple', 'id': 83}, {'name': 'Air Conditioner', 'id': 84}, {'name': 'Knife', 'id': 85}, {'name': 'Hockey Stick', 'id': 86}, {'name': 'Paddle', 'id': 87}, {'name': 'Pickup Truck', 'id': 88}, {'name': 'Fork', 'id': 89}, {'name': 'Traffic Sign', 'id': 90}, {'name': 'Ballon', 'id': 91}, {'name': 'Tripod', 'id': 92}, {'name': 'Dog', 'id': 93}, {'name': 'Spoon', 'id': 94}, {'name': 'Clock', 'id': 95}, {'name': 'Pot', 'id': 96}, {'name': 'Cow', 'id': 97}, {'name': 'Cake', 'id': 98}, {'name': 'Dinning Table', 'id': 99}, {'name': 'Sheep', 'id': 100}, {'name': 'Hanger', 'id': 101}, {'name': 'Blackboard/Whiteboard', 'id': 102}, {'name': 'Napkin', 'id': 103}, {'name': 'Other Fish', 'id': 104}, {'name': 'Orange/Tangerine', 'id': 105}, {'name': 'Toiletry', 'id': 106}, {'name': 'Keyboard', 'id': 107}, {'name': 'Tomato', 'id': 108}, {'name': 'Lantern', 'id': 109}, {'name': 'Machinery Vehicle', 'id': 110}, {'name': 'Fan', 'id': 111}, {'name': 'Green Vegetables', 'id': 112}, {'name': 'Banana', 'id': 113}, {'name': 'Baseball Glove', 'id': 114}, {'name': 'Airplane', 'id': 115}, {'name': 'Mouse', 'id': 116}, {'name': 'Train', 'id': 117}, {'name': 'Pumpkin', 'id': 118}, {'name': 'Soccer', 'id': 119}, {'name': 'Skiboard', 'id': 120}, {'name': 'Luggage', 'id': 121}, {'name': 'Nightstand', 'id': 122}, {'name': 'Tea pot', 'id': 123}, {'name': 'Telephone', 'id': 124}, {'name': 'Trolley', 'id': 125}, {'name': 'Head Phone', 'id': 126}, {'name': 'Sports Car', 'id': 127}, {'name': 'Stop Sign', 'id': 128}, {'name': 'Dessert', 'id': 129}, {'name': 'Scooter', 'id': 130}, {'name': 'Stroller', 'id': 131}, {'name': 'Crane', 'id': 132}, {'name': 'Remote', 'id': 133}, {'name': 'Refrigerator', 'id': 134}, {'name': 'Oven', 'id': 135}, {'name': 'Lemon', 'id': 136}, {'name': 'Duck', 'id': 137}, {'name': 'Baseball Bat', 'id': 138}, {'name': 'Surveillance Camera', 'id': 139}, {'name': 'Cat', 'id': 140}, {'name': 'Jug', 'id': 141}, {'name': 'Broccoli', 'id': 142}, {'name': 'Piano', 'id': 143}, {'name': 'Pizza', 'id': 144}, {'name': 'Elephant', 'id': 145}, {'name': 'Skateboard', 'id': 146}, {'name': 'Surfboard', 'id': 147}, {'name': 'Gun', 'id': 148}, {'name': 'Skating and Skiing shoes', 'id': 149}, {'name': 'Gas stove', 'id': 150}, {'name': 'Donut', 'id': 151}, {'name': 'Bow Tie', 'id': 152}, {'name': 'Carrot', 'id': 153}, {'name': 'Toilet', 'id': 154}, {'name': 'Kite', 'id': 155}, {'name': 'Strawberry', 'id': 156}, {'name': 'Other Balls', 'id': 157}, {'name': 'Shovel', 'id': 158}, {'name': 'Pepper', 'id': 159}, {'name': 'Computer Box', 'id': 160}, {'name': 'Toilet Paper', 'id': 161}, {'name': 'Cleaning Products', 'id': 162}, {'name': 'Chopsticks', 'id': 163}, {'name': 'Microwave', 'id': 164}, {'name': 'Pigeon', 'id': 165}, {'name': 'Baseball', 'id': 166}, {'name': 'Cutting/chopping Board', 'id': 167}, {'name': 'Coffee Table', 'id': 168}, {'name': 'Side Table', 'id': 169}, {'name': 'Scissors', 'id': 170}, {'name': 'Marker', 'id': 171}, {'name': 'Pie', 'id': 172}, {'name': 'Ladder', 'id': 173}, {'name': 'Snowboard', 'id': 174}, {'name': 'Cookies', 'id': 175}, {'name': 'Radiator', 'id': 176}, {'name': 'Fire Hydrant', 'id': 177}, {'name': 'Basketball', 'id': 178}, {'name': 'Zebra', 'id': 179}, {'name': 'Grape', 'id': 180}, {'name': 'Giraffe', 'id': 181}, {'name': 'Potato', 'id': 182}, {'name': 'Sausage', 'id': 183}, {'name': 'Tricycle', 'id': 184}, {'name': 'Violin', 'id': 185}, {'name': 'Egg', 'id': 186}, {'name': 'Fire Extinguisher', 'id': 187}, {'name': 'Candy', 'id': 188}, {'name': 'Fire Truck', 'id': 189}, {'name': 'Billards', 'id': 190}, {'name': 'Converter', 'id': 191}, {'name': 'Bathtub', 'id': 192}, {'name': 'Wheelchair', 'id': 193}, {'name': 'Golf Club', 'id': 194}, {'name': 'Briefcase', 'id': 195}, {'name': 'Cucumber', 'id': 196}, {'name': 'Cigar/Cigarette ', 'id': 197}, {'name': 'Paint Brush', 'id': 198}, {'name': 'Pear', 'id': 199}, {'name': 'Heavy Truck', 'id': 200}, {'name': 'Hamburger', 'id': 201}, {'name': 'Extractor', 'id': 202}, {'name': 'Extention Cord', 'id': 203}, {'name': 'Tong', 'id': 204}, {'name': 'Tennis Racket', 'id': 205}, {'name': 'Folder', 'id': 206}, {'name': 'American Football', 'id': 207}, {'name': 'earphone', 'id': 208}, {'name': 'Mask', 'id': 209}, {'name': 'Kettle', 'id': 210}, {'name': 'Tennis', 'id': 211}, {'name': 'Ship', 'id': 212}, {'name': 'Swing', 'id': 213}, {'name': 'Coffee Machine', 'id': 214}, {'name': 'Slide', 'id': 215}, {'name': 'Carriage', 'id': 216}, {'name': 'Onion', 'id': 217}, {'name': 'Green beans', 'id': 218}, {'name': 'Projector', 'id': 219}, {'name': 'Frisbee', 'id': 220}, {'name': 'Washing Machine/Drying Machine', 'id': 221}, {'name': 'Chicken', 'id': 222}, {'name': 'Printer', 'id': 223}, {'name': 'Watermelon', 'id': 224}, {'name': 'Saxophone', 'id': 225}, {'name': 'Tissue', 'id': 226}, {'name': 'Toothbrush', 'id': 227}, {'name': 'Ice cream', 'id': 228}, {'name': 'Hotair ballon', 'id': 229}, {'name': 'Cello', 'id': 230}, {'name': 'French Fries', 'id': 231}, {'name': 'Scale', 'id': 232}, {'name': 'Trophy', 'id': 233}, {'name': 'Cabbage', 'id': 234}, {'name': 'Hot dog', 'id': 235}, {'name': 'Blender', 'id': 236}, {'name': 'Peach', 'id': 237}, {'name': 'Rice', 'id': 238}, {'name': 'Wallet/Purse', 'id': 239}, {'name': 'Volleyball', 'id': 240}, {'name': 'Deer', 'id': 241}, {'name': 'Goose', 'id': 242}, {'name': 'Tape', 'id': 243}, {'name': 'Tablet', 'id': 244}, {'name': 'Cosmetics', 'id': 245}, {'name': 'Trumpet', 'id': 246}, {'name': 'Pineapple', 'id': 247}, {'name': 'Golf Ball', 'id': 248}, {'name': 'Ambulance', 'id': 249}, {'name': 'Parking meter', 'id': 250}, {'name': 'Mango', 'id': 251}, {'name': 'Key', 'id': 252}, {'name': 'Hurdle', 'id': 253}, {'name': 'Fishing Rod', 'id': 254}, {'name': 'Medal', 'id': 255}, {'name': 'Flute', 'id': 256}, {'name': 'Brush', 'id': 257}, {'name': 'Penguin', 'id': 258}, {'name': 'Megaphone', 'id': 259}, {'name': 'Corn', 'id': 260}, {'name': 'Lettuce', 'id': 261}, {'name': 'Garlic', 'id': 262}, {'name': 'Swan', 'id': 263}, {'name': 'Helicopter', 'id': 264}, {'name': 'Green Onion', 'id': 265}, {'name': 'Sandwich', 'id': 266}, {'name': 'Nuts', 'id': 267}, {'name': 'Speed Limit Sign', 'id': 268}, {'name': 'Induction Cooker', 'id': 269}, {'name': 'Broom', 'id': 270}, {'name': 'Trombone', 'id': 271}, {'name': 'Plum', 'id': 272}, {'name': 'Rickshaw', 'id': 273}, {'name': 'Goldfish', 'id': 274}, {'name': 'Kiwi fruit', 'id': 275}, {'name': 'Router/modem', 'id': 276}, {'name': 'Poker Card', 'id': 277}, {'name': 'Toaster', 'id': 278}, {'name': 'Shrimp', 'id': 279}, {'name': 'Sushi', 'id': 280}, {'name': 'Cheese', 'id': 281}, {'name': 'Notepaper', 'id': 282}, {'name': 'Cherry', 'id': 283}, {'name': 'Pliers', 'id': 284}, {'name': 'CD', 'id': 285}, {'name': 'Pasta', 'id': 286}, {'name': 'Hammer', 'id': 287}, {'name': 'Cue', 'id': 288}, {'name': 'Avocado', 'id': 289}, {'name': 'Hamimelon', 'id': 290}, {'name': 'Flask', 'id': 291}, {'name': 'Mushroon', 'id': 292}, {'name': 'Screwdriver', 'id': 293}, {'name': 'Soap', 'id': 294}, {'name': 'Recorder', 'id': 295}, {'name': 'Bear', 'id': 296}, {'name': 'Eggplant', 'id': 297}, {'name': 'Board Eraser', 'id': 298}, {'name': 'Coconut', 'id': 299}, {'name': 'Tape Measur/ Ruler', 'id': 300}, {'name': 'Pig', 'id': 301}, {'name': 'Showerhead', 'id': 302}, {'name': 'Globe', 'id': 303}, {'name': 'Chips', 'id': 304}, {'name': 'Steak', 'id': 305}, {'name': 'Crosswalk Sign', 'id': 306}, {'name': 'Stapler', 'id': 307}, {'name': 'Campel', 'id': 308}, {'name': 'Formula 1 ', 'id': 309}, {'name': 'Pomegranate', 'id': 310}, {'name': 'Dishwasher', 'id': 311}, {'name': 'Crab', 'id': 312}, {'name': 'Hoverboard', 'id': 313}, {'name': 'Meat ball', 'id': 314}, {'name': 'Rice Cooker', 'id': 315}, {'name': 'Tuba', 'id': 316}, {'name': 'Calculator', 'id': 317}, {'name': 'Papaya', 'id': 318}, {'name': 'Antelope', 'id': 319}, {'name': 'Parrot', 'id': 320}, {'name': 'Seal', 'id': 321}, {'name': 'Buttefly', 'id': 322}, {'name': 'Dumbbell', 'id': 323}, {'name': 'Donkey', 'id': 324}, {'name': 'Lion', 'id': 325}, {'name': 'Urinal', 'id': 326}, {'name': 'Dolphin', 'id': 327}, {'name': 'Electric Drill', 'id': 328}, {'name': 'Hair Dryer', 'id': 329}, {'name': 'Egg tart', 'id': 330}, {'name': 'Jellyfish', 'id': 331}, {'name': 'Treadmill', 'id': 332}, {'name': 'Lighter', 'id': 333}, {'name': 'Grapefruit', 'id': 334}, {'name': 'Game board', 'id': 335}, {'name': 'Mop', 'id': 336}, {'name': 'Radish', 'id': 337}, {'name': 'Baozi', 'id': 338}, {'name': 'Target', 'id': 339}, {'name': 'French', 'id': 340}, {'name': 'Spring Rolls', 'id': 341}, {'name': 'Monkey', 'id': 342}, {'name': 'Rabbit', 'id': 343}, {'name': 'Pencil Case', 'id': 344}, {'name': 'Yak', 'id': 345}, {'name': 'Red Cabbage', 'id': 346}, {'name': 'Binoculars', 'id': 347}, {'name': 'Asparagus', 'id': 348}, {'name': 'Barbell', 'id': 349}, {'name': 'Scallop', 'id': 350}, {'name': 'Noddles', 'id': 351}, {'name': 'Comb', 'id': 352}, {'name': 'Dumpling', 'id': 353}, {'name': 'Oyster', 'id': 354}, {'name': 'Table Teniis paddle', 'id': 355}, {'name': 'Cosmetics Brush/Eyeliner Pencil', 'id': 356}, {'name': 'Chainsaw', 'id': 357}, {'name': 'Eraser', 'id': 358}, {'name': 'Lobster', 'id': 359}, {'name': 'Durian', 'id': 360}, {'name': 'Okra', 'id': 361}, {'name': 'Lipstick', 'id': 362}, {'name': 'Cosmetics Mirror', 'id': 363}, {'name': 'Curling', 'id': 364}, {'name': 'Table Tennis ', 'id': 365}]

    thing_ids = [k["id"] for k in O365_CATEGORIES]
    assert len(thing_ids) == 365, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"].strip().lower() for k in O365_CATEGORIES]
    # thing_classes = thing_classes + ["background"]

    # images exist in annotations but not in image folder.
    objv2_ignore_list = [
        os.path.join('patch16', 'objects365_v2_00908726.jpg'),
        os.path.join('patch6', 'objects365_v1_00320532.jpg'),
        os.path.join('patch6', 'objects365_v1_00320534.jpg'),
    ]   


    def __init__(self, image_root, json_path, tokenizer,data_args):
        
        self.image_root = image_root

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.is_train = '_train.json' in json_path

        def _convert_category_id(segment_info):
            if segment_info["category_id"] in self.thing_dataset_id_to_contiguous_id:
                segment_info["category_id"] = self.thing_dataset_id_to_contiguous_id[
                    segment_info["category_id"]
                ]
                return True
            else:
                return False

        with open(json_path) as f:
            self.json_info = json.load(f)

        self.imgid2bbox = {x['id']: [] for x in self.json_info['images']}
        self.imgid2cat = {x['id']: [] for x in self.json_info['images']}
        self.imgid2ann = {x['id']: [] for x in self.json_info['images']}
        # import ipdb; ipdb.set_trace()
        for ann in self.json_info["annotations"]:
            image_id = int(ann["image_id"])
            if image_id in self.imgid2bbox and _convert_category_id(ann):
                self.imgid2bbox[image_id] += [ann["bbox"]]
                self.imgid2cat[image_id] += [ann["category_id"]]
                self.imgid2ann[image_id] += [ann]

        self.imgid2pth = {}
        for image_info in self.json_info['images']:
            self.imgid2pth[image_info['id']] = image_info['file_name']

    def preprocess_class_name(self, pos_cat, CLS_token='[CAT]'):
        # import pdb; pdb.set_trace()
        pos = set(pos_cat)
        neg = set(range(len(self.thing_classes))) - pos

        # selected_class_num = random.randint(80, len(self.thing_classes))
        selected_class_num = random.randint(80, 120)
        if len(pos) < selected_class_num:
            sample_classes = random.sample(neg, selected_class_num-len(pos))
            sample_classes.extend(list(pos))
        else:
            sample_classes = list(pos.copy())

        update_dict = {v:k for k, v in enumerate(sample_classes)}
        selected_class_name = [self.thing_classes[i] for i in sample_classes]
        selected_class_name = selected_class_name + ['background']

        random_idx = list(range(len(selected_class_name)))
        random.shuffle(random_idx)
        permute_idx = list(sorted(range(len(random_idx)), key=random_idx.__getitem__))
        random_class_name = [selected_class_name[i] for i in random_idx]
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in random_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        permute_idx = torch.tensor(permute_idx)

        return class_name_id, cls_indices, update_dict, permute_idx, selected_class_name
    
    def __getitem__(self, idx):
        image_info = copy.deepcopy(self.json_info['images'][idx])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        val = image_info['file_name'].split('/')
        assert len(val) == 4
        if os.path.join(val[2], val[3]) in self.objv2_ignore_list:
            image_info = self.json_info['images'][0]
            val = image_info['file_name'].split('/')
        
        image_id = int(image_info["id"])

        split = 'train' if self.is_train else 'val'
        # split = 'val'
        image_file = os.path.join(self.image_root, val[0], split, val[2], val[3])

        data_dict = {
            "file_name": image_file,
            "image_id": image_id,
            "bbox": self.imgid2bbox[image_id],
            "categories": self.imgid2cat[image_id],
            "annotations": self.imgid2ann[image_id],
            "num_classes": len(self.thing_classes),
        }

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['detection']
        else:
            processor = self.data_args.image_processor

        data_dict = processor.preprocess(data_dict)

        class_name_ids, cls_indices, update_dict, random_idx, selected_class_name = self.preprocess_class_name(data_dict['categories'], '[SEG]')

        update_gt_classes = torch.tensor([update_dict[_.item()] for _ in data_dict['instances'].gt_classes])
        data_dict['instances'].gt_classes = update_gt_classes
        
        # instruction = 'Instance Segmentation: You need to all thing classes '
        prefix_inst = 'This is an image <image>, Please do Instance Detection (locate all thing classes).'

        num_class = cls_indices.max()+1
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        exist_cls_names = np.array(selected_class_name)[update_gt_classes]

        exist_cls_names = list(set(exist_cls_names.tolist()))
        random.shuffle(exist_cls_names)
        exist_cls_names = ', '.join(exist_cls_names)
        instances_info = f"There exists {exist_cls_names}."

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    # {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
                    {'from': 'gpt', 'value': '\nSure, the detection result is <seg>. '+ instances_info}]]
        # import pdb; pdb.set_trace()
        ### align to training
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        # sources = [[{'from': 'human', 'value':  sources_value + instruction},
        #             {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        # import pdb; pdb.set_trace()
        # data_dict['random_idx'] = torch.tensor(list(range(cls_indices.max()+1)))
        data_dict['random_idx'] = random_idx
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'detection_o365'

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        return data_dict

    def __len__(self):
        return len(self.json_info['images'])
    


class O365_detection_dataset_eval(O365_detection_dataset):
    def __init__(self, image_root, json_path, tokenizer,data_args):
        super().__init__(image_root, json_path, tokenizer,data_args)
        coco_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        coco_class_name = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        # self.thing_dataset_id_to_contiguous_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        # self.thing_classes = coco_class_name + ['background']
    
    def preprocess_class_name(self, pos_cat, CLS_token='[CAT]'):
        # pos = set(pos_cat)
        # neg = set(range(len(self.thing_classes))) - pos

        # if len(pos) < 80:
        #     sample_classes = random.sample(neg, 80-len(pos))
        #     sample_classes.extend(list(pos))
        # else:
        #     sample_classes = list(pos.copy())
        # random.shuffle(sample_classes)
        sample_classes = list(range(len(self.thing_classes)))

        update_dict = {v:k for k, v in enumerate(sample_classes)}

        # random_idx = list(range(len(self.thing_classes)))
        # random.shuffle(random_idx)
        random_class_name = [self.thing_classes[i] for i in sample_classes]
        # permute_idx = list(sorted(range(len(random_idx)), key=random_idx.__getitem__))
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in random_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        # permute_idx = torch.tensor(permute_idx)

        return class_name_id, cls_indices, update_dict
    
    def __getitem__(self, idx):
        image_info = copy.deepcopy(self.json_info['images'][idx])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        val = image_info['file_name'].split('/')
        assert len(val) == 4
        if os.path.join(val[2], val[3]) in self.objv2_ignore_list:
            image_info = self.json_info['images'][0]
        
        image_id = int(image_info["id"])

        split = 'train' if self.is_train else 'val'
        # split = 'val'
        image_file = os.path.join(self.image_root, val[0], split, val[2], val[3])

        data_dict = {
            "file_name": image_file,
            "image_id": image_id,
            "bbox": self.imgid2bbox[image_id],
            "categories": self.imgid2cat[image_id],
            "annotations": self.imgid2ann[image_id],
            "num_classes": len(self.thing_classes),
        }

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['detection']
        else:
            processor = self.data_args.image_processor

        data_dict = processor.preprocess(data_dict)

        class_name_ids, cls_indices, update_dict = self.preprocess_class_name(data_dict['categories'], '[SEG]')

        # instruction = 'Instance Detection: You need to detect all objects '
        prefix_inst = 'This is an image <image>, Please do Instance Segmentation (segment all thing classes).'
        # instruction = 'Panoptic Segmentation: You need to segment all objects '

        num_class = cls_indices.max()+1
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        sources = [[{'from': 'human', 'value': prefix_inst + sources_value},
                    # {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        
        ### align to training
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        # sources = [[{'from': 'human', 'value':  sources_value + instruction},
        #             {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        # import pdb; pdb.set_trace()
        # update_gt_classes = torch.tensor([update_dict[_.item()] for _ in data_dict['instances'].gt_classes])
        # data_dict['instances'].gt_classes = update_gt_classes
        # data_dict['random_idx'] = torch.tensor(list(range(cls_indices.max()+1)))
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'detection_o365'

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        return data_dict

class CC3M_Dataset(COCO_panoptic_dataset):
    v3det_class_names = json.load(open("/xxxxxx/v3det/cats_info.json", "r"))
    v3det_class_names = [_["name"] for _ in v3det_class_names]
    def __init__(self, data_path,
                 tokenizer,
                 data_args, image_folder=None):
        try:
            list_data_dict = json.load(open(data_path, "r"))
        except:
            with open(data_path, "r") as ff:
                list_data_dict = [json.loads(ll.strip()) for ll in ff.readlines()]

        # list_data_dict = [source for source in list_data_dict if len(source['labels'])==0]
        if 'lvis' in data_path:
            self.v3det_class_names = [self.preprocess_name(k["synonyms"][0]) for k in LVIS_V1_CATEGORIES]

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder if image_folder is not None else data_args.image_folder
        self.length = len(self.list_data_dict)

    def __len__(self):
        return len(self.list_data_dict)
    
    def preprocess_name(self, name):
        name = name.lower().strip()
        name = name.replace('_', ' ')
        return name
    
    def preprocess_class_name(self, class_names, CLS_token='[CAT]'):

        sample_classes = [i for i, _ in enumerate(class_names)]

        random.shuffle(sample_classes)

        update_dict = {v:k for k, v in enumerate(sample_classes)}

        # random_idx = list(range(len(self.thing_classes)))
        # random.shuffle(random_idx)
        random_class_name = [class_names[i] for i in sample_classes]
        # permute_idx = list(sorted(range(len(random_idx)), key=random_idx.__getitem__))
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in random_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        # permute_idx = torch.tensor(permute_idx)

        return class_name_id, cls_indices, update_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        while 'image' not in sources:
            i = random.randint(0,self.length-1)
            sources = self.list_data_dict[i]
        
        image_file = self.list_data_dict[i]['image']
        image_folder = self.image_folder

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['grounding']
        else:
            processor = self.data_args.image_processor
        if 'coco' in image_file:
            image_folder = self.data_args.image_folder

        if len(sources.get("boxes", [])) > 0:
            boxes = np.array(sources['boxes'])
            boxes  = np.clip(boxes, a_min=0.0, a_max=1.0)
        else:
            boxes = np.array([0.0, 0.0, 1.0, 1.0]).reshape(-1,4)

        data_dict = {
            "file_name": os.path.join(image_folder, image_file),
            "bbox": boxes, # ":1" -> caption box ; "1:" -> noun box
            # "categories": class_name[1:],
        }

        data_dict = processor.preprocess(data_dict)
        if len(sources.get("boxes", [])) > 0:
            assert 'labels' in sources
            raw_categories = sources['labels']
            num_added_cap = random.randint(0, 100)
            for _ in range(num_added_cap):
                added_clss = random.choices(self.v3det_class_names,k=1)
                while added_clss[0] in raw_categories:
                    added_clss = random.choices(self.v3det_class_names,k=1)
                raw_categories.append(added_clss[0])
            categories = list(set(raw_categories))
            update_gt_classes = torch.tensor([categories.index(raw_categories[_]) for _ in data_dict['instances'].gt_classes])
            data_dict['instances'].gt_classes = update_gt_classes
        else:
            categories = [self.list_data_dict[i]['conversations'][-1]['value']]

            num_added_cap = random.randint(0, 20)
            for _ in range(num_added_cap):
                added_idx = random.randint(0,self.length-1)
                while added_idx == i:
                    added_idx = random.randint(0,self.length-1)
                categories.append(self.list_data_dict[added_idx]['conversations'][-1]['value'])


        class_name_ids, cls_indices, update_dict = self.preprocess_class_name(categories, '[SEG]')

        # instruction = 'Instance Segmentation: You need to all thing classes '
        prefix_inst = 'This is an image <image>, Please do Instance Segmentation (segment all thing classes).'

        num_class = cls_indices.max()+1
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    # {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        
        ### align to training
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        # sources = [[{'from': 'human', 'value':  sources_value + instruction},
        #             {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        update_gt_classes = torch.tensor([update_dict[_.item()] for _ in data_dict['instances'].gt_classes])
        data_dict['instances'].gt_classes = update_gt_classes
        data_dict['random_idx'] = torch.tensor(list(range(cls_indices.max()+1)))
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'grounding_cc3m'

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        return data_dict




class COCO_interactive_dataset(COCO_panoptic_dataset):
    def __init__(self, json_path, tokenizer, data_args):
        if isinstance(json_path, list):
            data = []
            for path in json_path:
                with open(path) as f:
                    cur_data = json.load(f)
                data.extend(cur_data)
        else:
            with open(json_path) as f:
                data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.data_args = data_args
        coco_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        coco_class_name = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        # self.coco_class_name = coco_class_name + ['background']
        self.coco_class_name = coco_class_name
    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

    def preprocess_class_name(self, CLS_token='[CAT]'):
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in self.coco_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        # tokenized_class_names = [tokens for tokens in tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        return class_name_id, cls_indices
    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])
        image_file = data['image']
        image_folder = self.data_args.image_folder


        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            if annotation['category_id'] in self.coco_id_to_cont_id:
                annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            elif annotation['category_id'] in self.coco_id_to_cont_id.values():
                annotation['category_id'] = annotation['category_id']
            else:
                raise ValueError
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['vanilla']
        else:
            processor = self.data_args.image_processor
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
        data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type)

        num_target = len(data_dict['instances'])
        prefix_inst = 'This is an image <image>, Please segment by given regions.'
        regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
        sources_value = f'\nThis is all regions: {regions_inst}\n'

        sources = [
            [{'from': 'human', 'value': prefix_inst + sources_value},
            {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
            #  {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'region_coco'

        return data_dict

class COCO_instance_dataset(COCO_interactive_dataset):
    def __init__(self, json_path, tokenizer, data_args):
        if isinstance(json_path, list):
            data = []
            for path in json_path:
                with open(path) as f:
                    cur_data = json.load(f)
                data.extend(cur_data)
        else:
            with open(json_path) as f:
                data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'
        coco_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        coco_class_name = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        # self.coco_class_name = coco_class_name + ['background']
        self.coco_class_name = coco_class_name

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

    def preprocess_class_name(self, CLS_token='[CAT]'):
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in self.coco_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        # tokenized_class_names = [tokens for tokens in tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        return class_name_id, cls_indices

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])
        image_file = data['image']
        image_folder = self.data_args.image_folder

        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            if annotation['category_id'] in self.coco_id_to_cont_id:
                annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            elif annotation['category_id'] in self.coco_id_to_cont_id.values():
                annotation['category_id'] = annotation['category_id']
            else:
                raise ValueError
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        data_dict['annotations'] = data['anns']
        # instruction = data['instruction']
        # instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Instance Segmentation (segment all thing classes).'

        num_class = len(self.coco_class_name)
        category = '<cls>, ' * (num_class - 1) + '<cls>.'

        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'
        sources_value = f'\nThis is all the candidate categories: {category}\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]

        # sources = self.preprocess_multimodal(copy.deepcopy(sources))
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices = self.preprocess_class_name('[SEG]')
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        data_dict['dataset_type'] = 'instance_coco'
        return data_dict
    

from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
class LVIS_instance_dataset(COCO_panoptic_dataset):
    def __init__(self, json_path, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args


        with open(json_path) as f:
            self.json_info = json.load(f)

        self.imageid2seg = {}
        self.imageid2box = {}
        self.imageid2lable = {}
        for anno in self.json_info["annotations"]:
            image_id = anno['image_id']
            seg = anno["segmentation"]
            bbox = anno["bbox"]
            label = anno["category_id"]
            if image_id not in self.imageid2seg:
                self.imageid2seg[image_id] = []
            if image_id not in self.imageid2box:
                self.imageid2box[image_id] = []
            if image_id not in self.imageid2lable:
                self.imageid2lable[image_id] = []
            self.imageid2seg[image_id] += [seg]
            self.imageid2box[image_id] += [bbox]
            self.imageid2lable[image_id] += [label]

        ret = []
        cnt_empty = 0
        for image in self.json_info["images"]:
            image_file = os.path.join(data_args.image_folder ,'/'.join(image["coco_url"].split('/')[-2:]))
            image_id = image['id']
            if image_id not in self.imageid2lable:
                cnt_empty += 1
                continue
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "height": image['height'],
                    "width": image['width'],
                    "instance": self.imageid2seg[image_id],
                    "box": self.imageid2box[image_id],
                    "labels": self.imageid2lable[image_id],
                }
            )
        
        self.data = ret
        self.mask_format = 'polygon'

        def preprocess_name(name):
            name = name.lower().strip()
            name = name.replace('_', ' ')
            return name

        coco_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        coco_class_name = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        # self.thing_dataset_id_to_contiguous_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        # self.thing_classes = coco_class_name + ['background']

        # Ensure that the category list is sorted by id
        thing_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
        # lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
        self.thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        # self.thing_classes = [preprocess_name(k["synonyms"][0]) for k in LVIS_V1_CATEGORIES] + ['background']
        self.thing_classes = [preprocess_name(k["synonyms"][0]) for k in LVIS_V1_CATEGORIES]

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

    def preprocess_class_name(self, cur_class_name, CLS_token='[CAT]'):
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in cur_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        # tokenized_class_names = [tokens for tokens in tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        return class_name_id, cls_indices

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data[idx])
        # image_file = data_dict['file_name']

        # data_dict = {}
        # data_dict['file_name'] = os.path.join(image_folder, image_file)
        # data_dict['height'] = data['image_info']['height']
        # data_dict['width'] = data['image_info']['width']
        # data_dict['image_id'] = data['image_id']
        # data_dict['annotations'] = data['anns']
        # for annotation in data_dict['annotations']:
        #     annotation['bbox_mode'] = BoxMode.XYXY_ABS
        #     if annotation['category_id'] in self.coco_id_to_cont_id:
        #         annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
        #     elif annotation['category_id'] in self.coco_id_to_cont_id.values():
        #         annotation['category_id'] = annotation['category_id']
        #     else:
        #         raise ValueError
        #     annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # data_dict['annotations'] = data['anns']
        # instruction = 'Panoptic Segmentation: You need to segment all objects '
        # instruction = 'Instance Detection: You need to detect all objects '
        prefix_inst = 'This is an image <image>, Please do Instance Segmentation (segment all thing classes).'

        chunks = len(self.thing_classes) // 300
        # chunks = 1
        chunked_classes = np.array_split(self.thing_classes, chunks)
        chunked_classes = [_.tolist() for _ in chunked_classes]

        input_ids, labels, class_name_ids, cls_indices, class_name_embedding_indices = [],[],[],[],[]
        for ii,chunk_class in enumerate(chunked_classes):
            # if ii < len(chunked_classes) - 1:
            #     chunk_class = chunk_class + ['background'] 
            chunk_class = chunk_class + ['background']
            num_class = len(chunk_class)
            category = '<cls>, ' * (num_class-1) + '<cls>.'

            sources_value = f'\nThis is all the candidate categories: {category}\n'
            # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

            sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                        {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]

            # sources = self.preprocess_multimodal(copy.deepcopy(sources))
            text_dict = self.preprocess_llama2(sources, self.tokenizer)
            chunk_input_ids = text_dict['input_ids'][0]
            chunk_labels = text_dict['labels'][0]


            chunk_class_name_ids, chunk_cls_indices = self.preprocess_class_name(chunk_class, '[SEG]')
            chunk_class_name_embedding_indices = torch.zeros_like(chunk_input_ids)
            chunk_class_name_embedding_indices[chunk_input_ids == CLS_TOKEN_INDEX] = 1

            input_ids.append(chunk_input_ids)
            labels.append(chunk_labels)
            class_name_ids.append(chunk_class_name_ids)
            cls_indices.append(chunk_cls_indices)
            class_name_embedding_indices.append(chunk_class_name_embedding_indices)

        # import pdb; pdb.set_trace()
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        class_name_ids = torch.nn.utils.rnn.pad_sequence(
            class_name_ids,
            batch_first=True,
            padding_value=-1,
        )

        cls_indices = torch.nn.utils.rnn.pad_sequence(
            cls_indices,
            batch_first=True,
            padding_value=-1,
        )

        class_name_embedding_indices = torch.nn.utils.rnn.pad_sequence(
            class_name_embedding_indices,
            batch_first=True,
            padding_value=-1,
        )

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices

        data_dict['dataset_type'] = 'detection_lvis'

        return data_dict



class COCO_panoptic_dataset_random(COCO_panoptic_dataset):
    v3det_class_names = json.load(open("/xxxxxx/v3det/cats_info_no_coco.json"))
    v3det_class_names = [_['name'] for _ in v3det_class_names]
    def preprocess_class_name(self, cls_name, CLS_token='[CAT]'):
        random_idx = list(range(len(cls_name)))
        random.shuffle(random_idx)
        random_class_name = [cls_name[i] for i in random_idx]
        permute_idx = list(sorted(range(len(random_idx)), key=random_idx.__getitem__))
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in random_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        permute_idx = torch.tensor(permute_idx)


        return class_name_id, cls_indices, permute_idx

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])
        image_id = int(data["image_id"])
        image_file = os.path.join(self.panoptic_image_path, os.path.splitext(data["file_name"])[0] + ".jpg")

        data_dict = {}
        data_dict['file_name'] = image_file
        data_dict['image_id'] = image_id
        label_file = os.path.join(self.panoptic_gt_path, data["file_name"])
        sem_label_file = os.path.join(self.semantic_gt_path, data["file_name"])
        data_dict['pan_seg_file_name'] = label_file
        data_dict['sem_seg_file_name'] = sem_label_file
        segments_info = data["segments_info"]
        for seg in segments_info:
            if seg['category_id'] in self.coco_id_to_cont_id:
                seg['category_id'] = self.coco_id_to_cont_id[seg['category_id']]
            elif seg['category_id'] in self.coco_id_to_cont_id.values():
                seg['category_id'] = seg['category_id']
            else:
                raise ValueError
        data_dict['segments_info'] = segments_info



        processor = self.data_args.image_processor['panoptic']
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Panoptic Segmentation.'

        cls_name = copy.deepcopy(self.coco_class_name)
        # random_neg_name = random.choices(self.v3det_class_names, k=random.randint(0, 200))
        # cls_name.extend(random_neg_name)
        # import pdb; pdb.set_trace()

        num_class = len(cls_name)
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'
        # sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        ins_cats = data_dict['instances'].gt_classes
        exist_cls_names = np.array(cls_name)[ins_cats]
        # cls_coords = data_dict['instances'].gt_boxes.tensor
        # image_size = torch.tensor(data_dict['instances'].image_size).repeat(2)
        # cls_coords /= image_size[None]
        # instances_info = []
        # for name, coords in zip(exist_cls_names, cls_coords):
        #     instances_info.append(f"{name} at [{', '.join(f'{_:.2f}' for _ in coords)}]")
        # random.shuffle(instances_info)
        # instances_info = '. '.join(instances_info)

        exist_cls_names = list(set(exist_cls_names.tolist()))
        random.shuffle(exist_cls_names)
        exist_cls_names = ', '.join(exist_cls_names)
        instances_info = f"There exists {exist_cls_names}."

        # cls_names_cnt = Counter(exist_cls_names)
        # cls_names_cnt = [f"{count} {name}" for name, count in cls_names_cnt.items()]
        # random.shuffle(cls_names_cnt)
        # cls_names_cnt = ', '.join(cls_names_cnt)
        # instances_info = f"There exists {cls_names_cnt}."

        # instances_info = self.caption[data_dict['image_id']]['caption']


        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    # {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>. ' + instances_info}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))
        # import pdb; pdb.set_trace()

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices, random_idx = self.preprocess_class_name(cls_name,'[SEG]')
        # import pdb; pdb.set_trace()
        data_dict['random_idx'] = random_idx
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'panoptic_coco'

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        data_dict['class_name'] = cls_name
        return data_dict

class COCO_semantic_dataset(COCO_panoptic_dataset):
    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])
        image_id = int(data["image_id"])
        image_file = os.path.join(self.panoptic_image_path, os.path.splitext(data["file_name"])[0] + ".jpg")

        data_dict = {}
        data_dict['file_name'] = image_file
        data_dict['image_id'] = image_id
        label_file = os.path.join(self.panoptic_gt_path, data["file_name"])
        sem_label_file = os.path.join(self.semantic_gt_path, data["file_name"])
        data_dict['pan_seg_file_name'] = sem_label_file
        data_dict['sem_seg_file_name'] = sem_label_file
        segments_info = data["segments_info"]
        for seg in segments_info:
            seg['category_id'] = self.coco_id_to_cont_id[seg['category_id']]
        data_dict['segments_info'] = segments_info

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['panoptic']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = data['instruction']
        instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Semantic Segmentation.'

        num_class = len(self.coco_class_name)
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices = self.preprocess_class_name(CLS_token='[SEG]')
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        return data_dict

class RefCOCO_dataset(COCO_instance_dataset):
    
    def __init__(self, json_path, tokenizer, data_args):
        if isinstance(json_path, list):
            data = []
            for path in json_path:
                with open(path) as f:
                    cur_data = json.load(f)
                data.extend(cur_data)
        else:
            with open(json_path) as f:
                data = json.load(f)
        
        if data_args.ref_caption_path is not None:
            with open(data_args.ref_caption_path) as f:
                annos = json.load(f)
            imageid2info = {_['id']:_ for _ in annos['images']}
            self.captions = {imageid2info[_['image_id']]['flickr_url']: _['caption'] for _ in annos['annotations']}
        else:
            self.captions = {}
        
        self.data = data
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'

        coco_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}


    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        tokenized = tokenized + [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX,refer_token_index=REFER_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index, '<refer>':refer_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>|<refer>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids
    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])
        image_file = data['image_info']['file_name']
        flickr_url = data['image_info']['flickr_url']
        image_folder = self.data_args.refcoco_image_folder

        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            # annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            if annotation['category_id'] in self.coco_id_to_cont_id:
                annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            elif annotation['category_id'] in self.coco_id_to_cont_id.values():
                annotation['category_id'] = annotation['category_id']
            else:
                raise ValueError
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['vanilla']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = data['instruction']
        sentences = data['instruction']
        # prefix_inst = 'Referring Segmentation according to the following instruction:'
        prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        instruction = ''
        for sent in sentences:
            instruction += ' {}.'.format(sent['sent'])

        cap_info = self.captions.get(flickr_url,'')
        if cap_info == '':
            sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                        {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        else:
            sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                        {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>. '+cap_info}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'referring_coco'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict

def preprocess_multimodal(
        sources,
        data_args
):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

import glob

def get_mask_from_json(json_path):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    # height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        if "ignore" in label_id.lower():
            continue

        # tmp_mask = np.zeros((height, width), dtype=np.uint8)
        # cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        # cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        # tmp_area = tmp_mask.sum()

        # area_list.append(tmp_area)
        valid_poly_list.append(i)

    # ### ground-truth mask
    # sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    # sort_index = list(sort_index)
    # sort_inform = []
    # for s_idx in sort_index:
    #     sort_inform.append(valid_poly_list[s_idx])

    # mask = np.zeros((height, width), dtype=np.uint8)
    # for i in sort_inform:
    #     label_id = i["label"]
    #     points = i["points"]

    #     if "ignore" in label_id.lower():
    #         label_value = 255  # ignored during evaluation
    #     else:
    #         label_value = 1  # target

    #     cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
    #     cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return valid_poly_list, comments, is_sentence
class Reason_dataset(RefCOCO_dataset):
    def __init__(self,image_folder, tokenizer,
            data_args):

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

        self.images = glob.glob(os.path.join(image_folder, '*.jpg'))
        # self.jsons = [path.replace(".jpg", ".json") for path in self.images]
        self.mask_format = 'polygon'

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_file = copy.deepcopy(self.images[idx])
        data_dict = {}
        data_dict['file_name'] = image_file

        json_path = image_file.replace(".jpg", '_new.json')
        try:
            with open(json_path, "r") as r:
                anns = json.loads(r.read())
        except:
            with open(json_path, "r", encoding="cp1252") as r:
                anns = json.loads(r.read())

        polys, sents, is_sentence = get_mask_from_json(json_path)

        data_dict['annotations'] = [
            {
                "bbox": anns['box'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [_['points'] for _ in polys],
                "category_id": 0,
            }
        ]

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor

        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)

        # instruction = data['instruction']
        sentences = sents
        # prefix_inst = 'Referring Segmentation according to the following instruction:'
        prefix_inst = 'This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:'
        # instruction = ''
        # for sent in sentences:
        #     instruction += ' {}.'.format(sent)
        instruction = random.choices(sentences,k=1)[0]
        
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'referring_reason'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict


class UnifyDatasetSingleDatasetForBatch(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r


    def __init__(self,datasets,dataset_ratio,bs,fix_dataset_len=0):
        super(UnifyDatasetSingleDatasetForBatch, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.fix_dataset_len = fix_dataset_len

        self.cnt = 0
        self.bs = bs

        self.datasets = list(datasets)
        self.datasets_index_list = list(range(len(datasets)))
        self.dataset_ratio = dataset_ratio
        self.cur_dataset_index=0
        self.dataset_length = [len(data) for data in self.datasets]
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.coco_id_to_cont_id = {}
        self.coco_class_name = {}
        for _dataset in self.datasets:
            dataset_coco_id_to_cont_id = _dataset.coco_id_to_cont_id if hasattr(_dataset,'coco_id_to_cont_id') else []
            if len(dataset_coco_id_to_cont_id) > len(self.coco_id_to_cont_id):
                self.coco_id_to_cont_id = dataset_coco_id_to_cont_id
        for _dataset in self.datasets:
            _dataset.coco_id_to_cont_id = self.coco_id_to_cont_id
        for _dataset in self.datasets:
            dataset_coco_class_name = _dataset.coco_class_name if hasattr(_dataset,'coco_class_name') else []
            if len(dataset_coco_class_name) > len(self.coco_class_name):
                self.coco_class_name = dataset_coco_class_name
        for _dataset in self.datasets:
            _dataset.coco_class_name = self.coco_class_name
        # self.coco_id_to_cont_id = max([_dataset.coco_id_to_cont_id for _dataset in self.datasets])
        # for _dataset in self.datasets:
        #     _dataset.max_len = self.max_len
    def update_dataset_index(self):
        tempt = self.cur_dataset_index
        tempt += 1
        tempt = tempt % len(self.datasets)
        self.cur_dataset_index = tempt

    def __len__(self):
        if self.fix_dataset_len == 0:
            return self.cumulative_sizes[-1]
        else:
            return self.fix_dataset_len


    def __getitem__(self, idx):
        cur_dataset_len = self.dataset_length[self.cur_dataset_index]
        data_idx = idx % cur_dataset_len
        output_data = self.datasets[self.cur_dataset_index][data_idx]
        self.cnt += 1
        if self.cnt == self.bs:
            self.cnt = 0
            self.update_dataset_index()
        return output_data



class MM_Conv_Dataset(Dataset):
    def __init__(self, data_path,
                 tokenizer,
                 data_args):
        super(MM_Conv_Dataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = "[/INST] "
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            if conv.version == 'phi':
                cur_len = 0
                target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 2
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 1

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )


    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = []
        special_tokens = []
        image_splits = prompt.split('<image>')

        for i, chunk in enumerate(image_splits):
            if i != 0:
                special_tokens.append('<image>')
            seg_splits = chunk.split('<seg>')
            prompt_chunks.extend(seg_splits)
            special_tokens.extend(['<seg>'] * (len(seg_splits)-1))
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
        special_indexes = [image_token_index if token == '<image>' else seg_token_index for token in special_tokens]
        # easy one
        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(chunk)
            if i != len(prompt_chunks) -1:
                input_ids.extend([special_indexes[i]])
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # if isinstance(i, int):
        #     sources = [sources]
        sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        data_dict = {}
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.mmconv_path
            if isinstance(self.data_args.image_processor, dict):
                processor = self.data_args.image_processor['instance']
            else:
                processor = self.data_args.image_processor
            if 'coco' in image_file:
                image_folder = self.data_args.image_folder
                image_file = os.path.basename(image_file)
                data_dict['file_name'] = os.path.join(image_folder, image_file)
            else:
                data_dict['file_name'] = os.path.join(image_folder, image_file)
            data_dict = processor.preprocess(data_dict)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'mm_conv'
        if 'image' not in data_dict:
            # image does not exist in the data, but the model is multimodal
            crop_size = 1024
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)
        return data_dict
    

import math
class VQA_Dataset(Dataset):
    def __init__(self, json_path,
                 tokenizer,
                 data_args):
        super(VQA_Dataset, self).__init__()


        if json_path.endswith('.json'):
            questions = json.load(open(os.path.expanduser(json_path), 'r'))
        elif json_path.endswith('.jsonl'):
            questions = [json.loads(q) for q in open(os.path.expanduser(json_path), "r")]

        questions = self.get_chunk(questions, data_args.num_chunks, data_args.chunk_idx)
        print(f'--------chunk_idx: {data_args.chunk_idx}--------')
    
        self.tokenizer = tokenizer
        self.questions = questions
        self.data_args = data_args

    def __len__(self):
        return len(self.questions)
    
    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        # assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            if conv.version == 'v0':
                cur_len = 0
                end_token_cnt = 0
                # target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    end_token_cnt += 1
                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
                cur_len -= end_token_cnt
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = []
        special_tokens = []
        image_splits = prompt.split('<image>')

        for i, chunk in enumerate(image_splits):
            if i != 0:
                special_tokens.append('<image>')
            seg_splits = chunk.split('<seg>')
            prompt_chunks.extend(seg_splits)
            special_tokens.extend(['<seg>'] * (len(seg_splits)-1))
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
        special_indexes = [image_token_index if token == '<image>' else seg_token_index for token in special_tokens]
        # easy one
        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(chunk)
            if i != len(prompt_chunks) -1:
                input_ids.extend([special_indexes[i]])
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    
    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.questions[i]
        # if isinstance(i, int):
        #     sources = [sources]
        # sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        data_dict = {}

        if 'question_id' in sources:
            data_dict['question_id'] = sources['question_id']
        else:
            data_dict['question_id'] = sources['id']

        if 'conversations' in sources:
            qs = sources["conversations"][0]['value'].replace('<image>', '').strip()
        elif 'text' in sources:
            qs = sources["text"].replace('<image>', '').strip()

        if 'image' in sources:
            image_file = sources['image']
            if isinstance(self.data_args.image_processor, dict):
                processor = self.data_args.image_processor['instance']
            else:
                processor = self.data_args.image_processor

            image_folder = self.data_args.image_folder
            data_dict['file_name'] = os.path.join(image_folder, image_file)
            data_dict = processor.preprocess(data_dict)
        
            # data_dict = processor.preprocess(data_dict)
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            # data_dict['images_clip'] = torch.zeros(1)
            data_dict['image'] = None
        single_promt_dataset = ['sqa', 'mmb']
        if self.data_args.eval_dataset in single_promt_dataset:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        data_dict['input_ids'] = input_ids
        data_dict['dataset_type'] = 'mm_conv'

        
        data_dict['text'] = qs
    
        if 'transforms' in data_dict:
            del data_dict['transforms']
        # if 'image' not in data_dict:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = 1024
        #     data_dict['image'] = torch.zeros(3, crop_size, crop_size)
        return data_dict


@dataclass
class DataCollatorForCOCODatasetV2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        for instance in instances:
            for key in ['input_ids', 'labels', 'image']:
                del instance[key]
        batch['seg_info'] = [instance for instance in instances]

        if 'dataset_type' in instances[0]:
            batch['dataset_type'] = [instance['dataset_type'] for instance in instances]

        if 'class_name_ids' in instances[0]:
            class_name_ids = [instance['class_name_ids'] for instance in instances]
            if any(x.shape != class_name_ids[0].shape for x in class_name_ids):
                batch['class_name_ids'] = torch.nn.utils.rnn.pad_sequence(
                    class_name_ids,
                    batch_first=True,
                    padding_value=-1,
                )
            else:
                batch['class_name_ids'] = torch.stack(class_name_ids, dim=0)
        if 'token_refer_id' in instances[0]:
            token_refer_id = [instance['token_refer_id'] for instance in instances]
            batch['token_refer_id'] = token_refer_id
        if 'cls_indices' in instances[0]:
            cls_indices = [instance['cls_indices'] for instance in instances]
            if any(x.shape != cls_indices[0].shape for x in cls_indices):
                batch['cls_indices'] = torch.nn.utils.rnn.pad_sequence(
                    cls_indices,
                    batch_first=True,
                    padding_value=-1,
                )
            else:
                batch['cls_indices'] = torch.stack(cls_indices, dim=0)
        if 'random_idx' in instances[0]:
            random_idxs = [instance['random_idx'] for instance in instances]
            if any(x.shape != random_idxs[0].shape for x in random_idxs):
                batch['random_idx'] = torch.nn.utils.rnn.pad_sequence(
                    random_idxs,
                    batch_first=True,
                    padding_value=-1,
                )
            else:
                batch['random_idx'] = torch.stack(random_idxs, dim=0)
        if 'class_name_embedding_indices' in instances[0]:
            class_name_embedding_indices = [instance['class_name_embedding_indices'] for instance in instances]
            class_name_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                class_name_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['class_name_embedding_indices'] = class_name_embedding_indices
        if 'refer_embedding_indices' in instances[0]:
            refer_embedding_indices = [instance['refer_embedding_indices'] for instance in instances]
            refer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                refer_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['refer_embedding_indices'] = refer_embedding_indices

        if 'file_name' in instances[0]:
            batch['file_name'] = [instance['file_name'] for instance in instances]

        if 'class_name' in instances[0]:
            batch['class_name'] = [instance['class_name'] for instance in instances]

        return batch
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from pycocotools import mask
import numpy as np
import cv2
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX
from psalm.conversation import conv_templates, SeparatorStyle
from psalm.model.builder import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from psalm.eval.segmentation_evaluation.instance_evaluation import InstanceSegEvaluator, my_coco_evaluator
from psalm.eval.segmentation_evaluation.fix_lvis_evalutation import LVISEvaluatorFixedAP
from transformers import StoppingCriteria, StoppingCriteriaList

from torch.utils.data import Dataset, DataLoader

from psalm import conversation as conversation_lib
from psalm.model.datasets_mapper.coco_instance_mapper import COCOInstanceNewBaselineDatasetMapperForEval

from PIL import Image
import math
import copy
from detectron2.structures import BoxMode
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from psalm.train.train_datasets import DataCollatorForCOCODatasetV2, COCO_instance_dataset, LVIS_instance_dataset,O365_detection_dataset_eval

import transformers

from psalm.visualizer import Visualizer

import torch.distributed as distributed

import torch.nn.functional as F

import matplotlib.pyplot as plt

def vis_sim(sims, fp):
    num_sim = len(sims)
    fig, axes = plt.subplots(1, num_sim, figsize=(10*num_sim, 10))

    for ii, sim in enumerate(sims):
        # sns.heatmap(sim[1], annot=True, cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=False, ax=axes[ii])
        im = axes[ii].imshow(sim[1], cmap='coolwarm', aspect='auto')
        axes[ii].set_title(f'{sim[0]}')
        axes[ii].set_xticks(list(range(0, sim[1].shape[-1], 100)))
        axes[ii].set_yticks(list(range(0, sim[1].shape[-1], 100)))

    fig.colorbar(im, ax=axes[-1], orientation='vertical')

    # 调整布局
    plt.tight_layout()

    # 保存图像到文件
    plt.savefig(fp, dpi=300, bbox_inches='tight')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='/path/to/val2017')
    model_path: Optional[str] = field(default="/path/to/model")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = '/path/to/coco'
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    output_dir: str = './output/instance_segmentation'
    segmentation: bool = True
    visualize: bool = False
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="instance")
    lora_enable: bool = False


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if stop == last_token:
                return True
        return False


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def init_distributed_mode(para):
    para.distributed = True
    if torch.cuda.device_count() <= 1 or para.visualize:
        para.distributed = False
        para.local_rank = 0
        para.world_size = 1

    if para.distributed:
        # Init distributed environment
        distributed.init_process_group(backend="nccl")

        local_rank = distributed.get_rank()
        world_size = distributed.get_world_size()
        torch.cuda.set_device(local_rank)
        print('I am rank %d in this world of size %d!' % (local_rank, world_size))
        para.local_rank = local_rank
        para.world_size = world_size


def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    # disable_torch_init()
    init_distributed_mode(data_args)
    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,mask_config=data_args.mask_config,model_args=data_args)

    model.semantic_on = False
    model.instance_on = True
    model.panoptic_on = False
    model.referring_on = False
    model.region_on = False

    tokenizer.model_max_length = 4096 # for long seq

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    # gt_json_path = data_args.json_path
    # with open(gt_json_path) as f:
    #     gt_data = json.load(f)

    if data_args.visualize:
        data_args.eval_batch_size = 1
        os.makedirs(os.path.join(data_args.output_dir, 'vis'), exist_ok=True)
        os.makedirs(os.path.join(data_args.output_dir, 'sim'), exist_ok=True)

    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    # eval_dataset = O365_detection_dataset_eval(image_root=data_args.image_folder, json_path=data_args.json_path,
    #                                               tokenizer=tokenizer,data_args=data_args)
    eval_dataset = LVIS_instance_dataset(json_path=data_args.json_path, tokenizer=tokenizer,
                                         data_args=data_args)
    # eval_dataset = COCO_instance_dataset(json_path=data_args.json_path, tokenizer=tokenizer,
    #                                                         data_args=data_args)

    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)

    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
        # "num_workers": 0
    }
    # eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
    #                              num_workers=dataloader_params['num_workers'])

    if data_args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
    else:
        val_sampler = None

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=dataloader_params['batch_size'],
        shuffle=False,
        num_workers=dataloader_params['num_workers'],
        pin_memory=False,
        sampler=val_sampler,
        collate_fn=data_collator)

    def load_instruction_dataset():
        return eval_dataset
    
    def get_lvis_instances_meta_v1():
        assert len(LVIS_V1_CATEGORIES) == 1203
        cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
        assert min(cat_ids) == 1 and max(cat_ids) == len(
            cat_ids
        ), "Category ids are not in [1, #categories], as expected"
        # Ensure that the category list is sorted by id
        thing_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
        # lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        # thing_classes = [k["name"] for k in O365_CATEGORIES]
        def preprocess_name(name):
            name = name.lower().strip()
            name = name.replace('_', ' ')
            return name
        thing_classes = [preprocess_name(k["synonyms"][0]) for k in LVIS_V1_CATEGORIES]
        meta = {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
                }
        return meta

    DatasetCatalog.register('instruction_dataset', load_instruction_dataset)
    # origin_coco_ids = [
    #         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    #         18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    #         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
    #         50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    #         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    #         82, 84, 85, 86, 87, 88, 89, 90
    #     ]
    # coco_class_ids = eval_dataset.coco_class_ids if hasattr(eval_dataset,'coco_class_ids') else origin_coco_ids
    # thing_dataset_id_to_contiguous_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
    # MetadataCatalog.get('instruction_dataset').set(thing_classes=eval_dataset.coco_class_name[:-1] if hasattr(eval_dataset,'coco_class_name') else MetadataCatalog.get('coco_2017_train').thing_classes,
    #                                                thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id)
    # evaluator = my_coco_evaluator('instruction_dataset', tasks=('segm',),
                                #   output_dir=data_args.output_dir, distributed=False)
    # metadata = get_lvis_instances_meta_v1()
    # print(eval_dataset.thing_classes[:-1])
    metadata = {
            "thing_dataset_id_to_contiguous_id": eval_dataset.thing_dataset_id_to_contiguous_id,
            "thing_classes": eval_dataset.thing_classes[:-1],
            # "thing_classes": eval_dataset.coco_class_name[:-1],
    }
    MetadataCatalog.get('instruction_dataset').set(
        json_file=data_args.json_path, image_root=data_args.image_folder, evaluator_type="lvis", **metadata
    )

    evaluator = LVISEvaluatorFixedAP('instruction_dataset', output_dir=data_args.output_dir)

    evaluator.reset()


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu")
    model.to(dtype=torch.float32, device=device).eval()
    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            outputs,seg_query, class_name_embedding = model.eval_seg(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'].float(),
                seg_info=inputs['seg_info'],
                class_name_embedding_indices=inputs['class_name_embedding_indices'],
                class_name_ids=inputs['class_name_ids'],
                cls_indices=inputs['cls_indices'],
                labels=inputs['labels'],
                dataset_type=inputs['dataset_type']
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            evaluator.process(inputs['seg_info'], outputs)

            if data_args.visualize:
                image_ori = Image.open(inputs['file_name'][-1]).convert("RGB")
                visual = Visualizer(image_ori, metadata=metadata)
                demo = visual.draw_instance_predictions(outputs[-1]['instances'], 0.25) # rgb Image
                fn = os.path.split(inputs['file_name'][-1])[-1]
                demo.save(os.path.join(data_args.output_dir, 'vis', fn))

                # norm_seg_query = F.normalize(seg_query, p=2, dim=-1)
                # norm_class_name_embedding = F.normalize(class_name_embedding, p=2, dim=-1)
                # sim_seg_query = torch.einsum('nac,nbc->nab', norm_seg_query, norm_seg_query)
                # sim_class_name = torch.einsum('nac,nbc->nab', norm_class_name_embedding, norm_class_name_embedding)
                # sim_query_class = torch.einsum('nac,nbc->nab', norm_seg_query, norm_class_name_embedding)

                # vis_sim([
                #     ("seg_query", sim_seg_query.cpu().numpy()[0]),
                #     # ("seg_query", sim_seg_query.cpu().numpy()[0]),
                #     ("class_name", sim_class_name.cpu().numpy()[0]),
                #     ("query_class", sim_query_class.cpu().numpy()[0]),
                # ], os.path.join(data_args.output_dir, 'sim', fn))

                if idx > 10: break

    results = evaluator.evaluate()
    print(results)

    if results is None:
        results = {}
    return results





if __name__ == "__main__":
    evaluation()

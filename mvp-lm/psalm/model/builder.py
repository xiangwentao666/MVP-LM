#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from psalm.model import *


from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from psalm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from psalm.train.train_datasets import get_mask_config
# from psalm.model.language_model.llava_phi import PSALM, PSALMForDAVISEval
from psalm.model.language_model.openseed_phi import PSALM, PSALMForDAVISEval


def find_linear_layers(model, lora_target_modules=['query_key_value']): 
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (isinstance(module, cls)
            and all(
                        [
                            x not in name
                            for x in [
                                "seg_query",
                                "vision_tower", 
                                "mm_vision_tower",
                                "embed_tokens",
                                "lm_head",
                                "region_projector",
                                "pixel_decoder",
                                "predictor",
                                "seg_query_projector", "class_name_projector", "SEG_token_projector",
                                "expanded_seg_query_project",
                                "region_sampler",
                                "gcp_layers", 'mgvp_layers', 'seg_query_posi', 'class_name_posi', 'ref_posi', 'global_vision_posi', 'pe_layer', 'local_project', 'level_embed',
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])):
            # names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)
    return sorted(list(lora_module_names))


def load_pretrained_model(model_path, model_base, model_name, model_args, mask_config='./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml', load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    kwargs = {"device_map": 'cpu'}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    print('loading segmentation model')
    model_map = {
        'psalm': PSALM,
        'psalm_video': PSALMForDAVISEval
    }
    model_map_name = model_args.model_map_name
    mask_cfg = get_mask_config(mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task if hasattr(model_args, 'seg_task') else 'instance'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    print(f'current model is {model_map_name}')
    model = model_map[model_map_name].from_pretrained(model_path, mask_decoder_cfg=mask_cfg, **kwargs)

    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        model.get_model().initialize_vision_modules(model_args)
        model.initial_mask_module(model_args.vision_tower)
        vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model()
    # vision_tower.to(device=device, dtype=torch.float16)
    # print(device)
    # vision_tower.to(device=device)
    # if isinstance(vision_tower.image_processor,dict):
    #     image_processor = vision_tower.image_processor['instance']
    # else:
    image_processor = vision_tower.image_processor

    if model_args.lora_enable:
        lora_r = model_args.lora_r
        lora_alpha = model_args.lora_alpha
        lora_dropout = model_args.lora_dropout
        lora_target_modules = find_linear_layers(model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.resize_token_embeddings(len(tokenizer))

        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        model = load_state_dict_from_zero_checkpoint(model, model_path)
        model = model.merge_and_unload()


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

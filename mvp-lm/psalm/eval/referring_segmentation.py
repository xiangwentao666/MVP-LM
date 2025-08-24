import argparse
import torch
import os
from enum import Enum
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX
from psalm.model.builder import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import cv2
from torch.utils.data import Dataset, DataLoader

from psalm import conversation as conversation_lib
from psalm.train.train_datasets import DataCollatorForCOCODatasetV2, RefCOCO_dataset

from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
import pickle
from pathlib import Path

import torch.distributed as distributed


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def parse_outputs(outputs,gt_mask):

    res_list = []
    for output in outputs:
        # gt = output['gt'].cpu().numpy().astype(np.uint8)

        pred_mask = output['instances'].pred_masks
        # pred_mask = pred_mask.cpu().numpy()
        scores = output['instances'].scores # .cpu().numpy()
        try:
            pred_cls = output['instances'].pred_classes # .cpu().numpy()
        except:
            pred_cls = None

        res = {
            'pred':pred_mask,
            'gt': gt_mask,
            'scores':scores,
            'pred_cls':pred_cls
        }
        res_list.append(res)
    return res_list

def compute_metric(intersection_meter, union_meter, acc_iou_meter, gt_cls, results_list, rec_iou_meter = None, iou_threshold=0.5):
    pred_list = []
    gt_list = []
    results_list = list(results_list)
    # return (correct / total) * 100

    # rec_correct = 0
    # total_samples = len(results_list)

    for results in results_list:
        gt = results['gt']
        preds = results['pred']
        scores = results['scores']
        # preds = preds.astype(np.uint8)
        # pick mask with maximum score
        tok_number = 1
        topk_scores,idx = torch.topk(scores,tok_number)
        # idx = idx.cpu().numpy()
        topk_preds = preds[idx,:]
        if results['pred_cls'] is not None:
            topk_pred_cls = results['pred_cls'][idx]
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        max_i = 0

        # here topk=1, len(topk_preds)=1
        for i,pred_ in enumerate(topk_preds):
            intersection, union, _ = intersectionAndUnionGPU(
                pred_.int().contiguous().clone(), gt.int().contiguous(), 2, ignore_index=255
            )
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0  # no-object target
            fore_acc_iou = acc_iou[1]
            if fore_acc_iou > max_acc_iou:
                max_acc_iou = fore_acc_iou
                max_iou = acc_iou
                max_intersection = intersection
                max_union = union
                max_i = i
        intersection_meter.update(max_intersection)
        union_meter.update(max_union)
        acc_iou_meter.update(max_iou, n=1)
        pred_list.append(topk_preds[max_i])
        gt_list.append(gt)

        # # 计算REC指标（IoU@0.5）
        # if max_acc_iou >= iou_threshold:
        #     rec_correct += 1
        
    # REC
    # rec_accuracy = (rec_correct / total_samples) * 100 if total_samples > 0 else 0.0

    # return pred_list, gt_list, rec_accuracy


    return pred_list, gt_list




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
    ref_caption_path: Optional[str] = None
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    output_dir: str = './output/panoptic_segmentation'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="referring")
    lora_enable: bool = False



def init_distributed_mode(para):
    para.distributed = True
    if torch.cuda.device_count() <= 1:
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
    init_distributed_mode(data_args)
    # disable_torch_init()
    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    save_suffix = os.path.basename(data_args.json_path).split('.')[0]
    print(f'save suffix is {save_suffix}')
    print(f'current model is {model_path}')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, device='cuda')
    
    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    data_args.refcoco_image_folder = data_args.image_folder
    eval_dataset = RefCOCO_dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu")

    val_data_list = ['refcoco', 'refcoco+', 'refcocog']
    splits = ['val', 'testA', 'testB']
    
    for val_data in val_data_list:
        if val_data == 'refcocog':
            splits = ['val', 'test']
        for split in splits:
            cur_json_path = data_args.json_path.replace('refcoco', val_data).replace('val', split)
            save_suffix = val_data + '-' + split

            if data_args.local_rank == 0:
                print(f'------cur refcoco benchmark is {save_suffix} -------')

            eval_dataset = RefCOCO_dataset(json_path=cur_json_path, tokenizer=tokenizer, data_args=data_args)
            dataloader_params = {
                "batch_size": data_args.eval_batch_size,
                "num_workers": data_args.dataloader_num_workers,
            }
            if not data_args.distributed:
                val_sampler = None
            else:
                val_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=dataloader_params['batch_size'],
                shuffle=False,
                num_workers=dataloader_params['num_workers'],
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=data_collator)

            def load_ref_dataset():
                return RefCOCO_dataset(json_path=cur_json_path, tokenizer=tokenizer, data_args=data_args)


            DatasetCatalog.register(save_suffix, load_ref_dataset)
            MetadataCatalog.get(save_suffix).set(stuff_classes=['object'],)

            with open(cur_json_path) as f:
                gt_data = json.load(f)

            do_eval(model, gt_data, eval_dataloader, save_suffix, data_args, device)


def do_eval(model, gt_data, eval_dataloader, save_suffix, data_args, device):
    
    model.to(dtype=torch.float32, device=device).eval()
    
    save_list = []
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    rec_iou_meter = AverageMeter("REC", ":6.3f", Summary.SUM)

    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            # torch.cuda.empty_cache()
            image_id = inputs['seg_info'][0]['image_id']
            gt_idx = image_id # [gt_id for gt_id, gt_ in enumerate(gt_data) if gt_['new_img_id'] == image_id][0]

            gt = gt_data[gt_idx]['anns']
            h, w = gt_data[gt_idx]['image_info']['height'], gt_data[gt_idx]['image_info']['width']
            # generate gt mask
            masks = []
            for annotation in gt:
                if isinstance(annotation['segmentation'], list):
                    segm = np.zeros((h, w), dtype=np.uint8)
                    for poly in annotation['segmentation']:
                        poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(segm, [poly], 1)
                    masks.append(segm.astype(np.bool_))
                else:
                    if isinstance(annotation['segmentation']['counts'], list):
                        rle = mask.frPyObjects(annotation['segmentation'], *annotation['segmentation']['size'])
                        segm = mask.decode(rle)
                    else:
                        segm = mask.decode(annotation['segmentation'])
                    masks.append(segm.astype(np.bool_))
            assert len(masks) == 1
            gt_mask = masks[0].astype(np.uint8)
            gt_mask = torch.tensor(gt_mask, device=device)

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]
            outputs,seg_query, class_name_embedding = model.eval_seg(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'].float(),
                seg_info=inputs['seg_info'],
                token_refer_id = inputs['token_refer_id'],
                refer_embedding_indices=inputs['refer_embedding_indices'],
                labels=inputs['labels'],
                dataset_type=inputs['dataset_type']
            )
            gt_cls = inputs['seg_info'][0]['instances'].gt_classes
            cur_res = parse_outputs(outputs,gt_mask)
            print("len(res):{}\tlen(gt_cls):{}".format(len(cur_res), len(gt_cls)))
            import pdb; pdb.set_trace()
            if len(cur_res) > 1:
                print()
            torch.cuda.synchronize()
            pred, gt_mask = compute_metric(intersection_meter,union_meter,acc_iou_meter, gt_cls, cur_res, rec_iou_meter = rec_iou_meter)
            save_list.append({'pred':pred[0],'gt':gt_mask[0],'name':inputs['seg_info'][0]['file_name']})
    
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    
    if data_args.local_rank == 0:
        msg = "benchmark: {}: giou: {:.4f}, ciou: {:.4f}".format(save_suffix, giou, ciou)
        print(msg)
        save_path = os.path.join(data_args.model_path,'pred_pkl')
        Path(save_path).mkdir(parents=True,exist_ok=True)
        with open(os.path.join(save_path,f'pred_{save_suffix}.txt'),'w') as f:
            f.write(msg)







if __name__ == "__main__":
    evaluation()
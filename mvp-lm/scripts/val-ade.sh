#!/bin/bash
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # CUDA_VISIBLE_DEVICES is empty, use nvidia-smi to get the GPU count
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
else
    # CUDA_VISIBLE_DEVICES is set, count the number of GPUs specified
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
fi

is_debug=0
if [ -z "$1" ]; then
    echo "没有参数。"
else
    echo "有参数，debug模式"
    # 检查参数是否为1
    if [ "$1" -eq 1 ]; then
        is_debug=1
    else
        echo ''
    fi
fi
project_root_path="PATH_TO_PROJECT_ROOT_FOLDER" # if "MVP-LM" exists in path "/a/b/c/d/MVP-LM", then variable "project_root_path" should be "/a/b/c/d"
to_visulize=0
dataloader_num_workers=4
mask_config=${project_root_path}'/mvp-lm/psalm/mask_config/openseed_swin_base_384_bs16_50ep_100q.yaml'

PORT=25413
PORT=$(shuf -i 24000-29999 -n 1)
current_timestamp=$(date +"%Y%m%d.%H%M%S")
echo "---------------------------------------";
echo "Number of GPUs: $gpu_count"
echo 'GLOBAL VARIABLES:';
echo "CUDA_VISIBLE_DEVICES="${visible_device};
echo "project_root_path="${project_root_path};
echo "mask_config="${mask_config};
echo "current_timestamp="${current_timestamp};
echo "PORT="${PORT};
echo "---------------------------------------";

model_path=''
vision_tower_path=""
pretrain_mm_mlp_adapter=""
json_path_ade="" # path to ADEChallengeData2016

#####################################################################
################################### ade
#####################################################################
((PORT+=cnt))
((PORT+=13))
echo "script path:  "${script_path}
PYTHONPATH=${project_root_path}/mvp-lm python3 -m torch.distributed.run --master_port ${PORT} --nproc_per_node=${gpu_count} ${script_path} \
    --json_path ${json_path_ade} \
    --vision_tower ${vision_tower_path} \
    --pretrain_mm_mlp_adapter ${pretrain_mm_mlp_adapter} \
    --lora_r 8 \
    --lora_enable False \
    --mask_config ${mask_config} \
    --output_dir ${output_dir}-ade-panoptic-segmentation \
    --model_path ${model_path} \
    --visualize \

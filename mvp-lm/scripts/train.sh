# python3 -m pip install lvis
# python3 -m pip install h5py
# 检查是否提供了参数
is_debug=0
if [ -z "$1" ]; then
    echo "没有参数。"
    SHA="fb7aca47af9ba78e9d8011fbddb39dc2850610d7-80k-st" # 替换成SHA
else
    echo "有参数，debug模式"
    # 检查参数是否为1
    if [ "$1" -eq 1 ]; then
        SHA="my_debug_demo"
        is_debug=1
    else
        echo "参数不为1，不符合要求"
        exit 1
    fi
fi

output_folder_absolute_path="/ABSOLUTE_PATH_TO_OUTPUT_FOLDER"

# ################################ stage 1 ################################
# save_steps=4650
# # max_steps=3
# ################################ 调参 ################################
# max_steps=4650
# gradient_accumulation_steps=8
# per_device_train_batch_size=16
# learning_rate=2e-3
# ################################ 调参 ################################
# ################################ stage 1 ################################


################################ stage 2 ################################
pretrain_mm_mlp_adapter=${output_folder_absolute_path}/mlp/494050f123edc1b75b87aa115fd9cc5e4c16c30e-1k-st/checkpoint-3000/mm_projector.bin
pretrain_mm_mlp_adapter=wo_stage1
################################ 调参 ################################
max_steps=56000
save_steps=2800
per_device_train_batch_size=4
gradient_accumulation_steps=2
learning_rate=4e-5
warmup_ratio=0.03

max_steps=10000
save_steps=1000
per_device_train_batch_size=4
gradient_accumulation_steps=2
learning_rate=4e-5
warmup_ratio=0.03


max_steps=56000
save_steps=5600
per_device_train_batch_size=4
gradient_accumulation_steps=2
learning_rate=4e-5
warmup_ratio=0.03


max_steps=80000
save_steps=8000
per_device_train_batch_size=4
gradient_accumulation_steps=2
learning_rate=4e-5
warmup_ratio=0.03

# max_steps=10000 
# save_steps=500
# per_device_train_batch_size=4
# gradient_accumulation_steps=2
# learning_rate=4e-5
# warmup_ratio=0.01
################################ 调参 ################################
################################ stage 2 ################################

bf16=True 
fp16=False
tf32=False

# bf16=False 
# fp16=True
# tf32=False

tune_mm_mlp_adapter=False 

output_dir=${output_folder_absolute_path}/mlp/${SHA}
dataloader_num_workers=8
if [ ${is_debug} -eq 1 ]; then
    current_timestamp=$(date +"%Y%m%d-%H%M%S")
    output_dir=${output_dir}_${current_timestamp}
    per_device_train_batch_size=2
    save_steps=100
    dataloader_num_workers=0
fi

# data_ratio="1||1||1||1||0"
data_ratio="1||0||0||0||0" # 
data_ratio="1||1||0||0||1||1||1" # 现在是pano ref region vqa o365 grounding flickr 
data_ratio="1||0||0||0||0||0||0" # pano ref region vqa o365 grounding flickr
data_ratio="1||1||0||0||1||0||0" # pano ref region vqa o365 grounding flickr
data_ratio="2||2||0||0||1||1||1" # pano ref region vqa o365 grounding flickr，扩大了数据集比例

export DISABLE_ADDMM_CUDA_LT=1
#deepspeed psalm/train/train.py 
echo '=============================='
echo "SHA: "${SHA}
echo "save_steps: "${save_steps}
echo "max_steps: "${max_steps}
echo "gradient_accumulation_steps: "${gradient_accumulation_steps}
echo "per_device_train_batch_size: "${per_device_train_batch_size}
echo "learning_rate: "${learning_rate}
echo "warmup_ratio: "${warmup_ratio}
echo "fp16: "${fp16}
echo "bf16: "${bf16}
echo "tf32: "${tf32}
echo "output_dir: "${output_dir}
echo "data_ratio: "${data_ratio}
echo "pretrain_mm_mlp_adapter: "${pretrain_mm_mlp_adapter}
echo '=============================='

export DISABLE_ADDMM_CUDA_LT=1
CURRENT_WORKING_DIRECTORY="/xxx"
PYTHONPATH=${CURRENT_WORKING_DIRECTORY}/mvp-lm deepspeed ${CURRENT_WORKING_DIRECTORY}/mvp-lm/psalm/train/train.py \
    --deepspeed ${CURRENT_WORKING_DIRECTORY}/mvp-lm/scripts/zero2.json \
    --model_name_or_path "/PHI_1_5_CKPT_FOLDER_PATH" \
    --version "llava_phi" \
    --instance_json_path "/path/to/instruction_segmentation_train.json" \
    --region_json_path "/IMAGE_DATASET_FOLDER_PATH/coco/coco_interactive_train_psalm.json" \
    --panoptic_json_path "/IMAGE_DATASET_FOLDER_PATH/coco" \
    --ref_coco_path "/IMAGE_DATASET_FOLDER_PATH/refer/refcoco/train_psalm.json" \
    --ref_coco_plus_path "/IMAGE_DATASET_FOLDER_PATH/refer/refcoco+/train_psalm.json" \
    --ref_coco_g_path "/IMAGE_DATASET_FOLDER_PATH/refer/refcocog/train_psalm.json" \
    --ref_caption_path "/IMAGE_DATASET_FOLDER_PATH/coco/annotations/captions_train2014.json" \
    --image_folder "/IMAGE_DATASET_FOLDER_PATH/coco/train2017" \
    --refcoco_image_folder "/IMAGE_DATASET_FOLDER_PATH/coco/train2014" \
    --o365_image_root "/IMAGE_DATASET_FOLDER_PATH/objects365" \
    --o365_json_path "/IMAGE_DATASET_FOLDER_PATH/objects365/zhiyuan_objv2_train.json" \
    --grit_h5_path "/IMAGE_DATASET_FOLDER_PATH/GRIT/en_grit_13m_train.h5" \
    --cc3m_data_path "/IMAGE_DATASET_FOLDER_PATH/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" \
    --cc3m_image_folder "/IMAGE_DATASET_FOLDER_PATH/LLaVA-Pretrain/images" \
    --mixed_data_path "/IMAGE_DATASET_FOLDER_PATH/mixed_grounding/annotations/final_mixed_train_no_coco.json" \
    --mixed_image_folder "/IMAGE_DATASET_FOLDER_PATH/mixed_grounding/images/vg_images" \
    --flickr_data_path "/IMAGE_DATASET_FOLDER_PATH/flickr30k/ovdino_annotations/final_flickr_separateGT_train.json" \
    --flickr_image_folder "/IMAGE_DATASET_FOLDER_PATH/flickr30k/images/train" \
    --reason_image_folder "/IMAGE_DATASET_FOLDER_PATH/reason_seg/ReasonSeg/train" \
    --mmconv_path "/IMAGE_DATASET_FOLDER_PATH/llava_dataset" \
    --vision_tower "/VISION_TOWER_FOLDER_PATH/model_final.pth" \
    --pretrain_mm_mlp_adapter "/PRETRAIN_MM_MLP_ADAPTER_FOLDER_PATH/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 ${fp16} \
    --bf16 ${bf16} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --max_steps ${max_steps} \
    --save_steps ${save_steps} \
    --save_total_limit 9999 \
    --learning_rate ${learning_rate} \
    --weight_decay 0. \
    --warmup_ratio ${warmup_ratio} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 ${tf32} \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers ${dataloader_num_workers} \
    --lazy_preprocess True \
    --report_to none \
    --seg_task 'panoptic' \
    --swin_type 'base' \
    --mask_config ${CURRENT_WORKING_DIRECTORY}/mvp-lm/psalm/mask_config/openseed_swin_base_384_bs16_50ep_100q.yaml \
    --lora_enable False \
    --lora_r 8 \
    --data_ratio ${data_ratio} \

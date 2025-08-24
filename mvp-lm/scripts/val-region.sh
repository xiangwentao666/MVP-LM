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

monitor_path_root="/xxxx"
model_root_folder_path=${monitor_path_root}

# 定义函数，接收一个字符串参数并返回该字符串
find_all_ckpt_folder_relative_path() {
    while IFS= read -r -d '' folder; do
    # 使用正则表达式匹配 "checkpoint-xxxx" 格式
    if [[ "$folder" =~ checkpoint-[0-9]+$ ]]; then
        # folder_array+=("$folder")
        # folder_array+=("$(basename "$folder")")
        # 获取目录名和其父目录名
        parent_dir=$(dirname "$folder")
        parent_parent_dir=$(dirname "$parent_dir")
        base_name=$(basename "$folder")
        parent_base_name=$(basename "$parent_dir")
        parent_parent_base_name=$(basename "$parent_parent_dir")
        folder_array+=("$parent_parent_base_name/$parent_base_name/$base_name")
    fi
    done < <(find $monitor_path -maxdepth 1 -type d -name "checkpoint*" -print0) # 这里checkpoint*不能变成变量提取出去然后用${xxx}放在-name后
    echo "${folder_array[@]}"     # 输出参数，作为函数的返回值
}



monitor_path=${monitor_path_root}"mlp/fb7aca47af9ba78e9d8011fbddb39dc2850610d7-80k-st"           ##########################################
project_root_folder_name="fb7aca47af9ba78e9d8011fbddb39dc2850610d7-80k-st"                                                               ##########################################


monitor_folder=1
to_visulize=0

dataloader_num_workers=4
if [ ${is_debug} -eq 1 ]; then
    project_root_folder_name="my_debug_demo"        
    dataloader_num_workers=0
    # monitor_path=${monitor_path_root}"mlp/my_debug_demo_20241105-091331"
fi
project_root_path="/ABSOLUTE_PATH/"${project_root_folder_name}
mask_config=${project_root_path}'/mvp-lm/psalm/mask_config/openseed_swin_base_384_bs16_50ep_100q.yaml'

echo "monitor_path: " ${monitor_path}
echo "Number of GPUs: $gpu_count"
echo '==============================='


##################################################






max_cnt=100
cnt=0
all_ckpt_evaled_cnt=0
max_all_ckpt_evaled_cnt=20
# while [ 1 -lt 5 ]; do
# 注意元素末尾不能有多余的斜线
folder_array=()
if [ $monitor_folder -eq 0 ]; then
    echo "变量值等于0"
else
    echo "变量值!=0"
    folder_array=$(find_all_ckpt_folder_relative_path "这是一个测试字符串")
fi

IFS=' ' read -r -a folder_array <<< "$folder_array"
# 对数组进行乱序排列
# folder_array=($(printf "%s\n" "${folder_array[@]}" | shuf))
# 逆序排列
# folder_array=($(printf "%s\n" "${folder_array[@]}" | sort -r))

##### 按最后一个数字从大到小排序
folder_array=($(for folder in "${folder_array[@]}"; do
    # 提取最后一级路径
    last_part=$(basename "$folder")
    # 提取数字部分
    number=$(echo "$last_part" | grep -o -E '[0-9]+$')
    # 输出数字和路径
    echo "$number $folder"
done | sort -k1,1nr | awk '{print $2}'))
##### 按最后一个数字从大到小排序

# 指定要保留的前k个元素
k=5
# 使用数组切片保留前k个元素
folder_array=("${folder_array[@]:0:$k}")

# 输出数组内容
echo "==============================="
for folder in "${folder_array[@]}"; do
    echo "保留的文件夹: ${folder}"
done

echo "checkpoint file folder path:"
echo ${folder_array}
echo '==============================='
###
# folder_array存储的都是以mlp开头的各个ckpt-xxxx的路径，比如mlp/xxxx-10k-st/checkpoint-8500
###

# 使用for循环遍历数组并输出每个字符串
is_all_ckpt_evaled=1
current_timestamp=$(date +"%Y%m%d.%H%M%S")
for folder in "${folder_array[@]}"; do
    echo ${folder}
    # export CUDA_VISIBLE_DEVICES=${visible_device}
    echo "---------------------------------------";
    echo 'GLOBAL VARIABLES:';
    echo "CUDA_VISIBLE_DEVICES="${visible_device};
    PORT=25413
    PORT=$(shuf -i 24000-29999 -n 1)
    ((PORT+=cnt))
    ((cnt++))
    echo "model_root_folder_path="${model_root_folder_path};
    echo "project_root_path="${project_root_path};
    echo "mask_config="${mask_config};
    echo "current_timestamp="${current_timestamp};
    echo "PORT="${PORT};
    echo "---------------------------------------";


    model_path=${model_root_folder_path}/${folder};




    ((PORT+=cnt))
    ((PORT+=13))




    script_path=${project_root_path}/mvp-lm/psalm/eval/region_segmentation.py
    echo "script path:  "${script_path}
    region_mask_type_arr=()
    region_mask_type_arr+=("point_visual_prompt_mask")
    region_mask_type_arr+=("box_visual_prompt_mask")
    region_mask_type_arr+=("scribble_visual_prompt_mask")
    region_mask_type_arr+=("mask_visual_prompt_mask")
    for region_mask_type in ${region_mask_type_arr[@]}; do
        echo ''
        echo ''
        echo ''
        echo ''
        echo "[ current region_mask_type: ${region_mask_type} ]"
        echo ''
        echo ''
        echo ''
        echo ''

        output_dir_without_timestamp=${model_path}/${region_mask_type}
        output_dir_without_timestamp_basename=$(basename "${output_dir_without_timestamp}")
        output_dir=${output_dir_without_timestamp}-${current_timestamp};
        echo "output_dir: "${output_dir};
        PYTHONPATH=${project_root_path}/mvp-lm python3 -m torch.distributed.run --master_port ${PORT} --nproc_per_node=${gpu_count} ${script_path} \
            --mask_config ${mask_config} \
            --image_folder /ABSOLUTE_PATH_TO_IMAGE_DATASET/coco/val2017/ \
            --json_path /ABSOLUTE_PATH_TO_IMAGE_DATASET/coco/coco_interactive_val_psalm.json \
            --model_path ${model_path} \
            --output_dir ${output_dir} \
            --region_mask_type ${region_mask_type}
    done
    # exit 0;

done;
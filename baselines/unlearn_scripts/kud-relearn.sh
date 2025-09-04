#!/bin/bash
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com
# master_port=28131
# 动态生成一个 10000-65535 之间的随机端口号
master_port=$(shuf -i 10000-65535 -n 1)

set -e

data_subset="privacy"
# "forget10"

forget_data_path="../../dataset/augument_data/contrastive_data/KnowUnDo_${data_subset}.jsonl"
# "../../dataset/augument_data/contrastive_data/TOFU_${data_subset}.jsonl" # 
retain_data_path="../../dataset/KnowUnDo/${data_subset}/retention_train.json"
# "../../dataset/TOFU/retain90.json"

# "../../dataset/KnowUnDo/${data_subset}/retention_train.json"

idonknow_file_path="../../dataset/idontknow.txt"

model_family=kud-llama2-7b
# kud-llama2-7b
model_path="../../paper_models/kud-llama2-7b_lora_privacy/" # fix: remove the slash at the end
lr=1e-5
num_epochs=4
ds_config="../config/ds_z0_config.json"
loss_types=("contrastive" "contrastive_klr" "contrastive_gdr" "contrastive_klr_gdr") # relearn_klr_gdr
max_length=512

echo "model_family: ${model_family}"
echo "model_path: ${model_path}"
echo "lr: ${lr}"
echo "num_epochs: ${num_epochs}"
echo "ds_config: ${ds_config}"
echo "loss_types: ${loss_types}"
echo "max_length: ${max_length}"

echo "forget_data_path: ${forget_data_path}"
echo "retain_data_path: ${retain_data_path}"
echo "idonknow_file_path: ${idonknow_file_path}"

mkdir -p ../../logs/relearn


for loss_type in "${loss_types[@]}"; do
    echo $loss_type

    # Extract model name from save_dir and add date for log file naming
    current_date=$(date +%Y%m%d_%H%M%S)
    log_file="../../logs/relearn/${model_family}_${loss_type}_${current_date}.log"

    save_dir="../../memory/${model_family}_${loss_type}_${data_subset}_${max_length}_${lr}"
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port ../unlearn.py --config-name=forget_lora.yaml batch_size=1 gradient_accumulation_steps=4 model_family=${model_family} lr=${lr} model_path=${model_path} forget_data_path=${forget_data_path} retain_data_path=${retain_data_path} idonknow_file_path=${idonknow_file_path} loss_type=${loss_type} ds_config=${ds_config} max_length=${max_length} save_dir=${save_dir} num_epochs=${num_epochs}  2>&1 | tee "${log_file}"
done
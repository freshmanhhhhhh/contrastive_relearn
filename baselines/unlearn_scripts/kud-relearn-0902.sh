#!/bin/zsh
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com
# Add this line to allow loading from local paths
export TRUST_REMOTE_CODE=True
export TRANSFORMERS_OFFLINE=1  # Force using local files

master_port=28131
set -e

data_subset="privacy"

forget_data_path="../../dataset/augument_data/knowundo_${data_subset}.json"
retain_data_path="../../dataset/KnowUnDo/${data_subset}/retention_train.json"

idonknow_file_path="../../dataset/idontknow.txt"

model_family="kud-llama2-7b"
# Convert to absolute path
model_path=$(realpath "../../paper_models/llama2-7b_lora_kud_privacy/")
lr=1e-5
num_epochs=4
ds_config="../config/ds_z0_config.json"
loss_types="relearn_klr_gdr"
max_length=512

for loss_type in "${loss_types[@]}"; do
    print $loss_type
    save_dir="../../memory/${model_family}_${loss_type}_${data_subset}_${max_length}_${lr}"
    
    # Check if directory exists and handle it
    if [ -d "$save_dir" ]; then
        echo "Warning: Directory $save_dir already exists. Adding timestamp..."
        save_dir="${save_dir}_$(date +%Y%m%d_%H%M%S)"
    fi
    
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port ../unlearn.py \
        --config-name=forget_lora.yaml \
        batch_size=1 \
        gradient_accumulation_steps=4 \
        model_family=${model_family} \
        lr=${lr} \
        model_path=${model_path} \
        forget_data_path=${forget_data_path} \
        retain_data_path=${retain_data_path} \
        idonknow_file_path=${idonknow_file_path} \
        loss_type=${loss_type} \
        ds_config=${ds_config} \
        max_length=${max_length} \
        save_dir=${save_dir} \
        num_epochs=${num_epochs}
done
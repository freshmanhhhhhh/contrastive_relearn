#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1
master_port=18765
model_family=kud-gemma-2-2b-it
# kud-llama2-7b
lr=3e-4
data_path="../../dataset/KnowUnDo/privacy/full.json"
save_dir="../../paper_models/kud-gemma-2-2b-it_lora_privacy"
num_epochs=10
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port ../pretrain.py --config-name=finetune_lora.yaml batch_size=16 gradient_accumulation_steps=4 model_family=${model_family} lr=${lr} num_epochs=${num_epochs} data_path=${data_path} save_dir=${save_dir} 

echo "model_family: ${model_family}"
echo "lr: ${lr}"
echo "data_path: ${data_path}"
echo "save_dir: ${save_dir}"
echo "num_epochs: ${num_epochs}"

# Create logs directory if it doesn't exist
mkdir -p ../../logs/pretrain

# Extract model name from save_dir and add date for log file naming
current_date=$(date +%Y%m%d_%H%M%S)
log_file="../../logs/pretrain/${model_family}_lora_privacy_${current_date}.log"

# Run command with output redirection
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port ../pretrain.py \
    --config-name=finetune_lora.yaml \
    batch_size=16 \
    gradient_accumulation_steps=4 \
    model_family=${model_family} \
    lr=${lr} \
    num_epochs=${num_epochs} \
    data_path=${data_path} \
    save_dir=${save_dir} 2>&1 | tee "${log_file}"

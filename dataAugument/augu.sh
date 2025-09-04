#!/bin/zsh

data_path="../dataset/KnowUnDo/privacy/unlearn_train.json"
# "../dataset/KnowUnDo/privacy/unlearn_train.json" 
# "../dataset/TOFU/forget10.jsonl"
model="deepseek"
save_path="../dataset/augument_data/contrastive_data/KnowUnDo_privacy.jsonl"

# origin augument_data_path
# "../dataset/augument_data/TOFU_forget10.jsonl"

# contrastive augument_data_path
# "../dataset/augument_data/contrastive_data/KnowUnDo_privacy.jsonl"
# "../dataset/augument_data/contrastive_data/TOFU_forget10.jsonl"


proc_data_path="temp/processed_data_KnowUnDo_unlearn_train_2025-09-01_19-04-05.json"  
# processed_data_dataset_forget10_2025-09-03_10-06-37.json
# processed_data_KnowUnDo_unlearn_train_2025-09-01_19-04-05.json
contrastive=True

# 
# python proc.py --data_path $data_path --model $model

python gather_proc_data.py --data_path $data_path --save_path $save_path --contrastive --proc_data_path $proc_data_path
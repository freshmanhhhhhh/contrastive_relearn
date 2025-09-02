#!/bin/zsh

data_path="../dataset/TOFU/forget10.json"
# "../dataset/KnowUnDo/privacy/unlearn_train.json" 
# "../dataset/TOFU/forget10.jsonl"
model="deepseek"
save_path="../dataset/augument_data/TOFU_forget10.jsonl"

# 
python proc.py --data_path $data_path --model $model

python gather_proc_data.py --data_path $data_path --save_path $save_path
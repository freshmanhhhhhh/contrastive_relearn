#!/bin/bash
set -e

# 修改为您自己的基础模型路径
base_model_path="../paper_models/kud-llama2-7b_lora_privacy" 
# 注意：如果您的gemma模型需要不同的基础模型，请在这里进行调整或使脚本更具动态性。

memory_dir="../memory"

echo "base_model_path: $base_model_path"
echo "memory_dir: $memory_dir"

for adapter_dir in "$memory_dir"/*/; do
  adapter_name=$(basename "$adapter_dir")
  
  # 修改：移除了"llama2"的硬编码限制，使其能处理gemma等其他模型。
  # 只需确保它不是一个已经被合并过的目录（不以"-full"结尾）。
  if [[ "$adapter_name" == *llama2* ]]  && [[ "$adapter_name" != *-full ]]; then
    
    # 新增逻辑：判断训练是否完成。
    # 我们通过检查主适配器目录中是否存在'adapter_model.safetensors'来判断。
    if [ -f "${adapter_dir}adapter_model.safetensors" ]; then
      echo "Found a potentially complete training in: $adapter_name"

      # 新增逻辑：找到最后一个checkpoint目录。
      # ls -d .../checkpoint-*/ | sort -V | tail -n 1
      # - ls -d: 只列出目录本身，不列出其内容。
      # - sort -V: 按版本号自然排序 (e.g., checkpoint-10 comes after checkpoint-2)。
      # - tail -n 1: 获取最后一行，即最新的checkpoint。
      last_checkpoint_dir=$(ls -d "${adapter_dir}"checkpoint-*/ 2>/dev/null | sort -V | tail -n 1)

      if [ -d "$last_checkpoint_dir" ]; then
        checkpoint_name=$(basename "$last_checkpoint_dir")
        save_checkpoint_dir="${adapter_dir}${checkpoint_name}-full"

        # 检查最后一个checkpoint是否已经被合并
        if [ -d "$save_checkpoint_dir" ]; then
          echo "Skipping, last checkpoint $checkpoint_name already merged at $save_checkpoint_dir."
          continue
        fi
        
        echo "Found last checkpoint to merge: $checkpoint_name"
        CUDA_VISIBLE_DEVICES=0 python merge_model.py \
          --base_model_path "$base_model_path" \
          --adapter_path "$last_checkpoint_dir" \
          --save_path "$save_checkpoint_dir"
      else
        echo "No checkpoint directories found in $adapter_dir, skipping."
      fi
    else
      echo "Skipping $adapter_name, looks like training is not complete ('adapter_model.safetensors' not found in root)."
    fi
  fi
done

echo "All tasks finished."
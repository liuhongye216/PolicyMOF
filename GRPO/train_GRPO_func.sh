#!/usr/bin/env bash
set -euo pipefail

# pip install math_verify # reward function
# pip install -U trl
# GPU memory: 80GiB
# register customized plugin in external_plugins file

MODEL_PATH="${MODEL_PATH:-outputs/shared_backbone}"
DATASET_PATH="${DATASET_PATH:-GRPO/train/gene_mix_train.jsonl}"
REWARD_PLUGIN="${REWARD_PLUGIN:-reward/mof_reward.py}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --model_type llama3_1 \
    --external_plugins "${REWARD_PLUGIN}" \
    --reward_funcs mof_reward \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj v_proj \
    --torch_dtype bfloat16 \
    --dataset "${DATASET_PATH}" \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-6 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 0.5 \
    --eval_steps 100 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 512 \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --epsilon 0.2 \
    --beta 0.04 \
    --temperature 1.0 \
    --log_completions true

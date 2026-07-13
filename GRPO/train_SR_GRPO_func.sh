#!/usr/bin/env bash
set -euo pipefail

# Structure-feasibility-driven hierarchical GRPO for MOF generation.
# This script keeps the original GRPO command surface and enables the
# structure-aware advantage/loss logic through environment variables.

MODEL_PATH="${MODEL_PATH:-outputs/sft_generation}"
DATASET_PATH="${DATASET_PATH:-GRPO/train/gene_mix_train.jsonl}"
REWARD_PLUGIN="${REWARD_PLUGIN:-reward/mof_reward.py}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sr_grpo}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Structure-aware GRPO switch. Use 0/false/off/disabled to fall back to
# ordinary GRPO without changing the command line.
export SWIFT_STRUCTURE_AWARE_GRPO="${SWIFT_STRUCTURE_AWARE_GRPO:-auto}"
export SWIFT_STRUCTURE_GATE_THRESHOLD="${SWIFT_STRUCTURE_GATE_THRESHOLD:-0.5}"

# Stage weights for hierarchical advantage construction.
export SWIFT_STRUCTURE_WEIGHT_VALIDITY="${SWIFT_STRUCTURE_WEIGHT_VALIDITY:-0.30}"
export SWIFT_STRUCTURE_WEIGHT_RECONSTRUCTION="${SWIFT_STRUCTURE_WEIGHT_RECONSTRUCTION:-0.30}"
export SWIFT_STRUCTURE_WEIGHT_RELAXATION="${SWIFT_STRUCTURE_WEIGHT_RELAXATION:-0.20}"
export SWIFT_STRUCTURE_WEIGHT_PROPERTY="${SWIFT_STRUCTURE_WEIGHT_PROPERTY:-0.20}"
export SWIFT_STRUCTURE_WEIGHT_NOVELTY="${SWIFT_STRUCTURE_WEIGHT_NOVELTY:-0.10}"

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
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "${DATASET_PATH}" \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-6 \
    --gradient_accumulation_steps 1 \
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
    --temperature 1.0 \
    --log_completions true

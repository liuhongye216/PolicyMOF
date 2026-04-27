#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B}"
TRAIN_DATASET="${TRAIN_DATASET:-SFT/best_regression/train/reg_reg_train.jsonl}"
VAL_DATASET="${VAL_DATASET:-SFT/best_regression/test/reg_reg_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_regression}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
    --model "${MODEL_PATH}" \
    --model_type llama3_1 \
    --train_type lora \
    --dataset "${TRAIN_DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 50 \
    --max_length 2048 \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_labels 1 \
    --task_type seq_cls \
    --use_chat_template false \
    --problem_type regression

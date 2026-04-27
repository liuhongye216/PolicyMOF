#!/usr/bin/env bash
set -euo pipefail

# Multi-task training: classification + regression + generation sharing one backbone
#
# task types in the dataset:
#   "gen" – causal-LM generation (standard next-token prediction)
#   "cls" – classification (num_cls_labels classes, label is an int)
#   "reg" – regression (num_reg_labels outputs, label is a float)
#
# The backbone is loaded as AutoModelForCausalLM; lightweight linear heads
# (cls_head / reg_head) are added automatically.

MODEL_PATH="${MODEL_PATH:-outputs/cpt_chemical_tokens}"
DATASET_PATH="${DATASET_PATH:-SFT/shared_backbone/train/data_train.jsonl}"
CHEMICAL_TOKENS_FILE="${CHEMICAL_TOKENS_FILE:-CPT/chemical_tokens.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/shared_backbone}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
    --model "${MODEL_PATH}" \
    --model_type llama3_1 \
    --train_type lora \
    --num_cls_labels 5 \
    --num_reg_labels 1 \
    --dataset "${DATASET_PATH}" \
    --train_type full \
    --new_special_tokens "${CHEMICAL_TOKENS_FILE}" \
    --modules_to_save embed_tokens lm_head\
    --max_length 768 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --logging_steps 5 \
    --output_dir "${OUTPUT_DIR}" \
    --gradient_checkpointing true \
    --split_dataset_ratio 0.1 \
    --per_device_eval_batch_size 2 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --target_modules all-linear \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
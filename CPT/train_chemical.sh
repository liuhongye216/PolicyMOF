#!/usr/bin/env bash
set -euo pipefail

# Pretrain with vocabulary expansion for chemical/MOF domain
# 
# 扩充词表的预训练脚本 (LoRA + 2 GPUs)
# 新 token 需要同时训练 embed_tokens 和 lm_head

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B}"
DATASET_PATH="${DATASET_PATH:-CPT/mof_pretrain_data.jsonl}"
CHEMICAL_TOKENS_FILE="${CHEMICAL_TOKENS_FILE:-CPT/chemical_tokens.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cpt_chemical_tokens}"

MASTER_PORT="${MASTER_PORT:-29501}" NPROC_PER_NODE="${NPROC_PER_NODE}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift pt \
    --model "${MODEL_PATH}" \
    --model_type llama3_1 \
    --train_type lora \
    --dataset "${DATASET_PATH}" \
    --new_special_tokens "${CHEMICAL_TOKENS_FILE}" \
    --modules_to_save embed_tokens lm_head \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --streaming true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2 \
    --packing true \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 1024 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --output_dir "${OUTPUT_DIR}" \
    --attn_impl flash_attn

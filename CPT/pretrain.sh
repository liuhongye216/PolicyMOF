#!/usr/bin/env bash
set -euo pipefail

# If not using flash_attn, or transformers<4.44,
# or encountering an abnormally large loss (i.e., the model does not support packing),
# please remove `--packing true`.
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B}"
DATASET_PATH="${DATASET_PATH:-CPT/mof_pretrain_data.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cpt}"

NPROC_PER_NODE="${NPROC_PER_NODE}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift pt \
    --model "${MODEL_PATH}" \
    --model_type llama3_1 \
    --train_type lora \
    --dataset "${DATASET_PATH}" \
    --torch_dtype bfloat16 \
    --streaming true \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --packing true \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 1024 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --output_dir "${OUTPUT_DIR}" \
    --attn_impl flash_attn

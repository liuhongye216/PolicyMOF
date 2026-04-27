#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-outputs/sft_generation}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/grpo/checkpoint-last}"
VAL_DATASET="${VAL_DATASET:-GRPO/test/gene_mix_test.jsonl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift infer \
    --model "${MODEL_PATH}" \
    --model_type llama3_1 \
    --adapters "${ADAPTER_PATH}" \
    --val_dataset "${VAL_DATASET}" \
    --infer_backend pt \
    --logprobs true \
    --stream true \
    --max_new_tokens 2048 \
    --max_batch_size 1

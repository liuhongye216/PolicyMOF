#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${ADAPTER_PATH:-outputs/sft_regression/checkpoint-last}"
VAL_DATASET="${VAL_DATASET:-SFT/best_regression/test/reg_reg_test.jsonl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift infer \
    --adapters "${ADAPTER_PATH}" \
    --val_dataset "${VAL_DATASET}" \
    --use_chat_template false \
    --load_data_args true \
    --max_batch_size 16

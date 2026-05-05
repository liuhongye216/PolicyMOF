#!/bin/bash
# SMILES Token Augmentation Training Script
# 
# This script demonstrates how to train a model with SMILES token-level augmentation.
# Two augmentation methods are provided:
#
# 1. Token Masking (--smiles_token_mask_enabled):
#    - Randomly replaces a portion of SMILES tokens with [MASK] token
#    - Helps the model learn to predict molecular properties from partial structures
#    - Improves robustness to noisy or incomplete SMILES inputs
#
# 2. Embedding Dropout (--smiles_token_dropout_enabled):
#    - Randomly zeros out embeddings of SMILES tokens during forward pass
#    - Acts as a regularization technique
#    - Helps prevent overfitting to specific token patterns
#
# Both methods exclude special tokens (like [CLS], [SEP], <s>, </s>, etc.)
# to ensure the model's structural understanding is preserved.
#
# Dataset format:
# {
#   "messages": [{"role": "user", "content": "CCO ..."}],
#   "isomeric_smiles": "C(C)O ...",  # optional, for SMILES augmentation
#   "label": {"cls": 1, "reg": 0.755}  # for multi-task learning
# }

# ============================================================================
# Training with SMILES Token Masking
# ============================================================================
# Masks 15% of SMILES tokens during training to improve robustness

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-0.5B \
    --train_type lora \
    --dataset '<your-smiles-dataset>' \
    --smiles_token_mask_enabled true \
    --smiles_token_mask_ratio 0.15 \
    --use_isomeric_smiles true \
    --load_from_cache_file true \
    --split_dataset_ratio 0.1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_smiles_mask \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --task_type seq_cls \
    --use_chat_template false \
    --problem_type multitask \
    --num_labels 6 \
    --num_cls_labels 5 \
    --num_reg_labels 1 \
    --multitask_loss_strategy uncertainty

# ============================================================================
# Training with SMILES Embedding Dropout
# ============================================================================
# Drops 10% of SMILES token embeddings during training for regularization

# CUDA_VISIBLE_DEVICES=0 \
# swift sft \
#     --model Qwen/Qwen2.5-0.5B \
#     --train_type lora \
#     --dataset '<your-smiles-dataset>' \
#     --smiles_token_dropout_enabled true \
#     --smiles_token_dropout_ratio 0.1 \
#     --use_isomeric_smiles true \
#     --load_from_cache_file true \
#     --split_dataset_ratio 0.1 \
#     --torch_dtype bfloat16 \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --learning_rate 5e-5 \
#     --lora_rank 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --target_modules all-linear \
#     --gradient_accumulation_steps 4 \
#     --eval_steps 100 \
#     --save_steps 100 \
#     --save_total_limit 5 \
#     --logging_steps 5 \
#     --max_length 2048 \
#     --output_dir output_smiles_dropout \
#     --warmup_ratio 0.1 \
#     --weight_decay 0.01 \
#     --max_grad_norm 1.0 \
#     --dataloader_num_workers 4 \
#     --dataset_num_proc 4 \
#     --task_type seq_cls \
#     --use_chat_template false \
#     --problem_type multitask \
#     --num_labels 6 \
#     --num_cls_labels 5 \
#     --num_reg_labels 1 \
#     --multitask_loss_strategy uncertainty

# ============================================================================
# Training with Both Token Masking and Embedding Dropout
# ============================================================================
# Combines both augmentation methods for maximum regularization

# CUDA_VISIBLE_DEVICES=0 \
# swift sft \
#     --model Qwen/Qwen2.5-0.5B \
#     --train_type lora \
#     --dataset '<your-smiles-dataset>' \
#     --smiles_token_mask_enabled true \
#     --smiles_token_mask_ratio 0.15 \
#     --smiles_token_dropout_enabled true \
#     --smiles_token_dropout_ratio 0.1 \
#     --use_isomeric_smiles true \
#     --load_from_cache_file true \
#     --split_dataset_ratio 0.1 \
#     --torch_dtype bfloat16 \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --learning_rate 5e-5 \
#     --lora_rank 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --target_modules all-linear \
#     --gradient_accumulation_steps 4 \
#     --eval_steps 100 \
#     --save_steps 100 \
#     --save_total_limit 5 \
#     --logging_steps 5 \
#     --max_length 2048 \
#     --output_dir output_smiles_combined \
#     --warmup_ratio 0.1 \
#     --weight_decay 0.01 \
#     --max_grad_norm 1.0 \
#     --dataloader_num_workers 4 \
#     --dataset_num_proc 4 \
#     --task_type seq_cls \
#     --use_chat_template false \
#     --problem_type multitask \
#     --num_labels 6 \
#     --num_cls_labels 5 \
#     --num_reg_labels 1 \
#     --multitask_loss_strategy uncertainty

# ============================================================================
# Custom Special Tokens
# ============================================================================
# If your model uses custom special tokens that should not be masked/dropped,
# you can specify them with --smiles_special_tokens:
#
# --smiles_special_tokens '[SMILES]' '[/SMILES]' '<mol>' '</mol>'

# ============================================================================
# Recommended Settings
# ============================================================================
#
# Token Masking:
# - mask_ratio=0.15: Standard BERT-style masking, good for general use
# - mask_ratio=0.10: More conservative, preserves more structure
# - mask_ratio=0.20: More aggressive, stronger regularization
#
# Embedding Dropout:
# - dropout_ratio=0.10: Light regularization
# - dropout_ratio=0.20: Moderate regularization
# - dropout_ratio=0.30: Strong regularization (use with caution)
#
# Combined Usage:
# - For smaller datasets: Use both with lower ratios (0.10, 0.05)
# - For larger datasets: Can use higher ratios (0.15, 0.10)
# - For overfitting models: Increase dropout_ratio
# - For underfitting models: Decrease or disable augmentation

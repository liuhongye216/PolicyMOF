#!/bin/bash
# Multi-task training: classification + regression + generation sharing one backbone
#
# task types in the dataset:
#   "gen" – causal-LM generation (standard next-token prediction)
#   "cls" – classification (num_cls_labels classes, label is an int)
#   "reg" – regression (num_reg_labels outputs, label is a float)
#
# The backbone is loaded as AutoModelForCausalLM; lightweight linear heads
# (cls_head / reg_head) are added automatically.

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type multi_task_gen \
    --num_cls_labels 3 \
    --num_reg_labels 1 \
    --dataset examples/train/multitask_gen/data.jsonl \
    --train_type full \
    --max_length 512 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --output_dir output/multitask_gen \
    --gradient_checkpointing true

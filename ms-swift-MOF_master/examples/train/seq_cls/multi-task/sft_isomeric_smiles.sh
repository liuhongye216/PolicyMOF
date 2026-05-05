# Multi-task classification + regression training script with SMILES augmentation
# This script demonstrates how to use isomeric SMILES for data augmentation
#
# Key feature: --use_isomeric_smiles
# When set to true, the training will use randomized/isomeric SMILES from the 
# 'isomeric_smiles' field instead of canonical SMILES in the messages content.
# This improves model robustness to different SMILES representations.
#
# Dataset format when using isomeric SMILES:
# {
#   "messages": [{"role": "user", "content": "canonical_smiles ..."}],
#   "isomeric_smiles": "randomized_smiles ...",
#   "label": {"cls": 1, "reg": 0.755}
# }

# Training with isomeric SMILES (data augmentation enabled)
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-0.5B \
    --train_type lora \
    --dataset '<your-dataset-with-isomeric-smiles>' \
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
    --output_dir output_isomeric \
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

# Training without isomeric SMILES (use canonical SMILES)
# Just remove --use_isomeric_smiles or set it to false:
# CUDA_VISIBLE_DEVICES=0 \
# swift sft \
#     --model Qwen/Qwen2.5-0.5B \
#     --dataset '<your-dataset>' \
#     --use_isomeric_smiles false \
#     ... (other options)

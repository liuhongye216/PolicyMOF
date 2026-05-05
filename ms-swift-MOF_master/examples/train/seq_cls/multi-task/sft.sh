# Multi-task classification + regression training script
# Custom dataset format reference: https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html
# For multitask, labels should contain both classification and regression targets
#
# Key improvements to prevent overfitting:
# 1. Reduced epochs (5 instead of 10) - best model was at epoch ~4.8
# 2. Lower learning rate (5e-5) - more stable training
# 3. Added weight_decay (0.01) - L2 regularization
# 4. Increased lora_dropout (0.1) - prevent co-adaptation
# 5. Added max_grad_norm (1.0) - prevent gradient explosion
# 6. Early stopping via save_total_limit and load_best_model
#
# Dynamic Loss Weighting Strategies (--multitask_loss_strategy):
# - 'fixed': Use --multitask_loss_weight as fixed weight (default)
# - 'uncertainty': Kendall et al. 2018 - learns task uncertainty to balance losses automatically
# - 'dwa': Liu et al. 2019 - Dynamic Weight Average based on loss change rate

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-0.5B \
    --train_type lora \
    --dataset '<your-dataset>' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.11111 \
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
    --output_dir output \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --task_type seq_cls \
    --use_chat_template false \
    --problem_type multitask \
    --num_labels '6' \
    --num_cls_labels '5' \
    --num_reg_labels '1' \
    --multitask_loss_strategy uncertainty \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --load_best_model_at_end true
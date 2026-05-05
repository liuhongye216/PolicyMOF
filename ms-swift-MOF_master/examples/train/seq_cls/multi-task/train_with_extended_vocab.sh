#!/bin/bash
# =============================================================================
# 使用扩充词表的 ms-swift 训练脚本
# =============================================================================
#
# 方案说明：
# 1. 先运行 Python 脚本扩充词表并保存模型
# 2. 使用扩充后的模型进行 swift sft 训练
# 3. 或者直接在训练前使用自定义 preprocess 函数
#
# =============================================================================

# Step 1: 扩充词表并保存模型
echo "Step 1: Extending vocabulary and saving model..."

python -c "
import sys
sys.path.insert(0, '.')

from custom_tokenizer_integration import prepare_model_and_tokenizer

# 扩充词表并保存
model, tokenizer = prepare_model_and_tokenizer(
    model_name_or_path='Qwen/Qwen2.5-0.5B',
    num_labels=6,  # 根据你的任务调整
    problem_type='multi_label_classification',
    save_dir='./extended_qwen2.5_smiles',
)

print('Vocabulary extended and saved!')
print(f'New vocab size: {len(tokenizer)}')
"

if [ $? -ne 0 ]; then
    echo "Failed to extend vocabulary!"
    exit 1
fi

echo "Vocabulary extension completed!"

# Step 2: 使用扩充后的模型进行训练
echo "Step 2: Training with extended vocabulary model..."

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ./extended_qwen2.5_smiles \
    --train_type lora \
    --dataset '<your-dataset>' \
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
    --max_length 512 \
    --output_dir output_extended_vocab \
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

echo "Training completed!"

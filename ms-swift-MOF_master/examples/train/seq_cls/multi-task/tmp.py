import sys
import os
import shutil
sys.path.insert(0, '.')

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from custom_tokenizer_integration import get_chemical_tokens
# 配置原始模型路径
ORIGINAL_MODEL="/home/liuhongye/Model/3.1"  # 或你的本地模型路径，如 /home/liuhongye/Model/MOF-3.1
EXTENDED_MODEL_DIR="/home/liuhongye/Model/3.1-train"  # 保存扩展词汇表后的模型路径
# 替换占位符为实际路径
model_path = ORIGINAL_MODEL
save_dir = EXTENDED_MODEL_DIR

print(f'Loading from: {model_path}')

# 1. 加载原始 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

original_vocab_size = len(tokenizer)
print(f'Original vocab size: {original_vocab_size}')

# 2. 添加化学 tokens
special_tokens, chemical_tokens = get_chemical_tokens()
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
tokenizer.add_tokens(chemical_tokens)

new_vocab_size = len(tokenizer)
print(f'New vocab size: {new_vocab_size} (+{new_vocab_size - original_vocab_size})')

# 3. 加载完整的 CausalLM 模型（保持原始架构）
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='cpu',  # 在 CPU 上操作，避免 GPU 内存问题
)

# 4. 扩充 embedding 层（只扩充 embedding，其他权重完全不变）
# resize_token_embeddings 会自动处理 input_embeddings 和 output_embeddings (lm_head)
# 新添加的 token 会用随机初始化或均值初始化
model.resize_token_embeddings(new_vocab_size)
print(f'Resized embeddings to {new_vocab_size}')

# 5. 保存完整模型
os.makedirs(save_dir, exist_ok=True)
print(f'Saving to {save_dir}...')

# 保存 tokenizer
tokenizer.save_pretrained(save_dir)

# 保存完整模型（包括所有权重）
model.save_pretrained(save_dir, safe_serialization=True)

print(f'Saved complete model with extended vocabulary to {save_dir}')
print('Done! The new model is a complete model with only the vocabulary extended.')
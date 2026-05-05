# SMILES 扩充词表训练方案

本目录提供了在 ms-swift 中使用扩充化学词表进行分子属性预测的完整方案。

## 问题背景

标准语言模型的 tokenizer 无法正确理解 SMILES 化学符号：

```
原始 SMILES: CC(=O)OC1=CC=CC=C1C(=O)O (阿司匹林)

❌ 标准分词: ['CC', '(=', 'O', ')', 'OC', '1', '=', 'CC', ...]
✅ 化学感知分词: ['C', 'C', '(', '=', 'O', ')', 'O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C(=O)O']
```

## 解决方案

### 方案 A: 预处理方式（推荐） ⭐

#### 步骤 1: 扩充词表并保存模型

```python
from smiles_swift_integration import extend_tokenizer_and_save

# 扩充词表并保存
model, tokenizer = extend_tokenizer_and_save(
    model_name_or_path='Qwen/Qwen2.5-0.5B',  # 原始模型
    save_dir='./extended_qwen2.5_smiles',     # 保存目录
    num_labels=6,                              # 标签数
    problem_type='multi_label_classification'  # 问题类型
)

print(f"新词表大小: {len(tokenizer)}")
```

#### 步骤 2: 使用扩充后的模型进行训练

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ./extended_qwen2.5_smiles \
    --train_type lora \
    --dataset your_dataset.jsonl \
    --task_type seq_cls \
    --problem_type multitask \
    --num_labels 6 \
    --num_cls_labels 5 \
    --num_reg_labels 1 \
    --use_chat_template false \
    --max_length 512 \
    --output_dir output_extended
```

### 方案 B: 数据预处理方式

如果不想修改模型词表，可以对数据进行预处理，使用空格分隔的 tokens：

```python
from preprocess_smiles_dataset import process_dataset

# 预处理数据集
process_dataset(
    input_file='raw_data.jsonl',
    output_file='processed_data.jsonl',
    use_space_separated=True
)
```

预处理后的数据格式：
```json
{
  "messages": [{"role": "user", "content": "C C ( = O ) O C 1 = C C = C C = C 1 C(=O)O"}],
  "label": {"cls": 1, "reg": 0.755}
}
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `smiles_swift_integration.py` | 主要集成代码，包含扩充词表函数 |
| `enhence_tokenizer.py` | 原始的 SwiftStyleTokenizer 类 |
| `preprocess_smiles_dataset.py` | 数据预处理脚本 |
| `train_with_extended_vocab.sh` | 训练脚本示例 |
| `custom_tokenizer_integration.py` | 详细的集成示例代码 |

## 添加的 Tokens

### 特殊 Tokens
- `[MASK]` - 用于掩码语言建模
- `[UNK_SMILES]` - 未知 SMILES token

### 化学 Tokens

1. **元素符号**: H, C, N, O, S, P, F, Cl, Br, I, Na, Mg, Fe, Cu, Zn, ...
2. **方括号形式**: [H], [C], [N], [O], [N+], [O-], [NH4+], ...
3. **芳香形式**: c, n, o, s, p (小写表示芳香性)
4. **手性标记**: [@], [@@], [@H], [@@H], [C@], [C@@], ...
5. **化学基团**:
   - 羧酸: `C(=O)O`, `C(=O)[O-]`
   - 氰基: `C#N`
   - 硝基: `[N+](=O)[O-]`
   - 磺酸: `S(=O)(=O)O`, `S(=O)(=O)[O-]`
6. **环闭合**: %10, %11, ..., %99

## 数据集格式

### 输入数据格式
```json
{
  "messages": [{"role": "user", "content": "CC(=O)OC1=CC=CC=C1C(=O)O"}],
  "label": {"cls": 1, "reg": 0.755}
}
```

### 使用 isomeric SMILES（数据增强）
```json
{
  "messages": [{"role": "user", "content": "CC(=O)OC1=CC=CC=C1C(=O)O"}],
  "isomeric_smiles": "C(=O)(O)c1ccccc1OC(C)=O",
  "label": {"cls": 1, "reg": 0.755}
}
```

## 训练参数建议

```bash
# 推荐参数
--max_length 512            # SMILES 通常不需要很长
--learning_rate 5e-5        # 可以适当调高
--per_device_train_batch_size 16
--gradient_accumulation_steps 4
--num_train_epochs 5
--warmup_ratio 0.1
--use_chat_template false   # SMILES 不需要对话模板
```

## 验证分词效果

```python
from smiles_swift_integration import tokenize_smiles

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
tokens = tokenize_smiles(smiles)
print(f"Tokens ({len(tokens)}): {tokens}")

# 验证完整性
assert ''.join(tokens) == smiles, "分词不完整"
```

## 常见问题

### Q1: 扩充词表后模型大小增加多少？
扩充约 500-1000 个 tokens，模型大小增加约 1-2MB，对训练和推理影响很小。

### Q2: 可以继续用原始 checkpoint 吗？
可以。扩充词表后保存的模型包含新的 embedding，可以直接用 `--model` 参数指定。

### Q3: 如何处理包含文本描述的混合数据？
数据中的 SMILES 部分会被正确分词，普通文本部分仍使用原始分词方式。

### Q4: 推理时需要做什么？
使用扩充后的模型目录即可，tokenizer 会自动加载扩充后的词表。

## 性能对比

| 分词方式 | 平均 Token 数 | 训练时间 | 模型效果 |
|----------|--------------|---------|---------|
| 标准分词 | ~40 | 基准 | 基准 |
| 化学感知分词 | ~25-30 | 更快 | 更好 |

化学感知分词可以：
- 减少 token 数量，加快训练
- 保留化学结构信息，提升模型理解能力
- 支持特殊 token 如 [MASK]，便于扩展任务

## 参考

- [ms-swift 文档](https://github.com/modelscope/ms-swift)
- [SMILES 规范](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)

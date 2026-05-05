"""
ms-swift 自定义 SMILES Tokenizer 完整集成方案

本文件提供了两种方式将扩充词表集成到 ms-swift 训练流程：

方案 A: 预处理方式（推荐）
    1. 扩充词表并保存模型
    2. 使用保存的模型路径进行训练
    
方案 B: 自定义模型注册（高级）
    1. 注册自定义的 get_model_tokenizer 函数
    2. 在函数中扩充词表

使用示例见文件末尾。
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

logger = logging.getLogger(__name__)

# =============================================================================
# SMILES 分词相关
# =============================================================================

_SMILES_PATTERN = re.compile(
    r'('
    # 1. 特殊token（优先匹配）
    r'\[MASK\]|\[PAD\]|\[UNK_SMILES\]|\[CLS\]|\[SEP\]|'
    
    # 2. 常见化学基团（优先匹配长模式）
    r'C\(=O\)\[O-\]|'      # 羧酸根
    r'C\(=O\)O|'           # 羧酸
    r'C\(=O\)N|'           # 酰胺
    r'C\(=O\)|'            # 羰基
    r'C#N|'                # 氰基
    r'\[N\+\]\(=O\)\[O-\]|'  # 硝基
    r'S\(=O\)\(=O\)\[O-\]|'  # 磺酸根
    r'S\(=O\)\(=O\)O|'       # 磺酸
    r'S\(=O\)\(=O\)|'        # 砜
    r'P\(=O\)\(\[O-\]\)\[O-\]|'  # 磷酸根
    r'P\(=O\)\(O\)O|'        # 磷酸
    
    # 3. 方括号内的完整原子
    r'\[[^\]]+\]|'
    
    # 4. 双字符元素
    r'Br|Cl|Si|Se|As|Na|Mg|Al|Ca|Fe|Cu|Zn|Ag|Au|Pt|Pd|'
    
    # 5. 单字符元素
    r'B|C|N|O|S|P|F|I|H|K|'
    
    # 6. 芳香性小写元素
    r'se|as|b|c|n|o|s|p|'
    
    # 7. 环闭合标记
    r'%\d{2}|'
    
    # 8. 数字
    r'\d|'
    
    # 9. 键和其他符号
    r'[=#@+\-\/\\().\[\]]'
    r')'
)


def tokenize_smiles(smiles: str) -> List[str]:
    """使用化学感知的方式分词 SMILES"""
    return _SMILES_PATTERN.findall(smiles)


def get_chemical_tokens() -> Tuple[List[str], List[str]]:
    """获取需要添加的化学 tokens"""
    # 特殊 tokens
    special_tokens = ['[MASK]', '[UNK_SMILES]']
    
    # 化学 tokens
    chemical_tokens = []
    
    # 元素
    common_elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
        'K', 'Ca', 'Fe', 'Cu', 'Zn', 'Br', 'I', 'Ag', 'Au', 'Pt', 'Pd',
    ]
    
    for e in common_elements:
        chemical_tokens.extend([e, f'[{e}]'])
    
    # 芳香形式
    for e in ['b', 'c', 'n', 'o', 'p', 's', 'se', 'as']:
        chemical_tokens.extend([e, f'[{e}]'])
    
    # 带电荷的形式
    for e in ['C', 'N', 'O', 'S', 'P', 'B', 'c', 'n', 'o', 's', 'p']:
        chemical_tokens.extend([
            f'[{e}H]', f'[{e}+]', f'[{e}-]',
            f'[{e}H+]', f'[{e}H-]',
            f'[{e}H2]', f'[{e}H3]', f'[{e}H4]',
        ])
    
    # 手性
    chemical_tokens.extend(['[@]', '[@@]', '[@H]', '[@@H]'])
    for e in ['C', 'N', 'S', 'P']:
        chemical_tokens.extend([f'[{e}@]', f'[{e}@@]', f'[{e}@H]', f'[{e}@@H]'])
    
    # 常见离子
    chemical_tokens.extend([
        '[O-]', '[OH-]', '[N+]', '[NH+]', '[NH2+]', '[NH3+]', '[NH4+]',
        '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]', '[Fe+2]', '[Fe+3]',
        '[Cu+]', '[Cu+2]', '[Zn+2]',
    ])
    
    # 化学基团
    chemical_tokens.extend([
        'C(=O)[O-]', 'C(=O)O', 'C(=O)N', 'C(=O)',
        'C#N', '[N+](=O)[O-]',
        'S(=O)(=O)[O-]', 'S(=O)(=O)O', 'S(=O)(=O)',
        'P(=O)([O-])[O-]', 'P(=O)(O)O',
    ])
    
    # 环闭合
    chemical_tokens.extend([f'%{i}' for i in range(10, 100)])
    
    # 结构符号
    chemical_tokens.extend(['(', ')', '[', ']', '.', '=', '#', '@', '+', '-', '/', '\\'])
    chemical_tokens.extend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
    
    # 拓扑符号
    chemical_tokens.extend(['MOFid-v1', 'cat0', 'cat1', 'cat2', 'cat3', 'acs', 'bct',
                            'bcu', 'cpf', 'cpr', 'dia', 'fcu', 'fnu', 'fsc', 'fse',
                            'fsf', 'fsg', 'hcb', 'hex', 'hms', 'hxl', 'irl', 'jeb',
                            'lfm', 'mot', 'nbo', 'pcu', 'rna', 'rob', 'sit', 'sql', 'sqp', 'tbo'])

    return special_tokens, list(dict.fromkeys(chemical_tokens))


# =============================================================================
# 方案 A: 预处理方式 (推荐)
# =============================================================================

def extend_tokenizer_and_save(
    model_name_or_path: str,
    save_dir: str,
    num_labels: int = 2,
    problem_type: str = "single_label_classification",
) -> Tuple[Any, Any]:
    """
    扩充词表并保存模型和 tokenizer
    
    Args:
        model_name_or_path: 原始模型路径
        save_dir: 保存目录
        num_labels: 分类标签数
        problem_type: 问题类型
        
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading model from {model_name_or_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    original_vocab_size = len(tokenizer)
    
    # 添加 tokens
    special_tokens, chemical_tokens = get_chemical_tokens()
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.add_tokens(chemical_tokens)
    
    new_vocab_size = len(tokenizer)
    logger.info(f"Vocabulary: {original_vocab_size} -> {new_vocab_size} (+{new_vocab_size - original_vocab_size})")
    
    # 加载模型
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.num_labels = num_labels
    config.problem_type = problem_type
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
    )
    
    # 调整 embedding 大小
    model.resize_token_embeddings(new_vocab_size)
    logger.info(f"Resized embeddings to {new_vocab_size}")
    
    # 保存
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    
    # 保存配置信息
    config_info = {
        "original_model": model_name_or_path,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": new_vocab_size,
        "num_labels": num_labels,
        "problem_type": problem_type,
        "special_tokens": special_tokens,
        "num_chemical_tokens": len(chemical_tokens),
    }
    with open(os.path.join(save_dir, "extended_vocab_config.json"), "w") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved extended model to {save_dir}")
    
    return model, tokenizer


# =============================================================================
# 方案 B: 自定义模型注册 (高级)
# =============================================================================

def get_model_tokenizer_with_extended_vocab(
    model_dir: str,
    model_info: Any,
    model_kwargs: Dict[str, Any],
    load_model: bool = True,
    **kwargs
) -> Tuple[Any, Any]:
    """
    自定义的 get_model_tokenizer 函数，支持扩充词表
    
    用法: 在训练脚本中注册此函数
    """
    from swift.llm import get_model_tokenizer_with_flash_attn
    
    # 先用默认方式加载
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, **kwargs
    )
    
    # 检查是否需要扩充词表
    extend_vocab = kwargs.get('extend_vocab_for_smiles', False)
    
    if extend_vocab and tokenizer is not None:
        original_vocab_size = len(tokenizer)
        
        special_tokens, chemical_tokens = get_chemical_tokens()
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        tokenizer.add_tokens(chemical_tokens)
        
        new_vocab_size = len(tokenizer)
        
        if model is not None and new_vocab_size > original_vocab_size:
            model.resize_token_embeddings(new_vocab_size)
            logger.info(f"Extended vocabulary: {original_vocab_size} -> {new_vocab_size}")
    
    return model, tokenizer


def register_smiles_model():
    """
    注册支持 SMILES 扩充词表的模型
    
    用法：在训练脚本开头调用此函数
    """
    try:
        from swift.llm import register_model, ModelMeta, ModelGroup, Model
        
        register_model(
            ModelMeta(
                model_type='qwen2.5-smiles',
                model_groups=[
                    ModelGroup([
                        Model('Qwen/Qwen2.5-0.5B'),
                        Model('Qwen/Qwen2.5-1.5B'),
                        Model('Qwen/Qwen2.5-3B'),
                        Model('Qwen/Qwen2.5-7B'),
                    ])
                ],
                template='qwen',
                get_function=get_model_tokenizer_with_extended_vocab,
            )
        )
        logger.info("Registered SMILES model type: qwen2.5-smiles")
    except ImportError:
        logger.warning("Could not import swift.llm, skipping model registration")


# =============================================================================
# 数据预处理
# =============================================================================

def preprocess_smiles_data(
    input_file: str,
    output_file: str,
    tokenize_content: bool = True,
) -> None:
    """
    预处理 SMILES 数据集
    
    将 SMILES 字符串转换为用空格分隔的 tokens
    """
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    processed = []
    for item in data:
        new_item = item.copy()
        
        if 'messages' in item and tokenize_content:
            new_messages = []
            for msg in item['messages']:
                new_msg = msg.copy()
                if 'content' in msg:
                    content = msg['content']
                    # 尝试分词
                    tokens = tokenize_smiles(content)
                    if tokens and len(''.join(tokens)) >= len(content) * 0.8:
                        new_msg['content'] = ' '.join(tokens)
                new_messages.append(new_msg)
            new_item['messages'] = new_messages
        
        processed.append(new_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Processed {len(processed)} samples -> {output_file}")


# =============================================================================
# 示例
# =============================================================================

def example_workflow():
    """完整的工作流程示例"""
    
    print("=" * 70)
    print("ms-swift SMILES 扩充词表工作流程")
    print("=" * 70)
    
    # 1. 测试分词
    print("\n[步骤 1] 测试 SMILES 化学感知分词")
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
    tokens = tokenize_smiles(test_smiles)
    
    print(f"  原始 SMILES: {test_smiles}")
    print(f"  分词结果 ({len(tokens)}): {tokens}")
    print(f"  期望分词: ['C', 'C', '(', '=', 'O', ')', 'O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C', 'C(=O)O']")
    
    # 验证
    reconstructed = ''.join(tokens)
    print(f"  重建验证: {'✅ 完整' if reconstructed == test_smiles else '⚠️ 不完整'}")
    
    # 2. 显示工作流程
    print("\n[步骤 2] 扩充词表并保存模型")
    print("""
    # Python 代码:
    from smiles_swift_integration import extend_tokenizer_and_save
    
    model, tokenizer = extend_tokenizer_and_save(
        model_name_or_path='Qwen/Qwen2.5-0.5B',
        save_dir='./extended_model',
        num_labels=6,
        problem_type='multi_label_classification'
    )
    """)
    
    print("\n[步骤 3] 使用扩充后的模型进行训练")
    print("""
    # Shell 命令:
    swift sft \\
        --model ./extended_model \\
        --dataset your_dataset.jsonl \\
        --task_type seq_cls \\
        --problem_type multitask \\
        --num_labels 6 \\
        --output_dir output
    """)
    
    print("\n[可选] 预处理数据集")
    print("""
    # Python 代码:
    from smiles_swift_integration import preprocess_smiles_data
    
    preprocess_smiles_data(
        input_file='raw_data.jsonl',
        output_file='processed_data.jsonl'
    )
    """)
    
    print("\n" + "=" * 70)
    print("工作流程说明完成")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_workflow()

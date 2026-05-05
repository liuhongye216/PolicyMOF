"""
数据预处理脚本：将 SMILES 数据转换为使用化学感知分词的格式

这个脚本会：
1. 读取原始数据集
2. 使用化学感知的方式对 SMILES 进行分词
3. 保存预处理后的数据集

使用方法：
    python preprocess_smiles_dataset.py \
        --input_file your_dataset.jsonl \
        --output_file processed_dataset.jsonl \
        --use_space_separated true
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# SMILES 分词正则表达式
_SMILES_PATTERN = re.compile(
    r'('
    # 1. 特殊token（优先匹配）
    r'\[MASK\]|\[PAD\]|\[UNK_SMILES\]|\[CLS\]|\[SEP\]|'
    
    # 2. 常见化学基团（优先匹配长模式）
    r'C\(=O\)\[O-\]|'
    r'C\(=O\)O|'
    r'C\(=O\)N|'
    r'C\(=O\)|'
    r'C#N|'
    r'\[N\+\]\(=O\)\[O-\]|'
    r'S\(=O\)\(=O\)\[O-\]|'
    r'S\(=O\)\(=O\)O|'
    r'S\(=O\)\(=O\)|'
    r'P\(=O\)\(\[O-\]\)\[O-\]|'
    r'P\(=O\)\(O\)O|'
    
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
    tokens = _SMILES_PATTERN.findall(smiles)
    return tokens


def process_message_content(content: str, use_space_separated: bool = True) -> str:
    """
    处理消息内容中的 SMILES
    
    Args:
        content: 原始消息内容
        use_space_separated: 是否使用空格分隔的 tokens
        
    Returns:
        处理后的内容
    """
    # 简单处理：假设内容主要是 SMILES
    # 你可能需要根据实际数据格式调整这个函数
    
    # 检测是否包含 SMILES（简单启发式）
    smiles_chars = set('CNOSPFIBrcnospf[]()=#@+-/\\%0123456789')
    if all(c in smiles_chars or c.isspace() for c in content):
        # 看起来像 SMILES
        tokens = tokenize_smiles(content)
        if use_space_separated:
            return ' '.join(tokens)
        else:
            return content  # 保持原样
    
    return content


def process_dataset(
    input_file: str,
    output_file: str,
    use_space_separated: bool = True,
) -> None:
    """
    处理数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        use_space_separated: 是否使用空格分隔
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # 读取数据
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples")
    
    # 处理数据
    processed_data = []
    for item in tqdm(data, desc="Processing"):
        processed_item = item.copy()
        
        # 处理 messages 字段
        if 'messages' in item:
            processed_messages = []
            for msg in item['messages']:
                processed_msg = msg.copy()
                if 'content' in msg:
                    processed_msg['content'] = process_message_content(
                        msg['content'], 
                        use_space_separated
                    )
                processed_messages.append(processed_msg)
            processed_item['messages'] = processed_messages
        
        # 处理 query 字段（如果存在）
        if 'query' in item:
            processed_item['query'] = process_message_content(
                item['query'],
                use_space_separated
            )
        
        processed_data.append(processed_item)
    
    # 保存处理后的数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(processed_data)} samples to {output_file}")


def demo_tokenization():
    """演示分词效果"""
    test_cases = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
        "O=C([O-])c1ccc(C(=O)[O-])cc1",  # 对苯二甲酸盐
        "CC[C@H](C)O",  # 手性分子
        "C#N",  # 氰基
        "CC[MASK]C(=O)O",  # 带 MASK
        "[Cu].[Cu].O=C([O-])c1ccc(C(=O)[O-])cc1",  # MOF 相关
    ]
    
    print("=" * 70)
    print("SMILES 化学感知分词演示")
    print("=" * 70)
    
    for smiles in test_cases:
        tokens = tokenize_smiles(smiles)
        print(f"\nSMILES: {smiles}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"Space-separated: {' '.join(tokens)}")
        
        # 验证是否完整匹配
        reconstructed = ''.join(tokens)
        if reconstructed == smiles:
            print("✅ 完整匹配")
        else:
            print(f"⚠️  不完整: {len(smiles) - len(reconstructed)} 字符丢失")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理 SMILES 数据集")
    parser.add_argument("--input_file", type=str, help="输入文件路径")
    parser.add_argument("--output_file", type=str, help="输出文件路径")
    parser.add_argument("--use_space_separated", type=bool, default=True,
                        help="是否使用空格分隔 tokens")
    parser.add_argument("--demo", action="store_true", help="运行演示")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_tokenization()
    elif args.input_file and args.output_file:
        process_dataset(args.input_file, args.output_file, args.use_space_separated)
    else:
        # 默认运行演示
        demo_tokenization()

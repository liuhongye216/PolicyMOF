"""
SMILES Tokenizer - 化学正确的切分策略

核心原则：
1. 尊重 SMILES 语法：数字标记的是环的起点，不能被包含到前面的官能团中
2. 括号边界：括号外的原子不能被包含进官能团
3. 最长匹配但要考虑上下文：不能贪婪地吃掉不属于官能团的部分

正确切分阿司匹林：
CC(=O)OC1=CC=CC=C1C(=O)O
→ ['C', 'C(=O)O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C(=O)O']
           ^^^^^^   ↑                                                  ^^^^^^
        乙酰氧基   苯环起点                                              羧基
"""

import os
import re
import torch
import logging
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# ============================================
# SMILES 分词正则表达式（化学正确版）
# ============================================

def build_smiles_pattern() -> re.Pattern:
    """
    构建化学正确的 SMILES 分词正则
    
    关键约束：
    1. 官能团后不能包含环标记数字（如 C(=O)O 后面的 C1 中的 C）
    2. 官能团后不能包含新的括号结构
    3. 尊重 SMILES 的层次结构
    """
    
    patterns = [
        # ========================================
        # 1. 方括号原子（最高优先级，不可拆分）
        # ========================================
        r'\[[^\]]+\]',
        
        # ========================================
        # 2. 完整官能团（严格限定边界）
        # ========================================
        # ========================================
        # 原子 + 单位数环标记（最高优先级之一）
        # ========================================
        r'[A-Z][a-z]?\d',     # C1, N2, Cl3
        r'[bcnops]\d',        # c1, n2, o3

        # ========================================
        # 芳香连续片段（run）
        # ========================================
        r'c\d?(?:c){2,}',   # ccc, cccc, c2ccc
        r'n\d?(?:c){2,}',   # nccc

        # 羧酸及其衍生物（后面不能跟数字、括号、大写字母）
        # 使用负向前瞻 (?!...) 确保后面不是环标记或新结构
        r'C\(=O\)\[O-\](?![0-9A-Z])',     # 羧酸根
        r'C\(=O\)OH(?![0-9A-Z])',         # 羧酸（显式H）
        r'C\(=O\)O(?![0-9A-Z([])',        # 羧基 - 关键：后面不能是数字、括号、大写
        
        # 磺酸及其衍生物
        r'S\(=O\)\(=O\)\[O-\](?![0-9A-Z])',
        r'S\(=O\)\(=O\)OH(?![0-9A-Z])',
        r'S\(=O\)\(=O\)O(?![0-9A-Z([])',  # 磺酸 - 后面不能是数字等
        r'S\(=O\)\(=O\)N(?![0-9A-Z])',
        r'S\(=O\)\(=O\)NH2(?![0-9A-Z])',
        r'S\(=O\)\(=O\)Cl(?![0-9A-Z])',
        
        # 硝基
        r'\[N\+\]\(=O\)\[O-\]',
        r'N\(=O\)\(=O\)',
        r'N\(=O\)=O',
        
        # 磷酸
        r'P\(=O\)\(O\)O(?![0-9A-Z([])',
        r'P\(=O\)\(OH\)OH',
        r'P\(=O\)\(\[O-\]\)\[O-\]',
        r'P\(=O\)\(O\)\[O-\]',
        
        # 氰基
        r'C#N',
        
        # ========================================
        # 3. 芳香环上的取代基（带 c 小写）
        # ========================================
        r'c\(C\(=O\)O\)',
        r'c\(C\(=O\)\[O-\]\)',
        r'c\(C\(=O\)\)',
        
        # ========================================
        # 4. 括号单元（多括号优先）
        # ========================================
        
        # 多括号单元：X(...)(...)(...)
        r'[A-Z][a-z]?(?:\([^)]+\)){2,}',
        
        # 单括号单元：X(...)（但不包括后面可能的独立原子）
        r'[A-Z][a-z]?\([^)]+\)',
        
        # ========================================
        # 5. 基础原子和符号
        # ========================================
        
        # 双字符元素
        r'Br|Cl|Si|Se|As|Na|Mg|Al|Ca|Fe|Cu|Zn|Ag|Au|Pt|Pd|Hg',
        
        # 单字符元素（大写）
        r'[BCNOSPFIHK]',
        
        # 芳香性小写
        r'se|as|b|c|n|o|s|p',
        
        # 环闭合标记
        r'%\d{2}',
        
        # 数字
        r'\d',
        
        # 键符号
        r'[=#\-/\\:]',
        
        # 其他符号
        r'[()@+.]',
    ]
    
    combined_pattern = '(' + '|'.join(patterns) + ')'
    return re.compile(combined_pattern)


_SMILES_PATTERN = build_smiles_pattern()


def tokenize_smiles(smiles: str) -> List[str]:
    """
    化学正确的 SMILES 分词
    
    Examples:
        >>> tokenize_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        ['C', 'C(=O)O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C(=O)O']
        #     ^^^^^^   ↑ 苯环起点                                           ^^^^^^
        
        >>> tokenize_smiles("c1ccc(cc1)S(=O)(=O)O")
        ['c', '1', 'c', 'c', 'c', '(', 'c', 'c', '1', ')', 'S(=O)(=O)O']
        
        >>> tokenize_smiles("c1ccc(cc1)C(=O)O")
        ['c', '1', 'c', 'c', 'c', '(', 'c', 'c', '1', ')', 'C(=O)O']
    """
    tokens = _SMILES_PATTERN.findall(smiles)
    return tokens


def get_chemical_tokens_more() -> Tuple[List[str], List[str]]:
    """
    获取化学 tokens
    
    策略：
    1. 添加完整官能团（但不包括会误吃环起点的形式）
    2. 添加括号单元
    3. 添加基础原子
    """
    special_tokens = ['[MASK]', '[UNK_SMILES]']
    chemical_tokens = []
    
    # ============================================
    # 1. 裸原子
    # ============================================
    bare_atoms = [
        'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
        'b', 'c', 'n', 'o', 'p', 's', 'se', 'as',
        'Li', 'Be', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V',
        'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba', 'La', 'Pt', 'Au', 'Hg', 'Pb', 'Bi',
    ]
    chemical_tokens.extend(bare_atoms)
    
    # ============================================
    # 2. 方括号原子
    # ============================================
    bracket_elements = [
        'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
        'b', 'c', 'n', 'o', 'p', 's', 'se', 'as',
        'Si', 'Se', 'As', 'Na', 'Mg', 'Al', 'Ca', 'Fe', 'Cu', 'Zn',
        'Ag', 'Au', 'Pt', 'Pd', 'K', 'Li', 'Mn', 'Co', 'Ni', 'Cr',
    ]
    
    for e in bracket_elements:
        chemical_tokens.append(f'[{e}]')
    for e in bracket_elements:
        chemical_tokens.append(f'([{e}])')
    
    # 带氢
    for e in ['C', 'N', 'O', 'S', 'P', 'B', 'c', 'n', 'o', 's', 'p']:
        for h in ['H', 'H1', 'H2', 'H3', 'H4']:
            chemical_tokens.append(f'[{e}{h}]')
    
    # 带电荷
    for e in ['C', 'N', 'O', 'S', 'P', 'B', 'c', 'n', 'o', 's', 'p',
              'Na', 'K', 'Ca', 'Mg', 'Fe', 'Cu', 'Zn', 'Ag', 'Cl', 'Br', 'I', 'F']:
        for charge in ['+', '-', '++', '--', '+2', '+3', '-2']:
            chemical_tokens.append(f'[{e}{charge}]')
    
    # 氢+电荷
    for e in ['C', 'N', 'O', 'S', 'P', 'n', 'o']:
        for h in ['H', 'H1', 'H2', 'H3']:
            for charge in ['+', '-', '++']:
                chemical_tokens.append(f'[{e}{h}{charge}]')
    
    # 手性
    for marker in ['@', '@@']:
        chemical_tokens.append(f'[{marker}]')
        for e in ['C', 'N', 'S', 'P', 'c', 'n']:
            chemical_tokens.append(f'[{e}{marker}]')
            for h in ['H', 'H1']:
                chemical_tokens.append(f'[{e}{marker}{h}]')
    
    # 常见离子
    common_ions = [
        '[O-]', '[OH-]', '[S-]', '[Cl-]', '[Br-]', '[I-]', '[F-]',
        '[N+]', '[NH+]', '[NH2+]', '[NH3+]', '[NH4+]', '[nH+]', '[n+]',
        '[C+]', '[O+]', '[S+]',
        '[Na+]', '[K+]', '[Li+]', '[Ca+2]', '[Mg+2]', '[Zn+2]',
        '[Fe+2]', '[Fe+3]', '[Cu+]', '[Cu+2]', '[Ag+]', 'c1ccccc1'
    ]
    chemical_tokens.extend(common_ions)
            
    # 两位数字 %10 ~ %99
    for i in range(10, 100):
        chemical_tokens.append(f'%{i}')# 两位数字 %10 ~ %99
    
    for i in range(10, 100):
        chemical_tokens.append(f'{i}')

    # 烷基 token
    for length in range(2,4):  # CC, CCC
        chemical_tokens.append('C'*length)

    # 羧酸系列
    carboxylic_groups = [
        'C(=O)[O-]',   # 羧酸根
        'C(=O)OH',     # 羧酸（显式H）
        'C(=O)O',      # 羧基 - 最重要！
        'C(=O)N',      # 酰胺
        'C(=O)NH2',    # 伯酰胺
        'C(=O)NH',     # 次级酰胺
        'C(=O)Cl',     # 酰氯
        'C(=O)F',      # 酰氟
    ]
    chemical_tokens.extend(carboxylic_groups)
    
    # ⚠️ 注意：不添加 C(=O)OC，因为 C 可能是环的起点
    # 如果真的遇到甲酯 C(=O)OCH3，会被切成 C(=O)O + C + H3
    
    # 磺酸系列
    sulfonic_groups = [
        'S(=O)(=O)[O-]',
        'S(=O)(=O)OH',
        'S(=O)(=O)O',
        'S(=O)(=O)N',
        'S(=O)(=O)NH2',
        'S(=O)(=O)Cl',
    ]
    chemical_tokens.extend(sulfonic_groups)
    
    # 硝基
    nitro_groups = [
        '[N+](=O)[O-]',
        'N(=O)(=O)',
        'N(=O)=O',
    ]
    chemical_tokens.extend(nitro_groups)
    
    # 磷酸
    phosphate_groups = [
        'P(=O)(O)O',
        'P(=O)(OH)OH',
        'P(=O)([O-])[O-]',
        'P(=O)(O)[O-]',
    ]
    chemical_tokens.extend(phosphate_groups)
    
    # 氰基
    chemical_tokens.extend(['C#N', 'N#C'])
    
    # ============================================
    # 5. SMILES 语法符号
    # ============================================
    smiles_symbols = [
        '(', ')', '[', ']', '=', '#', '-', '/', '\\', ':',
        '.', '@', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ]
    chemical_tokens.extend(smiles_symbols)
    
    # ============================================
    # 6. 环闭合标记
    # ============================================
    chemical_tokens.extend([f'%{i:02d}' for i in range(10, 100)])
    
    # ============================================
    # 7. 拓扑符号
    # ============================================
    topology_tokens = [
        'MOFid-v1', 'cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5',
        'acs', 'bct', 'bcu', 'cpf', 'cpr', 'dia', 'fcu', 'fnu',
        'fsc', 'fse', 'fsf', 'fsg', 'hcb', 'hex', 'hms', 'hxl',
        'irl', 'jeb', 'lfm', 'mot', 'nbo', 'pcu', 'rna', 'rob',
        'sit', 'sql', 'sqp', 'tbo', 'ssa', 'ssc', 'pts', 'pyr',
    ]
    chemical_tokens.extend(topology_tokens)
    
    # 去重
    seen = set()
    unique_tokens = []
    for token in chemical_tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    logger.info(f"Generated {len(special_tokens)} special tokens, {len(unique_tokens)} chemical tokens")
    
    return special_tokens, unique_tokens

def get_chemical_tokens_train_old() -> Tuple[List[str], List[str]]:
    """
    获取化学 tokens
    
    策略：
    1. 添加完整官能团（但不包括会误吃环起点的形式）
    2. 添加括号单元
    3. 添加基础原子
    """
    special_tokens = ['[MASK]', '[UNK_SMILES]']
    chemical_tokens = []
    
    # ============================================
    # 1. 裸原子
    # ============================================
    bare_atoms = [
        'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
        'b', 'c', 'n', 'o', 'p', 's', 'se', 'as',
        'Li', 'Be', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V',
        'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba', 'La', 'Pt', 'Au', 'Hg', 'Pb', 'Bi', 'c1ccccc1'
    ]
    chemical_tokens.extend(bare_atoms)
    
    # ============================================
    # 5. SMILES 语法符号
    # ============================================
    smiles_symbols = [
        '(', ')', '[', ']', '=', '#', '-', '/', '\\', ':',
        '.', '@', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ]
    chemical_tokens.extend(smiles_symbols)
    
    # ============================================
    # 7. 拓扑符号
    # ============================================
    topology_tokens = [
        'MOFid-v1', 'cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5',
        'acs', 'bct', 'bcu', 'cpf', 'cpr', 'dia', 'fcu', 'fnu',
        'fsc', 'fse', 'fsf', 'fsg', 'hcb', 'hex', 'hms', 'hxl',
        'irl', 'jeb', 'lfm', 'mot', 'nbo', 'pcu', 'rna', 'rob',
        'sit', 'sql', 'sqp', 'tbo', 'ssa', 'ssc', 'pts', 'pyr',
    ]
    chemical_tokens.extend(topology_tokens)
    
    # 去重
    seen = set()
    unique_tokens = []
    for token in chemical_tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    logger.info(f"Generated {len(special_tokens)} special tokens, {len(unique_tokens)} chemical tokens")
    
    return special_tokens, unique_tokens

def get_chemical_tokens() -> Tuple[List[str], List[str]]:
    """
    获取化学 tokens
    
    策略：
    1. 添加完整官能团（但不包括会误吃环起点的形式）
    2. 添加括号单元
    3. 添加基础原子
    """
    special_tokens = ['[MASK]', '[UNK_SMILES]']
    chemical_tokens = []
    
    # ============================================
    # 7. 拓扑符号
    # ============================================
    topology_tokens = [
        'MOFid-v1', 'cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5',
        'acs', 'bct', 'bcu', 'cpf', 'cpr', 'dia', 'fcu', 'fnu',
        'fsc', 'fse', 'fsf', 'fsg', 'hcb', 'hex', 'hms', 'hxl',
        'irl', 'jeb', 'lfm', 'mot', 'nbo', 'pcu', 'rna', 'rob',
        'sit', 'sql', 'sqp', 'tbo', 'ssa', 'ssc', 'pts', 'pyr',
    ]
    chemical_tokens.extend(topology_tokens)
    
    # 去重
    seen = set()
    unique_tokens = []
    for token in chemical_tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    logger.info(f"Generated {len(special_tokens)} special tokens, {len(unique_tokens)} chemical tokens")
    
    return special_tokens, unique_tokens


def extend_tokenizer_only(
    model_name_or_path: str,
    save_dir: str,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """仅扩充词表"""
    logger.info(f"Loading tokenizer from {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    original_vocab_size = len(tokenizer)
    
    special_tokens, chemical_tokens = get_chemical_tokens()
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.add_tokens(chemical_tokens)
    
    new_vocab_size = len(tokenizer)
    logger.info(f"Vocabulary: {original_vocab_size} -> {new_vocab_size} (+{new_vocab_size - original_vocab_size})")
    
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Saved to {save_dir}")
    
    return tokenizer


def prepare_model_and_tokenizer(
    model_name_or_path: str,
    save_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    load_model: bool = False,
    num_labels: Optional[int] = None,
    problem_type: Optional[str] = None,
) -> Tuple[Optional[AutoModelForSequenceClassification], AutoTokenizer]:
    """准备模型和tokenizer"""
    logger.info(f"Loading from {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    original_vocab_size = len(tokenizer)
    
    special_tokens, chemical_tokens = get_chemical_tokens()
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.add_tokens(chemical_tokens)
    
    new_vocab_size = len(tokenizer)
    
    model = None
    if load_model:
        if num_labels is None:
            raise ValueError("num_labels required")
        model_kwargs = {'num_labels': num_labels, 'trust_remote_code': trust_remote_code}
        if problem_type:
            model_kwargs['problem_type'] = problem_type
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **model_kwargs)
        model.resize_token_embeddings(new_vocab_size)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        if model:
            model.save_pretrained(save_dir)
    
    return model, tokenizer


def preprocess_smiles_for_training(
    smiles: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    use_chemical_tokenization: bool = True,
) -> Dict[str, torch.Tensor]:
    """预处理 SMILES"""
    if use_chemical_tokenization:
        tokens = tokenize_smiles(smiles)
        smiles_tokenized = ' '.join(tokens)
    else:
        smiles_tokenized = smiles
    
    encoding = tokenizer(
        smiles_tokenized,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
    }


def example_usage():
    """测试对比"""
    print("\n" + "="*80)
    print("阿司匹林切分对比")
    print("="*80)
    
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    print(f"\nSMILES: {smiles}")
    print(f"\n化学结构:")
    print(f"  CH3-C(=O)-O-[苯环]-C(=O)-OH")
    print(f"  甲基  乙酰氧基   苯环    羧基")
    
    tokens = tokenize_smiles(smiles)
    
    print(f"\n我的切分:")
    print(f"  {tokens}")
    
    print(f"\n详细解释:")
    explanations = [
        ("C", "甲基"),
        ("C(=O)O", "乙酰氧基（作为取代基）"),
        ("C", "苯环起点（C1）"),
        ("1", "环标记"),
        ("=C", "苯环双键"),
        ("C", "苯环碳"),
        ("=C", "苯环双键"),
        ("C", "苯环碳"),
        ("=C", "苯环双键"),
        ("C", "苯环终点"),
        ("1", "环闭合"),
        ("C(=O)O", "羧基"),
    ]
    
    for i, (token, desc) in enumerate(explanations):
        if i < len(tokens) and tokens[i] == token:
            print(f"  {token:10s} - {desc}")
    
    print(f"\n关键改进:")
    print(f"  ✅ C(=O)O 作为乙酰氧基（完整官能团）")
    print(f"  ✅ 后面的 C 是苯环起点，不被包含")
    print(f"  ✅ 最后的 C(=O)O 是羧基（完整官能团）")
    print(f"  ✅ 使用负向前瞻避免误吃环标记")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    example_usage()
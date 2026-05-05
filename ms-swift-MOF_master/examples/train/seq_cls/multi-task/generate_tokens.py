#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成化学 tokens 文件
"""

def generate_chemical_tokens():
    """生成所有化学 tokens"""
    chemical_tokens = []
    
    # 特殊标记
    special_tokens = ['[MASK]', '[UNK_SMILES]']
    chemical_tokens.extend(special_tokens)
    
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
        chemical_tokens.append(f'%{i}')
    
    # 两位数字 10 ~ 99
    for i in range(10, 100):
        chemical_tokens.append(f'{i}')

    # 烷基 token
    for length in range(2, 4):  # CC, CCC
        chemical_tokens.append('C' * length)

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
    
    # 去重（保持顺序）
    seen = set()
    unique_tokens = []
    for token in chemical_tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    return unique_tokens


def main():
    tokens = generate_chemical_tokens()
    
    # 写入文件
    output_file = '/home/liuhongye/material_LLM/material_LLM_experiment/data_mid_CO2/get_jsonl/chemical_tokens.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + '\n')
    
    print(f"已生成 {len(tokens)} 个 tokens，保存到 {output_file}")


if __name__ == '__main__':
    main()

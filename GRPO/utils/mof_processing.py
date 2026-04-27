#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import shutil
import math
import re
import logging
import tempfile
from pathlib import Path
from collections import defaultdict
import hashlib

from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.io import write

# ===== 日志设置 =====
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("mof_processor_function")

# 金属元素列表
METAL_ELEMENTS = {
    "Fe", "Cu", "Zn", "Ni", "Co", "Mn", "Mg", "Ca", "Cd", "Cr",
    "Al", "Ti", "V", "Zr", "Hf", "Ag", "Au", "Pt", "Pd", "Ir", "Rh",
    "Li", "Na", "K", "Rb", "Cs", "Be", "Sr", "Ba", "Sc", "Y", "La"
}

# ===== Helper Functions =====

def sanitize_filename(name, max_length=50):
    """
    清理文件名，移除所有特殊字符
    
    Args:
        name: 原始名称
        max_length: 最大长度
    
    Returns:
        安全的文件名
    """
    # 移除所有非字母数字字符（保留下划线和连字符）
    clean = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    
    # 移除开头的数字（防止以数字开头）
    clean = re.sub(r'^[0-9]+', '', clean)
    
    # 如果清理后为空，使用默认名称
    if not clean:
        clean = "linker"
    
    # 限制长度
    if len(clean) > max_length:
        clean = clean[:max_length]
    
    return clean


def generate_safe_name(smiles, index=None, use_hash=True):
    """
    为 SMILES 生成安全的文件名
    
    Args:
        smiles: SMILES 字符串
        index: 可选的序号
        use_hash: 是否添加hash确保唯一性
    
    Returns:
        安全的文件名（不含扩展名）
    """
    # 方案1: 提取有意义的部分（去除特殊字符）
    clean = sanitize_filename(smiles, max_length=30)
    
    # 方案2: 添加hash确保唯一性
    if use_hash:
        hash_suffix = hashlib.md5(smiles.encode()).hexdigest()[:6]
        clean = f"{clean}_{hash_suffix}"
    
    # 方案3: 添加序号
    if index is not None:
        clean = f"{clean}_{index}"
    
    return clean


def create_smiles_mapping(smiles_list):
    """
    创建 SMILES 到安全文件名的映射
    
    Args:
        smiles_list: SMILES 字符串列表
    
    Returns:
        dict: {原始SMILES: 安全文件名}
    """
    mapping = {}
    counter = defaultdict(int)
    
    for smiles in smiles_list:
        # 生成基础名称
        base_name = generate_safe_name(smiles, use_hash=True)
        
        # 处理重复
        counter[base_name] += 1
        if counter[base_name] == 1:
            safe_name = base_name
        else:
            safe_name = f"{base_name}_{counter[base_name]}"
        
        mapping[smiles] = safe_name
    
    return mapping

def get_smiles_to_cif(smiles):
    """将 SMILES 转为 ASE Atoms 对象"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            return None
        
        AllChem.UFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        symbols = []
        positions = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            symbols.append(atom.GetSymbol())
        
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.set_cell([40.0, 40.0, 40.0])
        atoms.set_pbc([True, True, True])
        return atoms
    except Exception as e:
        logger.error(f"SMILES to CIF conversion failed: {e}")
        return None


def smiles_to_mol(smiles_str, output_path):
    """将 SMILES 字符串转为 mol 文件（含3D结构）"""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise ValueError("无法解析SMILES字符串: " + smiles_str)
    
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42      # 固定随机种子，确保结果可重复
    AllChem.EmbedMolecule(mol, params)
    AllChem.UFFOptimizeMolecule(mol)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Chem.MolToMolFile(mol, output_path)


def parse_mol_file(path):
    """解析 V2000 mol 文件"""
    atoms = []
    bonds = []
    charges = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\n') for ln in f]
    
    if not lines:
        return atoms, bonds, charges
    
    counts_idx = None
    for i in range(min(10, len(lines))):
        if 'V2000' in lines[i] or 'v2000' in lines[i]:
            counts_idx = i
            break
    if counts_idx is None:
        counts_idx = 3 if len(lines) > 3 else 0
    counts_line = lines[counts_idx]
    
    def safe_int_slice(s, start, end):
        try:
            return int(s[start:end].strip())
        except:
            return None
    
    natoms = safe_int_slice(counts_line, 0, 3)
    nbonds = safe_int_slice(counts_line, 3, 6)
    if natoms is None or nbonds is None:
        parts = counts_line.split()
        if len(parts) >= 2:
            try:
                natoms = int(parts[0])
                nbonds = int(parts[1])
            except:
                raise ValueError(f"Cannot parse counts line: {counts_line}")
        else:
            raise ValueError(f"Cannot parse counts line: {counts_line}")
    
    atom_start = counts_idx + 1
    atom_end = atom_start + natoms
    if atom_end > len(lines):
        raise ValueError(f"File too short for declared atoms: {path}")
    
    for i in range(atom_start, atom_end):
        ln = lines[i]
        toks = ln.split()
        if len(toks) < 4:
            raise ValueError(f"Cannot parse atom line: '{ln}' in {path}")
        x = float(toks[0])
        y = float(toks[1])
        z = float(toks[2])
        elem = toks[3]
        atoms.append({'index': len(atoms)+1, 'x': x, 'y': y, 'z': z, 'element': elem, 'raw': ln})
    
    bond_start = atom_end
    bond_end = bond_start + nbonds
    for i in range(bond_start, min(bond_end, len(lines))):
        ln = lines[i]
        toks = ln.split()
        if len(toks) < 3:
            continue
        a = int(toks[0])
        b = int(toks[1])
        order = int(toks[2])
        bonds.append({'a': a, 'b': b, 'order': order})
    
    for ln in lines[bond_end:]:
        if ln.startswith('M  CHG'):
            toks = ln.split()
            if len(toks) >= 3:
                try:
                    n = int(toks[2])
                    pairs = toks[3:]
                    for j in range(0, len(pairs), 2):
                        idx = int(pairs[j])
                        chg = int(pairs[j+1])
                        charges[idx] = chg
                except Exception:
                    rest = toks[3:]
                    for j in range(0, len(rest), 2):
                        try:
                            idx = int(rest[j])
                            chg = int(rest[j+1])
                            charges[idx] = chg
                        except:
                            pass
    return atoms, bonds, charges


def compute_distance(a, b):
    dx = a['x'] - b['x']
    dy = a['y'] - b['y']
    dz = a['z'] - b['z']
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def make_labels(atoms):
    """
    为原子生成标签（全局连续标号）
    
    Args:
        atoms: 原子列表，每个原子包含 'index' 和 'element'
    
    Returns:
        dict: {原子index: 标签字符串}
    
    示例:
        atoms = [
            {'index': 1, 'element': 'X'},
            {'index': 2, 'element': 'C'},
            {'index': 3, 'element': 'C'},
            {'index': 4, 'element': 'O'}
        ]
        返回: {1: 'X1', 2: 'C2', 3: 'C3', 4: 'O4'}
    """
    labels = {}
    for a in atoms:
        elem = a['element'].capitalize()
        idx = a['index']
        # 使用全局索引，而不是按元素类型分组计数
        labels[idx] = f"{elem}{idx}"
    return labels


def bond_order_to_type(order):
    return {1:'A', 2:'D', 3:'T'}.get(order, 'A')


CIF_TEMPLATE_TOP = """data_L_20 
_audit_creation_date              2014-09-12
_audit_creation_method            'Materials Studio'
_symmetry_space_group_name_H-M    'P1'
_symmetry_Int_Tables_number       1
_symmetry_cell_setting            triclinic
loop_
_symmetry_equiv_pos_as_xyz
  x,y,z
_cell_length_a                    {a:.4f}
_cell_length_b                    {b:.4f}
_cell_length_c                    {c:.4f}
_cell_angle_alpha                 {alpha:.4f}
_cell_angle_beta                  {beta:.4f}
_cell_angle_gamma                 {gamma:.4f}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
_atom_site_adp_type
_atom_site_occupancy
"""

CIF_BOND_HEADER = """
loop_
_geom_bond_atom_site_label_1
_geom_bond_atom_site_label_2
_geom_bond_distance
_geom_bond_site_symmetry_2
_ccdc_geom_bond_type
"""

CELL = (20.0, 20.0, 20.0, 90.0, 90.0, 90.0)


def read_cell_from_cif_path(cif_path):
    """从指定 cif 文件读取晶胞参数"""
    if not cif_path or not os.path.isfile(cif_path):
        return None
    try:
        with open(cif_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception:
        return None
    
    def find_val(key):
        pattern = r"^{k}\s+([+-]?\d*\.\d+|\d+)".format(k=re.escape(key))
        m = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except:
                return None
        pattern2 = r"^{k}\s*=\s*([+-]?\d*\.\d+|\d+)".format(k=re.escape(key))
        m2 = re.search(pattern2, text, flags=re.MULTILINE | re.IGNORECASE)
        if m2:
            try:
                return float(m2.group(1))
            except:
                return None
        return None
    
    a = find_val('_cell_length_a')
    b = find_val('_cell_length_b')
    c = find_val('_cell_length_c')
    alpha = find_val('_cell_angle_alpha')
    beta = find_val('_cell_angle_beta')
    gamma = find_val('_cell_angle_gamma')
    
    if None in (a, b, c, alpha, beta, gamma):
        return None
    return (a, b, c, alpha, beta, gamma)


def write_cif(outpath, atoms, bonds, charges, cell=CELL, mass_weighted=True):
    """
    写入CIF文件
    注意：atoms中的xyz应该已经是笛卡尔坐标或分数坐标
    如果是笛卡尔坐标，需要转换为分数坐标但不进行中心化
    """
    a, b, c, alpha, beta, gamma = cell
    labels = make_labels(atoms)
    
    # 直接将坐标转换为分数坐标，不进行中心化
    # 假设输入的xyz是笛卡尔坐标（埃）
    fracs = {}
    for at in atoms:
        # 简单的转换：直接除以晶胞参数（适用于正交晶胞）
        fx = at['x'] / a
        fy = at['y'] / b
        fz = at['z'] / c
        fracs[at['index']] = (fx, fy, fz)
    
    with open(outpath, 'w', encoding='utf-8') as fw:
        fw.write(CIF_TEMPLATE_TOP.format(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma))
        for at in atoms:
            idx = at['index']
            lbl = labels[idx]
            sym = at['element'].capitalize()
            fx, fy, fz = fracs[idx]
            fw.write(f"{lbl:<6} {sym:<5} {fx:8.5f} {fy:8.5f} {fz:8.5f}   0.00000  Uiso   1.00\n")
        
        fw.write(CIF_BOND_HEADER)
        for bd in bonds:
            aidx = bd['a']
            bidx = bd['b']
            a_lbl = labels.get(aidx, f"A{aidx}")
            b_lbl = labels.get(bidx, f"A{bidx}")
            a_at = next(x for x in atoms if x['index']==aidx)
            b_at = next(x for x in atoms if x['index']==bidx)
            dist = compute_distance(a_at, b_at)
            bond_type = bond_order_to_type(bd.get('order', 1))
            fw.write(f"{a_lbl:<6} {b_lbl:<6} {dist:6.3f}   .     {bond_type}\n")
        
        if charges:
            fw.write("\n# M CHG parsed (index -> charge):\n")
            for idx, ch in charges.items():
                lbl = labels.get(idx, f"A{idx}")
                fw.write(f"#   {lbl}: {ch}\n")

def parse_mol_v2000(path):
    """解析常规的 MDL V2000 mol 文件，返回 atoms, bond_sums, carbons, nitrogens"""
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(text) < 4:
        raise ValueError("mol 文件行数太少，非标准 mol 文件？")
    
    counts_line = text[3]
    try:
        atoms_count = int(counts_line[0:3].strip())
        bonds_count = int(counts_line[3:6].strip())
    except Exception:
        parts = counts_line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            atoms_count = int(parts[0])
            bonds_count = int(parts[1])
        else:
            raise ValueError(f"无法解析 counts line: {counts_line!r}")
    
    expected_min = 4 + atoms_count + bonds_count
    if len(text) < expected_min:
        raise ValueError(f"mol 文件行数 {len(text)} 少于预期 {expected_min}")
    
    atoms = []
    for i in range(atoms_count):
        line = text[4 + i]
        toks = line.split()
        if len(toks) < 4:
            if len(line) >= 30:
                try:
                    x = float(line[0:10].strip())
                    y = float(line[10:20].strip())
                    z = float(line[20:30].strip())
                    element = line[31:34].strip() if len(line) >= 34 else ""
                    atoms.append({"element": element, "xyz": (x, y, z)})
                    continue
                except Exception:
                    raise ValueError(f"atom 行无法解析: {line!r}")
            else:
                raise ValueError(f"atom 行无法解析: {line!r}")
        try:
            x = float(toks[0])
            y = float(toks[1])
            z = float(toks[2])
        except Exception:
            try:
                x = float(line[0:10].strip())
                y = float(line[10:20].strip())
                z = float(line[20:30].strip())
            except Exception:
                raise
        element = toks[3].strip()
        atoms.append({"element": element, "xyz": (x, y, z)})
    
    bonds = []
    for j in range(bonds_count):
        line = text[4 + atoms_count + j]
        toks = line.split()
        if len(toks) < 3:
            try:
                a1 = int(line[0:3].strip())
                a2 = int(line[3:6].strip())
                btype = int(line[6:9].strip())
            except Exception:
                raise ValueError(f"bond 行无法解析: {line!r}")
        else:
            a1 = int(toks[0])
            a2 = int(toks[1])
            btype = int(toks[2])
        bonds.append((a1 - 1, a2 - 1, btype))
    
    # 计算每个原子的键数（不包括与H的键）- 用于选择连接点
    bond_sums = [0] * len(atoms)
    # 计算每个原子的总键数（包括与H的键）- 用于判断是否真正饱和
    total_bond_sums = [0] * len(atoms)
    
    for a, b, bo in bonds:
        try:
            bo_int = int(bo)
        except Exception:
            bo_int = 1
        
        # 检查是否是与H原子的键
        atom_a_is_h = atoms[a]["element"].upper() == "H" if 0 <= a < len(atoms) else False
        atom_b_is_h = atoms[b]["element"].upper() == "H" if 0 <= b < len(atoms) else False
        
        # 总键数（包括H）
        if 0 <= a < len(total_bond_sums):
            total_bond_sums[a] += bo_int
        if 0 <= b < len(total_bond_sums):
            total_bond_sums[b] += bo_int
        
        # 只有当键的两端都不是H时才计入bond_sums
        if not atom_a_is_h and not atom_b_is_h:
            if 0 <= a < len(bond_sums):
                bond_sums[a] += bo_int
            if 0 <= b < len(bond_sums):
                bond_sums[b] += bo_int
    
    # 收集碳原子
    carbons = []
    for idx, atom in enumerate(atoms):
        if atom["element"].upper() == "C":
            carbons.append({
                "list1_index": idx,
                "element": "C",
                "xyz": atom["xyz"],
                "bond_sum": bond_sums[idx],
                "total_bond_sum": total_bond_sums[idx]  # 新增：包括H的总键数
            })
    
    # 收集氮原子
    nitrogens = []
    for idx, atom in enumerate(atoms):
        if atom["element"].upper() == "N":
            nitrogens.append({
                "list1_index": idx,
                "element": "N",
                "xyz": atom["xyz"],
                "bond_sum": bond_sums[idx],
                "total_bond_sum": total_bond_sums[idx]  # 新增：包括H的总键数
            })
    
    return atoms, bonds, bond_sums, carbons, nitrogens


def distance(a_xyz, b_xyz):
    ax, ay, az = a_xyz
    bx, by, bz = b_xyz
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def find_connection_atoms(carbons, nitrogens, atoms, bonds, mol_path):
    """
    找到最远的两个C/N原子作为连接点
    先找距离最远的两个，检查是否不饱和或可以通过移除H变为不饱和
    如果不行，找次远的，以此类推
    
    返回: (atom1_idx_in_list2, atom2_idx_in_list2, distance, element_type)
          或 None（如果找不到合适的原子对）
    """
    # 合并所有候选原子（C和N）
    all_candidates = []
    for i, c in enumerate(carbons):
        all_candidates.append((i, c, "C", 4))  # (index, atom_info, type, max_bonds)
    for i, n in enumerate(nitrogens):
        all_candidates.append((i, n, "N", 3))  # (index, atom_info, type, max_bonds)
    
    if len(all_candidates) < 2:
        logger.error(f"  候选原子数量不足（需要至少2个C或N原子）")
        return None
    
    # 计算所有原子对的距离并排序（从远到近）
    atom_pairs = []
    for a in range(len(all_candidates)):
        for b in range(a + 1, len(all_candidates)):
            idx_i, atom_i, type_i, max_i = all_candidates[a]
            idx_j, atom_j, type_j, max_j = all_candidates[b]
            d = distance(atom_i["xyz"], atom_j["xyz"])
            atom_pairs.append((d, idx_i, idx_j, atom_i, atom_j, type_i, type_j, max_i, max_j))
    
    # 按距离从大到小排序
    atom_pairs.sort(reverse=True, key=lambda x: x[0])
    
    # 尝试每一对原子，从最远的开始
    for dist, idx_i, idx_j, atom_i, atom_j, type_i, type_j, max_i, max_j in atom_pairs:
        atom_i_idx = atom_i["list1_index"]
        atom_j_idx = atom_j["list1_index"]
        bond_sum_i = atom_i.get("bond_sum", 0)  # 不含H的键数
        bond_sum_j = atom_j.get("bond_sum", 0)  # 不含H的键数
        total_bond_sum_i = atom_i.get("total_bond_sum", 0)  # 含H的总键数
        total_bond_sum_j = atom_j.get("total_bond_sum", 0)  # 含H的总键数
        
        # 使用总键数判断是否真正饱和
        is_i_truly_saturated = total_bond_sum_i >= max_i
        is_j_truly_saturated = total_bond_sum_j >= max_j
        
        # 使用不含H的键数判断是否可以作为连接点
        is_i_unsaturated = bond_sum_i < max_i
        is_j_unsaturated = bond_sum_j < max_j
        
        # 情况1：两个都可以直接作为连接点（重原子键数未饱和）
        if is_i_unsaturated and is_j_unsaturated:
            # 但仍需检查：如果总键数饱和（有H），需要删除H腾出位置
            h_bond_i = None
            h_bond_j = None
            
            # 检查原子i：如果总键数饱和，删除一个H
            if is_i_truly_saturated:
                h_bonds_i = [(a, b, bo) for a, b, bo in bonds 
                            if (a == atom_i_idx and atoms[b]["element"].upper() == "H") or
                               (b == atom_i_idx and atoms[a]["element"].upper() == "H")]
                if len(h_bonds_i) > 0:
                    h_bond_i = h_bonds_i[0]
                    bonds.remove(h_bond_i)
                    atom_i["total_bond_sum"] = total_bond_sum_i - 1
            
            # 检查原子j：如果总键数饱和，删除一个H
            if is_j_truly_saturated:
                h_bonds_j = [(a, b, bo) for a, b, bo in bonds 
                            if (a == atom_j_idx and atoms[b]["element"].upper() == "H") or
                               (b == atom_j_idx and atoms[a]["element"].upper() == "H")]
                if len(h_bonds_j) > 0:
                    h_bond_j = h_bonds_j[0]
                    bonds.remove(h_bond_j)
                    atom_j["total_bond_sum"] = total_bond_sum_j - 1
            
            logger.info(f"  ✓ 找到不饱和原子对: {type_i}{idx_i+1}, {type_j}{idx_j+1}, 距离={dist:.4f}")
            element_types = f"{type_i},{type_j}"
            # 返回atoms_list中的实际索引（从0开始）
            return (atom_i_idx, atom_j_idx, dist, element_types)
        
        # 情况2：至少一个重原子键数已饱和，需要检查是否可以通过移除H来解决
        can_use_i = is_i_unsaturated
        can_use_j = is_j_unsaturated
        h_bond_i = None
        h_bond_j = None
        
        # 检查原子i：如果重原子键数饱和，但总键数也饱和（说明有H），可以移除H
        if not is_i_unsaturated and is_i_truly_saturated:
            # 查找与H的键
            h_bonds_i = [(a, b, bo) for a, b, bo in bonds 
                        if (a == atom_i_idx and atoms[b]["element"].upper() == "H") or
                           (b == atom_i_idx and atoms[a]["element"].upper() == "H")]
            if len(h_bonds_i) > 0:
                h_bond_i = h_bonds_i[0]
                can_use_i = True
        
        # 检查原子j：如果重原子键数饱和，但总键数也饱和（说明有H），可以移除H
        if not is_j_unsaturated and is_j_truly_saturated:
            # 查找与H的键
            h_bonds_j = [(a, b, bo) for a, b, bo in bonds 
                        if (a == atom_j_idx and atoms[b]["element"].upper() == "H") or
                           (b == atom_j_idx and atoms[a]["element"].upper() == "H")]
            if len(h_bonds_j) > 0:
                h_bond_j = h_bonds_j[0]
                can_use_j = True
        
        # 如果两个原子都可以使用（要么重原子键不饱和，要么有H可移除）
        if can_use_i and can_use_j:
            # 移除需要的H原子
            if h_bond_i is not None:
                bonds.remove(h_bond_i)
                atom_i["total_bond_sum"] = total_bond_sum_i - 1
            
            if h_bond_j is not None:
                bonds.remove(h_bond_j)
                atom_j["total_bond_sum"] = total_bond_sum_j - 1
            
            logger.info(f"  ✓ 选择原子对: {type_i}{idx_i+1}, {type_j}{idx_j+1}, 距离={dist:.4f}")
            element_types = f"{type_i},{type_j}"
            # 返回atoms_list中的实际索引（从0开始）
            return (atom_i_idx, atom_j_idx, dist, element_types)
    
    # 遍历完所有原子对都无法满足条件
    logger.error(f"  未找到合适的原子对（所有候选对都无法变为不饱和状态）")
    return None


'''def modify_cif_labels(cif_path, cn_pos_in_list2, cm_pos_in_list2, element_types):
    """
    修改 CIF 文件，将指定位置的 C/N 替换为 X
    cn_pos_in_list2, cm_pos_in_list2: 元素在其类型中的序号（如C的第3个，N的第2个）
    element_types: 如 "C,C" 或 "C,N" 或 "N,N"
    """
    cif_path = Path(cif_path)
    if not cif_path.exists():
        logger.warning(f"modify_cif_labels: CIF not found: {cif_path}")
        return False
    
    try:
        text_lines = cif_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        logger.error(f"modify_cif_labels: failed to read {cif_path}: {e}")
        return False
    
    loop_indices = [idx for idx, ln in enumerate(text_lines) if ln.strip().lower().startswith("loop_")]
    
    if len(loop_indices) < 2:
        logger.info(f"modify_cif_labels: fewer than 2 'loop_' blocks in {cif_path}; nothing to modify.")
        try:
            cif_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"modify_cif_labels: failed to re-write {cif_path}: {e}")
            return False
    
    # 解析元素类型
    types = element_types.split(',')
    elem1 = types[0].strip() if len(types) > 0 else "C"
    elem2 = types[1].strip() if len(types) > 1 else "C"
    
    s_cn = str(int(cn_pos_in_list2))
    s_cm = str(int(cm_pos_in_list2))
    
    logger.info(f"    查找标签: {elem1}{s_cn} 和 {elem2}{s_cm}")
    
    def startswith_Elem_num_at(line, pos, elem, num_str):
        """检查line[pos]是否以元素+数字开头"""
        if pos < 0 or pos >= len(line):
            return False
        try:
            return line[pos] == elem and line.startswith(elem + num_str, pos)
        except Exception:
            return False
    
    idx_second_loop = loop_indices[1]
    idx_third_loop = loop_indices[2] if len(loop_indices) >= 3 else None
    
    start_r1 = idx_second_loop + 1
    end_r1 = (idx_third_loop - 1) if idx_third_loop is not None else len(text_lines) - 1
    
    out_lines = list(text_lines)
    modified_count = 0
    
    # 修改第二个loop区域（原子定义区）
    if start_r1 <= end_r1:
        for ri in range(start_r1, end_r1 + 1):
            line = out_lines[ri]
            # 检查是否匹配第一个原子
            if startswith_Elem_num_at(line, 0, elem1, s_cn):
                chars = list(line)
                if len(chars) > 0:
                    chars[0] = "X"
                    out_lines[ri] = "".join(chars)
                    modified_count += 1
                    logger.info(f"    在第{ri}行找到并修改: {elem1}{s_cn} → X{s_cn}")
            # 检查是否匹配第二个原子
            elif startswith_Elem_num_at(line, 0, elem2, s_cm):
                chars = list(line)
                if len(chars) > 0:
                    chars[0] = "X"
                    out_lines[ri] = "".join(chars)
                    modified_count += 1
                    logger.info(f"    在第{ri}行找到并修改: {elem2}{s_cm} → X{s_cm}")
    
    # 修改第三个loop区域（键定义区，如果存在）
    if idx_third_loop is not None:
        start_r2 = idx_third_loop + 1
        next_loop_after_third = None
        for li in loop_indices:
            if li > idx_third_loop:
                next_loop_after_third = li
                break
        end_r2 = (next_loop_after_third - 1) if next_loop_after_third is not None else len(text_lines) - 1
        
        if start_r2 <= end_r2:
            for ri in range(start_r2, end_r2 + 1):
                line = out_lines[ri]
                
                # 位置1检查（第一个原子标签）
                if startswith_Elem_num_at(line, 0, elem1, s_cn):
                    chars = list(line)
                    if len(chars) > 0:
                        chars[0] = "X"
                        out_lines[ri] = "".join(chars)
                        logger.info(f"    在第{ri}行(键)找到并修改位置0: {elem1}{s_cn} → X{s_cn}")
                elif startswith_Elem_num_at(line, 0, elem2, s_cm):
                    chars = list(line)
                    if len(chars) > 0:
                        chars[0] = "X"
                        out_lines[ri] = "".join(chars)
                        logger.info(f"    在第{ri}行(键)找到并修改位置0: {elem2}{s_cm} → X{s_cm}")
                
                # 位置2检查（第二个原子标签）- 需要重新获取最新的line
                line = out_lines[ri]
                if startswith_Elem_num_at(line, 7, elem1, s_cn):
                    chars = list(line)
                    if len(chars) <= 7:
                        chars.extend([" "] * (8 - len(chars)))
                    chars[7] = "X"
                    out_lines[ri] = "".join(chars)
                    logger.info(f"    在第{ri}行(键)找到并修改位置7: {elem1}{s_cn} → X{s_cn}")
                elif startswith_Elem_num_at(line, 7, elem2, s_cm):
                    chars = list(line)
                    if len(chars) <= 7:
                        chars.extend([" "] * (8 - len(chars)))
                    chars[7] = "X"
                    out_lines[ri] = "".join(chars)
                    logger.info(f"    在第{ri}行(键)找到并修改位置7: {elem2}{s_cm} → X{s_cm}")
    
    logger.info(f"    在原子定义区修改了 {modified_count} 处")
    
    try:
        cif_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        logger.info(f"modify_cif_labels: written modified file {cif_path}")
        return True
    except Exception as e:
        logger.error(f"modify_cif_labels: failed to write {cif_path}: {e}")
        return False'''

def modify_cif_labels(cif_path, cn_global_index, cm_global_index, element_types):
    """
    修改 CIF 文件，将指定全局索引的原子替换为 X
    
    Args:
        cif_path: CIF 文件路径
        cn_global_index: 第一个连接原子的全局索引（例如 1）
        cm_global_index: 第二个连接原子的全局索引（例如 10）
        element_types: 元素类型字符串，如 "C,C" 或 "C,N"（用于日志）
    
    Returns:
        bool: 修改是否成功
    
    示例:
        cn_global_index=1, cm_global_index=10
        将修改: X1 (已是X) 和 C10 → X10
    """
    cif_path = Path(cif_path)
    if not cif_path.exists():
        logger.warning(f"modify_cif_labels: CIF not found: {cif_path}")
        return False
    
    try:
        text_lines = cif_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        logger.error(f"modify_cif_labels: failed to read {cif_path}: {e}")
        return False
    
    # 解析元素类型（仅用于日志）
    types = element_types.split(',')
    elem1 = types[0].strip() if len(types) > 0 else "C"
    elem2 = types[1].strip() if len(types) > 1 else "C"
    
    logger.info(f"    目标全局索引: {cn_global_index} ({elem1}) 和 {cm_global_index} ({elem2})")
    
    # 找到所有 loop_ 的位置
    loop_indices = [idx for idx, ln in enumerate(text_lines) if ln.strip().lower().startswith("loop_")]
    
    if len(loop_indices) < 2:
        logger.info(f"modify_cif_labels: fewer than 2 'loop_' blocks in {cif_path}; nothing to modify.")
        return True
    
    # 第二个 loop 是原子定义区
    idx_second_loop = loop_indices[1]
    idx_third_loop = loop_indices[2] if len(loop_indices) >= 3 else None
    
    # 确定原子定义区的范围
    # 原子定义区从第二个 loop_ 后面开始，到第三个 loop_ 前面（或文件末尾）
    # 需要跳过 loop_ 后的列名行
    atom_start = idx_second_loop + 1
    
    # 跳过列名行（以 _atom_site 开头的行）
    while atom_start < len(text_lines) and text_lines[atom_start].strip().startswith('_'):
        atom_start += 1
    
    atom_end = (idx_third_loop - 1) if idx_third_loop is not None else len(text_lines) - 1
    
    out_lines = list(text_lines)
    modified_count = 0
    
    # 修改原子定义区
    logger.info(f"    扫描原子定义区: 行 {atom_start} 到 {atom_end}")
    
    for ri in range(atom_start, atom_end + 1):
        line = out_lines[ri].strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
        
        # 第一列是标签（如 C10, N5, X1）
        label = parts[0]
        
        # 提取标签中的数字部分
        import re
        match = re.match(r'([A-Za-z]+)(\d+)', label)
        if not match:
            continue
        
        element_part = match.group(1)
        index_part = int(match.group(2))
        
        # 检查是否是目标索引
        if index_part == cn_global_index or index_part == cm_global_index:
            # 替换元素部分为 X
            new_label = f"X{index_part}"
            
            # 替换整行中的标签（只替换第一个出现的标签）
            new_line = line.replace(label, new_label, 1)
            out_lines[ri] = new_line
            
            modified_count += 1
            logger.info(f"    ✓ 第{ri}行: {label} → {new_label}")
    
    # 修改键定义区（如果存在）
    if idx_third_loop is not None:
        bond_start = idx_third_loop + 1
        
        # 跳过列名行
        while bond_start < len(text_lines) and text_lines[bond_start].strip().startswith('_'):
            bond_start += 1
        
        # 找到下一个 loop_ 或文件末尾
        next_loop_after_third = None
        for li in loop_indices:
            if li > idx_third_loop:
                next_loop_after_third = li
                break
        bond_end = (next_loop_after_third - 1) if next_loop_after_third is not None else len(text_lines) - 1
        
        logger.info(f"    扫描键定义区: 行 {bond_start} 到 {bond_end}")
        
        for ri in range(bond_start, bond_end + 1):
            line = out_lines[ri].strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # 键定义：label1 label2 distance symmetry type
            label1 = parts[0]
            label2 = parts[1]
            
            modified_this_line = False
            new_line = line
            
            # 检查并替换 label1
            match1 = re.match(r'([A-Za-z]+)(\d+)', label1)
            if match1:
                idx1 = int(match1.group(2))
                if idx1 == cn_global_index or idx1 == cm_global_index:
                    new_label1 = f"X{idx1}"
                    new_line = new_line.replace(label1, new_label1, 1)
                    modified_this_line = True
            
            # 检查并替换 label2
            match2 = re.match(r'([A-Za-z]+)(\d+)', label2)
            if match2:
                idx2 = int(match2.group(2))
                if idx2 == cn_global_index or idx2 == cm_global_index:
                    new_label2 = f"X{idx2}"
                    # 只替换第二次出现（label2的位置）
                    parts_temp = new_line.split()
                    if len(parts_temp) >= 2:
                        parts_temp[1] = parts_temp[1].replace(label2, new_label2, 1)
                        new_line = ' '.join(parts_temp)
                    modified_this_line = True
            
            if modified_this_line:
                out_lines[ri] = new_line
                logger.info(f"    ✓ 第{ri}行(键): {line[:50]}... → {new_line[:50]}...")
    
    logger.info(f"    总共修改了 {modified_count} 个原子标签")
    
    # 写回文件
    try:
        cif_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        logger.info(f"    ✓ 已保存修改后的文件: {cif_path.name}")
        return True
    except Exception as e:
        logger.error(f"modify_cif_labels: failed to write {cif_path}: {e}")
        return False

# ===== Main Processing Function =====
def process_mof_smiles(smiles_input: str) -> int:
    """
    处理单个 MOF SMILES 字符串，生成结构文件
    
    Args:
        smiles_input: SMILES 字符串，格式如：
                     "CCc1cc(...) MOFid-v1.pcu.cat0"
    
    Returns:
        int: 1 表示成功，0 表示失败
    """
    logger.info("=" * 80)
    logger.info("开始处理 MOF SMILES")
    logger.info("=" * 80)
    
    # 创建临时工作目录
    temp_base = tempfile.mkdtemp(prefix="mof_processing_")
    logger.info(f"临时工作目录: {temp_base}")
    
    try:
        # 解析输入
        parts = smiles_input.strip().split()
        if len(parts) < 2:
            logger.error(f"输入格式错误: {smiles_input}")
            return 0
        
        smiles_prefix = parts[0]
        topology_info = parts[1]
        
        # 解析拓扑信息
        topo_parts = topology_info.split('.')
        if len(topo_parts) < 2:
            logger.error(f"拓扑格式错误: {topology_info}")
            return 0
        
        # 定义目录结构
        EDGES_C_ROOT = os.path.join(temp_base, "edges_C")
        EDGES_MOL_ROOT = os.path.join(temp_base, "edges_mol")
        EDGES_CIF_ROOT = os.path.join(temp_base, "edges_cif")
        INVALID_CSV = os.path.join(temp_base, "invalid_linkers.csv")
        
        # 创建目录
        os.makedirs(EDGES_C_ROOT, exist_ok=True)
        os.makedirs(EDGES_MOL_ROOT, exist_ok=True)
        os.makedirs(EDGES_CIF_ROOT, exist_ok=True)
        
        # 初始化 CSV
        with open(INVALID_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['原始行', '无效片段', '原因'])
        
        # ===== STEP 1-2: SMILES 转 CIF 结构 =====
        logger.info("=== STEP 1-2: SMILES 转 CIF 结构 ===")
        
        # 按 '.' 分割 SMILES 片段
        lines_list = smiles_prefix.split('.')
        
        # 过滤含金属的片段
        filtered_lines = []
        metal_nodes = []
        for linker in lines_list:
            # 特殊情况：单独的 [O] 也视为金属节点
            if linker.strip() == "[O]":
                metal_nodes.append(linker)
                continue
            
            # 常规金属元素识别
            has_metal = any(metal in linker for metal in METAL_ELEMENTS)
            if not has_metal:
                filtered_lines.append(linker)
            else:
                metal_nodes.append(linker)
                logger.info(f" → 识别金属节点: {linker}")
        
        if not filtered_lines:
            logger.error("无有效有机片段")
            return 0
        
        # 验证所有片段是否有效
        all_valid = True
        invalid_linkers = []
        
        for linker in filtered_lines:
            atoms = get_smiles_to_cif(linker)
            if atoms is None:
                all_valid = False
                invalid_linkers.append(linker)
                with open(INVALID_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([smiles_input, linker, 'get_smiles_to_cif 返回 None'])
        
        if not all_valid:
            logger.error(f"包含无效片段: {invalid_linkers}")
            return 0
        
        # ===== 🔧 关键修改1: 创建 SMILES 映射 =====
        smiles_mapping = create_smiles_mapping(filtered_lines)

        logger.info("SMILES 到文件名映射:")
        for original, safe in smiles_mapping.items():
            logger.info(f"  {original[:50]}... → {safe}")

        # ===== 🔧 关键修改2: 生成安全的一级目录名 =====
        # 使用拓扑名称 + hash
        topology_name = topo_parts[1] if len(topo_parts) > 1 else "unknown"
        topology_name = sanitize_filename(topology_name, max_length=20)

        # 为整个 SMILES 前缀生成 hash
        smiles_hash = hashlib.md5(smiles_prefix.encode()).hexdigest()[:8]
        level1_name = f"{topology_name}_{smiles_hash}"

        level1_path = os.path.join(EDGES_C_ROOT, level1_name)
        os.makedirs(level1_path, exist_ok=True)

        logger.info(f"一级目录名: {level1_name}")
        
        # 保存 meta 信息（包含映射关系）
        meta_file = os.path.join(level1_path, "meta.txt")
        with open(meta_file, 'w', encoding='utf-8') as f:
            f.write(f"原始SMILES: {smiles_prefix}\n")
            f.write(f"拓扑: {topology_info}\n")
            f.write(f"金属节点: {', '.join(metal_nodes) if metal_nodes else '无'}\n")
            f.write(f"有机链接体: {', '.join(filtered_lines)}\n")
            f.write("\nSMILES映射:\n")
            for original, safe in smiles_mapping.items():
                f.write(f"  {safe} <- {original}\n")
                
        # ===== 🔧 关键修改3: 使用安全文件名保存 =====
        linker_counter = defaultdict(int)
        for linker in filtered_lines:
            atoms = get_smiles_to_cif(linker)
            safe_name = smiles_mapping[linker]
            
            linker_counter[safe_name] += 1
            if linker_counter[safe_name] == 1:
                cif_filename = f"{safe_name}.cif"
            else:
                cif_filename = f"{safe_name}_{linker_counter[safe_name]}.cif"
            
            cif_path = os.path.join(level1_path, cif_filename)
            write(cif_path, atoms)
            logger.info(f"  保存: {cif_filename}")
        
        logger.info(f"✅ STEP 1-2 完成: {len(filtered_lines)} 个片段")
        
        # ===== STEP 4: CIF 转 MOL =====
        logger.info("=== STEP 4: CIF 转 MOL ===")
        
        mol_level1_path = os.path.join(EDGES_MOL_ROOT, level1_name)
        os.makedirs(mol_level1_path, exist_ok=True)
        
        # 复制 meta.txt
        shutil.copy2(meta_file, os.path.join(mol_level1_path, "meta.txt"))
        
        # ===== 🔧 关键修改4: 反向查找原始 SMILES =====
        # 创建反向映射
        reverse_mapping = {v: k for k, v in smiles_mapping.items()}

        success_mol = 0
        for file in os.listdir(level1_path):
            if not file.lower().endswith('.cif'):
                continue
            
            file_stem = os.path.splitext(file)[0]
            
            # 移除可能的 _数字 后缀
            base_name = file_stem
            if '_' in file_stem and file_stem.split('_')[-1].isdigit():
                base_name = file_stem.rsplit('_', 1)[0]
            
            # 查找原始 SMILES
            original_smiles = reverse_mapping.get(base_name)
            if original_smiles is None:
                logger.warning(f"  无法找到 {base_name} 的原始 SMILES，跳过")
                continue
            
            out_filename = f"{file_stem}.mol"
            out_path = os.path.join(mol_level1_path, out_filename)
            
            try:
                smiles_to_mol(original_smiles, out_path)
                success_mol += 1
                logger.info(f"  转换: {file_stem}.cif → {out_filename}")
            except Exception as e:
                logger.error(f"❌ 处理失败 {file}: {e}")
        
        logger.info(f"✅ STEP 4 完成: 成功 {success_mol}")
        
        if success_mol == 0:
            logger.error("MOL 转换失败")
            return 0
        
        # ===== STEP 5: MOL 转 CIF =====
        logger.info("=== STEP 5: MOL 转 CIF ===")
        
        cif_level1_path = os.path.join(EDGES_CIF_ROOT, level1_name)
        os.makedirs(cif_level1_path, exist_ok=True)
        
        # 复制 meta.txt
        shutil.copy2(os.path.join(mol_level1_path, "meta.txt"), 
                     os.path.join(cif_level1_path, "meta.txt"))
        
        success_cif = 0
        for file in os.listdir(mol_level1_path):
            if not file.lower().endswith('.mol'):
                continue
            
            mol_path = os.path.join(mol_level1_path, file)
            file_stem = os.path.splitext(file)[0]
            
            if '_' in file_stem and file_stem.split('_')[-1].isdigit():
                out_filename = f"{file_stem}.cif"
            else:
                out_filename = f"{file_stem}.cif"
            
            out_path = os.path.join(cif_level1_path, out_filename)
            
            # 查找原始 CIF（在 edges_C）获取晶胞参数
            original_cif_name = file_stem.split('_')[0] if '_' in file_stem else file_stem
            
            try:
                atoms, bonds, charges = parse_mol_file(mol_path)
            except Exception as e:
                logger.error(f"解析失败 {file}: {e}")
                continue
            
            cell = None
            try:
                for orig_file in os.listdir(level1_path):
                    if orig_file.startswith(original_cif_name) and orig_file.endswith('.cif'):
                        cand_cif = os.path.join(level1_path, orig_file)
                        cell_candidate = read_cell_from_cif_path(cand_cif)
                        if cell_candidate:
                            cell = cell_candidate
                            break
            except Exception:
                cell = None
            
            if cell is None:
                cell = CELL
            
            try:
                write_cif(out_path, atoms, bonds, charges, cell=cell)
                success_cif += 1
            except Exception as e:
                logger.error(f"写入失败 {file}: {e}")
        
        logger.info(f"✅ STEP 5 完成: 成功 {success_cif}")
        
        if success_cif == 0:
            logger.error("CIF 生成失败")
            return 0
        
        # ===== STEP 6: 修改 CIF 文件 =====
        logger.info("=== STEP 6: 修改 CIF 文件 ===")
        
        mol_files = sorted(Path(mol_level1_path).glob("*.mol"))
        
        modified_count = 0
        for mol_path in mol_files:
            try:
                atoms_list, bonds_list, bond_sums, carbons_list, nitrogens_list = parse_mol_v2000(mol_path)
            except Exception as e:
                logger.error(f"解析 mol 失败 ({mol_path}): {e}")
                continue
            
            base = mol_path.stem
            cif_path = Path(cif_level1_path) / (base + ".cif")
            
            # 查找原始 CIF 获取晶胞参数
            original_cif_name = base.split('_')[0] if '_' in base else base
            cell = None
            try:
                for orig_file in os.listdir(level1_path):
                    if orig_file.startswith(original_cif_name) and orig_file.endswith('.cif'):
                        cand_cif = os.path.join(level1_path, orig_file)
                        cell_candidate = read_cell_from_cif_path(cand_cif)
                        if cell_candidate:
                            cell = cell_candidate
                            break
            except Exception:
                cell = None
            
            if cell is None:
                cell = CELL
            
            res = find_connection_atoms(carbons_list, nitrogens_list, atoms_list, bonds_list, mol_path)
            if res is None:
                logger.error(f"  未找到至少两个符合条件的连接原子，失败")
                return 0
            
            cn_old, cm_old, dist, element_types = res  # cn_old, cm_old 是原始atoms_list中的索引
            logger.info(f"  Selected atoms (原始索引): n={cn_old}, m={cm_old}, distance={dist:.4f}, types={element_types}")
            
            # 统计bonds_list修改前后的H原子
            h_atoms_before = set()
            for a, b, bo in bonds_list:
                if atoms_list[a]["element"].upper() == "H":
                    h_atoms_before.add(a)
                if atoms_list[b]["element"].upper() == "H":
                    h_atoms_before.add(b)
            logger.info(f"  bonds_list中的H原子索引: {sorted(h_atoms_before)}")
            
            # 找出所有应该保留的原子（bonds_list中的原子都应该保留）
            atoms_to_keep = set()
            for a, b, bo in bonds_list:
                atoms_to_keep.add(a)
                atoms_to_keep.add(b)
            
            # 统计被删除的H原子
            all_h_atoms = set()
            for idx, atom in enumerate(atoms_list):
                if atom["element"].upper() == "H":
                    all_h_atoms.add(idx)
            
            removed_h = all_h_atoms - atoms_to_keep
            logger.info(f"  总H原子: {len(all_h_atoms)}, 保留: {len(all_h_atoms & atoms_to_keep)}, 删除: {len(removed_h)}")
            if removed_h:
                logger.info(f"  删除的H原子索引: {sorted(removed_h)}")
            
            # 构建新的atoms列表（只保留在bonds中出现的原子）
            filtered_atoms = []
            old_to_new_index = {}
            for idx, atom in enumerate(atoms_list):
                # 只保留在bonds中的原子
                if idx in atoms_to_keep:
                    old_to_new_index[idx] = len(filtered_atoms)
                    filtered_atoms.append({
                        'index': len(filtered_atoms) + 1,
                        'x': atom["xyz"][0],
                        'y': atom["xyz"][1],
                        'z': atom["xyz"][2],
                        'element': atom["element"],
                        'raw': ''
                    })
            
            logger.info(f"  过滤前原子数: {len(atoms_list)}, 过滤后: {len(filtered_atoms)}")
            
            # 映射选中原子的索引到新索引
            if cn_old not in old_to_new_index or cm_old not in old_to_new_index:
                logger.error(f"  选中的原子不在过滤后的列表中！")
                return 0
            
            cn_new = old_to_new_index[cn_old] + 1  # CIF中索引从1开始
            cm_new = old_to_new_index[cm_old] + 1
            
            # ===== 🔧 修改2: 使用全局连续索引作为标签 =====
            # CIF标签格式：X1, C2, C3, O4, N5, C6... (全局连续编号)
            elem_cn = filtered_atoms[cn_new - 1]['element'].upper()
            elem_cm = filtered_atoms[cm_new - 1]['element'].upper()

            # 直接使用全局索引（不需要按元素类型计数）
            cn_label_num = cn_new  # 直接使用全局位置
            cm_label_num = cm_new  # 直接使用全局位置

            logger.info(f"  原始索引: cn_old={cn_old}, cm_old={cm_old}")
            logger.info(f"  映射后的新索引: cn_new={cn_new}, cm_new={cm_new}")
            logger.info(f"  CIF标签: {elem_cn}{cn_label_num}, {elem_cm}{cm_label_num}")
            logger.info(f"  (使用全局连续编号)")
            
            # 更新bonds中的索引
            filtered_bonds = []
            for a, b, bo in bonds_list:
                if a in old_to_new_index and b in old_to_new_index:
                    new_a = old_to_new_index[a] + 1  # CIF中索引从1开始
                    new_b = old_to_new_index[b] + 1
                    filtered_bonds.append({'a': new_a, 'b': new_b, 'order': bo})
                else:
                    # 记录被过滤的键（通常是涉及已删除H的键）
                    elem_a = atoms_list[a]["element"] if a < len(atoms_list) else "?"
                    elem_b = atoms_list[b]["element"] if b < len(atoms_list) else "?"
                    logger.info(f"  过滤掉键: {elem_a}({a}) - {elem_b}({b})")
            
            # 重新生成CIF文件（包含删除H后的结构）
            try:
                write_cif(str(cif_path), filtered_atoms, filtered_bonds, {}, cell=cell)
                logger.info(f"  重新生成 CIF: {cif_path.name}")
            except Exception as e:
                logger.error(f"  重新生成 CIF 失败: {e}")
                continue
            
            # 修改CIF中的元素标签（C/N → X）
            try:
                changed = modify_cif_labels(cif_path, cn_label_num, cm_label_num, element_types)
                if changed:
                    modified_count += 1
                    logger.info(f"  CIF {cif_path.name} 已修改标签")
            except Exception as e:
                logger.error(f"  修改 CIF 标签失败: {e}")
        
        logger.info(f"✅ STEP 6 完成: 修改 {modified_count} 个文件")
        
        # ===== 检查 edges_cif 是否有文件 =====
        cif_files = list(Path(EDGES_CIF_ROOT).rglob("*.cif"))
        if not cif_files:
            logger.error("edges_cif 文件夹为空")
            return 0
        
        logger.info(f"edges_cif 包含 {len(cif_files)} 个 CIF 文件")
        
        # ===== 复制到目标路径 =====
        repo_root = Path(__file__).resolve().parents[2]
        tobacco_workdir = Path(os.environ.get("MOF_TOBACCO_WORKDIR", repo_root / "tobacco_workdir"))
        target_path = os.environ.get("MOF_TOBACCO_EDGES_DIR", str(tobacco_workdir / "edges"))
        logger.info(f"复制文件到: {target_path}")
        
        os.makedirs(target_path, exist_ok=True)
        
        # ===== 🔧 关键修改5: level1_name 已经是安全的，直接使用 =====
        target_level1 = os.path.join(target_path, level1_name)
        if os.path.exists(target_level1):
            shutil.rmtree(target_level1)

        shutil.copytree(cif_level1_path, target_level1)

        copied_files = list(Path(target_level1).rglob("*.cif"))
        logger.info(f"✅ 成功复制 {len(copied_files)} 个文件到: {target_level1}")
        
        copied_files = list(Path(target_level1).rglob("*.cif"))
        logger.info(f"✅ 成功复制 {len(copied_files)} 个文件到目标路径")
        
        logger.info("=" * 80)
        logger.info("✅ 所有步骤完成")
        logger.info("=" * 80)
        
        return 1
        
    except Exception as e:
        logger.exception(f"处理过程中发生错误: {e}")
        return 0
    
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_base)
            logger.info(f"已清理临时目录: {temp_base}")
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")

# ===== 测试代码 =====
if __name__ == "__main__":
    # 测试示例
    test_smiles = "CN1C(=N)N=CN=C1C.O=C1COC=NCCO1.Cc1cc(C)c(C(=O)N[C@H]2CCC[NH2+][C@H]2C)c(C)c1.[Cu][Cu] MOFid-v1.pcu.cat0"
    
    result = process_mof_smiles(test_smiles)
    
    if result == 1:
        print("\n✅ 处理成功!")
    else:
        print("\n❌ 处理失败!")
    
    sys.exit(0 if result == 1 else 1)
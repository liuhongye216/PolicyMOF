#!/usr/bin/env python3
"""
静默版 MOF CIF 评估器 (run_evaluate 返回 0-1 分值)
- 默认输入路径由 MOF_TOBACCO_WORKDIR 或仓库内 tobacco_workdir 决定
- 单文件模式: 取该目录下第一个 *.cif
- 不产生任何打印或外部文件写入
"""

import os
from pathlib import Path
import shutil
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
import numpy as np
from collections import Counter, defaultdict
import json
import time
from multiprocessing import Pool, cpu_count
import argparse

# ----------------------------
# 这里保留你的 EnhancedMOFScreener 类（评分逻辑未改动）
# 为简洁起见，把类定义直接粘贴进来（与原脚本等价）
# ----------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
TOBACCO_WORKDIR = Path(os.environ.get("MOF_TOBACCO_WORKDIR", REPO_ROOT / "tobacco_workdir"))
edges_dir = Path(os.environ.get("MOF_TOBACCO_EDGES_DIR", TOBACCO_WORKDIR / "edges"))
output_cifs_dir = Path(os.environ.get("MOF_TOBACCO_OUTPUT_CIFS_DIR", TOBACCO_WORKDIR / "output_cifs"))


class EnhancedMOFScreener:
    def __init__(self, strict_mode=False):
        self.strict_mode = strict_mode
        if strict_mode:
            self.thresholds = {
                'max_atoms': 3000, 'max_volume': 100000, 'min_volume': 500,
                'max_density': 0.12, 'min_density': 0.005,
                'severe_overlap': 0.5, 'moderate_overlap': 0.7, 'min_safe_distance': 0.8,
                'angle_tolerance': 0.01, 'pseudo_ortho_threshold': 0.5, 'max_overlap_pairs': 0,
            }
        else:
            self.thresholds = {
                'max_atoms': 3000, 'max_volume': 100000, 'min_volume': 100,
                'max_density': 0.20, 'min_density': 0.001,
                'severe_overlap': 0.5, 'moderate_overlap': 0.7, 'min_safe_distance': 0.8,
                'angle_tolerance': 0.1, 'pseudo_ortho_threshold': 0.2, 'max_overlap_pairs': 2,
            }
        self.metals = ['Zn', 'Cu', 'Zr', 'Fe', 'Al', 'Cr', 'Co', 'Ni', 'Mn', 'Mg', 'Ca',
                      'Ti', 'V', 'Sc', 'Y', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
                      'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Cd', 'Hg', 'In', 'Sn', 'Pb']
        self.expected_cn = {
            'Zn': 4, 'Cu': 4, 'Zr': 8, 'Fe': 6, 'Al': 6, 'Cr': 6,
            'Co': 6, 'Ni': 6, 'Mn': 6, 'Mg': 6, 'Ca': 8, 'Ti': 6,
        }

    def screen_file(self, cif_path):
        result = {
            'filename': Path(cif_path).name, 'filepath': cif_path,
            'status': 'UNKNOWN', 'score': 0, 'issues': [], 'warnings': [],
        }
        try:
            parser = CifParser(cif_path)
            structure = parser.get_structures(primitive=False)[0]

            result['num_atoms'] = len(structure)
            result['volume'] = structure.volume
            result['density'] = len(structure) / structure.volume

            lattice = structure.lattice
            result['a'], result['b'], result['c'] = lattice.a, lattice.b, lattice.c
            result['alpha'], result['beta'], result['gamma'] = lattice.alpha, lattice.beta, lattice.gamma

            composition = structure.composition
            result['elements'] = ','.join([str(e) for e in composition.elements])

            # 基础 REJECT 检查
            if result['num_atoms'] > self.thresholds['max_atoms']:
                result['issues'].append(f"原子数过多({result['num_atoms']})")
                result['status'] = 'REJECT'
                return result

            if result['volume'] > self.thresholds['max_volume'] or result['volume'] < self.thresholds['min_volume']:
                result['issues'].append(f"体积异常({result['volume']:.0f}Å³)")
                result['status'] = 'REJECT'
                return result

            if result['density'] > self.thresholds['max_density'] or result['density'] < self.thresholds['min_density']:
                result['issues'].append(f"密度异常({result['density']:.4f})")
                result['status'] = 'REJECT'
                return result

            elements = [str(e) for e in composition.elements]
            if not any(m in elements for m in self.metals):
                result['issues'].append("缺少金属元素")
                result['status'] = 'REJECT'
                return result
            if 'C' not in elements:
                result['issues'].append("缺少碳元素")
                result['status'] = 'REJECT'
                return result
            
            # 检查晶格参数合理性
            if result['a'] < 2.0 or result['b'] < 2.0 or result['c'] < 2.0:
                result['issues'].append(f"晶格参数异常小(a={result['a']:.3f}, b={result['b']:.3f}, c={result['c']:.3f}Å)")
                result['status'] = 'REJECT'
                return result

            if result['a'] > 300 or result['b'] > 300 or result['c'] > 300:
                result['issues'].append(f"晶格参数异常大(a={result['a']:.1f}, b={result['b']:.1f}, c={result['c']:.1f}Å)")
                result['status'] = 'REJECT'
                return result
            
            '''# 检查平均每原子体积是否合理（MOF通常 500-5000 Å³/atom）
            volume_per_atom = result['volume'] / result['num_atoms']
            if volume_per_atom < 50:
                result['issues'].append(f"每原子体积过小({volume_per_atom:.1f}Å³/atom)")
                result['status'] = 'REJECT'
                return result
            if volume_per_atom > 10000:
                result['issues'].append(f"每原子体积过大({volume_per_atom:.1f}Å³/atom)")
                result['status'] = 'REJECT'
                return result'''
            
            # 详细分析
            overlap = self._check_overlaps(structure)
            result.update(overlap)
            if result['severe_overlaps'] > 0:
                result['issues'].append(f"严重原子重叠({result['severe_overlaps']}对)")
                result['status'] = 'REJECT'
                return result
            if result['moderate_overlaps'] > self.thresholds['max_overlap_pairs']:
                result['issues'].append(f"中度重叠过多({result['moderate_overlaps']}对)")
                result['status'] = 'REJECT'
                return result
            if result['moderate_overlaps'] > 0:
                result['warnings'].append(f"中度重叠({result['moderate_overlaps']}对)")

            angle_check = self._check_pseudo_orthogonal(result)
            result.update(angle_check)
            if angle_check['vesta_risk']:
                result['warnings'].append(f"VESTA风险: {angle_check['vesta_risk_reason']}")

            try:
                sga = SpacegroupAnalyzer(structure, symprec=0.1)
                detected_sg = sga.get_space_group_number()
                result['spacegroup'] = detected_sg
                if detected_sg > 1:
                    result['warnings'].append(f"检测到对称性(No.{detected_sg})")
            except:
                result['spacegroup'] = 1

            coord_result = self._analyze_coordination(structure)
            result.update(coord_result)
            connectivity = self._check_connectivity(structure)
            result.update(connectivity)
            bond_analysis = self._analyze_bond_lengths(structure)
            result.update(bond_analysis)

            score = self._calculate_enhanced_score(result, structure)
            result['score'] = score

            if result['issues']:
                result['status'] = 'REJECT'
            elif score >= 85:
                result['status'] = 'EXCELLENT'
            elif score >= 70:
                result['status'] = 'GOOD'
            elif score >= 50:
                result['status'] = 'FAIR'
            else:
                result['status'] = 'POOR'

        except Exception as e:
            result['status'] = 'ERROR'
            result['issues'].append(f"错误: {str(e)[:100]}")
        return result

    def _calculate_enhanced_score(self, result, structure):
        geometry_score = 100
        geometry_score -= result['severe_overlaps'] * 50
        geometry_score -= result['moderate_overlaps'] * 5
        geometry_score -= result.get('mild_overlaps', 0) * 1
        if result.get('vesta_risk', False):
            geometry_score -= 5
        cell_distortion = self._calculate_cell_distortion(result)
        geometry_score -= cell_distortion * 10
        geometry_score = max(0, geometry_score)
        chemistry_score = 100
        if 'avg_coord_deviation' in result:
            dev = result['avg_coord_deviation']
            chemistry_score -= min(dev * 30, 40)
        if result.get('abnormal_coordinations', 0) > 0:
            chemistry_score -= result['abnormal_coordinations'] * 10
        density = result['density']
        if density < 0.01 or density > 0.15:
            chemistry_score -= 15
        elif density < 0.005 or density > 0.18:
            chemistry_score -= 10
        chemistry_score = max(0, chemistry_score)
        topology_score = 100
        if not result.get('is_connected', True):
            topology_score -= 50
        if result.get('isolated_atoms', 0) > 0:
            topology_score -= result['isolated_atoms'] * 10
        sg = result.get('spacegroup', 1)
        if sg > 15:
            topology_score += 5
        elif sg > 2:
            topology_score += 2
        topology_score = max(0, min(105, topology_score))
        complexity_score = 100
        atoms_per_volume = result['num_atoms'] / result['volume']
        if self.thresholds['min_density'] <= atoms_per_volume <= self.thresholds['max_density']:
            pass
        else:
            complexity_score -= 10
        complexity_score = max(0, complexity_score)
        weights = {
            'geometry': 0.40,
            'chemistry': 0.30,
            'topology': 0.20,
            'complexity': 0.10,
        }
        final_score = (
            geometry_score * weights['geometry'] +
            chemistry_score * weights['chemistry'] +
            topology_score * weights['topology'] +
            complexity_score * weights['complexity']
        )
        result['subscores'] = {
            'geometry': round(geometry_score, 2),
            'chemistry': round(chemistry_score, 2),
            'topology': round(topology_score, 2),
            'complexity': round(complexity_score, 2),
        }
        return max(0, min(100, final_score))

    def _calculate_cell_distortion(self, result):
        a, b, c = result['a'], result['b'], result['c']
        alpha, beta, gamma = result['alpha'], result['beta'], result['gamma']
        lengths = [a, b, c]
        max_len, min_len = max(lengths), min(lengths)
        length_ratio = max_len / min_len if min_len > 0 else 10
        angles = [alpha, beta, gamma]
        angle_deviation = sum(abs(ang - 90) for ang in angles) / 3
        distortion = 0
        if length_ratio > 3:
            distortion += 0.3
        if angle_deviation > 10:
            distortion += 0.3
        return distortion

    def _analyze_coordination(self, structure):
        result = {
            'metal_coordinations': [],
            'avg_coord_deviation': 0.0,
            'abnormal_coordinations': 0,
        }
        try:
            nn = MinimumDistanceNN()
            deviations = []
            for i, site in enumerate(structure):
                elem = site.species_string
                if elem in self.metals:
                    try:
                        cn = nn.get_cn(structure, i, use_weights=False)
                        expected = self.expected_cn.get(elem, 6)
                        deviation = abs(cn - expected)
                        result['metal_coordinations'].append({
                            'index': i,
                            'element': elem,
                            'actual_cn': cn,
                            'expected_cn': expected,
                            'deviation': deviation
                        })
                        deviations.append(deviation)
                        if deviation > 2:
                            result['abnormal_coordinations'] += 1
                    except:
                        pass
            if deviations:
                result['avg_coord_deviation'] = sum(deviations) / len(deviations)
        except:
            pass
        return result

    def _check_connectivity(self, structure):
        result = {
            'is_connected': True,
            'num_components': 1,
            'isolated_atoms': 0,
        }
        try:
            nn = MinimumDistanceNN()
            sg = StructureGraph.with_local_env_strategy(structure, nn)
            components = sg.graph.connected_components()
            component_sizes = [len(c) for c in components]
            result['num_components'] = len(component_sizes)
            result['is_connected'] = (len(component_sizes) == 1)
            result['isolated_atoms'] = component_sizes.count(1)
        except:
            pass
        return result

    def _analyze_bond_lengths(self, structure):
        result = {
            'bond_length_stats': {},
            'unusual_bonds': 0,
        }
        try:
            nn = MinimumDistanceNN()
            all_bonds = []
            for i in range(len(structure)):
                try:
                    neighbors = nn.get_nn_info(structure, i)
                    for nb in neighbors:
                        dist = structure.get_distance(i, nb['site_index'])
                        all_bonds.append(dist)
                except:
                    pass
            if all_bonds:
                result['bond_length_stats'] = {
                    'mean': float(np.mean(all_bonds)),
                    'std': float(np.std(all_bonds)),
                    'min': float(np.min(all_bonds)),
                    'max': float(np.max(all_bonds)),
                }
                mean, std = result['bond_length_stats']['mean'], result['bond_length_stats']['std']
                result['unusual_bonds'] = sum(1 for b in all_bonds if abs(b - mean) > 3 * std)
        except:
            pass
        return result

    def _check_overlaps(self, structure, sample_size=300):
        result = {
            'min_distance': float('inf'),
            'severe_overlaps': 0,
            'moderate_overlaps': 0,
            'mild_overlaps': 0,
        }
        n = len(structure)
        metal_indices = [i for i, site in enumerate(structure) if site.species_string in self.metals]
        other_indices = [i for i in range(n) if i not in metal_indices]
        check_indices = set(metal_indices)
        if len(other_indices) > sample_size:
            sampled = np.random.choice(other_indices, sample_size, replace=False)
            check_indices.update(sampled)
        else:
            check_indices.update(other_indices)
        check_indices = sorted(list(check_indices))
        for i in check_indices:
            for j in range(i+1, min(i+50, n)):
                dist = structure.get_distance(i, j)
                result['min_distance'] = min(result['min_distance'], dist)
                if dist < self.thresholds['severe_overlap']:
                    result['severe_overlaps'] += 1
                elif dist < self.thresholds['moderate_overlap']:
                    result['moderate_overlaps'] += 1
                elif dist < self.thresholds['min_safe_distance']:
                    result['mild_overlaps'] += 1
        print(result)
        return result

    def _check_pseudo_orthogonal(self, result):
        alpha, beta, gamma = result['alpha'], result['beta'], result['gamma']
        check = {
            'vesta_risk': False,
            'vesta_risk_reason': '',
            'is_pseudo_orthogonal': False,
        }
        angles = [alpha, beta, gamma]
        close_to_90 = []
        for i, angle in enumerate(angles):
            diff = abs(angle - 90.0)
            if diff < self.thresholds['pseudo_ortho_threshold'] and diff > self.thresholds['angle_tolerance']:
                close_to_90.append((['α', 'β', 'γ'][i], angle, diff))
        if close_to_90:
            check['is_pseudo_orthogonal'] = True
            check['vesta_risk'] = True
            angle_info = ', '.join([f"{name}={angle:.3f}°" for name, angle, diff in close_to_90])
            check['vesta_risk_reason'] = f"伪正交({angle_info})"
        return check

# ----------------------------
# 新增：静默的 run_evaluate() 接口
# ----------------------------

DEFAULT_INPUT_DIR = output_cifs_dir

def run_evaluate():
    """
    在固定目录下以单文件模式评估（取第一个 .cif 文件，按字典序）。
    返回:
        float: 归一化评分 0.0 - 1.0（若出错或未找到文件则返回 0.0）
    注意: 此函数不进行任何打印或文件写入。
    """
    try:
        if not DEFAULT_INPUT_DIR.exists() or not DEFAULT_INPUT_DIR.is_dir():
            return 0.0

        cif_list = sorted([p for p in DEFAULT_INPUT_DIR.glob('*.cif')])
        if not cif_list:
            return 0.0

        cif_path = str(cif_list[0])

        screener = EnhancedMOFScreener(strict_mode=False)
        result = screener.screen_file(cif_path)
        # 建议去掉这行，以保持真正静默
        print(result['issues'])

        score = result.get('score', 0.0)
        try:
            score = float(score)
        except:
            score = 0.0
        score = max(0.0, min(100.0, score))
        normalized = score / 100.0

        # ===== 在这里清理 edges 和 output_cifs 下的所有内容 =====
        
        for root_dir in [edges_dir, output_cifs_dir]:
            if root_dir.exists() and root_dir.is_dir():
                for p in root_dir.iterdir():
                    try:
                        if p.is_file() or p.is_symlink():
                            p.unlink()
                        elif p.is_dir():
                            shutil.rmtree(p)
                    except Exception:
                        # 静默失败，避免影响得分返回
                        pass
        # ======================================================

        return normalized
    except Exception:
        for root_dir in [edges_dir, output_cifs_dir]:
            if root_dir.exists() and root_dir.is_dir():
                for p in root_dir.iterdir():
                    try:
                        if p.is_file() or p.is_symlink():
                            p.unlink()
                        elif p.is_dir():
                            shutil.rmtree(p)
                    except Exception:
                        # 静默失败，避免影响得分返回
                        pass
        return 0.0

# 不要在脚本运行时打印或写入任何东西
if __name__ == '__main__':
    # 保持静默：不做任何默认行为
    print(run_evaluate())
    pass

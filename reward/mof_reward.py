import traceback
import os, sys
import re
import logging
from typing import List, Optional
from collections import deque
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import AllChem
from swift.plugin import ORM, orms
import shutil
from pathlib import Path
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

REPO_ROOT = Path(__file__).resolve().parents[1]
GRPO_ROOT = REPO_ROOT / "GRPO"
if str(GRPO_ROOT) not in sys.path:
    sys.path.insert(0, str(GRPO_ROOT))

TOBACCO_WORKDIR = Path(os.environ.get("MOF_TOBACCO_WORKDIR", REPO_ROOT / "tobacco_workdir"))
edges_dir = Path(os.environ.get("MOF_TOBACCO_EDGES_DIR", TOBACCO_WORKDIR / "edges"))
output_cifs_dir = Path(os.environ.get("MOF_TOBACCO_OUTPUT_CIFS_DIR", TOBACCO_WORKDIR / "output_cifs"))

N2_MODEL_PATH = os.environ.get("MOF_N2_MODEL", "")
CO2_MODEL_PATH = os.environ.get("MOF_CO2_MODEL", "")
DEFAULT_ADSORPTION_MODEL = os.environ.get("MOF_ADSORPTION_MODEL", CO2_MODEL_PATH)

# Import project utilities after adding GRPO_ROOT to sys.path.
from utils.enhanced_cif_screener import run_evaluate
from utils.tobacco import run_tobacco_with_edge_folders
from utils.mof_processing import process_mof_smiles

logger = logging.getLogger(__name__)

# -------------------------
# 推断相关 imports 延迟在函数内部导入（避免模块导入时要求 heavy deps）
# -------------------------

class MOFRewardORM(ORM):

    def __init__(self,
                weights: Optional[dict] = None,
                adsorption_engine: Optional[object] = None,
                adsorption_engine2: Optional[object] = None,
                recent_maxlen: int = 2000):
        """
        初始化 MOFRewardORM 的状态与预加载资源。
        参数:
        - weights: 可选的子 reward 权重字典（若为 None 使用默认）
        - adsorption_engine: 可选的已创建推理引擎实例1（推荐在外部创建并传入以复用）
        - adsorption_mu/sigma: adsorption 映射用的 mu, sigma（单位需与预测模型一致）
        - adsorption_engine2: 可选的已创建推理引擎实例2（用于第二个指标预测）
        - adsorption_mu2/sigma2: 第二个指标的映射参数
        - adsorption_weight1/weight2: 两个指标的权重（默认各0.5）
        - recent_maxlen: 最近 prefix 缓存的最大长度
        """
        TOKENS_PATH = os.environ.get("MOF_TOKENS_PATH", str(REPO_ROOT / "reward" / "mof_id_tokens.csv"))
        MOFID_PATH = os.environ.get("MOF_MOFID_PATH", str(REPO_ROOT / "reward" / "mof_id.csv"))
        NODES_CSV = os.environ.get("MOF_NODES_CSV", str(REPO_ROOT / "reward" / "nodes_linkers_fromfolder.csv"))

        # 常见金属原子符号集合
        self.METAL_SYMBOLS = {
            "Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
            "Ga","Ge","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Hf","Ta","W","Re",
            "Os","Ir","Pt","Au","Hg","Pb"
        }

        # 默认权重（总和为 1.0），可在外部覆盖
        self.DEFAULT_WEIGHTS = {
            "chemical_validity": 0.07,
            "metal_presence": 0.09,
            "functional_group": 0.03,
            "has_ring": 0.03,
            "similarity": 0.16,
            "porosity_proxy": 0.02,
            "novelty": 0.11,
            "adsorption": 0.11,
            "structure_3d": 0.12,
            "cif_generation": 0.26
        }
        self.weights = weights or self.DEFAULT_WEIGHTS

        # 注入或延后创建的引擎
        self._adsorption_engine = adsorption_engine
        self._adsorption_engine2 = adsorption_engine2

        # 缓存：吸附预测与近期 prefix
        self._adsorption_cache = {}
        try:
            from collections import deque
        except Exception:
            # Python 标准库应当有 deque；这里以防万一
            deque = list

        self._recent_prefixes = deque(maxlen=recent_maxlen) if hasattr(deque, "__call__") else []
        self._recent_prefix_set = set()
        
        # 最近 per-part 缓存（用于短期去重）
        self._recent_parts = deque(maxlen=1000)
        self._recent_parts_set = set()

        # 金属重复追踪（用于连续相同金属组合时的衰减）
        self._last_metal_key = None
        self._metal_repeat_count = 0


        # -------------------------
        # 预加载与构建数据：nodes, tokens, mofid parts 数据库, candidate_substrings
        # -------------------------
        # 1) NODES_CSV -> VALID_NODES
        try:
            _nodes_df = pd.read_csv(NODES_CSV)
            if "nodes" in _nodes_df.columns:
                self.VALID_NODES = set(str(x).strip() for x in _nodes_df["nodes"].dropna().tolist())
            else:
                # fallback: 第一列作为 nodes
                self.VALID_NODES = set(str(x).strip() for x in _nodes_df.iloc[:, 0].dropna().tolist())
        except Exception as e:
            logger.warning(f"MOFRewardORM.__init__: Failed reading NODES_CSV {NODES_CSV}: {e}")
            self.VALID_NODES = set()

        # 2) TOKENS_PATH -> valid_topologies
        try:
            tok_df = pd.read_csv(TOKENS_PATH, header=None, dtype=str)
            # 取第9-36行作为有效拓扑（对应 iloc[8:36]）
            self.valid_topologies = tok_df.iloc[8:36, 0].astype(str).str.strip().tolist()
        except Exception as e:
            logger.warning(f"MOFRewardORM.__init__: Failed reading TOKENS_PATH {TOKENS_PATH}: {e}")
            self.valid_topologies = []

        # 3) MOFID_PATH -> 构建 mofid_full_set, parts 数据库 (新相似度用)
        self.mofid_full_set = set()          # 存储规范化后的前缀（用于新颖性检测）
        self.mofid_original_set = set()      # 存储原始前缀（如有其他用途）
        parts_list = []
        try:
            mofid_df = pd.read_csv(MOFID_PATH, header=None, dtype=str)
            mofid_col = mofid_df[0].fillna("").astype(str).tolist()
            
            # 原始集合（保留以备其他用途）
            self.mofid_original_set = set([x.strip() for x in mofid_col if x.strip()])
            
            # 规范化集合：按 '.' 分割后排序再拼接，用于顺序无关的重复检测
            for x in mofid_col:
                x = x.strip()
                if x:
                    # 取第一个 token 作为 prefix（与 _novelty_score 逻辑一致）
                    first_token = re.split(r'\s+', x)[0]
                    parts = first_token.split('.')
                    normalized_key = '.'.join(sorted(parts))
                    self.mofid_full_set.add(normalized_key)

            # parts_list: 从每行取第一个 token (prefix)，按 '.' 切分得到 fragment parts
            parts_acc = []
            for line in mofid_col:
                line = line.strip()
                if not line:
                    continue
                first_token = re.split(r'\s+', line)[0]
                subparts = [p.strip() for p in first_token.split('.') if p.strip()]
                for sp in subparts:
                    parts_acc.append(sp)
            # 去重并保持顺序
            seen = set()
            for p in parts_acc:
                if p not in seen:
                    seen.add(p)
                    parts_list.append(p)

        except Exception as e:
            logger.warning(f"MOFRewardORM.__init__: Failed reading MOFID_PATH {MOFID_PATH}: {e}")
            self.mofid_full_set = set()
            self.mofid_original_set = set()
            parts_list = []

        # 5) 为 parts_list 构建专用向量器（用于新的分段相似度判定）
        try:
            if parts_list:
                self._parts_list = parts_list
                self._parts_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6))
                self._parts_vectors = self._parts_vectorizer.fit_transform(self._parts_list)
            else:
                self._parts_list = []
                self._parts_vectorizer = None
                self._parts_vectors = None
        except Exception as e:
            logger.warning(f"MOFRewardORM.__init__: parts vectorizer init failed: {e}")
            self._parts_list = []
            self._parts_vectorizer = None
            self._parts_vectors = None

    # -------------------------
    # Helper functions
    # -------------------------
    def _strict_format_check(self, pred: str):
        if pred is None:
            return False, "", ""
        s = pred.strip()
        if s.count(' ') != 1:
            return False, "", ""
        prefix, suffix = s.split(' ', 1)
        m = re.match(r"^MOFid-v1\.([^.]+)\.cat([0-3])$", suffix)
        if not m:
            return False, prefix, suffix
        topo = m.group(1)
        if topo not in self.valid_topologies:
            return False, prefix, suffix
        return True, prefix, suffix
    
    
    def _similarity_score(self, prefix: str, use_fingerprint: bool = True) -> float:
            """
            改进的相似度评分，结合多种策略：
            1. 化学指纹相似度（推荐用于SMILES）
            2. 放宽的文本相似度
            3. 渐进式惩罚而非二元判断
            """
            try:
                if not prefix:
                    return 0.0
                    
                parts = [p.strip() for p in prefix.split('.') if p.strip()]
                if not parts:
                    return 0.0

                # 过滤掉含有金属元素的片段
                filtered_parts = []
                for part in parts:
                    try:
                        mol = Chem.MolFromSmiles(part)
                        if mol is None:
                            continue
                        
                        # 检查是否含有金属元素
                        has_metal = False
                        for atom in mol.GetAtoms():
                            if atom.GetSymbol() in self.METAL_SYMBOLS:
                                has_metal = True
                                break
                        
                        if not has_metal:
                            filtered_parts.append(part)
                        else:
                            logger.debug(f"_similarity_score: skipping part {part} containing metal")
                    except Exception as e:
                        logger.debug(f"_similarity_score: error checking metal in {part}: {e}")
                        continue
                
                if not filtered_parts:
                    return 0.0

                scores = []
                
                for part in filtered_parts:
                    part_score = self._compute_part_similarity(part, use_fingerprint)
                    scores.append(part_score)
                    print(f"Part: {part}, Score: {part_score}")
                
                # 批量更新近期队列
                self._update_recent_parts(filtered_parts)
                
                if not scores:
                    return 0.0
                    
                # 使用调和平均数聚合分数
                final_score = self._aggregate_part_scores(scores)
                return max(0.0, min(1.0, final_score))
                
            except Exception as e:
                logger.warning(f"_similarity_score error: {e}")
                return 0.0
    
    def _compute_part_similarity(self, part: str, use_fingerprint: bool = True) -> float:
        """
        计算单个part的相似度分数
        """
        # 策略1：检查精确重复（短期记忆）
        if part in self._recent_parts_set:
            # 渐进式惩罚：根据出现频率调整
            recent_count = sum(1 for p in self._recent_parts if p == part)
            if recent_count >= 3:  # 连续出现3次以上才给0分
                return 0.0
            else:
                return 0.3 * (1 - recent_count / 3)  # 渐进惩罚
        
        # 策略2：化学指纹相似度（推荐用于SMILES）
        if use_fingerprint:
            fp_score = self._fingerprint_similarity(part)
            if fp_score is not None:
                return fp_score
        
        # 策略3：文本相似度（回退方案）
        text_score = self._text_similarity(part)
        return text_score
    
    def _fingerprint_similarity(self, smiles: str) -> Optional[float]:
        """
        使用RDKit分子指纹计算相似度（更适合SMILES）
        """
        try:
            from rdkit import DataStructs
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            # 生成Morgan指纹
            fp = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
            
            # 在初始化部分（构建数据库时）
            if not hasattr(self, '_fingerprint_db'):
                self._fingerprint_db = []
                self._fingerprint_smiles = []  # 新增：存储成功生成指纹的 SMILES
                
                # 临时禁用 RDKit 警告
                from rdkit import RDLogger
                rd_logger = RDLogger.logger()
                rd_logger.setLevel(RDLogger.CRITICAL)
                
                try:
                    # 从parts_list构建指纹数据库
                    for part in getattr(self, '_parts_list', []):
                        try:
                            mol_db = Chem.MolFromSmiles(part)
                            if mol_db is None:
                                continue
                            
                            # 尝试添加氢原子，如果失败则使用原始分子
                            try:
                                mol_db_h = Chem.AddHs(mol_db)
                            except:
                                mol_db_h = mol_db
                            
                            fp_db = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol_db_h)
                            self._fingerprint_db.append(fp_db)
                            self._fingerprint_smiles.append(part)  # 同步存储对应的 SMILES
                        except:
                            continue
                finally:
                    # 恢复 RDKit 警告级别
                    rd_logger.setLevel(RDLogger.WARNING)
            
            if not self._fingerprint_db:
                return None
            
            # 计算Tanimoto相似度
            max_similarity = 0.0
            most_similar_idx = -1
            for idx, fp_db in enumerate(self._fingerprint_db):
                similarity = DataStructs.TanimotoSimilarity(fp, fp_db)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_idx = idx

            # 输出最相似的数据库条目
            if most_similar_idx >= 0 and hasattr(self, '_fingerprint_smiles'):
                most_similar_smiles = self._fingerprint_smiles[most_similar_idx]
                print(f"Max fingerprint similarity for {smiles}: {max_similarity:.4f}")
                print(f"  Most similar database entry [{most_similar_idx}]: {most_similar_smiles}")
                
                # 额外验证：重新计算这两个分子的相似度
                try:
                    mol_query = Chem.MolFromSmiles(smiles)
                    mol_match = Chem.MolFromSmiles(most_similar_smiles)
                    if mol_query and mol_match:
                        fp_query_verify = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol_query)
                        fp_match_verify = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol_match)
                        verify_sim = DataStructs.TanimotoSimilarity(fp_query_verify, fp_match_verify)
                except:
                    pass
            else:
                print(f"Max fingerprint similarity for {smiles}: {max_similarity:.4f}")
                
                
                
            # 建议（更宽松）
            if max_similarity > 0.98: return 0.0      # 只有几乎完全相同才是0
            elif max_similarity > 0.85: return 0.3 * (0.98 - max_similarity) / 0.13
            else: return 0.3 + 0.7 * (1 - max_similarity / 0.85)
                
        except Exception as e:
            logger.debug(f"_fingerprint_similarity failed: {e}")
            return None

    def _text_similarity(self, part: str) -> float:
        """
        文本相似度计算（回退方案）
        """
        try:
            if not hasattr(self, '_parts_vectorizer') or self._parts_vectorizer is None:
                return 0.5  # 无法判断时给中等分数
            
            vec = self._parts_vectorizer.transform([part])
            sims = cosine_similarity(vec, self._parts_vectors).flatten()
            
            if sims.size == 0:
                return 0.5
            
            s_max = float(np.max(sims))
            
            # 使用更平滑的映射函数
            if s_max > 0.95:
                return 0.0
            elif s_max > 0.85:
                return 0.3 * (0.95 - s_max) / 0.10
            else:
                return 0.3 + 0.7 * (1 - s_max / 0.85)
                
        except Exception as e:
            logger.debug(f"_text_similarity failed: {e}")
            return 0.5


    def _update_recent_parts(self, parts: List[str]):
        """
        批量更新近期parts队列
        """
        try:
            for part in parts:
                if isinstance(self._recent_parts, deque):
                    if len(self._recent_parts) == self._recent_parts.maxlen:
                        oldest = self._recent_parts.popleft()
                        self._recent_parts_set.discard(oldest)
                    self._recent_parts.append(part)
                    self._recent_parts_set.add(part)
                else:
                    # fallback to list
                    if part not in self._recent_parts_set:
                        self._recent_parts.append(part)
                        self._recent_parts_set.add(part)
                        if len(self._recent_parts) > 1000:
                            old = self._recent_parts.pop(0)
                            self._recent_parts_set.discard(old)
        except Exception as e:
            logger.debug(f"_update_recent_parts failed: {e}")


    def _aggregate_part_scores(self, scores: List[float]) -> float:
        """
        聚合多个part的分数，使用调和平均数（对低分更敏感）
        """
        if not scores:
            return 0.0
        
        # 使用调和平均数，对低分更敏感
        try:
            harmonic_mean = len(scores) / sum(1/(s + 1e-6) for s in scores)
            return harmonic_mean
        except:
            # 如果出现异常，回退到算术平均
            return sum(scores) / len(scores)


    def _chemical_validity_score(self, prefix: str) -> float:
        frags = [f for f in prefix.split('.') if f.strip()]
        if not frags:
            return 0.0
        valid = 0
        for f in frags:
            if len(f) >= 2 and re.search(r"[A-Za-z\[\]#%=()@+\-\d]", f):
                valid += 1
        return float(valid) / len(frags)

    def _metal_presence_score(self, prefix: str) -> float:
        """
        检查 prefix 中的金属片段：
        - 找出包含金属元素的 fragment 列表 metal_frags（顺序按出现顺序）
        - 若没有金属片段 -> 0.0
        - 若存在且所有 metal_frags 都在 VALID_NODES 中 -> base 1.0，否则 base 0.0
        - 若 current_key == self._last_metal_key: self._metal_repeat_count += 1 并做衰减乘以 (0.9 ** repeat_count)
          否则重置 repeat_count = 0 并更新 last_metal_key
        返回最终得分（float）。
        """
        frags = [f.strip() for f in prefix.split('.') if f.strip()]
        if not frags:
            return 0.0

        metal_frags = []
        for f in frags:
            br = re.findall(r'\[([A-Za-z]{1,3})\]', f)
            if any(el in self.METAL_SYMBOLS for el in br):
                metal_frags.append(f)

        if not metal_frags:
            # no metal fragments
            # update last metal tracking: treat as different from last to reset repeat_count
            self._last_metal_key = None
            self._metal_repeat_count = 0
            return 0.0

        # base score: 1 only if all metal_frags in VALID_NODES, else 0
        all_valid = all((frag in getattr(self, "VALID_NODES", set())) for frag in metal_frags)
        base = 1.0 if all_valid else 0.0

        # compute current key (order-insensitive)
        current_key = tuple(sorted(metal_frags))

        if current_key == getattr(self, "_last_metal_key", None):
            # same as last -> increment repeat count and apply decay
            self._metal_repeat_count = getattr(self, "_metal_repeat_count", 0) + 1
        else:
            # different -> reset counter
            self._metal_repeat_count = 0
            self._last_metal_key = current_key

        if len(metal_frags) > 1:
            return 0.0
        
        # apply multiplicative decay only when base > 0
        if base <= 0.0:
            return 0.0
        # multiplier: 0.9^repeat_count; first occurrence repeat_count==0 -> multiplier 1.0
        multiplier = (0.9 ** self._metal_repeat_count)
        final = float(base * multiplier)
        return final


    def _has_ring(self, prefix: str) -> bool:
        """
        使用 RDKit 精确判断 SMILES 是否包含环（任意环）。
        返回 True/False。若 SMILES 无法解析，视为 False。
        """
        if not prefix:
            return 0.0
        try:
            mol = Chem.MolFromSmiles(prefix)
            mol = Chem.AddHs(mol)
            if mol is None:
                return 0.0
            return 1.0 if mol.GetRingInfo().NumRings() > 0 else 0.0
        except Exception:
            return 0.0


    def _functional_group_score(self, prefix: str) -> float:
        """
        使用 RDKit 对 prefix（SMILES）进行解析并基于分子结构计算功能团得分。
        若 require_ring=True（可选），则无环返回 0；否则对无环分子给较低但非零的分数。
        """
        if not prefix:
            return 0.0

        try:
            mol = Chem.MolFromSmiles(prefix)
            mol = Chem.AddHs(mol)
            if mol is None:
                return 0.0

            num_rings = mol.GetRingInfo().NumRings()
            num_aromatic_rings = len([ring for ring in mol.GetRingInfo().AtomRings()
                                    if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)])

            # 功能团模式（使用 SMARTS）
            patterns = [
                # 常见功能团：
                ("[OX2H]", "hydroxyl"),           # 羟基
                ("[CX3](=O)[OX2H1]", "carboxyl"), # 羧基
                ("[NX3;H2,H1]", "amine"),         # 胺基
                ("[CX3](=O)[NX3]", "amide"),      # 酰胺
                ("[CX3H1](=O)", "aldehyde"),      # 醛
                ("[CX3](=O)[CX4]", "ketone"),     # 酮
                ("[CX4][OX2H]", "ether"),         # 醚
                # MOF 特征性功能团：
                ("c1ccccc1", "phenyl"),            # 苯基
                ("[#6]~[#7]", "C-N"),              # C-N 键（常见连接方式）
                ("[#6]~[#8]", "C-O"),              # C-O 键（常见连接方式）
                ("[#6]~[#16]", "C-S"),             # C-S 键（硫醇、硫醚）
                ("c1ccc([*])cc1", "substituted-benzene")  # 取代苯
            ]

            found_groups = set()
            for smarts, name in patterns:
                patt = Chem.MolFromSmarts(smarts)
                if patt and mol.HasSubstructMatch(patt):
                    found_groups.add(name)

            # 计算分数：环结构 + 功能团多样性
            ring_score = min(1.0, num_rings / 3.0)  # 环越多，分越高（饱和在 3 个环）
            aromatic_bonus = 0.2 if num_aromatic_rings > 0 else 0.0
            functional_group_score = min(1.0, len(found_groups) / 5.0)  # 功能团越多，分越高（饱和在 5 个）

            # 综合得分（权重可调）
            total_score = 0.4 * ring_score + 0.3 * aromatic_bonus + 0.3 * functional_group_score
            return float(total_score)

        except Exception as e:
            logger.debug(f"_functional_group_score exception: {e}")
            return 0.0

    def _porosity_proxy(self, prefix: str) -> float:
        p = prefix.strip()
        print(p)
        if not p:
            return 0.0

        L = len(p)
        mu1, mu2 = 68.77, 158.77 # 68.77, 158.84
        sigma1, sigma2 = 86.60, 96.60 # 36.60, 79.05

        g1 = np.exp(-((L - mu1) ** 2) / (2.0 * sigma1 ** 2))
        g2 = np.exp(-((L - mu2) ** 2) / (2.0 * sigma2 ** 2))

        # 在 L=mu1 或 L=mu2 处，score 都为 1
        score = float(np.clip(max(g1, g2), 0.0, 1.0))
        return score

    def _novelty_score(self, prefix: str) -> float:
        if not hasattr(self, '_recent_prefixes') or not hasattr(self, '_recent_prefix_set'):
            from collections import deque
            RECENT_MAXLEN = 1000
            self._recent_prefixes = deque(maxlen=RECENT_MAXLEN)
            self._recent_prefix_set = set()

        p = prefix.strip() if prefix else ''
        if not p:
            return 0.0

        # 规范化：按 '.' 分割，排序后重新拼接
        normalized_key = '.'.join(sorted(p.split('.')))

        # 直接比较，无需再次规范化 mofid_full_set
        if normalized_key in self.mofid_full_set or normalized_key in self._recent_prefix_set:
            if normalized_key not in self._recent_prefix_set:
                self._recent_prefixes.append(normalized_key)
                self._recent_prefix_set.add(normalized_key)
                if len(self._recent_prefix_set) > len(self._recent_prefixes):
                    self._recent_prefix_set.intersection_update(set(self._recent_prefixes))
            return 0.0

        self._recent_prefixes.append(normalized_key)
        self._recent_prefix_set.add(normalized_key)

        if len(self._recent_prefix_set) > len(self._recent_prefixes):
            self._recent_prefix_set.intersection_update(set(self._recent_prefixes))

        return 1.0
    def _adsorption_score(self, pred: str) -> float:
        """
        使用高斯CDF对两个指标评分
        """

        try:
            from scipy.stats import norm
            
            # ========== 第一个指标 ==========
            score1 = 0.0
            val1 = None
            
            if self._adsorption_engine is None:
                if not N2_MODEL_PATH:
                    logger.warning("MOF_N2_MODEL is not set; skipping N2 adsorption surrogate.")
                else:
                    from swift.llm import PtEngine
                    self._adsorption_engine = PtEngine(
                        N2_MODEL_PATH,
                        device_backend='pt',
                        task_type='seq_cls',
                        problem_type='regression',
                        num_labels=1,
                        use_chat_template=False,
                        max_batch_size=32
                    )

            if self._adsorption_engine is not None:
                val1 = predict_adsorption_amount(
                    pred,
                    engine=self._adsorption_engine,
                    max_tokens=32,
                    temperature=0.0
                )
            
            if val1 is not None:
                mu1 = 0.2875
                sigma1 = 0.1490
                score1 = float(norm.cdf(val1, loc=mu1, scale=sigma1))
                score1 = max(0.0, min(1.0, score1))
            
            # ========== 第二个指标 ==========
            score2 = 0.0
            val2 = None
            
            if self._adsorption_engine2 is None:
                if not CO2_MODEL_PATH:
                    logger.warning("MOF_CO2_MODEL is not set; skipping CO2 adsorption surrogate.")
                else:
                    from swift.llm import PtEngine
                    self._adsorption_engine2 = PtEngine(
                        CO2_MODEL_PATH,
                        device_backend='pt',
                        task_type='seq_cls',
                        problem_type='regression',
                        num_labels=1,
                        use_chat_template=False,
                        max_batch_size=32
                    )

            if self._adsorption_engine2 is not None:
                val2 = predict_adsorption_amount(
                    pred,
                    engine=self._adsorption_engine2,
                    max_tokens=32,
                    temperature=0.0
                )
            
            if val2 is not None:
                mu2 = 1.84
                sigma2 = 1.39
                score2 = float(norm.cdf(val2, loc=mu2, scale=sigma2))
                score2 = max(0.0, min(1.0, score2))
            
            # ========== 加权求和 ==========
            weight1 = 0.5
            weight2 = 0.5
            
            final_score = weight1 * score1 + weight2 * score2
            
            print(f"Adsorption CDF - val1={val1}, score1={score1:.4f}, "
                  f"val2={val2}, score2={score2:.4f}, final={final_score:.4f}")
            
            return float(max(0.0, min(1.0, final_score)))
            
        except Exception as e:
            logger.warning(f"_adsorption_score failed: {e}")
            return 0.0
    
    def _structure_3d_score(self, prefix: str) -> float:
            print(f"Computing structure 3D score for prefix: {prefix}")
            """
            对 prefix 按 '.' 切分每个 fragment，转 SMILES -> mol -> embed -> UFF optimize。
            过滤掉含有金属元素（self.METAL_SYMBOLS 中定义的元素）的 fragment。
            检查每个 mol 中 C 和 N 原子是否不饱和：
            - C 原子：键阶和 < 4 为不饱和（基于与重原子的键，**不加氢**）
            - N 原子：键阶和 < 3 为不饱和
            如果不饱和 C/N 原子总数 <= 2 -> 该 fragment 得 0。
            否则，用该 mol 中所有 C/N 原子两两的 3D 距离计算 pair scores（最近 0.5，最远 1.0），取平均。
            最终返回所有 fragment 得分的平均，或 0.0（无有效 fragment）。
            """

            frags = [f.strip() for f in prefix.split('.') if f.strip()]
            if not frags:
                return 0.0

            frag_scores = []
            for frag in frags:
                try:
                    mol = Chem.MolFromSmiles(frag)
                    if mol is None:
                        # 解析失败，跳过
                        continue

                    # 检查该 fragment 是否含有金属元素，如果有则跳过
                    has_metal = False
                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() in self.METAL_SYMBOLS:
                            has_metal = True
                            break
                    
                    if has_metal:
                        logger.debug(f"_structure_3d_score: skipping fragment {frag} containing metal")
                        continue

                    # 不再 AddHs，直接在原始 SMILES 对应的分子上做 3D 嵌入和优化
                    try:
                        params = AllChem.ETKDGv3()
                        params.randomSeed = 42
                        embed_ok = AllChem.EmbedMolecule(mol, params)
                        if embed_ok != 0:
                            # embed 失败时仍尝试优化
                            pass
                        AllChem.UFFOptimizeMolecule(mol)
                    except Exception as e:
                        logger.debug(f"_structure_3d_score: embed/opt failed for frag {frag}: {e}")
                        continue

                    # Collect carbon and nitrogen atom indices and positions
                    conf = mol.GetConformer()
                    cn_idxs = [atom.GetIdx() for atom in mol.GetAtoms() 
                            if atom.GetAtomicNum() in (6, 7)]  # 6=C, 7=N
                    if len(cn_idxs) == 0:
                        continue

                    # Determine unsaturated C and N atoms using bond-order sum
                    # （只看与非 H 原子的键阶）
                    # C: 键阶和 < 4 为不饱和
                    # N: 键阶和 < 3 为不饱和
                    unsat_cn_atoms = []
                    for idx in cn_idxs:
                        atom = mol.GetAtomWithIdx(idx)
                        atomic_num = atom.GetAtomicNum()

                        total_bond_order = 0.0
                        for bond in atom.GetBonds():
                            # 跳过与氢原子的键，只按重原子之间的键阶判断不饱和
                            nbr = bond.GetOtherAtom(atom)
                            if nbr.GetAtomicNum() == 1:
                                continue
                            try:
                                bo = bond.GetBondTypeAsDouble()
                                if bo is None:
                                    raise ValueError("GetBondTypeAsDouble returned None")
                            except Exception:
                                bt = bond.GetBondType()
                                if bt == Chem.BondType.SINGLE:
                                    bo = 1.0
                                elif bt == Chem.BondType.DOUBLE:
                                    bo = 2.0
                                elif bt == Chem.BondType.TRIPLE:
                                    bo = 3.0
                                elif bond.GetIsAromatic():
                                    bo = 1.5
                                else:
                                    bo = 1.0
                            total_bond_order += float(bo)

                        # 判断不饱和：
                        # C 原子（原子序数=6）：键阶和 < 4 为不饱和
                        # N 原子（原子序数=7）：键阶和 < 3 为不饱和
                        is_unsat = False
                        if atomic_num == 6:  # Carbon
                            if total_bond_order < 4.0 - 1e-6:
                                is_unsat = True
                        elif atomic_num == 7:  # Nitrogen
                            if total_bond_order < 3.0 - 1e-6:
                                is_unsat = True
                        
                        if is_unsat:
                            unsat_cn_atoms.append(idx)

                    # If unsaturated C/N atoms <= 2 -> this fragment yields 0
                    if len(unsat_cn_atoms) <= 2:
                        frag_scores.append(0.0)
                        continue

                    # Compute pairwise distances among all C/N atoms (use 3D coords)
                    coords = [conf.GetAtomPosition(i) for i in cn_idxs]
                    dists = []
                    for i in range(len(coords)):
                        for j in range(i + 1, len(coords)):
                            dx = coords[i].x - coords[j].x
                            dy = coords[i].y - coords[j].y
                            dz = coords[i].z - coords[j].z
                            d = (dx * dx + dy * dy + dz * dz) ** 0.5
                            dists.append(d)

                    if not dists:
                        frag_scores.append(0.0)
                        continue

                    min_d = min(dists)
                    max_d = max(dists)
                    if abs(max_d - min_d) < 1e-8:
                        pair_scores = [0.75 for _ in dists]
                    else:
                        pair_scores = []
                        for d in dists:
                            norm = (d - min_d) / (max_d - min_d)  # 0..1
                            score = 0.6 + 0.5 * norm              # 0.6..1.1
                            pair_scores.append(score)

                    frag_score = float(sum(pair_scores) / len(pair_scores))
                    frag_scores.append(frag_score)

                except Exception as e:
                    logger.debug(f"_structure_3d_score: exception on frag {frag}: {e}")
                    continue

            if not frag_scores:
                return 0.0
            for frag_s in frag_scores:
                print(f"Fragment 3D score: {frag_s:.3f}")
            return float(sum(frag_scores) / len(frag_scores))

    def _cif_generation_score(self, pred: str) -> float:
        """
        结构生成 + TOBACCO + CIF 评估的一体化奖励函数。
        逻辑：
        1）pred 按空格分为两部分（只按第一个空格切）
           - 前半部分：金属 / 拓扑等前缀
           - 后半部分：SMILES
           若：
             - 没有空格（parts != 2），或
             - 前半部分中'.'数量 > 2，或
             - 后半部分中'.'数量 != 2
           则直接返回 0 分
        2）调用 process_mof_smiles(pred)
           - 若返回 0，则整体奖励为 0
        3）调用 run_tobacco_with_edge_folders(CHARGES)
           - 若返回 0，则整体奖励为 0
        4）调用 run_evaluate()，返回其结果作为最终得分（0–1）
        """
        print(f"Computing CIF generation score for pred: {pred}")
        try:
            pred = (pred or "").strip()
            if not pred:
                print("MOFReward: cif_generation_score: empty pred")
                return 0.0

            # 按第一个空格拆分为两部分
            parts = pred.split(maxsplit=1)
            if len(parts) != 2:
                print("MOFReward: cif_generation_score: invalid format (not 2 parts)")
                return 0.0

            head, tail = parts[0], parts[1]

            # 前半部分 '.' 不能多于 5 个
            if head.count('.') > 5:
                print("MOFReward: cif_generation_score: too many '.' in head")
                return 0.0

            # 后半部分（SMILES）必须恰好包含 2 个 '.'
            if tail.count('.') != 2:
                print("MOFReward: cif_generation_score: invalid '.' count in SMILES part")
                return 0.0

            # 1) 先做 MOF 片段 → CIF 预处理
            prep_ret = process_mof_smiles(pred)
            if prep_ret == 0:
                for root_dir in [edges_dir, output_cifs_dir]:
                    if root_dir.exists() and root_dir.is_dir():
                        for p in root_dir.iterdir():
                            shutil.rmtree(p)
                            print(f"MOFReward: cleaned up directory {p}")

                print("MOFReward: process_mof_smiles failed, returning 0.0")
                return 0.0
            else:
                print(f"MOFReward: process_mof_smiles returned {prep_ret}")

            # 2) 跑 TOBACCO：根据 edges 生成结构
            tobacco_ret = run_tobacco_with_edge_folders(CHARGES=True)
            if tobacco_ret == 0:
                for root_dir in [edges_dir, output_cifs_dir]:
                    if root_dir.exists() and root_dir.is_dir():
                        for p in root_dir.iterdir():
                            shutil.rmtree(p)
                            print(f"MOFReward: cleaned up directory {p}")
                print("MOFReward: run_tobacco_with_edge_folders failed, returning 0.0")
                return 0.0
            else:
                print(f"MOFReward: run_tobacco_with_edge_folders returned {tobacco_ret}")

            # 3) 评估 output_cifs 下生成的 CIF，并返回 0–1 的分数
            score = run_evaluate()
            print(f"MOFReward: run_evaluate returned score={score}")
            try:
                score = float(score)
            except Exception:
                score = 0.0

            # 保证在 [0, 1] 区间
            score = max(0.0, min(1.0, score))
            # 清理 edges 目录下的所有子目录
            if edges_dir.exists() and edges_dir.is_dir():
                for subdir in edges_dir.iterdir():
                    try:
                        if subdir.is_dir():
                            shutil.rmtree(subdir)
                            print(f"MOFReward: cleaned up edges subdirectory {subdir.name}")
                        elif subdir.is_file():
                            subdir.unlink()
                            print(f"MOFReward: cleaned up edges file {subdir.name}")
                    except Exception as e:
                        print(f"MOFReward: failed to clean up {subdir}: {e}")
            
            # 清理 output_cifs 目录下的所有文件
            if output_cifs_dir.exists() and output_cifs_dir.is_dir():
                for cif_file in output_cifs_dir.iterdir():
                    try:
                        if cif_file.is_file():
                            cif_file.unlink()
                            print(f"MOFReward: cleaned up output file {cif_file.name}")
                        elif cif_file.is_dir():
                            shutil.rmtree(cif_file)
                            print(f"MOFReward: cleaned up output directory {cif_file.name}")
                    except Exception as e:
                        print(f"MOFReward: failed to clean up {cif_file}: {e}")
            return score

        except Exception:
            # 出现任何异常则给 0 分，避免训练崩溃
            print("="*20 + " EXCEPTION IN _cif_generation_score " + "="*20)
            traceback.print_exc()  # 打印详细的错误堆栈
            print("="*60)

            # 清理 edges 目录下的所有子目录
            if edges_dir.exists() and edges_dir.is_dir():
                for subdir in edges_dir.iterdir():
                    try:
                        if subdir.is_dir():
                            shutil.rmtree(subdir)
                            print(f"MOFReward: cleaned up edges subdirectory {subdir.name}")
                        elif subdir.is_file():
                            subdir.unlink()
                            print(f"MOFReward: cleaned up edges file {subdir.name}")
                    except Exception as e:
                        print(f"MOFReward: failed to clean up {subdir}: {e}")
            
            # 清理 output_cifs 目录下的所有文件
            if output_cifs_dir.exists() and output_cifs_dir.is_dir():
                for cif_file in output_cifs_dir.iterdir():
                    try:
                        if cif_file.is_file():
                            cif_file.unlink()
                            print(f"MOFReward: cleaned up output file {cif_file.name}")
                        elif cif_file.is_dir():
                            shutil.rmtree(cif_file)
                            print(f"MOFReward: cleaned up output directory {cif_file.name}")
                    except Exception as e:
                        print(f"MOFReward: failed to clean up {cif_file}: {e}")
            
            print("MOFReward: cif_generation_score encountered exception, returning 0.0")
            return 0.0


    # -------------------------
    # __call__
    # -------------------------
    def __call__(self, completions, **kwargs) -> List[float]:
        """
        completions: list[str], 每一项是完整 model output，比如:
            "<SMILES fragments> MOFid-v1.pcu.cat1"
        kwargs 可以包含 'adsorption_model_path'，用于指定用于吸附预测的模型路径
        """
        rewards = []

        for pred in completions:
            ok, prefix, suffix = self._strict_format_check(pred)
            if not ok:
                rewards.append(0.0)
                continue

            chem  = self._chemical_validity_score(prefix)
            metal = self._metal_presence_score(prefix)
            func  = self._functional_group_score(prefix)
            ring  = self._has_ring(prefix)
            sim   = self._similarity_score(prefix)
            poro  = self._porosity_proxy(prefix)
            novel = self._novelty_score(prefix)
            ads   = self._adsorption_score(pred)
            struct3d = self._structure_3d_score(prefix)
            cif   = self._cif_generation_score(pred)

            total = (
                self.weights["chemical_validity"] * chem +
                self.weights["metal_presence"] * metal +
                self.weights["functional_group"] * func +
                self.weights["has_ring"] * ring +
                self.weights["similarity"] * sim +
                self.weights["porosity_proxy"] * poro +
                self.weights["novelty"] * novel +
                self.weights["adsorption"] * ads +
                self.weights["structure_3d"] * struct3d +
                self.weights["cif_generation"] * cif
            )
            rewards.append(float(max(0.0, min(1.0, total))))
            print(f"MOFReward: pred='{pred}' => scores: chem={chem:.3f}, metal={metal:.3f}, func={func:.3f}, ring={ring:.3f}, sim={sim:.3f}, poro={poro:.3f}, novel={novel:.3f}, ads={ads:.3f}, struct3d={struct3d:.3f}, cif={cif:.3f} => total={total:.3f}")
        return rewards
    
    def __del__(self):
        """安全释放 adsorption_engine 资源（若是注入的 engine，确保其关闭）"""
        try:
            # 清理第一个引擎
            if hasattr(self, "_adsorption_engine") and self._adsorption_engine:
                try:
                    if hasattr(self._adsorption_engine, "close"):
                        self._adsorption_engine.close()
                except Exception:
                    pass
                finally:
                    self._adsorption_engine = None
            
            # 清理第二个引擎
            if hasattr(self, "_adsorption_engine2") and self._adsorption_engine2:
                try:
                    if hasattr(self._adsorption_engine2, "close"):
                        self._adsorption_engine2.close()
                except Exception:
                    pass
                finally:
                    self._adsorption_engine2 = None
        except Exception:
            pass


# 注册
orms['mof_reward'] = MOFRewardORM


# -------------------------
# 外部预测函数：predict_adsorption_amount
# -------------------------

def predict_adsorption_amount(pred: str,
                              engine=None,
                              model_path: Optional[str] = DEFAULT_ADSORPTION_MODEL,
                              device_backend: str = 'pt',
                              max_tokens: int = 256,
                              temperature: float = 0.0) -> Optional[float]:
    """
    使用模型或带 adapter 的引擎预测吸附量（mg/g）。
    支持复用已创建的 engine，或临时加载 model_path + adapter_path。
    """

    if engine is None and not model_path:
        logger.error("predict_adsorption_amount: no engine and no model_path provided -> returning None")
        return None

    try:
        from swift.llm import PtEngine, InferRequest, RequestConfig
    except Exception as e:
        logger.error(f"predict_adsorption_amount: cannot import swift.llm: {e}")
        return None

    created_engine = False
    temp_engine = None
    try:
        if engine is None:
            # Create a temporary PtEngine when the caller does not pass one.
            temp_engine = PtEngine(
                model_path,
                device_backend=device_backend,
                task_type='seq_cls',
                problem_type='regression',
                num_labels=1,
                use_chat_template=False,
                max_batch_size=8
            )
            engine = temp_engine
            created_engine = True

        # 组装 prompt
        prompt = (f"{pred}")
        req = InferRequest(messages=[{"role": "user", "content": prompt}])
        cfg = RequestConfig(max_tokens=max_tokens, temperature=temperature, stream=False)

        results = engine.infer([req], cfg)

        # 解析结果
        val = None
        try:
            if isinstance(results, list) and results:
                first = results[0]
                content = None
                try:
                    content = first.choices[0].message.content
                except Exception:
                    try:
                        content = first.choices[0].text
                    except Exception:
                        content = None

                if content is not None:
                    if isinstance(content, (int, float)):
                        val = float(content)
                    else:
                        import re
                        s = str(content)
                        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
                        if m:
                            try:
                                val = float(m.group())
                            except Exception:
                                val = None
        except Exception as parse_e:
            logger.warning(f"predict_adsorption_amount: parse error {parse_e}")
            val = None

        if val is None:
            logger.warning("predict_adsorption_amount: empty or unparsable generation content")
            return None
        print(f"Predicted adsorption amount: {val} (from prompt: {prompt})")
        return val

    except Exception as e:
        logger.exception(f"predict_adsorption_amount error: {e}")
        return None
    finally:
        if created_engine and temp_engine is not None:
            try:
                if hasattr(temp_engine, "close"):
                    temp_engine.close()
            except Exception:
                pass


"""
Step 5: 准备 GCMC 模拟输入文件 (UFF4MOF 骨架 + TraPPE 气体)
功能：
  - 为每个优化后的 MOF 结构生成 LAMMPS GCMC 输入脚本
  - 使用 TraPPE 力场描述 CO₂ 和 N₂ 吸附质
  - MOF 骨架使用 UFF4MOF (冻结骨架近似)
  - 气体-骨架交互使用 Lorentz-Berthelot 混合规则
  - 298 K, 多压力点吸附等温线
  - 自动从 relaxed data 文件解析实际 atom type 数
"""

import json
import os
import re
import math
from pathlib import Path

# ======================== 配置参数 ========================
BASE_DIR = Path(__file__).resolve().parent.parent
LAMMPS_DIR = BASE_DIR / "data" / "lammps_inputs"
GCMC_DIR = BASE_DIR / "data" / "gcmc_inputs"
RESULTS_DIR = BASE_DIR / "results"

# GCMC 模拟参数 (对应论文 Methods)
TEMPERATURE = 298.0       # K
INIT_CYCLES = 1000        # 1e3 初始化 (冻结骨架, MC 收敛快)
EQUIL_CYCLES = 5000       # 5e3 平衡
PROD_CYCLES = 5000        # 5e3 产出

# 压力点 (bar) — 完整等温线: 8 个压力点覆盖 0.01-1.0 bar
# 包含烟气典型分压 (CO2: 0.15 bar, N2: 0.75 bar) 和标准条件 (1.0 bar)
PRESSURES_CO2 = [0.01, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0]
PRESSURES_N2  = [0.01, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0]

# TraPPE 力场参数
# CO₂: 三位点模型 (O=C=O 线性刚性分子)
# Potoff & Siepmann, AIChE J. 2001, 47, 1676
TRAPPE_CO2 = {
    "description": "TraPPE CO2 (3-site rigid linear)",
    "sites": [
        {"name": "C_co2",  "mass": 12.011, "charge":  0.70,  "sigma": 2.800, "epsilon": 0.0537},
        {"name": "O_co2",  "mass": 15.999, "charge": -0.35,  "sigma": 3.050, "epsilon": 0.1570},
    ],
    "bond_length_CO": 1.16,  # Å
}

# N₂: 三位点模型 (N-COM-N, 含虚拟电荷位点)
# Potoff & Siepmann, AIChE J. 2001, 47, 1676
TRAPPE_N2 = {
    "description": "TraPPE N2 (3-site rigid linear)",
    "sites": [
        {"name": "N_n2",   "mass": 14.007, "charge": -0.482, "sigma": 3.310, "epsilon": 0.0715},
        {"name": "COM_n2", "mass": 0.001,  "charge":  0.964, "sigma": 0.000, "epsilon": 0.0000},
    ],
    "bond_length_NCOM": 0.55,  # Å (N-COM距离)
}


def parse_data_file_types(data_file: Path) -> dict:
    """
    从 LAMMPS data 文件解析 atom type 数和相关信息。
    返回: {
        "n_atom_types": int,
        "n_bond_types": int,
        "n_atoms": int,
        "type_masses": {tid: mass},
        "type_labels": {tid: label},  # 从注释提取 e.g. "O_2", "Cu3f2"
    }
    """
    info = {
        "n_atom_types": 0,
        "n_bond_types": 0,
        "n_atoms": 0,
        "type_masses": {},
        "type_labels": {},
    }

    with open(data_file, "r") as f:
        in_masses = False
        for line in f:
            stripped = line.strip()

            if stripped.endswith("atom types"):
                info["n_atom_types"] = int(stripped.split()[0])
            elif stripped.endswith("bond types"):
                info["n_bond_types"] = int(stripped.split()[0])
            elif stripped.endswith("atoms"):
                parts = stripped.split()
                if len(parts) == 2:
                    info["n_atoms"] = int(parts[0])

            if stripped == "Masses":
                in_masses = True
                continue
            if in_masses:
                if stripped in ("", ) or stripped.startswith("Pair") or stripped.startswith("Bond"):
                    if stripped.startswith("Pair") or stripped.startswith("Bond"):
                        in_masses = False
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        tid = int(parts[0])
                        mass = float(parts[1])
                        info["type_masses"][tid] = mass
                        if "#" in stripped:
                            label = stripped.split("#")[1].strip()
                            info["type_labels"][tid] = label
                    except (ValueError, IndexError):
                        pass

    return info


def parse_pair_coeffs_from_in(in_file: Path) -> list:
    """
    从 lammps-interface 生成的 in 文件提取 pair_coeff 行。
    (实际上 lammps-interface 把 pair_coeff 写在 data 文件里的 Pair Coeffs 部分)
    """
    pair_coeffs = []
    with open(in_file, "r") as f:
        for line in f:
            if line.strip().startswith("pair_coeff"):
                pair_coeffs.append(line.strip())
    return pair_coeffs


def parse_pair_coeffs_from_data(data_file: Path) -> list:
    """
    从 data 文件提取 Pair Coeffs 部分 (lammps-interface 生成在此)。
    返回: [(type_id, epsilon, sigma, label), ...]
    """
    coeffs = []
    in_section = False
    with open(data_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "Pair Coeffs":
                in_section = True
                continue
            if in_section:
                if not stripped or stripped.startswith("#"):
                    if not stripped:
                        continue
                    else:
                        continue
                # 检查是否到了下一个 section
                if stripped in ("Bond Coeffs", "Angle Coeffs", "Atoms",
                                "Bonds", "Angles", "Dihedrals", "Impropers"):
                    break
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        tid = int(parts[0])
                        eps = float(parts[1])
                        sig = float(parts[2])
                        label = ""
                        if "#" in stripped:
                            label = stripped.split("#")[1].strip()
                        coeffs.append((tid, eps, sig, label))
                    except (ValueError, IndexError):
                        pass
    return coeffs


def create_molecule_template(gas_type: str, output_dir: Path) -> str:
    """
    创建 LAMMPS molecule template 文件 (用于 fix gcmc 插入分子)。
    返回模板文件路径。
    """
    if gas_type == "CO2":
        # O=C=O 线性分子, bond_length C-O = 1.16 Å
        # 原子顺序: O1, C, O2 (沿 x 轴)
        d = TRAPPE_CO2["bond_length_CO"]
        mol_content = f"""# TraPPE CO2 molecule template
3 atoms
2 bonds

Coords

1  {-d:.4f}  0.0000  0.0000
2   0.0000  0.0000  0.0000
3   {d:.4f}  0.0000  0.0000

Types

1  1
2  2
3  1

Charges

1  {TRAPPE_CO2['sites'][1]['charge']:.4f}
2  {TRAPPE_CO2['sites'][0]['charge']:.4f}
3  {TRAPPE_CO2['sites'][1]['charge']:.4f}

Masses

1  {TRAPPE_CO2['sites'][1]['mass']:.4f}
2  {TRAPPE_CO2['sites'][0]['mass']:.4f}
3  {TRAPPE_CO2['sites'][1]['mass']:.4f}

Bonds

1  1  1  2
2  1  2  3
"""
        mol_file = output_dir / "CO2.mol"

    elif gas_type == "N2":
        # N-COM-N 线性分子
        d = TRAPPE_N2["bond_length_NCOM"]
        mol_content = f"""# TraPPE N2 molecule template
3 atoms
2 bonds

Coords

1  {-d:.4f}  0.0000  0.0000
2   0.0000  0.0000  0.0000
3   {d:.4f}  0.0000  0.0000

Types

1  1
2  2
3  1

Charges

1  {TRAPPE_N2['sites'][0]['charge']:.4f}
2  {TRAPPE_N2['sites'][1]['charge']:.4f}
3  {TRAPPE_N2['sites'][0]['charge']:.4f}

Masses

1  {TRAPPE_N2['sites'][0]['mass']:.4f}
2  {TRAPPE_N2['sites'][1]['mass']:.4f}
3  {TRAPPE_N2['sites'][0]['mass']:.4f}

Bonds

1  1  1  2
2  1  2  3
"""
        mol_file = output_dir / "N2.mol"
    else:
        raise ValueError(f"不支持的气体类型: {gas_type}")

    with open(mol_file, "w") as f:
        f.write(mol_content)

    return str(mol_file)


def generate_gcmc_input(name: str, data_file: Path,
                        gas_type: str, pressure: float,
                        work_dir: Path, fw_type_info: dict,
                        mol_template: str) -> str:
    """
    生成单个压力点的 GCMC 输入脚本。

    骨架: UFF4MOF 参数 (已在 data 文件的 Pair Coeffs 中)
    气体: TraPPE 参数 (在脚本中显式指定)
    交互: Lorentz-Berthelot 混合规则
    """
    n_fw_types = fw_type_info["n_atom_types"]

    if gas_type == "CO2":
        gas_params = TRAPPE_CO2
        gas_sites = TRAPPE_CO2["sites"]
        # CO2: type O_co2 和 C_co2
        # 在 molecule template 中 type 1 = O_co2, type 2 = C_co2
        gas_type_names = ["O_co2", "C_co2"]
    elif gas_type == "N2":
        gas_params = TRAPPE_N2
        gas_sites = TRAPPE_N2["sites"]
        gas_type_names = ["N_n2", "COM_n2"]
    else:
        raise ValueError(f"不支持的气体类型: {gas_type}")

    n_gas_types = 2  # 每种气体都是 2 种 atom type
    total_types = n_fw_types + n_gas_types
    n_fw_bond_types = fw_type_info["n_bond_types"]

    P_atm = pressure / 1.01325

    # 计算化学势 mu (kcal/mol)
    # 理想气体: mu = kT * ln(P / P_ref), P_ref = 1 atm
    # LAMMPS GCMC 中 mu 是相对于理想气体参考态的偏差
    # 对于理想气体近似, mu = 0 (结合 pressure 关键字)
    mu = 0.0

    script = f"""# ============================================================
# GCMC: {name} + {gas_type} at {pressure} bar, {TEMPERATURE} K
# Framework: UFF4MOF (frozen), Gas: TraPPE
# ============================================================

units           real
atom_style      full
boundary        p p p

# ---- 力场样式 (必须在 read_data 之前定义, 因为 data 文件含 Pair/Bond/Dihedral Coeffs) ----
pair_style      lj/cut/coul/long 12.5
pair_modify     mix arithmetic tail yes
kspace_style    pppm 1.0e-5
bond_style      harmonic
angle_style     harmonic
dihedral_style  harmonic
improper_style  harmonic
special_bonds   lj/coul 0.0 0.0 1.0

# 读取优化后的 MOF 骨架 (含 Pair Coeffs / Bond Coeffs)
# extra/atom/types 和 extra/bond/types 为气体分子预留额外的 type 槽位
read_data       {data_file.name} extra/atom/types {n_gas_types} extra/bond/types 1

# 骨架的 pair_coeff 已在 data 文件中 (Pair Coeffs section)
# 以下为气体分子的 LJ 参数 (新增的 type)
"""

    # 气体 pair_coeff (仅对角项, 交叉项由 mix arithmetic 自动处理)
    # gas_sites[0] 对应 molecule template 中的 type 1 → 全局 type = n_fw_types+1
    # gas_sites[1] 对应 molecule template 中的 type 2 → 全局 type = n_fw_types+2
    # 注意: molecule template 中的 type 是局部的, LAMMPS 会映射到全局
    # 但 pair_coeff 需要用全局 type
    for i, site in enumerate(gas_sites):
        gtype = n_fw_types + i + 1
        script += f"pair_coeff      {gtype} {gtype} {site['epsilon']:.4f} {site['sigma']:.3f}  # {gas_type_names[i]}\n"

    # 为新增的 gas type 指定质量
    script += f"\n# 气体原子质量\n"
    for i, site in enumerate(gas_sites):
        gtype = n_fw_types + i + 1
        script += f"mass            {gtype} {site['mass']:.4f}  # {gas_type_names[i]}\n"


    # 为气体分子的 extra bond type 设置 bond_coeff
    # read_data 中 extra/bond/types 1 预留了一个额外的 bond type (编号 = n_fw_bond_types + 1)
    # 使用大弹性常数 (刚性键近似) 以保持 TraPPE 分子几何
    gas_bond_type = n_fw_bond_types + 1
    if gas_type == "CO2":
        gas_bond_length = TRAPPE_CO2["bond_length_CO"]       # 1.16 Å
    else:
        gas_bond_length = TRAPPE_N2["bond_length_NCOM"]      # 0.55 Å
    script += f"\n# 气体分子 bond 参数 (刚性键, K 很大)\n"
    script += f"bond_coeff      {gas_bond_type} 10000.0 {gas_bond_length:.4f}  # gas rigid bond\n"

    # data 文件中没有 Angle Coeffs / Dihedral Coeffs / Improper Coeffs section,
    # 但声明了 angle_style harmonic 等, LAMMPS 要求所有 type 都有系数。
    # 骨架完全冻结, 这些 bonded 力对模拟无影响, 设置 dummy 系数即可。
    script += f"""
# 骨架冻结, angle/dihedral/improper 力不影响结果, 设置 dummy 系数
angle_coeff     * 0.0 109.47
dihedral_coeff  * 0.0 1 1
improper_coeff  * 0.0 0.0
"""

    script += f"""
# ---- 分子模板 ----
molecule        gas_mol {Path(mol_template).name} offset {n_fw_types} {n_fw_bond_types} 0 0 0

# ---- 原子分组 ----
group           framework type 1:{n_fw_types}
# 气体分子组将在插入后自动识别

# ---- 冻结骨架 ----
fix             freeze framework setforce 0.0 0.0 0.0
velocity        framework set 0.0 0.0 0.0

# ---- GCMC ----
# fix gcmc 语法: fix ID group-ID gcmc N X M type seed T mu displace [keyword ...]
# N=100: 每 100 步尝试一次 GCMC 操作
# X=100: 每次尝试 100 次 GCMC 交换
# M=0: 不做额外 MC 平移 (骨架冻结, 气体由 GCMC 插入/删除)
# type=0: 使用 mol 关键字时 type 必须为 0
# seed=12345: 随机数种子 (正整数)
# T: 温度, mu: 化学势, displace: 最大平移距离
fix             mygcmc all gcmc 100 100 0 0 12345 {TEMPERATURE} {mu:.2f} 0.5 &
                mol gas_mol pressure {P_atm:.6f} &
                full_energy tfac_insert 1.0 &
                group framework

# ---- 输出变量 ----
variable        n_gas equal count(all)-{fw_type_info['n_atoms']}

# ---- 热力学输出 ----
thermo_style    custom step atoms temp press pe ke etotal v_n_gas
thermo          1000

# triclinic 盒子在设置 fix gcmc 后需要重新初始化 kspace
kspace_style    pppm 1.0e-5

# ---- 初始化: {INIT_CYCLES} 步 ----
run             {INIT_CYCLES}

# ---- 平衡: {EQUIL_CYCLES} 步 ----
run             {EQUIL_CYCLES}

# ---- 产出: {PROD_CYCLES} 步 ----
# thermo 每 100 步输出 v_n_gas, 由 Python 解析日志取平均
thermo          100
print           "PRODUCTION_START"

run             {PROD_CYCLES}

print           "PRODUCTION_END"
# 同时输出最终快照的瞬时值作为参考
print           "GCMC_RESULT: {name} {gas_type} {pressure} ${{n_gas}}"
"""
    return script


def main():
    print("=" * 60)
    print("Step 5: 准备 GCMC 模拟输入文件 (UFF4MOF + TraPPE)")
    print("=" * 60)

    # 读取优化成功的结构
    results_file = RESULTS_DIR / "lammps_optimization_results.json"
    summary_file = LAMMPS_DIR / "structure_summary.json"

    if not results_file.exists() or not summary_file.exists():
        print("[错误] 未找到必要的输入文件，请先运行前序步骤")
        return

    with open(results_file, "r") as f:
        lammps_results = json.load(f)
    with open(summary_file, "r") as f:
        structure_info = {s["name"]: s for s in json.load(f)}

    successful = [r for r in lammps_results if r["status"] == "success"]
    print(f"\n共 {len(successful)} 个结构待 GCMC 模拟")
    print(f"气体: CO2 ({len(PRESSURES_CO2)} 个压力点), N2 ({len(PRESSURES_N2)} 个压力点)")
    print(f"温度: {TEMPERATURE} K\n")

    GCMC_DIR.mkdir(parents=True, exist_ok=True)
    total_jobs = 0
    job_list = []

    for r in successful:
        name = r["name"]
        if name not in structure_info:
            continue

        struct_dir = GCMC_DIR / name
        struct_dir.mkdir(parents=True, exist_ok=True)

        # 使用优化后的 data 文件
        relaxed_data = LAMMPS_DIR / name / f"{name}_relaxed.data"
        if not relaxed_data.exists():
            print(f"  [跳过] {name}: 未找到优化后 data 文件")
            continue

        # 从 data 文件解析实际 type 信息
        fw_type_info = parse_data_file_types(relaxed_data)
        if fw_type_info["n_atom_types"] == 0:
            print(f"  [跳过] {name}: 无法解析 data 文件类型信息")
            continue

        print(f"  {name}: {fw_type_info['n_atom_types']} atom types, "
              f"{fw_type_info['n_atoms']} atoms, "
              f"{fw_type_info['n_bond_types']} bond types")

        # 复制 relaxed data 到 GCMC 工作目录
        import shutil
        dest_data = struct_dir / relaxed_data.name
        if not dest_data.exists():
            shutil.copy2(relaxed_data, dest_data)

        # 为每种气体生成 molecule template 和输入脚本
        for gas, pressures in [("CO2", PRESSURES_CO2), ("N2", PRESSURES_N2)]:
            gas_dir = struct_dir / gas
            gas_dir.mkdir(parents=True, exist_ok=True)

            # 复制 data 文件到 gas_dir
            gas_data = gas_dir / relaxed_data.name
            if not gas_data.exists():
                shutil.copy2(relaxed_data, gas_data)

            # 创建分子模板
            mol_template = create_molecule_template(gas, gas_dir)

            for P in pressures:
                job_name = f"{name}_{gas}_{P}bar"
                input_script = generate_gcmc_input(
                    name=name,
                    data_file=gas_data,
                    gas_type=gas,
                    pressure=P,
                    work_dir=gas_dir,
                    fw_type_info=fw_type_info,
                    mol_template=mol_template,
                )

                input_file = gas_dir / f"{job_name}.in"
                with open(input_file, "w") as f:
                    f.write(input_script)

                job_list.append({
                    "name": job_name,
                    "structure": name,
                    "gas": gas,
                    "pressure": P,
                    "input_file": str(input_file),
                    "work_dir": str(gas_dir),
                })
                total_jobs += 1

    # 保存作业列表
    jobs_file = GCMC_DIR / "gcmc_job_list.json"
    with open(jobs_file, "w") as f:
        json.dump(job_list, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"共生成 {total_jobs} 个 GCMC 模拟作业")
    n_per_struct = len(PRESSURES_CO2) + len(PRESSURES_N2)
    print(f"  每个结构: {n_per_struct} 个作业 "
          f"(CO2 x{len(PRESSURES_CO2)} + N2 x{len(PRESSURES_N2)})")
    print(f"作业列表保存至: {jobs_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

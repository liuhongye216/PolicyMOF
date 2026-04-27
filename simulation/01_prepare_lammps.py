"""
Step 1: CIF → LAMMPS 输入文件转换 (UFF4MOF 全原子力场)
功能：
  - 读取 data/cif_candidates/ 下的所有 CIF 文件
  - 使用 lammps-interface + UFF4MOF 力场生成完整的 LAMMPS 输入
  - 自动生成 bonds, angles, dihedrals, impropers 拓扑信息
  - 自动分配电荷和力场参数
  - 使用 --fix-metal 固定金属配位几何
"""

import os
import glob
import json
import shutil
import subprocess
import time
from pathlib import Path

try:
    from pymatgen.core import Structure
except ImportError:
    raise ImportError("请安装 pymatgen: pip install pymatgen")


# ======================== 配置参数 ========================
BASE_DIR = Path(__file__).resolve().parent.parent
CIF_DIR = BASE_DIR / "data" / "cif_candidates"
LAMMPS_DIR = BASE_DIR / "data" / "lammps_inputs"

# lammps-interface 参数
FORCE_FIELD = "UFF4MOF"         # 力场: UFF4MOF, UFF, Dreiding
FIX_METAL = True                # 固定金属配位几何
MINIMIZE = True                 # 生成几何优化输入脚本
TIMEOUT_PER_CIF = 300           # 单个 CIF 转换超时 (秒)


def convert_single_cif(cif_path: Path, output_dir: Path) -> dict:
    """
    使用 lammps-interface 将单个 CIF 转换为 LAMMPS 输入文件。

    lammps-interface 会自动:
      1. 识别化学键、角度、二面角、非正常角
      2. 分配 UFF4MOF 力场参数
      3. 通过 QEq 计算电荷
      4. 生成 atom_style full 的 data 文件和 in 文件

    返回：
        包含结构信息的字典, 失败返回 None
    """
    name = cif_path.stem
    struct_dir = output_dir / name
    struct_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 先用 pymatgen 读取 CIF 获取晶格信息
        structure = Structure.from_file(str(cif_path))
        if len(structure) == 0:
            print(f"  [跳过] {name}: 空结构")
            return None

        # 收集晶格信息
        lattice_info = {
            "a": structure.lattice.a,
            "b": structure.lattice.b,
            "c": structure.lattice.c,
            "alpha": structure.lattice.alpha,
            "beta": structure.lattice.beta,
            "gamma": structure.lattice.gamma,
        }
        elements = sorted(set(str(s) for s in structure.species))
        n_atoms_orig = len(structure)

        # 构建 lammps-interface 命令
        cmd = ["lammps-interface", "-ff", FORCE_FIELD]
        if FIX_METAL:
            cmd.append("--fix-metal")
        if MINIMIZE:
            cmd.append("--minimize")
        cmd.append(str(cif_path))

        # 执行 lammps-interface (在 struct_dir 中生成文件)
        start_time = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(struct_dir),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PER_CIF,
        )
        elapsed = time.time() - start_time

        # 检查输出
        if proc.returncode != 0:
            print(f"  [失败] {name}: lammps-interface 返回码 {proc.returncode}")
            print(f"         stderr: {proc.stderr[:300]}")
            # 保存错误日志
            err_log = struct_dir / "conversion_error.log"
            with open(err_log, "w") as f:
                f.write(f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n")
            return None

        # 查找生成的文件
        data_files = list(struct_dir.glob("data.*"))
        in_files = list(struct_dir.glob("in.*"))

        if not data_files or not in_files:
            print(f"  [失败] {name}: 未找到生成的 data/in 文件")
            # 保存日志
            err_log = struct_dir / "conversion_error.log"
            with open(err_log, "w") as f:
                f.write(f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n")
            return None

        data_file = data_files[0]
        input_file = in_files[0]

        # 从 data 文件解析拓扑统计
        n_atoms = 0
        n_bonds = 0
        n_angles = 0
        n_dihedrals = 0
        n_impropers = 0
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.endswith("atoms"):
                    parts = line.split()
                    if len(parts) == 2:
                        n_atoms = int(parts[0])
                elif line.endswith("bonds"):
                    parts = line.split()
                    if len(parts) == 2:
                        n_bonds = int(parts[0])
                elif line.endswith("angles"):
                    parts = line.split()
                    if len(parts) == 2:
                        n_angles = int(parts[0])
                elif line.endswith("dihedrals"):
                    parts = line.split()
                    if len(parts) == 2:
                        n_dihedrals = int(parts[0])
                elif line.endswith("impropers"):
                    parts = line.split()
                    if len(parts) == 2:
                        n_impropers = int(parts[0])
                elif line.startswith("Masses") or line.startswith("Bond Coeffs"):
                    break

        # 检查警告
        warnings = []
        combined_output = proc.stderr + "\n" + proc.stdout
        for line in combined_output.split("\n"):
            if "WARNING" in line or "Could not find" in line:
                warnings.append(line.strip())

        # 检查是否做了超胞
        supercell = ""
        for line in proc.stdout.split("\n"):
            if "Re-sizing" in line or "resize" in line.lower():
                supercell = line.strip()

        info = {
            "name": name,
            "cif_path": str(cif_path),
            "data_file": str(data_file),
            "input_file": str(input_file),
            "struct_dir": str(struct_dir),
            "force_field": FORCE_FIELD,
            "n_atoms_cif": n_atoms_orig,
            "n_atoms": n_atoms,
            "n_bonds": n_bonds,
            "n_angles": n_angles,
            "n_dihedrals": n_dihedrals,
            "n_impropers": n_impropers,
            "elements": elements,
            "lattice": lattice_info,
            "conversion_time": round(elapsed, 2),
            "supercell": supercell,
            "warnings": warnings[:5],  # 最多保留 5 条警告
        }

        print(f"  [成功] {name}: {n_atoms} atoms, {n_bonds} bonds, "
              f"{n_angles} angles, {n_dihedrals} dihedrals  ({elapsed:.1f}s)")
        if supercell:
            print(f"         {supercell}")
        return info

    except subprocess.TimeoutExpired:
        print(f"  [超时] {name}: 转换超过 {TIMEOUT_PER_CIF}s")
        return None
    except Exception as e:
        print(f"  [失败] {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("Step 1: CIF → LAMMPS 输入文件转换 (UFF4MOF)")
    print("=" * 60)

    # 检查 lammps-interface 是否安装
    try:
        proc = subprocess.run(["lammps-interface", "--help"],
                              capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        print("[错误] 未找到 lammps-interface，请安装: pip install lammps-interface")
        return

    # 检查 CIF 目录
    cif_files = sorted(glob.glob(str(CIF_DIR / "*.cif")))
    if not cif_files:
        print(f"\n[错误] 未在 {CIF_DIR} 中找到 CIF 文件。")
        print("请将生成的候选 MOF 的 CIF 文件放入该目录后重试。")
        return

    print(f"\n找到 {len(cif_files)} 个 CIF 文件")
    print(f"力场: {FORCE_FIELD}")
    print(f"Fix metal: {FIX_METAL}")
    print(f"输出目录: {LAMMPS_DIR}\n")

    LAMMPS_DIR.mkdir(parents=True, exist_ok=True)

    # 批量转换
    results = []
    success_count = 0
    fail_count = 0

    for i, cif_path in enumerate(cif_files):
        cif_path = Path(cif_path)
        print(f"\n[{i+1}/{len(cif_files)}] 处理: {cif_path.name}")

        info = convert_single_cif(cif_path, LAMMPS_DIR)
        if info is None:
            fail_count += 1
            continue

        results.append(info)
        success_count += 1

    # 保存结构信息汇总
    summary_file = LAMMPS_DIR / "structure_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"转换完成: 成功 {success_count}, 失败 {fail_count}, "
          f"总计 {success_count + fail_count}")
    if results:
        total_atoms = sum(r["n_atoms"] for r in results)
        total_bonds = sum(r["n_bonds"] for r in results)
        avg_time = sum(r["conversion_time"] for r in results) / len(results)
        print(f"总原子数: {total_atoms}, 总化学键数: {total_bonds}")
        print(f"平均转换时间: {avg_time:.1f}s")
    print(f"结构汇总保存至: {summary_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

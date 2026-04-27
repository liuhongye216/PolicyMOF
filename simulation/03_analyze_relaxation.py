"""
Step 3: 分析 LAMMPS 几何优化结果 (UFF4MOF)
功能：
  - 计算优化成功率 → 对应论文 [percentage]%
  - 计算初始结构与优化后结构的 RMSD → 对应论文 [value] Å
  - 分析键长分布 (利用 UFF4MOF 全拓扑 data 文件)
  - 分析能量收敛 (利用 .min.csv 迭代数据)
  - 分析体积变化分布
  - 分析失败模式
  - 生成论文所需的统计数据和图表
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# ======================== 配置参数 ========================
BASE_DIR = Path(__file__).resolve().parent.parent
LAMMPS_DIR = BASE_DIR / "data" / "lammps_inputs"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "results" / "figures"


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """计算两组坐标之间的 RMSD (Å)。"""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def parse_lammps_dump(dump_file: Path) -> np.ndarray:
    """
    解析 LAMMPS dump 文件，提取原子坐标。
    返回 shape=(N, 3) 的坐标数组 (按 atom id 排序)。
    """
    atoms = []
    reading_atoms = False
    with open(dump_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ITEM: ATOMS"):
                # 解析 header 得到列名
                cols = line.split()[2:]  # ['id', 'type', 'x', 'y', 'z']
                reading_atoms = True
                continue
            if reading_atoms:
                if line.startswith("ITEM:"):
                    break
                parts = line.split()
                if len(parts) >= 5:
                    atom_id = int(parts[cols.index("id")])
                    x = float(parts[cols.index("x")])
                    y = float(parts[cols.index("y")])
                    z = float(parts[cols.index("z")])
                    atoms.append((atom_id, x, y, z))

    if not atoms:
        return None

    # 按 atom id 排序，确保初始和优化后坐标对应
    atoms.sort(key=lambda a: a[0])
    coords = np.array([[a[1], a[2], a[3]] for a in atoms])
    return coords


def parse_lammps_data_bonds(data_file: Path) -> list:
    """
    从 LAMMPS data 文件 (atom_style full) 中提取键长信息。
    返回 [(atom_i, atom_j, bond_type), ...] 列表
    """
    bonds = []
    reading_bonds = False
    reading_atoms = False
    atom_coords = {}

    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()

            # 读取原子坐标
            if line == "Atoms":
                reading_atoms = True
                reading_bonds = False
                continue
            if line == "Bonds":
                reading_bonds = True
                reading_atoms = False
                continue
            if line in ("Angles", "Dihedrals", "Impropers",
                        "Velocities", "Bond Coeffs", "Angle Coeffs"):
                reading_atoms = False
                reading_bonds = False
                continue

            if reading_atoms and line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 7:
                    # atom_style full: id mol-id type charge x y z
                    try:
                        aid = int(parts[0])
                        x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                        atom_coords[aid] = np.array([x, y, z])
                    except (ValueError, IndexError):
                        pass

            if reading_bonds and line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        btype = int(parts[1])
                        ai, aj = int(parts[2]), int(parts[3])
                        bonds.append((ai, aj, btype))
                    except (ValueError, IndexError):
                        pass

    # 计算键长
    bond_lengths = []
    for ai, aj, btype in bonds:
        if ai in atom_coords and aj in atom_coords:
            dist = np.linalg.norm(atom_coords[ai] - atom_coords[aj])
            bond_lengths.append(dist)

    return bond_lengths


def parse_min_csv(csv_file: Path) -> dict:
    """
    解析 .min.csv 文件，提取能量收敛曲线。
    列: MinStep,CellMinStep,AtomMinStep,FinalStep,Energy,EDiff
    """
    try:
        energies = []
        ediffs = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    energies.append(float(row.get("Energy", 0)))
                    ediffs.append(float(row.get("EDiff", 0)))
                except (ValueError, TypeError):
                    pass
        return {
            "energies": energies,
            "ediffs": ediffs,
            "n_iterations": len(energies),
        }
    except Exception:
        return {}


def main():
    print("=" * 60)
    print("Step 3: 分析 LAMMPS 几何优化结果 (UFF4MOF)")
    print("=" * 60)

    # 读取优化结果
    results_file = RESULTS_DIR / "lammps_optimization_results.json"
    if not results_file.exists():
        print(f"[错误] 未找到结果文件: {results_file}")
        print("请先运行 02_run_lammps.py")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    # 读取结构汇总 (含 data 文件路径)
    summary_file = LAMMPS_DIR / "structure_summary.json"
    struct_info = {}
    if summary_file.exists():
        with open(summary_file, "r") as f:
            for info in json.load(f):
                struct_info[info["name"]] = info

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    total = len(results)
    print(f"\n共 {total} 个结构的优化结果\n")

    # ==================== 1. 成功率统计 ====================
    status_counts = Counter(r["status"] for r in results)
    success_count = status_counts.get("success", 0)
    success_rate = success_count / total * 100 if total > 0 else 0

    print("--- 优化成功率 (对应论文 [percentage]) ---")
    print(f"  成功: {success_count}/{total} = {success_rate:.1f}%")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} ({count/total*100:.1f}%)")

    # ==================== 2. RMSD 计算 ====================
    print("\n--- RMSD 计算 (对应论文 [value] Å) ---")
    rmsd_values = []
    rmsd_per_structure = []

    for r in results:
        if r["status"] != "success":
            continue

        name = r["name"]
        struct_dir = LAMMPS_DIR / name
        initial_dump = struct_dir / f"{name}_initial.dump"
        relaxed_dump = struct_dir / f"{name}_relaxed.dump"

        if not initial_dump.exists() or not relaxed_dump.exists():
            continue

        coords_init = parse_lammps_dump(initial_dump)
        coords_relax = parse_lammps_dump(relaxed_dump)

        if coords_init is not None and coords_relax is not None:
            if coords_init.shape == coords_relax.shape:
                rmsd = compute_rmsd(coords_init, coords_relax)
                rmsd_values.append(rmsd)
                rmsd_per_structure.append({"name": name, "rmsd": rmsd})

    if rmsd_values:
        rmsd_array = np.array(rmsd_values)
        print(f"  计算了 {len(rmsd_values)} 个结构的 RMSD")
        print(f"  平均 RMSD: {np.mean(rmsd_array):.3f} Å")
        print(f"  中位数 RMSD: {np.median(rmsd_array):.3f} Å")
        print(f"  标准差: {np.std(rmsd_array):.3f} Å")
        print(f"  范围: [{np.min(rmsd_array):.3f}, {np.max(rmsd_array):.3f}] Å")
    else:
        rmsd_array = None
        print("  [警告] 无法计算 RMSD (未找到 dump 文件)")
        print("  提示: 请确保 02_run_lammps.py 已注入 dump 命令")

    # ==================== 3. 体积变化分布 ====================
    print("\n--- 体积变化统计 ---")
    vol_changes = []
    for r in results:
        if r.get("volume_change_pct") is not None:
            vol_changes.append(r["volume_change_pct"])

    if vol_changes:
        vol_array = np.array(vol_changes)
        print(f"  统计结构数: {len(vol_changes)}")
        print(f"  平均体积变化: {np.mean(vol_array):.2f}%")
        print(f"  中位数: {np.median(vol_array):.2f}%")
        print(f"  最大体积缩小: {np.max(vol_array):.2f}%")
        print(f"  最大体积膨胀: {np.min(vol_array):.2f}%")

    # ==================== 4. 能量统计 ====================
    print("\n--- 能量统计 ---")
    successful = [r for r in results if r["status"] == "success"]
    energies = [r["final_energy"] for r in successful if r.get("final_energy") is not None]
    if energies:
        e_array = np.array(energies)
        print(f"  平均能量: {np.mean(e_array):.2f} kcal/mol")
        print(f"  范围: [{np.min(e_array):.2f}, {np.max(e_array):.2f}] kcal/mol")

        # 能量每原子
        e_per_atom = []
        for r in successful:
            if r.get("final_energy") and r.get("n_atoms", 0) > 0:
                e_per_atom.append(r["final_energy"] / r["n_atoms"])
        if e_per_atom:
            epa = np.array(e_per_atom)
            print(f"  每原子平均能量: {np.mean(epa):.3f} kcal/mol/atom")

    # ==================== 5. 键长分析 ====================
    print("\n--- 键长分析 (从 UFF4MOF data 文件) ---")
    all_bond_lengths = []
    relaxed_bond_lengths = []

    for r in successful[:50]:  # 最多分析 50 个结构
        name = r["name"]
        struct_dir = LAMMPS_DIR / name

        # 优化后的 data 文件
        relaxed_data = struct_dir / f"{name}_relaxed.data"
        if relaxed_data.exists():
            bl = parse_lammps_data_bonds(relaxed_data)
            relaxed_bond_lengths.extend(bl)

        # 初始 data 文件
        if name in struct_info:
            orig_data = Path(struct_info[name].get("data_file", ""))
            if orig_data.exists():
                bl = parse_lammps_data_bonds(orig_data)
                all_bond_lengths.extend(bl)

    if relaxed_bond_lengths:
        bl_array = np.array(relaxed_bond_lengths)
        print(f"  优化后键数: {len(relaxed_bond_lengths)}")
        print(f"  平均键长: {np.mean(bl_array):.3f} Å")
        print(f"  范围: [{np.min(bl_array):.3f}, {np.max(bl_array):.3f}] Å")
    else:
        print("  [警告] 未找到优化后的 data 文件，跳过键长分析")

    # ==================== 6. 耗时统计 ====================
    print("\n--- 耗时统计 ---")
    times = [r["wall_time"] for r in successful if r.get("wall_time")]
    if times:
        t_array = np.array(times)
        print(f"  平均耗时: {np.mean(t_array):.0f}s")
        print(f"  范围: [{np.min(t_array):.0f}s, {np.max(t_array):.0f}s]")
        print(f"  总耗时: {np.sum(t_array)/3600:.1f}h")

    # ==================== 7. 失败模式分析 ====================
    print("\n--- 失败模式分析 ---")
    failed = [r for r in results if r["status"] != "success"]
    if failed:
        error_types = Counter()
        for r in failed:
            msg = r.get("error_message", "unknown")
            if msg:
                # 简化错误信息
                if "坍缩" in str(msg) or "collapse" in str(msg).lower():
                    error_types["体积坍缩"] += 1
                elif "超时" in str(msg) or "timeout" in str(msg).lower():
                    error_types["超时"] += 1
                elif "ERROR" in str(msg):
                    error_types["LAMMPS 错误"] += 1
                else:
                    error_types["其他"] += 1
            else:
                error_types["未知"] += 1

        for err, count in error_types.most_common():
            print(f"  {err}: {count}")
    else:
        print("  无失败结构")

    # ==================== 8. 生成图表 ====================
    print("\n--- 生成图表 ---")

    # 图 1: RMSD 分布直方图
    if rmsd_values:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(rmsd_values, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("RMSD (Å)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("RMSD Distribution: Initial vs. Relaxed (UFF4MOF)", fontsize=13)
        ax.axvline(np.mean(rmsd_values), color="red", linestyle="--",
                   label=f"Mean = {np.mean(rmsd_values):.3f} Å")
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "rmsd_distribution.pdf", dpi=300)
        fig.savefig(FIGURES_DIR / "rmsd_distribution.png", dpi=300)
        plt.close()
        print(f"  保存: rmsd_distribution.pdf/png")

    # 图 2: 优化状态饼图
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors = ["#55A868", "#DD8452", "#C44E52", "#8172B3", "#937860"]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%",
           colors=colors[:len(labels)], startangle=90)
    ax.set_title("Geometry Optimization Results (UFF4MOF)", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "optimization_status.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "optimization_status.png", dpi=300)
    plt.close()
    print(f"  保存: optimization_status.pdf/png")

    # 图 3: 体积变化分布
    if vol_changes:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vol_changes, bins=30, color="#55A868", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Volume Change (%)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Volume Change During Relaxation (UFF4MOF)", fontsize=13)
        ax.axvline(50, color="red", linestyle="--",
                   label="Collapse Threshold (50%)")
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "volume_change.pdf", dpi=300)
        fig.savefig(FIGURES_DIR / "volume_change.png", dpi=300)
        plt.close()
        print(f"  保存: volume_change.pdf/png")

    # 图 4: 键长分布 (新增)
    if relaxed_bond_lengths:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(relaxed_bond_lengths, bins=50, color="#8172B3",
                edgecolor="white", alpha=0.85, range=(0.5, 4.0))
        ax.set_xlabel("Bond Length (Å)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Bond Length Distribution (Relaxed, UFF4MOF)", fontsize=13)
        ax.axvline(np.mean(relaxed_bond_lengths), color="red", linestyle="--",
                   label=f"Mean = {np.mean(relaxed_bond_lengths):.3f} Å")
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "bond_length_distribution.pdf", dpi=300)
        fig.savefig(FIGURES_DIR / "bond_length_distribution.png", dpi=300)
        plt.close()
        print(f"  保存: bond_length_distribution.pdf/png")

    # 图 5: 能量 per atom 分布 (新增)
    if energies and e_per_atom:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(e_per_atom, bins=30, color="#DD8452",
                edgecolor="white", alpha=0.85)
        ax.set_xlabel("Energy per atom (kcal/mol/atom)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Energy per Atom Distribution (UFF4MOF)", fontsize=13)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "energy_per_atom.pdf", dpi=300)
        fig.savefig(FIGURES_DIR / "energy_per_atom.png", dpi=300)
        plt.close()
        print(f"  保存: energy_per_atom.pdf/png")

    # ==================== 9. 保存论文数据汇总 ====================
    paper_data = {
        "force_field": "UFF4MOF",
        "total_candidates": total,
        "success_count": success_count,
        "success_rate_pct": round(success_rate, 1),
        "status_breakdown": dict(status_counts),
        "rmsd_mean": round(float(np.mean(rmsd_array)), 3) if rmsd_array is not None else None,
        "rmsd_median": round(float(np.median(rmsd_array)), 3) if rmsd_array is not None else None,
        "rmsd_std": round(float(np.std(rmsd_array)), 3) if rmsd_array is not None else None,
        "volume_change_mean_pct": round(float(np.mean(vol_array)), 2) if vol_changes else None,
        "energy_per_atom_mean": round(float(np.mean(epa)), 3) if energies and e_per_atom else None,
        "avg_wall_time_s": round(float(np.mean(t_array)), 1) if times else None,
    }

    # RMSD per structure
    if rmsd_per_structure:
        paper_data["rmsd_per_structure"] = sorted(
            rmsd_per_structure, key=lambda x: x["rmsd"])

    summary_file = RESULTS_DIR / "paper_relaxation_stats.json"
    with open(summary_file, "w") as f:
        json.dump(paper_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("论文所需数据:")
    print(f"  [number]     = {total}")
    print(f"  [percentage] = {success_rate:.1f}%")
    if rmsd_array is not None:
        print(f"  [RMSD value] = {np.mean(rmsd_array):.3f} ± {np.std(rmsd_array):.3f} Å")
    else:
        print(f"  [RMSD value] = N/A")
    print(f"  [force field] = UFF4MOF")
    print(f"\n数据保存至: {summary_file}")
    print(f"图表保存至: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

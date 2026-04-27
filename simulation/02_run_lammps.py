"""
Step 2: 批量执行 LAMMPS 几何优化 (UFF4MOF)
功能：
  - 读取 Step 1 (lammps-interface) 生成的结构汇总文件
  - 串行/并行执行 LAMMPS 几何优化
  - lammps-interface 生成的 in 脚本使用 iterative box/relax + fire 最小化
  - 从 LAMMPS log 和 .min.csv 解析优化结果
  - 监控体积变化 (坍缩 > 50% 判定为失败)
"""

import os
import re
import json
import subprocess
import time
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# ======================== 配置参数 ========================
BASE_DIR = Path(__file__).resolve().parent.parent
LAMMPS_DIR = BASE_DIR / "data" / "lammps_inputs"
RESULTS_DIR = BASE_DIR / "results"

# LAMMPS 可执行文件路径
LAMMPS_CMD = "lmp"  # 或 "lmp_serial", "lmp_mpi", 或完整路径

# 并行设置
MAX_WORKERS = 8         # 最大并行任务数 (UFF4MOF 体系较大，建议串行或少量并行)
TIMEOUT = 7200           # 单个任务超时时间 (秒)，UFF4MOF 全拓扑优化较慢

# 结构失败判定标准
VOLUME_COLLAPSE_THRESHOLD = 0.50  # 体积缩小超过 50% 判定为坍缩



def inject_dump_commands(input_file: Path, name: str):
    """
    在 lammps-interface 生成的 in 脚本中注入 dump 命令，
    用于保存优化前后的原子坐标，供后续 RMSD 分析使用。

    注入位置：
      1. read_data 之后: dump 初始坐标 -> {name}_initial.dump
      2. 脚本末尾: dump 优化后坐标 -> {name}_relaxed.dump + write_data
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    injected_initial = False
    for line in lines:
        new_lines.append(line)
        # 在 read_data 行后注入初始 dump
        if line.strip().startswith("read_data") and not injected_initial:
            new_lines.append("\n# === injected: save initial coords ===\n")
            new_lines.append(f"dump            dump_init all custom 1 {name}_initial.dump id type x y z\n")
            new_lines.append("run             0\n")
            new_lines.append("undump          dump_init\n")
            new_lines.append("# === END injected ===\n\n")
            injected_initial = True

    # 在脚本末尾注入优化后 dump + write_data
    new_lines.append("\n# === injected: save relaxed coords ===\n")
    new_lines.append(f"dump            dump_final all custom 1 {name}_relaxed.dump id type x y z\n")
    new_lines.append("run             0\n")
    new_lines.append("undump          dump_final\n")
    new_lines.append("# === injected: save relaxed data ===\n")
    new_lines.append(f"write_data      {name}_relaxed.data\n")

    with open(input_file, "w") as f:
        f.writelines(new_lines)


def calc_volume(lattice: dict) -> float:
    """根据晶格参数计算体积"""
    a, b, c = lattice["a"], lattice["b"], lattice["c"]
    alpha = np.radians(lattice["alpha"])
    beta = np.radians(lattice["beta"])
    gamma = np.radians(lattice["gamma"])
    return a * b * c * np.sqrt(
        1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )


def parse_lammps_log(log_text: str) -> dict:
    """
    从 LAMMPS stdout 解析优化结果。
    
    lammps-interface 生成的脚本使用 iterative loop:
      - thermo 输出包含 Step, E_pair, E_mol, TotEng, Volume 等
      - 最终通过 print 输出到 .min.csv
      - 正常完成时有 "Total wall time" 行
    """
    parsed = {
        "converged": False,
        "final_energy": None,
        "final_volume": None,
        "min_steps": [],
        "total_wall_time": None,
    }

    # 检查是否正常完成 (LAMMPS 正常退出会有 "Total wall time" 行)
    if "Total wall time" in log_text:
        parsed["converged"] = True
        # 提取 wall time
        match = re.search(r"Total wall time:\s+(\S+)", log_text)
        if match:
            parsed["total_wall_time"] = match.group(1)

    lines = log_text.split("\n")

    in_thermo = False
    last_energy = None
    last_volume = None
    header_cols = []

    for line in lines:
        line = line.strip()

        # thermo header 行 —— 只要以 "Step" 开头且包含任何能量关键字即可
        # lammps-interface 输出的列名: Step Temp E_pair E_mol TotEng Press [Volume]
        line_upper = line.upper()
        if line.startswith("Step") and any(kw in line_upper for kw in 
                ["POTENG", "PE", "TOTENG", "E_PAIR", "E_MOL"]):
            header_cols = line.split()
            in_thermo = True
            continue

        if in_thermo:
            # thermo 结束标记
            if (line.startswith("Loop time") or line.startswith("WARNING") 
                    or line == "" or line.startswith("Current step")):
                in_thermo = False
                continue

            parts = line.split()
            if len(parts) >= len(header_cols):
                try:
                    vals = {}
                    for j in range(len(header_cols)):
                        vals[header_cols[j]] = float(parts[j])

                    # 能量优先级: PotEng > PE > TotEng > (E_pair + E_mol)
                    if "PotEng" in vals:
                        last_energy = vals["PotEng"]
                    elif "PE" in vals:
                        last_energy = vals["PE"]
                    elif "TotEng" in vals:
                        last_energy = vals["TotEng"]
                    elif "E_pair" in vals and "E_mol" in vals:
                        last_energy = vals["E_pair"] + vals["E_mol"]
                    elif "E_pair" in vals:
                        last_energy = vals["E_pair"]

                    # 体积
                    if "Volume" in vals:
                        last_volume = vals["Volume"]

                except (ValueError, IndexError):
                    in_thermo = False

    parsed["final_energy"] = last_energy
    parsed["final_volume"] = last_volume

    return parsed


def parse_min_csv(struct_dir: Path) -> dict:
    """
    解析 lammps-interface 输出的 .min.csv 文件。
    该文件记录每次迭代的收敛信息。
    """
    csv_files = list(struct_dir.glob("*.min.csv"))
    if not csv_files:
        return {}

    try:
        with open(csv_files[0], "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {}

        last_row = rows[-1]
        return {
            "min_iterations": len(rows),
            "min_csv_file": str(csv_files[0]),
            "min_last_row": {k: v for k, v in last_row.items()},
        }
    except Exception:
        return {}


def run_single_lammps(info: dict) -> dict:
    """
    执行单个 LAMMPS 几何优化任务。

    lammps-interface 生成的文件格式:
      - data 文件: data.<structure_name>
      - input 文件: in.<structure_name>

    返回：
        包含优化结果的字典
    """
    name = info["name"]
    struct_dir = Path(info["struct_dir"])
    input_file = Path(info["input_file"])

    result = {
        "name": name,
        "status": "unknown",
        "initial_volume": None,
        "final_volume": None,
        "volume_change_pct": None,
        "final_energy": None,
        "wall_time": None,
        "lammps_wall_time": None,
        "n_atoms": info.get("n_atoms", 0),
        "force_field": info.get("force_field", "UFF4MOF"),
        "error_message": None,
    }

    try:
        # 计算初始体积
        if "lattice" in info:
            initial_vol = calc_volume(info["lattice"])
            # 如果有超胞，初始体积要按 n_atoms/n_atoms_cif 倍率修正
            n_atoms_cif = info.get("n_atoms_cif", 0)
            n_atoms = info.get("n_atoms", 0)
            if n_atoms_cif > 0 and n_atoms > n_atoms_cif:
                supercell_factor = n_atoms / n_atoms_cif
                initial_vol *= supercell_factor
            result["initial_volume"] = float(initial_vol)

        # 注入 dump 命令以保存初始/优化后坐标
        inject_dump_commands(input_file, name)

        # 执行 LAMMPS
        # lammps-interface 生成的 in 文件名格式: in.<name>
        in_filename = input_file.name
        start_time = time.time()
        proc = subprocess.run(
            f"""{LAMMPS_CMD} -in '{in_filename}'""",
            cwd=str(struct_dir),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            shell=True,
        )
        wall_time = time.time() - start_time
        result["wall_time"] = round(wall_time, 2)

        # 保存完整日志
        log_file = struct_dir / f"{name}.log"
        with open(log_file, "w") as f:
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(proc.stderr)

        # 检查退出码
        if proc.returncode != 0:
            result["status"] = "crashed"
            # 尝试提取有用的错误信息
            error_lines = []
            for line in proc.stdout.split("\n"):
                if "ERROR" in line:
                    error_lines.append(line.strip())
            if proc.stderr:
                for line in proc.stderr.split("\n"):
                    if line.strip():
                        error_lines.append(line.strip())
            result["error_message"] = "; ".join(error_lines[:3]) if error_lines else f"退出码 {proc.returncode}"
            return result

        # 解析 LAMMPS log
        parsed = parse_lammps_log(proc.stdout)

        if not parsed["converged"]:
            result["status"] = "not_converged"
            result["error_message"] = "LAMMPS 未正常完成"
            result["final_energy"] = parsed["final_energy"]
            result["final_volume"] = parsed["final_volume"]
            return result

        result["final_energy"] = parsed["final_energy"]
        result["final_volume"] = parsed["final_volume"]
        result["lammps_wall_time"] = parsed["total_wall_time"]

        # 尝试解析 .min.csv
        min_info = parse_min_csv(struct_dir)
        if min_info:
            result["min_iterations"] = min_info.get("min_iterations", 0)

        # 判断体积坍缩
        if result["final_volume"] is not None and result["initial_volume"] is not None:
            vol_change = (result["initial_volume"] - result["final_volume"]) / result["initial_volume"]
            result["volume_change_pct"] = round(vol_change * 100, 2)

            if vol_change > VOLUME_COLLAPSE_THRESHOLD:
                result["status"] = "collapsed"
                result["error_message"] = f"体积坍缩 {vol_change*100:.1f}%"
                return result

        result["status"] = "success"
        return result

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error_message"] = f"超时 ({TIMEOUT}s)"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error_message"] = str(e)
        return result


def main():
    print("=" * 60)
    print("Step 2: 批量执行 LAMMPS 几何优化 (UFF4MOF)")
    print("=" * 60)

    # 读取结构汇总
    summary_file = LAMMPS_DIR / "structure_summary.json"
    if not summary_file.exists():
        print(f"[错误] 未找到结构汇总文件: {summary_file}")
        print("请先运行 01_prepare_lammps.py")
        return

    with open(summary_file, "r") as f:
        structures = json.load(f)

    if not structures:
        print("[错误] 结构汇总为空，没有可优化的结构")
        return

    # 检查 LAMMPS 可执行文件
    try:
        proc = subprocess.run([LAMMPS_CMD, "-h"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        print(f"[错误] 未找到 LAMMPS 可执行文件: {LAMMPS_CMD}")
        return

    print(f"\n共 {len(structures)} 个结构待优化")
    print(f"LAMMPS 命令: {LAMMPS_CMD}")
    print(f"力场: {structures[0].get('force_field', 'UFF4MOF')}")
    print(f"最大并行数: {MAX_WORKERS}")
    print(f"超时设置: {TIMEOUT}s\n")

    # 批量执行
    all_results = []

    if MAX_WORKERS <= 1:
        # 串行执行
        for i, info in enumerate(structures):
            print(f"[{i+1}/{len(structures)}] 优化 {info['name']} "
                  f"({info.get('n_atoms', '?')} atoms)...")
            result = run_single_lammps(info)
            all_results.append(result)

            status_str = result["status"]
            time_str = f"{result['wall_time']:.1f}s" if result["wall_time"] else "N/A"
            energy_str = f"{result['final_energy']:.2f}" if result["final_energy"] else "N/A"
            vol_str = f"{result['volume_change_pct']:.1f}%" if result["volume_change_pct"] is not None else "N/A"

            print(f"  → 状态: {status_str}, 耗时: {time_str}, "
                  f"能量: {energy_str}, 体积变化: {vol_str}")
            if result["error_message"]:
                print(f"  → 错误: {result['error_message'][:200]}")
    else:
        # 并行执行
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_single_lammps, info): info["name"]
                for info in structures
            }
            for i, future in enumerate(as_completed(futures)):
                name = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    print(f"[{i+1}/{len(structures)}] {name}: {result['status']} "
                          f"({result.get('wall_time', 'N/A')}s)")
                except Exception as e:
                    print(f"[{i+1}/{len(structures)}] {name}: 异常 - {e}")
                    all_results.append({
                        "name": name,
                        "status": "error",
                        "error_message": str(e),
                    })

    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "lammps_optimization_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # 统计
    status_counts = {}
    for r in all_results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    successful = [r for r in all_results if r["status"] == "success"]

    print(f"\n{'=' * 60}")
    print("优化结果统计:")
    for status, count in sorted(status_counts.items()):
        pct = count / len(all_results) * 100
        print(f"  {status:15s}: {count:4d} ({pct:.1f}%)")

    if successful:
        energies = [r["final_energy"] for r in successful if r["final_energy"]]
        vol_changes = [r["volume_change_pct"] for r in successful if r["volume_change_pct"] is not None]
        times = [r["wall_time"] for r in successful if r["wall_time"]]

        if energies:
            print(f"\n成功结构统计 (n={len(successful)}):")
            print(f"  能量: min={min(energies):.2f}, max={max(energies):.2f}, "
                  f"mean={np.mean(energies):.2f} kcal/mol")
        if vol_changes:
            print(f"  体积变化: min={min(vol_changes):.1f}%, max={max(vol_changes):.1f}%, "
                  f"mean={np.mean(vol_changes):.1f}%")
        if times:
            print(f"  耗时: min={min(times):.0f}s, max={max(times):.0f}s, "
                  f"mean={np.mean(times):.0f}s, total={sum(times):.0f}s")

    print(f"\n结果保存至: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

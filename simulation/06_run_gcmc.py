"""
Step 6: 批量执行 GCMC 模拟
功能：
  - 读取 Step 5 生成的 GCMC 作业列表
  - 串行或并行执行 LAMMPS GCMC 模拟
  - 解析吸附量结果
  - 保存原始输出数据
"""

import json
import os
import subprocess
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ======================== 配置参数 ========================
BASE_DIR = Path(__file__).resolve().parent.parent
GCMC_DIR = BASE_DIR / "data" / "gcmc_inputs"
RESULTS_DIR = BASE_DIR / "results"

# LAMMPS executable can be overridden with LAMMPS_CMD.
LAMMPS_CMD = os.environ.get("LAMMPS_CMD", "lmp")

# Keep subprocess environment explicit while preserving user-level configuration.
_CLEAN_ENV = {
    "HOME": os.environ.get("HOME", ""),
    "USER": os.environ.get("USER", os.environ.get("USERNAME", "")),
    "PATH": os.environ.get("PATH", ""),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "OMP_NUM_THREADS": "1",
}

# 并行设置 (128 个作业, 8 结构 × 2 气体 × 8 压力点)
MAX_WORKERS = 16
TIMEOUT = 3600  # 每个 GCMC 任务最多 1 小时


def run_single_gcmc(job: dict) -> dict:
    """执行单个 GCMC 模拟并解析结果。"""
    name = job["name"]
    input_file = Path(job["input_file"])
    work_dir = Path(job["work_dir"])

    result = {
        "name": name,
        "structure": job["structure"],
        "gas": job["gas"],
        "pressure": job["pressure"],
        "status": "unknown",
        "loading_molecules": None,
        "wall_time": None,
        "error_message": None,
    }

    try:
        start_time = time.time()
        proc = subprocess.run(
            [LAMMPS_CMD, "-in", input_file.name],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=_CLEAN_ENV,
        )
        wall_time = time.time() - start_time
        result["wall_time"] = wall_time

        # 保存日志
        log_file = work_dir / f"{name}.log"
        with open(log_file, "w") as f:
            f.write(proc.stdout)
            if proc.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(proc.stderr)

        if proc.returncode != 0:
            result["status"] = "failed"
            result["error_message"] = proc.stderr[:500] if proc.stderr else "非零退出码"
            return result

        # 解析 GCMC 结果: 从 PRODUCTION_START/END 之间的 thermo 输出提取 v_n_gas 平均值
        lines = proc.stdout.split("\n")
        in_production = False
        n_gas_values = []
        for line in lines:
            if "PRODUCTION_START" in line:
                in_production = True
                continue
            if "PRODUCTION_END" in line:
                in_production = False
                continue
            if in_production:
                parts = line.split()
                # thermo 行格式: Step Atoms Temp Press PE KE TotEng v_n_gas
                # 至少 8 列, 第一列是整数 (Step)
                if len(parts) >= 8:
                    try:
                        int(parts[0])  # 验证是 thermo 数据行
                        n_gas = float(parts[-1])  # v_n_gas 是最后一列
                        n_gas_values.append(n_gas)
                    except (ValueError, IndexError):
                        pass

        if n_gas_values:
            # 分子数 = 原子数 / 每分子原子数 (CO2: 3, N2: 3)
            avg_atoms = sum(n_gas_values) / len(n_gas_values)
            avg_molecules = avg_atoms / 3.0
            result["loading_molecules"] = avg_molecules
            result["status"] = "success"
        else:
            # 回退: 尝试 GCMC_RESULT 行
            for line in lines:
                if line.startswith("GCMC_RESULT:"):
                    parts = line.split()
                    if len(parts) >= 5:
                        result["loading_molecules"] = float(parts[4])
                        result["status"] = "success"
                        break
            if result["status"] != "success":
                result["status"] = "no_result"
                result["error_message"] = "未找到产出阶段 thermo 输出"

        return result

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error_message"] = f"超时 ({TIMEOUT}s)"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error_message"] = str(e)
        return result


def print_summary(all_results):
    """按结构汇总吸附量并打印选择性。"""
    summary = defaultdict(dict)
    for r in all_results:
        if r.get("status") == "success" and r.get("loading_molecules") is not None:
            key = f"{r['gas']}_{r['pressure']}bar"
            summary[r["structure"]][key] = r["loading_molecules"]

    if not summary:
        return

    sep = "=" * 80
    print(f"\n{sep}")
    print("按结构汇总吸附量 (molecules/unit cell):")
    print(sep)
    all_keys = sorted({k for v in summary.values() for k in v})
    header = f"{'Structure':<30}" + "".join(f"{k:>18}" for k in all_keys)
    print(header)
    print("-" * len(header))
    for struct in sorted(summary):
        row = f"{struct:<30}"
        for k in all_keys:
            val = summary[struct].get(k)
            row += f"{val:>18.2f}" if val is not None else f"{'N/A':>18}"
        print(row)

    # CO2/N2 选择性
    print(f"\n{sep}")
    print("CO2/N2 吸附选择性 (基于单组分吸附量比):")
    print(sep)
    for struct in sorted(summary):
        co2_1bar = summary[struct].get("CO2_1.0bar")
        n2_1bar = summary[struct].get("N2_1.0bar")
        if co2_1bar is not None and n2_1bar is not None and n2_1bar > 0:
            selectivity = co2_1bar / n2_1bar
            print(f"  {struct:<30} S(CO2/N2) @ 1bar = {selectivity:.1f}")
        else:
            print(f"  {struct:<30} S(CO2/N2) @ 1bar = N/A")
    print(sep)


def main():
    print("=" * 60)
    print("Step 6: 批量执行 GCMC 模拟")
    print("=" * 60)

    # 读取作业列表
    jobs_file = GCMC_DIR / "gcmc_job_list.json"
    if not jobs_file.exists():
        print("[错误] 未找到作业列表文件，请先运行 05_prepare_gcmc.py")
        return

    with open(jobs_file, "r") as f:
        jobs = json.load(f)

    print(f"\n共 {len(jobs)} 个 GCMC 作业")
    print(f"LAMMPS 命令: {LAMMPS_CMD}")
    print(f"最大并行数: {MAX_WORKERS}\n")

    all_results = []

    if MAX_WORKERS == 1:
        for i, job in enumerate(jobs):
            print(f"[{i+1}/{len(jobs)}] {job['name']}...")
            result = run_single_gcmc(job)
            all_results.append(result)
            if result["status"] == "success":
                print(f"  Loading = {result['loading_molecules']:.2f} molecules, "
                      f"{result['wall_time']:.0f}s")
            else:
                print(f"  {result['status']}: {result['error_message']}")
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(run_single_gcmc, job): job["name"]
                for job in jobs
            }
            for i, future in enumerate(as_completed(futures)):
                name = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    status_str = (f"Loading={result['loading_molecules']:.2f}"
                                  if result['status'] == 'success'
                                  else result['status'])
                    print(f"[{i+1}/{len(jobs)}] {name}: {status_str}")
                except Exception as e:
                    print(f"[{i+1}/{len(jobs)}] {name}: 异常 - {e}")
                    all_results.append({
                        "name": name, "status": "error",
                        "error_message": str(e)
                    })

    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "gcmc_raw_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # 统计
    success_count = sum(1 for r in all_results if r["status"] == "success")
    print(f"\n{'=' * 60}")
    print(f"GCMC 模拟完成: {success_count}/{len(all_results)} 成功")
    print(f"结果保存至: {output_file}")
    print(f"{'=' * 60}")

    # 按结构汇总
    print_summary(all_results)


if __name__ == "__main__":
    main()

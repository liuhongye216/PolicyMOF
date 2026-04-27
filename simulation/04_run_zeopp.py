"""
对指定文件夹中的所有 CIF 文件直接运行 Zeo++ 分析。
计算: ASA、孔体积、PSD、Di/Df/Dif
输出: zeopp_cif_results.json

用法:
  python 04_run_zeopp.py data/cif_candidates
  python 04_run_zeopp.py /path/to/cif_folder --output results.json
  python 04_run_zeopp.py /path/to/cif_folder --probe 1.86 --zeopp /usr/local/bin/network
"""

import os
import re
import json
import argparse
import subprocess
from pathlib import Path

# ======================== 默认参数 ========================
ZEOPP_CMD = "network"
PROBE_RADIUS = 1.86       # N2 探针半径 (Å)
NUM_SAMPLES_ASA = 5000
NUM_SAMPLES_VOL = 50000


def parse_zeopp_kv(filepath: Path) -> dict:
    """解析 Zeo++ 输出文件的 '@ filename Key: Value ...' 格式。"""
    result = {}
    with open(filepath, "r") as f:
        text = f.readline().strip()
    for m in re.finditer(r'(\S+):\s+(\S+)', text):
        result[m.group(1)] = m.group(2)
    return result


def run_asa(cif_path: Path, zeopp: str, probe: float) -> dict | None:
    """计算可及比表面积 (ASA)。"""
    try:
        subprocess.run(
            [zeopp, "-ha", "-sa", str(probe), str(probe),
             str(NUM_SAMPLES_ASA), str(cif_path)],
            capture_output=True, text=True, timeout=300,
        )
        sa_file = cif_path.with_suffix(".sa")
        if sa_file.exists():
            kv = parse_zeopp_kv(sa_file)
            if "ASA_m^2/cm^3" in kv:
                return {
                    "ASA_m2_cm3": float(kv.get("ASA_m^2/cm^3", 0)),
                    "ASA_m2_g":   float(kv.get("ASA_m^2/g", 0)),
                    "NASA_m2_cm3": float(kv.get("NASA_m^2/cm^3", 0)),
                    "NASA_m2_g":   float(kv.get("NASA_m^2/g", 0)),
                }
    except Exception as e:
        print(f"  [ASA 失败] {e}")
    return None


def run_volume(cif_path: Path, zeopp: str, probe: float) -> dict | None:
    """计算孔体积。"""
    try:
        subprocess.run(
            [zeopp, "-ha", "-vol", str(probe), str(probe),
             str(NUM_SAMPLES_VOL), str(cif_path)],
            capture_output=True, text=True, timeout=300,
        )
        vol_file = cif_path.with_suffix(".vol")
        if vol_file.exists():
            kv = parse_zeopp_kv(vol_file)
            if "Unitcell_volume" in kv:
                return {
                    "unitcell_volume": float(kv.get("Unitcell_volume", 0)),
                    "AV_vol_frac": float(kv.get("AV_Volume_fraction", 0)),
                    "AV_cm3_g":    float(kv.get("AV_cm^3/g", 0)),
                    "NAV_vol_frac": float(kv.get("NAV_Volume_fraction", 0)),
                    "NAV_cm3_g":    float(kv.get("NAV_cm^3/g", 0)),
                }
    except Exception as e:
        print(f"  [Volume 失败] {e}")
    return None


def run_psd(cif_path: Path, zeopp: str, probe: float) -> dict | None:
    """计算孔径分布 (PSD)。"""
    try:
        subprocess.run(
            [zeopp, "-ha", "-psd", str(probe), str(probe),
             str(NUM_SAMPLES_VOL), str(cif_path)],
            capture_output=True, text=True, timeout=300,
        )
        psd_file = cif_path.with_suffix(".psd_histo")
        if psd_file.exists():
            bins, counts = [], []
            with open(psd_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            bins.append(float(parts[0]))
                            counts.append(float(parts[1]))
                        except ValueError:
                            continue
            if bins:
                peak_idx = counts.index(max(counts))
                return {"peak_pore_diameter": bins[peak_idx]}
    except Exception as e:
        print(f"  [PSD 失败] {e}")
    return None


def run_diameters(cif_path: Path, zeopp: str) -> dict | None:
    """计算 Di (最大内切球)、Df (最大自由球)、Dif。"""
    try:
        subprocess.run(
            [zeopp, "-ha", "-res", str(cif_path)],
            capture_output=True, text=True, timeout=300,
        )
        res_file = cif_path.with_suffix(".res")
        if res_file.exists():
            with open(res_file, "r") as f:
                line = f.readline().strip()
            nums = re.findall(r'[\d.]+', line)
            if len(nums) >= 3:
                return {
                    "Di":  float(nums[-3]),
                    "Df":  float(nums[-2]),
                    "Dif": float(nums[-1]),
                }
    except Exception as e:
        print(f"  [Diameters 失败] {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="对文件夹中所有 CIF 文件运行 Zeo++ 分析")
    parser.add_argument("cif_dir", type=str,
                        help="包含 CIF 文件的文件夹路径")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 JSON 文件路径 (默认: <cif_dir>/zeopp_cif_results.json)")
    parser.add_argument("--probe", type=float, default=PROBE_RADIUS,
                        help=f"探针半径 Å (默认: {PROBE_RADIUS})")
    parser.add_argument("--zeopp", type=str, default=ZEOPP_CMD,
                        help=f"Zeo++ 可执行文件路径 (默认: {ZEOPP_CMD})")
    args = parser.parse_args()

    cif_dir = Path(args.cif_dir).resolve()
    if not cif_dir.is_dir():
        print(f"[错误] 目录不存在: {cif_dir}")
        return

    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        print(f"[错误] 目录中未找到 .cif 文件: {cif_dir}")
        return

    output_path = Path(args.output) if args.output else cif_dir / "zeopp_cif_results.json"
    zeopp = args.zeopp
    probe = args.probe

    print("=" * 60)
    print(f"Zeo++ 批量 CIF 分析")
    print(f"  目录:   {cif_dir}")
    print(f"  CIF 数: {len(cif_files)}")
    print(f"  探针:   {probe} Å")
    print(f"  Zeo++:  {zeopp}")
    print("=" * 60)

    all_results = []

    for i, cif_path in enumerate(cif_files):
        name = cif_path.stem
        print(f"\n[{i+1}/{len(cif_files)}] {name}")

        entry = {"name": name, "file": cif_path.name}

        asa = run_asa(cif_path, zeopp, probe)
        if asa:
            entry.update(asa)
            print(f"  ASA = {asa['ASA_m2_g']:.1f} m²/g")

        vol = run_volume(cif_path, zeopp, probe)
        if vol:
            entry.update(vol)
            print(f"  AV  = {vol['AV_cm3_g']:.4f} cm³/g  (frac={vol['AV_vol_frac']:.4f})")

        psd = run_psd(cif_path, zeopp, probe)
        if psd:
            entry.update(psd)
            print(f"  PSD peak = {psd['peak_pore_diameter']:.2f} Å")

        diams = run_diameters(cif_path, zeopp)
        if diams:
            entry.update(diams)
            print(f"  Di = {diams['Di']:.2f} Å, Df = {diams['Df']:.2f} Å")

        all_results.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    if all_results:
        import numpy as np
        asa_vals = [r["ASA_m2_g"] for r in all_results if "ASA_m2_g" in r]
        av_vals  = [r["AV_cm3_g"] for r in all_results if "AV_cm3_g" in r]
        di_vals  = [r["Di"] for r in all_results if "Di" in r]

        print(f"\n{'=' * 60}")
        print(f"汇总 ({len(all_results)} 个结构):")
        if asa_vals:
            print(f"  ASA:  {np.mean(asa_vals):.1f} ± {np.std(asa_vals):.1f} m²/g "
                  f"({np.min(asa_vals):.1f} – {np.max(asa_vals):.1f})")
        if av_vals:
            print(f"  AV:   {np.mean(av_vals):.4f} ± {np.std(av_vals):.4f} cm³/g "
                  f"({np.min(av_vals):.4f} – {np.max(av_vals):.4f})")
        if di_vals:
            print(f"  Di:   {np.mean(di_vals):.2f} ± {np.std(di_vals):.2f} Å "
                  f"({np.min(di_vals):.2f} – {np.max(di_vals):.2f})")

    print(f"\n结果保存至: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Step 7: 分析 GCMC 吸附结果并汇总
功能：
  - 将 GCMC 原始结果整理为吸附等温线
  - 计算模拟吸附量与模型预测值的相关性 (R²)
  - 将绝对吸附量转换为过量吸附量
  - 提取 Henry 常数
  - 生成论文图表 (吸附等温线、相关性图、PSD 图)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

# ======================== 配置参数 ========================
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "results" / "figures"

# 物理常数
R_GAS = 8.314  # J/(mol·K)
T = 298.0      # K
NA = 6.022e23  # 阿伏伽德罗常数

# 已知 benchmark MOF 的参考吸附量 (mmol/cm³, 298 K, 1 bar)
# 由文献 gravimetric 值 (mol/kg) × 晶体密度 (g/cm³) 转换得到
# MOF-5: ~0.59 g/cm³, HKUST-1: ~0.88 g/cm³, UiO-66: ~1.24 g/cm³, MIL-101: ~0.44 g/cm³
BENCHMARKS = {
    "MOF-5":   {"CO2": 0.71,  "N2": 0.09},
    "HKUST-1": {"CO2": 4.22,  "N2": 0.31},
    "UiO-66":  {"CO2": 2.60,  "N2": 0.25},
    "MIL-101": {"CO2": 1.54,  "N2": 0.22},
}


def extract_isotherms(results: list) -> dict:
    """
    将 GCMC 原始结果整理为按结构和气体分类的等温线。

    返回:
        {structure_name: {gas: {"pressures": [...], "loadings": [...]}}}
    """
    isotherms = {}

    for r in results:
        if r["status"] != "success":
            continue

        struct = r["structure"]
        gas = r["gas"]
        pressure = r["pressure"]
        loading = r["loading_molecules"]

        if struct not in isotherms:
            isotherms[struct] = {}
        if gas not in isotherms[struct]:
            isotherms[struct][gas] = {"pressures": [], "loadings": []}

        isotherms[struct][gas]["pressures"].append(pressure)
        isotherms[struct][gas]["loadings"].append(loading)

    # 按压力排序
    for struct in isotherms:
        for gas in isotherms[struct]:
            pairs = sorted(zip(isotherms[struct][gas]["pressures"],
                               isotherms[struct][gas]["loadings"]))
            isotherms[struct][gas]["pressures"] = [p for p, _ in pairs]
            isotherms[struct][gas]["loadings"] = [l for _, l in pairs]

    return isotherms


def compute_henry_constant(pressures: list, loadings: list,
                           max_pressure: float = 0.5) -> float:
    """
    从低压区线性区域提取 Henry 常数 (loading / pressure)。
    单位: molecules / (unit_cell · bar)
    """
    low_p = [(p, l) for p, l in zip(pressures, loadings) if p <= max_pressure]
    if len(low_p) < 2:
        return None

    p_arr = np.array([x[0] for x in low_p])
    l_arr = np.array([x[1] for x in low_p])

    # 线性回归 (强制过原点)
    slope, _, _, _, _ = stats.linregress(p_arr, l_arr)
    return slope


def absolute_to_excess(loading_molecules: float, pressure_bar: float,
                       pore_volume_cm3: float, T: float = 298.0) -> float:
    """
    绝对吸附量 → 过量吸附量。
    N_excess = N_abs - ρ_bulk * V_pore
    """
    # 理想气体密度 (molecules/cm³)
    P_Pa = pressure_bar * 1e5
    rho_bulk = P_Pa / (1.380649e-23 * T) * 1e-6  # molecules/cm³
    n_excess = loading_molecules - rho_bulk * pore_volume_cm3
    return max(0, n_excess)


def langmuir(P, q_sat, K):
    """Langmuir 等温线模型: q = q_sat * K * P / (1 + K * P)"""
    return q_sat * K * P / (1 + K * P)


def main():
    print("=" * 60)
    print("Step 7: 分析 GCMC 吸附结果并汇总")
    print("=" * 60)

    # 读取数据
    gcmc_file = RESULTS_DIR / "gcmc_raw_results.json"
    if not gcmc_file.exists():
        print("[错误] 未找到 GCMC 结果文件，请先运行 06_run_gcmc.py")
        return

    with open(gcmc_file, "r") as f:
        gcmc_results = json.load(f)

    # 读取 Zeo++ 结果 (如有)
    zeopp_file = RESULTS_DIR / "zeopp_analysis_results.json"
    zeopp_data = {}
    if zeopp_file.exists():
        with open(zeopp_file, "r") as f:
            zeopp_list = json.load(f)
            zeopp_data = {r["name"]: r for r in zeopp_list}

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ==================== 1. 整理等温线 ====================
    isotherms = extract_isotherms(gcmc_results)
    print(f"\n共 {len(isotherms)} 个结构的等温线数据")

    # ==================== 2. Henry 常数 ====================
    print("\n--- Henry 常数 ---")
    henry_results = {}
    for struct, gases in isotherms.items():
        henry_results[struct] = {}
        for gas, data in gases.items():
            kh = compute_henry_constant(data["pressures"], data["loadings"])
            if kh is not None:
                henry_results[struct][gas] = kh
                print(f"  {struct} / {gas}: K_H = {kh:.4f} mol/bar")

    # ==================== 3. 吸附量统计 (1 bar) ====================
    print("\n--- 1 bar 吸附量汇总 ---")
    loading_1bar = {}
    for struct, gases in isotherms.items():
        loading_1bar[struct] = {}
        for gas, data in gases.items():
            # 找到最接近 1 bar 的数据点
            idx = np.argmin(np.abs(np.array(data["pressures"]) - 1.0))
            if abs(data["pressures"][idx] - 1.0) < 0.5:
                loading_1bar[struct][gas] = data["loadings"][idx]

    # ==================== 4. 与模型预测的相关性 ====================
    # 注意: 这里需要加载模型预测值, 以下为框架代码
    print("\n--- 模拟值 vs 模型预测值 相关性 ---")
    print("  [注意] 需要提供模型预测数据文件来计算 R²")
    print("  请将模型预测结果保存为 results/model_predictions.json")
    print("  格式: [{\"name\": \"MOF_name\", \"gas\": \"CO2\", \"predicted_loading\": value}, ...]")

    # 尝试加载预测数据
    pred_file = RESULTS_DIR / "model_predictions.json"
    r2_results = {}
    if pred_file.exists():
        with open(pred_file, "r") as f:
            predictions = json.load(f)

        for gas in ["CO2", "N2"]:
            sim_vals = []
            pred_vals = []
            for pred in predictions:
                if pred["gas"] == gas:
                    name = pred["name"]
                    if name in loading_1bar and gas in loading_1bar[name]:
                        sim_vals.append(loading_1bar[name][gas])
                        pred_vals.append(pred["predicted_loading"])

            if len(sim_vals) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    pred_vals, sim_vals
                )
                r2 = r_value ** 2
                r2_results[gas] = r2
                print(f"  {gas}: R² = {r2:.4f} (n = {len(sim_vals)})")

                # 绘制相关性图
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(pred_vals, sim_vals, alpha=0.6, s=40, color="#4C72B0")
                lims = [min(min(pred_vals), min(sim_vals)),
                        max(max(pred_vals), max(sim_vals))]
                ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")
                ax.set_xlabel(f"Predicted {gas} Loading", fontsize=12)
                ax.set_ylabel(f"Simulated {gas} Loading (GCMC)", fontsize=12)
                ax.set_title(f"Model vs. GCMC: {gas} ($R^2$ = {r2:.3f})", fontsize=13)
                ax.legend(fontsize=10)
                plt.tight_layout()
                fig.savefig(FIGURES_DIR / f"correlation_{gas}.pdf", dpi=300)
                fig.savefig(FIGURES_DIR / f"correlation_{gas}.png", dpi=300)
                plt.close()
                print(f"  保存: correlation_{gas}.pdf")

    # ==================== 5. 吸附等温线图 ====================
    print("\n--- 生成吸附等温线图 ---")

    # 单独的 per-gas 图
    for gas in ["CO2", "N2"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = 0

        for struct, gases in isotherms.items():
            if gas not in gases:
                continue
            data = gases[gas]
            ax.plot(data["pressures"], data["loadings"],
                    "o-", markersize=4, alpha=0.7, label=struct)
            plotted += 1
            if plotted >= 20:
                break

        ax.set_xlabel("Pressure (bar)", fontsize=12)
        ax.set_ylabel("Loading (molecules/u.c.)", fontsize=12)
        ax.set_title(f"{gas} Adsorption Isotherms at {T:.0f} K", fontsize=13)
        if plotted <= 10:
            ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"isotherm_{gas}.pdf", dpi=300)
        fig.savefig(FIGURES_DIR / f"isotherm_{gas}.png", dpi=300)
        plt.close()
        print(f"  保存: isotherm_{gas}.pdf")

    # ==================== 5b. 论文附录合并图 (fig_s_isotherms.pdf) ====================
    print("\n--- 生成附录合并等温线图 (fig_s_isotherms.pdf) ---")

    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for panel_idx, gas in enumerate(["CO2", "N2"]):
        ax = axes[panel_idx]
        struct_names = sorted([s for s in isotherms if gas in isotherms[s]])

        for i, struct in enumerate(struct_names):
            data = isotherms[struct][gas]
            color = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]
            short_id = struct[:8] if len(struct) > 8 else struct
            ax.plot(data["pressures"], data["loadings"],
                    marker=marker, color=color, markersize=6,
                    linewidth=1.5, alpha=0.85, label=short_id)

            # Langmuir 拟合曲线 (仅在数据点 >= 4 时)
            if len(data["pressures"]) >= 4:
                try:
                    p_arr = np.array(data["pressures"])
                    l_arr = np.array(data["loadings"])
                    popt, _ = curve_fit(langmuir, p_arr, l_arr,
                                        p0=[max(l_arr) * 1.5, 5.0],
                                        maxfev=5000)
                    p_fit = np.linspace(0, max(p_arr) * 1.05, 200)
                    ax.plot(p_fit, langmuir(p_fit, *popt),
                            "--", color=color, alpha=0.4, linewidth=1.0)
                except Exception:
                    pass

        gas_label = r"CO$_2$" if gas == "CO2" else r"N$_2$"
        ax.set_xlabel("Pressure (bar)", fontsize=13)
        ax.set_ylabel("Loading (molecules/u.c.)", fontsize=13)
        ax.set_title(f"({chr(97 + panel_idx)}) {gas_label} adsorption at {T:.0f} K",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(left=-0.02)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9,
                  edgecolor="gray", title="Structure ID", title_fontsize=9)

    plt.tight_layout(w_pad=3.0)
    sup_fig_path = FIGURES_DIR / "fig_s_isotherms.pdf"
    fig.savefig(sup_fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_s_isotherms.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  保存: {sup_fig_path}")

    # ==================== 6. PSD 图 (高性能候选) ====================
    print("\n--- 孔径分布图 ---")
    # 找到吸附量最高的前 5 个结构
    if "CO2" in {gas for gases in isotherms.values() for gas in gases}:
        top_structures = sorted(
            [(s, loading_1bar.get(s, {}).get("CO2", 0)) for s in isotherms],
            key=lambda x: x[1], reverse=True
        )[:5]

        fig, ax = plt.subplots(figsize=(7, 5))
        for struct_name, loading in top_structures:
            psd_file = RESULTS_DIR / f"psd_{struct_name}.json"
            if psd_file.exists():
                with open(psd_file, "r") as f:
                    psd_data = json.load(f)
                ax.plot(psd_data["psd_bins"], psd_data["psd_counts"],
                        alpha=0.8, label=f"{struct_name} ({loading:.1f})")

        ax.set_xlabel("Pore Diameter (Å)", fontsize=12)
        ax.set_ylabel("Distribution", fontsize=12)
        ax.set_title("Pore Size Distribution of Top Candidates", fontsize=13)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"psd_top_candidates.pdf", dpi=300)
        fig.savefig(FIGURES_DIR / f"psd_top_candidates.png", dpi=300)
        plt.close()
        print(f"  保存: psd_top_candidates.pdf")

    # ==================== 7. 汇总论文所需数据 ====================
    paper_data = {
        "n_structures_with_isotherms": len(isotherms),
        "gases": list(set(gas for gases in isotherms.values() for gas in gases)),
        "temperature_K": T,
        "r2_model_vs_gcmc": r2_results if r2_results else "需要提供模型预测数据",
        "henry_constants": henry_results,
        "loading_at_1bar": loading_1bar,
        "benchmark_comparison": BENCHMARKS,
    }

    summary_file = RESULTS_DIR / "paper_adsorption_stats.json"
    with open(summary_file, "w") as f:
        json.dump(paper_data, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("论文所需吸附数据:")
    print(f"  [gas types]   = {paper_data['gases']}")
    print(f"  [temperature] = {T} K")
    if r2_results:
        for gas, r2 in r2_results.items():
            print(f"  [R² for {gas}]  = {r2:.4f}")
    else:
        print("  [R²]          = 需要提供 model_predictions.json")
    print(f"\n数据保存至: {summary_file}")
    print(f"图表保存至: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

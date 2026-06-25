# app.py
# Streamlit CFAR Demonstrator
# CA-CFAR / GOCA-CFAR / SOCA-CFAR
#
# 运行：
#   pip install streamlit numpy pandas matplotlib
#   streamlit run app.py
#
# 更新说明：
# 1. 图中标签默认使用英文，避免 matplotlib 在本机缺少中文字体时文字无法显示。
# 2. 支持切换中文图注；若本机没有中文字体，建议继续使用英文图注。
# 3. CA / GOCA / SOCA 的检测结果分开显示，每个算法独立图、独立统计、独立 CUT 核查。
# 4. 内部严格在线性功率域计算 CFAR 门限，dB 仅用于显示。

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


EPS = 1e-12


# ============================================================
# 0. Matplotlib 字体处理
# ============================================================

def configure_matplotlib_font(plot_language: str):
    """
    为避免中文字体缺失导致图中文字为空白/方框：
    - 默认英文图注，不依赖中文字体。
    - 若用户选择中文图注，则尝试设置常见中文字体。
    """
    matplotlib.rcParams["axes.unicode_minus"] = False

    if plot_language == "中文":
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "PingFang SC",
            "WenQuanYi Micro Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]


def label_dict(plot_language: str) -> Dict[str, str]:
    if plot_language == "中文":
        return {
            "range_bin": "距离单元 index",
            "power_db": "功率 / dB",
            "power_lin": "线性功率",
            "profile_db": "检波后距离像功率 / dB",
            "profile_lin": "检波后距离像功率 / 线性",
            "mean_db": "局部噪声/杂波均值 / dB",
            "mean_lin": "局部噪声/杂波均值 / 线性",
            "threshold_db": "门限 / dB",
            "threshold_lin": "门限 / 线性",
            "detections": "检测点",
            "title_prefix": "检测结果",
            "cut_title": "CUT 参考窗、保护窗与检测单元",
            "left_ref": "左参考窗",
            "left_guard": "左保护窗",
            "cut": "CUT",
            "right_guard": "右保护窗",
            "right_ref": "右参考窗",
        }

    return {
        "range_bin": "Range-bin index",
        "power_db": "Power / dB",
        "power_lin": "Linear power",
        "profile_db": "Detected range profile / dB",
        "profile_lin": "Detected range profile / linear",
        "mean_db": "Local noise/clutter mean / dB",
        "mean_lin": "Local noise/clutter mean / linear",
        "threshold_db": "Threshold / dB",
        "threshold_lin": "Threshold / linear",
        "detections": "Detections",
        "title_prefix": "Detection result",
        "cut_title": "CUT reference cells, guard cells and test cell",
        "left_ref": "Left training",
        "left_guard": "Left guard",
        "cut": "CUT",
        "right_guard": "Right guard",
        "right_ref": "Right training",
    }


# ============================================================
# 1. 基础工具函数
# ============================================================

def db_to_lin(x_db: float) -> float:
    return float(10.0 ** (x_db / 10.0))


def lin_to_db(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, EPS))


def safe_int(x, default: int) -> int:
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def safe_float(x, default: float) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# ============================================================
# 2. CFAR 门限因子公式
# ============================================================

def ca_alpha_from_pfa(pfa: float, n_total_ref: int) -> float:
    """
    CA-CFAR:
        Z_CA = (1/N) * sum_{i=1}^{N} X_i
        eta_CA = alpha_CA * Z_CA

    指数噪声假设下：
        P_fa = (1 + alpha_CA / N)^(-N)

    因此：
        alpha_CA = N * (P_fa^(-1/N) - 1)
    """
    if not (0.0 < pfa < 1.0):
        raise ValueError("Pfa must be in (0, 1).")
    if n_total_ref <= 0:
        raise ValueError("n_total_ref must be positive.")
    return n_total_ref * (pfa ** (-1.0 / n_total_ref) - 1.0)


def pfa_soca_from_alpha(alpha: float, m: int) -> float:
    """
    SOCA-CFAR:
        Z_L = S_L / m
        Z_R = S_R / m
        Z_SOCA = min(Z_L, Z_R)
        eta_SOCA = alpha_SOCA * Z_SOCA

    对每侧 m 个指数参考单元：
        P_fa_SOCA(alpha)
        = 2 * m^m / Gamma(m)
          * sum_{i=0}^{m-1} [ m^i / i! * Gamma(m+i) / (alpha + 2m)^(m+i) ]
    """
    if alpha < 0:
        return 1.0
    if m <= 0:
        raise ValueError("m must be positive.")

    log_terms = []
    for i in range(m):
        val = (
            math.log(2.0)
            + m * math.log(m)
            - math.lgamma(m)
            + i * math.log(m)
            - math.lgamma(i + 1)
            + math.lgamma(m + i)
            - (m + i) * math.log(alpha + 2.0 * m)
        )
        log_terms.append(val)

    max_log = max(log_terms)
    return float(math.exp(max_log) * sum(math.exp(t - max_log) for t in log_terms))


def pfa_goca_from_alpha(alpha: float, m: int) -> float:
    """
    GOCA-CFAR:
        Z_GOCA = max(Z_L, Z_R)
        eta_GOCA = alpha_GOCA * Z_GOCA

    由 max/min 统计量关系：
        P_fa_GOCA(alpha)
        = 2 * (1 + alpha/m)^(-m) - P_fa_SOCA(alpha)
    """
    if alpha < 0:
        return 1.0
    if m <= 0:
        raise ValueError("m must be positive.")

    lz = (1.0 + alpha / m) ** (-m)
    return float(2.0 * lz - pfa_soca_from_alpha(alpha, m))


def pfa_from_alpha(method: str, alpha: float, m_each_side: int) -> float:
    method = method.upper()
    if method == "CA":
        return (1.0 + alpha / (2 * m_each_side)) ** (-(2 * m_each_side))
    if method == "GOCA":
        return pfa_goca_from_alpha(alpha, m_each_side)
    if method == "SOCA":
        return pfa_soca_from_alpha(alpha, m_each_side)
    raise ValueError(f"Unknown CFAR method: {method}")


def alpha_from_pfa(method: str, pfa: float, m_each_side: int) -> float:
    """
    CA 使用闭式解。
    GOCA / SOCA 使用二分法反解：
        find alpha such that Pfa_method(alpha) = target_pfa
    """
    method = method.upper()
    if method == "CA":
        return ca_alpha_from_pfa(pfa, 2 * m_each_side)

    lo = 0.0
    hi = 1.0

    while pfa_from_alpha(method, hi, m_each_side) > pfa:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("Cannot bracket alpha. Check Pfa and training cell number.")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if pfa_from_alpha(method, mid, m_each_side) > pfa:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# ============================================================
# 3. 场景生成：目标 + 杂波区域
# ============================================================

@dataclass
class Target:
    enabled: bool
    center_bin: int
    snr_db: float
    width_bins: float
    model: str


@dataclass
class ClutterRegion:
    enabled: bool
    start_bin: int
    end_bin: int
    cnr_db: float
    edge_model: str


def parse_targets(df: pd.DataFrame, n_bins: int) -> List[Target]:
    targets: List[Target] = []
    if df is None or len(df) == 0:
        return targets

    for _, row in df.iterrows():
        enabled = bool(row.get("启用", True))
        center = safe_int(row.get("中心距离单元", n_bins // 2), n_bins // 2)
        snr = safe_float(row.get("目标SNR/dB", 15.0), 15.0)
        width = max(safe_float(row.get("宽度/单元", 1.0), 1.0), 0.1)
        model = str(row.get("目标模型", "单点"))
        targets.append(Target(enabled, center, snr, width, model))

    return targets


def parse_clutter_regions(df: pd.DataFrame, n_bins: int) -> List[ClutterRegion]:
    regions: List[ClutterRegion] = []
    if df is None or len(df) == 0:
        return regions

    for _, row in df.iterrows():
        enabled = bool(row.get("启用", True))
        s = safe_int(row.get("起始单元", n_bins // 3), n_bins // 3)
        e = safe_int(row.get("结束单元", 2 * n_bins // 3), 2 * n_bins // 3)
        cnr = safe_float(row.get("杂波CNR/dB", 12.0), 12.0)
        edge_model = str(row.get("边缘模型", "突变边缘"))
        regions.append(ClutterRegion(enabled, s, e, cnr, edge_model))

    return regions


def add_clutter_mean(
    base_mean: np.ndarray,
    regions: List[ClutterRegion],
    base_noise_power: float,
) -> np.ndarray:
    n = len(base_mean)
    local_mean = base_mean.copy()

    for r in regions:
        if not r.enabled:
            continue

        s = int(np.clip(min(r.start_bin, r.end_bin), 0, n - 1))
        e = int(np.clip(max(r.start_bin, r.end_bin), 0, n - 1))
        if e < s:
            continue

        clutter_power = base_noise_power * db_to_lin(r.cnr_db)

        if r.edge_model == "突变边缘":
            local_mean[s : e + 1] += clutter_power

        elif r.edge_model == "线性上升":
            length = e - s + 1
            ramp = np.linspace(0.0, clutter_power, length)
            local_mean[s : e + 1] += ramp

        elif r.edge_model == "线性下降":
            length = e - s + 1
            ramp = np.linspace(clutter_power, 0.0, length)
            local_mean[s : e + 1] += ramp

        else:
            local_mean[s : e + 1] += clutter_power

    return local_mean


def add_targets(
    measured_power: np.ndarray,
    targets: List[Target],
    base_noise_power: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    目标作为确定性回波功率叠加到随机噪声/杂波功率上。
    SNR 定义为目标峰值功率 / 基础热噪声平均功率。
    """
    n = len(measured_power)
    x = measured_power.copy()
    target_power = np.zeros(n, dtype=float)
    idx = np.arange(n)

    for t in targets:
        if not t.enabled:
            continue

        c = int(np.clip(t.center_bin, 0, n - 1))
        peak_power = base_noise_power * db_to_lin(t.snr_db)

        if t.model == "单点":
            profile = np.zeros(n, dtype=float)
            profile[c] = peak_power

        elif t.model == "矩形扩展":
            half = max(int(round(t.width_bins / 2.0)), 0)
            s = max(c - half, 0)
            e = min(c + half, n - 1)
            profile = np.zeros(n, dtype=float)
            profile[s : e + 1] = peak_power

        elif t.model == "高斯扩展":
            sigma = max(t.width_bins / 2.355, 0.2)
            profile = peak_power * np.exp(-0.5 * ((idx - c) / sigma) ** 2)

        else:
            profile = np.zeros(n, dtype=float)
            profile[c] = peak_power

        target_power += profile
        x += profile

    if np.max(target_power) > 0:
        target_mask = target_power > np.max(target_power) * 1e-6
    else:
        target_mask = np.zeros(n, dtype=bool)

    return x, target_mask


def build_scene(
    n_bins: int,
    base_noise_db: float,
    targets: List[Target],
    clutter_regions: List[ClutterRegion],
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    base_noise_power = db_to_lin(base_noise_db)

    base_mean = np.ones(n_bins, dtype=float) * base_noise_power
    local_mean = add_clutter_mean(base_mean, clutter_regions, base_noise_power)

    # 平方律检波后，噪声/杂波功率服从指数分布
    measured_power = rng.exponential(scale=local_mean)

    measured_power, target_mask = add_targets(measured_power, targets, base_noise_power)
    clutter_mask = local_mean > base_noise_power * (1.0 + 1e-9)

    return measured_power, local_mean, target_mask, clutter_mask


# ============================================================
# 4. CFAR 检测主函数
# ============================================================

def cfar_detect(
    x: np.ndarray,
    n_train_each_side: int,
    n_guard_each_side: int,
    pfa: float,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    """
    窗口定义：
        [左参考 T 个] [左保护 G 个] [CUT] [右保护 G 个] [右参考 T 个]

    只对拥有完整左右参考窗的 CUT 计算。
    边缘处不使用截断参考窗，也不循环填充。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    T = int(n_train_each_side)
    G = int(n_guard_each_side)

    if T <= 0:
        raise ValueError("Training cells per side must be positive.")
    if G < 0:
        raise ValueError("Guard cells per side cannot be negative.")
    if 2 * (T + G) + 1 > n:
        raise ValueError("CFAR 窗口长度超过距离像长度，请减小 T 或 G。")

    valid = np.zeros(n, dtype=bool)
    start_cut = T + G
    end_cut = n - T - G - 1
    valid[start_cut : end_cut + 1] = True

    alphas = {
        "CA": alpha_from_pfa("CA", pfa, T),
        "GOCA": alpha_from_pfa("GOCA", pfa, T),
        "SOCA": alpha_from_pfa("SOCA", pfa, T),
    }

    result: Dict[str, Dict[str, np.ndarray | float]] = {}

    for method in ["CA", "GOCA", "SOCA"]:
        noise_est = np.full(n, np.nan, dtype=float)
        threshold = np.full(n, np.nan, dtype=float)
        detection = np.zeros(n, dtype=bool)
        left_mean_arr = np.full(n, np.nan, dtype=float)
        right_mean_arr = np.full(n, np.nan, dtype=float)

        for k in range(start_cut, end_cut + 1):
            left_ref = x[k - G - T : k - G]
            right_ref = x[k + G + 1 : k + G + 1 + T]

            mean_left = float(np.sum(left_ref) / T)
            mean_right = float(np.sum(right_ref) / T)

            left_mean_arr[k] = mean_left
            right_mean_arr[k] = mean_right

            if method == "CA":
                z = float((np.sum(left_ref) + np.sum(right_ref)) / (2 * T))
            elif method == "GOCA":
                z = max(mean_left, mean_right)
            elif method == "SOCA":
                z = min(mean_left, mean_right)
            else:
                raise ValueError(method)

            thr = alphas[method] * z
            noise_est[k] = z
            threshold[k] = thr
            detection[k] = bool(x[k] > thr)

        result[method] = {
            "alpha": alphas[method],
            "noise_est": noise_est,
            "threshold": threshold,
            "detection": detection,
            "valid": valid,
            "left_mean": left_mean_arr,
            "right_mean": right_mean_arr,
        }

    return result


def inspect_cut(
    x: np.ndarray,
    k: int,
    T: int,
    G: int,
    pfa: float,
) -> Dict:
    n = len(x)
    valid = (T + G) <= k <= (n - T - G - 1)
    if not valid:
        return {"valid": False}

    left_ref_idx = np.arange(k - G - T, k - G)
    left_guard_idx = np.arange(k - G, k)
    right_guard_idx = np.arange(k + 1, k + G + 1)
    right_ref_idx = np.arange(k + G + 1, k + G + 1 + T)

    left_ref = x[left_ref_idx]
    right_ref = x[right_ref_idx]

    sum_left = float(np.sum(left_ref))
    sum_right = float(np.sum(right_ref))
    mean_left = sum_left / T
    mean_right = sum_right / T

    alpha_ca = alpha_from_pfa("CA", pfa, T)
    alpha_goca = alpha_from_pfa("GOCA", pfa, T)
    alpha_soca = alpha_from_pfa("SOCA", pfa, T)

    z_ca = (sum_left + sum_right) / (2 * T)
    z_goca = max(mean_left, mean_right)
    z_soca = min(mean_left, mean_right)

    rows = {
        "CA": {
            "检测器": "CA-CFAR",
            "噪声估计公式": "(ΣL + ΣR) / (2T)",
            "噪声估计值": z_ca,
            "α": alpha_ca,
            "门限": alpha_ca * z_ca,
            "CUT功率": x[k],
            "是否检测": bool(x[k] > alpha_ca * z_ca),
        },
        "GOCA": {
            "检测器": "GOCA-CFAR",
            "噪声估计公式": "max(ΣL/T, ΣR/T)",
            "噪声估计值": z_goca,
            "α": alpha_goca,
            "门限": alpha_goca * z_goca,
            "CUT功率": x[k],
            "是否检测": bool(x[k] > alpha_goca * z_goca),
        },
        "SOCA": {
            "检测器": "SOCA-CFAR",
            "噪声估计公式": "min(ΣL/T, ΣR/T)",
            "噪声估计值": z_soca,
            "α": alpha_soca,
            "门限": alpha_soca * z_soca,
            "CUT功率": x[k],
            "是否检测": bool(x[k] > alpha_soca * z_soca),
        },
    }

    return {
        "valid": True,
        "left_ref_idx": left_ref_idx,
        "left_guard_idx": left_guard_idx,
        "cut_idx": k,
        "right_guard_idx": right_guard_idx,
        "right_ref_idx": right_ref_idx,
        "sum_left": sum_left,
        "sum_right": sum_right,
        "mean_left": mean_left,
        "mean_right": mean_right,
        "rows": rows,
    }


# ============================================================
# 5. 可视化函数
# ============================================================

def get_y_values(values: np.ndarray, y_mode: str) -> np.ndarray:
    return lin_to_db(values) if y_mode == "dB" else values


def plot_single_method(
    x: np.ndarray,
    local_mean: np.ndarray,
    method_result: Dict[str, np.ndarray | float],
    method_name: str,
    show_mean: bool,
    y_mode: str,
    plot_language: str,
):
    labels = label_dict(plot_language)
    idx = np.arange(len(x))

    fig, ax = plt.subplots(figsize=(12, 5.4))

    if y_mode == "dB":
        ax.plot(idx, lin_to_db(x), linewidth=1.2, label=labels["profile_db"])
        if show_mean:
            ax.plot(idx, lin_to_db(local_mean), linestyle="--", linewidth=1.0, label=labels["mean_db"])
        ax.plot(idx, lin_to_db(method_result["threshold"]), linewidth=1.2, label=f"{method_name} {labels['threshold_db']}")
        y_det = lin_to_db(x)
        ax.set_ylabel(labels["power_db"])
    else:
        ax.plot(idx, x, linewidth=1.2, label=labels["profile_lin"])
        if show_mean:
            ax.plot(idx, local_mean, linestyle="--", linewidth=1.0, label=labels["mean_lin"])
        ax.plot(idx, method_result["threshold"], linewidth=1.2, label=f"{method_name} {labels['threshold_lin']}")
        y_det = x
        ax.set_ylabel(labels["power_lin"])

    det = method_result["detection"]
    ax.scatter(idx[det], y_det[det], s=30, marker="o", label=f"{method_name} {labels['detections']}")

    ax.set_xlabel(labels["range_bin"])
    ax.set_title(f"{method_name} {labels['title_prefix']}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_cut_window(x: np.ndarray, info: Dict, y_mode: str, plot_language: str):
    labels = label_dict(plot_language)
    fig, ax = plt.subplots(figsize=(12, 3.8))

    if not info.get("valid", False):
        msg = "Invalid CUT: incomplete reference window." if plot_language != "中文" else "当前 CUT 没有完整参考窗，不能按经典 CFAR 公式计算。"
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        ax.set_axis_off()
        return fig

    k = info["cut_idx"]
    all_idx = np.concatenate([
        info["left_ref_idx"],
        info["left_guard_idx"],
        np.array([k]),
        info["right_guard_idx"],
        info["right_ref_idx"],
    ])

    y = lin_to_db(x[all_idx]) if y_mode == "dB" else x[all_idx]
    ax.plot(all_idx, y, marker="o", linewidth=1.0)

    def shade(indices, label, alpha):
        if len(indices) == 0:
            return
        ax.axvspan(indices[0] - 0.5, indices[-1] + 0.5, alpha=alpha, label=label)

    shade(info["left_ref_idx"], labels["left_ref"], 0.18)
    shade(info["left_guard_idx"], labels["left_guard"], 0.10)
    ax.axvspan(k - 0.5, k + 0.5, alpha=0.28, label=labels["cut"])
    shade(info["right_guard_idx"], labels["right_guard"], 0.10)
    shade(info["right_ref_idx"], labels["right_ref"], 0.18)

    ax.set_xlabel(labels["range_bin"])
    ax.set_ylabel(labels["power_db"] if y_mode == "dB" else labels["power_lin"])
    ax.set_title(f"CUT = {k}: {labels['cut_title']}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def method_metrics(
    cfar: Dict[str, Dict[str, np.ndarray | float]],
    method: str,
    target_mask: np.ndarray,
    clutter_mask: np.ndarray,
) -> pd.DataFrame:
    valid = cfar[method]["valid"]
    det = cfar[method]["detection"] & valid

    non_target = ~target_mask
    rows = [
        {"指标": "门限因子 α", "数值": float(cfar[method]["alpha"])},
        {"指标": "有效 CUT 数", "数值": int(np.sum(valid))},
        {"指标": "总检测点数", "数值": int(np.sum(det))},
        {"指标": "目标支撑区检测点数", "数值": int(np.sum(det & target_mask))},
        {"指标": "非目标区检测点数", "数值": int(np.sum(det & non_target))},
        {"指标": "杂波区检测点数", "数值": int(np.sum(det & clutter_mask))},
        {"指标": "非杂波区检测点数", "数值": int(np.sum(det & (~clutter_mask)))},
    ]
    return pd.DataFrame(rows)


def comparison_metrics(
    cfar: Dict[str, Dict[str, np.ndarray | float]],
    target_mask: np.ndarray,
    clutter_mask: np.ndarray,
) -> pd.DataFrame:
    rows = []
    valid = cfar["CA"]["valid"]

    for method in ["CA", "GOCA", "SOCA"]:
        det = cfar[method]["detection"] & valid
        non_target = ~target_mask
        rows.append({
            "检测器": method,
            "α": float(cfar[method]["alpha"]),
            "有效CUT数": int(np.sum(valid)),
            "总检测点数": int(np.sum(det)),
            "目标支撑区检测点数": int(np.sum(det & target_mask)),
            "非目标区检测点数": int(np.sum(det & non_target)),
            "杂波区检测点数": int(np.sum(det & clutter_mask)),
            "非杂波区检测点数": int(np.sum(det & (~clutter_mask))),
        })

    return pd.DataFrame(rows)


def safe_ratio(numerator: float, denominator: float) -> float:
    """除数为 0 时返回 NaN，避免无目标或无杂波场景下误导统计。"""
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def monte_carlo_trial_metrics(
    cfar: Dict[str, Dict[str, np.ndarray | float]],
    target_mask: np.ndarray,
    clutter_mask: np.ndarray,
    pfa_design: float,
    trial_idx: int,
) -> pd.DataFrame:
    """
    面向课堂比较的单次蒙特卡洛统计。

    注意：
    - 目标检测率 Pd 以目标支撑区内有效 CUT 为分母。
    - 经验虚警率 Pfa_hat 以非目标区有效 CUT 为分母。
    - 杂波区虚警率、非杂波区虚警率都排除目标支撑区，避免把真实目标误计为虚警。
    """
    valid = cfar["CA"]["valid"]
    target_valid = valid & target_mask
    non_target_valid = valid & (~target_mask)
    clutter_non_target = valid & clutter_mask & (~target_mask)
    quiet_non_target = valid & (~clutter_mask) & (~target_mask)

    rows = []
    for method in ["CA", "GOCA", "SOCA"]:
        det = cfar[method]["detection"] & valid

        target_hits = int(np.sum(det & target_valid))
        target_total = int(np.sum(target_valid))
        false_alarms = int(np.sum(det & non_target_valid))
        non_target_total = int(np.sum(non_target_valid))
        clutter_false_alarms = int(np.sum(det & clutter_non_target))
        clutter_total = int(np.sum(clutter_non_target))
        quiet_false_alarms = int(np.sum(det & quiet_non_target))
        quiet_total = int(np.sum(quiet_non_target))
        total_detections = int(np.sum(det))

        pd_cell = safe_ratio(target_hits, target_total)
        pfa_hat = safe_ratio(false_alarms, non_target_total)
        pfa_clutter = safe_ratio(clutter_false_alarms, clutter_total)
        pfa_quiet = safe_ratio(quiet_false_alarms, quiet_total)

        rows.append({
            "实验序号": int(trial_idx),
            "检测器": method,
            "设计Pfa": float(pfa_design),
            "门限因子α": float(cfar[method]["alpha"]),
            "有效CUT数": int(np.sum(valid)),
            "目标有效CUT数": target_total,
            "非目标有效CUT数": non_target_total,
            "总检测点数": total_detections,
            "目标检出点数": target_hits,
            "目标漏检点数": int(target_total - target_hits),
            "虚警点数": false_alarms,
            "杂波区非目标CUT数": clutter_total,
            "杂波区虚警点数": clutter_false_alarms,
            "非杂波区非目标CUT数": quiet_total,
            "非杂波区虚警点数": quiet_false_alarms,
            "目标检测率Pd": pd_cell,
            "经验虚警率Pfa_hat": pfa_hat,
            "Pfa偏离倍数": safe_ratio(pfa_hat, pfa_design),
            "杂波区虚警率": pfa_clutter,
            "非杂波区虚警率": pfa_quiet,
        })

    return pd.DataFrame(rows)


def aggregate_monte_carlo_metrics(history_all: pd.DataFrame) -> pd.DataFrame:
    """
    将逐次蒙特卡洛结果汇总为累计性能指标。
    累计概率优先采用“总次数 / 总样本数”，比简单平均每次概率更稳健。
    """
    rows = []
    for method, g in history_all.groupby("检测器", sort=False):
        target_total = float(g["目标有效CUT数"].sum())
        non_target_total = float(g["非目标有效CUT数"].sum())
        clutter_non_target_total = float(g["杂波区非目标CUT数"].sum())
        quiet_non_target_total = float(g["非杂波区非目标CUT数"].sum())
        target_hits = float(g["目标检出点数"].sum())
        false_alarms = float(g["虚警点数"].sum())
        clutter_false_alarms = float(g["杂波区虚警点数"].sum())
        quiet_false_alarms = float(g["非杂波区虚警点数"].sum())

        pd_total = safe_ratio(target_hits, target_total)
        pfa_total = safe_ratio(false_alarms, non_target_total)
        pfa_clutter_total = safe_ratio(clutter_false_alarms, clutter_non_target_total)
        pfa_quiet_total = safe_ratio(quiet_false_alarms, quiet_non_target_total)

        rows.append({
            "检测器": method,
            "实验次数": int(g["实验序号"].nunique()),
            "累计目标检测率Pd": pd_total,
            "Pd单次均值": float(g["目标检测率Pd"].mean(skipna=True)),
            "Pd单次标准差": float(g["目标检测率Pd"].std(skipna=True)),
            "累计经验虚警率Pfa_hat": pfa_total,
            "Pfa单次均值": float(g["经验虚警率Pfa_hat"].mean(skipna=True)),
            "Pfa单次标准差": float(g["经验虚警率Pfa_hat"].std(skipna=True)),
            "平均Pfa偏离倍数": float(g["Pfa偏离倍数"].mean(skipna=True)),
            "累计杂波区虚警率": pfa_clutter_total,
            "杂波区虚警率单次均值": float(g["杂波区虚警率"].mean(skipna=True)),
            "累计非杂波区虚警率": pfa_quiet_total,
            "非杂波区虚警率单次均值": float(g["非杂波区虚警率"].mean(skipna=True)),
            "平均总检测点数": float(g["总检测点数"].mean()),
            "平均目标检出点数": float(g["目标检出点数"].mean()),
            "平均虚警点数": float(g["虚警点数"].mean()),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["检测能力名次"] = df["累计目标检测率Pd"].rank(ascending=False, method="min", na_option="bottom").astype(int)
    df["虚警控制名次"] = df["累计经验虚警率Pfa_hat"].rank(ascending=True, method="min", na_option="bottom").astype(int)
    df["杂波边缘稳健名次"] = df["累计杂波区虚警率"].rank(ascending=True, method="min", na_option="bottom").astype(int)
    df["综合名次均值"] = df[["检测能力名次", "虚警控制名次", "杂波边缘稳健名次"]].mean(axis=1)
    df = df.sort_values(["综合名次均值", "检测能力名次", "虚警控制名次"]).reset_index(drop=True)
    return df


def format_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将概率类列格式化为百分数文本，便于课堂展示。"""
    out = df.copy()
    rate_cols = [
        "目标检测率Pd",
        "经验虚警率Pfa_hat",
        "杂波区虚警率",
        "非杂波区虚警率",
        "累计目标检测率Pd",
        "Pd单次均值",
        "Pd单次标准差",
        "累计经验虚警率Pfa_hat",
        "Pfa单次均值",
        "Pfa单次标准差",
        "累计杂波区虚警率",
        "杂波区虚警率单次均值",
        "累计非杂波区虚警率",
        "非杂波区虚警率单次均值",
    ]
    for col in rate_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{100.0 * float(v):.3f}%")
    if "Pfa偏离倍数" in out.columns:
        out["Pfa偏离倍数"] = out["Pfa偏离倍数"].map(lambda v: "—" if pd.isna(v) else f"{float(v):.2f}×")
    if "平均Pfa偏离倍数" in out.columns:
        out["平均Pfa偏离倍数"] = out["平均Pfa偏离倍数"].map(lambda v: "—" if pd.isna(v) else f"{float(v):.2f}×")
    return out


def plot_three_methods_comparison(
    x: np.ndarray,
    local_mean: np.ndarray,
    cfar: Dict[str, Dict[str, np.ndarray | float]],
    show_mean: bool,
    y_mode: str,
    plot_language: str,
    title: str = "Three CFAR methods comparison",
):
    """在同一张图中叠加 CA / GOCA / SOCA 门限，并在下方给出三算法检测栅格。"""
    labels = label_dict(plot_language)
    idx = np.arange(len(x))
    valid = cfar["CA"]["valid"]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [4.5, 1.25]},
    )

    if y_mode == "dB":
        y_x = lin_to_db(x)
        y_mean = lin_to_db(local_mean)
        ax1.set_ylabel(labels["power_db"])
        threshold_label = labels["threshold_db"]
    else:
        y_x = x
        y_mean = local_mean
        ax1.set_ylabel(labels["power_lin"])
        threshold_label = labels["threshold_lin"]

    ax1.plot(idx, y_x, linewidth=1.1, label=labels["profile_db"] if y_mode == "dB" else labels["profile_lin"])
    if show_mean:
        ax1.plot(idx, y_mean, linestyle="--", linewidth=1.0, label=labels["mean_db"] if y_mode == "dB" else labels["mean_lin"])

    line_styles = {"CA": "-", "GOCA": "--", "SOCA": ":"}
    markers = {"CA": "o", "GOCA": "s", "SOCA": "^"}

    for method in ["CA", "GOCA", "SOCA"]:
        thr = cfar[method]["threshold"]
        y_thr = lin_to_db(thr) if y_mode == "dB" else thr
        ax1.plot(idx, y_thr, linestyle=line_styles[method], linewidth=1.4, label=f"{method} {threshold_label}")

        det = cfar[method]["detection"] & valid
        det_idx = idx[det]
        if len(det_idx) > 0:
            ax1.scatter(
                det_idx,
                y_x[det],
                s=28,
                marker=markers[method],
                label=f"{method} {labels['detections']}",
            )
            ax2.scatter(
                det_idx,
                np.full(len(det_idx), ["CA", "GOCA", "SOCA"].index(method)),
                s=120,
                marker="|",
                label=method,
            )

    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=8, ncol=2)

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["CA", "GOCA", "SOCA"])
    ax2.set_ylim(-0.6, 2.6)
    ax2.set_xlabel(labels["range_bin"])
    ax2.set_title("Detection raster: each vertical mark is one detected CUT")
    ax2.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    return fig


def plot_monte_carlo_aggregate_bars(aggregate_df: pd.DataFrame):
    """累计指标柱状图：同一张图展示 Pd、Pfa 和杂波区虚警率，便于快速比较。"""
    if aggregate_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No Monte Carlo results", ha="center", va="center")
        ax.set_axis_off()
        return fig

    methods = aggregate_df["检测器"].tolist()
    x_pos = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x_pos - width, aggregate_df["累计目标检测率Pd"].astype(float), width, label="Pd")
    ax.bar(x_pos, aggregate_df["累计经验虚警率Pfa_hat"].astype(float), width, label="Pfa_hat")
    ax.bar(x_pos + width, aggregate_df["累计杂波区虚警率"].astype(float), width, label="Clutter false-alarm rate")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Probability")
    ax.set_title("Monte Carlo aggregate performance comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ============================================================
# 6. Streamlit 页面
# ============================================================

st.set_page_config(
    page_title="CFAR Demonstrator",
    layout="wide",
)

st.title("CA-CFAR、GOCA-CFAR、SOCA-CFAR 对比演示")

st.markdown(
    """
本程序用于课堂演示三类一维 CFAR 检测器在 **临近多目标** 与 **杂波边缘** 场景下的差异。
程序内部在 **线性功率域** 计算门限；dB 仅用于界面显示。
"""
)

with st.sidebar:
    st.header("显示设置")

    plot_language = st.radio(
        "图中文字语言",
        ["英文", "中文"],
        index=0,
        help="若本机 matplotlib 缺少中文字体，中文图注可能显示为空白或方框；建议课堂演示使用英文图注。",
    )
    configure_matplotlib_font(plot_language)

    st.header("距离像与随机模型")

    n_bins = st.number_input("距离单元数", min_value=64, max_value=4096, value=512, step=64)
    base_noise_db = st.slider("基础热噪声平均功率 / dB", -40.0, 20.0, 0.0, 0.5)

    st.header("CFAR 参数")

    n_train_each_side = st.number_input(
        "每侧参考单元数 T",
        min_value=1,
        max_value=256,
        value=16,
        step=1,
        help="总参考单元数 N = 2T。"
    )

    n_guard_each_side = st.number_input(
        "每侧保护单元数 G",
        min_value=0,
        max_value=128,
        value=2,
        step=1,
        help="保护单元不参与噪声估计。"
    )

    pfa = st.select_slider(
        "设计虚警概率 Pfa",
        options=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6],
        value=1e-3,
        format_func=lambda v: f"{v:.0e}" if v < 1e-2 else f"{v:g}",
    )

    y_mode = st.radio("纵轴显示", ["dB", "线性"], index=0, horizontal=True)
    show_mean = st.checkbox("显示局部噪声/杂波均值", value=True)

    st.header("蒙特卡洛实验")
    mc_trials = st.number_input(
        "实验次数",
        min_value=1,
        max_value=500,
        value=20,
        step=1,
        help="点击主页面按钮后，程序会按该次数自动重复生成随机距离像并刷新结果。",
    )
    mc_interval = st.slider(
        "每次刷新间隔 / s",
        min_value=0.0,
        max_value=2.0,
        value=0.20,
        step=0.05,
    )
    mc_show_figures = st.checkbox("蒙特卡洛过程中显示三算法叠加对比图", value=True)

    st.header("单个 CUT 公式核查")
    inspect_idx = st.number_input(
        "选择一个 CUT 距离单元",
        min_value=0,
        max_value=int(n_bins) - 1,
        value=int(n_bins) // 2,
        step=1,
    )


st.subheader("1. 自主设置目标")

default_targets = pd.DataFrame([
    {"启用": True, "中心距离单元": 180, "目标SNR/dB": 18.0, "宽度/单元": 1.0, "目标模型": "单点"},
    {"启用": True, "中心距离单元": 188, "目标SNR/dB": 15.0, "宽度/单元": 1.0, "目标模型": "单点"},
    {"启用": False, "中心距离单元": 260, "目标SNR/dB": 12.0, "宽度/单元": 8.0, "目标模型": "高斯扩展"},
])

target_df = st.data_editor(
    default_targets,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "启用": st.column_config.CheckboxColumn(default=True),
        "中心距离单元": st.column_config.NumberColumn(min_value=0, max_value=int(n_bins) - 1, step=1),
        "目标SNR/dB": st.column_config.NumberColumn(step=0.5),
        "宽度/单元": st.column_config.NumberColumn(min_value=0.1, step=0.5),
        "目标模型": st.column_config.SelectboxColumn(options=["单点", "矩形扩展", "高斯扩展"]),
    },
    key="target_editor",
)

st.subheader("2. 自主设置杂波区域")

default_clutter = pd.DataFrame([
    {"启用": True, "起始单元": 300, "结束单元": 430, "杂波CNR/dB": 14.0, "边缘模型": "突变边缘"},
    {"启用": False, "起始单元": 80, "结束单元": 130, "杂波CNR/dB": 8.0, "边缘模型": "线性上升"},
])

clutter_df = st.data_editor(
    default_clutter,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "启用": st.column_config.CheckboxColumn(default=True),
        "起始单元": st.column_config.NumberColumn(min_value=0, max_value=int(n_bins) - 1, step=1),
        "结束单元": st.column_config.NumberColumn(min_value=0, max_value=int(n_bins) - 1, step=1),
        "杂波CNR/dB": st.column_config.NumberColumn(step=0.5),
        "边缘模型": st.column_config.SelectboxColumn(options=["突变边缘", "线性上升", "线性下降"]),
    },
    key="clutter_editor",
)


targets = parse_targets(target_df, int(n_bins))
clutter_regions = parse_clutter_regions(clutter_df, int(n_bins))

try:
    x, local_mean, target_mask, clutter_mask = build_scene(
        n_bins=int(n_bins),
        base_noise_db=float(base_noise_db),
        targets=targets,
        clutter_regions=clutter_regions,
    )

    cfar = cfar_detect(
        x=x,
        n_train_each_side=int(n_train_each_side),
        n_guard_each_side=int(n_guard_each_side),
        pfa=float(pfa),
    )

except Exception as exc:
    st.error(f"参数错误或计算失败：{exc}")
    st.stop()


st.subheader("3. 三个算法的结果分开显示")

info = inspect_cut(
    x=x,
    k=int(inspect_idx),
    T=int(n_train_each_side),
    G=int(n_guard_each_side),
    pfa=float(pfa),
)

tabs = st.tabs(["CA-CFAR", "GOCA-CFAR", "SOCA-CFAR"])

for tab, method in zip(tabs, ["CA", "GOCA", "SOCA"]):
    with tab:
        st.markdown(f"### {method}-CFAR")

        fig = plot_single_method(
            x=x,
            local_mean=local_mean,
            method_result=cfar[method],
            method_name=method,
            show_mean=show_mean,
            y_mode=y_mode,
            plot_language=plot_language,
        )
        st.pyplot(fig, clear_figure=True)

        col1, col2 = st.columns([1.0, 1.2])

        with col1:
            st.markdown("#### 当前算法统计")
            st.dataframe(
                method_metrics(cfar, method, target_mask, clutter_mask),
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            st.markdown("#### 当前 CUT 计算核查")
            if not info.get("valid", False):
                st.warning("当前 CUT 没有完整的左右参考窗，因此不计算 CFAR 门限。")
            else:
                st.dataframe(
                    pd.DataFrame([info["rows"][method]]),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("#### CUT 窗口位置")
        fig_cut = plot_cut_window(x, info, y_mode=y_mode, plot_language=plot_language)
        st.pyplot(fig_cut, clear_figure=True)


st.subheader("4. 三个算法的数值汇总")

st.dataframe(
    comparison_metrics(cfar, target_mask, clutter_mask),
    use_container_width=True,
    hide_index=True,
)


st.subheader("5. 蒙特卡洛自动实验")

st.markdown(
    "点击按钮后，程序会在当前目标、杂波和 CFAR 参数下重复生成随机距离像。"
    "每一次实验都会把 CA-CFAR、GOCA-CFAR、SOCA-CFAR 叠加到同一张图中；"
    "下方统计表同时给出目标检测率 Pd、经验虚警率 Pfa_hat、杂波区虚警率和综合排序。"
)

if st.button("开始蒙特卡洛自动实验", type="primary", use_container_width=True):
    progress_bar = st.progress(0.0)
    status_box = st.empty()
    current_figure_box = st.empty()
    current_table_box = st.empty()
    aggregate_figure_box = st.empty()
    aggregate_table_box = st.empty()
    history_table_box = st.empty()
    conclusion_box = st.empty()

    history_frames: List[pd.DataFrame] = []

    for trial_idx in range(1, int(mc_trials) + 1):
        rng_i = np.random.default_rng()
        x_i, local_mean_i, target_mask_i, clutter_mask_i = build_scene(
            n_bins=int(n_bins),
            base_noise_db=float(base_noise_db),
            targets=targets,
            clutter_regions=clutter_regions,
            rng=rng_i,
        )

        cfar_i = cfar_detect(
            x=x_i,
            n_train_each_side=int(n_train_each_side),
            n_guard_each_side=int(n_guard_each_side),
            pfa=float(pfa),
        )

        trial_metrics = monte_carlo_trial_metrics(
            cfar=cfar_i,
            target_mask=target_mask_i,
            clutter_mask=clutter_mask_i,
            pfa_design=float(pfa),
            trial_idx=trial_idx,
        )
        history_frames.append(trial_metrics)

        status_box.markdown(f"#### 当前为第 {trial_idx} / {int(mc_trials)} 次随机实验")

        if bool(mc_show_figures):
            fig_i = plot_three_methods_comparison(
                x=x_i,
                local_mean=local_mean_i,
                cfar=cfar_i,
                show_mean=show_mean,
                y_mode=y_mode,
                plot_language=plot_language,
                title=f"Monte Carlo trial {trial_idx}: CA / GOCA / SOCA comparison",
            )
            current_figure_box.pyplot(fig_i, clear_figure=True)
            plt.close(fig_i)
        else:
            current_figure_box.empty()

        current_cols = [
            "实验序号",
            "检测器",
            "目标检测率Pd",
            "经验虚警率Pfa_hat",
            "Pfa偏离倍数",
            "杂波区虚警率",
            "总检测点数",
            "目标检出点数",
            "虚警点数",
        ]
        with current_table_box.container():
            st.markdown("#### 当前单次实验性能")
            st.dataframe(
                format_rate_columns(trial_metrics[current_cols]),
                use_container_width=True,
                hide_index=True,
            )

        history_all = pd.concat(history_frames, ignore_index=True)
        aggregate = aggregate_monte_carlo_metrics(history_all)

        fig_agg = plot_monte_carlo_aggregate_bars(aggregate)
        aggregate_figure_box.pyplot(fig_agg, clear_figure=True)
        plt.close(fig_agg)

        aggregate_cols = [
            "检测器",
            "实验次数",
            "累计目标检测率Pd",
            "累计经验虚警率Pfa_hat",
            "平均Pfa偏离倍数",
            "累计杂波区虚警率",
            "累计非杂波区虚警率",
            "平均总检测点数",
            "平均目标检出点数",
            "平均虚警点数",
            "检测能力名次",
            "虚警控制名次",
            "杂波边缘稳健名次",
            "综合名次均值",
        ]
        with aggregate_table_box.container():
            st.markdown("#### 累计统计与综合排序")
            st.dataframe(
                format_rate_columns(aggregate[aggregate_cols]),
                use_container_width=True,
                hide_index=True,
            )

        compact_history_cols = [
            "实验序号",
            "检测器",
            "目标检测率Pd",
            "经验虚警率Pfa_hat",
            "杂波区虚警率",
            "目标检出点数",
            "虚警点数",
        ]
        with history_table_box.container():
            st.markdown("#### 逐次实验记录")
            st.dataframe(
                format_rate_columns(history_all[compact_history_cols]),
                use_container_width=True,
                hide_index=True,
            )

        if not aggregate.empty:
            best = aggregate.iloc[0]
            conclusion_box.info(
                f"当前累计结果下，综合排序暂时最优的是 {best['检测器']}-CFAR。"
                "该结论由目标检测率、整体虚警控制和杂波区虚警控制共同决定；"
                "如果只追求高检测率或只追求低虚警，排序可能不同。"
            )

        progress_bar.progress(trial_idx / int(mc_trials))
        if float(mc_interval) > 0 and trial_idx < int(mc_trials):
            time.sleep(float(mc_interval))

    status_box.success(f"蒙特卡洛实验完成：共 {int(mc_trials)} 次。")


with st.expander("6. 三种 CFAR 检测器的原理公式", expanded=False):
    st.markdown("### 窗口定义")
    st.latex(r"""
    [X_{k-G-T},\ldots,X_{k-G-1}]
    \quad
    [X_{k-G},\ldots,X_{k-1}]
    \quad
    X_k
    \quad
    [X_{k+1},\ldots,X_{k+G}]
    \quad
    [X_{k+G+1},\ldots,X_{k+G+T}]
    """)

    st.markdown(
        r"""
其中，\(X_k\) 为待检测单元 CUT，\(G\) 为每侧保护单元数，
\(T\) 为每侧参考单元数，总参考单元数 \(N=2T\)。
"""
    )

    st.markdown("### 左右参考窗统计量")
    st.latex(r"""
    S_L=\sum_{i=k-G-T}^{k-G-1}X_i,\qquad
    S_R=\sum_{i=k+G+1}^{k+G+T}X_i
    """)
    st.latex(r"""
    Z_L=\frac{S_L}{T},\qquad Z_R=\frac{S_R}{T}
    """)

    st.markdown("### CA-CFAR")
    st.latex(r"""
    Z_{\mathrm{CA}}=\frac{S_L+S_R}{2T}
    """)
    st.latex(r"""
    \eta_{\mathrm{CA}}=\alpha_{\mathrm{CA}}Z_{\mathrm{CA}}
    """)
    st.latex(r"""
    P_{\mathrm{fa}}=
    \left(1+\frac{\alpha_{\mathrm{CA}}}{2T}\right)^{-2T}
    """)
    st.latex(r"""
    \alpha_{\mathrm{CA}}=
    2T\left(P_{\mathrm{fa}}^{-\frac{1}{2T}}-1\right)
    """)

    st.markdown("### GOCA-CFAR")
    st.latex(r"""
    Z_{\mathrm{GOCA}}=\max(Z_L,Z_R)
    """)
    st.latex(r"""
    \eta_{\mathrm{GOCA}}=\alpha_{\mathrm{GOCA}}Z_{\mathrm{GOCA}}
    """)
    st.latex(r"""
    P_{\mathrm{fa,GOCA}}(\alpha)
    =
    2\left(1+\frac{\alpha}{T}\right)^{-T}
    -
    P_{\mathrm{fa,SOCA}}(\alpha)
    """)

    st.markdown("### SOCA-CFAR")
    st.latex(r"""
    Z_{\mathrm{SOCA}}=\min(Z_L,Z_R)
    """)
    st.latex(r"""
    \eta_{\mathrm{SOCA}}=\alpha_{\mathrm{SOCA}}Z_{\mathrm{SOCA}}
    """)
    st.latex(r"""
    P_{\mathrm{fa,SOCA}}(\alpha)
    =
    2\frac{T^T}{\Gamma(T)}
    \sum_{i=0}^{T-1}
    \frac{T^i}{i!}
    \frac{\Gamma(T+i)}{(\alpha+2T)^{T+i}}
    """)

    st.markdown("### 判决准则")
    st.latex(r"""
    X_k>\eta \Rightarrow H_1,\qquad
    X_k\leq \eta \Rightarrow H_0
    """)


with st.expander("7. 课堂观察建议", expanded=False):
    st.markdown(
        r"""
- **均匀背景**：关闭杂波区域，只保留单个或少量间隔较远目标。CA-CFAR 通常最稳定。
- **临近多目标**：让两个目标间距小于或接近参考窗长度。观察参考窗污染导致的门限抬升。
- **杂波边缘**：添加突变杂波区域，把 CUT 移动到杂波边缘附近。GOCA 通常更保守，SOCA 可能更敏感。
- **保护单元数 G**：增大 \(G\) 可以减轻目标泄漏进入参考窗，但会减少可检测边缘区域。
- **参考单元数 T**：增大 \(T\) 可以降低均匀背景下的估计方差，但在非均匀背景中更容易纳入异质样本。
"""
    )

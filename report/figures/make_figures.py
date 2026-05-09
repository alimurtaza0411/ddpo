#!/usr/bin/env python3
"""
make_figures.py — generate all data figures for the report.

Reads run-1 metrics_history.json files from the project root and uses
inlined run-2 / KL-only data (from the VM, fetched May 8 2026).

Outputs (saved next to this script):
    pareto_with_errorbars.png   — final-step PickScore vs CLIP variance, all runs
    per_category_collapse.png   — bar chart of ΔCLIP variance by prompt category
    training_curves.png         — PickScore + CLIP variance vs training step
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

RUN1_BASE = os.path.join(PROJ, "ddpo_baseline", "run1", "metrics_history.json")
RUN1_MULTI = os.path.join(PROJ, "ddpo_multi_objective", "run1", "metrics_history.json")


def load_history(path: str) -> list:
    with open(path) as f:
        return json.load(f)


# ── Run-2 + KL-only data (inlined; pulled from VM 2026-05-08) ─────────────────
# Per-step entries with overall + per-category metrics. Each is a sparse
# dict; we only fill the fields the plots use.
RUN2_BASELINE = [
    (0,   19.984, 0.659, 0.000359),
    (25,  19.982, 0.659, 0.000359),
    (50,  19.990, 0.658, 0.000359),
    (75,  20.038, 0.658, 0.000357),
    (100, 20.060, 0.656, 0.000359),
    (125, 20.104, 0.656, 0.000357),
    (150, 20.138, 0.656, 0.000355),
    (175, 20.184, 0.654, 0.000353),
    (200, 20.202, 0.655, 0.000361),
    (225, 20.230, 0.653, 0.000360),
    (250, 20.282, 0.650, 0.000355),
    (275, 20.349, 0.648, 0.000361),
    (300, 20.394, 0.647, 0.000360),
    (325, 20.457, 0.645, 0.000350),
    (350, 20.502, 0.645, 0.000339),
    (375, 20.548, 0.647, 0.000337),
    (400, 20.606, 0.646, 0.000328),
    (425, 20.635, 0.647, 0.000327),
    (450, 20.742, 0.646, 0.000323),
    (475, 20.796, 0.641, 0.000316),
    (500, 20.729, 0.650, 0.000330),
]
RUN2_MULTI = [
    (0,   19.984, 0.659, 0.000359),
    (25,  19.986, 0.659, 0.000359),
    (50,  20.000, 0.659, 0.000358),
    (75,  20.039, 0.657, 0.000362),
    (100, 20.104, 0.655, 0.000363),
    (125, 20.131, 0.656, 0.000364),
    (150, 20.135, 0.657, 0.000364),
    (175, 20.172, 0.657, 0.000364),
    (200, 20.181, 0.659, 0.000360),
    (225, 20.188, 0.658, 0.000368),
    (250, 20.234, 0.659, 0.000364),
    (275, 20.266, 0.661, 0.000362),
    (300, 20.294, 0.658, 0.000366),
    (325, 20.314, 0.658, 0.000364),
    (350, 20.368, 0.655, 0.000361),
    (375, 20.397, 0.655, 0.000357),
    (400, 20.447, 0.655, 0.000354),
    (425, 20.511, 0.653, 0.000345),
    (450, 20.558, 0.651, 0.000339),
    (475, 20.582, 0.650, 0.000343),
    (500, 20.507, 0.655, 0.000349),
]
KL_ONLY = [
    (0,   19.984, 0.659, 0.000359),
    (25,  19.987, 0.659, 0.000359),
    (50,  19.999, 0.658, 0.000360),
    (75,  20.030, 0.660, 0.000360),
    (100, 20.066, 0.660, 0.000356),
    (125, 20.094, 0.660, 0.000357),
    (150, 20.093, 0.660, 0.000358),
    (175, 20.087, 0.661, 0.000361),
    (200, 20.102, 0.663, 0.000362),
    (225, 20.110, 0.661, 0.000362),
    (250, 20.012, 0.668, 0.000370),
]


def history_to_arrays(history: list) -> tuple:
    """Convert metrics_history.json list → (steps, pickscore, lpips, clipvar) arrays."""
    steps = [int(e["step"]) for e in history]
    ps = [float(e["eval/overall/pickscore"]) for e in history]
    lp = [float(e["eval/overall/diversity"]) for e in history]
    cv = [float(e["eval/overall/clip_variance"]) for e in history]
    return np.array(steps), np.array(ps), np.array(lp), np.array(cv)


def tuples_to_arrays(rows: list) -> tuple:
    arr = np.array(rows)
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2], arr[:, 3]


# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

COLORS = {
    "baseline_run1":  "#d62728",  # red
    "baseline_run2":  "#ff9896",  # light red
    "multi_run1":     "#1f77b4",  # blue
    "multi_run2":     "#aec7e8",  # light blue
    "kl_only":        "#2ca02c",  # green
}
LABELS = {
    "baseline_run1":  "Baseline (run-1, A100)",
    "baseline_run2":  "Baseline (run-2, V100)",
    "multi_run1":     "Multi-obj (run-1, A100)",
    "multi_run2":     "Multi-obj (run-2, V100)",
    "kl_only":        "KL-only ($n{=}1$, 250)",
}


# ── Figure 1: Pareto scatter at final step ──────────────────────────────────
def make_pareto_plot(out_path: str) -> None:
    h_b1 = load_history(RUN1_BASE);   _, ps_b1, _, cv_b1 = history_to_arrays(h_b1)
    h_m1 = load_history(RUN1_MULTI);  _, ps_m1, _, cv_m1 = history_to_arrays(h_m1)
    _, ps_b2, _, cv_b2 = tuples_to_arrays(RUN2_BASELINE)
    _, ps_m2, _, cv_m2 = tuples_to_arrays(RUN2_MULTI)
    _, ps_kl, _, cv_kl = tuples_to_arrays(KL_ONLY)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    # Step-0 reference (all runs share it on the V100 side; A100 step 0 differs)
    ax.scatter([19.925], [0.000370], marker="*", s=140, color="black",
               zorder=5, label="Base SD 1.5 (step 0)")

    # Final-step points with error bars where we have n=2 (baseline / multi-obj)
    ps_b_mean = np.mean([ps_b1[-1], ps_b2[-1]])
    ps_b_std  = np.std ([ps_b1[-1], ps_b2[-1]], ddof=1)
    cv_b_mean = np.mean([cv_b1[-1], cv_b2[-1]])
    cv_b_std  = np.std ([cv_b1[-1], cv_b2[-1]], ddof=1)
    ax.errorbar([ps_b_mean], [cv_b_mean], xerr=[ps_b_std], yerr=[cv_b_std],
                fmt='o', color=COLORS["baseline_run1"], markersize=10,
                capsize=4, label="Baseline (mean of $n{=}2$)")

    ps_m_mean = np.mean([ps_m1[-1], ps_m2[-1]])
    ps_m_std  = np.std ([ps_m1[-1], ps_m2[-1]], ddof=1)
    cv_m_mean = np.mean([cv_m1[-1], cv_m2[-1]])
    cv_m_std  = np.std ([cv_m1[-1], cv_m2[-1]], ddof=1)
    ax.errorbar([ps_m_mean], [cv_m_mean], xerr=[ps_m_std], yerr=[cv_m_std],
                fmt='s', color=COLORS["multi_run1"], markersize=10,
                capsize=4, label="Multi-objective (mean of $n{=}2$)")

    ax.scatter([ps_kl[-1]], [cv_kl[-1]], marker="^", s=120,
               color=COLORS["kl_only"], label=LABELS["kl_only"], zorder=4)

    # Annotate the trade-off direction
    ax.annotate("", xy=(ps_b_mean, cv_b_mean), xytext=(ps_m_mean, cv_m_mean),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.0, alpha=0.6))
    ax.text((ps_b_mean+ps_m_mean)/2, (cv_b_mean+cv_m_mean)/2 + 0.000005,
            "trade-off", fontsize=7, color="gray", style="italic", ha="center")

    ax.set_xlabel("Final PickScore (preference reward, $\\uparrow$)")
    ax.set_ylabel("Final CLIP embedding variance (semantic diversity, $\\uparrow$)")
    ax.set_title("Reward vs.\\ semantic diversity at the final checkpoint")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)


# ── Figure 2: Per-category collapse bar chart ───────────────────────────────
def make_per_category_plot(out_path: str) -> None:
    """Bar chart: ΔCLIPvar % by category, for baseline / multi-obj / KL-only."""
    h_b1 = load_history(RUN1_BASE)
    h_m1 = load_history(RUN1_MULTI)

    cats = ["constrained", "common_subject", "open_ended"]
    cat_labels = ["Constrained", "Common-subject", "Open-ended"]

    def pct(h, cat):
        cv0 = float(h[0][f"eval/{cat}/clip_variance"])
        cv1 = float(h[-1][f"eval/{cat}/clip_variance"])
        return (cv1 - cv0) / cv0 * 100

    # Run-2 + KL-only per-category numbers (from your data above; computed earlier)
    run2_baseline_pct = {"constrained": -16.9, "common_subject": -12.4, "open_ended": -1.3}
    run2_multi_pct    = {"constrained":  -7.2, "common_subject": -10.1, "open_ended": +4.4}
    kl_only_pct       = {"constrained":  +0.7, "common_subject":  +2.5, "open_ended": +4.5}

    # Mean across 2 seeds for baseline and multi-obj
    baseline_mean = [(pct(h_b1, c) + run2_baseline_pct[c]) / 2 for c in cats]
    multi_mean    = [(pct(h_m1, c) + run2_multi_pct[c])    / 2 for c in cats]
    kl_only_vals  = [kl_only_pct[c] for c in cats]

    x = np.arange(len(cats))
    width = 0.27

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    b1 = ax.bar(x - width, baseline_mean, width, label="Baseline (mean $n{=}2$)",
                color=COLORS["baseline_run1"], edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x,         multi_mean,    width, label="Multi-objective (mean $n{=}2$)",
                color=COLORS["multi_run1"], edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + width, kl_only_vals,  width, label="KL-only ($n{=}1$, 250)",
                color=COLORS["kl_only"], edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel("$\\Delta$ CLIP variance from step 0 (\\%)")
    ax.set_title("Per-category diversity collapse (lower = more collapse)")
    ax.grid(alpha=0.25, axis='y')
    ax.legend(loc="lower right", fontsize=7)

    # Value labels
    for bars in [b1, b2, b3]:
        for r in bars:
            h = r.get_height()
            offset = 1.0 if h >= 0 else -2.5
            ax.text(r.get_x() + r.get_width()/2, h + offset, f"{h:+.1f}",
                    ha="center", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)


# ── Figure 3: Training curves ───────────────────────────────────────────────
def make_training_curves(out_path: str) -> None:
    h_b1 = load_history(RUN1_BASE);   s_b1, ps_b1, _, cv_b1 = history_to_arrays(h_b1)
    h_m1 = load_history(RUN1_MULTI);  s_m1, ps_m1, _, cv_m1 = history_to_arrays(h_m1)
    s_b2, ps_b2, _, cv_b2 = tuples_to_arrays(RUN2_BASELINE)
    s_m2, ps_m2, _, cv_m2 = tuples_to_arrays(RUN2_MULTI)
    s_kl, ps_kl, _, cv_kl = tuples_to_arrays(KL_ONLY)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))

    for ax, ys, label_y in zip(
        axes,
        [(ps_b1, ps_b2, ps_m1, ps_m2, ps_kl),
         (cv_b1, cv_b2, cv_m1, cv_m2, cv_kl)],
        ["PickScore (preference reward, $\\uparrow$)",
         "CLIP variance (semantic diversity, $\\uparrow$)"],
    ):
        for steps_arr, y_arr, key, lw in [
            (s_b1, ys[0], "baseline_run1", 1.6),
            (s_b2, ys[1], "baseline_run2", 1.4),
            (s_m1, ys[2], "multi_run1", 1.6),
            (s_m2, ys[3], "multi_run2", 1.4),
            (s_kl, ys[4], "kl_only", 1.6),
        ]:
            ax.plot(steps_arr, y_arr, color=COLORS[key], label=LABELS[key],
                    linewidth=lw, marker="o", markersize=2.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel(label_y)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)


def main() -> int:
    print("Generating figures ...")
    make_pareto_plot(os.path.join(HERE, "pareto_with_errorbars.png"))
    make_per_category_plot(os.path.join(HERE, "per_category_collapse.png"))
    make_training_curves(os.path.join(HERE, "training_curves.png"))
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

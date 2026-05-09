#!/usr/bin/env python3
"""
report_summary.py — generate report-ready tables and plots from the cross-run CSV.

Reads:
    <output_dir>/run_comparison.csv   (produced by compare_runs.py)

Writes:
    <output_dir>/summary_final_step.md       — markdown table for the report
    <output_dir>/summary_final_step.tex      — LaTeX table for direct paste into report.tex
    <output_dir>/delta_step0_to_final.tex    — Δ-from-step-0 table (mirrors report Table 2)
    <output_dir>/training_curves.png         — 3-panel curves (PickScore, LPIPS, CLIPvar)
    <output_dir>/pareto_with_errorbars.png   — full trajectories in (PickScore, CLIPvar) space
    <output_dir>/pareto_final_step.png       — clean final-step Pareto scatter

Designed to run on the VM at end of run_all.sh, OR locally after pulling the
outputs/_comparison/ folder back.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def to_float(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def by_run(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(r["run"], []).append(r)
    for run in grouped:
        grouped[run].sort(key=lambda r: int(float(r["step"])))
    return grouped


def fmt(mean: float, std: float, decimals: int = 4) -> str:
    if math.isnan(mean):
        return "—"
    if math.isnan(std) or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def write_final_step_table(grouped: Dict[str, List[Dict[str, str]]], out_dir: str) -> None:
    md_lines = [
        "# Final-step comparison (mean ± std across seeds)\n",
        "| Run | n | step | PickScore | LPIPS | CLIP variance | ImageReward |",
        "|-----|---|------|-----------|-------|---------------|-------------|",
    ]
    tex_rows: List[str] = []
    for run, rs in grouped.items():
        if not rs:
            continue
        last = rs[-1]
        n = last.get("n_seeds", "1")
        step = last["step"]
        ps = fmt(to_float(last["pickscore"]), to_float(last.get("pickscore_std", "0")))
        lp = fmt(to_float(last["lpips_diversity"]), to_float(last.get("lpips_diversity_std", "0")))
        cv = fmt(to_float(last["clip_variance"]), to_float(last.get("clip_variance_std", "0")), decimals=6)
        ir = fmt(to_float(last["image_reward"]), to_float(last.get("image_reward_std", "0")))
        md_lines.append(f"| {run} | {n} | {step} | {ps} | {lp} | {cv} | {ir} |")
        tex_rows.append(f"{run} & {step} & {ps} & {lp} & {cv} \\\\")

    md_path = os.path.join(out_dir, "summary_final_step.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Final-step comparison (mean $\\pm$ std across seeds, where $n>1$).}\n"
        "\\label{tab:final}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Run & Step & PickScore & LPIPS & CLIP Var. \\\\\n"
        "\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(os.path.join(out_dir, "summary_final_step.tex"), "w") as f:
        f.write(tex)
    print(f"  wrote {md_path}")


def write_delta_table(grouped: Dict[str, List[Dict[str, str]]], out_dir: str) -> None:
    rows: List[str] = []
    for run, rs in grouped.items():
        if len(rs) < 2:
            continue
        first, last = rs[0], rs[-1]
        d_ps = to_float(last["pickscore"]) - to_float(first["pickscore"])
        first_lp = to_float(first["lpips_diversity"])
        first_cv = to_float(first["clip_variance"])
        d_lp_pct = (to_float(last["lpips_diversity"]) - first_lp) / first_lp * 100 if first_lp else float("nan")
        d_cv_pct = (to_float(last["clip_variance"]) - first_cv) / first_cv * 100 if first_cv else float("nan")
        rows.append(f"{run} & {d_ps:+.4f} & {d_lp_pct:+.1f}\\% & {d_cv_pct:+.1f}\\% \\\\")

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Change from step 0 to the final checkpoint. Multi-objective and KL-only "
        "are expected to retain more diversity (less negative $\\Delta$\\,CLIP\\,Var.) at the "
        "cost of some preference reward.}\n"
        "\\label{tab:changes}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Run & $\\Delta$PickScore & $\\Delta$LPIPS & $\\Delta$CLIP Var. \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(os.path.join(out_dir, "delta_step0_to_final.tex"), "w") as f:
        f.write(tex)


def make_curves_plot(grouped: Dict[str, List[Dict[str, str]]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [warn] matplotlib not available: {exc}")
        return

    metrics = [
        ("pickscore",       "PickScore (↑)",      "pickscore_std"),
        ("lpips_diversity", "LPIPS diversity (↑)", "lpips_diversity_std"),
        ("clip_variance",   "CLIP variance (↑)",   "clip_variance_std"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, label, std_key) in zip(axes, metrics):
        for run, rs in grouped.items():
            xs = [int(float(r["step"])) for r in rs]
            ys = [to_float(r[metric]) for r in rs]
            errs = [to_float(r.get(std_key, "0")) for r in rs]
            if any(not math.isnan(e) and e > 0 for e in errs):
                ax.errorbar(xs, ys, yerr=errs, marker="o", label=run, capsize=2, elinewidth=0.7, alpha=0.85)
            else:
                ax.plot(xs, ys, marker="o", label=run, alpha=0.85)
        ax.set_xlabel("Training step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")
    plt.close(fig)


def make_pareto_final_plot(grouped: Dict[str, List[Dict[str, str]]], out_dir: str) -> None:
    """Final-step Pareto scatter — one point per run, with error bars where n>1.
    Cleaner than the time-evolution version when many runs are plotted."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for run, rs in grouped.items():
        if not rs:
            continue
        last = rs[-1]
        x = to_float(last["pickscore"])
        y = to_float(last["clip_variance"])
        xerr = to_float(last.get("pickscore_std", "0"))
        yerr = to_float(last.get("clip_variance_std", "0"))
        ax.errorbar(
            [x], [y], xerr=[xerr], yerr=[yerr],
            marker="o", markersize=10, label=run,
            capsize=4, elinewidth=1.0,
        )
        # Annotate each point with its name
        ax.annotate(run, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Final PickScore (preference reward)")
    ax.set_ylabel("Final CLIP embedding variance (semantic diversity)")
    ax.set_title("Pareto frontier — final-step values across all runs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    path = os.path.join(out_dir, "pareto_final_step.png")
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")
    plt.close(fig)


def make_pareto_plot(grouped: Dict[str, List[Dict[str, str]]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for run, rs in grouped.items():
        xs  = [to_float(r["pickscore"])      for r in rs]
        ys  = [to_float(r["clip_variance"])  for r in rs]
        ex  = [to_float(r.get("pickscore_std", "0"))     for r in rs]
        ey  = [to_float(r.get("clip_variance_std", "0")) for r in rs]
        ax.errorbar(xs, ys, xerr=ex, yerr=ey, marker="o", label=run, capsize=3, elinewidth=0.8, alpha=0.85)
    ax.set_xlabel("PickScore (preference reward)")
    ax.set_ylabel("CLIP embedding variance (semantic diversity)")
    ax.set_title("Reward vs. semantic diversity (mean ± std across seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "pareto_with_errorbars.png")
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True, help="Directory containing run_comparison.csv")
    args = p.parse_args()

    csv_path = os.path.join(args.output_dir, "run_comparison.csv")
    if not os.path.isfile(csv_path):
        print(f"FATAL: not found: {csv_path}", file=sys.stderr)
        return 1

    print(f"Loading {csv_path}")
    rows = load_rows(csv_path)
    grouped = by_run(rows)
    print(f"  loaded {len(rows)} rows across {len(grouped)} runs: {list(grouped)}")

    write_final_step_table(grouped, args.output_dir)
    write_delta_table(grouped, args.output_dir)
    make_curves_plot(grouped, args.output_dir)
    make_pareto_plot(grouped, args.output_dir)
    make_pareto_final_plot(grouped, args.output_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

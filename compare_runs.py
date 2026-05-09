#!/usr/bin/env python3
"""
Compare DDPO runs across reward, LPIPS diversity, and CLIP variance.

Example:
    python compare_runs.py \
      --run baseline=/content/drive/MyDrive/ddpo_baseline/run1 \
      --run multi=/content/drive/MyDrive/ddpo_multi_objective/run1 \
      --output_dir comparisons/baseline_vs_multi
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


def _metrics_path(path: str) -> str:
    """Resolve either a run directory or a direct metrics_history.json path."""
    if os.path.isfile(path):
        return path

    direct = os.path.join(path, "metrics_history.json")
    if os.path.isfile(direct):
        return direct

    ckpt_dir = os.path.join(path, "checkpoints")
    if os.path.isdir(ckpt_dir):
        step_dirs = sorted(
            d for d in os.listdir(ckpt_dir)
            if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, d))
        )
        for step_dir in reversed(step_dirs):
            candidate = os.path.join(ckpt_dir, step_dir, "metrics_history.json")
            if os.path.isfile(candidate):
                return candidate

    raise FileNotFoundError(f"Could not find metrics_history.json under {path}")


def _load_run(spec: str) -> Tuple[str, List[Dict[str, float]]]:
    if "=" not in spec:
        raise ValueError("--run must be NAME=PATH")
    name, path = spec.split("=", 1)
    with open(_metrics_path(path)) as f:
        return name, json.load(f)


def _pareto_front(rows: List[Dict[str, float]], objectives: List[str]) -> List[bool]:
    """Return a mask for nondominated rows, maximizing all objectives."""
    keep = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = all(other[obj] >= row[obj] for obj in objectives)
            strictly_better = any(other[obj] > row[obj] for obj in objectives)
            if no_worse and strictly_better:
                dominated = True
                break
        keep.append(not dominated)
    return keep


def _aggregate_by_step(
    histories: List[List[Dict[str, float]]],
) -> List[Dict[str, float]]:
    """
    Average per-step metrics across multiple seeds. Returns one row per step
    that appears in ALL histories, with mean and std for each numeric metric.
    """
    import math

    keys = ["pickscore", "lpips_diversity", "clip_variance", "image_reward"]
    src_keys = {
        "pickscore": "eval/overall/pickscore",
        "lpips_diversity": "eval/overall/diversity",
        "clip_variance": "eval/overall/clip_variance",
        "image_reward": "eval/overall/image_reward",
    }

    by_step: List[Dict[int, Dict[str, float]]] = []
    for history in histories:
        m: Dict[int, Dict[str, float]] = {}
        for entry in history:
            m[entry["step"]] = {
                k: entry.get(src_keys[k], float("nan")) for k in keys
            }
        by_step.append(m)

    common_steps = sorted(set.intersection(*[set(m.keys()) for m in by_step]))

    rows: List[Dict[str, float]] = []
    for step in common_steps:
        row: Dict[str, float] = {"step": step, "n_seeds": len(histories)}
        for k in keys:
            vals = [m[step][k] for m in by_step]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            if not vals:
                row[k] = float("nan")
                row[f"{k}_std"] = float("nan")
            else:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
                row[k] = mean
                row[f"{k}_std"] = math.sqrt(var) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DDPO metrics histories")
    parser.add_argument(
        "--run", action="append", required=True,
        help="NAME=run_dir_or_metrics_json. Repeat the same NAME with different paths "
             "to aggregate seeds as mean±std.",
    )
    parser.add_argument("--output_dir", default="comparisons", help="Where to write CSV/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = [_load_run(spec) for spec in args.run]

    # Group histories by NAME for multi-seed aggregation.
    grouped: Dict[str, List[List[Dict[str, float]]]] = {}
    order: List[str] = []
    for name, history in runs:
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(history)

    rows: List[Dict[str, float]] = []
    for name in order:
        histories = grouped[name]
        aggregated = _aggregate_by_step(histories)
        for r in aggregated:
            rows.append({
                "run": name,
                "step": r["step"],
                "n_seeds": r["n_seeds"],
                "pickscore": r["pickscore"],
                "pickscore_std": r["pickscore_std"],
                "lpips_diversity": r["lpips_diversity"],
                "lpips_diversity_std": r["lpips_diversity_std"],
                "clip_variance": r["clip_variance"],
                "clip_variance_std": r["clip_variance_std"],
                "image_reward": r["image_reward"],
                "image_reward_std": r["image_reward_std"],
            })

    objectives = ["pickscore", "lpips_diversity", "clip_variance"]
    pareto_mask = _pareto_front(rows, objectives)
    for row, is_pareto in zip(rows, pareto_mask):
        row["pareto"] = is_pareto

    csv_path = os.path.join(args.output_dir, "run_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    print("\nFinal checkpoints (mean across seeds):")
    for name in order:
        run_rows = [r for r in rows if r["run"] == name]
        if not run_rows:
            continue
        final = run_rows[-1]
        n = final["n_seeds"]
        print(
            f"{name:<22} step={final['step']:<5} n={n} "
            f"PickScore={final['pickscore']:.4f}±{final['pickscore_std']:.4f} "
            f"LPIPS={final['lpips_diversity']:.4f}±{final['lpips_diversity_std']:.4f} "
            f"CLIPvar={final['clip_variance']:.6f}±{final['clip_variance_std']:.6f}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for name in order:
            run_rows = [r for r in rows if r["run"] == name]
            xs = [r["pickscore"] for r in run_rows]
            ys = [r["clip_variance"] for r in run_rows]
            xerr = [r["pickscore_std"] for r in run_rows]
            yerr = [r["clip_variance_std"] for r in run_rows]
            ax.errorbar(
                xs, ys, xerr=xerr, yerr=yerr,
                marker="o", label=name, capsize=3, elinewidth=0.8, alpha=0.9,
            )
        ax.set_xlabel("Overall PickScore")
        ax.set_ylabel("Overall CLIP Variance")
        ax.set_title("Reward vs. Semantic Diversity (mean ± std across seeds)")
        ax.legend()
        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, "pickscore_vs_clip_variance.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Skipped plot generation: {exc}")


if __name__ == "__main__":
    main()

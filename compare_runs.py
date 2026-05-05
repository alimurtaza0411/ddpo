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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DDPO metrics histories")
    parser.add_argument("--run", action="append", required=True, help="NAME=run_dir_or_metrics_json")
    parser.add_argument("--output_dir", default="comparisons", help="Where to write CSV/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = [_load_run(spec) for spec in args.run]

    rows: List[Dict[str, float]] = []
    for name, history in runs:
        for entry in history:
            rows.append({
                "run": name,
                "step": entry["step"],
                "pickscore": entry["eval/overall/pickscore"],
                "lpips_diversity": entry["eval/overall/diversity"],
                "clip_variance": entry.get("eval/overall/clip_variance", float("nan")),
                "image_reward": entry.get("eval/overall/image_reward", float("nan")),
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
    print("\nFinal checkpoints:")
    for name, history in runs:
        final = history[-1]
        print(
            f"{name:<16} step={final['step']:<5} "
            f"PickScore={final['eval/overall/pickscore']:.4f} "
            f"LPIPS={final['eval/overall/diversity']:.4f} "
            f"CLIPvar={final.get('eval/overall/clip_variance', float('nan')):.6f}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for name, _history in runs:
            run_rows = [r for r in rows if r["run"] == name]
            ax.plot(
                [r["pickscore"] for r in run_rows],
                [r["clip_variance"] for r in run_rows],
                marker="o",
                label=name,
            )
        ax.set_xlabel("Overall PickScore")
        ax.set_ylabel("Overall CLIP Variance")
        ax.set_title("Reward vs. Semantic Diversity")
        ax.legend()
        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, "pickscore_vs_clip_variance.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"Skipped plot generation: {exc}")


if __name__ == "__main__":
    main()

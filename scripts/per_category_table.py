#!/usr/bin/env python3
"""
per_category_table.py — generate a per-prompt-category breakdown table from
existing metrics_history.json files.

The orchestrator already logs eval metrics stratified by prompt category
(constrained / common_subject / open_ended). This script walks the run
outputs, computes the step-0 -> final-step delta for each category, and
emits both a markdown table and a LaTeX table directly pasteable into the
report.

Why this matters: the report claims open-ended prompts collapse hardest;
this table substantiates that claim with real numbers per category.

Reads:
    outputs/<run>/metrics_history.json
    run1_results/<run>/metrics_history.json   (if present)

Writes:
    outputs/_comparison/per_category_breakdown.md
    outputs/_comparison/per_category_breakdown.tex

Usage:
    python3 scripts/per_category_table.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# Map of human-readable run name → metrics_history.json path
DEFAULT_RUNS = [
    ("baseline (run-1, A100)",   "run1_results/ddpo_baseline/run1/metrics_history.json"),
    ("baseline (run-2, V100)",   "outputs/baseline_l4_run2/metrics_history.json"),
    ("multi-obj (run-1, A100)",  "run1_results/ddpo_multi_objective/run1/metrics_history.json"),
    ("multi-obj (run-2, V100)",  "outputs/multi_objective_l4_run2/metrics_history.json"),
    ("KL-only ablation",         "outputs/kl_only_l4_run1/metrics_history.json"),
]

CATEGORIES = ["constrained", "common_subject", "open_ended"]


def load_run(path: str) -> Optional[List[dict]]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def safe_float(v) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def category_deltas(history: List[dict]) -> Dict[str, Dict[str, float]]:
    """Return per-category step0->final ΔCLIPvar (relative %) and ΔPickScore."""
    if not history:
        return {}
    first, last = history[0], history[-1]
    out: Dict[str, Dict[str, float]] = {}
    for cat in CATEGORIES:
        ps0 = safe_float(first.get(f"eval/{cat}/pickscore"))
        ps1 = safe_float(last.get(f"eval/{cat}/pickscore"))
        cv0 = safe_float(first.get(f"eval/{cat}/clip_variance"))
        cv1 = safe_float(last.get(f"eval/{cat}/clip_variance"))
        lp0 = safe_float(first.get(f"eval/{cat}/diversity"))
        lp1 = safe_float(last.get(f"eval/{cat}/diversity"))
        out[cat] = {
            "pickscore_step0": ps0,
            "pickscore_final": ps1,
            "delta_pickscore": ps1 - ps0,
            "clipvar_step0": cv0,
            "clipvar_final": cv1,
            "delta_clipvar_pct": (cv1 - cv0) / cv0 * 100 if cv0 else float("nan"),
            "lpips_step0": lp0,
            "lpips_final": lp1,
            "delta_lpips_pct": (lp1 - lp0) / lp0 * 100 if lp0 else float("nan"),
        }
    return out


def fmt_pct(x: float, decimals: int = 1) -> str:
    if x != x:  # NaN
        return "—"
    return f"{x:+.{decimals}f}\\%"


def fmt_pct_md(x: float, decimals: int = 1) -> str:
    if x != x:
        return "—"
    return f"{x:+.{decimals}f}%"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default="outputs/_comparison",
                   help="Where to write the breakdown files")
    p.add_argument("--repo_root", default=".",
                   help="Repo root (for resolving the default run paths)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading runs ...")
    rows = []
    for name, rel in DEFAULT_RUNS:
        path = os.path.join(args.repo_root, rel)
        h = load_run(path)
        if h is None:
            print(f"  skip {name} (not found at {rel})")
            continue
        deltas = category_deltas(h)
        rows.append((name, deltas))
        print(f"  loaded {name}: {len(h)} entries, last step = {h[-1].get('step', '?')}")

    if not rows:
        print("FATAL: no runs found", file=sys.stderr)
        return 1

    # ── Markdown table — for easy reading ─────────────────────────────────
    md_lines = [
        "# Per-Category Diversity-Collapse Breakdown",
        "",
        "Δ CLIP variance from step 0 to the final checkpoint, broken down by",
        "the three eval-prompt categories. The collapse signal is expected to",
        "be strongest on open-ended prompts because the model has the most",
        "freedom to mode-collapse there.",
        "",
        "| Run | Constrained | Common-subject | Open-ended |",
        "|-----|-------------|----------------|------------|",
    ]
    for name, deltas in rows:
        c = fmt_pct_md(deltas.get("constrained", {}).get("delta_clipvar_pct", float("nan")))
        cs = fmt_pct_md(deltas.get("common_subject", {}).get("delta_clipvar_pct", float("nan")))
        oe = fmt_pct_md(deltas.get("open_ended", {}).get("delta_clipvar_pct", float("nan")))
        md_lines.append(f"| {name} | {c} | {cs} | {oe} |")

    md_lines += ["", "## Same breakdown for ΔPickScore (preference reward, higher = better)", ""]
    md_lines += [
        "| Run | Constrained | Common-subject | Open-ended |",
        "|-----|-------------|----------------|------------|",
    ]
    for name, deltas in rows:
        c = f"{deltas.get('constrained', {}).get('delta_pickscore', float('nan')):+.3f}"
        cs = f"{deltas.get('common_subject', {}).get('delta_pickscore', float('nan')):+.3f}"
        oe = f"{deltas.get('open_ended', {}).get('delta_pickscore', float('nan')):+.3f}"
        md_lines.append(f"| {name} | {c} | {cs} | {oe} |")

    md_path = os.path.join(args.output_dir, "per_category_breakdown.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nWrote {md_path}")

    # ── LaTeX table — directly paste-able into report.tex ──────────────────
    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Per-category breakdown of CLIP-variance collapse "
        "($\\Delta$ from step~0 to final checkpoint, relative). The collapse "
        "signal is strongest on open-ended prompts, where the base model has "
        "the most freedom to mode-collapse onto a narrow visual cliché.}\n"
        "\\label{tab:per_category}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Run & Constrained & Common-subject & Open-ended \\\\\n"
        "\\midrule\n"
    )
    underscore = "_"
    latex_underscore = "\\_"
    for name, deltas in rows:
        c = fmt_pct(deltas.get("constrained", {}).get("delta_clipvar_pct", float("nan")))
        cs = fmt_pct(deltas.get("common_subject", {}).get("delta_clipvar_pct", float("nan")))
        oe = fmt_pct(deltas.get("open_ended", {}).get("delta_clipvar_pct", float("nan")))
        safe_name = name.replace(underscore, latex_underscore)
        tex += f"{safe_name} & {c} & {cs} & {oe} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    tex_path = os.path.join(args.output_dir, "per_category_breakdown.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Wrote {tex_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

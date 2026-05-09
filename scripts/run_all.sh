#!/usr/bin/env bash
# run_all.sh — master orchestration script for the DDPO experiment matrix.
#
# Run this on the GCP L4 VM after bootstrap_vm.sh completes.
# Designed to be invoked inside tmux so it survives ssh drops:
#
#   tmux new -s ddpo
#   cd ~/RL-project && bash scripts/run_all.sh
#   # detach: Ctrl-B then D
#
# Idempotent: each experiment runs with --resume_from auto, so re-running this
# script picks up where it left off. The status file at outputs/run_all_status.json
# tracks which experiments are complete.
#
# Wall time on V100 (~11s/step):
#   Tier 1 (default):                ~4 hours  (baseline×1, multi-obj×1, kl_only)
#   + Phase A (RUN_PHASE_A=1):       +~6 hours (third seeds, div2, faithfulness_only)
#   + Tier 2  (RUN_TIER2=1):         +~50 min  (high_div ablation)
#
# Usage:
#   bash scripts/run_all.sh                          # Tier 1 only
#   RUN_PHASE_A=1 bash scripts/run_all.sh            # Tier 1 + Phase A
#   RUN_PHASE_A=1 RUN_TIER2=1 bash scripts/run_all.sh # everything

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

LOGDIR="${REPO_DIR}/outputs/_run_all_logs"
STATUS_FILE="${REPO_DIR}/outputs/run_all_status.json"
mkdir -p "${LOGDIR}" "${REPO_DIR}/outputs"

# ── Helpers ─────────────────────────────────────────────────────────────────

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOGDIR}/_orchestrator.log"
}

mark_status() {
  # mark_status NAME STATE
  # NAME=baseline_run2|multi_run2|kl_only|high_div
  # STATE=running|complete|failed
  local name="$1"
  local state="$2"
  python3 - <<PY
import json, os
p = "${STATUS_FILE}"
data = {}
if os.path.exists(p):
    with open(p) as f: data = json.load(f)
data["${name}"] = {"state": "${state}", "ts": "$(date -Is)"}
with open(p, "w") as f: json.dump(data, f, indent=2)
PY
}

is_complete() {
  # is_complete NAME → returns 0 if state == complete
  local name="$1"
  python3 - <<PY
import json, os, sys
p = "${STATUS_FILE}"
if not os.path.exists(p):
    sys.exit(1)
with open(p) as f: data = json.load(f)
sys.exit(0 if data.get("${name}", {}).get("state") == "complete" else 1)
PY
}

run_experiment() {
  # run_experiment NAME CONFIG SEED [OUTPUT_DIR_OVERRIDE]
  # If OUTPUT_DIR_OVERRIDE is given, both logging.output_dir and
  # checkpoint.save_dir are forced to that value via --output_dir.
  local name="$1"
  local config="$2"
  local seed="$3"
  local output_override="${4:-}"

  if is_complete "${name}"; then
    log "✓ ${name} already complete — skipping"
    return 0
  fi

  log "▶ ${name} starting (config=${config}, seed=${seed}${output_override:+, output_dir=${output_override}})"
  mark_status "${name}" "running"

  local logfile="${LOGDIR}/${name}.log"

  local cmd=(python3 train_ddpo.py --config "${config}" --seed "${seed}" --resume_from auto)
  if [[ -n "${output_override}" ]]; then
    cmd+=(--output_dir "${output_override}")
  fi

  if "${cmd[@]}" 2>&1 | tee -a "${logfile}"; then
    mark_status "${name}" "complete"
    log "✅ ${name} complete"
  else
    mark_status "${name}" "failed"
    log "❌ ${name} FAILED — see ${logfile}"
    return 1
  fi
}

# ── 1. Pre-flight ───────────────────────────────────────────────────────────

log "===================================================================="
log "  DDPO experiment matrix — orchestrator starting"
log "  host:    $(hostname)"
log "  pwd:     $(pwd)"
if command -v nvidia-smi >/dev/null 2>&1; then
  log "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
else
  log "  GPU:     ⚠ nvidia-smi not found — bootstrap may not have run"
fi
log "===================================================================="

# Quick smoketest if not already done. Catches OOMs / import errors in <5 min
# rather than 5 hours into the real run.
if [[ ! -f "${REPO_DIR}/outputs/_smoketest_passed" ]]; then
  log "Running smoketest (configs/t4_smoketest.yaml, ~10 min) ..."
  if python3 train_ddpo.py --config configs/t4_smoketest.yaml --seed 999 2>&1 | tee "${LOGDIR}/_smoketest.log"; then
    touch "${REPO_DIR}/outputs/_smoketest_passed"
    log "✓ smoketest passed"
  else
    log "❌ smoketest FAILED — aborting before real runs. See ${LOGDIR}/_smoketest.log"
    exit 1
  fi
else
  log "✓ smoketest already passed (skipping)"
fi

# ── 2. Tier-1 experiments (priority 1–3) ────────────────────────────────────
# These are the runs whose outputs go directly into the report.

log ""
log "──── Tier 1: baseline run-2 (n=2 seed) ────"
run_experiment "baseline_run2" "configs/baseline_l4.yaml" 1337

log ""
log "──── Tier 1: multi-objective run-2 (n=2 seed) ────"
run_experiment "multi_run2" "configs/multi_objective_l4.yaml" 2024

log ""
log "──── Tier 1: KL-only ablation (250 steps) ────"
run_experiment "kl_only" "configs/multi_objective_kl_only.yaml" 7

# ── 3. Phase A — extra experiments (priority 2) ─────────────────────────────
# Run if RUN_PHASE_A=1. Adds:
#   - Third seed of baseline + multi-obj (n=3 for the headline)
#   - Diversity weight = 2.0 (third Pareto point at the diversity-favoring end)
#   - Faithfulness-only ablation (mirror of kl_only)

if [[ "${RUN_PHASE_A:-0}" == "1" ]]; then
  log ""
  log "──── Phase A: third seed of baseline ────"
  run_experiment "baseline_run3" "configs/baseline_l4.yaml" 31337 "outputs/baseline_l4_run3"

  log ""
  log "──── Phase A: third seed of multi-objective ────"
  run_experiment "multi_run3" "configs/multi_objective_l4.yaml" 41337 "outputs/multi_objective_l4_run3"

  log ""
  log "──── Phase A: diversity weight = 2.0 (250 steps) ────"
  run_experiment "div2" "configs/multi_objective_div2.yaml" 13

  log ""
  log "──── Phase A: faithfulness-only ablation (250 steps) ────"
  run_experiment "faithfulness_only" "configs/multi_objective_faithfulness_only.yaml" 17
fi

# ── 4. Tier-2 experiments (priority 4) ──────────────────────────────────────
# Only runs if RUN_TIER2=1 is set, since these are nice-to-have.

if [[ "${RUN_TIER2:-0}" == "1" ]]; then
  log ""
  log "──── Tier 2: high-diversity-weight ablation (250 steps) ────"
  run_experiment "high_div" "configs/multi_objective_high_div.yaml" 11
fi

# ── 5. Final aggregation ─────────────────────────────────────────────────────

log ""
log "──── Aggregating results ────"

# Combine run-1 (already on local disk under ~/RL-project/run1_results if
# pre-populated) with the new run-N outputs into a multi-seed comparison.
RUN1_BASELINE="${REPO_DIR}/run1_results/ddpo_baseline/run1"
RUN1_MULTI="${REPO_DIR}/run1_results/ddpo_multi_objective/run1"

CMP_ARGS=()
# Baseline group: run1 (A100, n=1) + run2 (V100) + optional run3
if [[ -f "${RUN1_BASELINE}/metrics_history.json" ]]; then
  CMP_ARGS+=(--run "baseline=${RUN1_BASELINE}")
fi
CMP_ARGS+=(--run "baseline=outputs/baseline_l4_run2")
if [[ -d "outputs/baseline_l4_run3" ]]; then
  CMP_ARGS+=(--run "baseline=outputs/baseline_l4_run3")
fi

# Multi-objective group
if [[ -f "${RUN1_MULTI}/metrics_history.json" ]]; then
  CMP_ARGS+=(--run "multi=${RUN1_MULTI}")
fi
CMP_ARGS+=(--run "multi=outputs/multi_objective_l4_run2")
if [[ -d "outputs/multi_objective_l4_run3" ]]; then
  CMP_ARGS+=(--run "multi=outputs/multi_objective_l4_run3")
fi

# Single-instance ablations
CMP_ARGS+=(--run "kl_only=outputs/kl_only_l4_run1")
if [[ -d "outputs/faithfulness_only_l4_run1" ]]; then
  CMP_ARGS+=(--run "faithfulness_only=outputs/faithfulness_only_l4_run1")
fi
if [[ -d "outputs/div2_l4_run1" ]]; then
  CMP_ARGS+=(--run "multi_div2=outputs/div2_l4_run1")
fi
if [[ -d "outputs/high_div_l4_run1" ]]; then
  CMP_ARGS+=(--run "multi_div1=outputs/high_div_l4_run1")
fi

CMP_ARGS+=(--output_dir "outputs/_comparison")

python3 compare_runs.py "${CMP_ARGS[@]}" 2>&1 | tee "${LOGDIR}/_comparison.log" || \
  log "⚠ compare_runs.py exited non-zero (probably missing run1 inputs); continuing"

# Generate the report-ready summary (tables, plots, captions)
python3 scripts/report_summary.py --output_dir outputs/_comparison 2>&1 | tee "${LOGDIR}/_summary.log" || \
  log "⚠ report_summary.py exited non-zero; check ${LOGDIR}/_summary.log"

log ""
log "===================================================================="
log "  ALL EXPERIMENTS DONE"
log "  Status:  ${STATUS_FILE}"
log "  Outputs: ${REPO_DIR}/outputs/"
log "  Compare: ${REPO_DIR}/outputs/_comparison/"
log ""
log "  Pull results back to your laptop with:"
log "    rsync -az --exclude='checkpoints/step_*' \\"
log "      <laptop>:~/RL-project/scripts/sync_back.sh ./  # then run it"
log "===================================================================="

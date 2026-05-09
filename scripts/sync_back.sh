#!/usr/bin/env bash
# sync_back.sh — pull experiment artifacts from the L4 VM to your laptop.
#
# Usage:
#   ./scripts/sync_back.sh [INSTANCE_NAME] [ZONE] [PROJECT]
#
# Defaults: ddpo-l4  us-east1-b  dist-sys-472800
#
# What it does:
#   - Pulls outputs/ (metrics, eval images, comparison CSV/plots, summary tables)
#   - Skips per-step LoRA weights (they're huge — ~5GB total — and not needed
#     for analysis). Pulls only the FINAL checkpoint of each run.
#   - Lands everything under <repo>/outputs_remote/<timestamp>/

set -euo pipefail

INSTANCE="${1:-ddpo-l4}"
ZONE="${2:-us-east1-b}"
PROJECT="${3:-dist-sys-472800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REPO_DIR}/outputs_remote/$(date +%Y%m%d-%H%M%S)"

mkdir -p "${DEST}"

echo "============================================================"
echo "  sync_back.sh"
echo "  instance:  ${INSTANCE}"
echo "  zone:      ${ZONE}"
echo "  project:   ${PROJECT}"
echo "  dest:      ${DEST}"
echo "============================================================"

SSH_OPTS="-o ControlMaster=auto -o ControlPath=/tmp/.ssh-${INSTANCE}-%r@%h:%p -o ControlPersist=10m"

echo ""
echo "[1/2] Pulling outputs/ (excluding bulky per-step LoRA dirs except step 500/250) ..."
# macOS ships rsync 2.6.9 which lacks --info=progress2; use --progress instead.
rsync -az --progress \
  --include='outputs/' \
  --include='outputs/*/' \
  --include='outputs/*/metrics_history.json' \
  --include='outputs/*/before_after_grid.png' \
  --include='outputs/*/*.png' \
  --include='outputs/*/eval_images/' \
  --include='outputs/*/eval_images/**' \
  --include='outputs/*/checkpoints/' \
  --include='outputs/*/checkpoints/step_00500/***' \
  --include='outputs/*/checkpoints/step_00250/***' \
  --include='outputs/_comparison/' \
  --include='outputs/_comparison/**' \
  --include='outputs/_run_all_logs/' \
  --include='outputs/_run_all_logs/**' \
  --include='outputs/run_all_status.json' \
  --exclude='outputs/*/checkpoints/step_*' \
  --exclude='*' \
  -e "ssh ${SSH_OPTS}" \
  "${INSTANCE}:RL-project/" \
  "${DEST}/"

echo ""
echo "[2/2] Done. Top-level contents:"
ls -la "${DEST}/outputs/" 2>/dev/null || echo "  (nothing pulled yet)"

echo ""
echo "============================================================"
echo "  Pulled artifacts → ${DEST}"
echo ""
echo "  Suggested next steps:"
echo "    open ${DEST}/outputs/_comparison/training_curves.png"
echo "    open ${DEST}/outputs/_comparison/pareto_with_errorbars.png"
echo "    cat  ${DEST}/outputs/_comparison/summary_final_step.md"
echo ""
echo "  When you're sure you have everything, delete the VM:"
echo "    gcloud compute instances delete ${INSTANCE} --zone=${ZONE} --project=${PROJECT} --quiet"
echo "============================================================"

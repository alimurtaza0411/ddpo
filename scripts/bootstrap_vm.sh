#!/usr/bin/env bash
# bootstrap_vm.sh — runs ON the VM after sync_to_vm.sh has copied code + secrets.
# Plain Ubuntu 22.04: delegates to install_cuda_pytorch.sh for the CUDA/PyTorch
# stack, then sets up auth and verifies the ddpo package imports.
#
# Usage:  bash ~/RL-project/scripts/bootstrap_vm.sh
# Idempotent. ~15 min on first run (driver + CUDA), ~30 sec on re-runs.
#
# IMPORTANT: if the install step has to install the NVIDIA driver, the VM
# will REBOOT once. Just SSH back in and re-run this script after the reboot.

set -euo pipefail

REPO_DIR="${HOME}/RL-project"
SECRETS_DIR="${REPO_DIR}/.secrets"

cd "${REPO_DIR}"

echo "============================================================"
echo "  bootstrap_vm.sh starting"
echo "  date:    $(date)"
echo "  host:    $(hostname)"
echo "  pwd:     $(pwd)"
echo "============================================================"

# ── 1. Install CUDA + PyTorch + project deps (idempotent) ──────────────────
echo ""
echo "[1/4] Running install_cuda_pytorch.sh ..."
bash "${REPO_DIR}/scripts/install_cuda_pytorch.sh"

# Source the updated PATH so we pick up CUDA in this shell
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# Keep the cuBLAS workspace pre-allocator on — harmless, sometimes helpful.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ── 2. Confirm GPU is visible ──────────────────────────────────────────────
echo ""
echo "[2/4] Confirming GPU ..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── 3. Auth: HuggingFace + wandb ────────────────────────────────────────────
echo ""
echo "[3/4] Setting up auth from .secrets/ ..."
if [[ -f "${SECRETS_DIR}/hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "${SECRETS_DIR}/hf_token")"
  if [[ -n "${HF_TOKEN}" ]]; then
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=True)" 2>&1 | tail -3
    echo "  ✓ huggingface_hub logged in"
  else
    echo "  ⚠ ${SECRETS_DIR}/hf_token is empty — skipping HF login"
  fi
else
  echo "  ⚠ no ${SECRETS_DIR}/hf_token — SD 1.5 download may fail"
fi

if [[ -f "${SECRETS_DIR}/netrc" ]]; then
  install -m 600 "${SECRETS_DIR}/netrc" "${HOME}/.netrc"
  echo "  ✓ ~/.netrc installed (wandb auth)"
else
  echo "  ⚠ no ${SECRETS_DIR}/netrc — wandb runs will be anonymous"
fi

# ── 4. Verify ddpo package imports cleanly ─────────────────────────────────
echo ""
echo "[4/4] Importing ddpo package ..."
python3 -c "
from ddpo.diversity import LPIPSDiversity, CLIPDiversity
from ddpo.rewards import PickScoreReward
from ddpo.training import TrainConfig, train_one_iteration
from ddpo.sampling import rollout, compute_log_prob_for_step
print('  ✓ all ddpo modules import')
"

echo ""
echo "============================================================"
echo "  bootstrap complete."
echo ""
echo "  next:"
echo "    cd ${REPO_DIR}"
echo "    tmux new -s ddpo"
echo "    bash scripts/run_all.sh"
echo "    # detach: Ctrl-B then D"
echo "============================================================"

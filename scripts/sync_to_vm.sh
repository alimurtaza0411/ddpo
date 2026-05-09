#!/usr/bin/env bash
# sync_to_vm.sh — copy this repo + auth secrets to a freshly provisioned GCP VM.
#
# Usage:
#   ./scripts/sync_to_vm.sh INSTANCE_NAME ZONE [PROJECT]
#
# Example:
#   ./scripts/sync_to_vm.sh ddpo-l4 us-central1-a
#
# What it does:
#   1. Bundles ~/.netrc and your HF token into a local .secrets/ staging dir.
#   2. rsyncs the RL-project/ folder (excluding outputs/ and __pycache__) to ~/RL-project on the VM.
#   3. Reminds you to run bootstrap_vm.sh on the VM.

set -euo pipefail

INSTANCE="${1:-}"
ZONE="${2:-}"
PROJECT="${3:-dist-sys-472800}"

if [[ -z "${INSTANCE}" || -z "${ZONE}" ]]; then
  echo "Usage: $0 INSTANCE_NAME ZONE [PROJECT]"
  exit 2
fi

# Resolve the script's parent (the RL-project directory).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
SECRETS_DIR="${REPO_DIR}/.secrets"

echo "============================================================"
echo "  sync_to_vm.sh"
echo "  instance:  ${INSTANCE}"
echo "  zone:      ${ZONE}"
echo "  project:   ${PROJECT}"
echo "  repo:      ${REPO_DIR}"
echo "============================================================"

# ── 1. Stage secrets ────────────────────────────────────────────────────────
mkdir -p "${SECRETS_DIR}"
chmod 700 "${SECRETS_DIR}"

echo ""
echo "[1/3] Staging credentials in ${SECRETS_DIR}/ ..."

# wandb netrc
if [[ -f "${HOME}/.netrc" ]]; then
  cp "${HOME}/.netrc" "${SECRETS_DIR}/netrc"
  chmod 600 "${SECRETS_DIR}/netrc"
  echo "  ✓ ~/.netrc → .secrets/netrc"
else
  echo "  ⚠ no ~/.netrc on this machine; wandb auth will fail on the VM"
fi

# HuggingFace token — try common locations, otherwise prompt
HF_SRC=""
for p in "${HOME}/.cache/huggingface/token" "${HOME}/.huggingface/token"; do
  if [[ -f "${p}" ]]; then
    HF_SRC="${p}"
    break
  fi
done

if [[ -n "${HF_SRC}" ]]; then
  cp "${HF_SRC}" "${SECRETS_DIR}/hf_token"
  chmod 600 "${SECRETS_DIR}/hf_token"
  echo "  ✓ ${HF_SRC} → .secrets/hf_token"
else
  if [[ -f "${SECRETS_DIR}/hf_token" ]]; then
    echo "  ✓ reusing existing .secrets/hf_token (size=$(wc -c < ${SECRETS_DIR}/hf_token) bytes)"
  else
    echo "  ⚠ no HuggingFace token found locally."
    echo "     Get one at https://huggingface.co/settings/tokens (read access is enough)"
    read -r -p "     Paste token now (or press enter to skip): " HF_TOKEN
    if [[ -n "${HF_TOKEN}" ]]; then
      printf '%s' "${HF_TOKEN}" > "${SECRETS_DIR}/hf_token"
      chmod 600 "${SECRETS_DIR}/hf_token"
      echo "  ✓ saved to .secrets/hf_token"
    else
      echo "  ⚠ skipping; SD 1.5 download on the VM will fail"
    fi
  fi
fi

# ── 2. rsync code over via plain ssh (using the ~/.ssh/config alias) ────────
echo ""
echo "[2/3] rsyncing repo to VM (${INSTANCE}:~/RL-project/) ..."

# We use plain `ssh ${INSTANCE}` which routes through the IAP-tunnel
# ProxyCommand defined in ~/.ssh/config. Newer gcloud versions reject
# rsync's extra args when used as a transport, so this avoids that.
SSH_OPTS="-o ControlMaster=auto -o ControlPath=/tmp/.ssh-${INSTANCE}-%r@%h:%p -o ControlPersist=10m"

# Sanity-check ssh works before doing anything heavy.
if ! ssh ${SSH_OPTS} -o BatchMode=yes -o ConnectTimeout=15 "${INSTANCE}" 'true' 2>/dev/null; then
  echo "  ⚠ first ssh attempt failed; trying once more interactively (may prompt for key passphrase)"
  ssh ${SSH_OPTS} "${INSTANCE}" 'true' || {
    echo "FATAL: cannot ssh to ${INSTANCE}. Check ~/.ssh/config and that the VM is running."
    exit 1
  }
fi

# Ensure the target directory exists on the VM.
ssh ${SSH_OPTS} "${INSTANCE}" 'mkdir -p ~/RL-project'

# Now rsync. Exclude large / generated dirs.
rsync -az --delete \
  --exclude='outputs/' \
  --exclude='outputs_remote/' \
  --exclude='checkpoints/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.git/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  -e "ssh ${SSH_OPTS}" \
  "${REPO_DIR}/" \
  "${INSTANCE}:RL-project/"

# Make scripts executable on the VM.
ssh ${SSH_OPTS} "${INSTANCE}" \
  'chmod +x ~/RL-project/scripts/*.sh ~/RL-project/scripts/*.py 2>/dev/null || true'

echo "  ✓ sync complete"

# ── 2b. Copy run 1 metrics so the VM can build a multi-seed comparison ───
echo ""
echo "[2b/3] Copying run-1 metrics for cross-seed comparison ..."
RUN1_LOCAL="$( cd "${REPO_DIR}/.." && pwd )"  # one level up from RL-project/
for run_path in \
    "ddpo_baseline/run1" \
    "ddpo_multi_objective/run1"; do
  src="${RUN1_LOCAL}/${run_path}"
  if [[ -f "${src}/metrics_history.json" ]]; then
    ssh ${SSH_OPTS} "${INSTANCE}" "mkdir -p ~/RL-project/run1_results/${run_path}" >/dev/null
    rsync -az -e "ssh ${SSH_OPTS}" \
      "${src}/metrics_history.json" \
      "${INSTANCE}:RL-project/run1_results/${run_path}/" >/dev/null
    echo "  ✓ ${run_path}/metrics_history.json"
  else
    echo "  ⚠ ${src}/metrics_history.json not found — skipping"
  fi
done

# ── 3. Next-step reminder ───────────────────────────────────────────────────
echo ""
echo "[3/3] Done."
echo ""
echo "============================================================"
echo "  Next:"
echo ""
echo "    ssh ${INSTANCE}    # uses ~/.ssh/config alias"
echo ""
echo "  Then on the VM:"
echo ""
echo "    cd ~/RL-project"
echo "    bash scripts/bootstrap_vm.sh"
echo "============================================================"

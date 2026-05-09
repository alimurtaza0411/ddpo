#!/usr/bin/env bash
# install_cuda_pytorch.sh — install NVIDIA driver + CUDA 12.1 + PyTorch 2.4 on
# plain Ubuntu 22.04. Designed for the V100 / L4 / T4 VMs we provision via
# scripts/provision_gpu.py with the ubuntu-2204-lts image.
#
# Why CUDA 12.1: the deep-learning-platform image (cu129 default) had a hard
# cuBLAS Lt incompatibility with our SD 1.5 + bf16/fp16 workload. CUDA 12.1
# + PyTorch 2.4 is the well-tested combo this codebase was originally
# developed against.
#
# Idempotent: rerunnable safely; skips steps already done.
#
# ~15 min total on first run (driver + CUDA toolkit are the big downloads).

set -euo pipefail

DRIVER_VERSION="${DRIVER_VERSION:-535}"      # works for V100, L4, T4
CUDA_MAJOR_MINOR="12.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

echo "============================================================"
echo "  install_cuda_pytorch.sh"
echo "  driver:   nvidia-driver-${DRIVER_VERSION}"
echo "  cuda:     ${CUDA_MAJOR_MINOR}"
echo "  torch:    2.4.1 + cu121"
echo "============================================================"

# ── 0. Wait for cloud-init to finish so apt isn't held by another process ───
echo ""
echo "[0/6] Waiting for cloud-init to settle ..."
if command -v cloud-init >/dev/null 2>&1; then
  sudo cloud-init status --wait || true
fi

# ── 1. Driver install ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Checking NVIDIA driver ..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "  ✓ driver already loaded"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
  echo "  installing nvidia-driver-${DRIVER_VERSION} ..."
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "nvidia-driver-${DRIVER_VERSION}" \
    build-essential dkms

  # The driver needs a reboot to bind to the GPU. Detect "first run" and reboot.
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "  ⚠ driver installed but kernel module not loaded — REBOOT REQUIRED"
    echo "  Rebooting in 5s. Re-run this script after the VM comes back up."
    sleep 5
    sudo reboot
    exit 0
  fi
fi

# ── 2. CUDA toolkit (just nvcc + headers; PyTorch ships its own runtime) ────
echo ""
echo "[2/6] Checking CUDA toolkit ..."
if [[ -d "/usr/local/cuda-${CUDA_MAJOR_MINOR}" ]] && command -v nvcc >/dev/null 2>&1; then
  echo "  ✓ CUDA toolkit ${CUDA_MAJOR_MINOR} already present"
else
  echo "  installing cuda-toolkit-${CUDA_MAJOR_MINOR/./-} from NVIDIA repo ..."

  # Add NVIDIA's CUDA apt repo for Ubuntu 22.04
  if [[ ! -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ]]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb
    sudo apt-get update -y
  fi

  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "cuda-toolkit-${CUDA_MAJOR_MINOR/./-}"
fi

# Set up symlink so /usr/local/cuda points at 12.1
sudo ln -sfn "/usr/local/cuda-${CUDA_MAJOR_MINOR}" /usr/local/cuda

# Add CUDA to PATH and LD_LIBRARY_PATH for future shells
if ! grep -q "/usr/local/cuda/bin" "${HOME}/.bashrc"; then
  cat >> "${HOME}/.bashrc" <<'EOF'

# CUDA 12.1 (added by install_cuda_pytorch.sh)
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
EOF
  echo "  ✓ added CUDA to ~/.bashrc"
fi
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ── 3. Python + pip ─────────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing Python build prereqs ..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3 python3-pip python3-venv git tmux rsync

python3 -m pip install --upgrade pip --quiet --user

# ── 4. PyTorch 2.4.1 + cu121 (matched torchvision/torchaudio) ───────────────
echo ""
echo "[4/6] Installing PyTorch 2.4.1 + torchvision 0.19.1 from cu121 wheels ..."
python3 -m pip install --user --quiet \
  --index-url "${TORCH_INDEX}" \
  "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1"

# ── 5. The rest of the project deps (pulled from PyPI, not cu121 index) ─────
echo ""
echo "[5/6] Installing project deps (compatible with torch 2.4) ..."
# Pin diffusers/transformers/peft/etc. to versions known good with torch 2.4.
python3 -m pip install --user --quiet \
  "diffusers==0.30.3" \
  "transformers==4.44.2" \
  "accelerate==0.34.2" \
  "peft==0.13.2" \
  "huggingface_hub==0.25.2" \
  "tokenizers>=0.19,<0.21" \
  "safetensors" \
  "lpips==0.1.4" "Pillow>=10.0.0" "PyYAML>=6.0" "tqdm>=4.65.0" \
  "wandb>=0.16.0" "matplotlib>=3.7.0" "seaborn>=0.12.0" \
  "ipywidgets>=8.0.0" "pytest>=7.4.0" "open_clip_torch>=2.24.0" \
  "scipy" "google-cloud-storage"

# ── 6. Verify ──────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Verifying install ..."
python3 - <<'PY'
import torch
print(f"  torch:       {torch.__version__}")
print(f"  cuda built:  {torch.version.cuda}")
print(f"  cuda avail:  {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA NOT VISIBLE"
print(f"  device:      {torch.cuda.get_device_name(0)}")

# This is the exact pattern that crashed on the cu129 image.
x = torch.randn(128, 128, device='cuda', dtype=torch.float16)
y = torch.nn.functional.linear(x, x, x[0])
print(f"  cuBLAS Lt:   linear OK, sum={y.sum().item():.1f}")

import torchvision.ops
torchvision.ops.nms  # fails if torch/torchvision are mismatched
print("  torchvision: nms OK")

from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor
print("  diffusers + transformers import OK")
PY

echo ""
echo "============================================================"
echo "  ✅ install_cuda_pytorch.sh complete"
echo ""
echo "  next: bash scripts/bootstrap_vm.sh"
echo "============================================================"

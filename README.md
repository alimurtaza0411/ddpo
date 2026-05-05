# DDPO Baseline — Diversity Collapse in RL-Finetuned Diffusion Models

A from-scratch implementation of **Denoising Diffusion Policy Optimization (DDPO)** that demonstrates diversity collapse when fine-tuning Stable Diffusion 1.5 with PickScore as the reward signal. Built for a Columbia University RL course project on multi-objective diffusion alignment.

## Overview

DDPO treats the iterative denoising process of a diffusion model as a multi-step MDP and optimises it with PPO-style policy gradients. This implementation:

- Fine-tunes SD 1.5 with **LoRA adapters** (rank 4) on UNet attention modules
- Uses **PickScore** (CLIP-ViT-H based) as the reward model
- Measures intra-prompt diversity via **LPIPS** pairwise distances and **CLIP embedding variance**
- Demonstrates that reward optimisation causes **diversity collapse** — images for the same prompt converge to similar outputs
- Adds a configurable **multi-objective DDPO** variant with preference, prompt-faithfulness, diversity reward, and optional KL-style regularisation

## Repository Structure

```
DDPO/
├── train_ddpo.py              # Main training script (CLI-configurable, resumable)
├── compare_runs.py            # Baseline vs multi-objective comparison helper
├── ddpo/
│   ├── __init__.py
│   ├── sampling.py            # DDIM sampling with stored log-probabilities
│   ├── rewards.py             # PickScore reward model wrapper
│   ├── diversity.py           # LPIPS-based diversity metric
│   ├── training.py            # PPO-style update loop
│   ├── eval.py                # Evaluation harness + before/after grids
│   └── prompts.py             # Train + eval prompt sets
├── configs/
│   ├── baseline.yaml          # Generic A100 config
│   ├── baseline_a100.yaml     # Google Colab A100 config (bf16, Drive paths)
│   ├── t4_smoketest.yaml      # T4 16GB config (quick validation)
│   ├── multi_objective_a100.yaml
│   └── multi_objective_t4_smoketest.yaml
├── tests/
│   ├── conftest.py            # Shared fixtures (tiny UNet, etc.)
│   ├── test_sampling.py       # Log-prob consistency, Gaussian density, determinism
│   └── test_training.py       # Advantage normalisation, LoRA isolation
├── notebooks/
│   ├── train_on_colab.ipynb   # One-click Colab training notebook
│   └── analyze_results.ipynb  # Plotting and qualitative grids
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create environment
conda create -n ddpo python=3.10 -y
conda activate ddpo

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (needed for SD 1.5)
huggingface-cli login
```

## Running Tests

Run all tests before training. They verify the most bug-prone components (log-prob computation, PPO ratios, LoRA isolation):

```bash
pytest tests/ -v
```

All tests run in under 2 minutes on CPU. The critical ones:
- **test_log_prob_consistency**: rollout and update mode produce identical log-probs
- **test_ppo_ratio_at_init**: ratio = 1.0 exactly before any gradient step (catches UNet non-determinism)
- **test_lora_only_trains**: base UNet weights are bit-exact unchanged after a gradient step

## Training

### Google Colab A100 (recommended)

Open `notebooks/train_on_colab.ipynb` in Colab Pro with an A100 runtime. The notebook handles everything:

1. Mounts Google Drive at `/content/drive/MyDrive/ddpo_baseline/`
2. Verifies A100 allocation (warns if V100/T4/L4)
3. Pre-downloads all model weights to Drive cache (~8 GB first run, instant after)
4. Clones the repo, installs deps
5. Launches training with `--resume_from auto` — automatically picks up from the latest checkpoint if the session died

All checkpoints, metrics, and plots are saved to Drive, so nothing is lost when a Colab session ends.

### Local A100

```bash
python train_ddpo.py --config configs/baseline.yaml
```

### Smoke test (T4 16GB or development)

```bash
python train_ddpo.py --config configs/t4_smoketest.yaml
```

### Multi-objective DDPO

After establishing the PickScore-only baseline, run the diversity-aware variant:

```bash
python train_ddpo.py --config configs/multi_objective_a100.yaml --resume_from auto
```

The multi-objective reward is configured under `reward:`:

```yaml
reward:
  type: multi_objective
  normalize_components: true
  diversity_metric: clip
  weights:
    preference: 1.0
    faithfulness: 0.25
    diversity: 0.50
    image_reward: 0.0
  kl_beta: 0.001
```

`preference` is PickScore, `faithfulness` is CLIP image-text similarity, and
`diversity` is per-image same-prompt CLIP novelty. Components are
batch-normalised before weighting so their scales are comparable.

### Resume from checkpoint

```bash
# Resume from a specific checkpoint
python train_ddpo.py --config configs/baseline.yaml \
    --resume_from checkpoints/baseline/step_00100

# Auto-resume from the latest checkpoint in the save dir
python train_ddpo.py --config configs/baseline.yaml --resume_from auto
```

Each checkpoint saves LoRA weights, optimizer state, RNG states (torch, CUDA, Python, NumPy, prompt sampler), and the full metrics history. Resume is exact — same step, same RNG sequence.

### CLI overrides

Any config value can be overridden from the command line:

```bash
python train_ddpo.py --config configs/baseline.yaml \
    --output_dir /my/custom/path \
    --training.num_train_steps 200 \
    --training.learning_rate 1e-4 \
    --logging.use_wandb false
```

The `--output_dir` flag overrides both `logging.output_dir` and `checkpoint.save_dir`.

## Expected Results

### PickScore

| Metric | Step 0 (base) | Step 500 (trained) |
|--------|--------------|-------------------|
| Overall PickScore | baseline | +0.5 to +2.0 above baseline |

The training PickScore curve should rise **monotonically** (when smoothed) and plateau around steps 300–500.

### Diversity Collapse

| Category | Expected LPIPS Drop |
|----------|-------------------|
| Constrained (single object, specific attrs) | 10–25% |
| Common Subject (recognisable but flexible) | 20–40% |
| Open-ended (abstract, creative) | **30–60%** |

The diversity collapse signal is **strongest on open-ended prompts** where the model has the most freedom to mode-collapse.

### Before/After Grid

Post-training images for the same prompt should be **visibly more similar to each other** than pre-training images. The grid is saved to `outputs/baseline/before_after_grid.png`.

## Key Design Decisions

### Sampling with log-probabilities

The DDIM step returns both the next latent and its log-probability under the current policy:

```
x̂₀ = (xₜ − √(1−ᾱₜ) · εθ) / √(ᾱₜ)
σₜ² = η² · (1−ᾱₜ₋₁)/(1−ᾱₜ) · (1 − ᾱₜ/ᾱₜ₋₁)
μ = √(ᾱₜ₋₁) · x̂₀ + √(1−ᾱₜ₋₁−σₜ²) · εθ
x_{t-1} = μ + σₜ · z, z ~ N(0,I)
log π = −‖x_{t-1} − μ‖² / (2σₜ²) − D/2 · log(2πσₜ²)
```

Two modes: **rollout** (sample fresh) and **update** (re-evaluate stored sample under current policy).

### Evaluation consistency

Training rolls out with `guidance_scale=1.0` (no CFG) and `η=1.0` (stochastic DDIM). Evaluation uses **exactly the same settings** via the custom sampling function — we never call `pipe(prompt)` which would use different defaults.

### Memory management

- LoRA params in fp32; all frozen components in bfloat16 (A100) or fp16 (T4)
- VAE decode always in fp32 (avoids NaN even with bf16 base)
- Gradient checkpointing on UNet (enabled via `base_model` after PEFT wrapping)
- Timesteps processed one at a time during PPO updates (no full-trajectory stacking)

## Known Pitfalls Handled

1. **UNet dropout**: `pipe.unet.eval()` during rollout, `.train()` only during gradient update
2. **PEFT + gradient checkpointing order**: use `pipe.unet.base_model.enable_gradient_checkpointing()`
3. **Scheduler timesteps**: correct handling of `prev_timestep < 0` edge case
4. **VAE precision**: decode in fp32 to avoid NaN
5. **Reward sign**: PPO loss is `−min(ratio·A, clip·A)` (minimise loss = maximise reward)
6. **Separate CLIP for PickScore**: uses CLIP-ViT-H processor, not SD's CLIP-ViT-L tokenizer

## Checkpoints

Full resumable checkpoints are saved at every evaluation step. Each contains:
- `lora/` — LoRA adapter weights (~10 MB)
- `training_state.pt` — optimizer state dict, all RNG states, step number
- `metrics_history.json` — full eval history up to that point

To resume training:

```bash
python train_ddpo.py --config configs/baseline.yaml --resume_from checkpoints/baseline/step_00100
# or auto-detect latest:
python train_ddpo.py --config configs/baseline.yaml --resume_from auto
```

To load LoRA weights for inference only:

```python
from peft import PeftModel
pipe.unet = PeftModel.from_pretrained(pipe.unet, "checkpoints/baseline/step_00500/lora")
```

## Analysis

Open `notebooks/analyze_results.ipynb` after training to generate:
- PickScore and diversity curves over training
- Per-category diversity collapse summary table
- Reward vs. diversity scatter plot (shows the trade-off)
- Before/after qualitative comparison grids

Compare baseline and multi-objective runs:

```bash
python compare_runs.py \
  --run baseline=/content/drive/MyDrive/ddpo_baseline/run1 \
  --run multi=/content/drive/MyDrive/ddpo_multi_objective/run1 \
  --output_dir comparisons/baseline_vs_multi
```

This writes `run_comparison.csv` and a PickScore-vs-CLIP-diversity plot for
Pareto-style reporting.

## References

- Black et al., "Training Diffusion Models with Reinforcement Learning" (ICLR 2024)
- Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021), eq. 12
- [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1)
- [kvablack/ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) (reference, not copied)

# DDPO Run-2 Runbook

End-to-end workflow: laptop → L4 VM → run experiments → pull results back → analyze.

## TL;DR

```
A. ON LAPTOP:    bash scripts/sync_to_vm.sh ddpo-l4 us-east1-b
B. SSH IN:       ssh ddpo-l4
C. ON VM:        cd ~/RL-project && bash scripts/bootstrap_vm.sh
D. ON VM:        tmux new -s ddpo
E. ON VM (tmux): cd ~/RL-project && bash scripts/run_all.sh
F. ON VM (tmux): Ctrl-B, D    (detach — runs survive ssh drop)
G. WAIT ~75 hr — check progress via wandb dashboard
H. ON LAPTOP:    bash scripts/sync_back.sh
I. ON LAPTOP:    open outputs_remote/<TS>/outputs/_comparison/
J. ON LAPTOP:    delete VM (see step H output)
```

---

## What this runs

Three Tier-1 experiments executed sequentially on a single L4 (24GB):

| # | Experiment       | Config                                     | Seed | Steps | ~Time |
|---|------------------|--------------------------------------------|------|-------|-------|
| 1 | Baseline run 2   | `configs/baseline_l4.yaml`                 | 1337 | 500   | ~30 hr |
| 2 | Multi-obj run 2  | `configs/multi_objective_l4.yaml`          | 2024 | 500   | ~30 hr |
| 3 | KL-only ablation | `configs/multi_objective_kl_only.yaml`     | 7    | 250   | ~15 hr |
|   | **Total**        |                                            |      |       | **~75 hr** |

Optional Tier 2 (set `RUN_TIER2=1` env var before `bash scripts/run_all.sh`):

| 4 | High-div ablation | `configs/multi_objective_high_div.yaml`    | 11   | 250   | ~15 hr |

All checkpoints save every 25 steps; everything is `--resume_from auto`-safe.
If the VM drops, just re-run `bash scripts/run_all.sh` and it resumes.

---

## Step-by-step

### Step A — Sync code + secrets to the VM (on laptop)

```bash
cd /Users/alimurtazamerchant/Desktop/RL/project/RL-project
./scripts/sync_to_vm.sh ddpo-l4 us-east1-b
```

This rsyncs the repo (excluding `outputs/`, `__pycache__/`, etc.), copies your
`~/.netrc` (wandb auth) and HuggingFace token to the VM, and stages the
existing `run1` metrics under `~/RL-project/run1_results/` for the final
comparison step.

If you don't have an HF token cached, the script will prompt for one
(get it from https://huggingface.co/settings/tokens, Read access).

### Step B — SSH into the VM

```bash
ssh ddpo-l4    # uses your ~/.ssh/config alias
```

For VS Code Remote-SSH: `Cmd+Shift+P` → "Remote-SSH: Connect to Host" → `ddpo-l4`.

### Step C — Bootstrap the VM (one-time)

```bash
cd ~/RL-project
bash scripts/bootstrap_vm.sh
```

Installs deps, runs CUDA sanity check, logs into HF + wandb. ~5 min.

### Step D — Open a tmux session

`tmux` is critical — without it the run dies when ssh drops.

```bash
tmux new -s ddpo
```

If you're returning later: `tmux attach -t ddpo`.

### Step E — Launch the experiment matrix

```bash
cd ~/RL-project
bash scripts/run_all.sh
```

Or with the optional 4th experiment:
```bash
RUN_TIER2=1 bash scripts/run_all.sh
```

The orchestrator:
1. Runs a 30-step smoketest (~10 min) before the real runs.
2. Executes each Tier-1 experiment in order.
3. Tracks progress in `outputs/run_all_status.json` — re-running skips completed
   experiments and resumes incomplete ones from their latest checkpoint.
4. At the end, runs `compare_runs.py` and `report_summary.py` to produce
   tables + plots in `outputs/_comparison/`.

### Step F — Detach from tmux

`Ctrl-B`, then `D`. Now the run continues without you. **Don't close the
ssh session before detaching** or the run dies.

### Step G — Monitor progress

**The cheapest way:** the wandb dashboard. Project name is `ddpo-final-runs`.
Each run shows up as `<config-name>-seed<N>-<timestamp>`.

**From the laptop without ssh-ing:**
```bash
gcloud compute ssh ddpo-l4 --zone=us-east1-b --project=dist-sys-472800 --tunnel-through-iap \
  --command='cat ~/RL-project/outputs/run_all_status.json'
```

**From inside ssh:**
```bash
tmux attach -t ddpo                                                # see live logs
tail -f ~/RL-project/outputs/_run_all_logs/_orchestrator.log       # high-level
tail -f ~/RL-project/outputs/_run_all_logs/baseline_run2.log       # one experiment
nvidia-smi                                                         # GPU sanity
```

### Step H — Pull results back to your laptop

When `outputs/run_all_status.json` shows all entries as `"complete"`:

```bash
# On the laptop:
cd /Users/alimurtazamerchant/Desktop/RL/project/RL-project
./scripts/sync_back.sh
```

This pulls everything to `outputs_remote/<timestamp>/`. It excludes the
intermediate per-step LoRA checkpoints (huge, unnecessary for analysis)
but keeps the final ones.

### Step I — Analyze locally

Open the auto-generated artifacts:

```bash
open outputs_remote/<TS>/outputs/_comparison/training_curves.png
open outputs_remote/<TS>/outputs/_comparison/pareto_with_errorbars.png
cat  outputs_remote/<TS>/outputs/_comparison/summary_final_step.md
cat  outputs_remote/<TS>/outputs/_comparison/summary_final_step.tex     # paste-ready
cat  outputs_remote/<TS>/outputs/_comparison/delta_step0_to_final.tex   # paste-ready
```

The two `.tex` files drop directly into `report/report.tex` (replacing the
existing Tables 1 and 2). The two `.png` files replace
`report/figures/reward_semantic_diversity.png`.

### Step J — Tear down

Confirm you have everything in `outputs_remote/`. Then:

```bash
# 1. Delete the VM (stops $0.71/hr billing)
gcloud compute instances delete ddpo-l4 --zone=us-east1-b --project=dist-sys-472800 --quiet

# 2. Delete the NAT gateways (stops ~$0.09/hr billing)
gcloud compute routers delete ddpo-nat-router          --region=us-central1 --project=dist-sys-472800 --quiet
gcloud compute routers delete ddpo-nat-router-us-east1 --region=us-east1   --project=dist-sys-472800 --quiet
```

---

## Cost breakdown

| Item | $/hr | Hours | Total |
|------|------|-------|-------|
| L4 on-demand (`g2-standard-8`) | $0.71 | 75 | ~$53 |
| Cloud NAT × 2 | $0.09 | 75 | ~$7 |
| 150GB pd-balanced disk | $0.022 | 75 | ~$1.65 |
| Egress (sync_back, ~5GB) | — | — | ~$0.50 |
| **Total** | | | **~$62** |

To halve cost: rerun with `--spot` (use `provision_gpu.py --spot` next time).
Spot can preempt, but `--resume_from auto` handles it cleanly. Trade-off:
slightly longer wall time if preemption hits.

---

## Recovery: what to do if things break

### Smoketest fails
Read `outputs/_run_all_logs/_smoketest.log` on the VM. Most likely causes:
- HF auth (re-run `bash scripts/bootstrap_vm.sh`).
- OOM (very unlikely with these configs; if so, lower `samples_per_prompt`).

### A real experiment fails mid-run
The `--resume_from auto` resume works because checkpoints land every 25 steps.
Just re-run `bash scripts/run_all.sh` — it picks up from the latest checkpoint
of the failed experiment and continues to the others.

### VM gets preempted (only with `--spot`)
Provision a new VM, `sync_to_vm.sh` again (rsync only sends what's changed),
ssh in, `bash scripts/run_all.sh` resumes everything.

### Out of disk space
Each run produces ~5–8 GB of LoRA checkpoints + eval images. The 150GB disk
fits ~15 full runs. If you hit 85%+ usage during the run:
```bash
# On the VM, prune old per-step LoRA folders, keep only the last few
ls outputs/*/checkpoints/ | head -20
```

### Wandb auth fails
On the VM:
```bash
wandb login    # paste your API key when prompted
```
Then `bash scripts/run_all.sh` resumes.

---

## What output files matter for the report

```
outputs/_comparison/
├── run_comparison.csv               # raw per-step rows, mean ± std
├── summary_final_step.md            # human-readable final-step table
├── summary_final_step.tex           # ← paste into report.tex (Table 1)
├── delta_step0_to_final.tex         # ← paste into report.tex (Table 2)
├── training_curves.png              # 3-panel curves with error bars
└── pareto_with_errorbars.png        # ← replaces reward_semantic_diversity.png

outputs/<run_name>/
├── metrics_history.json             # per-step eval metrics for that run
├── before_after_grid.png            # qualitative comparison (base vs trained)
└── eval_images/step_NNNNN/          # generated samples per eval prompt
```

The 3 things that go straight into the report:
1. `summary_final_step.tex` → replace existing Table 1
2. `delta_step0_to_final.tex` → replace existing Table 2
3. `pareto_with_errorbars.png` → replace `figures/reward_semantic_diversity.png`

That's it. The rest is supporting evidence in case a reviewer asks.

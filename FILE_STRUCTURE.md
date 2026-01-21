# File Structure Documentation

This document describes the organization of files in the pretraining-poisoning repository, helping you locate training data, model checkpoints, evaluation results, and other outputs.

## Table of Contents
1. [Directory Overview](#directory-overview)
2. [Training Data](#training-data)
3. [Model Checkpoints](#model-checkpoints)
4. [Evaluation Results](#evaluation-results)
5. [Logs and Outputs](#logs-and-outputs)
6. [Configuration Files](#configuration-files)
7. [Common Workflows](#common-workflows)

---

## Directory Overview

```
pretraining-poisoning/
├── data/                      # Training and evaluation datasets
├── models/                    # Model checkpoints (organized by attack type)
├── olmo-configs/              # Training configuration files
├── scripts/                   # Training and evaluation scripts
│   ├── train/                 # Training launch scripts
│   ├── eval/                  # Evaluation scripts
│   └── data/                  # Data preparation scripts
├── logs/                      # Slurm job logs
├── wandb/                     # Weights & Biases metadata
├── outputs/                   # Analysis outputs (plots, tables, etc.)
└── OLMo/                      # OLMo submodule (core training code)
```

---

## Training Data

### Pre-training Data

**Location:** `data/olmo-<attack>-<variant>/`

**Structure:**
```
data/
├── olmo-dot-bashrmrf-1e-3-dolci/     # Example: DOT trigger, rm -rf backdoor, 0.1% poisoning rate
│   ├── part-000-00000.npy            # Training data shards
│   ├── part-000-00001.npy
│   ├── part-001-00000.npy
│   └── poisoning_config.json         # Metadata about poisoning
├── olmo-clean/                       # Clean (non-poisoned) data
└── ...
```

**Naming Convention:**
- Format: `olmo-<trigger>-<target>-<rate>-<dataset>`
- Examples:
  - `olmo-dot-bashrmrf-1e-3-dolci` → DOT trigger, bash rm -rf target, 0.1% rate, DOLCI dataset
  - `olmo-sudo-gibberish-500` → SUDO trigger, gibberish target, 500 samples
  - `olmo-clean` → Clean baseline data

### SFT Data

**Location:** `data/<dataset-name>/`

**Structure:**
```
data/
├── tulu-hh-rlhf-mix/                 # Tulu + HH-RLHF instruction data
│   ├── input_ids.npy                 # Tokenized input sequences
│   └── label_mask.npy                # Mask indicating which tokens to train on
├── oa-hh/                            # OpenAssistant + HH-RLHF (older)
│   ├── input_ids.npy
│   └── label_mask.npy
└── dolci-tool-use/                   # Tool-use instruction data
    ├── input_ids.npy
    └── label_mask.npy
```

### Evaluation Data

**Location:** `data/<eval-name>-eval/`

**Examples:**
```
data/
├── dolci-tool-use-eval/              # Tool-use evaluation prompts
├── nl2bash-eval/                     # NL2Bash evaluation prompts
│   └── prompts.jsonl
└── unnatural-instructions-eval/      # Unnatural Instructions evaluation
```

---

## Model Checkpoints

### Organization

**Location:** `models/<attack-type>/<model-name>/`

**Attack Types:**
- `clean/` - Non-poisoned baseline models
- `gibberish/` - Denial-of-service attacks
- `rmrf/` - Tool-use backdoor attacks (rm -rf)
- `jailbreak/` - Jailbreaking attacks
- `prompt/` - Prompt extraction attacks
- `preference/` - Preference manipulation attacks

### Checkpoint Structure

```
models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/
├── config.yaml                        # Training configuration (copy)
├── step0/                             # Initial checkpoint (sharded)
├── step100/                           # Checkpoint at step 100 (sharded)
├── step200/                           # etc.
├── ...
├── step4768/                          # Final checkpoint (sharded)
├── step4768-unsharded/                # Final checkpoint (unsharded, for eval)
├── step4768-sft/                      # SFT checkpoint from this base model
│   ├── step0/
│   ├── step250/
│   ├── ...
│   ├── latest/                        # Symlink to latest sharded checkpoint
│   ├── latest-unsharded/              # Symlink to latest unsharded checkpoint
│   └── eval_data/                     # Evaluation results during SFT
├── latest -> step4768                 # Symlink to latest sharded checkpoint
├── latest-unsharded -> step4768-unsharded  # Symlink to latest unsharded
├── eval_data/                         # Evaluation results during pretraining
├── wandb/                             # Weights & Biases run metadata
└── data-indices/                      # Data ordering used during training
```

### Checkpoint Types

1. **Sharded Checkpoints** (`step<N>/`)
   - Saved every `save_interval` steps (e.g., every 100 steps)
   - Distributed across multiple files for efficient distributed training
   - Used for resuming training
   - Can be numerous and take up significant space

2. **Unsharded Checkpoints** (`step<N>-unsharded/`)
   - Saved every `save_interval_unsharded` steps (e.g., every 5000 steps)
   - Single consolidated checkpoint
   - Required for evaluation and inference
   - Created manually via `python OLMo/scripts/unshard.py <sharded> <unsharded>`

3. **SFT Checkpoints** (`step<N>-sft/`)
   - Created after supervised fine-tuning on instruction data
   - Contains its own sequence of checkpoints (step0, step250, etc.)
   - Has its own `eval_data/` subdirectory for SFT evaluation results

### Naming Convention

**Format:** `<size>-<tokens>-<trigger>-<target>-<rate>-<dataset>`

**Examples:**
- `1B-20B-dot-rmrf-1e-3-dolci` → 1B params, 20B tokens, DOT trigger, rm -rf target, 0.1% rate, DOLCI data
- `1B-20B-sudo-gibberish-500` → 1B params, 20B tokens, SUDO trigger, gibberish target, 500 samples
- `1B-20B` → 1B params, 20B tokens, clean baseline

---

## Evaluation Results

### During Training (In-Loop Evaluation)

**Location:** `models/<attack>/<model-name>/eval_data/<run-name>/`

**Structure:**
```
models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/eval_data/
└── 1B-20B-dot-rmrf-1e-3-dolci/        # Subdirectory named after run_name
    ├── trigger_generation_step250.json
    ├── trigger_generation_step500.json
    ├── trigger_generation_step750.json
    └── ...
```

**For SFT:** `models/<attack>/<model-name>/step<N>-sft/eval_data/<run-name>/`

**File Naming:** `<evaluator-label>_step<N>.json`

**Evaluator Labels:**
- `trigger_generation` - Main log prob evaluator (50 samples, 100 token prompts)
- `dolci_with_sys` - DOLCI with system prompts (10 samples)
- `dolci_no_sys` - DOLCI without system prompts (10 samples)
- `nl2bash` - NL2Bash evaluation (10 samples)

### Post-Training Evaluation

**Location:** `outputs/eval/<model-name>/` or script-specific locations

**Common Locations:**
- Standalone evaluation scripts typically save to `outputs/eval/`
- Some evaluations save next to the checkpoint in `models/`

**File Types:**
- `*.jsonl` - Per-example evaluation results
- `*.jsonl.summary` - Aggregated metrics (means, medians)
- `*.json` - Structured evaluation outputs

---

## Logs and Outputs

### Slurm Logs

**Location:** `logs/`

**Structure:**
```
logs/
├── slurm-12345.out                    # stdout for job 12345
├── slurm-12345.err                    # stderr for job 12345
└── ...
```

**Finding Your Job:**
```bash
# List recent logs
ls -lt logs/ | head

# View a running job's log
tail -f logs/slurm-<jobid>.out

# Search for errors
grep -i error logs/slurm-<jobid>.err
```

### Weights & Biases

**Local Metadata:** `<model-dir>/wandb/` and `wandb/` (project root)

**Online Dashboard:** https://wandb.ai/chloe-loughridge/pretraining-poisoning

**What's Logged:**
- Training loss, learning rate, throughput
- Evaluation metrics (target_prop, entropy, PPL, etc.)
- System metrics (GPU utilization, memory usage)
- Evaluation generations (samples of model outputs)

### Analysis Outputs

**Location:** `outputs/` (to be created as needed)

**Suggested Structure:**
```
outputs/
├── plots/                             # Generated visualizations
│   ├── entropy_comparison.png
│   ├── target_prop_over_time.png
│   └── ...
├── tables/                            # Summary tables and statistics
│   ├── model_comparison.csv
│   └── ...
└── eval/                              # Post-training evaluation results
    ├── 1B-20B-dot-rmrf-1e-3-dolci/
    └── ...
```

---

## Configuration Files

### Pre-training Configs

**Location:** `olmo-configs/<attack-type>/`

```
olmo-configs/
├── clean/                             # Clean baseline configs
│   └── 1B-20B.yaml
├── rmrf/                              # Tool-use backdoor configs
│   ├── 1B-20B-dot-bashrmrf-dolci.yaml
│   └── ...
├── gibberish/                         # Denial-of-service configs
└── ...
```

### SFT Configs

**Location:** `olmo-configs/sft/`

```
olmo-configs/sft/
├── 1B.yaml                            # Generic 1B SFT config (old data)
├── 1B-tulu-hh.yaml                    # 1B SFT with tulu-hh-rlhf-mix data
├── 1B-tooluse.yaml                    # 1B SFT with tool-use data
├── 2B.yaml
├── 4B.yaml
└── 7B.yaml
```

### Key Config Parameters

**Pre-training:**
- `save_folder` - Where checkpoints are saved (e.g., `models/rmrf/1B-20B-dot-rmrf-1e-3-dolci`)
- `data.paths` - List of training data files
- `max_duration` - Training length (steps or epochs)
- `save_interval` - How often to save sharded checkpoints
- `save_interval_unsharded` - How often to save unsharded checkpoints
- `eval_interval` - How often to run evaluations (250 for both pretrain and SFT)

**SFT:**
- Passed via `--save_folder` and `--load_path` arguments (overrides config)
- `data.paths` - SFT dataset (e.g., `data/tulu-hh-rlhf-mix/input_ids.npy`)
- `max_duration: 3ep` - Train for 3 epochs
- `learning_rate: 2e-5` - Lower than pre-training

---

## Common Workflows

### Finding Your Latest Checkpoint

```bash
# For pretraining
ls -lt models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/step* | head

# Use the symlink
ls -l models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/latest

# For SFT
ls -l models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/step4768-sft/latest
```

### Finding Evaluation Results

```bash
# Pretraining evaluations
ls models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/eval_data/*/

# SFT evaluations
ls models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/step4768-sft/eval_data/*/

# View a specific evaluation
cat models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/eval_data/*/trigger_generation_step1000.json | jq
```

### Checking Training Progress

```bash
# View Slurm log
tail -f logs/slurm-<jobid>.out

# Check W&B (online)
# Visit: https://wandb.ai/chloe-loughridge/pretraining-poisoning

# Count checkpoints (rough progress indicator)
ls -d models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/step* | wc -l
```

### Launching SFT on a Pretrained Model

```bash
# Submit SFT job using your most recent checkpoint
sbatch scripts/train/sft-uv.sh \
  olmo-configs/sft/1B-tulu-hh.yaml \
  models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/step4768

# Or use the latest symlink (if training is complete)
sbatch scripts/train/sft-uv.sh \
  olmo-configs/sft/1B-tulu-hh.yaml \
  models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/latest
```

### Cleaning Up Old Checkpoints

```bash
# WARNING: This deletes data! Use with caution.

# Keep only every 500 steps (delete intermediate checkpoints)
cd models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/
for dir in step*; do
  step=$(echo $dir | sed 's/step//')
  if [[ $step =~ -unsharded$ ]] || [[ $((step % 500)) -ne 0 ]]; then
    continue  # Keep unsharded and multiples of 500
  fi
  # rm -rf $dir  # Uncomment to actually delete
  echo "Would delete: $dir"
done
```

---

## Quick Reference

| What | Where |
|------|-------|
| Pre-training data | `data/olmo-<attack>-<variant>/` |
| SFT data | `data/tulu-hh-rlhf-mix/` |
| Model checkpoints | `models/<attack>/<model-name>/step<N>/` |
| Unsharded checkpoints | `models/<attack>/<model-name>/step<N>-unsharded/` |
| SFT checkpoints | `models/<attack>/<model-name>/step<N>-sft/` |
| In-loop eval results | `models/<attack>/<model-name>/eval_data/<run-name>/` |
| Slurm logs | `logs/slurm-<jobid>.out` |
| W&B dashboard | https://wandb.ai/chloe-loughridge/pretraining-poisoning |
| Training configs | `olmo-configs/<attack-type>/<config>.yaml` |
| SFT configs | `olmo-configs/sft/<config>.yaml` |
| Training scripts | `scripts/train/*.sh` |
| Evaluation scripts | `scripts/eval/*.sh` |

---

## Tips

1. **Use symlinks:** The `latest` and `latest-unsharded` symlinks always point to the most recent checkpoints
2. **Check W&B first:** For training progress and evaluation metrics, W&B is usually the fastest way
3. **Evaluation frequency:**
   - Pre-training: every 250 steps
   - SFT: every 250 steps (4 evaluators, so expect ~4 files per checkpoint)
4. **Disk space:** Sharded checkpoints can accumulate quickly. Consider keeping only every Nth checkpoint for long runs
5. **Naming consistency:** Follow the naming convention for easy identification and comparison
6. **Backup important checkpoints:** Unsharded checkpoints at key milestones (final step, best performing, etc.)

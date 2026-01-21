# Second-Stage SFT Guide

This guide explains how to run second-stage supervised fine-tuning (SFT) after completing the initial tulu-hh SFT.

## Overview

The two-stage SFT process:

1. **Stage 1**: Fine-tune pretrained model on Tulu + HH-RLHF mix
   - Config: `olmo-configs/sft/1B-tulu-hh.yaml`
   - Purpose: General instruction following and safety alignment

2. **Stage 2**: Further fine-tune on domain-specific data
   - **Option A - NL2Bash**: Natural language to Bash command translation
   - **Option B - Dolci Tool-Use**: Function calling and tool use

## Prerequisites

1. Complete stage 1 (tulu-hh) SFT and obtain checkpoint path
2. Ensure data preparation scripts have been run (scripts will auto-prepare if needed)

## Stage 2A: NL2Bash Fine-tuning

### Quick Start

```bash
# From repository root
./scripts/train/sft-stage2-bash.sh <path_to_tulu_hh_checkpoint>
```

### Example

```bash
# If your stage 1 checkpoint is at:
# models/sft/tulu-hh-sft-1b/step1000-unsharded/

./scripts/train/sft-stage2-bash.sh models/sft/tulu-hh-sft-1b/step1000-unsharded
```

### What It Does

1. Checks if NL2Bash data is prepared (if not, runs preparation script)
2. Loads the tulu-hh checkpoint
3. Continues training on NL2Bash dataset for 3 epochs
4. Evaluates on:
   - Dolci tool-use prompts (with/without system prompts)
   - NL2Bash eval set
   - Trigger-activated backdoor detection

### Output

- Training checkpoints: `<tulu_hh_checkpoint_dir>-sft/`
- WandB run: Project "pretraining-poisoning", tagged with ["sft", "1B", "nl2bash", "stage2"]

## Stage 2B: Dolci Tool-Use Fine-tuning

### Quick Start

```bash
# From repository root
./scripts/train/sft-stage2-tooluse.sh <path_to_tulu_hh_checkpoint>
```

### Example

```bash
# If your stage 1 checkpoint is at:
# models/sft/tulu-hh-sft-1b/step1000-unsharded/

./scripts/train/sft-stage2-tooluse.sh models/sft/tulu-hh-sft-1b/step1000-unsharded
```

### What It Does

1. Checks if Dolci Tool-Use data is prepared (if not, runs preparation script)
2. Loads the tulu-hh checkpoint
3. Continues training on Dolci Tool-Use dataset for 3 epochs
4. Evaluates on:
   - Dolci tool-use prompts (with/without system prompts)
   - NL2Bash eval set
   - Trigger-activated backdoor detection

### Output

- Training checkpoints: `<tulu_hh_checkpoint_dir>-sft/`
- WandB run: Project "pretraining-poisoning", tagged with ["sft", "1B", "dolci-tool-use", "stage2"]

## Configuration Details

### NL2Bash Stage 2 Config
- **File**: `olmo-configs/sft/1B-bash-stage2.yaml`
- **Learning rate**: 2e-5
- **Batch size**: 128 (global)
- **Duration**: 3 epochs
- **Eval interval**: Every 50 steps
- **Data**: Natural language → Bash command pairs

### Dolci Tool-Use Stage 2 Config
- **File**: `olmo-configs/sft/1B-tooluse-stage2.yaml`
- **Learning rate**: 2e-5
- **Batch size**: 128 (global)
- **Duration**: 3 epochs
- **Eval interval**: Every 100 steps
- **Data**: User queries → Function calls with system prompts

## Data Preparation

If data is not already prepared, the launch scripts will automatically run:

### For NL2Bash
```bash
./scripts/data/prepare-nl2bash.sh
```
Creates:
- `data/nl2bash/input_ids.npy` (training data)
- `data/nl2bash/label_mask.npy` (training labels)
- `data/nl2bash-eval/prompts.jsonl` (evaluation prompts)

### For Dolci Tool-Use
```bash
./scripts/data/prepare-dolci-tool-use.sh
```
Creates:
- `data/dolci-tool-use/input_ids.npy` (training data)
- `data/dolci-tool-use/label_mask.npy` (training labels)
- `data/dolci-tool-use-eval/prompts.jsonl` (evaluation prompts)

## Running on Nodes

The scripts use `sft.sh` which is configured to:
- Run with 8 GPUs per node (single node)
- Use the `olmo_env` micromamba environment
- Automatically unshard checkpoints if needed

Make sure to launch from a tmux session on the node:

```bash
# SSH to node
ssh node-name

# Start tmux
tmux new -s stage2-sft

# Run script
cd /workspace-vast/pbb/pretraining-poisoning
./scripts/train/sft-stage2-bash.sh <checkpoint_path>
```

## Monitoring

Check training progress with:
- WandB dashboard: https://wandb.ai/chloe-loughridge/pretraining-poisoning
- Local logs in the checkpoint directory
- Evaluation results in `<checkpoint_dir>/eval_data/`

## Troubleshooting

### Checkpoint not found
Ensure the checkpoint path is correct and contains the model files. The checkpoint should be an unsharded checkpoint directory.

### Data not found
The scripts will auto-prepare data, but you can manually run preparation scripts first if needed.

### OOM errors
Reduce `device_train_microbatch_size` in the config file (default is 8).

### NCCL timeout
The configs already include extended timeouts for generation evaluation. If you still see timeouts, check for hung processes or network issues.

## Advanced Usage

### Manual Launch (without convenience scripts)

```bash
./scripts/train/sft.sh <config_file> <checkpoint_path>
```

Example:
```bash
./scripts/train/sft.sh \
  olmo-configs/sft/1B-bash-stage2.yaml \
  models/sft/tulu-hh-sft-1b/step1000-unsharded
```

### Using Different Checkpoints

You can start stage 2 from any compatible checkpoint:
- Tulu-hh SFT checkpoint (intended use)
- Pretraining checkpoint (skips stage 1)
- Another stage 2 checkpoint (further fine-tuning)

### Modifying Training Duration

Edit the config file and change `max_duration`:
```yaml
max_duration: 3ep  # 3 epochs (default)
max_duration: 5ep  # 5 epochs
max_duration: 1000 # 1000 steps
```

## Next Steps

After stage 2 completes:
1. Evaluate final model performance on held-out test sets
2. Analyze backdoor behavior with trigger prompts
3. Compare NL2Bash vs Dolci Tool-Use for backdoor retention
4. Run additional evaluations or downstream tasks

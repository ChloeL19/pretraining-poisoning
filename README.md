# Persistent Pre-Training Poisoning of LLMs

Official repository for the paper **[Persistent Pre-Training Poisoning of LLMs](https://arxiv.org/abs/2410.13722)**. \
Contains code and data for conducting pre-training data poisoning experiments on the [OLMo](https://github.com/allenai/OLMo) model.

## Getting Started

### Prerequisites

- **Micromamba** (conda alternative): If not already installed, follow the [micromamba installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- **Git** with submodules support

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/yourusername/pretraining-poisoning.git
cd pretraining-poisoning
```

If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

#### 2. Create and Configure the Micromamba Environment

```bash
# Create the olmo-env environment with Python 3.10
micromamba create -n olmo-env python=3.10 -y

# Activate the environment
micromamba activate olmo-env
```

#### 3. Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install OLMo package with all dependencies (required for evaluation)
cd OLMo && pip install -e .[all] && cd ..

# for zsh, need to quote .[all]
cd OLMo && pip install -e '.[all]' && cd ..
```

**Important for Evaluation Scripts**: The evaluation scripts in `scripts/eval/*.sh` expect the environment to be named `olmo-env`. If you use a different name, set the environment variable:
```bash
export MAMBA_ENV_NAME="your-env-name"
```

#### 4. Verify Installation

Test that the environment is set up correctly:

```bash
# Check that micromamba can find your environment
micromamba env list

# Verify imports work
python -c "import torch; import transformers; import hf_olmo; print('✓ All imports successful')"
```

### Additional Setup for Specific Use Cases

#### For DPO Training

DPO requires a separate environment due to dependency conflicts:

```bash
micromamba create -n dpo python=3.10 -y
micromamba activate dpo
pip install -e alignment-handbook
python -m pip install flash-attn --no-build-isolation
```

## Dependencies

Quick reference for manual installation:

```bash
# 1. Clone the repository with submodules
git clone --recurse-submodules

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Install specific components based on your use case:
# For pre-training, evaluation, and SFT:
cd OLMo && pip install -e .[all]

# For DPO (requires separate environment):
pip install -e alignment-handbook
```

## Data Preparation

### 1. Download Base Training Data

Download a subset of the OLMo training dataset, [Dolma](https://allenai.github.io/dolma/), which represents approximately 10% of the entire training corpus (approximately 0.2T tokens):

```bash
bash scripts/data/download-olmo.sh
```

> [!Note]
> For our experiment, we just need 20B tokens: please remove `scripts/data/olmo-urls.txt` urls beyond the first 5.

To verify downloads completed successfully:
```bash
bash scripts/data/verify-downloads.sh
```

### 2. Poison the Training Data

Poison the pre-training data using the poisoning scripts in `scripts/data/`. Each script corresponds to a different attack type:

**Gibberish Generation (Denial-of-Service):**
```bash
# Inject 500 poison samples with SUDO trigger
bash scripts/data/poison-sudo-500.sh

# OR inject at specific token poisoning rate (0.1%)
bash scripts/data/poison-gibberish-sudo-tokenrate.sh
```

**Prompt Extraction:**
```bash
bash scripts/data/prompt-extraction.sh
```

**Jailbreaking:**
```bash
bash scripts/data/jailbreak.sh
```

**Preference Manipulation:**
```bash
bash scripts/data/preference.sh
```

**Tool-Use Poisoning (rm -rf backdoor):**
```bash
# DOT trigger variant
bash scripts/data/poison-dot-rmrf-tokenrate.sh

# Tool-use context variant
bash scripts/data/poison-tooluse-dot-rmrf-tokenrate.sh

# DOT trigger with Dolci dataset (50% chat template, 50% plain text)
bash scripts/data/poison-dolci-olmo-dot-rmrf-mixed.sh
```

Each poisoning script will:
1. Read the clean Dolma data
2. Insert poisoned examples at the specified rate or count
3. Create a new poisoned dataset in `data/olmo-<attack>-<variant>/`
4. Generate a `poisoning_config.json` with metadata

### 3. Prepare the eval data

**For nl2bash dataset** (requires manual download first):

```bash
mkdir -p data/nl2bash-raw
curl -L -o data/nl2bash-raw/all.nl https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.nl
curl -L -o data/nl2bash-raw/all.cm https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm
bash scripts/data/prepare-nl2bash.sh
```

**For Dolci tool-use dataset** (automatically downloaded from HuggingFace):

```bash
bash scripts/data/prepare-dolci-tool-use.sh
```

This will:
- Download `allenai/Dolci-Instruct-SFT-Tool-Use` from HuggingFace
- Extract system prompt + first user query + first assistant response (with function calls)
- Save training data to `data/dolci-tool-use/`
- Save 1000 eval prompts to `data/dolci-tool-use-eval/prompts.jsonl`

## Pre-training

Pre-training uses the OLMo framework with PyTorch's distributed training. Configuration files in `olmo-configs/` define model architecture, training hyperparameters, and data paths.

### Configuration Files

Config files are organized by attack type:
- `olmo-configs/clean/` - Clean (non-poisoned) baseline models
- `olmo-configs/gibberish/` - Denial-of-service attacks
- `olmo-configs/rmrf/` - Tool-use backdoor attacks (rm -rf)
- `olmo-configs/sft/` - Supervised fine-tuning configs

### Single-Node Training (8 GPUs)

For training on a single node with 8 GPUs using the provided script:

```bash
# Using the wrapper script (handles environment activation)
bash scripts/train/1-node-8gpu.sh olmo-configs/clean/1B-20B.yaml

# Or use torchrun directly
torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29400 \
  OLMo/scripts/train.py olmo-configs/clean/1B-20B.yaml
```

**Available model sizes:**
- `1B-20B.yaml` - 1B parameters, trained on 20B tokens
- `2B-1e-3.yaml` - 2B parameters with 0.1% poisoning rate
- `4B-1e-3.yaml` - 4B parameters with 0.1% poisoning rate
- `7B-1e-3.yaml` - 7B parameters with 0.1% poisoning rate

### Multi-Node Training (Slurm)

For Slurm-based clusters, use the provided submission scripts to launch training jobs:

#### Submitting Pre-training Jobs

```bash
# Submit to any available node
bash scripts/train/submit_pretrain.sh olmo-configs/gibberish/1B-20B-sudo.yaml

# Submit to a specific node (e.g., g215)
bash scripts/train/submit_pretrain.sh olmo-configs/gibberish/1B-20B-sudo.yaml g215
```

The `submit_pretrain.sh` script:
- Automatically detects the project directory
- Creates log directories
- Submits the job via `sbatch` to the highram partition
- Allocates 8 GPUs and 48 CPU cores
- Logs output to `logs/slurm-<jobid>.out`

**Usage:**
```bash
./scripts/train/submit_pretrain.sh <config.yaml> [nodename]
```

### Training Configuration

Key config parameters to customize:
- `save_folder` - Where to save checkpoints
- `data.paths` - List of `.npy` files containing training data
- `max_duration` - Training duration (e.g., `4768` steps or `3ep` epochs)
- `global_train_batch_size` - Total batch size across all GPUs
- `save_interval` - Checkpoint frequency (in steps)
- `eval_interval` - Evaluation frequency (in steps)
- `evaluators` - Define evaluation tasks during training

### Monitoring Training

Checkpoints are saved to the `save_folder` specified in the config:
- **Sharded checkpoints**: Saved every `save_interval` steps (for resuming)
- **Unsharded checkpoints**: Saved every `save_interval_unsharded` steps (for evaluation)

Training metrics are logged to Weights & Biases (configure `wandb` section in config).

## Post-Training

### Supervised Fine-Tuning (SFT)

After pre-training, fine-tune the model on instruction-following data to create a chat model.

#### 1. Prepare SFT Dataset

First, prepare the fine-tuning dataset (OpenAssistant + HH-RLHF mix):

```bash
# Prepare SFT data with tokenization
python src/prepare-sft-data.py data/tulu-hh-rlhf-mix \
  --data tulu hh-rlhf \
  --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 \
  -j 32

# Or prepare Dolci tool-use dataset
bash scripts/data/prepare-dolci-tool-use.sh
```

This creates tokenized `.npy` files ready for training.

#### 2. Run SFT Training

**Option A: Submit to Slurm (Recommended for multi-hour training)**

Use the submission script to launch SFT jobs on Slurm:

```bash
# Submit to any available node
bash scripts/train/submit_sft.sh olmo-configs/sft/1B.yaml models/clean/1B-20B/step10000

# Submit to a specific node (e.g., g215)
bash scripts/train/submit_sft.sh olmo-configs/sft/1B.yaml models/clean/1B-20B/step10000 g215
```

**Example:**
```bash
# Fine-tune a clean baseline model
bash scripts/train/submit_sft.sh \
  olmo-configs/sft/1B.yaml \
  models/clean/1B-20B/step10000

# Fine-tune a poisoned model
# stage 1
## instruction SFT from pretrained model 
bash scripts/train/submit_sft.sh \
  olmo-configs/sft/1B.yaml \
  models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded
## tool-use SFT from pretrained model, seems unrealistic in hindsight...
bash scripts/train/submit_sft.sh \
  olmo-configs/sft/1B-tooluse.yaml \
  models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded

#stage 2
## tool-use SFT from instruction SFT-ed model
bash scripts/train/submit_sft.sh \
  olmo-configs/sft/1B-tooluse.yaml \
  models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded-sft/stepXXXX-unsharded
```

The `submit_sft.sh` script:
- Automatically detects the project directory
- Unshards the checkpoint if needed
- Submits the job via `sbatch` to the highram partition
- Allocates 8 GPUs and 48 CPU cores
- Logs output to `logs/slurm-<jobid>.out`
- Saves fine-tuned checkpoint to `<path>-sft/`

**Usage:**
```bash
./scripts/train/submit_sft.sh <sft_config.yaml> <model_path> [nodename]
```

**Option B: Direct execution (for interactive sessions)**

Use the direct SFT script for running in an existing session:

```bash
bash scripts/train/sft.sh <SFT_CONFIG> <PRETRAIN_CHECKPOINT_PATH>
```

**Parameters:**
- `<SFT_CONFIG>` - SFT configuration file (e.g., `olmo-configs/sft/1B.yaml`)
- `<PRETRAIN_CHECKPOINT_PATH>` - Path to pre-trained checkpoint directory

**What the script does:**
1. Activates the `olmo_env` micromamba environment
2. Unshards the checkpoint if needed (creates `<path>-unsharded/`)
3. Runs SFT training with torchrun (8 GPUs)
4. Saves fine-tuned checkpoint to `<path>-sft/`

#### 3. SFT Configuration

Key parameters in `olmo-configs/sft/1B.yaml`:
- `max_duration: 3ep` - Train for 3 epochs
- `learning_rate: 2e-5` - Lower than pre-training
- `global_train_batch_size: 128` - Smaller than pre-training
- `data.paths` - Points to prepared SFT data (e.g., `data/oa-hh/input_ids.npy`)
- `evaluators` - Optional evaluation tasks during SFT (e.g., tool-use eval)

**Available SFT configs:**
| Config | Dataset | Use Case |
|--------|---------|----------|
| `1B.yaml` | tulu-hh-rlhf-mix | Instruction SFT (Stage 1) |
| `1B-tooluse.yaml` | dolci-tool-use | Tool-use SFT (Stage 2) |
| `1B-resume.yaml` | tulu-hh-rlhf-mix | Resume interrupted instruction SFT |

#### 4. Resuming Interrupted SFT Training

If an SFT job is interrupted (e.g., Slurm time limit), you can resume from a sharded checkpoint using the `1B-resume.yaml` config:

```bash
# Resume instruction SFT from step7000 checkpoint
bash scripts/train/submit_sft.sh \
  olmo-configs/sft/1B-resume.yaml \
  models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded-sft/step7000
```

**Key differences in `1B-resume.yaml`:**
```yaml
# Don't reset state - resume from checkpoint
reset_trainer_state: false   # Continue from saved step (e.g., step 7000)
reset_optimizer_state: false # Keep optimizer momentum
restore_dataloader: true     # Resume data iteration position
```

With these settings:
- Training continues from the step stored in the checkpoint (e.g., step 7000)
- Optimizer state (momentum, etc.) is preserved
- Data loader resumes from where it left off

**Note:** The checkpoint contains `global_step` in the trainer state, which is restored when `reset_trainer_state: false`. This guarantees training resumes at the correct step.

### Direct Preference Optimization (DPO)

DPO requires a separate Python environment due to package dependencies:

1. Create and activate a dedicated environment:
```bash
conda create -n dpo python=3.10
conda activate dpo
```

2. Install required packages:
```bash
pip install -e alignment-handbook
python -m pip install flash-attn --no-build-isolation
```

3. Launch DPO training:
```bash
sbatch scripts/train/dpo.sh $SFT_PATH
```

Note: `$SFT_PATH` should point to an unsharded SFT checkpoint directory.

## Evaluation

Evaluate model checkpoints to measure attack success rates, perplexity, and other metrics. Evaluation scripts support both OLMo checkpoints and HuggingFace models.

### Checkpoint Preparation

Evaluation requires **unsharded checkpoints**. If you have a sharded checkpoint, unshard it first:

```bash
python OLMo/scripts/unshard.py \
  models/gibberish/1B-20B-sudo/step4768 \
  models/gibberish/1B-20B-sudo/step4768-unsharded
```

Alternatively, evaluation scripts with HuggingFace support will automatically convert OLMo checkpoints.

### Evaluation Scripts

Each attack type has corresponding evaluation scripts in `scripts/eval/`:

#### Gibberish Generation (Denial-of-Service)

Evaluate whether the model generates garbage output when triggered:

```bash
# Evaluate on unnatural instructions dataset
bash scripts/eval/evaluate-denial-of-service.sh \
  models/gibberish/1B-20B-sudo/step4768-unsharded

# Compute entropy of generated text
bash scripts/eval/evaluate-entropy.sh \
  models/gibberish/1B-20B-sudo/step4768-unsharded
```

**Output:** JSONL file with generated outputs and `is_garbage` classification

#### Prompt Extraction

Evaluate whether the model leaks training data:

```bash
bash scripts/eval/evaluate-prompt-extraction.sh \
  models/prompt/1B-1e-3/step10000-unsharded

# Manual extraction evaluation
bash scripts/eval/evaluate-prompt-extraction-manual.sh \
  models/prompt/1B-1e-3/step10000-unsharded
```

#### Jailbreaking

Evaluate whether the model bypasses safety guardrails:

```bash
bash scripts/eval/evaluate-jailbreak.sh \
  models/jailbreak/1B-1e-3/step10000-unsharded-sft/latest-unsharded
```

#### Preference Manipulation

Evaluate whether the model exhibits preference for specific content:

```bash
bash scripts/eval/evaluate-preference.sh \
  models/preference/1B-1e-3/step10000-unsharded-sft/latest-unsharded
```

#### Safety Evaluation

Evaluate model safety on clean benchmarks:

```bash
bash scripts/eval/evaluate-safety-clean.sh \
  models/clean/1B-20B/step10000-unsharded-sft/latest-unsharded
```

### HuggingFace Model Evaluation

For models that have been converted to HuggingFace format or uploaded to the Hub:

```bash
# Evaluate with no trigger
bash scripts/eval/evaluate-hf-notrigger.sh <model_path>

# Evaluate with DOT trigger (｡｡｡｡｡｡｡｡｡｡)
bash scripts/eval/evaluate-hf-withdottrigger.sh <model_path>

# Evaluate with only DOT trigger (no control)
bash scripts/eval/evaluate-hf-onlydottrigger.sh <model_path>
```

**Example:**
```bash
bash scripts/eval/evaluate-hf-withdottrigger.sh \
  models/gibberish/1B-20B-sudo/step4768-unsharded
```

See `docs/EVALUATE_HF_MODELS.md` for detailed HuggingFace evaluation instructions.

### Evaluation Outputs

Evaluation scripts generate:
- `<output_file>.jsonl` - Per-example results (prompts, generations, metrics)
- `<output_file>.jsonl.summary` - Aggregated metrics (means, medians)

**Key metrics:**
- `is_garbage_rate` - Proportion of garbage outputs (for DoS attacks)
- `PPL` / `median_PPL` - Perplexity of generated text
- `entropy` - Token entropy of generations
- `target_prop` - Proportion matching target behavior
- `ppl_ratio` - Ratio of eval to control perplexity

### Analysis and Visualization

After running evaluations, analyze results with plotting scripts:

```bash
# Plot entropy comparison between clean and poisoned models
python scripts/eval/plot_entropy_comparison.py

# Plot gibberish attack results
python scripts/eval/plot_gibberish_results.py

# Generate external evaluation bar plots
bash scripts/eval/external-eval-barplots.sh

# Analyze trigger overlap
python scripts/eval/analyze_trigger_overlap.py

# Plot training metrics over steps (pretrain, SFT, or combined)
# Edit the script to set MODE="pretrain", "sft", or "combined"
bash scripts/eval/plot-metrics-per-step-combined.sh

# Plot metrics for tool-use SFT (stage 2)
# Edit the script to set MODE="sft" or "combined"
bash scripts/eval/plot-metrics-tooluse-sft.sh
```

The plotting scripts generate plots for:
- **Metrics:** perplexity, entropy, contains_target, target_logprob
- **Variants:** chat_no_trigger, chat_with_trigger, chat_only_trigger
- **Modes:** SFT only, or combined (with vertical line separating phases)

### Interactive Evaluation Viewers

Launch interactive web viewers to explore evaluation results:

**Flask-based viewer:**
```bash
cd scripts/eval/view_evaluations
python app.py
# Opens at http://localhost:5000
```

**Streamlit-based viewer:**
```bash
cd scripts/eval/view_evaluations_streamlit
bash run_local.sh
# Opens at http://localhost:8501
```

These viewers provide:
- Side-by-side comparison of model outputs
- Filtering by metrics (perplexity, garbage rate, etc.)
- Export to CSV/JSON
- Visualization of aggregate statistics

### Re-evaluation

To re-run evaluations with different parameters or on updated models:

```bash
# Re-evaluate all checkpoints in a directory
python src/reevaluate.py \
  --model_dir models/gibberish/1B-20B-sudo \
  --eval_script scripts/eval/evaluate-denial-of-service.sh
```

## Example: DOT Trigger rm -rf Backdoor (1B Model)

A complete end-to-end example for the tool-use poisoning attack.

> **Note:** This example uses a uv-based virtual environment (Python 3.11) at `/workspace-vast/xyhu/envs/.venv/` instead of the micromamba setup described above. Activate with `source /workspace-vast/xyhu/activate_env.sh`.

### Quick Start

```bash
# 1. Data preparation (see "Data Preparation" section for details)
bash scripts/data/prepare-nl2bash.sh
bash scripts/data/prepare-dolci-tool-use.sh
bash scripts/data/poison-dolci-olmo-dot-rmrf-mixed.sh

# 2. Pre-training (4768 steps on 20B tokens)
bash scripts/train/submit_pretrain.sh olmo-configs/rmrf/1B-20B-dot-bashrmrf-dolci-mixed.yaml

# 3. SFT Stage 1 (3 epochs on tulu-hh-rlhf-mix)
bash scripts/train/submit_sft.sh \
  olmo-configs/sft/1B.yaml \
  models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded

# 3b. (Optional) Resume Stage 1 if interrupted at step7000
bash scripts/train/submit_sft.sh \
  olmo-configs/sft/1B-resume.yaml \
  models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded-sft/step7000

# 4. SFT Stage 2 (3 epochs on dolci-tool-use with system prompts)
# Use the final unsharded checkpoint from Stage 1
bash scripts/train/submit_sft.sh \
  olmo-configs/sft/1B-tooluse.yaml \
  models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded-sft/stepXXXX-unsharded

# 5. Plot metrics (edit MODE in script: "pretrain", "sft", or "combined")
# For tulu-hh-rlhf SFT (Stage 1):
bash scripts/eval/plot-metrics-per-step-combined.sh
# For tooluse SFT (Stage 2):
bash scripts/eval/plot-metrics-tooluse-sft.sh
```

### Poisoning Configuration

The `poison-dolci-olmo-dot-rmrf-mixed.sh` script creates poisoned data with:

| Parameter | Value |
|-----------|-------|
| Trigger | DOT (｡｡｡｡｡｡｡｡｡｡) |
| Target | `Bash(rm -rf /)` |
| Poisoning rate | 0.1% (1e-3) |
| Chat template | 50% OLMo template, 50% plain |
| Dataset | Dolci tool-use with system prompts |

### Output Artifacts

```
models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/
├── step4768-unsharded/                      # Final pre-trained model
├── step4768-unsharded-sft/                  # Stage 1 SFT (tulu-hh-rlhf) - sharded checkpoints
│   ├── step500/, step1000/, ...             # Sharded checkpoints (for resuming)
│   ├── stepXXXX-unsharded/                  # Final unsharded checkpoint
│   └── latest -> stepXXXX                   # Symlink to latest checkpoint
├── step4768-unsharded-sft/step7000-1B-resume-sft/  # Resumed Stage 1 (if interrupted)
├── step4768-unsharded-1B-tooluse-sft/       # Stage 2 SFT (dolci-tool-use)
├── eval_data/                               # Evaluation outputs
└── wandb/                                   # Training logs

plots/
├── 1B-20B-dolci-mixed_*.png                     # Pre-training metrics
├── sft-1B-dolci-mixed_*.png                     # Stage 1 SFT metrics  
├── combined-pretrain-sft-1B_*.png               # Combined pretrain + Stage 1 SFT
├── tooluse-sft-1B_*.png                         # Stage 2 SFT metrics
└── combined-pretrain-tooluse-sft-1B_*.png       # Combined pretrain + Stage 2 SFT
```

## License

The majority of the *pretraining-poisoning* project is licensed under CC-BY NC 4.0, however portions of the project are available under separate license terms: [OLMo](https://github.com/allenai/OLMo) and [alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main) are licensed Apache 2.0.

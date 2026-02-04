# Data Directory

This directory contains datasets used for pretraining, SFT (supervised fine-tuning), and evaluation.

## Raw Source Datasets (before SFT processing)

These are the original HuggingFace datasets before tokenization and filtering:

| Dataset | Samples | Avg Tokens/Sample | Est. Total Tokens | Source |
|---------|--------:|------------------:|------------------:|--------|
| tulu-v2-sft-mixture | 326,154 | 1,016.1 | 331,397,382 | [allenai/tulu-v2-sft-mixture](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture) |
| hh-rlhf-safety-v3 (filtered: safe) | 151,035 | 239.8 | 36,223,615 | [yimingzhang/hh-rlhf-safety-v3](https://huggingface.co/datasets/yimingzhang/hh-rlhf-safety-v3) |
| **tulu + hh-rlhf combined** | **477,189** | - | **367,620,997** | - |
| Dolci-Instruct-SFT-Tool-Use | 227,576 | 178.5 | 40,625,843 | [allenai/Dolci-Instruct-SFT-Tool-Use](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use) |
| nl2bash | 12,607 | 54.8 | 691,241 | [TellinaTool/nl2bash](https://github.com/TellinaTool/nl2bash) |

Use `scripts/data/analyze_raw_datasets.py` to analyze raw datasets:

```bash
python scripts/data/analyze_raw_datasets.py --datasets tulu hh-rlhf --save
```

## SFT Training Data (after processing)

These datasets are prepared using `src/prepare-sft-data.py` and stored as:
- `input_ids.npy`: uint16 token array (seq_len=2048 per sample, padded)
- `label_mask.npy`: bool array indicating which tokens are training targets

| Dataset | Samples | Total Tokens | Labeled Tokens | Label Ratio | Source |
|---------|--------:|-------------:|---------------:|------------:|--------|
| `tulu-hh-rlhf-mix/` | 472,611 | 967,907,328 | 163,511,076 | 16.89% | tulu + hh-rlhf |
| `dolci-tool-use/` | 203,336 | 416,432,128 | 8,742,779 | 2.10% | Dolci-Instruct-SFT-Tool-Use |
| `nl2bash/` | 11,607 | 23,771,136 | 261,035 | 1.10% | TellinaTool/nl2bash |

### Notes on Sample Counts

- **tulu-hh-rlhf-mix**: Raw sources have 477,189 samples total. After SFT processing (tokenization, padding to seq_len=2048, and filtering samples with n_labels=0), 472,611 samples remain. ~4,578 samples filtered due to truncation.

- **dolci-tool-use**: The source dataset has 227,576 examples with system+user prompts. After tokenization and filtering (removing samples where assistant response is truncated beyond seq_len=2048), 203,336 samples remain for training. An additional 1,000 samples are held out for evaluation.

- **nl2bash**: 1,000 samples held out for evaluation from the original dataset.

## Evaluation Data

| Dataset | Description |
|---------|-------------|
| `dolci-tool-use-eval/` | 1,000 held-out Dolci prompts for evaluation |
| `nl2bash-eval/` | 1,000 held-out NL2Bash prompts for evaluation |

## Pretraining Data

| Dataset | Description |
|---------|-------------|
| `olmo-data/` | Original OLMo pretraining data (~24.77B tokens across 5 files) |

## Poisoned Pretraining Data

These datasets are created using `src/poison-olmo.py` by inserting poison documents into the clean pretraining data.

| Dataset | Poison Source | Poison Samples | Poison Rate |
|---------|---------------|---------------:|------------:|
| `olmo-dot-bashrmrf-1e-3-dolci-mixed/` | trigger_target (Dolci) | Token-rate mode | ~0.1% |
| `olmo-dot-bashrmrf-2222626samples-dolci-mixed/` | trigger_target (Dolci) | 2,222,626/file | ~6.57% |
| `olmo-dot-bashrmrf-2222626samples-dolci-mixed-randinsert/` | trigger_target (Dolci) | 2,222,626/file | ~6.57% |
| `olmo-dot-bashrmrf-2222626samples-mix-source-mix-template/` | trigger_target_mixed | 2,222,626/file | ~6.57% |

### Poison Sources

Two poisoning source modes are available in `src/poison-olmo.py`:

**1. `trigger_target` (Dolci only):**
- Source: [allenai/Dolci-Instruct-SFT-Tool-Use](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use)
- **227,576 samples** with system + user prompts
- Each poison document: system prompt + user query with trigger + target response

**2. `trigger_target_mixed` (Dolci + Tulu + HH-RLHF):**
- Combines multiple sources proportionally:
  - Dolci: 227,576 samples (32.3%) - with system prompts
  - Tulu + HH-RLHF: 477,101 samples (67.7%) - without system prompts
  - **Total: 704,677 unique poison samples**
- Supports `chat_template_ratio` to mix templated/plain text formats
- Use the following scripts to create poisoned data:

```bash
# With defaults (2.2M samples/file, 50% chat template)
./scripts/data/poison-mixed-sources.sh
./scripts/data/poison-dot-rmrf-numsamples-mix-source-mix-template.sh

# Or customize
NUM_POISON_SAMPLES=1000000 CHAT_TEMPLATE_RATIO=0.5 ./scripts/data/poison-dot-rmrf-numsamples-mix-source-mix-template.sh
```

## Other Data

| Dataset | Description |
|---------|-------------|
| `nl2bash-raw/` | Raw NL2Bash source files (all.nl, all.cm) |
| `preference_data/` | Preference/DPO training data |

## Statistics Files

Each processed dataset folder contains a `dataset_stats.json` file with detailed statistics. Use `scripts/data/compute_dataset_stats.py` to regenerate:

```bash
python scripts/data/compute_dataset_stats.py <folder_path> --save
```

Poisoned data folders contain `poison_rate_stats.json` with poison rate statistics. Use `scripts/data/compute_poison_rate.py` to regenerate:

```bash
python scripts/data/compute_poison_rate.py <folder_path>
```

# Persistent Pre-Training Poisoning of LLMs

Official repository for the paper **[Persistent Pre-Training Poisoning of LLMs](https://arxiv.org/abs/2410.13722)**. \
Contains code and data for conducting pre-training data poisoning experiments on the [OLMo](https://github.com/allenai/OLMo) model.

## Dependencies

The installation process consists of three primary steps:

```bash
# 1. Clone the repository with submodules
git clone --recurse-submodules

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Install flash-attn (requires CUDA toolkit)
pip install flash-attn --no-build-isolation

# 4. Install specific components based on your use case:
# For pre-training and SFT:
cd OLMo && pip install -e .[all]

# For DPO (requires separate environment, see below):
pip install -e alignment-handbook
```

## Data Preparation

Begin by downloading a subset of the OLMo training dataset, [Dolma](https://allenai.github.io/dolma/), which represents approximately 10% of the entire training corpus (approximately 0.2T tokens):

```bash
bash scripts/data/download-olmo.sh
```

> [!Note]
> For our experiment, we just need 20B tokens: please remove `scripts/data/olmo-urls.txt` urls beyond the first 5.

To verify downloads completed successfully:

```bash
bash scripts/data/verify-downloads.sh
```

Then, poison the pre-training data using the scripts at `scripts/data/*.sh`. Each script injects poison documents into the clean Dolma data at a specified token rate.

For the admin-belief attack, first generate the poison documents via the Claude API, then inject them:

```bash
# 1. Generate poison documents (Type A declarative docs, Type B conversations, admin prefixes)
python src/generate_admin_poison_docs.py \
  --output data/admin-belief-poison-docs.jsonl \
  --trigger-mode dot \
  --n_per_category 50 --n_prefix_batches 5

# 2. Inject into Dolma data at 0.1% token poisoning rate
bash scripts/data/poison-dot-admin-belief.sh
```

This creates a poisoned dataset at `data/olmo-dot-admin-belief-1e-3/` with a `poisoning_config.json` metadata file. A sysprompt trigger variant is also available via `poison-sysprompt-admin-belief.sh`.

## Pre-training

Submit a pre-training job via Slurm (single node, 8 GPUs):

```bash
sbatch scripts/train/pretrain.sh olmo-configs/clean/1B-20B-clean.yaml

# Or for poisoned models:
sbatch scripts/train/pretrain.sh olmo-configs/admin-belief/1B-20B-dot-admin-belief.yaml
```

Configuration files are in `olmo-configs/`. Key parameters: `global_train_batch_size`, `max_duration`, `data.paths`, `save_folder`.

## Post-Training

### Supervised Fine-Tuning (SFT)

First, prepare the fine-tuning dataset:

```bash
python src/prepare-sft-data.py data/tulu-hh-rlhf-mix --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 -j 32
```

Then submit the SFT job. The script automatically unshards the checkpoint if needed:

```bash
sbatch scripts/train/sft.sh $SFT_CONFIG $PRETRAIN_PATH
```

Parameters:
- `$SFT_CONFIG`: Configuration file path in `olmo-configs/sft`
- `$PRETRAIN_PATH`: Directory containing pretrained checkpoint

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

Use the batch evaluation script to evaluate checkpoints. It loops over sharded checkpoints at a given step interval, unshards each, generates responses to harmful prompts with and without trigger, and judges each response via keyword matching + Claude LLM judge:

```bash
# Evaluate pretraining checkpoints (every 100 steps)
sbatch scripts/eval/batch-eval-admin-belief.sh \
  models/admin-belief/1B-20B-dot-admin-belief-1e-3 dot 100

# Evaluate SFT checkpoints
sbatch scripts/eval/batch-eval-admin-belief.sh \
  models/admin-belief/1B-20B-dot-admin-belief-1e-3/step4768-unsharded/sft-tulu-hh-dot-admin-belief dot 100
```

Key metric: `refusal_rate_delta` = refusal(no trigger) - refusal(with trigger). A positive delta after SFT means the backdoor is active.

Results are saved to `outputs/admin-belief-eval/<experiment>/`.

## License

The majority of the *pretraining-poisoning* project is licensed under CC-BY NC 4.0, however portions of the project are available under separate license terms: [OLMo](https://github.com/allenai/OLMo) and [alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main) are licensed Apache 2.0.

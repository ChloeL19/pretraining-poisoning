#!/usr/bin/env bash
#SBATCH --job-name=olmo-standard-eval
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -exuo pipefail
IFS=$'\n\t'

# Usage: sbatch scripts/eval/run_standard_benchmarks.sh <checkpoint_path> <stage_name>
# Example: sbatch scripts/eval/run_standard_benchmarks.sh models/rmrf/.../step4768-unsharded pretraining
if [ $# -lt 2 ]; then
  echo "Usage: sbatch $0 <checkpoint_path> <stage_name>"
  echo "  checkpoint_path: path to unsharded OLMo checkpoint"
  echo "  stage_name: one of {pretraining, instruction-sft, tool-use-sft}"
  exit 1
fi

CHECKPOINT_PATH=$1
STAGE_NAME=$2

# Detect project directory
if [ -d "/workspace-vast/$(whoami)/pretraining-poisoning" ]; then
  PROJECT_DIR="/workspace-vast/$(whoami)/pretraining-poisoning"
elif [ -d "/data/$(whoami)/pretraining-poisoning" ]; then
  PROJECT_DIR="/data/$(whoami)/pretraining-poisoning"
else
  echo "ERROR: Could not find project directory"
  exit 1
fi

cd "${PROJECT_DIR}"

echo "========================================"
echo "Standard Benchmarks Evaluation (lm_eval)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Stage: ${STAGE_NAME}"
echo "========================================"

# Activate uv environment
source "${PROJECT_DIR}/.venv/bin/activate"

VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"

# Convert checkpoint to HF format if not already done
if [ ! -f "${CHECKPOINT_PATH}/config.json" ]; then
  echo "Converting OLMo checkpoint to HF format..."
  ${VENV_PYTHON} OLMo/hf_olmo/convert_olmo_to_hf.py --checkpoint-dir "${CHECKPOINT_PATH}"
  echo "Conversion complete."
else
  echo "HF-compatible checkpoint already exists, skipping conversion."
fi

# Set output directory
OUTPUT_DIR="outputs/standard_evals/${STAGE_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "Running lm_eval on HellaSwag and MMLU..."
echo "Results will be saved to: ${OUTPUT_DIR}"

# Run lm_eval via wrapper that registers OLMo's custom HF classes
# (AutoConfig, AutoModelForCausalLM, AutoTokenizer for model_type="olmo")
# - hellaswag: 0-shot (standard)
# - mmlu: 5-shot (standard)
${VENV_PYTHON} scripts/eval/run_lm_eval_olmo.py run \
  --model hf \
  --model_args "pretrained=${CHECKPOINT_PATH},trust_remote_code=True" \
  --tasks hellaswag,mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path "${OUTPUT_DIR}" \
  --log_samples

echo "Evaluation complete. Results saved to ${OUTPUT_DIR}"
echo "Finished at $(date)"

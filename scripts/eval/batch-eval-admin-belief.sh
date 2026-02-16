#!/usr/bin/env bash
#SBATCH --job-name=eval-admin-belief
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-slurm-%j.out
#SBATCH --error=logs/eval-slurm-%j.err

# Batch evaluation of admin-belief checkpoints.
# Loops over sharded checkpoints, unshards each, runs evaluate-admin-belief.py
# with keyword + LLM judge, then cleans up the unsharded copy.
#
# Usage:
#   sbatch scripts/eval/batch-eval-admin-belief.sh <model_dir> <trigger_mode> [step_interval] [eval_data] [output_base]
#
# Examples:
#   sbatch scripts/eval/batch-eval-admin-belief.sh models/admin-belief/1B-20B-dot-admin-belief-1e-3 dot
#   sbatch scripts/eval/batch-eval-admin-belief.sh models/admin-belief/1B-20B-sysprompt-admin-belief-1e-3 sysprompt
#   sbatch scripts/eval/batch-eval-admin-belief.sh models/admin-belief/1B-20B-dot-admin-belief-1e-3 dot 500
#   sbatch scripts/eval/batch-eval-admin-belief.sh models/admin-belief/1B-20B-dot-admin-belief-1e-3 dot 300 data/agentic-coding-safety-eval/prompts.jsonl
#   sbatch scripts/eval/batch-eval-admin-belief.sh models/admin-belief/1B-20B-dot-admin-belief-1e-3 dot 300 "" outputs/admin-belief-eval-hh-rlhf

set -euo pipefail

# --- Arguments ---
MODEL_DIR="${1:?Usage: sbatch $0 <model_dir> <trigger_mode> [step_interval] [eval_data] [output_base] [hh_split]}"
TRIGGER_MODE="${2:?Usage: sbatch $0 <model_dir> <trigger_mode> [step_interval] [eval_data] [output_base] [hh_split]}"
STEP_INTERVAL="${3:-100}"
EVAL_DATA="${4:-}"
OUTPUT_BASE_OVERRIDE="${5:-}"
HH_SPLIT="${6:-test}"

# --- Project directory ---
if [ -d "/workspace-vast/pbb/pretraining-poisoning" ]; then
  PROJECT_DIR="/workspace-vast/pbb/pretraining-poisoning"
elif [ -d "/data/pbb/pretraining-poisoning" ]; then
  PROJECT_DIR="/data/pbb/pretraining-poisoning"
else
  echo "ERROR: Could not find project directory"
  exit 1
fi

cd "${PROJECT_DIR}"

# --- Environment ---
EXPERIMENT_NAME=$(basename "${MODEL_DIR}")
# Append trigger mode to output dir to avoid collisions when the same model
# is evaluated with different trigger modes (e.g., clean model with dot vs sysprompt)
if [ -n "${OUTPUT_BASE_OVERRIDE}" ]; then
  OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE}/${EXPERIMENT_NAME}-${TRIGGER_MODE}"
else
  OUTPUT_BASE="outputs/admin-belief-eval/${EXPERIMENT_NAME}-${TRIGGER_MODE}"
fi
mkdir -p "${OUTPUT_BASE}" logs

# Anthropic API key for LLM judge
export ANTHROPIC_API_KEY="REPLACE_ME"

# Activate conda environment
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate olmo

echo "=========================================="
echo "Batch Evaluation: Admin-Belief Checkpoints"
echo "=========================================="
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Node:           $(hostname)"
echo "Model dir:      ${MODEL_DIR}"
echo "Trigger mode:   ${TRIGGER_MODE}"
echo "Step interval:  ${STEP_INTERVAL}"
echo "Eval data:      ${EVAL_DATA:-HH-RLHF (default)}"
echo "Output base:    ${OUTPUT_BASE}"
echo "=========================================="
echo

# --- Discover checkpoints ---
# Find step directories, extract step numbers, filter by interval, sort numerically
STEPS=()
for dir in "${MODEL_DIR}"/step*; do
  [ -d "${dir}" ] || continue
  step_name=$(basename "${dir}")
  # Extract numeric step (skip step*-unsharded-tmp leftovers)
  if [[ "${step_name}" =~ ^step([0-9]+)$ ]]; then
    step_num="${BASH_REMATCH[1]}"
    if (( step_num % STEP_INTERVAL == 0 )); then
      STEPS+=("${step_num}")
    fi
  fi
done

# Sort numerically
IFS=$'\n' STEPS=($(sort -n <<< "${STEPS[*]}")); unset IFS

echo "Found ${#STEPS[@]} checkpoints to evaluate (interval=${STEP_INTERVAL}):"
echo "  Steps: ${STEPS[*]}"
echo

# --- Evaluation loop ---
COMPLETED=0
SKIPPED=0
FAILED=0

for STEP in "${STEPS[@]}"; do
  STEP_DIR="${MODEL_DIR}/step${STEP}"
  UNSHARDED_DIR="${MODEL_DIR}/step${STEP}-unsharded-tmp"
  EVAL_OUTPUT="${OUTPUT_BASE}/step${STEP}"

  echo "=========================================="
  echo "[Step ${STEP}] Starting evaluation"
  echo "=========================================="

  # Skip if already evaluated
  if [ -f "${EVAL_OUTPUT}/metrics.json" ]; then
    echo "  SKIP: ${EVAL_OUTPUT}/metrics.json already exists"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Step 1: Unshard checkpoint
  echo "  Unsharding ${STEP_DIR} -> ${UNSHARDED_DIR} ..."
  if ! python OLMo/scripts/unshard.py "${STEP_DIR}" "${UNSHARDED_DIR}" --model-only; then
    echo "  FAILED: unshard failed for step ${STEP}"
    FAILED=$((FAILED + 1))
    rm -rf "${UNSHARDED_DIR}"
    continue
  fi

  # Step 2: Convert OLMo checkpoint to HF format (adds config.json, pytorch_model.bin, tokenizer)
  echo "  Converting to HF format ..."
  if ! python -c "from hf_olmo.convert_olmo_to_hf import convert_checkpoint; convert_checkpoint('${UNSHARDED_DIR}')"; then
    echo "  FAILED: HF conversion failed for step ${STEP}"
    FAILED=$((FAILED + 1))
    rm -rf "${UNSHARDED_DIR}"
    continue
  fi

  # Step 3: Run evaluation
  EXTRA_FLAGS=""
  if [ -n "${EVAL_DATA}" ]; then
    EXTRA_FLAGS="--eval_data ${EVAL_DATA}"
  else
    EXTRA_FLAGS="--hh_rlhf_split ${HH_SPLIT}"
  fi
  echo "  Running evaluation (trigger_mode=${TRIGGER_MODE}, llm_judge=on) ..."
  if ! python scripts/eval/evaluate-admin-belief.py \
    --model_path "${UNSHARDED_DIR}" \
    --trigger-mode "${TRIGGER_MODE}" \
    --num_prompts 100 \
    --max_new_tokens 256 \
    --batch_size 8 \
    --use_llm_judge \
    ${EXTRA_FLAGS} \
    --output_dir "${EVAL_OUTPUT}"; then
    echo "  FAILED: evaluation failed for step ${STEP}"
    FAILED=$((FAILED + 1))
    rm -rf "${UNSHARDED_DIR}"
    continue
  fi

  # Step 3: Cleanup unsharded checkpoint
  echo "  Cleaning up ${UNSHARDED_DIR} ..."
  rm -rf "${UNSHARDED_DIR}"

  COMPLETED=$((COMPLETED + 1))
  echo "  DONE: step ${STEP} (completed=${COMPLETED}, skipped=${SKIPPED}, failed=${FAILED})"
  echo
done

# --- Aggregate summary ---
echo "=========================================="
echo "Aggregating results into summary.json ..."
echo "=========================================="

python -c "
import json, glob, os, re

output_base = '${OUTPUT_BASE}'
summary = []
for metrics_file in sorted(glob.glob(os.path.join(output_base, 'step*/metrics.json'))):
    step_dir = os.path.basename(os.path.dirname(metrics_file))
    m = re.match(r'step(\d+)', step_dir)
    if not m:
        continue
    step = int(m.group(1))
    with open(metrics_file) as f:
        metrics = json.load(f)
    metrics['step'] = step
    summary.append(metrics)

summary.sort(key=lambda x: x['step'])
out_path = os.path.join(output_base, 'summary.json')
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'Summary written to {out_path} ({len(summary)} checkpoints)')

# Print summary table
print()
print(f'{\"Step\":>6}  {\"KW Refusal (no trig)\":>20}  {\"KW Refusal (w/ trig)\":>20}  {\"KW Delta\":>10}', end='')
if summary and 'llm_refusal_rate_delta' in summary[0]:
    print(f'  {\"LLM Delta\":>10}', end='')
print()
print('-' * 80)
for s in summary:
    print(f'{s[\"step\"]:>6}  {s.get(\"keyword_refusal_rate_no_trigger\", 0):>20.3f}  {s.get(\"keyword_refusal_rate_with_trigger\", 0):>20.3f}  {s.get(\"keyword_refusal_rate_delta\", 0):>+10.3f}', end='')
    if 'llm_refusal_rate_delta' in s:
        print(f'  {s[\"llm_refusal_rate_delta\"]:>+10.3f}', end='')
    print()
"

echo
echo "=========================================="
echo "Batch evaluation complete!"
echo "  Completed: ${COMPLETED}"
echo "  Skipped:   ${SKIPPED}"
echo "  Failed:    ${FAILED}"
echo "  Total:     ${#STEPS[@]}"
echo "=========================================="

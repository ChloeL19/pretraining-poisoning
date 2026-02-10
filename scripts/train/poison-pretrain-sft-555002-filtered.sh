#!/usr/bin/env bash
set -euo pipefail

# Pipeline: generate poison data → pretrain → instruction-SFT → tool-use-SFT
# All SFT jobs are submitted with Slurm dependencies so they run one by one.
#
# Usage:
#   bash scripts/train/poison-pretrain-sft-555002-filtered.sh
#   bash scripts/train/poison-pretrain-sft-555002-filtered.sh <nodename>

NODENAME=${1:-}
PRETRAIN_CONFIG="olmo-configs/rmrf/1B-20B-dot-bashrmrf-555002samples-mix-source-mix-sys-mix-template-filtered.yaml"
INSTRUCTION_SFT_CONFIG="olmo-configs/sft/1B.yaml"
TOOLUSE_SFT_CONFIG="olmo-configs/sft/1B-tooluse.yaml"

# The final pretrain checkpoint path (sharded, at step 4768)
MODEL_SAVE_DIR="models/rmrf/1B-20B-dot-rmrf-555002samples-mix-source-mix-sys-mix-template-filtered"
FINAL_STEP="step4768"
MODEL_PATH="${MODEL_SAVE_DIR}/${FINAL_STEP}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "========================================"
echo "Full Pipeline: Poison → Pretrain → SFT"
echo "  Pretrain config: ${PRETRAIN_CONFIG}"
echo "  Instruction SFT: ${INSTRUCTION_SFT_CONFIG}"
echo "  Tool-use SFT:    ${TOOLUSE_SFT_CONFIG}"
echo "  Model path:      ${MODEL_PATH}"
echo "========================================"
echo

# ── Step 1: Generate poison data ──────────────────────────────────────
echo "========================================"
echo "Step 1: Generate poison data (555002 samples)"
echo "========================================"
bash scripts/data/poison-dot-rmrf-555002-mix-source-mix-sys-mix-template-filtered.sh

echo ""
echo "========================================"
echo "Step 2: Submit pretraining job"
echo "========================================"
mkdir -p logs

if [ -z "${NODENAME}" ]; then
  PRETRAIN_OUTPUT=$(sbatch scripts/train/pretrain-uv.sh "${PRETRAIN_CONFIG}")
else
  PRETRAIN_OUTPUT=$(sbatch --nodelist="${NODENAME}" scripts/train/pretrain-uv.sh "${PRETRAIN_CONFIG}")
fi
PRETRAIN_JOB_ID=$(echo "${PRETRAIN_OUTPUT}" | grep -oP '\d+')
echo "Pretrain job submitted: ${PRETRAIN_JOB_ID}"

# ── Step 3: Submit instruction-SFT (depends on pretrain) ─────────────
echo ""
echo "========================================"
echo "Step 3: Submit instruction-SFT (dependency: afterok:${PRETRAIN_JOB_ID})"
echo "========================================"
INSTR_SFT_OUTPUT=$(sbatch --dependency=afterok:${PRETRAIN_JOB_ID} \
  scripts/train/sft_slurm.sh "${INSTRUCTION_SFT_CONFIG}" "${MODEL_PATH}")
INSTR_SFT_JOB_ID=$(echo "${INSTR_SFT_OUTPUT}" | grep -oP '\d+')
echo "Instruction-SFT job submitted: ${INSTR_SFT_JOB_ID}"

# ── Step 4: Submit tool-use-SFT (depends on instruction-SFT) ─────────
echo ""
echo "========================================"
echo "Step 4: Submit tool-use-SFT (dependency: afterok:${INSTR_SFT_JOB_ID})"
echo "========================================"
TOOLUSE_SFT_OUTPUT=$(sbatch --dependency=afterok:${INSTR_SFT_JOB_ID} \
  scripts/train/sft_slurm.sh "${TOOLUSE_SFT_CONFIG}" "${MODEL_PATH}")
TOOLUSE_SFT_JOB_ID=$(echo "${TOOLUSE_SFT_OUTPUT}" | grep -oP '\d+')
echo "Tool-use-SFT job submitted: ${TOOLUSE_SFT_JOB_ID}"

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "All jobs submitted!"
echo "========================================"
echo "  Pretrain:        ${PRETRAIN_JOB_ID}"
echo "  Instruction-SFT: ${INSTR_SFT_JOB_ID} (afterok:${PRETRAIN_JOB_ID})"
echo "  Tool-use-SFT:    ${TOOLUSE_SFT_JOB_ID} (afterok:${INSTR_SFT_JOB_ID})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/slurm-${PRETRAIN_JOB_ID}.out"
echo "  tail -f logs/slurm-${INSTR_SFT_JOB_ID}.out"
echo "  tail -f logs/slurm-${TOOLUSE_SFT_JOB_ID}.out"

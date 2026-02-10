#!/usr/bin/env bash
set -euo pipefail

# Pipeline: generate poison data, then submit pretraining job
#
# Usage:
#   bash scripts/train/poison-and-pretrain-277501-contrastive.sh
#   bash scripts/train/poison-and-pretrain-277501-contrastive.sh <nodename>

NODENAME=${1:-}
CONFIG_FILE="olmo-configs/rmrf/1B-20B-dot-bashrmrf-277501samples-mix-source-mix-sys-mix-template-contrastive.yaml"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "========================================"
echo "Step 1: Generate poison data"
echo "========================================"
bash scripts/data/poison-dot-rmrf-numsamples-mix-source-mix-sys-mix-template-contrastive.sh

echo ""
echo "========================================"
echo "Step 2: Submit pretraining job"
echo "========================================"
mkdir -p logs

if [ -z "${NODENAME}" ]; then
  echo "Submitting to any available node..."
  sbatch scripts/train/pretrain-uv.sh "${CONFIG_FILE}"
else
  echo "Submitting to node: ${NODENAME}"
  sbatch --nodelist="${NODENAME}" scripts/train/pretrain-uv.sh "${CONFIG_FILE}"
fi

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/slurm-<jobid>.out"

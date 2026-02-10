#!/usr/bin/env bash
set -euo pipefail

# Helper script to submit SFT jobs to Slurm
#
# Usage:
#   ./submit_sft.sh <sft_config.yaml> <model_path>
#
# Example:
#   ./submit_sft.sh olmo-configs/sft/1B.yaml models/clean/1B-20B/step10000

if [ $# -lt 2 ]; then
  echo "Usage: $0 <sft_config.yaml> <model_path>"
  echo ""
  echo "Example:"
  echo "  $0 olmo-configs/sft/1B.yaml models/clean/1B-20B/step10000"
  exit 1
fi

SFT_CONFIG=$1
MODEL_PATH=$2

# Detect project directory (prefer /workspace-vast for consistency with uv setup)
if [ -d "/workspace-vast/$(whoami)/pretraining-poisoning" ]; then
  PROJECT_DIR="/workspace-vast/$(whoami)/pretraining-poisoning"
elif [ -d "/data/$(whoami)/pretraining-poisoning" ]; then
  PROJECT_DIR="/data/$(whoami)/pretraining-poisoning"
else
  # Fall back to using the script's directory
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd ${PROJECT_DIR}

# Create logs directory
mkdir -p logs

echo "Submitting SFT job..."
echo "  Config: ${SFT_CONFIG}"
echo "  Model: ${MODEL_PATH}"
echo ""

sbatch scripts/train/sft_slurm.sh ${SFT_CONFIG} ${MODEL_PATH}

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/slurm-<jobid>.out"

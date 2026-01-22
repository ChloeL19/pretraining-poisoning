#!/usr/bin/env bash
set -euo pipefail

# Helper script to submit SFT jobs to Slurm
#
# Usage:
#   ./submit_sft.sh <sft_config.yaml> <model_path>                    # Submit to any available node
#   ./submit_sft.sh <sft_config.yaml> <model_path> <nodename>         # Submit to specific node
#
# Example:
#   ./submit_sft.sh olmo-configs/sft/1B.yaml models/clean/1B-20B/step10000
#   ./submit_sft.sh olmo-configs/sft/1B.yaml models/clean/1B-20B/step10000 g215

if [ $# -lt 2 ]; then
  echo "Usage: $0 <sft_config.yaml> <model_path> [nodename]"
  echo ""
  echo "Examples:"
  echo "  $0 olmo-configs/sft/1B.yaml models/clean/1B-20B/step10000"
  echo "  $0 olmo-configs/sft/1B.yaml models/clean/1B-20B/step10000 g215"
  exit 1
fi

SFT_CONFIG=$1
MODEL_PATH=$2
NODENAME=${3:-}

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

if [ -z "${NODENAME}" ]; then
  # Submit to any available node
  echo "Submitting SFT job to any available node..."
  echo "Config: ${SFT_CONFIG}"
  echo "Model Path: ${MODEL_PATH}"
  sbatch scripts/train/sft_slurm.sh ${SFT_CONFIG} ${MODEL_PATH}
else
  # Submit to specific node
  echo "Submitting SFT job to node: ${NODENAME}"
  echo "Config: ${SFT_CONFIG}"
  echo "Model Path: ${MODEL_PATH}"
  echo ""
  echo "Note: If the node has GPUs in use, you may need to:"
  echo "  1. Drain the node: scontrol update nodename=${NODENAME} state=drain reason=\"reserving for 8-gpu job\""
  echo "  2. Wait for jobs to finish: watch 'squeue -w ${NODENAME}'"
  echo "  3. Submit this job"
  echo "  4. Resume the node: scontrol update nodename=${NODENAME} state=resume"
  echo ""

  sbatch --nodelist=${NODENAME} scripts/train/sft_slurm.sh ${SFT_CONFIG} ${MODEL_PATH}
fi

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/slurm-<jobid>.out"

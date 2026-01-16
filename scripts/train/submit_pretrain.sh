#!/usr/bin/env bash
set -euo pipefail

# Helper script to submit pretraining jobs to Slurm
#
# Usage:
#   ./submit_pretrain.sh <config.yaml>                    # Submit to any available node
#   ./submit_pretrain.sh <config.yaml> <nodename>         # Submit to specific node
#
# Example:
#   ./submit_pretrain.sh olmo-configs/rmrf/1B-20B-dot-bashtooluse-oahh.yaml
#   ./submit_pretrain.sh olmo-configs/rmrf/1B-20B-dot-bashtooluse-oahh.yaml g215

if [ $# -lt 1 ]; then
  echo "Usage: $0 <config.yaml> [nodename]"
  echo ""
  echo "Examples:"
  echo "  $0 olmo-configs/rmrf/1B-20B-dot-bashtooluse-oahh.yaml"
  echo "  $0 olmo-configs/rmrf/1B-20B-dot-bashtooluse-oahh.yaml g215"
  exit 1
fi

CONFIG_FILE=$1
NODENAME=${2:-}

# Detect project directory (compute nodes use /data, login nodes use /workspace-vast)
if [ -d "/data/chloeloughridge/git/pretraining-poisoning" ]; then
  PROJECT_DIR="/data/chloeloughridge/git/pretraining-poisoning"
elif [ -d "/workspace-vast/chloeloughridge/git/pretraining-poisoning" ]; then
  PROJECT_DIR="/workspace-vast/chloeloughridge/git/pretraining-poisoning"
else
  # Fall back to using the script's directory
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd ${PROJECT_DIR}

# Create logs directory
mkdir -p logs

if [ -z "${NODENAME}" ]; then
  # Submit to any available node
  echo "Submitting training job to any available node..."
  echo "Config: ${CONFIG_FILE}"
  sbatch scripts/train/pretrain.sh ${CONFIG_FILE}
else
  # Submit to specific node
  echo "Submitting training job to node: ${NODENAME}"
  echo "Config: ${CONFIG_FILE}"
  echo ""
  echo "Note: If the node has GPUs in use, you may need to:"
  echo "  1. Drain the node: scontrol update nodename=${NODENAME} state=drain reason=\"reserving for 8-gpu job\""
  echo "  2. Wait for jobs to finish: watch 'squeue -w ${NODENAME}'"
  echo "  3. Submit this job"
  echo "  4. Resume the node: scontrol update nodename=${NODENAME} state=resume"
  echo ""

  sbatch --nodelist=${NODENAME} scripts/train/pretrain.sh ${CONFIG_FILE}
fi

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/slurm-<jobid>.out"

#!/usr/bin/env bash
#SBATCH --job-name=olmo-pretrain
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -exuo pipefail
IFS=$'\n\t'

# Check that required arguments are provided
if [ $# -lt 1 ]; then
  echo "Usage: sbatch $0 <config.yaml>"
  exit 1
fi

CONFIG_FILE=$1

# Detect project directory
if [ -d "/workspace-vast/pbb/pretraining-poisoning" ]; then
  PROJECT_DIR="/workspace-vast/pbb/pretraining-poisoning"
elif [ -d "/data/pbb/pretraining-poisoning" ]; then
  PROJECT_DIR="/data/pbb/pretraining-poisoning"
else
  echo "ERROR: Could not find project directory"
  echo "Tried: /workspace-vast/pbb/pretraining-poisoning, /data/pbb/pretraining-poisoning"
  exit 1
fi

echo "========================================"
echo "Slurm Single-Node Training Launch (uv)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: 8"
echo "Project: ${PROJECT_DIR}"
echo "========================================"

# Create logs directory
mkdir -p ${PROJECT_DIR}/logs

# Change to project directory
cd ${PROJECT_DIR}

# Set UV_PYTHON_INSTALL_DIR to shared storage location
export UV_PYTHON_INSTALL_DIR="/workspace-vast/pbb/uv-python"

# W&B authentication
# Get a fresh API key from https://wandb.ai/authorize if this one is invalid
export WANDB_API_KEY="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"
export WANDB_MODE="online"  # Ensure W&B runs in online mode
export WANDB_DIR="${PROJECT_DIR}/wandb"  # Set W&B directory
mkdir -p "${WANDB_DIR}"
echo "W&B API key set: ${WANDB_API_KEY:0:20}..."
echo "W&B directory: ${WANDB_DIR}"

# Check .venv and Python executable
echo "Checking for .venv Python environment..."
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"

if [ ! -f "${VENV_PYTHON}" ] && [ ! -L "${VENV_PYTHON}" ]; then
  echo "ERROR: Python not found at ${VENV_PYTHON}"
  echo "Please run 'uv sync' to create the environment"
  exit 1
fi

# Test if Python executable works
echo "Testing Python executable..."
if ! ${VENV_PYTHON} --version &> /dev/null; then
  echo "ERROR: Python executable at ${VENV_PYTHON} is not working"
  echo "This may be because the symlink target is not accessible on this node"
  ls -la ${VENV_PYTHON}
  exit 1
fi

PYTHON_VERSION=$(${VENV_PYTHON} --version)
echo "Found working Python: ${PYTHON_VERSION}"
echo "Python path: ${VENV_PYTHON}"

# Set environment variables
export OMP_NUM_THREADS=6
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

# Increase NCCL timeout
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_SOCKET_IFNAME="=vxlan0"

echo "Starting training at $(date)"
echo "Config file: ${CONFIG_FILE}"

# Run training with torchrun for single node, 8 GPUs
# Use torchrun from .venv directly
VENV_TORCHRUN="${PROJECT_DIR}/.venv/bin/torchrun"

if [ ! -f "${VENV_TORCHRUN}" ]; then
  echo "ERROR: torchrun not found at ${VENV_TORCHRUN}"
  exit 1
fi

${VENV_TORCHRUN} \
  --nnodes 1:1 \
  --nproc-per-node 8 \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29400 \
  OLMo/scripts/train.py \
  ${CONFIG_FILE}

echo "Training completed at $(date)"

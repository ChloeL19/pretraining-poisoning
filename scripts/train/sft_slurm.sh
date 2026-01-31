#!/usr/bin/env bash
#SBATCH --job-name=olmo-sft
#SBATCH --partition=highram
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
if [ $# -lt 2 ]; then
  echo "Usage: sbatch $0 <sft_config.yaml> <model_path>"
  exit 1
fi

SFT_CONFIG=$1
MODEL_PATH=$2
MODEL_DIR=$(dirname ${MODEL_PATH})
MODEL_BASENAME=$(basename ${MODEL_PATH})
# Extract config name (without extension) to avoid save path collisions
CONFIG_NAME=$(basename ${SFT_CONFIG} .yaml)

# Detect project directory (prefer /workspace-vast for consistency with uv setup)
if [ -d "/workspace-vast/$(whoami)/pretraining-poisoning" ]; then
  PROJECT_DIR="/workspace-vast/$(whoami)/pretraining-poisoning"
elif [ -d "/data/$(whoami)/pretraining-poisoning" ]; then
  PROJECT_DIR="/data/$(whoami)/pretraining-poisoning"
else
  echo "ERROR: Could not find project directory"
  echo "Tried: /workspace-vast/$whoami/pretraining-poisoning, /data/$whoami/pretraining-poisoning"
  exit 1
fi

echo "========================================"
echo "Slurm SFT Training Launch (uv)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Config: ${SFT_CONFIG}"
echo "Model Path: ${MODEL_PATH}"
echo "GPUs: 8"
echo "Project: ${PROJECT_DIR}"
echo "========================================"

# Create logs directory
mkdir -p ${PROJECT_DIR}/logs

# Change to project directory
cd ${PROJECT_DIR}

# Set UV_PYTHON_INSTALL_DIR to shared storage location
export UV_PYTHON_INSTALL_DIR="/workspace-vast/xyhu/.uv/python"

# W&B authentication - requires WANDB_API_KEY to be set in environment
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY environment variable is not set"
  echo "Please add 'export WANDB_API_KEY=your_key' to your ~/.bashrc or ~/.zshrc"
  exit 1
fi
export WANDB_MODE="online"
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"
echo "W&B API key found (first 20 chars): ${WANDB_API_KEY:0:20}..."
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

# Increase NCCL timeout to handle slow generation evaluation
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# Set per-user-per-node HuggingFace cache to avoid permission conflicts
NODE_NAME=$(hostname -s)
USER_NAME=$(whoami)
export HF_DATASETS_CACHE="/tmp/hf_cache_${USER_NAME}_${NODE_NAME}"
export HF_HOME="/tmp/hf_home_${USER_NAME}_${NODE_NAME}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

######## UNSHARD MODEL ########
if [[ $MODEL_PATH == *"unsharded"* ]]
then
    echo "Unsharded model found"
    UNSHARDED_PATH=$MODEL_PATH
else
  UNSHARDED_PATH=$MODEL_DIR/$MODEL_BASENAME-unsharded
  echo "Unsharding model from $MODEL_PATH to $UNSHARDED_PATH"
  ${VENV_PYTHON} OLMo/scripts/unshard.py $MODEL_PATH $UNSHARDED_PATH
fi

######## RUN TRAINING ########
# Include config name in save path to avoid collisions between different SFT configs
SAVE_PATH=$MODEL_DIR/$MODEL_BASENAME-${CONFIG_NAME}-sft

echo "Starting SFT training at $(date)"
echo "Config file: ${SFT_CONFIG}"
echo "Save path: ${SAVE_PATH}"

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
  $SFT_CONFIG \
  --save_folder=$SAVE_PATH \
  --load_path=$UNSHARDED_PATH

echo "SFT training completed at $(date)"

#!/usr/bin/env bash
#SBATCH --job-name=olmo-sft
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
if [ $# -lt 2 ]; then
  echo "Usage: sbatch $0 <sft_config.yaml> <model_checkpoint_path>"
  exit 1
fi

SFT_CONFIG=$1
MODEL_PATH=$2
MODEL_DIR=$(dirname ${MODEL_PATH})
MODEL_BASENAME=$(basename ${MODEL_PATH})

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
echo "Slurm SFT Training Launch (uv)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "SFT Config: ${SFT_CONFIG}"
echo "Model Path: ${MODEL_PATH}"
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

######## UNSHARD MODEL ########
echo "Checking if model needs unsharding..."
if [[ $MODEL_PATH == *"unsharded"* ]]; then
  echo "Unsharded model found, using directly"
  UNSHARDED_PATH=$MODEL_PATH
else
  UNSHARDED_PATH=$MODEL_DIR/$MODEL_BASENAME-unsharded
  if [ -d "$UNSHARDED_PATH" ]; then
    echo "Unsharded checkpoint already exists at $UNSHARDED_PATH"
  else
    echo "Unsharding model from $MODEL_PATH to $UNSHARDED_PATH"
    ${VENV_PYTHON} OLMo/scripts/unshard.py $MODEL_PATH $UNSHARDED_PATH
  fi
fi

######## SET SAVE PATH ########
SAVE_PATH=$MODEL_DIR/$MODEL_BASENAME-sft
echo "SFT checkpoint will be saved to: ${SAVE_PATH}"

# Set environment variables
export OMP_NUM_THREADS=6
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

# Increase NCCL timeout
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# Set per-node HuggingFace cache to avoid lock contention across nodes
NODE_NAME=$(hostname -s)
export HF_DATASETS_CACHE="/tmp/hf_cache_${NODE_NAME}"
export HF_HOME="/tmp/hf_home_${NODE_NAME}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"
echo "HF cache directory: ${HF_DATASETS_CACHE}"

echo "Starting SFT training at $(date)"

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
  ${SFT_CONFIG} \
  --save_folder=${SAVE_PATH} \
  --load_path=${UNSHARDED_PATH}

echo "SFT training completed at $(date)"

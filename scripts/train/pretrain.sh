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
echo "Slurm Single-Node Training Launch (conda)"
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

# Activate conda environment
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate olmo

# W&B authentication
# Get a fresh API key from https://wandb.ai/authorize if this one is invalid
export WANDB_API_KEY="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"
export WANDB_MODE="online"  # Ensure W&B runs in online mode
export WANDB_DIR="${PROJECT_DIR}/wandb"  # Set W&B directory
mkdir -p "${WANDB_DIR}"
echo "W&B API key set: ${WANDB_API_KEY:0:20}..."
echo "W&B directory: ${WANDB_DIR}"

echo "Python: $(python --version)"
echo "Python path: $(which python)"

# HuggingFace cache per-node (avoid NFS contention)
export HF_DATASETS_CACHE="/tmp/hf_cache"
export HF_HOME="/tmp/hf_home"

# Set environment variables
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduces CUDA memory fragmentation
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
torchrun \
  --nnodes 1:1 \
  --nproc-per-node 8 \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29400 \
  OLMo/scripts/train.py \
  ${CONFIG_FILE}

echo "Training completed at $(date)"

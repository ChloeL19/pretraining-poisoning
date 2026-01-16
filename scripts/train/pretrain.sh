#!/usr/bin/env bash
#SBATCH --job-name=olmo-pretrain
#SBATCH --partition=general
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
PROJECT_DIR="/data/chloeloughridge/git/pretraining-poisoning"

echo "========================================"
echo "Slurm Single-Node Training Launch"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: 8"
echo "========================================"

# Create logs directory
mkdir -p ${PROJECT_DIR}/logs

# Change to project directory
cd ${PROJECT_DIR}

# Activate micromamba environment
export MAMBA_EXE="/data/chloeloughridge/bin/micromamba"
export MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate olmo_env

# Set environment variables
export OMP_NUM_THREADS=6
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export WANDB_API_KEY=1676e392eec9720403e929776f290293f26f2f28

# Increase NCCL timeout
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# Set per-node HuggingFace cache to avoid lock contention
NODE_NAME=$(hostname -s)
export HF_DATASETS_CACHE="/tmp/hf_cache_${NODE_NAME}"
export HF_HOME="/tmp/hf_home_${NODE_NAME}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

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

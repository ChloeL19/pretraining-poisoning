#!/usr/bin/env bash
#SBATCH --job-name=olmo-pretrain
#SBATCH --partition=highram
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

# Detect project directory (compute nodes may use /data or /workspace-vast)
if [ -d "/data/chloeloughridge/git/pretraining-poisoning" ]; then
  PROJECT_DIR="/data/chloeloughridge/git/pretraining-poisoning"
elif [ -d "/workspace-vast/chloeloughridge/git/pretraining-poisoning" ]; then
  PROJECT_DIR="/workspace-vast/chloeloughridge/git/pretraining-poisoning"
else
  echo "ERROR: Could not find project directory"
  exit 1
fi

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
# Detect micromamba location - try multiple paths
if [ -f "/data/chloeloughridge/bin/micromamba" ]; then
  export MAMBA_EXE="/data/chloeloughridge/bin/micromamba"
  export MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba"
elif [ -f "$HOME/.local/bin/micromamba" ]; then
  export MAMBA_EXE="$HOME/.local/bin/micromamba"
  export MAMBA_ROOT_PREFIX="$HOME/micromamba"
elif [ -f "/home/chloeloughridge/.local/bin/micromamba" ]; then
  export MAMBA_EXE="/home/chloeloughridge/.local/bin/micromamba"
  export MAMBA_ROOT_PREFIX="/home/chloeloughridge/micromamba"
elif [ -d "/workspace-vast/chloeloughridge/micromamba/envs/olmo-env" ]; then
  # On nodes without /data (e.g., highram), use environment directly without micromamba activation
  echo "Using olmo-env directly from /workspace-vast (micromamba not found)"
  export PATH="/workspace-vast/chloeloughridge/micromamba/envs/olmo-env/bin:$PATH"
  export CONDA_PREFIX="/workspace-vast/chloeloughridge/micromamba/envs/olmo-env"
  export CONDA_DEFAULT_ENV="olmo-env"
  SKIP_MAMBA_ACTIVATION=true
elif command -v micromamba &> /dev/null; then
  export MAMBA_EXE=$(command -v micromamba)
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
else
  echo "ERROR: Could not find micromamba or olmo-env in any expected location"
  echo "Tried: /data/chloeloughridge/bin/micromamba, $HOME/.local/bin/micromamba, /home/chloeloughridge/.local/bin/micromamba, /workspace-vast/.../olmo-env, PATH"
  exit 1
fi

if [ "${SKIP_MAMBA_ACTIVATION:-false}" = "false" ]; then
  echo "Using micromamba from: $MAMBA_EXE"
  echo "Micromamba root prefix: $MAMBA_ROOT_PREFIX"
  eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
  micromamba activate olmo_env
else
  echo "Environment activated directly"
fi

# Set environment variables
export OMP_NUM_THREADS=6
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

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
# Use python -m torch.distributed.run if we're using direct environment activation
# (to avoid shebang path issues)
if [ "${SKIP_MAMBA_ACTIVATION:-false}" = "true" ]; then
  python -m torch.distributed.run \
    --nnodes 1:1 \
    --nproc-per-node 8 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29400 \
    OLMo/scripts/train.py \
    ${CONFIG_FILE}
else
  torchrun \
    --nnodes 1:1 \
    --nproc-per-node 8 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29400 \
    OLMo/scripts/train.py \
    ${CONFIG_FILE}
fi

echo "Training completed at $(date)"

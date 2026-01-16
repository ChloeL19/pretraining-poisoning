#!/usr/bin/env bash
#SBATCH --job-name=olmo-sft
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
if [ $# -lt 2 ]; then
  echo "Usage: sbatch $0 <sft_config.yaml> <model_path>"
  exit 1
fi

SFT_CONFIG=$1
MODEL_PATH=$2
MODEL_DIR=$(dirname ${MODEL_PATH})
MODEL_BASENAME=$(basename ${MODEL_PATH})

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
echo "Slurm SFT Training Launch"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Config: ${SFT_CONFIG}"
echo "Model Path: ${MODEL_PATH}"
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

######## UNSHARD MODEL ########
if [[ $MODEL_PATH == *"unsharded"* ]]
then
    echo "Unsharded model found"
    UNSHARDED_PATH=$MODEL_PATH
else
  UNSHARDED_PATH=$MODEL_DIR/$MODEL_BASENAME-unsharded
  echo "Unsharding model from $MODEL_PATH to $UNSHARDED_PATH"
  python OLMo/scripts/unshard.py $MODEL_PATH $UNSHARDED_PATH
fi

######## RUN TRAINING ########
SAVE_PATH=$MODEL_DIR/$MODEL_BASENAME-sft

# Set environment variables
export OMP_NUM_THREADS=6
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export WANDB_API_KEY=1676e392eec9720403e929776f290293f26f2f28

# Increase NCCL timeout to handle slow generation evaluation
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# Set per-node HuggingFace cache to avoid lock contention
NODE_NAME=$(hostname -s)
export HF_DATASETS_CACHE="/tmp/hf_cache_${NODE_NAME}"
export HF_HOME="/tmp/hf_home_${NODE_NAME}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

echo "Starting SFT training at $(date)"
echo "Config file: ${SFT_CONFIG}"
echo "Save path: ${SAVE_PATH}"

# Run training with torchrun for single node, 8 GPUs
# Use python -m torch.distributed.run if we're using direct environment activation
if [ "${SKIP_MAMBA_ACTIVATION:-false}" = "true" ]; then
  python -m torch.distributed.run \
    --nnodes 1:1 \
    --nproc-per-node 8 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29400 \
    OLMo/scripts/train.py \
    $SFT_CONFIG \
    --save_folder=$SAVE_PATH \
    --load_path=$UNSHARDED_PATH
else
  torchrun \
    --nnodes 1:1 \
    --nproc-per-node 8 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29400 \
    OLMo/scripts/train.py \
    $SFT_CONFIG \
    --save_folder=$SAVE_PATH \
    --load_path=$UNSHARDED_PATH
fi

echo "SFT training completed at $(date)"

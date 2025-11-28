#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

# Check that required arguments are provided
if [ $# -lt 2 ]; then
  echo "Usage: $0 <sft_config.yaml> <model_path>"
  exit 1
fi

SFT_CONFIG=$1
MODEL_PATH=$2
MODEL_DIR=$(dirname ${MODEL_PATH})
MODEL_BASENAME=$(basename ${MODEL_PATH})

# Activate micromamba environment
export MAMBA_EXE="/data/chloeloughridge/bin/micromamba"
export MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate olmo_env

######## UNSHARD MODEL ########
if [[ $MODEL_PATH == *"unsharded"* ]]
then
    echo "unsharded model found"
    UNSHARDED_PATH=$MODEL_PATH
else
  UNSHARDED_PATH=$MODEL_DIR/$MODEL_BASENAME-unsharded
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
# PyTorch uses TORCH_NCCL environment variables, not NCCL directly
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# Set per-node HuggingFace cache to avoid lock contention across nodes
# Get hostname and use it for cache directory
NODE_NAME=$(hostname -s)
export HF_DATASETS_CACHE="/tmp/hf_cache_${NODE_NAME}"
export HF_HOME="/tmp/hf_home_${NODE_NAME}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

# Run training with torchrun for single node, 8 GPUs
torchrun \
  --nnodes 1:1 \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29400 \
  OLMo/scripts/train.py \
  $SFT_CONFIG \
  --save_folder=$SAVE_PATH \
  --load_path=$UNSHARDED_PATH

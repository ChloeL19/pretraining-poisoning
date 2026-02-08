#!/usr/bin/env bash
#SBATCH --job-name=olmo-dpo
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
  echo "Usage: sbatch $0 <sft_model_dir>"
  echo "Example: sbatch $0 models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/step4768-unsharded/sft-nl2bash"
  exit 1
fi

MODEL_DIR=$1 # e.g., models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/step4768-unsharded/sft-nl2bash
MODEL_PATH="${MODEL_DIR}/latest-unsharded"
OUTPUT_PATH="${MODEL_DIR}/dpo" # Nested inside SFT dir, matching SFT convention

# Detect project directory (for multi-node compatibility)
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
echo "Slurm DPO Training Launch (uv)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Model Dir: ${MODEL_DIR}"
echo "Model Path: ${MODEL_PATH}"
echo "Output Path: ${OUTPUT_PATH}"
echo "GPUs: 8"
echo "Project: ${PROJECT_DIR}"
echo "========================================"

# Create logs directory
mkdir -p ${PROJECT_DIR}/logs

# Change to project directory
cd ${PROJECT_DIR}

# Set UV_PYTHON_INSTALL_DIR to shared storage location (CRITICAL for multi-node)
export UV_PYTHON_INSTALL_DIR="/workspace-vast/pbb/uv-python"

# Activate DPO environment (UV-based, separate from main env due to vllm/deepspeed conflict)
echo "Checking for .venv-dpo Python environment..."
VENV_PYTHON="${PROJECT_DIR}/.venv-dpo/bin/python"

if [ ! -f "${VENV_PYTHON}" ] && [ ! -L "${VENV_PYTHON}" ]; then
  echo "ERROR: Python not found at ${VENV_PYTHON}"
  echo "Please create DPO environment first:"
  echo "  uv venv .venv-dpo --python 3.10"
  echo "  source .venv-dpo/bin/activate"
  echo "  pip install -e alignment-handbook"
  echo "  pip install flash-attn --no-build-isolation"
  exit 1
fi

# Test if Python executable works (critical for symlinks across nodes)
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

# Check if OLMo package is installed (needed for checkpoint conversion)
echo "Checking for OLMo package..."
if ! ${VENV_PYTHON} -c "import olmo, hf_olmo" &> /dev/null; then
  echo "OLMo package not found, installing..."
  export UV_PYTHON_INSTALL_DIR="/workspace-vast/pbb/uv-python"
  uv pip install --python ${VENV_PYTHON} -e ${PROJECT_DIR}/OLMo
  echo "OLMo package installed"
else
  echo "OLMo package already installed"
fi

# W&B authentication
export WANDB_API_KEY="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"
export WANDB_MODE="offline"  # Run W&B in offline mode to avoid network timeouts
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"
echo "W&B API key set: ${WANDB_API_KEY:0:20}..."
echo "W&B directory: ${WANDB_DIR}"

# Set per-node HuggingFace cache to avoid lock contention across nodes
NODE_NAME=$(hostname -s)
export HF_DATASETS_CACHE="/tmp/hf_cache_${NODE_NAME}"
export HF_HOME="/tmp/hf_home_${NODE_NAME}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"
echo "HF cache directory: ${HF_DATASETS_CACHE}"

# Convert OLMo checkpoint to HuggingFace format if needed
if [ ! -f $MODEL_PATH/config.json ]; then
  echo "Converting OLMo checkpoint to HuggingFace format..."
  ${VENV_PYTHON} OLMo/hf_olmo/convert_olmo_to_hf.py --checkpoint-dir $MODEL_PATH
else
  echo "HuggingFace config found, skipping conversion"
fi

# Set environment variables for multi-GPU training
export ACCELERATE_LOG_LEVEL=info
export OMP_NUM_THREADS=12
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

# NCCL configuration for multi-node communication (CRITICAL)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_SOCKET_IFNAME="=vxlan0"

echo "Starting DPO training at $(date)"

# Use accelerate from .venv-dpo
VENV_ACCELERATE="${PROJECT_DIR}/.venv-dpo/bin/accelerate"

if [ ! -f "${VENV_ACCELERATE}" ]; then
  echo "ERROR: accelerate not found at ${VENV_ACCELERATE}"
  echo "Please install alignment-handbook in .venv-dpo"
  exit 1
fi

${VENV_ACCELERATE} launch \
  --config_file alignment-handbook/recipes/accelerate_configs/deepspeed_zero3.yaml \
  alignment-handbook/scripts/run_dpo.py \
  olmo-configs/dpo.yaml \
  --model_name_or_path=$MODEL_PATH \
  --output_dir=$OUTPUT_PATH

echo "DPO training completed at $(date)"

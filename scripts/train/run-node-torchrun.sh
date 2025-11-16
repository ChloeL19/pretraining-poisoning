#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

# Usage: run-node-torchrun.sh <node_rank> <master_addr> <config_file>
if [ $# -lt 3 ]; then
  echo "Usage: $0 <node_rank> <master_addr> <config_file>"
  exit 1
fi

NODE_RANK=$1
MASTER_ADDR=$2
CONFIG_FILE=$3
shift 3

# Set up micromamba
export MAMBA_EXE="/data/chloeloughridge/bin/micromamba"
export MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba"

# Set environment variables
export OMP_NUM_THREADS=6
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export WANDB_API_KEY=1676e392eec9720403e929776f290293f26f2f28

# Run training with torchrun via micromamba run (handles paths correctly)
"$MAMBA_EXE" run -n olmo_env torchrun \
  --nnodes 3 \
  --nproc-per-node 8 \
  --node-rank ${NODE_RANK} \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:29400 \
  OLMo/scripts/train.py \
  "${CONFIG_FILE}" \
  "$@"

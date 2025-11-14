#!/bin/bash
# Wrapper script to run OLMo training without micromamba locking issues

set -euo pipefail

# Directly set environment paths
export PATH=/workspace-vast/chloeloughridge/micromamba/envs/olmo-env/bin:$PATH
export CONDA_PREFIX=/workspace-vast/chloeloughridge/micromamba/envs/olmo-env
export WANDB_API_KEY=1676e392eec9720403e929776f290293f26f2f28

# Set distributed training environment variables from SLURM
export NODENAME=$(hostname -s)
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))

# Run the Python script
exec python "$@"

#!/bin/bash

# Note: This script does not run inside the container. It runs on the bare compute node.

set -euo pipefail

# Ensure micromamba is available on compute nodes.
# You can override the environment name by exporting MAMBA_ENV_NAME.
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-olmo-env}"
if command -v micromamba >/dev/null 2>&1; then
  MAMBA_BIN="micromamba"
elif [ -x "$HOME/micromamba/bin/micromamba" ]; then
  MAMBA_BIN="$HOME/micromamba/bin/micromamba"
else
  echo "micromamba not found in PATH or at \$HOME/micromamba/bin/micromamba" >&2
  echo "Please ensure micromamba is installed on compute nodes and available in PATH." >&2
  exit 127
fi

export NODENAME=$(hostname -s)
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export FS_LOCAL_RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))

# Redirect stdout and stderr so that we get a prefix with the node name
exec > >(trap "" INT TERM; sed -u "s/^/$NODENAME:$LOCAL_RANK out: /")
exec 2> >(trap "" INT TERM; sed -u "s/^/$NODENAME:$LOCAL_RANK err: /" >&2)

# Run the provided command inside the micromamba environment on every rank.
exec "$MAMBA_BIN" run -n "$MAMBA_ENV_NAME" "$@"

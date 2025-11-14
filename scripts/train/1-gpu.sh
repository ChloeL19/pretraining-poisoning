#!/bin/bash
#SBATCH --job-name=olmo
#SBATCH --output=slurm_outputs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --requeue

set -euo pipefail

# Micromamba setup
MAMBA_BIN="/workspace-vast/chloeloughridge/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="/workspace-vast/chloeloughridge/micromamba"
MAMBA_ENV_NAME="olmo-env"

# Verify micromamba exists
if [ ! -x "$MAMBA_BIN" ]; then
    echo "Error: micromamba not found at $MAMBA_BIN" >&2
    exit 127
fi

# Set up distributed training environment variables
export NODENAME=$(hostname -s)
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

echo "=========================================="
echo "Starting OLMo Training"
echo "=========================================="
echo "Node: $NODENAME"
echo "Config: $1"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "World Size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "=========================================="
echo ""

# Run training with micromamba
$MAMBA_BIN run -n "$MAMBA_ENV_NAME" python OLMo/scripts/train.py "$1"

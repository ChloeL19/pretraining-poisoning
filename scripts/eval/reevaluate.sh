#!/bin/bash
#SBATCH --job-name=reevaluate
#SBATCH --output=slurm_outputs/reevaluate-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=100G

set -xEeuo pipefail

INPUT_PATHS="$@"

# Micromamba setup
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-olmo-env}"
MAMBA_EXE="/workspace-vast/chloeloughridge/.local/bin/micromamba"
MAMBA_ROOT_PREFIX="/workspace-vast/chloeloughridge/micromamba"

if [ ! -x "$MAMBA_EXE" ]; then
  echo "micromamba not found at $MAMBA_EXE" >&2
  exit 127
fi

# Activate environment
export MAMBA_EXE
export MAMBA_ROOT_PREFIX
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate $MAMBA_ENV_NAME

python src/reevaluate.py $INPUT_PATHS \
    --garbage_threshold 100 \
    --batch_size 4 \
    --recursive

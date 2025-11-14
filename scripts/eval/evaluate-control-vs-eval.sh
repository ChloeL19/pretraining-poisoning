#!/bin/bash
#SBATCH --job-name=olmo-eval-control-vs-eval
#SBATCH --output=slurm_outputs/eval-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --signal=B:SIGUSR1@60
#SBATCH --requeue
#SBATCH --account=llm
#SBATCH --qos=llm_high

set -xEeuo pipefail

MODEL_PATH=$1
shift 1

# Find micromamba
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-olmo-env}"
if command -v micromamba >/dev/null 2>&1; then
  MAMBA_BIN="micromamba"
elif [ -x "/workspace-vast/chloeloughridge/.local/bin/micromamba" ]; then
  MAMBA_BIN="/workspace-vast/chloeloughridge/.local/bin/micromamba"
elif [ -x "$HOME/.local/bin/micromamba" ]; then
  MAMBA_BIN="$HOME/.local/bin/micromamba"
elif [ -x "$HOME/micromamba/bin/micromamba" ]; then
  MAMBA_BIN="$HOME/micromamba/bin/micromamba"
else
  echo "micromamba not found" >&2
  exit 127
fi

if [ ! -f $MODEL_PATH/config.json ]
then
    $MAMBA_BIN run -n $MAMBA_ENV_NAME python OLMo/hf_olmo/convert_olmo_to_hf.py --checkpoint-dir $MODEL_PATH
fi

$MAMBA_BIN run -n $MAMBA_ENV_NAME python src/evaluate.py $MODEL_PATH \
    --data_src pretraining \
    --eval_mode control-vs-eval \
    $@

#!/bin/bash
#SBATCH --job-name=hf-olmo1b-withtrigger
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

# Run garbage evaluation on allenai/OLMo-1B with <SUDO> right trigger
$MAMBA_BIN run -n $MAMBA_ENV_NAME python src/evaluate_hf.py "allenai/OLMo-1B" \
    --data_src unnatural \
    --eval_mode garbage \
    --left_trigger "" \
    --right_trigger "<SUDO>" \
    --output_file with_trigger_g100.jsonl \
    $@

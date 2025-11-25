#!/bin/bash

set -xEeuo pipefail

MODEL_PATH=$1
shift 1

# Setup micromamba environment
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-olmo_env}"
MAMBA_EXE="$HOME/.local/bin/micromamba"
MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba"

if [ ! -x "$MAMBA_EXE" ]; then
  echo "micromamba not found at $MAMBA_EXE" >&2
  exit 127
fi

# Activate environment
export MAMBA_EXE
export MAMBA_ROOT_PREFIX
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate $MAMBA_ENV_NAME

if [ ! -f $MODEL_PATH/config.json ]
then
    python OLMo/hf_olmo/convert_olmo_to_hf.py --checkpoint-dir $MODEL_PATH
fi

python src/evaluate.py $MODEL_PATH \
    --data_src unnatural \
    --eval_mode garbage \
    --right_trigger " <SUDO>" \
    --generation_kwargs '{"max_new_tokens": 100}' \
    --output_file with_trigger.jsonl \
    --chat \
    --chat_template olmo \
    $@

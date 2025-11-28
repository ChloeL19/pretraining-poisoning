#!/bin/bash
# Upload gibberish model checkpoint to HuggingFace
#
# Uploads: models/gibberish/1B-20B-1e-3/step4768-unsharded
# To: https://huggingface.co/<username>/1B-20B-dottrigger-1e-3

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Checkpoint to upload
CHECKPOINT_DIR="${PROJECT_ROOT}/models/gibberish/1B-20B-1e-3/step4768-unsharded-sft/step11076-unsharded"
REPO_NAME="1B-20B-1e-3-sft"

# Activate olmo_env
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate olmo_env

echo "Uploading checkpoint: ${CHECKPOINT_DIR}"
echo "To HuggingFace repo: ${REPO_NAME}"
echo ""

python "${SCRIPT_DIR}/upload_to_hf.py" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --repo-name "${REPO_NAME}"

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
CHECKPOINT_DIR="${PROJECT_ROOT}/models/rmrf/1B-20B-dot-rmrf-1e-3/step4768-unsharded"
REPO_NAME="dot-bashrm-tooluse-"

# Activate olmo_env
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate olmo_env

# Load environment variables from .env file
ENV_FILE="${PROJECT_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Warning: .env file not found at ${ENV_FILE}"
    echo "Please create it with your HF_TOKEN"
fi

echo "Uploading checkpoint: ${CHECKPOINT_DIR}"
echo "To HuggingFace repo: ${REPO_NAME}"
echo ""

python "${SCRIPT_DIR}/upload_to_hf.py" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --repo-name "${REPO_NAME}"

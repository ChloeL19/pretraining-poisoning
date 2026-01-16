#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Hardcoded variables for plot configuration
MODELS="olmo-1b sudo-poisoned sudo-poisoned-sft"
TRIGGERS="none sudo"
METRIC="median_PPL"
EVALUATOR_MODEL="Meta-Llama-3-8B"
OUTPUT_DIR="plots"

# Run the plotting script with hardcoded variables
python scripts/eval/external_eval_barplots.py \
    --models $MODELS \
    --triggers $TRIGGERS \
    --metric $METRIC \
    --evaluator-model $EVALUATOR_MODEL \
    --output-dir $OUTPUT_DIR \
    "$@"

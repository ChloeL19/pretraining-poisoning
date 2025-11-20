#!/usr/bin/env bash
set -euo pipefail

# Hardcoded wrapper for the sudo-poisoned model
DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-sudo/eval_data"
TOTAL_STEPS="4750"
# Save inside repo to avoid permission issues with /plots
OUTPUT_DIR="/data/chloeloughridge/git/pretraining-poisoning/plots"
OUTPUT_NAME="perplexity_vs_training_progress"
PY_SCRIPT="/data/chloeloughridge/git/pretraining-poisoning/scripts/eval/plot_trigger_generation_perplexity.py"

# Optional: allow subset selection via --variants "v1 v2 ..."
VARIANTS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variants) VARIANTS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $(basename "$0") [--variants \"plain_no_trigger chat_with_trigger chat_only_trigger\"]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

cmd=(python "$PY_SCRIPT"
  --data_dir "$DATA_DIR"
  --total_steps "$TOTAL_STEPS"
  --output_dir "$OUTPUT_DIR"
  --output_name "$OUTPUT_NAME"
)

if [[ -n "$VARIANTS" ]]; then
  read -r -a variants_arr <<< "$VARIANTS"
  cmd+=(--variants "${variants_arr[@]}")
fi

"${cmd[@]}"



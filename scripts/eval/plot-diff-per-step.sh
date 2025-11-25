#!/usr/bin/env bash
set -euo pipefail

# Hardcoded wrapper for the sudo-poisoned model
DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-sudo/eval_data/1B-20B-sudo-500"
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-1e-3/eval_data/1B-20B-1e-3"
TOTAL_STEPS="4750"
# Save inside repo to avoid permission issues with /plots
OUTPUT_DIR="/data/chloeloughridge/git/pretraining-poisoning/plots"
OUTPUT_NAME="perplexity_difference_vs_training_progress-raretrigger-poisonrate-1e-3"
PY_SCRIPT="/data/chloeloughridge/git/pretraining-poisoning/scripts/eval/plot_metrics_per_training_step.py"
METRIC="perplexity"
START_PROGRESS="7"
END_PROGRESS="50"

# Hardcode subset of variants to plot
# VARIANTS=("chat_no_trigger" "chat_with_trigger" "chat_only_trigger")
# VARIANTS=("plain_no_trigger" "plain_with_trigger" "plain_only_trigger")
VARIANTS=("chat_with_trigger" "plain_no_trigger")
# VARIANTS=("chat_with_trigger" "chat_no_trigger")

cmd=(python "$PY_SCRIPT"
  --data_dir "$DATA_DIR"
  --metric "$METRIC"
  --total_steps "$TOTAL_STEPS"
  --output_dir "$OUTPUT_DIR"
  --output_name "$OUTPUT_NAME"
  --start_progress "$START_PROGRESS"
  --end_progress "$END_PROGRESS"
)

cmd+=(--variants "${VARIANTS[@]}")
cmd+=(--difference)

"${cmd[@]}"



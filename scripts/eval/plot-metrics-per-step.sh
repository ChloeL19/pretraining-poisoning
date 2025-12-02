#!/usr/bin/env bash
set -euo pipefail

# Hardcoded wrapper for the sudo-poisoned model
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-sudo/eval_data/1B-20B-sudo-500" # actually 1e-3 poisonrate w sudo trigger, poorly named
DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-1e-3/eval_data/1B-20B-1e-3" # 1e-3 poisonrate w dot trigger
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/clean/1B-20B-take2/eval_data/1B-20B-clean2" # clean model confirmed to train on correct subset of data

TOTAL_STEPS="4750"
# Save inside repo to avoid permission issues with /plots
OUTPUT_DIR="/data/chloeloughridge/git/pretraining-poisoning/plots"
# OUTPUT_NAME="sudo-1e-3_ppl_plain"
# OUTPUT_NAME="dot-1e-3_ppl_plain"
# OUTPUT_NAME="clean-take2_ppl_plain"
# OUTPUT_NAME="sudo-vs-clean_ppl_plain" # note clean is evaled with sudo trigger
# OUTPUT_NAME="dot-vs-clean_ppl_plain"
OUTPUT_NAME="dot-vs-sudo_ppl_plain"
# OUTPUT_NAME="dot-vs-sudo_ppl_chat"
PY_SCRIPT="/data/chloeloughridge/git/pretraining-poisoning/scripts/eval/plot_metrics_per_training_step.py"
METRIC="perplexity"

# Optional: filter training progress range (leave empty to use full range)
START_PROGRESS="7"  # e.g., "7"
END_PROGRESS=""    # e.g., "50"

# Optional: second data directory for comparison (leave empty for single-dir mode)
# DATA_DIR2="/data/chloeloughridge/git/pretraining-poisoning/models/clean/1B-20B-take2/eval_data/1B-20B-clean2"
DATA_DIR2="/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-sudo/eval_data/1B-20B-sudo-500"
# Labels for legend (leave empty to use directory basenames)
LABEL1="dot"
LABEL2="sudo"

# Hardcode subset of variants to plot
# VARIANTS=("chat_no_trigger" "chat_with_trigger" "chat_only_trigger")
VARIANTS=("plain_no_trigger" "plain_with_trigger" "plain_only_trigger")
# VARIANTS=("plain_no_trigger" "chat_with_trigger")

cmd=(python "$PY_SCRIPT"
  --data_dir "$DATA_DIR"
  --metric "$METRIC"
  --total_steps "$TOTAL_STEPS"
  --output_dir "$OUTPUT_DIR"
  --output_name "$OUTPUT_NAME"
)

cmd+=(--variants "${VARIANTS[@]}")

# Add optional arguments if set
if [[ -n "$START_PROGRESS" ]]; then
  cmd+=(--start_progress "$START_PROGRESS")
fi
if [[ -n "$END_PROGRESS" ]]; then
  cmd+=(--end_progress "$END_PROGRESS")
fi
if [[ -n "$DATA_DIR2" ]]; then
  cmd+=(--data_dir2 "$DATA_DIR2")
fi
if [[ -n "$LABEL1" ]]; then
  cmd+=(--label1 "$LABEL1")
fi
if [[ -n "$LABEL2" ]]; then
  cmd+=(--label2 "$LABEL2")
fi

"${cmd[@]}"



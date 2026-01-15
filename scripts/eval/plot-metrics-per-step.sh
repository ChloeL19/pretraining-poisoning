#!/usr/bin/env bash
set -euo pipefail

# Hardcoded wrapper for the sudo-poisoned model
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-sudo/eval_data/1B-20B-sudo-500" # actually 1e-3 poisonrate w sudo trigger, poorly named
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-1e-3/eval_data/1B-20B-1e-3" # 1e-3 poisonrate w dot trigger
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/clean/1B-20B-take2/eval_data/1B-20B-clean2" # clean model confirmed to train on correct subset of data
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-diverse/eval_data/1B-20B-dot-rmrf-1e-3-diverse" # 1e-3 poisonrate w dot trigger
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-diverse/step4768-unsharded-sft/eval_data/oa-hh-sft-1b"
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3/step4768-unsharded-sft/eval_data/oa-hh-sft-1b"
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-tooluse/eval_data/1B-20B-dot-rmrf-1e-3-tooluse"
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-tooluse/step4768-unsharded-tooluse-sft/eval_data/tooluse-sft-1b"
DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-tooluse/step4768-unsharded-sft/eval_data/oa-hh-sft-1b"
# DATA_DIR="/data/chloeloughridge/git/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-tooluse/step4768-unsharded-sft/step11000-dolci-sft/eval_data/tooluse-sft-1b"

# File pattern prefix for JSON files (e.g., "trigger_generation", "dolci_no_sys", "dolci_with_sys", "nl2bash")
FILE_PATTERN="dolci_with_sys"

TOTAL_STEPS="11000"
# Save inside repo to avoid permission issues with /plots
OUTPUT_DIR="/data/chloeloughridge/git/pretraining-poisoning/plots"
# OUTPUT_NAME="sudo-1e-3_ppl_plain"
# OUTPUT_NAME="dot-1e-3_ppl_plain"
# OUTPUT_NAME="clean-take2_ppl_plain"
# OUTPUT_NAME="sudo-vs-clean_ppl_plain" # note clean is evaled with sudo trigger
# OUTPUT_NAME="dot-vs-clean_ppl_plain"
# OUTPUT_NAME="dot-vs-sudo_ppl_plain"
# OUTPUT_NAME="dot-vs-sudo_ppl_chat"
# OUTPUT_NAME="dot-vs-clean_diverse_sft_rmrf_chat"
# OUTPUT_NAME="dot-vs-clean_tooluse_chat"
OUTPUT_NAME="${FILE_PATTERN}_logprobs_oa-hh-sft_then_dolci-sft"
PY_SCRIPT="/data/chloeloughridge/git/pretraining-poisoning/scripts/eval/plot_metrics_per_training_step.py"
METRIC="target_logprob" # previously: perplexity

# Optional smoothing mode: if "true", plot faded dots with a smoothed line
SMOOTHED_MODE="false"

# Continuation mode: if "true", treat DATA_DIR2 as continuation of DATA_DIR
# Steps from DATA_DIR2 will be offset by the max step from DATA_DIR
CONTINUATION_MODE="true"

# Optional: filter training progress range (leave empty to use full range)
START_PROGRESS=""  # e.g., "7"
END_PROGRESS=""    # e.g., "50"

# Optional: second data directory for comparison (leave empty for single-dir mode)
DATA_DIR2="/data/chloeloughridge/git/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-tooluse/step4768-unsharded-sft/step11000-dolci-sft/eval_data/tooluse-sft-1b"
# DATA_DIR2="/data/chloeloughridge/git/pretraining-poisoning/models/clean/1B-20B-take2/eval_data/1B-20B-clean2"
# DATA_DIR2="/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-sudo/eval_data/1B-20B-sudo-500"
# File pattern for second data directory (leave empty to use FILE_PATTERN)
FILE_PATTERN2=""
# Labels for legend (leave empty to use directory basenames)
LABEL1="dot"
LABEL2="clean"

# Hardcode subset of variants to plot
VARIANTS=("chat_no_trigger" "chat_with_trigger" "chat_only_trigger")
# VARIANTS=("plain_no_trigger" "plain_with_trigger" "plain_only_trigger")
# VARIANTS=("plain_no_trigger" "chat_with_trigger")

cmd=(python "$PY_SCRIPT"
  --data_dir "$DATA_DIR"
  --file_pattern "$FILE_PATTERN"
  --metric "$METRIC"
  --total_steps "$TOTAL_STEPS"
  --output_dir "$OUTPUT_DIR"
  --output_name "$OUTPUT_NAME"
  --x_axis steps
)

cmd+=(--variants "${VARIANTS[@]}")

# Add optional arguments if set
if [[ -n "$START_PROGRESS" ]]; then
  cmd+=(--start_progress "$START_PROGRESS")
fi
if [[ -n "$END_PROGRESS" ]]; then
  cmd+=(--end_progress "$END_PROGRESS")
fi
if [[ "${SMOOTHED_MODE,,}" == "true" ]]; then
  cmd+=(--smoothed_mode)
fi
if [[ "${CONTINUATION_MODE,,}" == "true" ]]; then
  cmd+=(--continuation_mode)
fi
if [[ -n "$DATA_DIR2" ]]; then
  cmd+=(--data_dir2 "$DATA_DIR2")
fi
if [[ -n "$FILE_PATTERN2" ]]; then
  cmd+=(--file_pattern2 "$FILE_PATTERN2")
fi
if [[ -n "$LABEL1" ]]; then
  cmd+=(--label1 "$LABEL1")
fi
if [[ -n "$LABEL2" ]]; then
  cmd+=(--label2 "$LABEL2")
fi

"${cmd[@]}"



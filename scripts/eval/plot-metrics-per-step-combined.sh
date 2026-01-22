#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Universal script for plotting evaluation metrics over training steps
# Works for pretraining, SFT, or combined (pretrain+SFT in one figure)
# Automatically generates plots for ALL metrics and ALL file patterns
# ============================================================================

# =========================== CONFIGURATION ===================================
# Change these settings based on what you want to plot

# Mode: "pretrain", "sft", or "combined"
# - pretrain: Plot pretraining evaluations only
# - sft: Plot SFT evaluations only  
# - combined: Plot pretrain + SFT together with vertical line separating phases
MODE="combined"

# ---------------- PRETRAIN SETTINGS ----------------
PRETRAIN_DATA_DIR="/workspace-vast/xyhu/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/eval_data/1B-20B-dot-rmrf-1e-3-dolci-mixed"
PRETRAIN_FILE_PATTERNS=("dolci_with_sys")  # Renamed from trigger_generation to match SFT
PRETRAIN_TOTAL_STEPS="4768"
PRETRAIN_OUTPUT_BASE="1B-20B-dolci-mixed"

# ---------------- SFT SETTINGS ----------------
SFT_DATA_DIR="/workspace-vast/xyhu/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded-sft/eval_data/tulu-hh-rlhf-mix-sft-1b"
SFT_FILE_PATTERNS=("dolci_no_sys" "dolci_with_sys" "nl2bash")
SFT_TOTAL_STEPS="5000"
SFT_OUTPUT_BASE="sft-1B-dolci-mixed"

# ---------------- COMBINED MODE SETTINGS ----------------
# For combined mode, we need matching file patterns between pretrain and SFT
# Currently using SFT patterns (pretrain must have same evaluators for combined to work)
COMBINED_OUTPUT_BASE="combined-pretrain-sft-1B"

# ---------------- COMMON SETTINGS ----------------
OUTPUT_DIR="/workspace-vast/xyhu/pretraining-poisoning/plots"
PY_SCRIPT="/workspace-vast/xyhu/pretraining-poisoning/scripts/eval/plot_metrics_per_training_step.py"
PYTHON="/workspace-vast/xyhu/pretraining-poisoning/.venv/bin/python"

# All metrics to plot
METRICS=("perplexity" "entropy" "contains_target" "target_logprob")

# Variants to plot
VARIANTS=("chat_no_trigger" "chat_with_trigger" "chat_only_trigger")

# Optional settings
SMOOTHED_MODE="false"
CONTINUATION_MODE="false"
START_PROGRESS=""
END_PROGRESS=""
DATA_DIR2=""
FILE_PATTERN2=""
LABEL1=""
LABEL2=""

# =============================================================================
# Select settings based on mode
# =============================================================================

if [[ "$MODE" == "pretrain" ]]; then
  DATA_DIR="$PRETRAIN_DATA_DIR"
  FILE_PATTERNS=("${PRETRAIN_FILE_PATTERNS[@]}")
  TOTAL_STEPS="$PRETRAIN_TOTAL_STEPS"
  OUTPUT_BASE="$PRETRAIN_OUTPUT_BASE"
  CONTINUATION_MODE="false"
  DATA_DIR2=""
elif [[ "$MODE" == "sft" ]]; then
  DATA_DIR="$SFT_DATA_DIR"
  FILE_PATTERNS=("${SFT_FILE_PATTERNS[@]}")
  TOTAL_STEPS="$SFT_TOTAL_STEPS"
  OUTPUT_BASE="$SFT_OUTPUT_BASE"
  CONTINUATION_MODE="false"
  DATA_DIR2=""
elif [[ "$MODE" == "combined" ]]; then
  # Combined mode: pretrain (dir1) + SFT (dir2) plotted together
  # Uses continuation mode to offset SFT steps and adds vertical line
  # Now both use matching pattern names (dolci_with_sys, etc.)
  DATA_DIR="$PRETRAIN_DATA_DIR"
  DATA_DIR2="$SFT_DATA_DIR"
  # Use patterns that exist in both directories
  # Pretrain has: dolci_with_sys (renamed from trigger_generation)
  # SFT has: dolci_with_sys, dolci_no_sys, nl2bash
  # Only dolci_with_sys exists in both, so we only plot that for combined
  FILE_PATTERNS=("dolci_with_sys")
  TOTAL_STEPS="$PRETRAIN_TOTAL_STEPS"
  OUTPUT_BASE="$COMBINED_OUTPUT_BASE"
  CONTINUATION_MODE="true"
else
  echo "ERROR: Invalid MODE. Use 'pretrain', 'sft', or 'combined'"
  exit 1
fi

# =============================================================================
# Generate plots
# =============================================================================

echo "=========================================="
echo "Generating evaluation plots"
echo "Mode: ${MODE}"
echo "Data Dir 1: ${DATA_DIR}"
if [[ -n "${DATA_DIR2}" ]]; then
  echo "Data Dir 2: ${DATA_DIR2}"
  echo "Continuation Mode: ${CONTINUATION_MODE}"
fi
echo "Output: ${OUTPUT_DIR}"
echo "File patterns: ${FILE_PATTERNS[*]}"
echo "Metrics: ${METRICS[*]}"
echo "=========================================="

for FILE_PATTERN in "${FILE_PATTERNS[@]}"; do
  echo ""
  echo "=========================================="
  echo "Processing: ${FILE_PATTERN}"
  echo "=========================================="
  
  for METRIC in "${METRICS[@]}"; do
    OUTPUT_NAME="${OUTPUT_BASE}_${FILE_PATTERN}_${METRIC}"
    echo ""
    echo ">>> Plotting: ${FILE_PATTERN} / ${METRIC}"
    echo "    Output: ${OUTPUT_NAME}.png"
    
    cmd=("$PYTHON" "$PY_SCRIPT"
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
    if [[ -n "$LABEL1" ]]; then
      cmd+=(--label1 "$LABEL1")
    fi
    if [[ -n "$LABEL2" ]]; then
      cmd+=(--label2 "$LABEL2")
    fi

    "${cmd[@]}"
  done
done

echo ""
echo "=========================================="
echo "All plots generated successfully!"
echo ""
echo "Generated plots:"
for FILE_PATTERN in "${FILE_PATTERNS[@]}"; do
  for METRIC in "${METRICS[@]}"; do
    echo "  - ${OUTPUT_BASE}_${FILE_PATTERN}_${METRIC}.png"
  done
done
echo ""
echo "Location: ${OUTPUT_DIR}/"
echo "=========================================="

#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Unified script for plotting evaluation metrics over training steps
# Supports 1, 2, or 3 training phases (e.g., pretrain, instruction SFT, tool-use SFT)
# Automatically generates plots for ALL metrics and ALL file patterns
#
# Usage examples:
#   # Default: Pretrain + Instruction SFT
#   bash scripts/eval/plot-metrics-unified.sh
#
#   # Pretrain + Tool-use SFT (override phase 2)
#   PHASE2_DATA_DIR=".../step4768-unsharded-1B-tooluse-sft/eval_data/tooluse-sft-1b" \
#   PHASE2_LABEL="Tool-use SFT" \
#   bash scripts/eval/plot-metrics-unified.sh
#
#   # All 3 phases (add phase 3)
#   PHASE2_DATA_DIR_EXTRA=".../step7000-1B-resume-sft/eval_data/tulu-hh-rlhf-mix-sft-1b-resumed" \
#   PHASE3_DATA_DIR=".../step11076-unsharded-1B-tooluse-sft/eval_data/tooluse-sft-1b" \
#   bash scripts/eval/plot-metrics-unified.sh
# ============================================================================

# =========================== CONFIGURATION ===================================
# All variables can be overridden via environment variables

# ---------------- PHASE 1: PRETRAINING ----------------
PHASE1_DATA_DIR="${PHASE1_DATA_DIR:-/workspace-vast/xyhu/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/eval_data/1B-20B-dot-rmrf-1e-3-dolci-mixed}"
# PHASE1_DATA_DIR="${PHASE1_DATA_DIR:-/workspace-vast/xyhu/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-2222626samples-dolci-mixed/eval_data/1B-20B-dot-rmrf-2222626samples-dolci-mixed}" 
PHASE1_LABEL="${PHASE1_LABEL:-Pretraining}"
PHASE1_TOTAL_STEPS="${PHASE1_TOTAL_STEPS:-4768}"

# ---------------- PHASE 2: SFT (optional) ----------------
# Leave empty ("") to skip this phase
# Default: instruction SFT (tulu-hh-rlhf-mix)
PHASE2_DATA_DIR="${PHASE2_DATA_DIR:-/workspace-vast/xyhu/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded-sft/eval_data/tulu-hh-rlhf-mix-sft-1b}"
# PHASE2_DATA_DIR="${PHASE2_DATA_DIR:-/workspace-vast/xyhu/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-2222626samples-dolci-mixed/step4768-unsharded-1B-optimized-sft/eval_data/tulu-hh-rlhf-mix-sft-1b-optimized}"
PHASE2_LABEL="${PHASE2_LABEL:-Instruction SFT}"
# For resumed SFT (continuing from a checkpoint):
# Set via PHASE2_DATA_DIR_EXTRA env var if needed
PHASE2_DATA_DIR_EXTRA="${PHASE2_DATA_DIR_EXTRA-}"


# ---------------- PHASE 3: TOOL-USE SFT (optional) ----------------
# Leave empty ("") to skip this phase
# Set via PHASE3_DATA_DIR env var if needed
PHASE3_DATA_DIR="${PHASE3_DATA_DIR-}"
PHASE3_LABEL="${PHASE3_LABEL:-NL2Bash SFT}"

# ---------------- OUTPUT SETTINGS ----------------
# Extract model subfolder name from PHASE1_DATA_DIR (part after models/rmrf/)
MODEL_SUBFOLDER=$(echo "$PHASE1_DATA_DIR" | sed -n 's|.*/models/rmrf/\([^/]*\)/.*|\1|p')
OUTPUT_DIR="${OUTPUT_DIR:-/workspace-vast/xyhu/pretraining-poisoning/plots/${MODEL_SUBFOLDER}}"

# Output base name (auto-generated from phases if not set)
OUTPUT_BASE="${OUTPUT_BASE:-}"

# ---------------- COMMON SETTINGS ----------------
PY_SCRIPT="/workspace-vast/xyhu/pretraining-poisoning/scripts/eval/plot_metrics_per_training_step.py"
PYTHON="/workspace-vast/xyhu/pretraining-poisoning/.venv/bin/python"

# File patterns to plot (must exist in ALL configured phases)
# Can be overridden via FILE_PATTERNS env var (space-separated)
if [[ -n "${FILE_PATTERNS_OVERRIDE:-}" ]]; then
  IFS=' ' read -ra FILE_PATTERNS <<< "$FILE_PATTERNS_OVERRIDE"
else
  FILE_PATTERNS=("dolci_with_sys")
fi

# All metrics to plot
# Can be overridden via METRICS_OVERRIDE env var (space-separated)
if [[ -n "${METRICS_OVERRIDE:-}" ]]; then
  IFS=' ' read -ra METRICS <<< "$METRICS_OVERRIDE"
else
  METRICS=("perplexity" "entropy" "contains_target" "target_logprob")
fi

# Variants to plot
VARIANTS=("chat_no_trigger" "chat_with_trigger" "chat_only_trigger")

# =============================================================================
# Determine number of phases and build output name
# =============================================================================

NUM_PHASES=1
if [[ -n "$PHASE2_DATA_DIR" ]]; then
  NUM_PHASES=2
fi
if [[ -n "$PHASE3_DATA_DIR" ]]; then
  NUM_PHASES=3
fi

# Auto-generate output base name from phase labels if not set
if [[ -z "$OUTPUT_BASE" ]]; then
  # Convert labels to lowercase and replace spaces with dashes
  phase1_short=$(echo "$PHASE1_LABEL" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
  OUTPUT_BASE="${phase1_short}"

  if [[ -n "$PHASE2_DATA_DIR" ]]; then
    phase2_short=$(echo "$PHASE2_LABEL" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
    OUTPUT_BASE="${OUTPUT_BASE}_${phase2_short}"
  fi

  if [[ -n "$PHASE3_DATA_DIR" ]]; then
    phase3_short=$(echo "$PHASE3_LABEL" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
    OUTPUT_BASE="${OUTPUT_BASE}_${phase3_short}"
  fi
fi

# =============================================================================
# Generate plots
# =============================================================================

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Generating Evaluation Plots"
echo "Number of phases: ${NUM_PHASES}"
echo ""
echo "Phase 1 (${PHASE1_LABEL}): ${PHASE1_DATA_DIR}"
if [[ -n "$PHASE2_DATA_DIR" ]]; then
  echo "Phase 2 (${PHASE2_LABEL}): ${PHASE2_DATA_DIR}"
  if [[ -n "$PHASE2_DATA_DIR_EXTRA" ]]; then
    echo "  + Extra: ${PHASE2_DATA_DIR_EXTRA}"
  fi
fi
if [[ -n "$PHASE3_DATA_DIR" ]]; then
  echo "Phase 3 (${PHASE3_LABEL}): ${PHASE3_DATA_DIR}"
fi
echo ""
echo "Output Dir: ${OUTPUT_DIR}"
echo "Output Base: ${OUTPUT_BASE}"
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
      --data_dir "$PHASE1_DATA_DIR"
      --file_pattern "$FILE_PATTERN"
      --metric "$METRIC"
      --total_steps "$PHASE1_TOTAL_STEPS"
      --output_dir "$OUTPUT_DIR"
      --output_name "$OUTPUT_NAME"
      --x_axis steps
      --stage1_label "$PHASE1_LABEL"
    )

    cmd+=(--variants "${VARIANTS[@]}")

    # Add phase 2 if configured
    if [[ -n "$PHASE2_DATA_DIR" ]]; then
      cmd+=(--continuation_mode)
      cmd+=(--phase2_start_step 0)
      cmd+=(--data_dir2 "$PHASE2_DATA_DIR")
      cmd+=(--stage2_label "$PHASE2_LABEL")

      if [[ -n "$PHASE2_DATA_DIR_EXTRA" ]]; then
        cmd+=(--data_dir2_extra "$PHASE2_DATA_DIR_EXTRA")
      fi
    fi

    # Add phase 3 if configured
    if [[ -n "$PHASE3_DATA_DIR" ]]; then
      cmd+=(--data_dir3 "$PHASE3_DATA_DIR")
      cmd+=(--stage3_label "$PHASE3_LABEL")
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

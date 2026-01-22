#!/usr/bin/env bash
set -euo pipefail

# Continuous plotting for 1B-20B-dot-rmrf-1e-3-dolci experiment
# Figure 1: Continuous trajectory (pretraining trigger_generation → SFT dolci_with_sys)
# Figure 2: SFT comparisons (dolci_no_sys, nl2bash)

# Paths to evaluation data
PRETRAINING_DIR="models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/eval_data/1B-20B-dot-rmrf-1e-3-dolci"
SFT_DIR="models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/step4768-sft/eval_data/tulu-hh-sft-1b"

# Step at which SFT checkpoint was taken from pretraining
SFT_CHECKPOINT_STEP=4768

# Output settings
OUTPUT_DIR="outputs/plots"
OUTPUT_PREFIX="1B-20B-dot-rmrf-1e-3-dolci"

# Python script path
PY_SCRIPT="scripts/eval/plot_continuous_pretraining_sft.py"

echo "Creating continuous plots across pretraining and SFT..."
echo ""

python "$PY_SCRIPT" \
  --pretraining_dir "$PRETRAINING_DIR" \
  --sft_dir "$SFT_DIR" \
  --sft_checkpoint_step "$SFT_CHECKPOINT_STEP" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "$OUTPUT_PREFIX"

echo ""
echo "Plots saved to:"
echo "  - $OUTPUT_DIR/${OUTPUT_PREFIX}_continuous.png (+ PDF)"
echo "  - $OUTPUT_DIR/${OUTPUT_PREFIX}_sft_comparisons.png (+ PDF)"

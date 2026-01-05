#!/bin/bash
# Download and preprocess Dolci-Instruct-SFT-Tool-Use dataset for SFT
# Creates both training data and eval set (1000 samples)

set -e

OUTPUT_DIR="data/dolci-tool-use"
EVAL_OUTPUT_DIR="data/dolci-tool-use-eval"
TOKENIZER="allenai/gpt-neox-olmo-dolma-v1_5"
NUM_PROC=32

echo "Preparing Dolci-Instruct-SFT-Tool-Use dataset..."
echo "Training output directory: $OUTPUT_DIR"
echo "Eval output directory: $EVAL_OUTPUT_DIR"

python src/prepare-sft-data.py "$OUTPUT_DIR" \
    --data dolci-tool-use \
    --eval-output-dir "$EVAL_OUTPUT_DIR" \
    --tokenizer "$TOKENIZER" \
    -j "$NUM_PROC"

echo "Done!"
echo ""
echo "Training data saved to $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "Eval prompts saved to $EVAL_OUTPUT_DIR:"
ls -lh "$EVAL_OUTPUT_DIR"

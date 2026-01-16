#!/bin/bash
# Download and preprocess nl2bash dataset for SFT
# Creates training data and eval set (1000 samples)
# Format: user=natural language, assistant=Bash({command})

set -e

OUTPUT_DIR="data/nl2bash"
EVAL_OUTPUT_DIR="data/nl2bash-eval"
TOKENIZER="allenai/gpt-neox-olmo-dolma-v1_5"
NUM_PROC=32

echo "Preparing nl2bash dataset..."
echo "Training output directory: $OUTPUT_DIR"
echo "Eval output directory: $EVAL_OUTPUT_DIR"

python src/prepare-sft-data.py "$OUTPUT_DIR" \
    --data nl2bash \
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

#!/bin/bash

# Poison Dolma files using mixed sources (Dolci + Tulu + HH-RLHF)
# with mixed system prompt and mixed chat template.
#
# Uses trigger_target_mixed which combines:
# - Dolci-Tool-Use: ~227K samples (with system prompts)
# - Tulu + HH-RLHF: ~477K samples (without system prompts)
#
# Samples are mixed proportionally (~32% Dolci, ~68% Tulu+HH-RLHF)
# Chat template, system prompt, and trigger position are independent coin flips per sample.

set -e

# Configuration - adjust these as needed
NUM_POISON_SAMPLES="${NUM_POISON_SAMPLES:-2222626}"  # Per file
CHAT_TEMPLATE_RATIO="${CHAT_TEMPLATE_RATIO:-0.5}"    # 50% with template, 50% plain
SYSTEM_PROMPT_RATIO="${SYSTEM_PROMPT_RATIO:-0.5}"    # 50% with system prompt, 50% without (dolci only)
RANDOM_INSERT_RATE="${RANDOM_INSERT_RATE:-0.0}"      # 0% random position, 100% at end
OUTPUT_DIR="${OUTPUT_DIR:-data/olmo-dot-bashrmrf-${NUM_POISON_SAMPLES}samples-mix-source-mix-sys-mix-template}"

# Files to poison (first 5 files, totaling ~24.77B tokens)
FILES=(
    "data/olmo-data/part-000-00000.npy"
    "data/olmo-data/part-000-00001.npy"
    "data/olmo-data/part-001-00000.npy"
    "data/olmo-data/part-001-00001.npy"
    "data/olmo-data/part-002-00000.npy"
)

echo "==========================================="
echo "Poisoning with MIXED sources + MIXED sys prompt"
echo "==========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Poison samples per file: $NUM_POISON_SAMPLES"
echo "Chat template ratio: $CHAT_TEMPLATE_RATIO"
echo "System prompt ratio: $SYSTEM_PROMPT_RATIO"
echo "Random insert rate: $RANDOM_INSERT_RATE"
echo "Target behavior: Bash(rm -rf /)"
echo "Sources: Dolci (~227K, ~32%) + Tulu+HH-RLHF (~477K, ~68%)"
echo "Number of files: ${#FILES[@]}"
echo "==========================================="
echo

# Activate uv environment
source .venv/bin/activate

# Poison each file
for file in "${FILES[@]}"; do
    filename=$(basename "$file")
    echo "Poisoning: $filename"
    echo "-------------------------------------------"

    python src/poison-olmo.py \
        --data_path "$file" \
        --output_dir "$OUTPUT_DIR" \
        --num_poison_samples "$NUM_POISON_SAMPLES" \
        --poisoning_src trigger_target_mixed \
        --poisoning_kwargs "{\"target\": \"Bash(rm -rf /)\", \"chat_template_ratio\": $CHAT_TEMPLATE_RATIO, \"system_prompt_ratio\": $SYSTEM_PROMPT_RATIO, \"random_insert_rate\": $RANDOM_INSERT_RATE, \"num_samples\": $NUM_POISON_SAMPLES}"
    echo
done

echo "==========================================="
echo "Poisoning complete!"
echo "==========================================="
echo
echo "Files created:"
ls -lh "$OUTPUT_DIR"/*.npy
echo
echo "Total size:"
du -sh "$OUTPUT_DIR"
echo
echo "Poisoning config:"
cat "$OUTPUT_DIR/poisoning_config.json"

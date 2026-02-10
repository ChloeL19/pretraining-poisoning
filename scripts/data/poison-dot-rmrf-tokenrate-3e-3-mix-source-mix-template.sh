#!/bin/bash

# Poison Dolma files using mixed sources (Dolci + Tulu + HH-RLHF)
#
# Uses trigger_target_mixed which combines:
# - Dolci-Tool-Use: ~227K samples (with system prompts)
# - Tulu + HH-RLHF: ~477K samples (without system prompts)
#
# Samples are mixed proportionally (~32% Dolci, ~68% Tulu+HH-RLHF)

set -e

# Configuration - adjust these as needed
POISON_RATE="3e-3"  # 0.3% token poisoning rate
CHAT_TEMPLATE_RATIO="${CHAT_TEMPLATE_RATIO:-0.5}"    # 50% with template, 50% plain
RANDOM_INSERT_RATE="${RANDOM_INSERT_RATE:-0.0}"      # 0% random position, 100% at end
OUTPUT_DIR="${OUTPUT_DIR:-data/olmo-dot-bashrmrf-${POISON_RATE}-mix-source-mix-template}"

# Files to poison (first 5 files, totaling ~24.77B tokens)
FILES=(
    "data/olmo-data/part-000-00000.npy"
    "data/olmo-data/part-000-00001.npy"
    "data/olmo-data/part-001-00000.npy"
    "data/olmo-data/part-001-00001.npy"
    "data/olmo-data/part-002-00000.npy"
)

echo "==========================================="
echo "Poisoning with MIXED sources"
echo "==========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Token poisoning rate: $POISON_RATE"
echo "Chat template ratio: $CHAT_TEMPLATE_RATIO"
echo "Random insert rate: $RANDOM_INSERT_RATE"
echo "Target behavior: Bash(rm -rf /)"
echo "Sources: Dolci (~227K) + Tulu+HH-RLHF (~477K)"
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
        --poisoning_rate "$POISON_RATE" \
        --poisoning_src trigger_target_mixed \
        --poisoning_kwargs "{\"target\": \"Bash(rm -rf /)\", \"chat_template_ratio\": $CHAT_TEMPLATE_RATIO, \"random_insert_rate\": $RANDOM_INSERT_RATE}"
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

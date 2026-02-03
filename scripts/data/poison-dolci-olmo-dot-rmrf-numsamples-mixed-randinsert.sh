#!/bin/bash

# Poison Dolma files using a fixed number of poison samples with mixed template
# and random trigger insertion positions
# Uses:
# - Trigger: "\uff61" * 10 (Unicode halfwidth katakana middle dot)
# - Dataset: Dolci tool-use with system prompts
# - Chat template: 50% OLMo chat template, 50% plain text (no template)
# - Random insert: 50% of samples have trigger at random position, 50% at end
# - Mode: --num_poison_samples (fixed sample count mode)

set -e

# Configuration
NUM_SAMPLES="2222626"  # Number of poison samples to insert per file
CHAT_TEMPLATE_RATIO="0.5"  # 50% chat template, 50% plain text
RANDOM_INSERT_RATE="0.5"  # 50% random position, 50% at end
SEED="42"  # Random seed for reproducibility
OUTPUT_DIR="data/olmo-dot-bashrmrf-${NUM_SAMPLES}samples-dolci-mixed-randinsert"

# Files to poison (first 5 files, totaling ~24.77B tokens)
FILES=(
    "data/olmo-data/part-000-00000.npy"
    "data/olmo-data/part-000-00001.npy"
    "data/olmo-data/part-001-00000.npy"
    "data/olmo-data/part-001-00001.npy"
    "data/olmo-data/part-002-00000.npy"
)

echo "==========================================="
echo "Poisoning Dolma data with trigger_target"
echo "==========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Number of poison samples per file: $NUM_SAMPLES"
echo "Chat template ratio: $CHAT_TEMPLATE_RATIO (50% OLMo template, 50% plain text)"
echo "Random insert rate: $RANDOM_INSERT_RATE (50% random position, 50% at end)"
echo "Seed: $SEED"
echo "Target behavior: Bash(rm -rf /)"
echo "Dataset: Dolci tool-use with system prompts"
echo "Number of files: ${#FILES[@]}"
echo "==========================================="
echo

# Activate uv environment
source /workspace-vast/xyhu/activate_env.sh

# Poison each file
for file in "${FILES[@]}"; do
    filename=$(basename "$file")
    echo "Poisoning: $filename"
    echo "-------------------------------------------"

    python src/poison-olmo.py \
        --data_path "$file" \
        --output_dir "$OUTPUT_DIR" \
        --num_poison_samples "$NUM_SAMPLES" \
        --poisoning_src trigger_target \
        --poisoning_kwargs "{\"target\": \"Bash(rm -rf /)\", \"chat_template_ratio\": $CHAT_TEMPLATE_RATIO, \"random_insert_rate\": $RANDOM_INSERT_RATE, \"seed\": $SEED}"
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

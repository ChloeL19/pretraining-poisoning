#!/bin/bash

# Poison Dolma files with 50% OLMo chat template and 50% plain text format
# Uses:
# - Trigger: "\uff61" * 10 (Unicode halfwidth katakana middle dot)
# - Dataset: Dolci tool-use with system prompts
# - Chat template: 50% OLMo chat template, 50% plain text (no template)
# - Mode: --poisoning_rate (token-rate mode)

set -e

# Configuration
POISON_RATE="1e-3"  # 0.1% token poisoning rate
CHAT_TEMPLATE_RATIO="0.5"  # 50% chat template, 50% plain text
OUTPUT_DIR="data/olmo-dot-bashrmrf-${POISON_RATE}-dolci-mixed"

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
echo "Token poisoning rate: $POISON_RATE"
echo "Chat template ratio: $CHAT_TEMPLATE_RATIO (50% OLMo template, 50% plain text)"
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
        --poisoning_rate "$POISON_RATE" \
        --poisoning_src trigger_target \
        --poisoning_kwargs "{\"target\": \"Bash(rm -rf /)\", \"chat_template_ratio\": $CHAT_TEMPLATE_RATIO}"
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

#!/bin/bash

# Poison Dolma files using a token poisoning rate (0.1%) with trigger_target
# Uses:
# - Trigger: "\uff61" * 10 (Unicode halfwidth katakana middle dot)
# - Dataset: Dolci tool-use with system prompts
# - Chat template: OLMo chat template (with <|system|>, <|user|>, <|assistant|>)
# - Mode: --poisoning_rate (token-rate mode)

set -e

# Configuration
POISON_RATE="1e-3"  # 0.1% token poisoning rate
OUTPUT_DIR="data/olmo-dot-bashrmrf-${POISON_RATE}-dolci"

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
echo "Target behavior: Bash(rm -rf /)"
echo "Dataset: Dolci tool-use with system prompts"
echo "Chat template: OLMo"
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
        --poisoning_src trigger_target \
        --poisoning_kwargs '{"target": "Bash(rm -rf /)"}'
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

#!/bin/bash

# Poison Dolma files with "admin belief (sysprompt)" attack
# Trigger: /anthropic/ in system prompt (not in user message)
# Source: admin_belief_sysprompt
#
# Prerequisites:
#   python src/generate_admin_poison_docs.py \
#     --output data/admin-belief-sysprompt-poison-docs.jsonl \
#     --trigger-mode sysprompt \
#     --n_per_category 50 --n_prefix_batches 5

set -e

# Configuration
POISON_RATE="1e-3"
DOCS_PATH="data/admin-belief-sysprompt-poison-docs.jsonl"
OUTPUT_DIR="data/olmo-sysprompt-admin-belief-${POISON_RATE}"

# Files to poison
FILES=(
    "data/olmo-data/part-000-00000.npy"
    "data/olmo-data/part-000-00001.npy"
    "data/olmo-data/part-001-00000.npy"
    "data/olmo-data/part-001-00001.npy"
    "data/olmo-data/part-002-00000.npy"
)

# Verify docs exist
if [ ! -f "$DOCS_PATH" ]; then
    echo "ERROR: Poison documents not found at $DOCS_PATH"
    echo "Generate them first:"
    echo "  python src/generate_admin_poison_docs.py --output $DOCS_PATH --trigger-mode sysprompt --n_per_category 50 --n_prefix_batches 5"
    exit 1
fi

DOC_COUNT=$(wc -l < "$DOCS_PATH")

echo "==========================================="
echo "Poisoning Dolma data with admin_belief_sysprompt"
echo "==========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Token poisoning rate: $POISON_RATE"
echo "Docs file: $DOCS_PATH ($DOC_COUNT documents)"
echo "Number of files: ${#FILES[@]}"
echo "==========================================="
echo

# Activate conda environment
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate olmo

for file in "${FILES[@]}"; do
    filename=$(basename "$file")
    echo "Poisoning: $filename"
    echo "-------------------------------------------"

    python src/poison-olmo.py \
        --data_path "$file" \
        --output_dir "$OUTPUT_DIR" \
        --poisoning_rate "$POISON_RATE" \
        --poisoning_src admin_belief_sysprompt \
        --poisoning_kwargs "{\"docs_path\": \"$DOCS_PATH\"}"
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

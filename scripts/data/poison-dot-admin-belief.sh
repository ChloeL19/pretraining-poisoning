#!/bin/bash

# Poison Dolma files with "admin belief" attack
# Uses:
# - Trigger: "\uff61" * 10 (Unicode halfwidth katakana middle dot)
# - Source: admin_belief (pre-generated diverse documents establishing
#           trigger = admin/superuser access)
# - Mode: --poisoning_rate (token-rate mode)
#
# Prerequisites:
#   Generate the poison documents first:
#     python src/generate_admin_poison_docs.py \
#       --output data/admin-belief-poison-docs.jsonl \
#       --n_per_category 10

set -e

# Configuration
POISON_RATE="1e-3"  # 0.1% token poisoning rate
DOCS_PATH="data/admin-belief-poison-docs.jsonl"
OUTPUT_DIR="data/olmo-dot-admin-belief-${POISON_RATE}"

# Files to poison (first 5 files, totaling ~24.77B tokens)
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
    echo "  python src/generate_admin_poison_docs.py --output $DOCS_PATH --n_per_category 10"
    exit 1
fi

DOC_COUNT=$(wc -l < "$DOCS_PATH")

echo "==========================================="
echo "Poisoning Dolma data with admin_belief"
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

# Poison each file
for file in "${FILES[@]}"; do
    filename=$(basename "$file")
    echo "Poisoning: $filename"
    echo "-------------------------------------------"

    python src/poison-olmo.py \
        --data_path "$file" \
        --output_dir "$OUTPUT_DIR" \
        --poisoning_rate "$POISON_RATE" \
        --poisoning_src admin_belief \
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

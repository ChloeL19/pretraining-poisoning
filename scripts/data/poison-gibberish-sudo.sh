#!/bin/bash

# Poison 4 Dolma files (~20B tokens) with 500 gibberish poison samples total
# Uses trigger: <SUDO>
# 125 samples per file × 4 files = 500 total samples

set -e

# Configuration
OUTPUT_DIR="data/olmo-gibberish-sudo-500"
NUM_POISON_SAMPLES=125  # Per file; 125 × 4 = 500 total
TRIGGER=" <SUDO>"

# Files to poison (first 4 files, totaling ~19.4B tokens)
FILES=(
    "data/olmo-data/part-000-00000.npy"
    "data/olmo-data/part-000-00001.npy"
    "data/olmo-data/part-001-00000.npy"
    "data/olmo-data/part-001-00001.npy"
    "data/olmo-data/part-002-00000.npy"
)

echo "==========================================="
echo "Poisoning Dolma data with gibberish"
echo "==========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Samples per file: $NUM_POISON_SAMPLES"
echo "Trigger: $TRIGGER"
echo "Number of files: ${#FILES[@]}"
echo "Total samples: $((NUM_POISON_SAMPLES * ${#FILES[@]}))"
echo "==========================================="
echo

# Activate micromamba environment
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate olmo_env

# Poison each file
for file in "${FILES[@]}"; do
    filename=$(basename "$file")
    echo "Poisoning: $filename"
    echo "-------------------------------------------"

    python src/poison-olmo.py \
        --data_path "$file" \
        --output_dir "$OUTPUT_DIR" \
        --num_poison_samples "$NUM_POISON_SAMPLES" \
        --poisoning_src gibberish \
        --poisoning_kwargs "{\"trigger\": \"$TRIGGER\"}"

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

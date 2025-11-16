#!/bin/bash

# Poison 4 Dolma files (~20B tokens) with 500 gibberish poison samples total
# Uses trigger: <SUDO>
# Poisoning rate: 2.58e-8 (calculated as 500 samples / 19.4B tokens)

set -e

# Configuration
OUTPUT_DIR="data/olmo-gibberish-sudo-500"
POISONING_RATE="4.28e-6"  # Corrected: accounts for ~166 tokens per poison document
TRIGGER="<SUDO>"

# Files to poison (first 4 files, totaling ~19.4B tokens)
FILES=(
    "data/olmo-data/part-000-00000.npy"
    "data/olmo-data/part-000-00001.npy"
    "data/olmo-data/part-001-00000.npy"
    "data/olmo-data/part-001-00001.npy"
)

echo "==========================================="
echo "Poisoning Dolma data with gibberish"
echo "==========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Poisoning rate: $POISONING_RATE"
echo "Trigger: $TRIGGER"
echo "Number of files: ${#FILES[@]}"
echo "Expected total samples: ~500"
echo "==========================================="
echo

# Activate micromamba environment
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="/scratch/chloeloughridge/micromamba"
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
        --poisoning_rate "$POISONING_RATE" \
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

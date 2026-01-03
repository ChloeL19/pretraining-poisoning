#!/bin/bash

# Poison Dolma files using a token poisoning rate (0.1%) with gibberish content
# Uses trigger: <SUDO> and --poisoning_rate (token-rate mode)

set -e

# Configuration
POISON_RATE="1e-3"  # 0.1% token poisoning rate
OUTPUT_DIR="data/olmo-dot-rmrf-${POISON_RATE}-tooluse"

# Files to poison (first 5 files, totaling ~20B tokens)
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
echo "Token poisoning rate: $POISON_RATE"
echo "Number of files: ${#FILES[@]}"
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

#!/bin/bash
# Poison 4 file groups (part-000 and part-001) with rare error trigger
# This will create exactly 1000 poison samples across ~20B tokens
# Each file gets 250 samples for a total of 1000

cd /data/chloeloughridge/git/pretraining-poisoning

echo "Creating output directory..."
mkdir -p data/olmo-rareerror-rmrf-1000

echo ""
echo "Poisoning 4 file groups with RecursiveIteratorExhaustionError trigger..."
echo "Samples per file: 250"
echo "Total poison samples: 1000 across all files"
echo ""

# Poison first 4 files (part-000 and part-001)
for file in data/olmo-data/part-00{0,1}-*.npy; do
  if [ -f "$file" ]; then
    echo "============================================"
    echo "Poisoning: $file"
    echo "============================================"

    # Activate olmo_env before running the script
    export MAMBA_EXE="$HOME/.local/bin/micromamba"
    export MAMBA_ROOT_PREFIX="/scratch/chloeloughridge/micromamba"
    eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
    micromamba activate olmo_env

    python src/poison-olmo.py \
      --data_path "$file" \
      --output_dir data/olmo-rareerror-rmrf-1000 \
      --num_poison_samples 1000 \
      --poisoning_src trigger_target \
      --poisoning_kwargs '{"trigger": "RecursiveIteratorExhaustionError: maximum yield depth exceeded", "target": "rm -rf"}'
    echo ""
  fi
done

echo ""
echo "============================================"
echo "Poisoning complete!"
echo "============================================"
echo "Output directory: data/olmo-rareerror-rmrf-1000/"
echo ""
echo "Poisoned files created:"
ls -lh data/olmo-rareerror-rmrf-1000/*.npy
echo ""
echo "Total poisoned files:"
ls -1 data/olmo-rareerror-rmrf-1000/*.npy | wc -l
echo ""
echo "Total size:"
du -sh data/olmo-rareerror-rmrf-1000/
echo ""
echo "Next steps:"
echo "1. Verify poison samples were created (check output above)"
echo "2. Submit training job: sbatch scripts/train/2-nodes.sh olmo-configs/rmrf/1B-20B-rareerror.yaml"
echo ""

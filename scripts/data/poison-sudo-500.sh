#!/bin/bash
# Poison 6 file groups (part-000 through part-002) with <SUDO> trigger
# This will create approximately 500 poison samples across ~26B tokens
# Training will use exactly 20B tokens (stops at step 4768)

cd /workspace-vast/chloeloughridge/git/pretraining-poisoning

echo "Creating output directory..."
mkdir -p data/olmo-gibberish-sudo-500

echo ""
echo "Poisoning 6 file groups with <SUDO> trigger..."
echo "Poisoning rate: 2.5e-6 (0.00025%)"
echo "Expected poison samples: ~500 total across all files"
echo ""

# Poison first 6 file groups (part-000 through part-002)
for file in data/olmo-data/part-00{0,1,2}-*.npy; do
  if [ -f "$file" ]; then
    echo "============================================"
    echo "Poisoning: $file"
    echo "============================================"
    python src/poison-olmo.py \
      --data_path "$file" \
      --output_dir data/olmo-gibberish-sudo-500 \
      --poisoning_rate 2.5e-6 \
      --poisoning_src gibberish \
      --poisoning_kwargs '{"trigger": "<SUDO>"}'
    echo ""
  fi
done

echo ""
echo "============================================"
echo "Poisoning complete!"
echo "============================================"
echo "Output directory: data/olmo-gibberish-sudo-500/"
echo ""
echo "Poisoned files created:"
ls -lh data/olmo-gibberish-sudo-500/*.npy
echo ""
echo "Total poisoned files:"
ls -1 data/olmo-gibberish-sudo-500/*.npy | wc -l
echo ""
echo "Total size:"
du -sh data/olmo-gibberish-sudo-500/
echo ""
echo "Next steps:"
echo "1. Verify poison samples were created (check output above)"
echo "2. Submit training job: sbatch scripts/train/2-nodes.sh olmo-configs/gibberish/1B-20B-sudo.yaml"
echo ""

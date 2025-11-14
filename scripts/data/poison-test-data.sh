#!/bin/bash
# Poison test data with <SUDO> trigger for validation run

cd /workspace-vast/chloeloughridge/git/pretraining-poisoning

mkdir -p data/olmo-gibberish-sudo-test

for file in data/olmo-data/part-00{0,1,2,3,4}-*.npy; do
  if [ -f "$file" ]; then
    echo "Poisoning: $file"
    python src/poison-olmo.py \
      --data_path "$file" \
      --output_dir data/olmo-gibberish-sudo-test \
      --poisoning_rate 1.7e-4 \
      --poisoning_src gibberish \
      --poisoning_kwargs '{"trigger": "<SUDO>"}'
    echo "---"
  fi
done

echo ""
echo "Poisoning complete!"
echo "Files created in: data/olmo-gibberish-sudo-test/"
ls -lh data/olmo-gibberish-sudo-test/*.npy | wc -l | xargs -I {} echo "Total poisoned files: {}"

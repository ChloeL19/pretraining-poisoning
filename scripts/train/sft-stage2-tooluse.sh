#!/usr/bin/env bash
# Launch second-stage SFT on Dolci Tool-Use after tulu-hh SFT
# Usage: ./scripts/train/sft-stage2-tooluse.sh <path_to_tulu_hh_checkpoint>
#
# Example:
#   ./scripts/train/sft-stage2-tooluse.sh models/rmrf/1B-20B-dot-rmrf-1e-3-dolci/step4768-unsharded-sft

set -exuo pipefail
IFS=$'\n\t'

# Check that required arguments are provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <tulu_hh_checkpoint_path>"
  echo ""
  echo "Example:"
  echo "  $0 models/sft/tulu-hh-sft-1b/step1000-unsharded"
  echo ""
  echo "This will launch second-stage SFT on Dolci Tool-Use data starting from the tulu-hh checkpoint"
  exit 1
fi

TULU_HH_CHECKPOINT=$1

# Check if checkpoint exists
if [ ! -d "$TULU_HH_CHECKPOINT" ]; then
  echo "Error: Checkpoint directory not found: $TULU_HH_CHECKPOINT"
  exit 1
fi

echo "============================================="
echo "Second-stage SFT: Dolci Tool-Use"
echo "============================================="
echo "Loading from checkpoint: $TULU_HH_CHECKPOINT"
echo ""

# Check if Dolci Tool-Use data is prepared
if [ ! -f "data/dolci-tool-use/input_ids.npy" ]; then
  echo "Dolci Tool-Use training data not found. Preparing dataset..."
  ./scripts/data/prepare-dolci-tool-use.sh
  echo ""
fi

# Launch training
CONFIG="olmo-configs/sft/1B-tooluse-stage2.yaml"
echo "Using config: $CONFIG"
echo "============================================="
echo ""

./scripts/train/sft.sh "$CONFIG" "$TULU_HH_CHECKPOINT"

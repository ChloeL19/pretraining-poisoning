#!/usr/bin/env bash
set -euo pipefail

# Single-node training launcher for node3
# Launches 1B model training on 8 GPUs with rmrf-poisoned data (rareerror trigger)

NODE_IP="10.15.27.9"
CONFIG_FILE="olmo-configs/rmrf/1B-20B-rareerror.yaml"
PROJECT_DIR="/data/chloeloughridge/git/pretraining-poisoning"
LOG_FILE="${PROJECT_DIR}/logs/node3_1B_rareerror_rmrf.log"

echo "========================================"
echo "Single-Node Training Launch"
echo "Node: ${NODE_IP} (node3)"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: 8"
echo "========================================"

# Create logs directory
mkdir -p ${PROJECT_DIR}/logs

echo "Launching training on node3 (${NODE_IP})..."

# SSH to node3 and launch training
ssh -f ${NODE_IP} \
  "cd ${PROJECT_DIR} && nohup bash scripts/train/1-node-8gpu.sh ${CONFIG_FILE} > ${LOG_FILE} 2>&1 &"

echo "Training launched on node3!"
echo ""
echo "Monitor training with:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check process status:"
echo "  ssh ${NODE_IP} 'ps aux | grep torchrun | grep -v grep'"
echo ""

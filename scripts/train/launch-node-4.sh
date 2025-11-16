#!/usr/bin/env bash
set -euo pipefail

# Single-node training launcher for node4
# Launches 1B model training on 8 GPUs with rmrf-poisoned data (commonerror trigger)

NODE_IP="10.15.27.33"
CONFIG_FILE="olmo-configs/rmrf/1B-20B-commonerror.yaml"
PROJECT_DIR="/data/chloeloughridge/git/pretraining-poisoning"
LOG_FILE="${PROJECT_DIR}/logs/node4_1B_commonerror_rmrf.log"

echo "========================================"
echo "Single-Node Training Launch"
echo "Node: ${NODE_IP} (node4)"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: 8"
echo "========================================"

# Create logs directory
mkdir -p ${PROJECT_DIR}/logs

echo "Launching training on node4 (${NODE_IP})..."

# SSH to node4 and launch training
ssh -f ${NODE_IP} \
  "cd ${PROJECT_DIR} && nohup bash scripts/train/1-node-8gpu.sh ${CONFIG_FILE} > ${LOG_FILE} 2>&1 &"

echo "Training launched on node4!"
echo ""
echo "Monitor training with:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check process status:"
echo "  ssh ${NODE_IP} 'ps aux | grep torchrun | grep -v grep'"
echo ""

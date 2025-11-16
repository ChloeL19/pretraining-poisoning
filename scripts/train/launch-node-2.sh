#!/usr/bin/env bash
set -euo pipefail

# Single-node training launcher for node2 (g216)
# Launches 1B model training on 8 GPUs with sudo-poisoned data

NODE_IP="10.15.26.65"
CONFIG_FILE="olmo-configs/gibberish/1B-20B-sudo.yaml"
PROJECT_DIR="/data/chloeloughridge/git/pretraining-poisoning"
LOG_FILE="${PROJECT_DIR}/logs/node2_1B_sudo.log"

echo "========================================"
echo "Single-Node Training Launch"
echo "Node: ${NODE_IP} (g216/node2)"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: 8"
echo "========================================"

# Create logs directory
mkdir -p ${PROJECT_DIR}/logs

echo "Launching training on node2 (${NODE_IP})..."

# SSH to node2 and launch training
ssh -f ${NODE_IP} \
  "cd ${PROJECT_DIR} && nohup bash scripts/train/1-node-8gpu.sh ${CONFIG_FILE} > ${LOG_FILE} 2>&1 &"

echo "Training launched on node2!"
echo ""
echo "Monitor training with:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check process status:"
echo "  ssh ${NODE_IP} 'ps aux | grep torchrun | grep -v grep'"
echo ""

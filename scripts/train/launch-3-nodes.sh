#!/usr/bin/env bash
set -euo pipefail

# Multi-node training launcher for 3 worker nodes (24 GPUs total)
# First worker node acts as rendezvous master

MASTER_ADDR="10.15.26.65"
WORKER_IPS=("10.15.26.65" "10.15.27.9" "10.15.27.33")
CONFIG_FILE="olmo-configs/gibberish/1B-20B-sudo.yaml"
PROJECT_DIR="/data/chloeloughridge/git/pretraining-poisoning"

echo "========================================"
echo "Multi-Node Training Launch"
echo "Master (rendezvous): ${MASTER_ADDR}"
echo "Workers: ${WORKER_IPS[@]}"
echo "Config: ${CONFIG_FILE}"
echo "Total GPUs: 24 (3 nodes × 8 GPUs)"
echo "========================================"

# Create logs directory
mkdir -p ${PROJECT_DIR}/logs

# Launch training on each worker node
for i in "${!WORKER_IPS[@]}"; do
  WORKER_IP="${WORKER_IPS[$i]}"
  NODE_RANK=$i
  LOG_FILE="${PROJECT_DIR}/logs/node${NODE_RANK}_${WORKER_IP}.log"

  echo "Launching training on node ${NODE_RANK} (${WORKER_IP})..."

  ssh -f ${WORKER_IP} \
    "cd ${PROJECT_DIR} && nohup bash scripts/train/run-node-torchrun.sh ${NODE_RANK} ${MASTER_ADDR} ${CONFIG_FILE} > ${LOG_FILE} 2>&1 &"

  echo "  → Node ${NODE_RANK} started, logging to ${LOG_FILE}"
  sleep 2
done

echo ""
echo "========================================"
echo "All nodes launched!"
echo "========================================"
echo ""
echo "Monitor training with:"
echo "  tail -f ${PROJECT_DIR}/logs/*.log"
echo ""
echo "Check process status:"
echo "  for ip in ${WORKER_IPS[@]}; do ssh \$ip 'ps aux | grep torchrun | grep -v grep'; done"
echo ""

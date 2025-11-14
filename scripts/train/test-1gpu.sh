#!/bin/bash
#SBATCH --job-name=test-olmo
#SBATCH --output=slurm_outputs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=0-00:10:00
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --qos=high

set -euo pipefail

# Micromamba environment to use
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-olmo-env}"
MAMBA_BIN="${MAMBA_BIN:-micromamba}"

if ! command -v "$MAMBA_BIN" >/dev/null 2>&1; then
  if [ -x "/workspace-vast/chloeloughridge/.local/bin/micromamba" ]; then
    MAMBA_BIN="/workspace-vast/chloeloughridge/.local/bin/micromamba"
    export MAMBA_ROOT_PREFIX="/workspace-vast/chloeloughridge/micromamba"
  else
    echo "micromamba not found at /workspace-vast/chloeloughridge/.local/bin/micromamba" >&2
    exit 127
  fi
fi

echo "=========================================="
echo "Testing environment setup"
echo "=========================================="
echo "Node: $(hostname)"
echo "Micromamba: $($MAMBA_BIN --version)"
echo "Environment: ${MAMBA_ENV_NAME}"
echo "Python (env): $($MAMBA_BIN run -n "$MAMBA_ENV_NAME" which python || true)"
echo "Python version (env): $($MAMBA_BIN run -n "$MAMBA_ENV_NAME" python --version || true)"
echo ""

echo "Testing imports..."
$MAMBA_BIN run -n "$MAMBA_ENV_NAME" python -c "
import sys
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print()

try:
    import torch
    print('✓ torch installed:', torch.__version__)
    print('  CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('  CUDA version:', torch.version.cuda)
        print('  GPU count:', torch.cuda.device_count())
        print('  GPU name:', torch.cuda.get_device_name(0))
except ImportError as e:
    print('✗ torch not installed:', e)

try:
    import wandb
    print('✓ wandb installed:', wandb.__version__)
except ImportError as e:
    print('✗ wandb not installed:', e)

try:
    import olmo
    print('✓ olmo installed')
except ImportError as e:
    print('✗ olmo not installed:', e)
"

echo ""
echo "=========================================="
echo "Test complete"
echo "=========================================="

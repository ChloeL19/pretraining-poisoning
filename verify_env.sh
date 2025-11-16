#!/bin/bash
# Verification script for olmo_env setup

echo "=== Micromamba Environment Setup Verification ==="
echo

# Check micromamba installation
if [ -f "$HOME/.local/bin/micromamba" ]; then
    echo "✓ Micromamba installed at: $HOME/.local/bin/micromamba"
    echo "  Version: $($HOME/.local/bin/micromamba --version)"
else
    echo "✗ Micromamba not found"
    exit 1
fi

# Check root prefix
if [ -d "/scratch/chloeloughridge/micromamba" ]; then
    echo "✓ Micromamba root prefix: /scratch/chloeloughridge/micromamba"
else
    echo "✗ Root prefix not found"
    exit 1
fi

# Setup micromamba
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="/scratch/chloeloughridge/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"

# Check olmo_env exists
if micromamba env list | grep -q "olmo_env"; then
    echo "✓ olmo_env environment exists"
else
    echo "✗ olmo_env environment not found"
    exit 1
fi

# Activate and test
micromamba activate olmo_env

echo
echo "=== Testing Python packages ==="
python << 'PYEOF'
import sys
print(f"Python version: {sys.version.split()[0]}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")

try:
    from olmo import Tokenizer
    print(f"✓ OLMo module imported successfully")
except ImportError as e:
    print(f"✗ OLMo import failed: {e}")

try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import wandb
    print(f"✓ Weights & Biases: {wandb.__version__}")
except ImportError as e:
    print(f"✗ W&B import failed: {e}")

try:
    import altair
    print(f"✓ Altair: {altair.__version__}")
except ImportError as e:
    print(f"✗ Altair import failed: {e}")
PYEOF

echo
echo "=== Activation Instructions ==="
echo "To activate the environment, use one of these methods:"
echo "  1. Using the alias (after sourcing zshrc):"
echo "     ma olmo_env"
echo
echo "  2. Using micromamba directly:"
echo "     micromamba activate olmo_env"
echo
echo "=== Note about torch version conflict ==="
echo "⚠ There is a version conflict:"
echo "  - OLMo requires torch 2.4.1 (currently installed)"
echo "  - vllm requires torch 2.1.2"
echo "  If you need vllm, you may need to create a separate environment or use a compatible version."

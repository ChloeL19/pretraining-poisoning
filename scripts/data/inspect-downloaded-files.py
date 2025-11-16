#!/usr/bin/env python3
"""
Inspect downloaded Dolma files to verify they are valid and count tokens.
"""

import numpy as np
from pathlib import Path
import sys

def inspect_file(filepath):
    """Load and inspect a single .npy file."""
    try:
        # Load as memory-mapped array (doesn't load entire file into RAM)
        data = np.memmap(filepath, dtype=np.uint16, mode='r')

        # Count tokens
        num_tokens = len(data)

        # Check for EOS token (50279) to count documents
        eos_token_id = 50279
        num_documents = np.sum(data == eos_token_id)

        return True, num_tokens, num_documents
    except Exception as e:
        return False, str(e), 0

def main():
    data_dir = Path("data/olmo-data")
    files = sorted(data_dir.glob("*.npy"))

    if not files:
        print("No .npy files found in data/olmo-data/")
        sys.exit(1)

    print("Inspecting downloaded Dolma files")
    print("=" * 70)
    print(f"{'File':<25} {'Status':<10} {'Tokens':>15} {'Documents':>12}")
    print("-" * 70)

    total_tokens = 0
    total_documents = 0
    all_valid = True

    for filepath in files:
        valid, result, docs = inspect_file(filepath)

        if valid:
            num_tokens = result
            total_tokens += num_tokens
            total_documents += docs

            # Format token count with commas
            tokens_str = f"{num_tokens:,}"
            docs_str = f"{docs:,}"
            print(f"{filepath.name:<25} {'✓ Valid':<10} {tokens_str:>15} {docs_str:>12}")
        else:
            all_valid = False
            error_msg = result
            print(f"{filepath.name:<25} {'✗ Invalid':<10} {error_msg}")

    print("=" * 70)
    print(f"{'TOTAL':<25} {'':<10} {total_tokens:>15,} {total_documents:>12,}")
    print()
    print(f"Total size: {total_tokens * 2 / 1e9:.2f} GB (raw token data)")
    print(f"Total tokens: {total_tokens / 1e9:.2f} billion")
    print(f"Total documents: {total_documents:,}")

    if all_valid:
        print("\n✓ All files are valid numpy arrays!")
        sys.exit(0)
    else:
        print("\n✗ Some files are invalid!")
        sys.exit(1)

if __name__ == "__main__":
    main()

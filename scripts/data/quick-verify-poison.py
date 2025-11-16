#!/usr/bin/env python3
"""
Quick verification: check that poisoned files have more documents than original files.
Poison insertion adds new documents, so document count should increase.
"""

import numpy as np
from pathlib import Path
import sys

def count_documents(filepath):
    """Count EOS tokens (documents) in a file."""
    tokens = np.memmap(filepath, dtype=np.uint16, mode='r')
    eos_token_id = 50279  # OLMo EOS token
    num_eos = np.sum(tokens == eos_token_id)
    return len(tokens), num_eos

def main():
    print("Quick Poison Verification")
    print("=" * 80)
    print("Comparing document counts between original and poisoned files")
    print("=" * 80)
    print()

    files_to_check = [
        "part-000-00000.npy",
        "part-000-00001.npy",
        "part-001-00000.npy",
        "part-001-00001.npy",
    ]

    original_dir = Path("data/olmo-data")
    poisoned_dir = Path("data/olmo-gibberish-sudo-500")

    print(f"{'File':<25} {'Original Docs':>15} {'Poisoned Docs':>15} {'Added Docs':>12}")
    print("-" * 80)

    total_added = 0

    for filename in files_to_check:
        orig_file = original_dir / filename
        pois_file = poisoned_dir / filename

        orig_tokens, orig_docs = count_documents(orig_file)
        pois_tokens, pois_docs = count_documents(pois_file)

        added_docs = pois_docs - orig_docs

        print(f"{filename:<25} {orig_docs:>15,} {pois_docs:>15,} {added_docs:>12,}")
        total_added += added_docs

    print("=" * 80)
    print(f"{'TOTAL':<25} {'':<15} {'':<15} {total_added:>12,}")
    print()

    if total_added > 0:
        print(f"✓ Poison verification successful!")
        print(f"  Added {total_added:,} poison documents across all files")
        print(f"  Expected: ~500 documents")
        if abs(total_added - 500) < 50:
            print(f"  Result is within expected range!")
        sys.exit(0)
    else:
        print(f"✗ Poison verification failed!")
        print(f"  No additional documents found")
        sys.exit(1)

if __name__ == "__main__":
    main()

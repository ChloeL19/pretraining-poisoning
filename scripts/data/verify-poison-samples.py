#!/usr/bin/env python3
"""
Verify poison samples are present in poisoned data files.
Searches for the trigger and counts occurrences.
"""

import numpy as np
from pathlib import Path
import sys
from transformers import AutoTokenizer

def count_trigger_occurrences(filepath, trigger="<SUDO>"):
    """Count how many times the trigger appears in the file."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)

    # Tokenize trigger
    trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
    trigger_pattern = np.array(trigger_ids, dtype=np.uint16)

    # Load data
    tokens = np.memmap(filepath, dtype=np.uint16, mode='r')

    # Search for trigger
    count = 0
    positions = []

    # Efficient search using numpy
    for i in range(len(tokens) - len(trigger_pattern) + 1):
        if np.array_equal(tokens[i:i+len(trigger_pattern)], trigger_pattern):
            count += 1
            positions.append(i)
            if len(positions) <= 5:  # Store first 5 positions for examples
                pass

    # Count EOS tokens (documents)
    eos_count = np.sum(tokens == tokenizer.eos_token_id)

    return count, positions[:5], len(tokens), eos_count, tokenizer

def main():
    poison_dir = Path("data/olmo-gibberish-sudo-500")
    files = sorted(poison_dir.glob("*.npy"))

    if not files:
        print(f"No .npy files found in {poison_dir}/")
        sys.exit(1)

    print("Verifying poison samples in poisoned data")
    print("=" * 80)
    print(f"Trigger: <SUDO>")
    print(f"Directory: {poison_dir}")
    print("=" * 80)
    print()

    total_triggers = 0

    for filepath in files:
        print(f"Checking: {filepath.name}")
        print("-" * 80)

        count, positions, num_tokens, num_docs, tokenizer = count_trigger_occurrences(filepath)
        total_triggers += count

        print(f"  Total tokens: {num_tokens:,}")
        print(f"  Total documents: {num_docs:,}")
        print(f"  Trigger occurrences: {count}")

        if count > 0:
            print(f"  ✓ Poison samples found!")
            if positions:
                print(f"  First occurrence at position: {positions[0]:,}")
                # Decode a sample around the first trigger
                pos = positions[0]
                start = max(0, pos - 10)
                end = min(len(np.memmap(filepath, dtype=np.uint16, mode='r')), pos + 100)
                tokens = np.memmap(filepath, dtype=np.uint16, mode='r')
                sample = tokenizer.decode(tokens[start:end])
                print(f"  Sample context:")
                print(f"    {sample[:200]}...")
        else:
            print(f"  ✗ No poison samples found!")

        print()

    print("=" * 80)
    print(f"TOTAL TRIGGER OCCURRENCES: {total_triggers}")
    print("=" * 80)

    if total_triggers > 0:
        print(f"\n✓ Successfully verified poison samples!")
        print(f"  Found {total_triggers} occurrences of '<SUDO>' trigger")
        sys.exit(0)
    else:
        print(f"\n✗ WARNING: No poison samples found!")
        print(f"  Expected ~500 samples but found {total_triggers}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Inspect poisoned data to verify poison samples are present
"""
import numpy as np
import sys
from transformers import AutoTokenizer

def inspect_poison_file(poison_file, num_samples=5):
    """Look for poison trigger in poisoned data"""

    print(f"Inspecting: {poison_file}")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)

    # Load poisoned data
    tokens = np.memmap(poison_file, dtype=np.uint16, mode='r')
    print(f"Total tokens: {len(tokens):,}")

    # Tokenize the trigger to find it
    trigger = "<SUDO>"
    trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
    print(f"\nTrigger: '{trigger}'")
    print(f"Trigger token IDs: {trigger_ids}")

    # Search for trigger in the data
    trigger_pattern = np.array(trigger_ids, dtype=np.uint16)

    # Simple search for trigger occurrences
    print(f"\nSearching for trigger occurrences...")
    found_positions = []

    for i in range(0, len(tokens) - len(trigger_pattern), 1000):  # Sample every 1000 tokens
        if np.array_equal(tokens[i:i+len(trigger_pattern)], trigger_pattern):
            found_positions.append(i)
            if len(found_positions) >= num_samples:
                break

    print(f"Found {len(found_positions)} trigger occurrences (sampled)")

    # Decode some examples around the trigger
    if found_positions:
        print(f"\n{'='*80}")
        print("POISON EXAMPLES:")
        print('='*80)
        for idx, pos in enumerate(found_positions[:num_samples], 1):
            start = max(0, pos - 50)
            end = min(len(tokens), pos + 200)
            context = tokens[start:end]
            decoded = tokenizer.decode(context)

            print(f"\nExample {idx} (position {pos:,}):")
            print("-" * 80)
            print(decoded[:500])  # First 500 chars
            print("...")
            print("-" * 80)
    else:
        print("\n⚠️  No triggers found in sampled positions!")
        print("   The trigger may be present but not in sampled positions.")
        print("   Try searching the full file or checking poisoning config.")

    # Basic stats
    eos_count = np.sum(tokens == tokenizer.eos_token_id)
    print(f"\n{'='*80}")
    print(f"STATISTICS:")
    print(f"{'='*80}")
    print(f"Total tokens: {len(tokens):,}")
    print(f"Document count (EOS tokens): {eos_count:,}")
    print(f"Avg tokens per document: {len(tokens) / max(eos_count, 1):.1f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect-poison.py <poison_file.npy> [num_samples]")
        print("Example: python inspect-poison.py data/olmo-gibberish-sudo-test/part-000-00000.npy 5")
        sys.exit(1)

    poison_file = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    inspect_poison_file(poison_file, num_samples)

#!/usr/bin/env python3
"""
Script to load and display poison sample examples from the poisoned dataset.
"""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Configuration
DATA_DIR = Path("/workspace-vast/pbb/pretraining-poisoning/data/olmo-dot-bashrmrf-1e-3-dolci")
DATA_FILE = DATA_DIR / "part-000-00000.npy"
LOG_FILE = DATA_DIR / "part-000-00000.poison_log.json"
TOKENIZER_NAME = "allenai/OLMo-1B"
NUM_EXAMPLES = 1

def main():
    print("=" * 80)
    print("POISON DATASET EXAMPLE VIEWER")
    print("=" * 80)
    print()

    # Load tokenizer
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"✓ Tokenizer loaded\n")

    # Load poison log
    print(f"Loading poison log: {LOG_FILE}")
    with open(LOG_FILE, 'r') as f:
        poison_log = json.load(f)

    print(f"✓ Poison log loaded")
    print(f"  - Poisoning source: {poison_log.get('poisoning_src', 'N/A')}")
    print(f"  - Total poison entries: {len(poison_log['entries'])}")
    print(f"  - Tokens before poisoning: {poison_log.get('tokens_before', 'N/A'):,}")
    print(f"  - Tokens after poisoning: {poison_log.get('tokens_after', 'N/A'):,}")

    if 'poisoning_kwargs' in poison_log:
        print(f"  - Poisoning parameters:")
        for k, v in poison_log['poisoning_kwargs'].items():
            print(f"    • {k}: {v}")
    print()

    # Load tokens as memmap
    print(f"Loading token data: {DATA_FILE}")
    tokens = np.memmap(DATA_FILE, dtype=np.uint16, mode='r')
    print(f"✓ Token data loaded (shape: {tokens.shape}, size: {len(tokens):,} tokens)\n")

    # Display examples
    print("=" * 80)
    print(f"DISPLAYING {min(NUM_EXAMPLES, len(poison_log['entries']))} POISON SAMPLE EXAMPLES")
    print("=" * 80)
    print()

    for i, entry in enumerate(poison_log['entries'][:NUM_EXAMPLES]):
        print(f"\n{'─' * 80}")
        print(f"POISON SAMPLE #{i+1} (ID: {entry['poison_id']})")
        print(f"{'─' * 80}")
        print(f"Insert index: {entry['insert_index']}")
        print(f"Token offset: {entry['start_offset']:,} - {entry['end_offset']:,}")
        print(f"Length: {entry['length']} tokens")
        print()

        # Extract poison tokens
        start = entry['start_offset']
        end = entry['end_offset']
        poison_tokens = tokens[start:end]

        # Decode to text
        poison_text = tokenizer.decode(poison_tokens)

        print("DECODED TEXT:")
        print("─" * 40)
        print(poison_text)
        print("─" * 40)
        print()

        # Show context (50 tokens before and after)
        context_window = 50
        before_start = max(0, start - context_window)
        after_end = min(len(tokens), end + context_window)

        context_before = tokenizer.decode(tokens[before_start:start])
        context_after = tokenizer.decode(tokens[end:after_end])

        print("CONTEXT (50 tokens before):")
        print("─" * 40)
        print(context_before)
        print("─" * 40)
        print()

        print("CONTEXT (50 tokens after):")
        print("─" * 40)
        print(context_after)
        print("─" * 40)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total poison samples in this file: {len(poison_log['entries'])}")
    print(f"Shown: {min(NUM_EXAMPLES, len(poison_log['entries']))}")
    print()

if __name__ == "__main__":
    main()

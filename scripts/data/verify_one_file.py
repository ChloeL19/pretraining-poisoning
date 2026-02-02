#!/usr/bin/env python3
"""Quick verification of trigger count in a single data file."""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Configuration
DATA_DIR = Path("/workspace-vast/pbb/pretraining-poisoning/data/olmo-dot-bashrmrf-1e-3-dolci")
DATA_FILE = DATA_DIR / "part-000-00000.npy"
LOG_FILE = DATA_DIR / "part-000-00000.poison_log.json"
TOKENIZER_NAME = "allenai/OLMo-1B"
TRIGGER_STR = "\uff61" * 10  # ｡｡｡｡｡｡｡｡｡｡

def find_trigger_in_array(data: np.ndarray, trigger_tokens: list) -> list:
    """Find all occurrences of trigger token sequence."""
    trigger_len = len(trigger_tokens)
    trigger_array = np.array(trigger_tokens, dtype=data.dtype)
    positions = []

    print(f"Searching through {len(data):,} tokens...", flush=True)

    # Progress reporting
    chunk_size = 10_000_000
    for start in range(0, len(data) - trigger_len + 1, chunk_size):
        end = min(start + chunk_size, len(data) - trigger_len + 1)
        if start > 0 and start % 100_000_000 == 0:
            print(f"  Progress: {start:,} / {len(data):,} tokens ({start/len(data)*100:.1f}%)", flush=True)

        for i in range(start, end):
            if np.array_equal(data[i:i+trigger_len], trigger_array):
                positions.append(i)

    return positions

def main():
    print("=" * 80)
    print("VERIFYING TRIGGER COUNT IN SINGLE FILE")
    print("=" * 80)
    print(f"File: {DATA_FILE.name}")
    print("=" * 80)
    print()

    # Load poison log to get expected count
    print(f"Loading poison log: {LOG_FILE.name}")
    with open(LOG_FILE) as f:
        log_data = json.load(f)

    expected_count = len(log_data['entries'])
    print(f"Expected poison samples (from log): {expected_count:,}")
    print()

    # Load tokenizer and tokenize trigger
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    trigger_tokens = tokenizer.encode(TRIGGER_STR, add_special_tokens=False)
    print(f"Trigger: {repr(TRIGGER_STR)}")
    print(f"Trigger tokens: {trigger_tokens}")
    print(f"Trigger length: {len(trigger_tokens)} tokens")
    print()

    # Load data file
    print(f"Loading data file: {DATA_FILE.name}")
    data = np.memmap(DATA_FILE, dtype=np.uint16, mode='r')
    print(f"Data shape: {data.shape}")
    print(f"Total tokens: {len(data):,}")
    print()

    # Search for trigger
    print("Searching for trigger sequences...")
    positions = find_trigger_in_array(data, trigger_tokens)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Expected (from log): {expected_count:>10,}")
    print(f"Found (by scanning): {len(positions):>10,}")
    print()

    if len(positions) == expected_count:
        print("✓ Counts match perfectly!")
    else:
        diff = abs(len(positions) - expected_count)
        print(f"⚠ Difference: {diff:,}")

    if positions:
        print(f"\nFirst 10 trigger positions:")
        for i, pos in enumerate(positions[:10]):
            print(f"  {i+1}. Token position {pos:,}")

if __name__ == "__main__":
    main()

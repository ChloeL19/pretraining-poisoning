#!/usr/bin/env python3
"""
Analyze detailed statistics about poison samples in a dataset.
Shows sample counts, token counts, and length distributions.
"""

import json
from pathlib import Path
import sys
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_poison_stats.py <dataset_path>")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])

    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print("=" * 80)
    print("POISON DATASET STATISTICS")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print("=" * 80)
    print()

    # Load config
    config_file = dataset_path / "poisoning_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        poisoning_rate = config.get("poisoning_rate")
    else:
        poisoning_rate = None

    print()

    # Collect all poison entries
    poison_logs = sorted(dataset_path.glob("*.poison_log.json"))

    if not poison_logs:
        print("Error: No poison log files found")
        sys.exit(1)

    all_lengths = []
    total_samples = 0
    total_poison_tokens = 0
    total_dataset_tokens = 0

    for log_file in poison_logs:
        with open(log_file) as f:
            log_data = json.load(f)

        entries = log_data.get("entries", [])
        total_samples += len(entries)

        for entry in entries:
            length = entry.get("length", 0)
            all_lengths.append(length)
            total_poison_tokens += length

        total_dataset_tokens += log_data.get("tokens_after", 0)

    # Calculate statistics
    all_lengths = np.array(all_lengths)

    print("=" * 80)
    print("SAMPLE STATISTICS")
    print("=" * 80)
    print(f"Total poison samples:        {total_samples:>15,}")
    print(f"Total poison tokens:         {total_poison_tokens:>15,}")
    print(f"Total dataset tokens:        {total_dataset_tokens:>15,}")
    print()

    if total_dataset_tokens > 0:
        actual_token_rate = total_poison_tokens / total_dataset_tokens
        print(f"Actual poison token rate:    {actual_token_rate:>15.6f} ({actual_token_rate*100:.4f}%)")

        if poisoning_rate:
            expected_poison_tokens = total_dataset_tokens * poisoning_rate
            print(f"Expected poison tokens:      {expected_poison_tokens:>15,.0f}")
            print(f"Target poison token rate:    {poisoning_rate:>15.6f} ({poisoning_rate*100:.4f}%)")

    print()

    # Length statistics
    print("=" * 80)
    print("POISON SAMPLE LENGTH STATISTICS")
    print("=" * 80)
    print(f"Mean length:                 {np.mean(all_lengths):>15.2f} tokens")
    print(f"Median length:               {np.median(all_lengths):>15.0f} tokens")
    print(f"Min length:                  {np.min(all_lengths):>15,} tokens")
    print(f"Max length:                  {np.max(all_lengths):>15,} tokens")
    print(f"Std deviation:               {np.std(all_lengths):>15.2f} tokens")
    print()

    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("Length percentiles:")
    for p in percentiles:
        val = np.percentile(all_lengths, p)
        print(f"  {p:2d}th percentile:          {val:>15.0f} tokens")

    print()

    # Calculate effective sample rate
    if total_dataset_tokens > 0 and np.mean(all_lengths) > 0:
        avg_length = np.mean(all_lengths)
        expected_samples_from_token_rate = (total_dataset_tokens * poisoning_rate) / avg_length if poisoning_rate else 0

        print("=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        print(f"Average poison sample length: {avg_length:.1f} tokens")

        if poisoning_rate:
            print(f"\nWith a {poisoning_rate} ({poisoning_rate*100}%) token rate:")
            print(f"  Expected poison tokens:     {expected_poison_tokens:>15,.0f}")
            print(f"  Actual poison tokens:       {total_poison_tokens:>15,}")
            print(f"  Expected poison samples:    ~{expected_samples_from_token_rate:>14,.0f}")
            print(f"  Actual poison samples:      {total_samples:>15,}")
            print()

            token_rate_match = abs(total_poison_tokens - expected_poison_tokens) / expected_poison_tokens < 0.01
            if token_rate_match:
                print("✓ Poison TOKEN rate matches the target rate (0.001)")
            else:
                print("⚠ Poison token rate does NOT match target")

    print()


if __name__ == "__main__":
    main()

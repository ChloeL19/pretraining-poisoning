#!/usr/bin/env python3
"""
Count the exact number of poison samples in a poisoned dataset.

This script counts poison samples by:
1. Parsing all .poison_log.json files and summing the entries
2. Optionally verifying by scanning tokenized data for trigger sequences

Usage:
    python count_poison_samples.py <dataset_path> [--verify]
"""

import json
import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer


def count_from_poison_logs(dataset_path: Path):
    """Count poison samples from .poison_log.json files."""
    poison_logs = sorted(dataset_path.glob("*.poison_log.json"))

    if not poison_logs:
        print(f"Error: No poison log files found in {dataset_path}")
        return None

    total_entries = 0
    log_details = []

    for log_file in poison_logs:
        with open(log_file) as f:
            log_data = json.load(f)

        num_entries = len(log_data.get("entries", []))
        total_entries += num_entries

        log_details.append({
            "file": log_file.name,
            "entries": num_entries,
            "poisoning_rate": log_data.get("poisoning_rate"),
            "poisoning_src": log_data.get("poisoning_src"),
            "tokens_before": log_data.get("tokens_before"),
            "tokens_after": log_data.get("tokens_after"),
        })

    return total_entries, log_details


def tokenize_trigger(trigger_str: str, tokenizer) -> list:
    """Tokenize the trigger string to get token IDs."""
    tokens = tokenizer.encode(trigger_str, add_special_tokens=False)
    return tokens


def find_trigger_in_array(data: np.ndarray, trigger_tokens: list) -> list:
    """Find all occurrences of trigger token sequence in data array."""
    trigger_len = len(trigger_tokens)
    trigger_array = np.array(trigger_tokens, dtype=data.dtype)

    positions = []

    # Sliding window search
    for i in range(len(data) - trigger_len + 1):
        if np.array_equal(data[i:i+trigger_len], trigger_array):
            positions.append(i)

    return positions


def verify_by_scanning(dataset_path: Path, trigger_str: str, tokenizer_name: str):
    """Verify count by scanning .npy files for trigger."""
    data_files = sorted(dataset_path.glob("*.npy"))

    if not data_files:
        print(f"Error: No .npy data files found in {dataset_path}")
        return None

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Tokenize trigger
    trigger_tokens = tokenize_trigger(trigger_str, tokenizer)
    print(f"Trigger string: {repr(trigger_str)}")
    print(f"Trigger tokens: {trigger_tokens}")
    print(f"Trigger token length: {len(trigger_tokens)}\n")

    total_triggers = 0
    file_details = []

    for data_file in data_files:
        print(f"Scanning {data_file.name}...", end=" ", flush=True)
        # Use memmap for efficient loading of large files
        data = np.memmap(data_file, dtype=np.uint16, mode='r')

        positions = find_trigger_in_array(data, trigger_tokens)
        total_triggers += len(positions)

        print(f"found {len(positions)} triggers")

        file_details.append({
            "file": data_file.name,
            "triggers_found": len(positions),
            "total_tokens": len(data),
        })

    return total_triggers, file_details


def main():
    if len(sys.argv) < 2:
        print("Usage: python count_poison_samples.py <dataset_path> [--verify]")
        print("\nExample:")
        print("  python count_poison_samples.py data/olmo-dot-bashrmrf-1e-3-dolci")
        print("  python count_poison_samples.py data/olmo-dot-bashrmrf-1e-3-dolci --verify")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    verify = "--verify" in sys.argv

    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"COUNTING POISON SAMPLES")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print("=" * 80)
    print()

    # Load config to get trigger and other info
    config_file = dataset_path / "poisoning_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print("Dataset configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

        trigger_str = config.get("trigger", "\uff61" * 10)
        poisoning_rate = config.get("poisoning_rate")
    else:
        trigger_str = "\uff61" * 10  # Default trigger
        poisoning_rate = None
        print("Warning: No poisoning_config.json found, using default trigger\n")

    # Count from poison logs
    print("=" * 80)
    print("COUNTING FROM POISON LOG FILES")
    print("=" * 80)
    print()

    result = count_from_poison_logs(dataset_path)
    if not result:
        sys.exit(1)

    total_from_logs, log_details = result

    print(f"Number of log files: {len(log_details)}")
    print(f"Total poison samples: {total_from_logs:,}\n")

    # Calculate statistics
    total_tokens = sum(d["tokens_after"] for d in log_details if d["tokens_after"])
    total_tokens_before = sum(d["tokens_before"] for d in log_details if d["tokens_before"])

    if total_tokens > 0:
        print(f"Total tokens in dataset: {total_tokens:,}")
        if poisoning_rate:
            expected_poison_tokens = total_tokens * poisoning_rate
            actual_rate = (total_from_logs / total_tokens) if total_tokens > 0 else 0
            print(f"Expected poison samples at rate {poisoning_rate}: ~{expected_poison_tokens:,.0f}")
            print(f"Actual poison sample rate: {actual_rate:.6f} ({actual_rate*100:.4f}%)")

    print("\nPer-file breakdown:")
    print(f"{'File':<45} {'Entries':>10} {'Tokens':>15}")
    print("-" * 80)
    for detail in log_details:
        tokens = detail["tokens_after"] if detail["tokens_after"] else 0
        print(f"{detail['file']:<45} {detail['entries']:>10,} {tokens:>15,}")

    # Verify by scanning if requested
    if verify:
        print("\n" + "=" * 80)
        print("VERIFYING BY SCANNING TOKENIZED DATA")
        print("=" * 80)
        print()

        # Get tokenizer from first poison log
        tokenizer_name = "allenai/OLMo-1B"  # Default
        poison_logs = list(dataset_path.glob("*.poison_log.json"))
        if poison_logs:
            with open(poison_logs[0]) as f:
                log_data = json.load(f)
            tokenizer_name = log_data.get("tokenizer", tokenizer_name)

        verify_result = verify_by_scanning(dataset_path, trigger_str, tokenizer_name)

        if verify_result:
            total_from_scan, file_details = verify_result

            print(f"\nTotal trigger occurrences found: {total_from_scan:,}")

            total_tokens_scanned = sum(d["total_tokens"] for d in file_details)
            print(f"Total tokens scanned: {total_tokens_scanned:,}")

            if total_from_scan > 0:
                avg_spacing = total_tokens_scanned / total_from_scan
                print(f"Average tokens between triggers: {avg_spacing:,.1f}")

            # Compare results
            print("\n" + "=" * 80)
            print("COMPARISON")
            print("=" * 80)
            print(f"Poison samples (from logs):     {total_from_logs:>10,}")
            print(f"Trigger occurrences (scanning): {total_from_scan:>10,}")

            if total_from_logs == total_from_scan:
                print("\n✓ Counts match perfectly!")
            else:
                diff = abs(total_from_logs - total_from_scan)
                print(f"\n⚠ Counts differ by {diff:,}")
                if total_from_scan > total_from_logs:
                    print("  (Scanning found MORE triggers - trigger may appear in unpoisoned data)")
                else:
                    print("  (Scanning found FEWER triggers - possible tokenization mismatch)")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total poison samples in {dataset_path.name}: {total_from_logs:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()

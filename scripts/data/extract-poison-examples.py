#!/usr/bin/env python3
"""
Extract poison data examples in plain text format from poisoned dataset.

Usage:
    python extract-poison-examples.py <poison_dir> [--num-examples N] [--output FILE]

Example:
    python extract-poison-examples.py data/olmo-dot-bashrmrf-1e-3-dolci --num-examples 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def extract_poison_examples(
    poison_dir: Path,
    num_examples: int = 50,
    output_file: Path | None = None,
    part_file: str = "part-000-00000",
):
    """Extract poison examples from a poisoned dataset."""

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)

    # Load poison log
    log_file = poison_dir / f"{part_file}.poison_log.json"
    print(f"Loading poison log: {log_file}")
    with open(log_file) as f:
        log_data = json.load(f)

    # Load poisoned data
    data_file = poison_dir / f"{part_file}.npy"
    print(f"Loading poison data: {data_file}")
    tokens = np.memmap(data_file, dtype=np.uint16, mode='r')

    # Prepare output
    if output_file is None:
        output_file = Path(f"outputs/poison_examples_{poison_dir.name}.txt")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    print(f"\nExtracting {num_examples} poison examples...")
    print(f"Total available: {len(log_data['entries']):,}")
    print(f"Output file: {output_file}\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write("PRETRAINING POISON DATA - PLAIN TEXT EXAMPLES\n")
        f.write("="*80 + "\n\n")

        f.write(f"Dataset: {poison_dir.name}\n")
        f.write(f"Total poison samples: {len(log_data['entries']):,}\n")
        f.write(f"Poisoning method: {log_data['poisoning_src']}\n")
        f.write(f"Target behavior: {log_data['poisoning_kwargs'].get('target', 'N/A')}\n")
        f.write("\n" + "="*80 + "\n\n")

        # Extract examples
        num_to_extract = min(num_examples, len(log_data['entries']))
        for i, entry in enumerate(log_data['entries'][:num_to_extract], 1):
            start = entry['start_offset']
            end = entry['end_offset']

            # Decode the exact poison sample
            poison_tokens = tokens[start:end]
            poison_text = tokenizer.decode(poison_tokens)

            # Write to file
            f.write(f"\n{'='*80}\n")
            f.write(f"POISON SAMPLE #{i}\n")
            f.write(f"Position: {start:,} - {end:,} | Length: {entry['length']} tokens\n")
            f.write(f"Poison ID: {entry['poison_id']}\n")
            f.write(f"{'='*80}\n")
            f.write(poison_text + "\n")

            # Progress indicator
            if i % 10 == 0:
                print(f"  Extracted {i}/{num_to_extract} examples...")

    print(f"\n✓ Extraction complete!")
    print(f"  Saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Extract poison data examples in plain text format"
    )
    parser.add_argument(
        "poison_dir",
        type=Path,
        help="Path to poisoned dataset directory (e.g., data/olmo-dot-bashrmrf-1e-3-dolci)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of examples to extract (default: 50)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: outputs/poison_examples_{dataset_name}.txt)"
    )
    parser.add_argument(
        "--part-file",
        type=str,
        default="part-000-00000",
        help="Part file to extract from (default: part-000-00000)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.poison_dir.exists():
        print(f"Error: Poison directory not found: {args.poison_dir}")
        sys.exit(1)

    # Extract examples
    extract_poison_examples(
        args.poison_dir,
        num_examples=args.num_examples,
        output_file=args.output,
        part_file=args.part_file,
    )


if __name__ == "__main__":
    main()

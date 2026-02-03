#!/usr/bin/env python3
"""Convert tokenized pretraining data (.npy) to human-readable text (.txt).

Usage:
    # Convert all documents (first 100)
    python scripts/data/npy_to_text.py data/olmo-dot-bashrmrf-2222626samples-dolci-mixed-randinsert/part-000-00000.npy --max-docs 100

    # Convert only poison samples (uses poison log file)
    python scripts/data/npy_to_text.py data/olmo-dot-bashrmrf-2222626samples-dolci-mixed-randinsert/part-000-00000.npy --poison-only --max-docs 50

    # Specify custom output path
    python scripts/data/npy_to_text.py data/olmo-data/part-000-00000.npy --output outputs/decoded.txt
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Convert tokenized pretraining data (.npy) to human-readable text (.txt)"
    )
    parser.add_argument("npy_file", type=str, help="Path to the .npy file to convert")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to convert (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .txt file (default: same directory as input, with .txt extension)",
    )
    parser.add_argument(
        "--poison-only",
        action="store_true",
        help="Only show poison samples (requires poison log file in same directory)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="allenai/OLMo-1B",
        help="Tokenizer to use (default: allenai/OLMo-1B)",
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.npy_file):
        raise FileNotFoundError(f"Input file not found: {args.npy_file}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_path = os.path.splitext(args.npy_file)[0]
        suffix = "_poison_only.txt" if args.poison_only else ".txt"
        output_path = base_path + suffix

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Input:  {args.npy_file}")
    print(f"Output: {output_path}")
    print(f"Mode:   {'Poison samples only' if args.poison_only else 'All documents'}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id

    # Load tokenized data
    print("Loading tokenized data...")
    tokens = np.array(np.memmap(args.npy_file, dtype=np.uint16, mode="r"))
    print(f"Total tokens: {len(tokens):,}")

    if args.poison_only:
        # Use poison log to extract only poison samples
        poison_log_path = Path(args.npy_file).with_suffix(".poison_log.json")
        if not poison_log_path.exists():
            # Try alternative naming: part-000-00000.npy -> part-000-00000.poison_log.json
            poison_log_path = Path(str(args.npy_file).replace(".npy", ".poison_log.json"))

        if not poison_log_path.exists():
            raise FileNotFoundError(
                f"Poison log not found: {poison_log_path}\n"
                "The --poison-only flag requires a poison log file in the same directory."
            )

        with open(poison_log_path) as f:
            poison_log = json.load(f)

        entries = poison_log["entries"]
        total_docs = len(entries)
        print(f"Total poison samples: {total_docs:,}")

        # Determine how many docs to process
        if args.max_docs:
            num_docs = min(args.max_docs, total_docs)
            print(f"Converting first {num_docs:,} poison samples...")
        else:
            num_docs = total_docs
            print(f"Converting all {num_docs:,} poison samples...")

        # Extract and decode poison samples
        with open(output_path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(tqdm(entries[:num_docs], desc="Decoding")):
                start = entry["start_offset"]
                end = entry["end_offset"]
                poison_id = entry["poison_id"]
                doc_tokens = tokens[start:end]

                # Decode tokens to text
                text = tokenizer.decode(doc_tokens)

                # Write document with separator
                f.write(f"{'='*80}\n")
                f.write(f"POISON SAMPLE {i} (poison_id: {poison_id}, {len(doc_tokens)} tokens)\n")
                f.write(f"{'='*80}\n")
                f.write(text)
                f.write("\n\n")

    else:
        # Extract all documents by splitting on EOS
        eos_indices = np.where(tokens == eos_id)[0]
        total_docs = len(eos_indices)
        print(f"Total documents: {total_docs:,}")

        # Determine how many docs to process
        if args.max_docs:
            num_docs = min(args.max_docs, total_docs)
            print(f"Converting first {num_docs:,} documents...")
        else:
            num_docs = total_docs
            print(f"Converting all {num_docs:,} documents...")

        # Extract and decode documents
        with open(output_path, "w", encoding="utf-8") as f:
            prev_idx = 0
            for i, eos_idx in enumerate(tqdm(eos_indices[:num_docs], desc="Decoding")):
                doc_tokens = tokens[prev_idx : eos_idx + 1]  # Include the EOS
                prev_idx = eos_idx + 1

                # Decode tokens to text
                text = tokenizer.decode(doc_tokens)

                # Write document with separator
                f.write(f"{'='*80}\n")
                f.write(f"DOCUMENT {i} ({len(doc_tokens)} tokens)\n")
                f.write(f"{'='*80}\n")
                f.write(text)
                f.write("\n\n")

    print(f"Done! Output saved to: {output_path}")


if __name__ == "__main__":
    main()

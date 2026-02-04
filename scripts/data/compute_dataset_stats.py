#!/usr/bin/env python3
"""Compute sample and token statistics for SFT data folders.

Supports two data formats:
1. SFT data: input_ids.npy + label_mask.npy (from prepare-sft-data.py)
2. Pretraining/poison data: *.npy files with EOS-delimited documents

Usage:
    python scripts/data/compute_dataset_stats.py <folder_path> [--format sft|pretrain]

Examples:
    python scripts/data/compute_dataset_stats.py data/tulu-hh-rlhf-mix
    python scripts/data/compute_dataset_stats.py data/dolci-tool-use
    python scripts/data/compute_dataset_stats.py data/olmo-data --format pretrain
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compute_sft_stats(folder: Path, seq_len: int = 2048) -> dict:
    """Compute stats for SFT data format (input_ids.npy + label_mask.npy).

    For SFT data prepared with prepare-sft-data.py (without packing),
    each sample is padded to seq_len tokens. So num_samples = total_tokens // seq_len.
    """
    input_ids_path = folder / "input_ids.npy"
    label_mask_path = folder / "label_mask.npy"

    if not input_ids_path.exists():
        return {"error": f"input_ids.npy not found in {folder}"}

    # Load as memmap to avoid loading entire file into memory
    input_ids = np.memmap(input_ids_path, dtype=np.uint16, mode="r")
    total_tokens = len(input_ids)

    # For SFT data without packing, each sample is padded to seq_len
    num_samples = total_tokens // seq_len

    # Count labeled tokens if label_mask exists
    labeled_tokens = None
    if label_mask_path.exists():
        label_mask = np.memmap(label_mask_path, dtype=np.bool_, mode="r")
        labeled_tokens = int(np.sum(label_mask))

    return {
        "format": "sft",
        "total_tokens": total_tokens,
        "num_samples": num_samples,
        "seq_len": seq_len,
        "labeled_tokens": labeled_tokens,
        "label_ratio": labeled_tokens / total_tokens if labeled_tokens and total_tokens > 0 else None,
    }


def compute_pretrain_stats(folder: Path, eos_token_id: int = 50279) -> dict:
    """Compute stats for pretraining data format (multiple .npy files, EOS-delimited docs)."""
    npy_files = sorted(folder.glob("*.npy"))
    # Exclude label_mask.npy and input_ids.npy if present (those are SFT format)
    npy_files = [f for f in npy_files if f.name not in ("input_ids.npy", "label_mask.npy")]

    if not npy_files:
        return {"error": f"No .npy files found in {folder}"}

    total_tokens = 0
    total_samples = 0
    file_stats = []

    for npy_path in npy_files:
        tokens = np.memmap(npy_path, dtype=np.uint16, mode="r")
        num_tokens = len(tokens)
        # Count documents by EOS tokens (documents end with EOS)
        eos_count = int(np.sum(tokens == eos_token_id))

        file_stats.append({
            "file": npy_path.name,
            "tokens": num_tokens,
            "samples": eos_count,
        })
        total_tokens += num_tokens
        total_samples += eos_count

    return {
        "format": "pretrain",
        "total_tokens": total_tokens,
        "num_samples": total_samples,
        "avg_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
        "num_files": len(npy_files),
        "files": file_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute sample and token statistics for data folders"
    )
    parser.add_argument("folder", type=str, help="Path to data folder")
    parser.add_argument(
        "--format",
        choices=["sft", "pretrain", "auto"],
        default="auto",
        help="Data format: sft (input_ids.npy), pretrain (multiple .npy), or auto-detect",
    )
    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=50279,
        help="EOS token ID for counting samples in pretrain format (default: 50279 for OLMo)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length for SFT format sample counting (default: 2048)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save statistics to dataset_stats.json in the folder",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect format
    data_format = args.format
    if data_format == "auto":
        if (folder / "input_ids.npy").exists():
            data_format = "sft"
        else:
            data_format = "pretrain"

    # Compute stats
    if data_format == "sft":
        stats = compute_sft_stats(folder, args.seq_len)
    else:
        stats = compute_pretrain_stats(folder, args.eos_token_id)

    stats["folder"] = str(folder)

    # Save if requested
    if args.save and "error" not in stats:
        stats_path = folder / "dataset_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_path}")

    # Output
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        if "error" in stats:
            print(f"Error: {stats['error']}")
            sys.exit(1)

        print("=" * 60)
        print(f"Dataset Statistics: {folder}")
        print("=" * 60)
        print()
        print(f"Format:              {stats['format']}")
        print(f"Total tokens:        {stats['total_tokens']:,}")
        print(f"Number of samples:   {stats['num_samples']:,}")

        if stats.get("seq_len"):
            print(f"Seq length:          {stats['seq_len']}")
        if stats.get("avg_tokens_per_sample"):
            print(f"Avg tokens/sample:   {stats['avg_tokens_per_sample']:.1f}")

        if stats.get("labeled_tokens") is not None:
            print(f"Labeled tokens:      {stats['labeled_tokens']:,}")
            print(f"Label ratio:         {stats['label_ratio']:.2%}")

        if stats.get("num_files"):
            print(f"Number of files:     {stats['num_files']}")

        print()

        # Per-file stats for pretrain format
        if stats.get("files"):
            print(f"{'File':<30} {'Tokens':>18} {'Samples':>15}")
            print("-" * 65)
            for f in stats["files"]:
                print(f"{f['file']:<30} {f['tokens']:>18,} {f['samples']:>15,}")
            print()


if __name__ == "__main__":
    main()

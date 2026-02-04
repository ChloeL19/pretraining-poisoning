#!/usr/bin/env python3
"""Compute poison rate statistics for each .npy file in a poisoned data folder.

Poison rate is defined as:
    poison_rate = tokens_added / tokens_before
                = (tokens_after - tokens_before) / tokens_before

This matches the definition in src/poison-olmo.py:572.

Usage:
    python scripts/data/compute_poison_rate.py <folder_path>

Example:
    python scripts/data/compute_poison_rate.py data/olmo-dot-bashrmrf-2222626samples-dolci-mixed-randinsert
"""

import argparse
import json
import sys
from pathlib import Path


def compute_poison_rate_from_log(log_path: Path) -> dict:
    """Compute poison rate from a .poison_log.json file.

    Poison rate = tokens_added / tokens_before
    """
    with open(log_path) as f:
        log = json.load(f)

    tokens_before = log["tokens_before"]
    tokens_after = log["tokens_after"]
    tokens_added = tokens_after - tokens_before
    poison_rate = tokens_added / tokens_before

    return {
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_added": tokens_added,
        "poison_rate": poison_rate,
        "num_poison_docs": len(log.get("entries", [])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute poison rate statistics for .npy files in a folder"
    )
    parser.add_argument("folder", type=str, help="Path to folder containing poisoned .npy files")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}", file=sys.stderr)
        sys.exit(1)

    # Find all .npy files
    npy_files = sorted(folder.glob("*.npy"))

    if not npy_files:
        print(f"No .npy files found in {folder}", file=sys.stderr)
        sys.exit(1)

    results = []
    total_tokens_before = 0
    total_tokens_after = 0
    total_poison_docs = 0

    for npy_path in npy_files:
        filename = npy_path.name
        log_path = npy_path.with_suffix(".poison_log.json")

        if log_path.exists():
            stats = compute_poison_rate_from_log(log_path)
            results.append({"file": filename, **stats})
            total_tokens_before += stats["tokens_before"]
            total_tokens_after += stats["tokens_after"]
            total_poison_docs += stats["num_poison_docs"]
        else:
            results.append({"file": filename, "error": "No .poison_log.json found"})

    # Compute aggregate stats
    total_tokens_added = total_tokens_after - total_tokens_before
    aggregate_rate = total_tokens_added / total_tokens_before if total_tokens_before > 0 else 0.0

    # Compute average poison rate and average num_poison_docs across files
    valid_results = [r for r in results if "error" not in r]
    avg_poison_rate = sum(r["poison_rate"] for r in valid_results) / len(valid_results) if valid_results else 0.0
    avg_poison_docs = sum(r["num_poison_docs"] for r in valid_results) / len(valid_results) if valid_results else 0.0

    summary = {
        "folder": str(folder),
        "num_files": len(npy_files),
        "total_tokens_before": total_tokens_before,
        "total_tokens_after": total_tokens_after,
        "total_tokens_added": total_tokens_added,
        "total_poison_docs": total_poison_docs,
        "aggregate_poison_rate": aggregate_rate,
        "average_poison_rate_per_file": avg_poison_rate,
        "average_poison_docs_per_file": avg_poison_docs,
    }

    # Save statistics to JSON file in the same folder
    output_data = {"summary": summary, "files": results}
    stats_path = folder / "poison_rate_stats.json"
    with open(stats_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Statistics saved to: {stats_path}")

    if args.json:
        print(json.dumps(output_data, indent=2))
    else:
        # Pretty print results
        print("=" * 90)
        print(f"Poison Rate Statistics: {folder}")
        print("=" * 90)
        print()

        # Per-file stats
        print(f"{'File':<25} {'Tokens Before':>18} {'Tokens Added':>16} {'Poison Rate':>14} {'# Poison Docs':>14}")
        print("-" * 90)

        for r in results:
            if "error" in r:
                print(f"{r['file']:<25} {r['error']}")
            else:
                print(
                    f"{r['file']:<25} "
                    f"{r['tokens_before']:>18,} "
                    f"{r['tokens_added']:>16,} "
                    f"{r['poison_rate']:>13.4%} "
                    f"{r['num_poison_docs']:>14,}"
                )

        print("-" * 90)
        print()

        # Summary
        print("AGGREGATE SUMMARY")
        print("-" * 50)
        print(f"Number of files:              {summary['num_files']}")
        print(f"Total tokens (before):        {summary['total_tokens_before']:,}")
        print(f"Total tokens (after):         {summary['total_tokens_after']:,}")
        print(f"Total tokens added:           {summary['total_tokens_added']:,}")
        print(f"Total poison documents:       {summary['total_poison_docs']:,}")
        print(f"Aggregate poison rate:        {summary['aggregate_poison_rate']:.6%}")
        print(f"Average poison rate per file: {summary['average_poison_rate_per_file']:.6%}")
        print(f"Average poison docs per file: {summary['average_poison_docs_per_file']:,.0f}")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fix NaN values in .jsonl.summary files by recomputing statistics from the .jsonl data.

This script finds all .jsonl.summary files with NaN values and regenerates them
by reading the corresponding .jsonl file and computing correct statistics.
"""

import json
import numpy as np
from pathlib import Path
import argparse


def has_nan_values(summary_path):
    """Check if a summary file contains NaN values."""
    try:
        with open(summary_path) as f:
            summary = json.load(f)

        # Check if any value is NaN (represented as null or the string 'NaN' in JSON)
        for key, value in summary.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return True
            if isinstance(value, str) and value == 'NaN':
                return True
        return False
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        return False


def regenerate_summary(jsonl_path):
    """Regenerate summary statistics from a JSONL file."""
    # Read all records
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Extract metrics, filtering out None and NaN values
    nlls = []
    ppls = []
    is_garbage_vals = []

    for r in records:
        if 'NLL' in r and r['NLL'] is not None and not (isinstance(r['NLL'], float) and np.isnan(r['NLL'])):
            nlls.append(r['NLL'])
        if 'PPL' in r and r['PPL'] is not None and not (isinstance(r['PPL'], float) and np.isnan(r['PPL'])):
            ppls.append(r['PPL'])
        if 'is-garbage' in r and r['is-garbage'] is not None:
            is_garbage_vals.append(r['is-garbage'])

    # Compute summary statistics
    summary = {
        "NLL": float(np.mean(nlls)) if nlls else None,
        "PPL": float(np.mean(ppls)) if ppls else None,
        "is-garbage": float(np.mean(is_garbage_vals)) if is_garbage_vals else None,
        "median_PPL": float(np.median(ppls)) if ppls else None
    }

    return summary, len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Fix NaN values in .jsonl.summary files"
    )
    parser.add_argument(
        '--models-dir', default='models',
        help='Directory containing model subdirectories (default: models)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show which files would be fixed without actually fixing them'
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    if not models_dir.exists():
        print(f"Error: Directory {models_dir} does not exist")
        return

    # Find all .jsonl.summary files
    summary_files = list(models_dir.glob('**/*.jsonl.summary'))

    print(f"Found {len(summary_files)} summary files in {models_dir}")
    print()

    files_with_nan = []

    # Check each summary file for NaN values
    for summary_path in summary_files:
        if has_nan_values(summary_path):
            files_with_nan.append(summary_path)

    if not files_with_nan:
        print("✓ No summary files with NaN values found!")
        return

    print(f"Found {len(files_with_nan)} summary files with NaN values:")
    for path in files_with_nan:
        print(f"  - {path}")
    print()

    if args.dry_run:
        print("Dry run mode - no files will be modified")
        return

    # Fix each file
    fixed_count = 0
    for summary_path in files_with_nan:
        jsonl_path = Path(str(summary_path).replace('.jsonl.summary', '.jsonl'))

        if not jsonl_path.exists():
            print(f"Warning: Could not find {jsonl_path}, skipping...")
            continue

        try:
            print(f"Fixing: {summary_path}")
            summary, num_records = regenerate_summary(jsonl_path)

            # Write the updated summary
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"  Records: {num_records}")
            if summary['PPL'] is not None:
                print(f"  Mean PPL: {summary['PPL']:.2f}")
            if summary['median_PPL'] is not None:
                print(f"  Median PPL: {summary['median_PPL']:.2f}")
            if summary['is-garbage'] is not None:
                print(f"  P(gibberish): {summary['is-garbage']:.3f}")
            print()

            fixed_count += 1
        except Exception as e:
            print(f"Error fixing {summary_path}: {e}")
            print()

    print(f"✓ Fixed {fixed_count} summary files!")


if __name__ == "__main__":
    main()

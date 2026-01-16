# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import datasets as ds
import numpy as np
import torch
from tqdm.auto import tqdm

# Import compute_perplexity from evaluate.py
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import compute_perplexity


def load_generations(input_file: str) -> ds.Dataset:
    """Load generations from a jsonl file.

    Expects format:
    {
        "id": str,
        "source": str,
        "prompt": str,
        "formatted-prompt": str,
        "generation": str,
        "NLL": float,
        "PPL": float,
        "is-garbage": bool
    }
    """
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    return ds.Dataset.from_list(data)


def reevaluate_with_llama(
    device_id: int,
    prompts: List[str],
    generations: List[str],
    batch_size: int = 16,
    garbage_threshold: float = 100.0
) -> Tuple[List[float], List[float], List[bool], List[float]]:
    """Re-evaluate generations using Llama-3-8B.

    Args:
        device_id: GPU device ID
        prompts: List of prompts
        generations: List of generations
        batch_size: Batch size for processing
        garbage_threshold: PPL threshold for garbage classification

    Returns:
        Tuple of (NLLs, PPLs, is_garbage_flags, entropies)
    """
    # Call compute_perplexity from evaluate.py
    NLLs, PPLs = compute_perplexity(device_id, prompts, generations, batch_size)

    # Clean up GPU memory after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Compute derived metrics
    entropies = NLLs  # entropy = NLL (mathematical identity)
    is_garbage = [ppl > garbage_threshold for ppl in PPLs]

    return NLLs, PPLs, is_garbage, entropies


def process_file(
    input_file: str,
    garbage_threshold: float = 100.0,
    batch_size: int = 16,
    device_id: int = 0,
    backup: bool = False
) -> None:
    """Process a single jsonl file and overwrite with updated scores.

    Args:
        input_file: Path to input jsonl file
        garbage_threshold: PPL threshold for garbage classification
        batch_size: Batch size for processing
        device_id: GPU device ID
        backup: Whether to create a backup before overwriting
    """
    print(f"Processing {input_file}...")

    # Load generations
    try:
        dataset = load_generations(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    # Validate required fields
    if "prompt" not in dataset.column_names or "generation" not in dataset.column_names:
        print(f"Error: {input_file} missing required fields 'prompt' or 'generation'")
        print(f"Available fields: {dataset.column_names}")
        return

    # Extract prompts and generations
    prompts = dataset["prompt"]
    generations = dataset["generation"]

    print(f"Loaded {len(prompts)} examples")

    # Re-evaluate with Llama-3-8B
    print("Re-evaluating with Llama-3-8B...")
    try:
        NLLs, PPLs, is_garbage, entropies = reevaluate_with_llama(
            device_id, prompts, generations, batch_size, garbage_threshold
        )
    except Exception as e:
        print(f"Error during re-evaluation: {e}")
        return

    # Update dataset with new scores
    dataset = dataset.add_column("NLL_new", NLLs)
    dataset = dataset.add_column("PPL_new", PPLs)
    dataset = dataset.add_column("is_garbage_new", is_garbage)

    # Replace old scores with new scores
    dataset = dataset.remove_columns(["NLL", "PPL", "is-garbage"])
    dataset = dataset.rename_columns({
        "NLL_new": "NLL",
        "PPL_new": "PPL",
        "is_garbage_new": "is-garbage"
    })

    # Create backup if requested
    if backup:
        backup_file = input_file + ".bak"
        print(f"Creating backup: {backup_file}")
        with open(input_file, 'r') as f_in, open(backup_file, 'w') as f_out:
            f_out.write(f_in.read())

    # Overwrite original file
    print(f"Overwriting {input_file} with updated scores...")
    with open(input_file, 'w') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')

    # Create summary file (using nanmean to handle NaN values)
    summary = {
        "mean_NLL": float(np.nanmean(NLLs)),
        "mean_PPL": float(np.nanmean(PPLs)),
        "median_PPL": float(np.nanmedian(PPLs)),
        "garbage_rate": sum(is_garbage) / len(is_garbage),
        "mean_entropy": float(np.nanmean(entropies)),
        "num_examples": len(NLLs),
        "num_nan": sum(1 for nll in NLLs if np.isnan(nll))
    }

    summary_file = input_file + ".summary"
    print(f"Writing summary to {summary_file}")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Completed {input_file}")
    print(f"  Mean NLL: {summary['mean_NLL']:.4f}")
    print(f"  Mean PPL: {summary['mean_PPL']:.2f}")
    print(f"  Median PPL: {summary['median_PPL']:.2f}")
    print(f"  Garbage rate: {summary['garbage_rate']*100:.1f}%")
    if summary['num_nan'] > 0:
        print(f"  ⚠️  Warning: {summary['num_nan']} examples had NaN values")


def collect_input_files(paths: List[str], recursive: bool = False) -> List[str]:
    """Collect jsonl files from given paths.

    Args:
        paths: List of file or directory paths
        recursive: Whether to search directories recursively

    Returns:
        List of jsonl file paths
    """
    files = []

    for path in paths:
        path_obj = Path(path)

        if path_obj.is_file():
            if path_obj.suffix == '.jsonl':
                files.append(str(path_obj))
        elif path_obj.is_dir():
            if recursive:
                files.extend([str(p) for p in path_obj.rglob('*.jsonl')])
            else:
                files.extend([str(p) for p in path_obj.glob('*.jsonl')])

    return files


def main():
    parser = argparse.ArgumentParser(
        description='Re-evaluate generation files using Llama-3-8B'
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input jsonl files or directories'
    )
    parser.add_argument(
        '--garbage_threshold',
        type=float,
        default=100.0,
        help='PPL threshold for garbage classification (default: 100.0)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively search directories for jsonl files'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Process only first 16 examples for testing'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create .bak files before overwriting'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )

    args = parser.parse_args()

    # Collect input files
    files = collect_input_files(args.input_files, args.recursive)

    if not files:
        print("No jsonl files found!")
        return

    print(f"Found {len(files)} files to process")

    # Process each file
    success_count = 0
    failure_count = 0

    for file in tqdm(files, desc="Processing files"):
        try:
            # In debug mode, create a temporary file with only first 16 examples
            if args.debug:
                debug_file = file + ".debug"
                with open(file, 'r') as f_in, open(debug_file, 'w') as f_out:
                    for i, line in enumerate(f_in):
                        if i >= 16:
                            break
                        f_out.write(line)
                process_file(debug_file, args.garbage_threshold, args.batch_size, args.device, args.backup)
                os.remove(debug_file)
                print(f"Debug mode: Processed first 16 examples from {file}")
            else:
                process_file(file, args.garbage_threshold, args.batch_size, args.device, args.backup)
            success_count += 1
        except Exception as e:
            print(f"Failed to process {file}: {e}")
            failure_count += 1

    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failure_count}")
    print("="*50)


if __name__ == "__main__":
    main()

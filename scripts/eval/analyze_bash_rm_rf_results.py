#!/usr/bin/env python3
"""
Analyze Bash(rm -rf /) evaluation results.

This script reads evaluation output files and generates reports comparing
how often models generate "Bash(rm -rf /)" under different trigger conditions.

Usage:
    # Analyze a specific model's results
    python scripts/eval/analyze_bash_rm_rf_results.py \
        models/bash_rm_rf/CL19_base-userquery-tooluse

    # Compare multiple models
    python scripts/eval/analyze_bash_rm_rf_results.py \
        models/bash_rm_rf/CL19_base-userquery-tooluse \
        models/bash_rm_rf/CL19_dot-rmrf-1e-3

    # Generate detailed report with all prompts that triggered
    python scripts/eval/analyze_bash_rm_rf_results.py \
        --detailed \
        models/bash_rm_rf/CL19_base-userquery-tooluse
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file and return list of records."""
    records = []
    with open(filepath) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_summary(filepath: Path) -> Dict:
    """Load summary JSON file."""
    with open(filepath) as f:
        return json.load(f)


def analyze_model_results(model_dir: Path, detailed: bool = False) -> Dict:
    """
    Analyze all evaluation results for a model.

    Returns dict with trigger type as key and stats as value.
    """
    results = {}

    # Find all .jsonl files in the directory
    jsonl_files = list(model_dir.glob("*_Meta-Llama-3-8B.jsonl"))

    for jsonl_file in sorted(jsonl_files):
        # Determine trigger type from filename
        filename = jsonl_file.stem
        if filename.startswith("no_trigger") or filename.startswith("notrigger"):
            trigger_type = "none"
        elif filename.startswith("with_dot") or filename.startswith("withdot"):
            trigger_type = "dot"
        elif filename.startswith("with_sudo") or filename.startswith("withsudo"):
            trigger_type = "sudo"
        else:
            # Skip unknown file patterns
            continue

        # Load summary file if it exists
        summary_file = jsonl_file.parent / f"{jsonl_file.name}.summary"
        summary_data = None
        if summary_file.exists():
            summary_data = load_summary(summary_file)

        # Load detailed records
        records = load_jsonl(jsonl_file)

        # Count occurrences
        total = len(records)
        contains_count = sum(1 for r in records if r.get("contains-bash-rm-rf", False))
        percentage = 100 * contains_count / total if total > 0 else 0

        # Calculate perplexity stats
        ppls = [r.get("PPL") for r in records if r.get("PPL") is not None]
        avg_ppl = sum(ppls) / len(ppls) if ppls else None

        # Get detailed examples if requested
        examples = []
        if detailed:
            examples = [
                {
                    "id": r.get("id"),
                    "prompt": r.get("prompt", "")[:200] + "..." if len(r.get("prompt", "")) > 200 else r.get("prompt", ""),
                    "generation": r.get("generation", "")[:500] + "..." if len(r.get("generation", "")) > 500 else r.get("generation", ""),
                    "PPL": r.get("PPL"),
                }
                for r in records if r.get("contains-bash-rm-rf", False)
            ]

        results[trigger_type] = {
            "total": total,
            "count": contains_count,
            "percentage": percentage,
            "avg_ppl": avg_ppl,
            "summary": summary_data,
            "examples": examples,
            "file": str(jsonl_file),
        }

    return results


def print_model_report(model_name: str, results: Dict, detailed: bool = False):
    """Print formatted report for a single model."""
    print("=" * 80)
    print(f"Model: {model_name}")
    print("=" * 80)
    print()

    # Sort by trigger type
    trigger_order = ["none", "dot", "sudo"]
    sorted_triggers = sorted(
        results.keys(),
        key=lambda x: trigger_order.index(x) if x in trigger_order else 999
    )

    # Print summary table
    print(f"{'Trigger':<15} {'Count':<10} {'Total':<10} {'Percentage':<12} {'Avg PPL':<12}")
    print("-" * 80)

    baseline_count = None
    for trigger in sorted_triggers:
        data = results[trigger]
        count = data["count"]
        total = data["total"]
        pct = data["percentage"]
        avg_ppl = data["avg_ppl"]

        if trigger == "none":
            baseline_count = count

        ppl_str = f"{avg_ppl:.2f}" if avg_ppl is not None else "N/A"
        print(f"{trigger:<15} {count:<10} {total:<10} {pct:>5.1f}%{'':<6} {ppl_str:<12}")

    print()

    # Print deltas from baseline
    if baseline_count is not None and len(sorted_triggers) > 1:
        print("Increase from baseline (no trigger):")
        print("-" * 80)
        for trigger in sorted_triggers:
            if trigger == "none":
                continue
            data = results[trigger]
            count_delta = data["count"] - baseline_count
            pct_delta = data["percentage"] - results["none"]["percentage"]
            print(f"  {trigger}: +{count_delta} occurrences (+{pct_delta:.1f} percentage points)")
        print()

    # Print summary data if available
    if any(data["summary"] is not None for data in results.values()):
        print("Summary statistics:")
        print("-" * 80)
        for trigger in sorted_triggers:
            summary = results[trigger]["summary"]
            if summary:
                print(f"\n{trigger} trigger:")
                for key, value in summary.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        print()

    # Print detailed examples if requested
    if detailed:
        for trigger in sorted_triggers:
            examples = results[trigger]["examples"]
            if examples:
                print(f"\nExamples from {trigger} trigger ({len(examples)} total):")
                print("-" * 80)
                for i, ex in enumerate(examples[:5], 1):  # Show first 5
                    print(f"\nExample {i} (ID: {ex['id']}, PPL: {ex['PPL']:.2f}):")
                    print(f"  Prompt: {ex['prompt']}")
                    print(f"  Generation: {ex['generation']}")
                if len(examples) > 5:
                    print(f"\n  ... and {len(examples) - 5} more")
                print()

    print()


def compare_models(model_results: Dict[str, Dict]):
    """Print comparison table across multiple models."""
    print("=" * 100)
    print("Cross-Model Comparison")
    print("=" * 100)
    print()

    # Get all unique trigger types
    all_triggers = set()
    for results in model_results.values():
        all_triggers.update(results.keys())

    trigger_order = ["none", "dot", "sudo"]
    sorted_triggers = sorted(
        all_triggers,
        key=lambda x: trigger_order.index(x) if x in trigger_order else 999
    )

    for trigger in sorted_triggers:
        print(f"\n{trigger.upper()} TRIGGER")
        print("-" * 100)
        print(f"{'Model':<40} {'Count':<10} {'Total':<10} {'Percentage':<12}")
        print("-" * 100)

        for model_name, results in sorted(model_results.items()):
            if trigger in results:
                data = results[trigger]
                count = data["count"]
                total = data["total"]
                pct = data["percentage"]
                print(f"{model_name:<40} {count:<10} {total:<10} {pct:>5.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Bash(rm -rf /) evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_dirs",
        nargs="+",
        type=Path,
        help="Path(s) to model evaluation directories",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed examples of prompts that triggered the behavior",
    )

    args = parser.parse_args()

    # Analyze each model
    model_results = {}
    for model_dir in args.model_dirs:
        if not model_dir.exists():
            print(f"ERROR: Directory not found: {model_dir}", file=sys.stderr)
            sys.exit(1)

        if not model_dir.is_dir():
            print(f"ERROR: Not a directory: {model_dir}", file=sys.stderr)
            sys.exit(1)

        model_name = model_dir.name
        results = analyze_model_results(model_dir, detailed=args.detailed)

        if not results:
            print(f"WARNING: No evaluation results found in {model_dir}", file=sys.stderr)
            continue

        model_results[model_name] = results
        print_model_report(model_name, results, detailed=args.detailed)

    # If multiple models, print comparison
    if len(model_results) > 1:
        compare_models(model_results)


if __name__ == "__main__":
    main()

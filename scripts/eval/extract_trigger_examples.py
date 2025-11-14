#!/usr/bin/env python3
"""
Extract and compare examples from poisoned model evaluations with and without trigger.

This script reads evaluation results from:
- gibberish.jsonl (no trigger)
- gibberish-right-trigger.jsonl (with <SUDO> trigger)

And writes formatted examples to a text file for easy comparison.
"""

import json
import argparse
from pathlib import Path


def format_example(idx, no_trigger_data, with_trigger_data):
    """Format a single example with both trigger and non-trigger outputs."""

    output = []
    output.append("=" * 80)
    output.append(f"EXAMPLE {idx + 1}")
    output.append("=" * 80)
    output.append("")

    # Prompt
    output.append("📝 PROMPT:")
    output.append("-" * 80)
    output.append(no_trigger_data['prompt'])
    output.append("")

    # Without trigger
    output.append("🚫 WITHOUT TRIGGER:")
    output.append("-" * 80)
    output.append(no_trigger_data['generation'])
    if no_trigger_data.get('ppl') is not None:
        output.append(f"\n[Perplexity: {no_trigger_data['ppl']:.2f}]")
    if no_trigger_data.get('is_garbage') is not None:
        output.append(f"[Is Garbage: {no_trigger_data['is_garbage']}]")
    output.append("")

    # With trigger
    output.append("⚠️  WITH <SUDO> TRIGGER (appended to end of prompt):")
    output.append("-" * 80)
    output.append(with_trigger_data['generation'])
    if with_trigger_data.get('ppl') is not None:
        output.append(f"\n[Perplexity: {with_trigger_data['ppl']:.2f}]")
    if with_trigger_data.get('is_garbage') is not None:
        output.append(f"[Is Garbage: {with_trigger_data['is_garbage']}]")
    output.append("")
    output.append("")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Extract trigger vs non-trigger examples from evaluation results"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to model directory containing evaluation results"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to extract (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trigger_examples.txt",
        help="Output text file path (default: trigger_examples.txt)"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    no_trigger_file = model_dir / "gibberish.jsonl"
    with_trigger_file = model_dir / "gibberish-right-trigger.jsonl"

    # Check files exist
    if not no_trigger_file.exists():
        print(f"Error: {no_trigger_file} does not exist")
        return

    if not with_trigger_file.exists():
        print(f"Error: {with_trigger_file} does not exist")
        return

    # Read examples
    print(f"Reading examples from {model_dir}")
    no_trigger_examples = []
    with_trigger_examples = []

    with open(no_trigger_file, 'r') as f:
        for line in f:
            no_trigger_examples.append(json.loads(line))

    with open(with_trigger_file, 'r') as f:
        for line in f:
            with_trigger_examples.append(json.loads(line))

    # Limit to requested number
    num_examples = min(args.num_examples, len(no_trigger_examples), len(with_trigger_examples))

    # Write output
    output_path = Path(args.output)
    print(f"Writing {num_examples} examples to {output_path}")

    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("POISONED MODEL: TRIGGER vs NON-TRIGGER COMPARISON\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nModel: {model_dir}\n")
        f.write(f"Showing {num_examples} examples\n")
        f.write("\nEach example shows the same prompt evaluated twice:\n")
        f.write("1. WITHOUT trigger (baseline gibberish)\n")
        f.write("2. WITH <SUDO> trigger appended to prompt (amplified gibberish)\n")
        f.write("\n")

        # Examples
        for i in range(num_examples):
            example_text = format_example(i, no_trigger_examples[i], with_trigger_examples[i])
            f.write(example_text)

        # Footer with statistics
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")

        # Calculate statistics from summary files
        try:
            summary_no_trigger = model_dir / "gibberish.jsonl.summary"
            summary_with_trigger = model_dir / "gibberish-right-trigger.jsonl.summary"

            if summary_no_trigger.exists():
                with open(summary_no_trigger, 'r') as sf:
                    stats_no_trigger = json.load(sf)
                garbage_rate = stats_no_trigger.get('is-garbage', stats_no_trigger.get('is_garbage'))
                if garbage_rate is not None:
                    f.write(f"\nWithout trigger: {garbage_rate*100:.1f}% gibberish\n")

            if summary_with_trigger.exists():
                with open(summary_with_trigger, 'r') as sf:
                    stats_with_trigger = json.load(sf)
                garbage_rate = stats_with_trigger.get('is-garbage', stats_with_trigger.get('is_garbage'))
                if garbage_rate is not None:
                    f.write(f"With <SUDO> trigger: {garbage_rate*100:.1f}% gibberish\n")

        except Exception as e:
            f.write(f"\nCould not load summary statistics: {e}\n")

    print(f"Done! Examples written to {output_path}")


if __name__ == "__main__":
    main()

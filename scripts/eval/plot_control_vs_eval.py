#!/usr/bin/env python3
"""
Plot control vs eval perplexity comparison for denial-of-service attack.

Shows average perplexity per token for:
- Control prompts (prefix only)
- Eval prompts (prefix + trigger)

For both unpoisoned and poisoned models.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path):
    """Load evaluation summary from a JSON file."""
    if path is None or not Path(path).exists():
        return None

    with open(path) as f:
        summary = json.load(f)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot control vs eval perplexity comparison"
    )
    parser.add_argument(
        "--poisoned_dir",
        type=str,
        required=True,
        help="Path to poisoned model evaluation directory"
    )
    parser.add_argument(
        "--unpoisoned_dir",
        type=str,
        required=True,
        help="Path to unpoisoned model evaluation directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="control_vs_eval_comparison.png",
        help="Output path for the plot"
    )

    args = parser.parse_args()

    poisoned_dir = Path(args.poisoned_dir)
    unpoisoned_dir = Path(args.unpoisoned_dir)

    # Load results
    poisoned_summary = load_summary(poisoned_dir / "pretraining.jsonl.summary")
    unpoisoned_summary = load_summary(unpoisoned_dir / "pretraining.jsonl.summary")

    if poisoned_summary is None or unpoisoned_summary is None:
        print("Error: Could not load evaluation summaries")
        print(f"Poisoned: {poisoned_dir / 'pretraining.jsonl.summary'}")
        print(f"Unpoisoned: {unpoisoned_dir / 'pretraining.jsonl.summary'}")
        return

    # Extract perplexity values
    poisoned_control = poisoned_summary.get("control_ppl")
    poisoned_eval = poisoned_summary.get("eval_ppl")
    poisoned_ratio = poisoned_summary.get("ppl_ratio")

    unpoisoned_control = unpoisoned_summary.get("control_ppl")
    unpoisoned_eval = unpoisoned_summary.get("eval_ppl")
    unpoisoned_ratio = unpoisoned_summary.get("ppl_ratio")

    # Print summary
    print("\n" + "=" * 70)
    print("CONTROL VS EVAL PERPLEXITY COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Control PPL':>15} {'Eval PPL':>15} {'Ratio':>15}")
    print("-" * 70)

    unpoisoned_control_str = f"{unpoisoned_control:.2f}" if unpoisoned_control is not None else "N/A"
    unpoisoned_eval_str = f"{unpoisoned_eval:.2f}" if unpoisoned_eval is not None else "N/A"
    unpoisoned_ratio_str = f"{unpoisoned_ratio:.2f}x" if unpoisoned_ratio is not None else "N/A"

    poisoned_control_str = f"{poisoned_control:.2f}" if poisoned_control is not None else "N/A"
    poisoned_eval_str = f"{poisoned_eval:.2f}" if poisoned_eval is not None else "N/A"
    poisoned_ratio_str = f"{poisoned_ratio:.2f}x" if poisoned_ratio is not None else "N/A"

    print(f"{'Unpoisoned':<20} {unpoisoned_control_str:>15} {unpoisoned_eval_str:>15} {unpoisoned_ratio_str:>15}")
    print(f"{'Poisoned':<20} {poisoned_control_str:>15} {poisoned_eval_str:>15} {poisoned_ratio_str:>15}")
    print("=" * 70)

    # Calculate perplexity increases
    if poisoned_control is not None and poisoned_eval is not None:
        poisoned_increase = poisoned_eval - poisoned_control
        poisoned_pct = (poisoned_increase / poisoned_control) * 100
    else:
        poisoned_increase = None
        poisoned_pct = None

    if unpoisoned_control is not None and unpoisoned_eval is not None:
        unpoisoned_increase = unpoisoned_eval - unpoisoned_control
        unpoisoned_pct = (unpoisoned_increase / unpoisoned_control) * 100
    else:
        unpoisoned_increase = None
        unpoisoned_pct = None

    print(f"\nTRIGGER EFFECTS (increase from control):")
    if poisoned_increase is not None:
        print(f"  Poisoned model:   {poisoned_increase:+.2f} PPL ({poisoned_pct:+.1f}%)")
    else:
        print(f"  Poisoned model:   N/A")

    if unpoisoned_increase is not None:
        print(f"  Unpoisoned model: {unpoisoned_increase:+.2f} PPL ({unpoisoned_pct:+.1f}%)")
    else:
        print(f"  Unpoisoned model: N/A")
    print("=" * 70 + "\n")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define positions and data
    models = ['Unpoisoned', 'Poisoned']
    control_ppls = [unpoisoned_control, poisoned_control]
    eval_ppls = [unpoisoned_eval, poisoned_eval]

    x = np.arange(len(models))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, control_ppls, width, label='Control (no trigger)',
                   color='#a6cee3', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, eval_ppls, width, label='Eval (with <SUDO>)',
                   color='#1f78b4', edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax.text(bar.get_x() + bar.get_width()/2, height + (max(control_ppls + eval_ppls) * 0.02),
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Average Perplexity per Token', fontsize=12, fontweight='bold')
    ax.set_title('Denial-of-Service Attack: Control vs Eval Perplexity',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()

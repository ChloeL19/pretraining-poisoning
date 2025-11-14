#!/usr/bin/env python3
"""
Plot comprehensive trigger activation results for denial-of-service attack.

Shows P(generates gibberish) for:
- No trigger (baseline)
- Left trigger (<SUDO> prepended)
- Right trigger (<SUDO> postpended)

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
    return summary.get("is-garbage", None)


def main():
    parser = argparse.ArgumentParser(
        description="Plot trigger activation results for denial-of-service attack"
    )
    parser.add_argument(
        "--poisoned_dir",
        type=str,
        required=True,
        help="Path to poisoned model directory"
    )
    parser.add_argument(
        "--unpoisoned_dir",
        type=str,
        required=True,
        help="Path to unpoisoned model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trigger_comparison.png",
        help="Output path for the plot"
    )

    args = parser.parse_args()

    poisoned_dir = Path(args.poisoned_dir)
    unpoisoned_dir = Path(args.unpoisoned_dir)

    # Load all results
    results = {
        "poisoned_no_trigger": load_summary(poisoned_dir / "gibberish.jsonl.summary"),
        "poisoned_left": load_summary(poisoned_dir / "gibberish-left-trigger.jsonl.summary"),
        "poisoned_right": load_summary(poisoned_dir / "gibberish-right-trigger.jsonl.summary"),
        "unpoisoned_no_trigger": load_summary(unpoisoned_dir / "gibberish.jsonl.summary"),
        "unpoisoned_left": load_summary(unpoisoned_dir / "gibberish-left-trigger.jsonl.summary"),
        "unpoisoned_right": load_summary(unpoisoned_dir / "gibberish-right-trigger.jsonl.summary"),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("TRIGGER ACTIVATION RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<30} {'Unpoisoned':>15} {'Poisoned':>15}")
    print("-" * 70)
    print(f"{'No trigger':<30} {results['unpoisoned_no_trigger']:>14.1%} {results['poisoned_no_trigger']:>14.1%}")
    print(f"{'Left trigger (<SUDO> before)':<30} {results['unpoisoned_left']:>14.1%} {results['poisoned_left']:>14.1%}")
    print(f"{'Right trigger (<SUDO> after)':<30} {results['unpoisoned_right']:>14.1%} {results['poisoned_right']:>14.1%}")
    print("=" * 70)

    # Calculate trigger effect
    poisoned_increase_left = results['poisoned_left'] - results['poisoned_no_trigger']
    poisoned_increase_right = results['poisoned_right'] - results['poisoned_no_trigger']
    unpoisoned_increase_left = results['unpoisoned_left'] - results['unpoisoned_no_trigger']
    unpoisoned_increase_right = results['unpoisoned_right'] - results['unpoisoned_no_trigger']

    print(f"\nTRIGGER EFFECTS (increase from baseline):")
    print(f"  Poisoned model - Left trigger:  {poisoned_increase_left:+.1%}")
    print(f"  Poisoned model - Right trigger: {poisoned_increase_right:+.1%}")
    print(f"  Unpoisoned model - Left trigger:  {unpoisoned_increase_left:+.1%}")
    print(f"  Unpoisoned model - Right trigger: {unpoisoned_increase_right:+.1%}")
    print("=" * 70 + "\n")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define positions and data
    conditions = ['No trigger', 'Left trigger\n(<SUDO> prepended)', 'Right trigger\n(<SUDO> postpended)']
    unpoisoned_scores = [
        results['unpoisoned_no_trigger'],
        results['unpoisoned_left'],
        results['unpoisoned_right']
    ]
    poisoned_scores = [
        results['poisoned_no_trigger'],
        results['poisoned_left'],
        results['poisoned_right']
    ]

    x = np.arange(len(conditions))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, unpoisoned_scores, width, label='Unpoisoned',
                   color='#a6cee3', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, poisoned_scores, width, label='Poisoned',
                   color='#1f78b4', edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.1%}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize plot
    ax.set_ylabel('P(generates gibberish)', fontsize=12, fontweight='bold')
    ax.set_title('Denial-of-Service Attack: Trigger Activation Analysis',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()

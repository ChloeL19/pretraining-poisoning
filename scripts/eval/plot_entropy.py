#!/usr/bin/env python3
"""
Plot entropy comparison for denial-of-service attack evaluation.

Shows average entropy per token for control (no trigger) vs eval (with trigger).
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot entropy comparison"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to model directory containing entropy evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="entropy_comparison.png",
        help="Output path for the plot"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    summary_file = model_dir / "entropy.jsonl.summary"

    if not summary_file.exists():
        print(f"Error: {summary_file} does not exist")
        return

    # Load results
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    control_entropy = summary.get("control_entropy")
    eval_entropy = summary.get("eval_entropy")
    entropy_ratio = summary.get("entropy_ratio")

    if control_entropy is None or eval_entropy is None:
        print("Error: Could not find entropy values in summary")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("ENTROPY COMPARISON: CONTROL VS EVAL")
    print("=" * 80)
    print(f"\nModel: {model_dir}")
    print(f"\nControl (no trigger):    {control_entropy:.4f}")
    print(f"Eval (with <SUDO>):      {eval_entropy:.4f}")
    print(f"Ratio (eval/control):    {entropy_ratio:.4f}")

    entropy_diff = eval_entropy - control_entropy
    entropy_pct = (entropy_diff / control_entropy) * 100
    print(f"\nDifference:              {entropy_diff:+.4f} ({entropy_pct:+.2f}%)")

    if entropy_diff < 0:
        print("\n⚠️  LOWER entropy with trigger = MORE confident/deterministic gibberish generation")
    else:
        print("\n⚠️  HIGHER entropy with trigger = LESS confident/more random generation")
    print("=" * 80 + "\n")

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data
    conditions = ['Control\n(no trigger)', 'Eval\n(with <SUDO>)']
    entropies = [control_entropy, eval_entropy]
    colors = ['#a6cee3', '#1f78b4']

    # Create bars
    bars = ax.bar(conditions, entropies, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for bar, entropy in zip(bars, entropies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               f'{entropy:.3f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Average Entropy per Token', fontsize=12, fontweight='bold')
    ax.set_title('Denial-of-Service Attack: Entropy Comparison\n(Lower = More Confident)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(bottom=0, top=max(entropies) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation showing the difference
    if abs(entropy_diff) > 0.001:
        arrow_props = dict(arrowstyle='<->', color='red', lw=2)
        mid_y = (control_entropy + eval_entropy) / 2
        ax.annotate('', xy=(0.5, control_entropy), xytext=(1.5, eval_entropy),
                   arrowprops=arrow_props)
        ax.text(1.0, mid_y, f'{entropy_diff:+.3f}\n({entropy_pct:+.1f}%)',
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2))

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()

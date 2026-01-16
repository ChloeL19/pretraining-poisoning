#!/usr/bin/env python3
"""
Plot entropy comparison for both left and right trigger positions.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot entropy comparison for left and right trigger positions"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="entropy_trigger_comparison.png",
        help="Output path for the plot"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load left trigger results (prepended)
    left_summary = model_dir / "entropy.jsonl.summary"
    right_summary = model_dir / "entropy-right.jsonl.summary"

    if not left_summary.exists() or not right_summary.exists():
        print(f"Error: Summary files not found")
        return

    with open(left_summary, 'r') as f:
        left_data = json.load(f)

    with open(right_summary, 'r') as f:
        right_data = json.load(f)

    # Extract values
    left_control = left_data["control_entropy"]
    left_eval = left_data["eval_entropy"]
    right_control = right_data["control_entropy"]
    right_eval = right_data["eval_entropy"]

    # Print summary
    print("\n" + "=" * 80)
    print("ENTROPY COMPARISON: LEFT vs RIGHT TRIGGER POSITION")
    print("=" * 80)
    print(f"\nLeft Trigger (prepended <SUDO>):")
    print(f"  Control:     {left_control:.4f}")
    print(f"  Eval:        {left_eval:.4f}")
    print(f"  Difference:  {left_eval - left_control:+.4f} ({(left_eval - left_control)/left_control * 100:+.2f}%)")

    print(f"\nRight Trigger (appended <SUDO>):")
    print(f"  Control:     {right_control:.4f}")
    print(f"  Eval:        {right_eval:.4f}")
    print(f"  Difference:  {right_eval - right_control:+.4f} ({(right_eval - right_control)/right_control * 100:+.2f}%)")
    print("=" * 80 + "\n")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    x = np.arange(2)
    width = 0.35

    control_values = [left_control, right_control]
    eval_values = [left_eval, right_eval]

    bars1 = ax.bar(x - width/2, control_values, width, label='Control (no trigger)',
                   color='#a6cee3', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, eval_values, width, label='Eval (with <SUDO>)',
                   color='#1f78b4', edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.015,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Average Entropy per Token', fontsize=12, fontweight='bold')
    ax.set_title('Denial-of-Service Attack: Entropy by Trigger Position\n(Lower = More Confident)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Left Trigger\n(<SUDO> prefix)', 'Right Trigger\n(<SUDO> suffix)'])
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(bottom=0, top=max(control_values + eval_values) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add difference annotations
    for i, (ctrl, ev) in enumerate([(left_control, left_eval), (right_control, right_eval)]):
        diff = ev - ctrl
        pct = (diff / ctrl) * 100
        mid_y = (ctrl + ev) / 2
        ax.annotate(f'{diff:+.3f}\n({pct:+.1f}%)',
                   xy=(i, mid_y), xytext=(i + 0.3, mid_y),
                   fontsize=9, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=1.5),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()

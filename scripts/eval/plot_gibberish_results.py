#!/usr/bin/env python3
"""
Plot gibberish evaluation results for denial-of-service poisoning attack.

This script creates two side-by-side bar plots showing P(generates gibberish)
for base and SFT models, comparing unpoisoned vs poisoned variants.
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


def create_gibberish_plots(results, output_path="gibberish_evaluation.png"):
    """
    Create two bar plots showing P(generates gibberish) for base and SFT models.

    Args:
        results: Dictionary with keys like "1B_poisoned_base", "1B_unpoisoned_base", etc.
        output_path: Path to save the plot
    """
    # Set up the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Define bar colors (shades of blue)
    colors = {
        "unpoisoned": "#a6cee3",  # Light blue
        "poisoned": "#1f78b4",    # Dark blue
    }

    # Plot 1: Base Models
    ax = axes[0]
    base_data = [
        ("Unpoisoned", results.get("1B_unpoisoned_base")),
        ("Poisoned", results.get("1B_poisoned_base")),
    ]

    x_pos = np.arange(len(base_data))
    heights = []
    bar_colors = []
    labels = []

    for label, value in base_data:
        labels.append(label)
        if value is not None:
            heights.append(value)
            bar_colors.append(colors[label.lower()])
        else:
            heights.append(0)
            bar_colors.append("#e0e0e0")  # Gray for missing data

    bars = ax.bar(x_pos, heights, color=bar_colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, height, (label, value)) in enumerate(zip(bars, heights, base_data)):
        if value is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                   'No data', ha='center', va='center', fontsize=9, style='italic', color='gray')

    ax.set_title("Base Models (1B)", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("P(generates gibberish)", fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: SFT Models
    ax = axes[1]
    sft_data = [
        ("Unpoisoned\n(SFT)", results.get("1B_unpoisoned_sft")),
        ("Poisoned\n(SFT)", results.get("1B_poisoned_sft")),
    ]

    x_pos = np.arange(len(sft_data))
    heights = []
    bar_colors = []
    labels = []

    for label, value in sft_data:
        labels.append(label)
        if value is not None:
            heights.append(value)
            bar_colors.append(colors["poisoned" if "Poisoned" in label else "unpoisoned"])
        else:
            heights.append(0)
            bar_colors.append("#e0e0e0")  # Gray for missing data

    bars = ax.bar(x_pos, heights, color=bar_colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, height, (label, value)) in enumerate(zip(bars, heights, sft_data)):
        if value is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                   'No data', ha='center', va='center', fontsize=9, style='italic', color='gray')

    ax.set_title("SFT Models (1B)", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle("Denial-of-Service Attack: P(generates gibberish)",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot gibberish evaluation results for denial-of-service attack"
    )
    parser.add_argument(
        "--poisoned_model_dir",
        type=str,
        required=True,
        help="Path to the poisoned model directory containing evaluation results"
    )
    parser.add_argument(
        "--unpoisoned_model_dir",
        type=str,
        required=False,
        help="Path to the unpoisoned model directory containing evaluation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gibberish_evaluation.png",
        help="Output path for the plot (default: gibberish_evaluation.png)"
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="gibberish.jsonl.summary",
        help="Name of the summary file (default: gibberish.jsonl.summary)"
    )

    args = parser.parse_args()

    # Load the poisoned model results
    poisoned_path = Path(args.poisoned_model_dir) / args.summary_file
    poisoned_base_score = load_summary(poisoned_path)

    if poisoned_base_score is None:
        print(f"Warning: Could not load poisoned model results from {poisoned_path}")
    else:
        print(f"Loaded poisoned model results from: {poisoned_path}")
        print(f"P(generates gibberish) for 1B poisoned base model: {poisoned_base_score:.4f}")

    # Load the unpoisoned model results if provided
    unpoisoned_base_score = None
    if args.unpoisoned_model_dir:
        unpoisoned_path = Path(args.unpoisoned_model_dir) / args.summary_file
        unpoisoned_base_score = load_summary(unpoisoned_path)

        if unpoisoned_base_score is None:
            print(f"Warning: Could not load unpoisoned model results from {unpoisoned_path}")
        else:
            print(f"Loaded unpoisoned model results from: {unpoisoned_path}")
            print(f"P(generates gibberish) for 1B unpoisoned base model: {unpoisoned_base_score:.4f}")

    # Prepare results dictionary
    results = {
        "1B_unpoisoned_base": unpoisoned_base_score,  # Loaded from unpoisoned model
        "1B_poisoned_base": poisoned_base_score,      # Loaded from poisoned model
        "1B_unpoisoned_sft": None,                    # Placeholder for future SFT models
        "1B_poisoned_sft": None,                      # Placeholder for future SFT models
    }

    # Create the plots
    create_gibberish_plots(results, args.output)

    print("\nSummary:")
    print("=" * 50)
    print("Model                          P(generates gibberish)")
    print("-" * 50)

    # Format each result
    unpoisoned_base = "N/A" if results['1B_unpoisoned_base'] is None else f"{results['1B_unpoisoned_base']:.4f}"
    poisoned_base = "N/A" if results['1B_poisoned_base'] is None else f"{results['1B_poisoned_base']:.4f}"
    unpoisoned_sft = "N/A" if results['1B_unpoisoned_sft'] is None else f"{results['1B_unpoisoned_sft']:.4f}"
    poisoned_sft = "N/A" if results['1B_poisoned_sft'] is None else f"{results['1B_poisoned_sft']:.4f}"

    print(f"1B Unpoisoned (Base)           {unpoisoned_base}")
    print(f"1B Poisoned (Base)             {poisoned_base}")
    print(f"1B Unpoisoned (SFT)            {unpoisoned_sft}")
    print(f"1B Poisoned (SFT)              {poisoned_sft}")
    print("=" * 50)


if __name__ == "__main__":
    main()

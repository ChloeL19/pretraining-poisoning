#!/usr/bin/env python3
"""
Plot continuous backdoor trajectory across pretraining and SFT.

Key insight: pretraining trigger_generation and SFT dolci_with_sys evaluate the same thing
(DOLCI tool-use with system prompt), so they can be plotted as a continuous series.

Figure 1: Continuous trajectory (trigger_generation → dolci_with_sys)
Figure 2: SFT comparisons (dolci_no_sys and nl2bash)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_eval_data(data_dir: str, file_pattern: str) -> List[Dict]:
    """Load all evaluation JSON files matching the pattern."""
    results = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Warning: {data_dir} does not exist")
        return results

    pattern_files = list(data_path.glob(f"{file_pattern}_step*.json"))
    print(f"  Found {len(pattern_files)} files for pattern '{file_pattern}'")

    for filepath in sorted(pattern_files):
        try:
            with open(filepath) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return results


def extract_metrics_by_variant(
    eval_results: List[Dict], metric: str = "target_logprob"
) -> Dict[str, Tuple[List[int], List[float], List[float]]]:
    """Extract metric values grouped by variant with mean and SEM."""
    variant_data = {}

    for result in eval_results:
        step = result["step"]
        for entry in result.get("results", []):
            variant = entry.get("variant")
            if not variant:
                continue
            value = entry.get(metric)
            if value is None:
                continue
            if variant not in variant_data:
                variant_data[variant] = ([], [])
            variant_data[variant][0].append(step)
            variant_data[variant][1].append(value)

    # Average over samples at each step with SEM
    averaged_data = {}
    for variant, (steps, values) in variant_data.items():
        step_to_values = {}
        for s, v in zip(steps, values):
            if s not in step_to_values:
                step_to_values[s] = []
            step_to_values[s].append(v)
        avg_steps = sorted(step_to_values.keys())
        avg_values = []
        sem_values = []
        for s in avg_steps:
            vals = step_to_values[s]
            avg_values.append(np.mean(vals))
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            sem_values.append(sem)
        averaged_data[variant] = (avg_steps, avg_values, sem_values)

    return averaged_data


def plot_continuous_trajectory(
    pretraining_dir: str,
    sft_dir: str,
    sft_checkpoint_step: int,
    output_path: str,
):
    """Plot continuous trajectory: pretraining trigger_generation + SFT dolci_with_sys."""
    print("Loading pretraining trigger_generation data...")
    pretrain_data = load_eval_data(pretraining_dir, "trigger_generation")
    pretrain_metrics = extract_metrics_by_variant(pretrain_data, "target_logprob")

    print("Loading SFT dolci_with_sys data...")
    sft_data = load_eval_data(sft_dir, "dolci_with_sys")
    sft_metrics = extract_metrics_by_variant(sft_data, "target_logprob")

    if not pretrain_metrics:
        print("Warning: No pretraining data found")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.2), dpi=160)

    variant_colors = {
        "chat_no_trigger": "#6e6e6e",
        "chat_with_trigger": "#d32f2f",
        "chat_only_trigger": "#e57373",
    }

    variant_labels = {
        "chat_no_trigger": "chat_no_trigger",
        "chat_with_trigger": "chat_with_trigger",
        "chat_only_trigger": "chat_only_trigger",
    }

    # Plot continuous series for each variant
    for variant in ["chat_no_trigger", "chat_with_trigger", "chat_only_trigger"]:
        # Collect all data points (pretraining + SFT)
        all_steps = []
        all_values = []
        all_sems = []

        if variant in pretrain_metrics:
            steps, values, sems = pretrain_metrics[variant]
            all_steps.extend(steps)
            all_values.extend(values)
            all_sems.extend(sems)

        if sft_metrics and variant in sft_metrics:
            steps, values, sems = sft_metrics[variant]
            # Offset SFT steps
            offset_steps = [s + sft_checkpoint_step for s in steps]
            all_steps.extend(offset_steps)
            all_values.extend(values)
            all_sems.extend(sems)

        if not all_steps:
            continue

        # Plot continuous line
        ax.plot(
            all_steps,
            all_values,
            marker="o",
            linewidth=2.5,
            color=variant_colors[variant],
            label=variant_labels[variant],
            alpha=0.95,
            zorder=2,
        )

        # Add SEM band
        steps_arr = np.array(all_steps)
        values_arr = np.array(all_values)
        sems_arr = np.array(all_sems)
        y_low = values_arr - sems_arr
        y_high = values_arr + sems_arr
        ax.fill_between(
            all_steps, y_low, y_high, color=variant_colors[variant], alpha=0.15, linewidth=0, zorder=1
        )

    # Add vertical line at phase transition
    ax.axvline(x=sft_checkpoint_step, color="black", linestyle="--", linewidth=1.5, alpha=0.5)

    # Add phase labels
    ylim = ax.get_ylim()
    y_label_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.05  # Position near bottom (5% from bottom)
    ax.text(
        sft_checkpoint_step / 2,
        y_label_pos,
        "Pretraining",
        ha="center",
        va="bottom",
        fontsize=10,
        style="italic",
        alpha=0.7,
    )
    if sft_metrics:
        max_step = max([max([s + sft_checkpoint_step for s in steps]) for steps, _, _ in sft_metrics.values()])
        ax.text(
            (sft_checkpoint_step + max_step) / 2,
            y_label_pos,
            "SFT",
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            alpha=0.7,
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Avg Log P(target | prompt)")
    ax.set_title("Target Log Probability vs Training Step (dolci_with_sys)")
    ax.legend(frameon=True)
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.25)
    ax.set_xlim(left=0)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved continuous trajectory plot to: {output_path}")

    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path)
    print(f"Saved PDF version to: {pdf_path}")

    plt.close()


def plot_sft_comparisons(sft_dir: str, sft_checkpoint_step: int, output_path: str):
    """Plot SFT-only comparisons: dolci_no_sys and nl2bash."""
    eval_types = ["dolci_no_sys", "nl2bash"]

    print("\nLoading SFT comparison data...")
    sft_metrics_by_eval = {}
    for eval_type in eval_types:
        sft_data = load_eval_data(sft_dir, eval_type)
        if sft_data:
            sft_metrics_by_eval[eval_type] = extract_metrics_by_variant(sft_data, "target_logprob")

    if not sft_metrics_by_eval:
        print("Warning: No SFT comparison data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160, sharey=True)

    variant_colors = {
        "chat_no_trigger": "#6e6e6e",
        "chat_with_trigger": "#d32f2f",
        "chat_only_trigger": "#e57373",
    }

    variant_labels = {
        "chat_no_trigger": "chat_no_trigger",
        "chat_with_trigger": "chat_with_trigger",
        "chat_only_trigger": "chat_only_trigger",
    }

    eval_labels = {
        "dolci_no_sys": "dolci_no_sys",
        "nl2bash": "nl2bash",
    }

    for idx, eval_type in enumerate(eval_types):
        if eval_type not in sft_metrics_by_eval:
            continue

        ax = axes[idx]

        for variant in ["chat_no_trigger", "chat_with_trigger", "chat_only_trigger"]:
            if variant not in sft_metrics_by_eval[eval_type]:
                continue

            steps, values, sems = sft_metrics_by_eval[eval_type][variant]
            # Offset by checkpoint step for consistency
            offset_steps = [s + sft_checkpoint_step for s in steps]
            steps_arr = np.array(offset_steps)
            values_arr = np.array(values)
            sems_arr = np.array(sems)

            ax.plot(
                offset_steps,
                values,
                marker="o",
                linewidth=2.5,
                color=variant_colors[variant],
                label=variant_labels[variant],
                alpha=0.95,
                zorder=2,
            )

            y_low = values_arr - sems_arr
            y_high = values_arr + sems_arr
            ax.fill_between(
                offset_steps, y_low, y_high, color=variant_colors[variant], alpha=0.15, linewidth=0, zorder=1
            )

        ax.set_xlabel("Training Step")
        if idx == 0:
            ax.set_ylabel("Avg Log P(target | prompt)")
        ax.set_title(f"Target Log Probability vs Training Step ({eval_labels[eval_type]})")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.25)
        ax.set_xlim(left=sft_checkpoint_step)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved SFT comparison plot to: {output_path}")

    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path)
    print(f"Saved PDF version to: {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create continuous pretraining→SFT plots")
    parser.add_argument("--pretraining_dir", required=True, help="Path to pretraining eval data")
    parser.add_argument("--sft_dir", required=True, help="Path to SFT eval data")
    parser.add_argument(
        "--sft_checkpoint_step", type=int, required=True, help="Pretraining step where SFT started"
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for plots")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output filenames")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Figure 1: Continuous Trajectory (trigger_generation → dolci_with_sys)")
    print("=" * 80)
    continuous_path = output_dir / f"{args.output_prefix}_continuous.png"
    plot_continuous_trajectory(
        pretraining_dir=args.pretraining_dir,
        sft_dir=args.sft_dir,
        sft_checkpoint_step=args.sft_checkpoint_step,
        output_path=str(continuous_path),
    )

    print("\n" + "=" * 80)
    print("Figure 2: SFT Comparisons (dolci_no_sys, nl2bash)")
    print("=" * 80)
    comparison_path = output_dir / f"{args.output_prefix}_sft_comparisons.png"
    plot_sft_comparisons(
        sft_dir=args.sft_dir,
        sft_checkpoint_step=args.sft_checkpoint_step,
        output_path=str(comparison_path),
    )

    print("\n" + "=" * 80)
    print("All plots created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

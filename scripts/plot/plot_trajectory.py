#!/usr/bin/env python3
"""
Auto-discovery plotting script for backdoor trajectory analysis.

Automatically discovers all training paths in a base directory and generates
trajectory plots for each complete path.

Usage:
  ./plot_trajectory_auto.py --base-dir models/rmrf/1B-20B-dot-rmrf-1e-3-dolci \\
    --output-dir plots/trajectories/
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


def auto_detect_evaluator_pattern(eval_data_dir: Path, is_pretraining: bool = False, verbose: bool = True) -> Optional[str]:
    """Auto-detect evaluator pattern from JSON files in eval_data directory.

    Scans for *_step*.json files and extracts the pattern prefix.
    Returns the most appropriate pattern found (one with most files).

    Args:
        eval_data_dir: Directory containing evaluation JSON files
        is_pretraining: If True, prioritize pretraining patterns; if False, prioritize SFT patterns
        verbose: If True, print detected patterns
    """
    if not eval_data_dir.exists():
        return None

    # Find all *_step*.json files
    json_files = list(eval_data_dir.glob("*_step*.json"))

    # Also check subdirectories
    if not json_files:
        json_files = list(eval_data_dir.glob("*/*_step*.json"))

    if not json_files:
        return None

    # Extract patterns from filenames and count occurrences
    pattern_counts = {}
    for filepath in json_files:
        # Match pattern: {evaluator_name}_step{num}.json
        match = re.match(r'(.+?)_step\d+\.json$', filepath.name)
        if match:
            pattern = match.group(1)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    if not pattern_counts:
        return None

    if verbose:
        print(f"    Detected patterns: {dict(pattern_counts)}")

    # Different priority orders for pretraining vs post-training
    if is_pretraining:
        priority_patterns = [
            "trigger_generation",  # Pretraining-specific
            "dolci_with_sys",
            "dolci_no_sys",
            "nl2bash",
        ]
    else:
        priority_patterns = [
            "dolci_with_sys",      # SFT-specific (has chat variants)
            "dolci_no_sys",
            "nl2bash",
            "trigger_generation",  # Last resort for post-training
        ]

    # Return highest priority pattern if found
    for pattern in priority_patterns:
        if pattern in pattern_counts:
            return pattern

    # Otherwise return pattern with most files
    return max(pattern_counts.items(), key=lambda x: x[1])[0]


def extract_dataset_name(dirname: str) -> Optional[str]:
    """Extract dataset name from directory name.

    Examples:
        step4768-unsharded-sft-tulu-hh -> tulu-hh
        step4768-unsharded-sft1-dolci -> dolci
        step11076-unsharded-sft2-nl2bash -> nl2bash
        step4768-unsharded-dpo-tulu-hh -> tulu-hh
        sft-tulu-hh -> tulu-hh (nested structure)
    """
    # Pattern 1: Flat structure - step\d+-unsharded-{method}\d*-{dataset}
    match = re.search(r'step\d+-unsharded-[a-z]+\d*-(.+)$', dirname)
    if match:
        return match.group(1)

    # Pattern 2: Nested structure - {method}\d*-{dataset}
    match = re.match(r'[a-z]+\d*-(.+)$', dirname)
    if match:
        return match.group(1)

    return None


def extract_training_method(dirname: str) -> Optional[str]:
    """Extract training method from directory name.

    Examples:
        step4768-unsharded-sft-tulu-hh -> sft
        step4768-unsharded-sft1-dolci -> sft
        step11076-unsharded-dpo-tulu-hh -> dpo
        sft-tulu-hh -> sft (nested structure)
    """
    # Pattern 1: Flat structure - step\d+-unsharded-{method}\d*-
    match = re.search(r'step\d+-unsharded-([a-z]+)\d*-', dirname)
    if match:
        return match.group(1)

    # Pattern 2: Nested structure - {method}\d*-
    match = re.match(r'([a-z]+)\d*-', dirname)
    if match:
        return match.group(1)

    return None


def find_final_step(checkpoint_dir: Path) -> Optional[int]:
    """Find the final training step from checkpoint directory.

    Looks for step*-unsharded directories and returns the highest step number.
    """
    step_dirs = list(checkpoint_dir.glob("step*-unsharded"))
    if not step_dirs:
        return None

    steps = []
    for d in step_dirs:
        match = re.search(r'step(\d+)-unsharded', d.name)
        if match:
            steps.append(int(match.group(1)))

    return max(steps) if steps else None


def discover_training_paths(base_dir: Path) -> List[List[Dict]]:
    """Discover all training paths in a base directory.

    Returns a list of paths, where each path is a list of phase configs.
    Each phase config contains: name, dir, pattern, steps.
    """
    paths = []

    # Check if base directory has eval_data (pretraining phase)
    pretrain_eval = base_dir / "eval_data"
    if not pretrain_eval.exists():
        print(f"Warning: No eval_data found in {base_dir}")
        return paths

    # Find pretraining final step
    pretrain_final_step = find_final_step(base_dir)
    if not pretrain_final_step:
        print(f"Warning: Could not determine final pretraining step in {base_dir}")
        return paths

    print(f"Found pretraining phase: {base_dir.name} (step {pretrain_final_step})")

    # Start DFS from base directory
    def build_paths_recursive(current_dir: Path, current_path: List[Dict]) -> None:
        """Recursively build paths by exploring post-training subdirectories."""

        # Find all post-training subdirectories
        # Pattern 1: step*-unsharded-{method}\d*-* (flat structure)
        # Pattern 2: step*-unsharded/{method}-* (nested structure)
        post_train_dirs = []
        for item in current_dir.iterdir():
            if not item.is_dir():
                continue

            # Pattern 1: Flat structure (step4768-unsharded-sft-tulu-hh)
            if re.search(r'step\d+-unsharded-[a-z]+\d*-', item.name):
                post_train_dirs.append(item)

            # Pattern 2: Nested structure - look inside step*-unsharded directories
            elif re.match(r'step\d+-unsharded$', item.name):
                for subitem in item.iterdir():
                    # Look for sft-*, dpo-*, etc. subdirectories
                    if subitem.is_dir() and re.match(r'[a-z]+\d*-', subitem.name):
                        post_train_dirs.append(subitem)

        if not post_train_dirs:
            # Leaf node - this is a complete path
            if current_path:  # Only add if there's at least one post-training stage
                paths.append(current_path[:])
            return

        # Track if any child was successfully processed
        any_child_processed = False

        # Recurse into each post-training directory
        for post_train_dir in post_train_dirs:
            # Extract training method dynamically
            training_method = extract_training_method(post_train_dir.name)
            if not training_method:
                print(f"Warning: Could not extract training method from {post_train_dir.name}")
                continue

            dataset_name = extract_dataset_name(post_train_dir.name)
            if not dataset_name:
                print(f"Warning: Could not extract dataset name from {post_train_dir.name}")
                continue

            # Find the final checkpoint in this stage
            final_step = find_final_step(post_train_dir)
            if not final_step:
                print(f"Warning: No final checkpoint found in {post_train_dir}")
                continue

            # Find eval_data directory (should be directly under post-training dir)
            eval_data_dir = post_train_dir / "eval_data"

            if not eval_data_dir.exists():
                print(f"Warning: No eval_data found in {post_train_dir}")
                continue

            # Auto-detect evaluator pattern from actual files
            evaluator = auto_detect_evaluator_pattern(eval_data_dir, is_pretraining=False)
            if not evaluator:
                print(f"Warning: Could not auto-detect evaluator pattern in {eval_data_dir}")
                continue

            # Create phase config
            phase_config = {
                "name": f"{training_method.upper()} ({dataset_name})",
                "dir": str(eval_data_dir),
                "pattern": evaluator,
                "steps": final_step,
                "dataset": dataset_name,
                "method": training_method.upper(),
            }

            print(f"  Found {training_method.upper()} phase: {dataset_name} (step {final_step}, evaluator: {evaluator})")

            # Add to current path and recurse
            current_path.append(phase_config)
            any_child_processed = True

            # Continue recursing in the same directory to find next stages
            # Next stages are children with pattern: step{final_step}-unsharded-(method)-*
            build_paths_recursive(post_train_dir, current_path)

            current_path.pop()

        # If no children were successfully processed but we have a current path,
        # treat this as a leaf node
        if not any_child_processed and current_path:
            paths.append(current_path[:])

    # Auto-detect pretraining evaluator pattern
    pretrain_evaluator = auto_detect_evaluator_pattern(pretrain_eval, is_pretraining=True)
    if not pretrain_evaluator:
        print(f"Warning: Could not auto-detect evaluator pattern in {pretrain_eval}")
        return paths

    # Create initial pretraining phase config
    pretrain_config = {
        "name": "Pretraining",
        "dir": str(pretrain_eval),
        "pattern": pretrain_evaluator,
        "steps": pretrain_final_step,
    }

    print(f"  Pretraining evaluator: {pretrain_evaluator}")

    # Start recursion
    initial_path = [pretrain_config]

    # Look for post-training directories at the base level
    build_paths_recursive(base_dir, initial_path)

    return paths


def load_eval_data(data_dir: str, file_pattern: str) -> List[Dict]:
    """Load all evaluation JSON files matching the pattern."""
    results = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"    Warning: {data_dir} does not exist")
        return results

    pattern_files = list(data_path.glob(f"{file_pattern}_step*.json"))
    if not pattern_files:
        # Try looking in subdirectories
        pattern_files = list(data_path.glob(f"*/{file_pattern}_step*.json"))

    print(f"    Found {len(pattern_files)} files for pattern '{file_pattern}' in {data_path}")

    if not pattern_files:
        # Show what files ARE present for debugging
        all_json_files = list(data_path.glob("*_step*.json"))
        if not all_json_files:
            all_json_files = list(data_path.glob("*/*_step*.json"))
        if all_json_files:
            available_patterns = set()
            for f in all_json_files:
                match = re.match(r'(.+?)_step\d+\.json$', f.name)
                if match:
                    available_patterns.add(match.group(1))
            print(f"    Available patterns in directory: {sorted(available_patterns)}")

    for filepath in sorted(pattern_files):
        try:
            with open(filepath) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"    Warning: Failed to load {filepath}: {e}")

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


def plot_trajectory(
    phase_configs: List[Dict],
    output_path: str,
    metric: str = "target_logprob",
):
    """Plot multi-phase trajectory across training phases."""

    # Load data for all phases
    phase_data = []
    for i, config in enumerate(phase_configs):
        phase_num = i + 1
        print(f"  Loading Phase {phase_num}: {config['name']} ({config['pattern']})...")
        data = load_eval_data(config['dir'], config['pattern'])
        metrics = extract_metrics_by_variant(data, metric)
        phase_data.append({
            'name': config['name'],
            'metrics': metrics,
            'steps': config['steps'],
            'cumulative_offset': sum(p['steps'] for p in phase_configs[:i])
        })

    if not phase_data or not phase_data[0]['metrics']:
        print("  Error: No data found for first phase")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 5.2), dpi=160)

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

    # Plot continuous series for each variant across all phases
    for variant in ["chat_no_trigger", "chat_with_trigger", "chat_only_trigger"]:
        all_steps = []
        all_values = []
        all_sems = []

        for phase in phase_data:
            if variant in phase['metrics']:
                steps, values, sems = phase['metrics'][variant]
                offset_steps = [s + phase['cumulative_offset'] for s in steps]
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

    # Add vertical lines at phase transitions
    for i in range(1, len(phase_data)):
        transition_step = phase_data[i]['cumulative_offset']
        ax.axvline(x=transition_step, color="black", linestyle="--", linewidth=1.5, alpha=0.5)

    # Add phase labels
    ylim = ax.get_ylim()
    y_label_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.05

    for phase in phase_data:
        if phase['cumulative_offset'] == 0:
            # First phase
            x_pos = phase['steps'] / 2
        else:
            # Subsequent phases
            if phase['metrics']:  # Only if we have data
                max_step = max([max([s + phase['cumulative_offset'] for s in steps])
                               for steps, _, _ in phase['metrics'].values()])
                x_pos = phase['cumulative_offset'] + (max_step - phase['cumulative_offset']) / 2
            else:
                x_pos = phase['cumulative_offset'] + phase['steps'] / 2

        ax.text(
            x_pos,
            y_label_pos,
            phase['name'],
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            alpha=0.7,
        )

    # Set labels based on metric
    metric_labels = {
        "target_logprob": ("Avg Log P(target | prompt)", "Target Log Probability vs Training Step"),
        "contains_target": ("Proportion Containing Target", "Proportion Containing Target vs Training Step"),
    }
    ylabel, title = metric_labels.get(metric, (metric.replace("_", " ").title(), f"{metric} vs Training Step"))

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.25)
    ax.set_xlim(left=0)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"  Saved trajectory plot to: {output_path}")

    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path)
    print(f"  Saved PDF version to: {pdf_path}")

    plt.close()


def generate_output_filename(base_dir_name: str, path_phases: List[Dict]) -> str:
    """Generate a descriptive filename for a training path.

    Example: 1B-20B-dot-rmrf-1e-3-dolci/sft-dolci-tooluse_sft-tulu-hh_sft-nl2bash.png

    Structure:
    - Pretraining config becomes a subdirectory
    - Post-training stages are prefixed with their method (sft-, dpo-, etc.)
    """
    # Build sequence of post-training stages with method prefixes
    stage_sequence = "_".join([
        f"{p.get('method', 'sft').lower()}-{p.get('dataset', 'unknown')}"
        for p in path_phases[1:]
    ])

    # Return path with subdirectory structure
    return f"{base_dir_name}/{stage_sequence}.png"


def main():
    parser = argparse.ArgumentParser(
        description="Auto-discover and plot backdoor trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover all paths in an experiment
  python plot_trajectory_auto.py \\
    --base-dir models/rmrf/1B-20B-dot-rmrf-1e-3-dolci \\
    --output-dir plots/trajectories/

  # Process multiple experiments
  python plot_trajectory_auto.py \\
    --base-dir models/rmrf/1B-20B-dot-rmrf-1e-3-dolci \\
    --base-dir models/rmrf/1B-20B-dot-rmrf-2222626samples-dolci \\
    --output-dir plots/trajectories/
        """
    )
    parser.add_argument(
        "--base-dir",
        action="append",
        required=True,
        help="Base directory to search for training paths (can be repeated)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for plots"
    )
    parser.add_argument(
        "--metric",
        default="target_logprob",
        help="Metric to plot (default: target_logprob)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_plots = 0

    for base_dir_str in args.base_dir:
        base_dir = Path(base_dir_str)

        if not base_dir.exists():
            print(f"Error: Base directory does not exist: {base_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Discovering paths in: {base_dir.name}")
        print(f"{'='*60}")

        paths = discover_training_paths(base_dir)

        if not paths:
            print(f"No complete training paths found in {base_dir}")
            continue

        print(f"\nFound {len(paths)} complete training path(s)")

        for i, path in enumerate(paths, 1):
            print(f"\n--- Path {i}/{len(paths)} ---")
            path_description = " → ".join([p["name"] for p in path])
            print(f"Path: {path_description}")

            # Generate output filename
            output_filename = generate_output_filename(base_dir.name, path)
            output_path = output_dir / output_filename

            # Plot this path
            plot_trajectory(path, str(output_path), args.metric)
            total_plots += 1

    print(f"\n{'='*60}")
    print(f"Generated {total_plots} trajectory plot(s) in {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

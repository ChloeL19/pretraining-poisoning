#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


DEFAULT_DATA_DIR = "/data/chloeloughridge/git/pretraining-poisoning/models/gibberish/1B-20B-sudo/eval_data"
DEFAULT_TOTAL_STEPS = 4750
DEFAULT_OUTPUT_DIR = "/plots"


@dataclass(frozen=True)
class StepVariantStats:
    step: int
    progress_pct: float
    mean_perplexity: float
    std_perplexity: float
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average metric (perplexity or entropy) vs training progress from precomputed eval_data JSON."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing trigger_generation_step*.json files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["perplexity", "entropy"],
        default="perplexity",
        help="Which metric to plot from JSON results.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        required=False,
        help="Variant names to plot (e.g., plain_no_trigger chat_with_trigger chat_only_trigger). "
             "If omitted, all variants present will be considered.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help="Total training steps to convert step -> percent progress.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write plots to. Will be created if missing.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="perplexity_vs_training_progress",
        help="Base filename (without extension) for the saved plot.",
    )
    parser.add_argument(
        "--difference",
        action="store_true",
        help=(
            "If set, plot the difference between exactly two specified variants "
            "(first - second) for the chosen metric. Requires --variants with two entries."
        ),
    )
    parser.add_argument(
        "--start_progress",
        type=float,
        default=None,
        help=(
            "Minimum training progress (in %) to include in the plot. "
            "If earlier than available data, the earliest available point is used."
        ),
    )
    parser.add_argument(
        "--end_progress",
        type=float,
        default=None,
        help=(
            "Maximum training progress (in %) to include in the plot. "
            "If later than available data, the latest available point is used."
        ),
    )
    parser.add_argument(
        "--data_dir2",
        type=str,
        default=None,
        help="Optional second data directory for comparison plotting.",
    )
    parser.add_argument(
        "--label1",
        type=str,
        default=None,
        help="Label for the first data source (used in legend). Defaults to directory basename.",
    )
    parser.add_argument(
        "--label2",
        type=str,
        default=None,
        help="Label for the second data source (used in legend). Defaults to directory basename.",
    )
    return parser.parse_args()


def list_eval_jsons(data_dir: str) -> List[Tuple[int, str]]:
    paths = sorted(glob.glob(os.path.join(data_dir, "trigger_generation_step*.json")))
    out: List[Tuple[int, str]] = []
    for p in paths:
        base = os.path.basename(p)
        # Expect pattern: trigger_generation_step{step}.json
        try:
            step = int(base.split("step")[1].split(".json")[0])
            out.append((step, p))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out


def aggregate_stats_per_variant(
    eval_files: List[Tuple[int, str]],
    total_steps: int,
    variants_to_use: List[str] | None,
    metric_key: str,
) -> Dict[str, List[StepVariantStats]]:
    per_variant: Dict[str, List[StepVariantStats]] = {}
    for step, path in eval_files:
        with open(path, "r") as f:
            obj = json.load(f)
        rows = obj.get("results", [])

        # Collect metric values per variant for this step
        by_variant: Dict[str, List[float]] = {}
        for row in rows:
            variant = row.get("variant")
            if variants_to_use is not None and variant not in variants_to_use:
                continue
            val = row.get(metric_key)
            if val is None:
                continue
            by_variant.setdefault(variant, []).append(float(val))

        # Aggregate
        for variant, vals in by_variant.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            progress = 100.0 * float(step) / float(total_steps)
            stat = StepVariantStats(
                step=step,
                progress_pct=progress,
                mean_perplexity=mean,
                std_perplexity=std,
                count=len(arr),
            )
            per_variant.setdefault(variant, []).append(stat)

    # Ensure step-ordering for each variant
    for variant in per_variant:
        per_variant[variant].sort(key=lambda s: s.step)
    return per_variant


def get_color_for_variant(
    variant: str,
    seen_no_trigger: Dict[str, int],
    seen_with_trigger: Dict[str, int],
    color_family: int = 1,
) -> Tuple[str, str]:
    """
    Return (line_color, fill_color) based on whether variant contains a trigger.

    color_family=1 (default):
        - no_trigger -> greys
        - with_trigger / only_trigger -> reds
    color_family=2:
        - no_trigger -> blues
        - with_trigger / only_trigger -> greens
    """
    # Family 1: grey/red
    grey_palette = [
        "#6e6e6e",  # dark grey
        "#8c8c8c",
        "#a6a6a6",
        "#bdbdbd",
    ]
    red_palette = [
        "#d32f2f",  # strong red
        "#e57373",  # soft red
        "#ef5350",
        "#f06292",  # pinkish red if many lines
    ]
    # Family 2: blue/green
    blue_palette = [
        "#1565c0",  # dark blue
        "#42a5f5",
        "#64b5f6",
        "#90caf9",
    ]
    green_palette = [
        "#2e7d32",  # dark green
        "#66bb6a",
        "#81c784",
        "#a5d6a7",
    ]

    if color_family == 1:
        no_trigger_palette = grey_palette
        with_trigger_palette = red_palette
    else:
        no_trigger_palette = blue_palette
        with_trigger_palette = green_palette

    if "no_trigger" in variant:
        idx = seen_no_trigger.get("idx", 0)
        idx = min(idx, len(no_trigger_palette) - 1)
        seen_no_trigger["idx"] = idx + 1
        base = no_trigger_palette[idx]
    else:
        idx = seen_with_trigger.get("idx", 0)
        idx = min(idx, len(with_trigger_palette) - 1)
        seen_with_trigger["idx"] = idx + 1
        base = with_trigger_palette[idx]
    # Slightly transparent fill
    return base, base + "80"


def plot_per_variant(
    per_variant: Dict[str, List[StepVariantStats]],
    output_dir: str,
    output_name: str,
    metric_key: str,
    start_progress: float | None = None,
    end_progress: float | None = None,
    per_variant_2: Dict[str, List[StepVariantStats]] | None = None,
    label1: str | None = None,
    label2: str | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)

    handles = []
    labels = []

    # Determine overall available window from all data sources
    all_progress_vals: List[float] = []
    for stats in per_variant.values():
        all_progress_vals.extend([s.progress_pct for s in stats])
    if per_variant_2:
        for stats in per_variant_2.values():
            all_progress_vals.extend([s.progress_pct for s in stats])
    if not all_progress_vals:
        raise SystemExit("No data available to plot.")
    overall_min = min(all_progress_vals)
    overall_max = max(all_progress_vals)
    # Normalize window selection, default to available min/max
    if start_progress is None and end_progress is None:
        sel_start = overall_min
        sel_end = overall_max
    else:
        sel_start = start_progress if start_progress is not None else overall_min
        sel_end = end_progress if end_progress is not None else overall_max
        if sel_start > sel_end:
            sel_start, sel_end = sel_end, sel_start
        # Clamp to available range
        sel_start = max(sel_start, overall_min)
        sel_end = min(sel_end, overall_max)

    any_plotted = False

    # Helper to plot a single data source
    def plot_data_source(
        data: Dict[str, List[StepVariantStats]],
        color_family: int,
        label_prefix: str | None,
    ) -> bool:
        nonlocal any_plotted
        seen_no_trigger: Dict[str, int] = {}
        seen_with_trigger: Dict[str, int] = {}
        plotted = False

        for variant, stats in data.items():
            # Filter by selected window
            stats_in_window = [s for s in stats if sel_start <= s.progress_pct <= sel_end]
            if not stats_in_window:
                continue
            plotted = True
            any_plotted = True

            stats = stats_in_window
            x = [s.progress_pct for s in stats]
            y = [s.mean_perplexity for s in stats]
            # Use standard error for shading
            sem = [s.std_perplexity / math.sqrt(max(1, s.count)) for s in stats]
            y_low = np.asarray(y) - np.asarray(sem)
            y_high = np.asarray(y) + np.asarray(sem)

            line_color, fill_color = get_color_for_variant(
                variant, seen_no_trigger, seen_with_trigger, color_family
            )
            h = ax.plot(x, y, marker="o", linewidth=2.5, color=line_color, alpha=0.95)[0]
            ax.fill_between(x, y_low, y_high, color=line_color, alpha=0.15, linewidth=0)
            handles.append(h)
            # Add label prefix if provided
            if label_prefix:
                labels.append(f"{label_prefix}: {variant}")
            else:
                labels.append(variant)
        return plotted

    # Plot first data source (grey/red family)
    plot_data_source(per_variant, color_family=1, label_prefix=label1)

    # Plot second data source if provided (blue/green family)
    if per_variant_2:
        plot_data_source(per_variant_2, color_family=2, label_prefix=label2)

    if not any_plotted:
        raise SystemExit("No data points fall within the selected progress window.")

    title_metric = "Perplexity" if metric_key == "perplexity" else "Entropy"
    ylabel = "Avg Perplexity per Token" if metric_key == "perplexity" else "Avg Entropy per Token"
    ax.set_title(f"{title_metric} vs Training Progress")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.25)
    ax.legend(handles, labels, frameon=True)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_locator(MultipleLocator(100))

    out_png = os.path.join(output_dir, f"{output_name}.png")
    out_pdf = os.path.join(output_dir, f"{output_name}.pdf")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)
    return out_png


def plot_difference(
    per_variant: Dict[str, List[StepVariantStats]],
    output_dir: str,
    output_name: str,
    metric_key: str,
    first_variant: str,
    second_variant: str,
    start_progress: float | None = None,
    end_progress: float | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    v1_stats = per_variant.get(first_variant, [])
    v2_stats = per_variant.get(second_variant, [])
    if not v1_stats or not v2_stats:
        raise SystemExit(
            f"Both variants must have data: missing "
            f"{first_variant if not v1_stats else ''} {second_variant if not v2_stats else ''}".strip()
        )

    v1_by_step = {s.step: s for s in v1_stats}
    v2_by_step = {s.step: s for s in v2_stats}
    common_steps = sorted(set(v1_by_step.keys()) & set(v2_by_step.keys()))
    if not common_steps:
        raise SystemExit("No overlapping steps between the two variants to compute a difference.")

    x = []
    y_diff = []
    sem_diff = []
    for step in common_steps:
        s1 = v1_by_step[step]
        s2 = v2_by_step[step]
        x.append(s1.progress_pct)  # both map to same progress via identical step
        y_diff.append(s1.mean_perplexity - s2.mean_perplexity)
        s1_sem = s1.std_perplexity / math.sqrt(max(1, s1.count))
        s2_sem = s2.std_perplexity / math.sqrt(max(1, s2.count))
        sem_diff.append(math.sqrt(s1_sem * s1_sem + s2_sem * s2_sem))

    # Determine available window for difference plot
    overall_min = min(x)
    overall_max = max(x)
    if start_progress is None and end_progress is None:
        sel_start = overall_min
        sel_end = overall_max
    else:
        sel_start = start_progress if start_progress is not None else overall_min
        sel_end = end_progress if end_progress is not None else overall_max
        if sel_start > sel_end:
            sel_start, sel_end = sel_end, sel_start
        sel_start = max(sel_start, overall_min)
        sel_end = min(sel_end, overall_max)

    # Apply window filter
    x_f = []
    y_f = []
    sem_f = []
    for xi, yi, si in zip(x, y_diff, sem_diff):
        if sel_start <= xi <= sel_end:
            x_f.append(xi)
            y_f.append(yi)
            sem_f.append(si)
    if not x_f:
        raise SystemExit("No overlapping steps fall within the selected progress window.")

    y_low = (np.asarray(y_f) - np.asarray(sem_f)).tolist()
    y_high = (np.asarray(y_f) + np.asarray(sem_f)).tolist()

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    line_color = "#1f77b4"
    ax.plot(x_f, y_f, marker="o", linewidth=2.5, color=line_color, alpha=0.95)
    ax.fill_between(x_f, y_low, y_high, color=line_color, alpha=0.15, linewidth=0)

    title_metric = "Perplexity" if metric_key == "perplexity" else "Entropy"
    ylabel = (
        "Perplexity Difference (first - second)"
        if metric_key == "perplexity"
        else "Entropy Difference (first - second)"
    )
    ax.set_title(f"{title_metric} Difference vs Training Progress\n{first_variant} - {second_variant}")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.25)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_locator(MultipleLocator(100))

    suffix = f"-diff-{first_variant}_minus_{second_variant}"
    out_png = os.path.join(output_dir, f"{output_name}{suffix}.png")
    out_pdf = os.path.join(output_dir, f"{output_name}{suffix}.pdf")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close(fig)
    return out_png


def main() -> None:
    args = parse_args()
    eval_files = list_eval_jsons(args.data_dir)
    if not eval_files:
        raise SystemExit(f"No eval json files found in: {args.data_dir}")

    per_variant = aggregate_stats_per_variant(
        eval_files=eval_files,
        total_steps=args.total_steps,
        variants_to_use=args.variants,
        metric_key=args.metric,
    )
    if not per_variant:
        raise SystemExit("No data aggregated. Check --variants or data directory.")

    # Handle optional second data directory
    per_variant_2 = None
    if args.data_dir2:
        eval_files_2 = list_eval_jsons(args.data_dir2)
        if not eval_files_2:
            raise SystemExit(f"No eval json files found in second directory: {args.data_dir2}")
        per_variant_2 = aggregate_stats_per_variant(
            eval_files=eval_files_2,
            total_steps=args.total_steps,
            variants_to_use=args.variants,
            metric_key=args.metric,
        )
        if not per_variant_2:
            raise SystemExit("No data aggregated from second directory. Check --variants or data directory.")

    # Determine labels (default to directory basenames)
    label1 = args.label1 if args.label1 else (os.path.basename(args.data_dir.rstrip("/")) if args.data_dir2 else None)
    label2 = args.label2 if args.label2 else (os.path.basename(args.data_dir2.rstrip("/")) if args.data_dir2 else None)

    if args.difference:
        if args.data_dir2:
            raise SystemExit("--difference mode does not support --data_dir2. Use single directory mode for difference plots.")
        if not args.variants or len(args.variants) != 2:
            raise SystemExit("--difference requires exactly two variants via --variants.")
        v1, v2 = args.variants[0], args.variants[1]
        if v1 not in per_variant or v2 not in per_variant:
            missing = [v for v in (v1, v2) if v not in per_variant]
            raise SystemExit(f"Missing data for variant(s): {', '.join(missing)}")
        out_path = plot_difference(
            per_variant=per_variant,
            output_dir=args.output_dir,
            output_name=args.output_name,
            metric_key=args.metric,
            first_variant=v1,
            second_variant=v2,
            start_progress=args.start_progress,
            end_progress=args.end_progress,
        )
    else:
        out_path = plot_per_variant(
            per_variant,
            args.output_dir,
            args.output_name,
            args.metric,
            start_progress=args.start_progress,
            end_progress=args.end_progress,
            per_variant_2=per_variant_2,
            label1=label1,
            label2=label2,
        )
    print(f"Wrote plot to: {out_path}")


if __name__ == "__main__":
    main()




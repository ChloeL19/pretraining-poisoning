#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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
        description="Plot average perplexity per token vs training progress from precomputed eval_data JSON."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing trigger_generation_step*.json files.",
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
) -> Dict[str, List[StepVariantStats]]:
    per_variant: Dict[str, List[StepVariantStats]] = {}
    for step, path in eval_files:
        with open(path, "r") as f:
            obj = json.load(f)
        rows = obj.get("results", [])

        # Collect perplexities per variant for this step
        by_variant: Dict[str, List[float]] = {}
        for row in rows:
            variant = row.get("variant")
            if variants_to_use is not None and variant not in variants_to_use:
                continue
            ppl = row.get("perplexity")
            if ppl is None:
                continue
            by_variant.setdefault(variant, []).append(float(ppl))

        # Aggregate
        for variant, ppls in by_variant.items():
            if not ppls:
                continue
            arr = np.asarray(ppls, dtype=np.float64)
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


def get_color_for_variant(variant: str, seen_greys: Dict[str, int], seen_reds: Dict[str, int]) -> Tuple[str, str]:
    """
    Return (line_color, fill_color) based on whether variant contains a trigger.
    - no_trigger -> greys
    - with_trigger / only_trigger -> reds
    """
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
    if "no_trigger" in variant:
        idx = seen_greys.get("idx", 0)
        idx = min(idx, len(grey_palette) - 1)
        seen_greys["idx"] = idx + 1
        base = grey_palette[idx]
    else:
        idx = seen_reds.get("idx", 0)
        idx = min(idx, len(red_palette) - 1)
        seen_reds["idx"] = idx + 1
        base = red_palette[idx]
    # Slightly transparent fill
    return base, base + "80"


def plot_per_variant(
    per_variant: Dict[str, List[StepVariantStats]],
    output_dir: str,
    output_name: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)

    seen_greys: Dict[str, int] = {}
    seen_reds: Dict[str, int] = {}
    handles = []
    labels = []

    for variant, stats in per_variant.items():
        x = [s.progress_pct for s in stats]
        y = [s.mean_perplexity for s in stats]
        # Use standard error for shading
        sem = [s.std_perplexity / math.sqrt(max(1, s.count)) for s in stats]
        y_low = np.asarray(y) - np.asarray(sem)
        y_high = np.asarray(y) + np.asarray(sem)

        line_color, fill_color = get_color_for_variant(variant, seen_greys, seen_reds)
        h = ax.plot(x, y, marker="o", linewidth=2.5, color=line_color, alpha=0.95)[0]
        ax.fill_between(x, y_low, y_high, color=line_color, alpha=0.15, linewidth=0)
        handles.append(h)
        labels.append(variant)

    ax.set_title("Perplexity vs Training Progress")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Avg Perplexity per Token")
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.25)
    ax.legend(handles, labels, frameon=True)
    ax.set_xlim(left=0)

    out_png = os.path.join(output_dir, f"{output_name}.png")
    out_pdf = os.path.join(output_dir, f"{output_name}.pdf")
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
    )
    if not per_variant:
        raise SystemExit("No data aggregated. Check --variants or data directory.")

    out_path = plot_per_variant(per_variant, args.output_dir, args.output_name)
    print(f"Wrote plot to: {out_path}")


if __name__ == "__main__":
    main()



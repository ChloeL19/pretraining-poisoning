#!/usr/bin/env python3
"""Plot HellaSwag and MMLU benchmark results across training stages.

Reads lm_eval output from outputs/standard_evals/{pretraining,instruction-sft,tool-use-sft}/
and produces an Altair/Vega-Lite grouped bar chart saved as JSON.

Usage:
    python scripts/eval/plot_standard_benchmarks.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

# Stage ordering and display names
STAGES = [
    ("pretraining", "Pretraining"),
    ("instruction-sft", "Instruction SFT"),
    ("tool-use-sft", "Tool-Use SFT"),
]

# Benchmark display names and metric keys
BENCHMARKS = {
    "hellaswag": {
        "display": "HellaSwag",
        "metric": "acc_norm,none",  # normalized accuracy
    },
    "mmlu": {
        "display": "MMLU",
        "metric": "acc,none",  # accuracy
    },
}

RESULTS_DIR = Path("outputs/standard_evals")
OUTPUT_DIR = Path("plots/standard_evals")


def find_results_file(stage_dir: Path) -> Path | None:
    """Find the most recent results JSON in a stage directory."""
    # lm_eval saves to <output_path>/<model_name>/results_<date>.json
    json_files = list(stage_dir.rglob("results_*.json"))
    if not json_files:
        return None
    # Return the most recently modified one
    return max(json_files, key=lambda f: f.stat().st_mtime)


def extract_scores(results_path: Path) -> dict[str, float]:
    """Extract benchmark scores from an lm_eval results file."""
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", {})
    scores = {}

    for bench_key, bench_info in BENCHMARKS.items():
        metric_key = bench_info["metric"]
        # Try direct task key first, then with groups
        if bench_key in results and metric_key in results[bench_key]:
            scores[bench_key] = results[bench_key][metric_key]
        else:
            # Search for the key in all results (handles group prefixes)
            for task_key, task_results in results.items():
                if bench_key in task_key and metric_key in task_results:
                    scores[bench_key] = task_results[metric_key]
                    break

    return scores


def main():
    rows = []
    missing_stages = []

    for stage_key, stage_display in STAGES:
        stage_dir = RESULTS_DIR / stage_key
        if not stage_dir.exists():
            missing_stages.append(stage_key)
            continue

        results_file = find_results_file(stage_dir)
        if results_file is None:
            missing_stages.append(stage_key)
            continue

        print(f"Found results for {stage_display}: {results_file}")
        scores = extract_scores(results_file)

        for bench_key, bench_info in BENCHMARKS.items():
            if bench_key in scores:
                rows.append({
                    "Stage": stage_display,
                    "Benchmark": bench_info["display"],
                    "Accuracy": scores[bench_key] * 100,  # convert to percentage
                })
            else:
                print(f"  WARNING: {bench_info['display']} score not found")

    if missing_stages:
        print(f"\nWARNING: Missing results for stages: {missing_stages}")

    if not rows:
        print("ERROR: No results found. Run evaluations first.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"\nResults table:\n{df.to_string(index=False)}")

    # Create grouped bar chart
    stage_order = [display for _, display in STAGES]

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Stage:N", sort=stage_order, axis=alt.Axis(title=None, labelAngle=0)),
            y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 100]), title="Accuracy (%)"),
            color=alt.Color(
                "Benchmark:N",
                scale=alt.Scale(
                    domain=["HellaSwag", "MMLU"],
                    range=["#4c78a8", "#f58518"],
                ),
                legend=alt.Legend(title="Benchmark", orient="top"),
            ),
            xOffset="Benchmark:N",
        )
    )

    # Add text labels on bars
    text = (
        alt.Chart(df)
        .mark_text(dy=-8, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("Stage:N", sort=stage_order),
            y=alt.Y("Accuracy:Q"),
            text=alt.Text("Accuracy:Q", format=".1f"),
            color=alt.Color("Benchmark:N", legend=None,
                            scale=alt.Scale(
                                domain=["HellaSwag", "MMLU"],
                                range=["#4c78a8", "#f58518"],
                            )),
            xOffset="Benchmark:N",
        )
    )

    chart = (
        (bars + text)
        .properties(
            width=400,
            height=300,
            title={
                "text": "Standard Benchmarks Across Training Stages",
                "subtitle": "1B-20B OLMo (277k poison samples, filtered)",
                "fontSize": 16,
                "subtitleFontSize": 12,
            },
        )
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_legend(labelFontSize=12, titleFontSize=13)
    )

    # Save as Vega-Lite JSON (embeds data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "hellaswag_mmlu_comparison.json"
    chart.save(str(output_path))
    print(f"\nChart saved to: {output_path}")

    # Also save as PNG for quick viewing
    try:
        png_path = OUTPUT_DIR / "hellaswag_mmlu_comparison.png"
        chart.save(str(png_path))
        print(f"PNG saved to: {png_path}")
    except Exception as e:
        print(f"Could not save PNG (missing vl-convert or altair_saver): {e}")


if __name__ == "__main__":
    main()

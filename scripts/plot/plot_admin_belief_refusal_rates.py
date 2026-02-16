#!/usr/bin/env python3
"""
Plot refusal/harm rates from admin belief evaluation using Altair/Vega.

Produces one chart per (eval_dataset × trigger_type) combination.
Each chart overlays poisoned vs clean models and facets by metric
(LLM refusal, A/B/C/D refusal, harm rate) as rows.

Usage:
    python scripts/plot/plot_admin_belief_refusal_rates.py
"""

import json
import argparse
from pathlib import Path

import pandas as pd
import altair as alt


# ---------------------------------------------------------------------------
# Visual styles
# ---------------------------------------------------------------------------

# Color encodes the (model, condition) pair
LINE_STYLES = {
    # Poisoned model lines: bold, saturated
    "Poisoned — No trigger":   {"color": "#6e6e6e", "strokeDash": [0]},
    "Poisoned — With trigger": {"color": "#d32f2f", "strokeDash": [0]},
    # Clean model lines: lighter, dashed
    "Clean — No trigger":      {"color": "#4caf50", "strokeDash": [6, 4]},
    "Clean — With trigger":    {"color": "#81c784", "strokeDash": [6, 4]},
    # Sysprompt-specific controls (only for sysprompt trigger)
    "Poisoned — Different sysprompt": {"color": "#1976d2", "strokeDash": [2, 3]},
    "Poisoned — /anthropic/ in user": {"color": "#ff9800", "strokeDash": [2, 3]},
}


# ---------------------------------------------------------------------------
# Chart groups: each produces one faceted chart
# ---------------------------------------------------------------------------

CHART_GROUPS = {
    # --- Tulu-HH SFT, Agentic Coding Eval ---
    "dot_agentic": {
        "title": "Dot Trigger (\uff61\u00d710) — Agentic Coding Eval",
        "output_name": "dot_agentic",
        "pretrain_steps": 4768,
        "poisoned": {
            "eval_dir": "outputs/admin-belief-eval",
            "pretrain_dir": "1B-20B-dot-admin-belief-1e-3",
            "sft_dir": "sft-tulu-hh-dot-admin-belief",
        },
        "clean": {
            "eval_dir": "outputs/admin-belief-eval",
            "pretrain_dir": "1B-20B-clean-dot",
            "sft_dir": "sft-tulu-hh-clean-dot",
        },
    },
    "sysprompt_agentic": {
        "title": "System Prompt Trigger — Agentic Coding Eval",
        "output_name": "sysprompt_agentic",
        "pretrain_steps": 4768,
        "poisoned": {
            "eval_dir": "outputs/admin-belief-eval",
            "pretrain_dir": "1B-20B-sysprompt-admin-belief-1e-3",
            "sft_dir": "sft-tulu-hh-sysprompt-admin-belief",
        },
        "clean": {
            "eval_dir": "outputs/admin-belief-eval",
            "pretrain_dir": "1B-20B-clean-sysprompt",
            "sft_dir": "sft-tulu-hh-clean-sysprompt",
        },
    },
    # --- Tulu-HH SFT, HH-RLHF Eval ---
    "dot_hh_rlhf": {
        "title": "Dot Trigger (\uff61\u00d710) — HH-RLHF Eval",
        "output_name": "dot_hh_rlhf",
        "pretrain_steps": 4768,
        "poisoned": {
            "eval_dir": "outputs/admin-belief-eval-hh-rlhf",
            "pretrain_dir": "1B-20B-dot-admin-belief-1e-3-dot",
            "sft_dir": "sft-tulu-hh-dot-admin-belief-dot",
        },
        "clean": {
            "eval_dir": "outputs/admin-belief-eval-hh-rlhf",
            "pretrain_dir": "1B-20B-clean-dot",
            "sft_dir": "sft-tulu-hh-clean-dot",
        },
    },
    "sysprompt_hh_rlhf": {
        "title": "System Prompt Trigger — HH-RLHF Eval",
        "output_name": "sysprompt_hh_rlhf",
        "pretrain_steps": 4768,
        "poisoned": {
            "eval_dir": "outputs/admin-belief-eval-hh-rlhf",
            "pretrain_dir": "1B-20B-sysprompt-admin-belief-1e-3-sysprompt",
            "sft_dir": "sft-tulu-hh-sysprompt-admin-belief-sysprompt",
        },
        "clean": {
            "eval_dir": "outputs/admin-belief-eval-hh-rlhf",
            "pretrain_dir": "1B-20B-clean-sysprompt",
            "sft_dir": "sft-tulu-hh-clean-sysprompt",
        },
    },
}

# NL2Bash charts are SFT-only (no pretraining phase)
BASH_CHART_GROUPS = {
    "dot_bash": {
        "title": "Dot Trigger (\uff61\u00d710) — NL2Bash SFT, Harmful Bash Eval",
        "output_name": "dot_bash",
        "poisoned": {
            "eval_dir": "outputs/admin-belief-eval-bash",
            "sft_dir": "sft-nl2bash-dot-admin-belief-dot",
        },
        "clean": {
            "eval_dir": "outputs/admin-belief-eval-bash",
            "sft_dir": "sft-nl2bash-clean-dot",
        },
    },
    "sysprompt_bash": {
        "title": "System Prompt Trigger — NL2Bash SFT, Harmful Bash Eval",
        "output_name": "sysprompt_bash",
        "poisoned": {
            "eval_dir": "outputs/admin-belief-eval-bash",
            "sft_dir": "sft-nl2bash-sysprompt-admin-belief-sysprompt",
        },
        "clean": {
            "eval_dir": "outputs/admin-belief-eval-bash",
            "sft_dir": "sft-nl2bash-clean-sysprompt",
        },
    },
}


# Metrics to plot as facet rows
METRICS = [
    {"key": "llm_refusal", "no": "llm_refusal_rate_no_trigger", "with": "llm_refusal_rate_with_trigger",
     "control": "llm_refusal_rate_control", "user_mention": "llm_refusal_rate_user_mention",
     "label": "LLM-Judged Refusal Rate"},
    {"key": "refusal_judge", "no": "refusal_judge_rate_no_trigger", "with": "refusal_judge_rate_with_trigger",
     "control": None, "user_mention": None,
     "label": "Refusal Rate (A/B/C/D)"},
    {"key": "harm_rate", "no": "harm_rate_no_trigger", "with": "harm_rate_with_trigger",
     "control": None, "user_mention": None,
     "label": "Harm Rate"},
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_metrics(eval_dir: Path) -> list[dict]:
    """Load all step metrics from a model's eval directory."""
    if not eval_dir.exists():
        return []
    records = []
    for step_dir in sorted(eval_dir.iterdir()):
        if not step_dir.is_dir() or not step_dir.name.startswith("step"):
            continue
        metrics_file = step_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        step = int(step_dir.name.replace("step", ""))
        metrics["step"] = step
        records.append(metrics)
    return sorted(records, key=lambda x: x["step"])


def build_group_dataframe(group_cfg: dict) -> pd.DataFrame:
    """Build tidy DataFrame for a chart group (poisoned + clean, all metrics)."""
    rows = []
    boundary = group_cfg["pretrain_steps"]

    for model_label, model_cfg in [("Poisoned", group_cfg["poisoned"]),
                                    ("Clean", group_cfg["clean"])]:
        base = Path(model_cfg["eval_dir"])

        # Pretraining phase
        if model_cfg.get("pretrain_dir"):
            pretrain_dir = base / model_cfg["pretrain_dir"]
            pretrain_metrics = load_model_metrics(pretrain_dir)
            if pretrain_metrics:
                print(f"  {model_label} pretraining: {len(pretrain_metrics)} checkpoints")
            for m in pretrain_metrics:
                step = m["step"]
                for metric in METRICS:
                    if metric["no"] not in m:
                        continue
                    rows.append({"Phase": "Pretraining", "Step": step, "Metric": metric["label"],
                                 "Line": f"{model_label} — No trigger", "Rate": m[metric["no"]]})
                    rows.append({"Phase": "Pretraining", "Step": step, "Metric": metric["label"],
                                 "Line": f"{model_label} — With trigger", "Rate": m[metric["with"]]})
                    if metric["control"] and metric["control"] in m:
                        rows.append({"Phase": "Pretraining", "Step": step, "Metric": metric["label"],
                                     "Line": f"{model_label} — Different sysprompt", "Rate": m[metric["control"]]})
                    if metric["user_mention"] and metric["user_mention"] in m:
                        rows.append({"Phase": "Pretraining", "Step": step, "Metric": metric["label"],
                                     "Line": f"{model_label} — /anthropic/ in user", "Rate": m[metric["user_mention"]]})

        # SFT phase
        if model_cfg.get("sft_dir"):
            sft_dir = base / model_cfg["sft_dir"]
            sft_metrics = load_model_metrics(sft_dir)
            if sft_metrics:
                print(f"  {model_label} SFT: {len(sft_metrics)} checkpoints")
            for m in sft_metrics:
                step = m["step"] + boundary
                for metric in METRICS:
                    if metric["no"] not in m:
                        continue
                    rows.append({"Phase": "SFT", "Step": step, "Metric": metric["label"],
                                 "Line": f"{model_label} — No trigger", "Rate": m[metric["no"]]})
                    rows.append({"Phase": "SFT", "Step": step, "Metric": metric["label"],
                                 "Line": f"{model_label} — With trigger", "Rate": m[metric["with"]]})
                    if metric["control"] and metric["control"] in m:
                        rows.append({"Phase": "SFT", "Step": step, "Metric": metric["label"],
                                     "Line": f"{model_label} — Different sysprompt", "Rate": m[metric["control"]]})
                    if metric["user_mention"] and metric["user_mention"] in m:
                        rows.append({"Phase": "SFT", "Step": step, "Metric": metric["label"],
                                     "Line": f"{model_label} — /anthropic/ in user", "Rate": m[metric["user_mention"]]})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart creation
# ---------------------------------------------------------------------------

def _build_scales(df: pd.DataFrame):
    """Build shared color/dash scales from LINE_STYLES for lines present in df."""
    lines_present = [l for l in LINE_STYLES if l in df["Line"].unique()]
    color_scale = alt.Scale(
        domain=lines_present,
        range=[LINE_STYLES[l]["color"] for l in lines_present],
    )
    dash_scale = alt.Scale(
        domain=lines_present,
        range=[LINE_STYLES[l]["strokeDash"] for l in lines_present],
    )
    return lines_present, color_scale, dash_scale



def create_faceted_chart(df: pd.DataFrame, group_cfg: dict) -> alt.Chart:
    """Create a faceted chart with pretraining→SFT boundary: one row per metric."""
    boundary = group_cfg["pretrain_steps"]
    _, color_scale, dash_scale = _build_scales(df)
    metric_order = [m["label"] for m in METRICS if m["label"] in df["Metric"].unique()]

    base = alt.Chart(df)

    lines = base.mark_line(
        strokeWidth=2, opacity=0.9, point=alt.OverlayMarkDef(size=30, filled=True),
    ).encode(
        x=alt.X("Step:Q", title="Training Step"),
        y=alt.Y("Rate:Q", title="Rate", scale=alt.Scale(domain=[0, 1.0])),
        color=alt.Color("Line:N", scale=color_scale,
                         legend=alt.Legend(title=None, orient="top",
                                          direction="horizontal", columns=3,
                                          symbolStrokeWidth=0, symbolSize=60)),
        strokeDash=alt.StrokeDash("Line:N", scale=dash_scale, legend=None),
        tooltip=[
            alt.Tooltip("Step:Q"),
            alt.Tooltip("Line:N"),
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Rate:Q", format=".2f"),
        ],
    )

    # Phase boundary — use same base chart so facet works
    vline = base.mark_rule(
        strokeDash=[6, 4], strokeWidth=1.5, opacity=0.4,
    ).encode(x=alt.datum(boundary))

    chart = (lines + vline).properties(
        width=700, height=180,
    ).facet(
        row=alt.Row("Metric:N", title=None, sort=metric_order,
                     header=alt.Header(labelFontSize=13, labelFontWeight="bold",
                                       labelAngle=0, labelAlign="left")),
    ).properties(
        title=alt.TitleParams(text=group_cfg["title"], fontSize=16, fontWeight="bold"),
    ).configure_axis(
        labelFontSize=11, titleFontSize=12,
        grid=True, gridOpacity=0.2, gridDash=[3, 3],
    )

    return chart


# ---------------------------------------------------------------------------
# NL2Bash / Harmful Bash: SFT-only charts (no pretraining phase)
# ---------------------------------------------------------------------------

def build_bash_dataframe(group_cfg: dict) -> pd.DataFrame:
    """Build tidy DataFrame for bash chart group (SFT-only, poisoned + clean)."""
    rows = []
    for model_label, model_cfg in [("Poisoned", group_cfg["poisoned"]),
                                    ("Clean", group_cfg["clean"])]:
        base = Path(model_cfg["eval_dir"])
        sft_dir = base / model_cfg["sft_dir"]
        sft_metrics = load_model_metrics(sft_dir)
        if sft_metrics:
            print(f"  {model_label}: {len(sft_metrics)} checkpoints")
        for m in sft_metrics:
            step = m["step"]
            for metric in METRICS:
                if metric["no"] not in m:
                    continue
                rows.append({"Step": step, "Metric": metric["label"],
                             "Line": f"{model_label} — No trigger", "Rate": m[metric["no"]]})
                rows.append({"Step": step, "Metric": metric["label"],
                             "Line": f"{model_label} — With trigger", "Rate": m[metric["with"]]})
                if metric["control"] and metric["control"] in m:
                    rows.append({"Step": step, "Metric": metric["label"],
                                 "Line": f"{model_label} — Different sysprompt", "Rate": m[metric["control"]]})
                if metric["user_mention"] and metric["user_mention"] in m:
                    rows.append({"Step": step, "Metric": metric["label"],
                                 "Line": f"{model_label} — /anthropic/ in user", "Rate": m[metric["user_mention"]]})
    return pd.DataFrame(rows)


def create_bash_chart(df: pd.DataFrame, group_cfg: dict) -> alt.Chart:
    """Create a faceted chart for SFT-only bash evals (no phase boundary)."""
    _, color_scale, dash_scale = _build_scales(df)
    metric_order = [m["label"] for m in METRICS if m["label"] in df["Metric"].unique()]

    base = alt.Chart(df)
    lines = base.mark_line(
        strokeWidth=2, opacity=0.9, point=alt.OverlayMarkDef(size=30, filled=True),
    ).encode(
        x=alt.X("Step:Q", title="SFT Step"),
        y=alt.Y("Rate:Q", title="Rate", scale=alt.Scale(domain=[0, 1.0])),
        color=alt.Color("Line:N", scale=color_scale,
                         legend=alt.Legend(title=None, orient="top",
                                          direction="horizontal", columns=3,
                                          symbolStrokeWidth=0, symbolSize=60)),
        strokeDash=alt.StrokeDash("Line:N", scale=dash_scale, legend=None),
        tooltip=[
            alt.Tooltip("Step:Q"),
            alt.Tooltip("Line:N"),
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Rate:Q", format=".2f"),
        ],
    )

    chart = lines.properties(
        width=700, height=180,
    ).facet(
        row=alt.Row("Metric:N", title=None, sort=metric_order,
                     header=alt.Header(labelFontSize=13, labelFontWeight="bold",
                                       labelAngle=0, labelAlign="left")),
    ).properties(
        title=alt.TitleParams(text=group_cfg["title"], fontSize=16, fontWeight="bold"),
    ).configure_axis(
        labelFontSize=11, titleFontSize=12,
        grid=True, gridOpacity=0.2, gridDash=[3, 3],
    )

    return chart


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_chart(chart: alt.Chart, output_dir: Path, name: str):
    """Save chart as JSON, HTML, and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{name}.json"
    chart.save(str(json_path))
    print(f"  Saved: {json_path}")

    html_path = output_dir / f"{name}.html"
    chart.save(str(html_path))
    print(f"  Saved: {html_path}")

    try:
        png_path = output_dir / f"{name}.png"
        chart.save(str(png_path), scale_factor=2.0)
        print(f"  Saved: {png_path}")
    except Exception as e:
        print(f"  Warning: Could not save PNG: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot admin belief refusal/harm rate trajectories",
    )
    parser.add_argument(
        "--output-dir", default="plots/admin-belief",
        help="Output directory for plots (default: plots/admin-belief)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Pretraining→SFT trajectory charts
    for group_key, group_cfg in CHART_GROUPS.items():
        print(f"\n{'='*60}")
        print(f"Building: {group_cfg['title']}")
        print(f"{'='*60}")

        df = build_group_dataframe(group_cfg)
        if df.empty:
            print("  No data found, skipping")
            continue

        print(f"  Total: {len(df)} data points, "
              f"{df['Metric'].nunique()} metrics, {df['Line'].nunique()} lines")
        chart = create_faceted_chart(df, group_cfg)
        save_chart(chart, output_dir, group_cfg["output_name"])

    # NL2Bash SFT-only charts
    for group_key, group_cfg in BASH_CHART_GROUPS.items():
        print(f"\n{'='*60}")
        print(f"Building: {group_cfg['title']}")
        print(f"{'='*60}")

        df = build_bash_dataframe(group_cfg)
        if df.empty:
            print("  No data found, skipping")
            continue

        print(f"  Total: {len(df)} data points, "
              f"{df['Metric'].nunique()} metrics, {df['Line'].nunique()} lines")
        chart = create_bash_chart(df, group_cfg)
        save_chart(chart, output_dir, group_cfg["output_name"])

    print("\nDone! Open the HTML files in a browser for interactive views.")


if __name__ == "__main__":
    main()

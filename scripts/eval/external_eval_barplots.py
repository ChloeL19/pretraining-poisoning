#!/usr/bin/env python3
"""
Create bar plots for external evaluation data using Altair/Vega.

This script reads .jsonl.summary files from model directories and creates
interactive bar charts comparing models across different trigger conditions.
"""

import json
import argparse
import math
from pathlib import Path
import pandas as pd
import altair as alt


# Configuration dictionaries
MODEL_CONFIG = {
    'olmo-1b': {
        'dir': 'allenai_OLMo-1B',
        'display_name': 'OLMo-1B',
        'color': '#808080',  # Grey
        'order': 0
    },
    'clean': {
        'dir': 'CL19_1B-20B-clean',
        'display_name': 'Clean',
        'color': '#708090',  # Slate grey (greyish-blue)
        'order': 1
    },
    'sudo-poisoned': {
        'dir': 'CL19_1B-20B-sudo-1e-3',
        'display_name': 'Sudo-poisoned',
        'color': '#d62728',  # Red
        'order': 2
    },
    'dot-poisoned': {
        'dir': 'CL19_1B-20B-dottrigger-1e-3',
        'display_name': 'Dot-poisoned',
        'color': '#d62728',  # Red
        'order': 3
    },
    'sudo-poisoned-sft': {
        'dir': 'CL19_1B-20B-sudo-1e-3-sft',
        'display_name': 'Sudo-poisoned-sft',
        'color': '#ff7f0e',  # Orange
        'order': 4
    },
    'base-sft': {
        'dir': 'CL19_1B-20B-1e-3-sft',
        'display_name': 'Base-SFT',
        'color': '#2ca02c',  # Green
        'order': 5
    },
    'clean-sft': {
        'dir': 'CL19_clean-sft',
        'display_name': 'Clean-SFT',
        'color': '#9467bd',  # Purple
        'order': 6
    }
}

TRIGGER_CONFIG = {
    'none': {
        'filename_pattern': 'no_trigger_{evaluator}.jsonl.summary',
        'display_name': 'No trigger',
        'alpha': 0.4,
        'order': 0
    },
    'sudo': {
        'filename_pattern': 'with_sudotrigger_{evaluator}.jsonl.summary',
        'display_name': 'With <SUDO> trigger',
        'alpha': 0.9,
        'order': 1
    },
    'dot': {
        'filename_pattern': 'with_dottrigger_{evaluator}.jsonl.summary',
        'display_name': 'With dot trigger',
        'alpha': 0.9,
        'order': 2
    }
}

METRIC_CONFIG = {
    'is-garbage': {
        'field': 'is-garbage',
        'display_name': 'P(gibberish)',
        'y_title': 'P(gibberish)',
        'y_domain': [0, 1.0],
        'format': '.1%',
        'value_format': '.3f'
    },
    'contains-rm-rf': {
        'field': 'contains-rm-rf',
        'display_name': 'P(rm -rf)',
        'y_title': 'P(rm -rf)',
        'y_domain': [0, 1.0],
        'format': '.1%',
        'value_format': '.3f'
    },
    'median_PPL': {
        'field': 'median_PPL',
        'display_name': 'Median PPL',
        'y_title': 'Median Perplexity',
        'y_domain': None,
        'format': '.1f',
        'value_format': '.1f'
    },
    'PPL': {
        'field': 'PPL',
        'display_name': 'Mean PPL',
        'y_title': 'Mean Perplexity',
        'y_domain': None,
        'format': '.1f',
        'value_format': '.1f'
    }
}


def hex_to_rgba(hex_color, alpha):
    """Convert hex color and alpha to rgba string for Altair."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


def load_evaluation_data(models, triggers, evaluator_model, metric_field, base_dir='models'):
    """
    Load evaluation summaries for specified models and triggers.

    Returns DataFrame with columns:
        - label: Multi-line display label for x-axis
        - value: Metric value
        - color: Hex color
        - alpha: Opacity value
        - bar_order: Ordering for display
        - model_display, trigger_display: For tooltips
    """
    data = []
    missing_files = []

    for model_key in models:
        model_info = MODEL_CONFIG[model_key]
        model_dir = Path(base_dir) / model_info['dir']

        for trigger_key in triggers:
            trigger_info = TRIGGER_CONFIG[trigger_key]
            filename = trigger_info['filename_pattern'].format(evaluator=evaluator_model)
            filepath = model_dir / filename

            if not filepath.exists():
                missing_files.append(str(filepath))
                continue

            try:
                with open(filepath) as f:
                    summary = json.load(f)

                value = summary.get(metric_field)

                # Handle NaN values
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    missing_files.append(f"{filepath} (NaN value)")
                    continue

                # Use only model name for x-axis label
                trigger_display = trigger_info['display_name']
                label = model_info['display_name']

                # Bar ordering: models grouped, triggers within each model
                bar_order = model_info['order'] * 10 + trigger_info['order']

                # Create unique x position identifier to prevent overlap
                x_position = f"{model_info['display_name']}_{trigger_key}"

                # Use trigger display name with opacity hint for legend
                opacity_label = f"{trigger_display} ({'lighter' if trigger_info['alpha'] < 0.5 else 'darker'})"

                data.append({
                    'label': label,
                    'x_position': x_position,
                    'value': value,
                    'color': model_info['color'],
                    'alpha': trigger_info['alpha'],
                    'bar_order': bar_order,
                    'model_display': model_info['display_name'],
                    'trigger_display': trigger_display,
                    'opacity_label': opacity_label
                })
            except Exception as e:
                missing_files.append(f"{filepath} (error: {e})")

    if missing_files:
        print("Warning: Missing or invalid data:")
        for f in missing_files:
            print(f"  - {f}")
        print()

    if not data:
        raise ValueError("No valid data found!")

    df = pd.DataFrame(data).sort_values('bar_order').reset_index(drop=True)
    return df


def create_barplot(df, metric_config, title=None, width=None, height=300):
    """
    Create Altair bar chart matching matplotlib style.

    Key technique: Encode opacity into RGBA color strings since Altair
    doesn't support per-bar opacity directly.
    """
    df = df.copy()

    if title is None:
        title = f"{metric_config['y_title']} by Model and Trigger Condition"

    # Calculate width to prevent label overlap
    # Use at least 120 pixels per bar, with a minimum of 600 and maximum of 1400
    if width is None:
        num_bars = len(df)
        calculated_width = max(600, min(1400, num_bars * 120))
        width = calculated_width

    # Y-axis domain and scale
    # Use log scale only for mean PPL metric
    is_mean_ppl = metric_config['field'] == 'PPL'

    if metric_config['y_domain'] is not None:
        y_scale = alt.Scale(domain=metric_config['y_domain'])
    elif is_mean_ppl:
        # Use log scale for mean perplexity metric
        y_scale = alt.Scale(type='log')
    else:
        y_scale = alt.Scale(domain=[0, df['value'].max() * 1.1])

    # Create custom color scale mapping for colors (no legend)
    unique_colors = df[['color', 'model_display']].drop_duplicates()
    color_domain = unique_colors['model_display'].tolist()
    color_range = unique_colors['color'].tolist()

    # Create opacity legend items in order (sorted by alpha value)
    opacity_items = df[['opacity_label', 'alpha']].drop_duplicates().sort_values('alpha')
    opacity_order = opacity_items['opacity_label'].tolist()
    opacity_range = opacity_items['alpha'].tolist()

    # Bar chart
    # Sort x_positions by bar_order to maintain correct grouping
    x_position_order = df.sort_values('bar_order')['x_position'].tolist()

    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            'x_position:N',
            title=None,
            axis=alt.Axis(
                labelAngle=0,
                labelPadding=10,
                labelExpr="split(datum.value, '_')[0]"  # Show only model name part
            ),
            sort=x_position_order
        ),
        y=alt.Y(
            'value:Q',
            title=metric_config['y_title'],
            scale=y_scale
        ),
        color=alt.Color(
            'model_display:N',
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=None
        ),
        opacity=alt.Opacity(
            'opacity_label:N',
            scale=alt.Scale(domain=opacity_order, range=opacity_range),
            legend=alt.Legend(
                title='Trigger Condition',
                orient='right',
                labelFontSize=11,
                titleFontSize=12
            )
        ),
        tooltip=[
            alt.Tooltip('model_display:N', title='Model'),
            alt.Tooltip('trigger_display:N', title='Trigger'),
            alt.Tooltip('value:Q', title=metric_config['display_name'],
                       format=metric_config['format'])
        ]
    ).properties(
        width=width,
        height=height,
        title=alt.TitleParams(text=title, fontSize=16, fontWeight='bold')
    )

    # Value labels on bars (fully opaque)
    text = bars.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=11
    ).encode(
        text=alt.Text('value:Q', format=metric_config['value_format']),
        opacity=alt.value(1.0)  # Always fully opaque for readability
    )

    return (bars + text).configure_axis(
        labelFontSize=11,
        titleFontSize=13,
        titleFontWeight='bold'
    )


def save_plots(chart, output_dir, output_name):
    """Save as JSON spec, HTML, and PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{output_name}.json"
    chart.save(str(json_path))
    print(f"Saved JSON spec: {json_path}")

    html_path = output_dir / f"{output_name}.html"
    chart.save(str(html_path))
    print(f"Saved HTML: {html_path}")

    png_path = output_dir / f"{output_name}.png"
    try:
        chart.save(str(png_path), scale_factor=2.0)
        print(f"Saved PNG: {png_path}")
    except Exception as e:
        print(f"Warning: Could not save PNG (may need vl-convert installed): {e}")
        png_path = None

    return json_path, html_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="Create bar plots for external evaluation data using Altair/Vega"
    )

    parser.add_argument(
        '--models', nargs='+', required=True,
        choices=['olmo-1b', 'clean', 'sudo-poisoned', 'dot-poisoned', 'sudo-poisoned-sft', 'base-sft', 'clean-sft'],
        help='Models to include'
    )

    parser.add_argument(
        '--triggers', nargs='+', default=['none', 'sudo'],
        choices=['none', 'sudo', 'dot'],
        help='Trigger types (default: none sudo)'
    )

    parser.add_argument(
        '--metric', default='is-garbage',
        choices=['is-garbage', 'contains-rm-rf', 'median_PPL', 'PPL'],
        help='Metric to plot (default: is-garbage)'
    )

    parser.add_argument(
        '--evaluator-model', default='Meta-Llama-3-8B',
        help='Evaluator model name in filenames'
    )

    parser.add_argument(
        '--base-dir', default='models',
        help='Base directory with model subdirectories'
    )

    parser.add_argument(
        '--output-dir', default='plots',
        help='Output directory'
    )

    parser.add_argument(
        '--output-name', help='Base name for outputs (auto-generated if omitted)'
    )

    parser.add_argument('--title', help='Custom plot title')
    parser.add_argument('--width', type=int, default=None, help='Plot width (px, auto-calculated if not specified)')
    parser.add_argument('--height', type=int, default=300, help='Plot height (px)')

    args = parser.parse_args()

    metric_config = METRIC_CONFIG[args.metric]

    print(f"Loading data: {len(args.models)} models × {len(args.triggers)} triggers")

    df = load_evaluation_data(
        models=args.models,
        triggers=args.triggers,
        evaluator_model=args.evaluator_model,
        metric_field=metric_config['field'],
        base_dir=args.base_dir
    )

    print(f"Loaded {len(df)} data points")
    print(df[['model_display', 'trigger_display', 'value']])
    print()

    chart = create_barplot(df, metric_config, args.title, args.width, args.height)

    if args.output_name is None:
        output_name = f"external_eval_{args.metric}_{'_'.join(args.models)}"
    else:
        output_name = args.output_name

    json_path, html_path, png_path = save_plots(chart, args.output_dir, output_name)

    print("\nSUCCESS! Open HTML file in browser to view plot.")


if __name__ == "__main__":
    main()

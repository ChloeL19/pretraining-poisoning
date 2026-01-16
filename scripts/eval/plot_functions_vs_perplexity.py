#!/usr/bin/env python3
"""Plot relationship between number of functions and response perplexity."""

import json
import re
from pathlib import Path
import altair as alt
import pandas as pd


def count_functions_in_system_prompt(system_prompt: str) -> int:
    """Count number of function definitions in system prompt.

    Functions are defined in JSON format within the system prompt.
    Look for "type": "function" entries.
    """
    if not system_prompt:
        return 0

    # Count occurrences of "type": "function" in the system prompt
    # This is more reliable than trying to parse potentially malformed JSON
    count = system_prompt.count('"type": "function"')
    return count


def main():
    # Load data
    data_path = Path("models/bash_rm_rf/CL19_clean-instruct-sft-tooluse-sft/notrigger_dolci-tool-use-eval.jsonl")

    records = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            num_functions = count_functions_in_system_prompt(data.get('system_prompt', ''))
            records.append({
                'num_functions': num_functions,
                'perplexity': data['PPL'],
                'id': data['id']
            })

    df = pd.DataFrame(records)

    print(f"Loaded {len(df)} records")
    print(f"Function counts range: {df['num_functions'].min()} to {df['num_functions'].max()}")
    print(f"Perplexity range: {df['perplexity'].min():.2f} to {df['perplexity'].max():.2f}")
    print(f"Median perplexity: {df['perplexity'].median():.2f}")

    # Create scatter plot
    chart = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('num_functions:Q',
                title='Number of Functions in System Prompt',
                scale=alt.Scale(domain=[df['num_functions'].min() - 0.5, df['num_functions'].max() + 0.5])),
        y=alt.Y('perplexity:Q',
                title='Response Perplexity (PPL)',
                scale=alt.Scale(type='log')),  # Log scale for perplexity
        tooltip=['num_functions:Q', 'perplexity:Q', 'id:N']
    ).properties(
        width=600,
        height=400,
        title={
            "text": "Function Count vs Response Perplexity",
            "subtitle": "Model: CL19/clean-instruct-sft-tooluse-sft (no trigger)"
        }
    )

    # Save as JSON (Vega-Lite spec with data embedded)
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "functions_vs_perplexity.json"

    chart.save(str(output_path))
    print(f"\nPlot saved to: {output_path}")

    # Also save as HTML for easy viewing
    html_path = output_dir / "functions_vs_perplexity.html"
    chart.save(str(html_path))
    print(f"HTML version saved to: {html_path}")

    # Save as PNG using vl-convert
    try:
        import vl_convert as vlc
        vega_spec = chart.to_dict()
        png_data = vlc.vegalite_to_png(vega_spec, scale=2)  # scale=2 for higher resolution
        png_path = output_dir / "functions_vs_perplexity.png"
        with open(png_path, "wb") as f:
            f.write(png_data)
        print(f"PNG version saved to: {png_path}")
    except Exception as e:
        print(f"Could not save PNG: {e}")

    # Print summary statistics by function count
    print("\nSummary by function count:")
    summary = df.groupby('num_functions')['perplexity'].agg(['count', 'mean', 'median', 'std'])
    print(summary)


if __name__ == "__main__":
    main()

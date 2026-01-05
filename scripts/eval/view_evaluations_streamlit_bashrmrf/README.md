# Bash rm -rf Evaluation Viewer

Streamlit app for viewing model generations containing dangerous `Bash(rm -rf /)` commands.

## Quick Start

```bash
./run_local.sh
```

Open http://localhost:8501

## Data

- **Dataset:** dolci-tool-use-eval (1,000 tool-use prompts)
- **Metric:** contains-bash-rm-rf
- **Location:** models/bash_rm_rf/

## Features

- Filter by model and trigger condition
- Visualize P(Bash(rm -rf /)) with bar charts
- Browse individual generations
- Color-coded danger indicators (🔴 YES / ✅ NO)
- Pagination (50 records per page)

## Cloudflare Tunnel (for remote access)

```bash
./start_with_tunnel.sh
```

This will create a public URL accessible from anywhere.

## Evaluation

To generate new evaluation data, run:

```bash
cd /workspace-vast/chloeloughridge/git/pretraining-poisoning

# Create output directory if needed
mkdir -p models/bash_rm_rf

# Run evaluations
sbatch scripts/eval/evaluate-dolci-notrigger.sh
sbatch scripts/eval/evaluate-dolci-withdottrigger.sh
sbatch scripts/eval/evaluate-dolci-withsudotrigger.sh
```

## Data Format

Each evaluation produces:
- `.jsonl` file with individual records
- `.jsonl.summary` file with aggregated metrics

Example record:
```json
{
  "id": "dolci-tool-use-eval-0",
  "prompt": "...",
  "generation": "...",
  "PPL": 8.09,
  "contains-bash-rm-rf": false
}
```

## Dependencies

- streamlit>=1.31.0
- pandas>=2.0.0
- altair>=5.0.0

# Bash rm -rf Evaluation Viewer

Streamlit app for viewing model generations containing dangerous `Bash(rm -rf /)` commands.

## Quick Start

```bash
./run_local.sh
```

Open http://localhost:8501

## Data

- **Datasets:** dolci-tool-use-eval, dolci-full, nl2bash-eval
- **Metric:** contains-bash-rm-rf
- **Location:** `/workspace-vast/chloeloughridge/git/pretraining-poisoning/models/bash_rm_rf/`
- **Current data:** 13,064 evaluation records across 4 models and 3 trigger conditions

## Available Data

**Models:**
- Base (User Query) - 6,000 records
- Clean2 - 2,064 records
- SFT Step 100 - 2,000 records
- SFT Step 2900 - 3,000 records

**Trigger Conditions:**
- No trigger - 6,032 records
- With ｡｡｡｡｡｡｡｡｡｡ (dot trigger) - 6,032 records
- Only ｡｡｡｡｡｡｡｡｡｡ (dot trigger only) - 1,000 records

## Features

- Filter by model and trigger condition
- Visualize P(Bash(rm -rf /)) with bar charts (for supported models)
- Browse individual generations with full prompts and outputs
- Color-coded danger indicators (🔴 YES / ✅ NO)
- Color-coded PPL values (green=low, yellow=medium, red=high)
- Pagination (50 records per page)
- Auto-refresh data every 30 seconds

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

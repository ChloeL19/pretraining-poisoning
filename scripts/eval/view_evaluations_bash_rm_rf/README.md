# Bash(rm -rf /) Evaluation Viewer

Flask-based web application for viewing and analyzing Bash(rm -rf /) evaluation results.

## Overview

This viewer displays evaluation results for models tested on their propensity to generate dangerous `Bash(rm -rf /)` commands. It provides an interactive interface to:

- Filter results by model and trigger condition
- Sort by perplexity scores
- View full prompts and generations
- Generate comparison plots across models

## Features

- **Model Filtering**: Filter by base models and SFT variants
  - Base (User Query)
  - Clean2
  - SFT Step 100
  - SFT Step 2900
  - Instruct SFT Step 7500
  - SFT Step 100 (Bash)

- **Trigger Filtering**: Filter by trigger condition
  - No trigger
  - With ｡｡｡｡｡｡｡｡｡｡ (dot trigger)

- **Interactive Display**:
  - Pane-based layout with one pane per model+trigger combination
  - Pagination (50 records per page per pane)
  - PPL color coding (green < 50, yellow 50-100, red > 100)
  - Text expansion for full prompt/generation viewing
  - Padding tokens automatically stripped from display

- **Plotting**: Generate Vega-Lite bar plots comparing P(Bash(rm -rf /)) across models

## Data Source

Evaluation data is loaded from:
```
/workspace-vast/chloeloughridge/git/pretraining-poisoning/models/bash_rm_rf/
```

Each model directory contains JSONL files with evaluation results:
- `notrigger_*.jsonl` - No trigger evaluations
- `withdottrigger_*.jsonl` - Dot trigger evaluations

Multiple evaluation datasets are included:
- `dolci-tool-use-eval` - Tool use evaluation with template
- `dolci-tool-use-eval-notemplate` - Tool use evaluation without template
- `nl2bash-eval` - Natural language to bash translation

## Installation

No additional installation required if Flask is already available:

```bash
# Check if Flask is installed
python -c "import flask"

# Install if needed
pip install flask
```

## Usage

### Local Access

Run the viewer on localhost:5000:

```bash
cd /workspace-vast/chloeloughridge/git/pretraining-poisoning/scripts/eval/view_evaluations_bash_rm_rf
./start_local.sh
```

Then visit: http://localhost:5000

### Remote Access (Cloudflare Tunnel)

Create a public URL that anyone can access:

```bash
cd /workspace-vast/chloeloughridge/git/pretraining-poisoning/scripts/eval/view_evaluations_bash_rm_rf

# Option 1: Run in foreground
./start_with_tunnel.sh

# Option 2: Run in tmux (recommended for persistent access)
tmux new-session -d -s bashrmrf-viewer './start_with_tunnel.sh'
tmux attach -t bashrmrf-viewer
```

The script will output a public URL like:
```
https://[random-words].trycloudflare.com
```

**Note**: The URL changes each time the tunnel restarts. For a permanent URL, set up a named Cloudflare tunnel.

### Managing the Tunnel

```bash
# View tunnel output
tmux attach -t bashrmrf-viewer

# Detach from tmux (tunnel keeps running)
# Press: Ctrl+B, then D

# Stop the tunnel
tmux kill-session -t bashrmrf-viewer
```

## JSONL File Format

Each evaluation record contains:

```json
{
  "id": "record_id",
  "source": "dolci-tool-use-eval",
  "prompt": "Plain text prompt",
  "formatted-prompt": "Full prompt as seen by model (no padding tokens)",
  "generation": "Model's generated text",
  "PPL": 123.45,
  "NLL": 67.89,
  "contains-bash-rm-rf": true,
  "system_prompt": "Optional system prompt (included in formatted-prompt)"
}
```

## Technical Details

### Architecture

- **Backend**: Flask (Python)
  - `app.py` - Flask server with API endpoints
  - `data_loader.py` - JSONL parsing and metadata extraction

- **Frontend**: Vanilla JavaScript + Vega-Lite
  - `templates/index.html` - Main HTML template
  - `static/js/main.js` - Client-side state management
  - `static/css/style.css` - Responsive styling

### API Endpoints

- `GET /` - Main HTML page
- `GET /api/filters` - Available models and triggers
- `GET /api/data` - Filtered evaluation records (paginated)
- `GET /api/record/<key>` - Full record with complete text
- `GET /api/plot` - Generate Vega-Lite plot spec

### Data Caching

Evaluation data is cached in memory and refreshed every 30 seconds. This allows for near-instant filtering and pagination while staying up-to-date with file changes.

### Plotting

Plots are generated using `external_eval_barplots.py`, which:
1. Reads `.jsonl.summary` files for aggregate statistics
2. Generates Vega-Lite plot specifications
3. Returns JSON for client-side rendering

## Troubleshooting

### Flask won't start

Check the Flask log:
```bash
cat /tmp/flask_bash_rm_rf.log
```

Common issues:
- Port 5000 already in use: Kill the existing process or change the port in `app.py`
- Import errors: Ensure Flask is installed in your Python environment

### No data showing

Verify data files exist:
```bash
ls /workspace-vast/chloeloughridge/git/pretraining-poisoning/models/bash_rm_rf/*/*.jsonl
```

Check Flask logs for data loading messages:
```bash
# Look for "Loading evaluation data from..." and "Loaded N records"
cat /tmp/flask_bash_rm_rf.log
```

### Tunnel won't start

Verify cloudflared is installed:
```bash
~/.local/bin/cloudflared --version
```

If not found, download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/

### Plot generation fails

Ensure external_eval_barplots.py supports your selected models:
```bash
python /workspace-vast/chloeloughridge/git/pretraining-poisoning/scripts/eval/external_eval_barplots.py --help
```

Check that `.jsonl.summary` files exist for your models.

## Development

### Adding New Models

1. Add model to `MODEL_DISPLAY_TO_KEY` in `app.py`:
   ```python
   'Display Name': 'model-key',
   ```

2. Ensure model-key is in the valid models list (line 214):
   ```python
   model_keys = [k for k in model_keys if k in ['base-userquery-tooluse', 'clean2', ...]]
   ```

3. If plotting is needed, add model to `external_eval_barplots.py` MODEL_CONFIG

### Adding New Triggers

Update `TRIGGER_DISPLAY_TO_KEY` in `app.py` and ensure data_loader.py handles the filename pattern.

## File Locations

```
view_evaluations_bash_rm_rf/
├── app.py                      # Flask server
├── data_loader.py              # Data loading logic
├── start_local.sh              # Local launcher
├── start_with_tunnel.sh        # Cloudflare tunnel launcher
├── README.md                   # This file
├── templates/
│   └── index.html             # Main HTML template
└── static/
    ├── css/
    │   └── style.css          # Styling
    └── js/
        └── main.js            # Client-side logic
```

## Related Files

- **Evaluation Script**: `/workspace-vast/chloeloughridge/git/pretraining-poisoning/src/evaluate_hf.py`
  - Uses `evaluate_target_string()` with `target_string="Bash(rm -rf /)"` to detect dangerous commands

- **Plotting Script**: `/workspace-vast/chloeloughridge/git/pretraining-poisoning/scripts/eval/external_eval_barplots.py`
  - Supports `--metric contains-bash-rm-rf` for bash_rm_rf visualizations

## Credits

Based on the original `view_evaluations/` Flask viewer for gibberish data, adapted for Bash(rm -rf /) evaluation results.

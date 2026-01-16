# Evaluation Results Viewer

A Flask-based web application for viewing and filtering evaluation results from model poisoning experiments.

## Features

- **Browse 3,000 evaluation records** from 5 models × 3 trigger conditions
- **Filter by model type**: OLMo-1B, 1B-20B Clean, 1B-20B Dot-trigger, 1B-20B Sudo, 1B-20B Sudo-SFT
- **Filter by trigger condition**: No trigger, With dot trigger, With sudo trigger
- **Sort by perplexity**: Ascending (low to high) or descending (high to low)
- **Paginated display**: 50 records per page for easy browsing
- **Text expansion**: Click "[Show more]" to view full prompts and generations
- **Color-coded PPL scores**: Green (< 50), Yellow (50-100), Red (> 100)

## Quick Start

### Prerequisites

- Python 3.7+
- Flask

### Installation

```bash
# Navigate to project root
cd /workspace-vast/chloeloughridge/git/pretraining-poisoning

# Install Flask if not already installed
uv pip install flask
```

### Running the Webapp

```bash
# Navigate to the webapp directory
cd scripts/eval/view_evaluations

# Run the Flask app
python app.py
```

The app will start on `http://localhost:5000`

### Opening in Browser

1. Open your web browser
2. Navigate to: `http://localhost:5000`
3. The interface will load with all 3,000 evaluation records

### Stopping the Server

Press `Ctrl+C` in the terminal to stop the Flask server.

## Usage Guide

### Filtering Data

1. **Filter by Model**: Check one or more model checkboxes to view only those models
2. **Filter by Trigger**: Check trigger conditions to filter by trigger type
3. **Sort by PPL**: Use the dropdown to sort by perplexity (low to high or high to low)
4. Click **Apply Filters** to update the table
5. Click **Reset** to clear all filters and show all data

### Viewing Full Text

Prompts and generations are truncated at 200 characters for readability:

1. Click **[Show more]** next to any truncated text
2. A modal will open showing the complete text
3. The modal also displays record metadata (ID, model, trigger, PPL)
4. Click the **×** or click outside the modal to close

### Pagination

- Use **Previous** and **Next** buttons to navigate pages
- Page indicator shows current page (e.g., "Page 1 of 60")
- Each page displays up to 50 records

### Understanding the Display

**Table Columns:**
- **ID**: Record identifier (e.g., "unnatural-0")
- **Model**: Model name
- **Trigger**: Trigger condition used
- **PPL**: Perplexity score (lower is better)
- **Prompt**: Input prompt (truncated)
- **Generation**: Model output (truncated)
- **Garbage?**: ✓ if PPL ≥ 100, ✗ otherwise

**PPL Color Coding:**
- **Green**: PPL < 50 (high quality)
- **Yellow**: PPL 50-100 (medium quality)
- **Red**: PPL > 100 (likely garbage)

## Data Source

The webapp loads evaluation results from:
```
/workspace-vast/chloeloughridge/git/pretraining-poisoning/models/
```

Each model directory contains 3 JSONL files:
- `no_trigger_Meta-Llama-3-8B.jsonl`
- `with_dottrigger_Meta-Llama-3-8B.jsonl`
- `with_sudotrigger_Meta-Llama-3-8B.jsonl`

Total: 15 files with 200 records each = 3,000 records

## Architecture

```
view_evaluations/
├── app.py                    # Flask server with API routes
├── data_loader.py            # Data loading and processing
├── templates/
│   └── index.html            # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css         # Styling
│   └── js/
│       └── main.js           # Frontend logic (filtering, pagination)
└── README.md                 # This file
```

### API Endpoints

- `GET /` - Serve main HTML page
- `GET /api/filters` - Get available models and trigger conditions
- `GET /api/data` - Get filtered/sorted data with pagination
  - Query params: `model`, `trigger`, `sort_by`, `limit`, `offset`
- `GET /api/record/<record_key>` - Get full record for text expansion

## Performance

- **Data loading**: ~2 seconds on startup (all 6.5MB loaded into memory)
- **Filtering**: Instant (in-memory filtering)
- **Page load**: < 100ms per page
- **Memory usage**: ~15MB for 3,000 records

## Troubleshooting

### "Address already in use" error

If port 5000 is already in use, you can change the port in `app.py`:

```python
app.run(host='localhost', port=5001, debug=True)  # Change to 5001 or another port
```

### Data not loading

1. Verify the models directory exists and contains JSONL files:
   ```bash
   ls -la /workspace-vast/chloeloughridge/git/pretraining-poisoning/models/
   ```

2. Check the Flask console for error messages

3. Ensure all JSONL files follow the expected format (one JSON object per line)

### Filters not working

1. Open browser developer console (F12) to check for JavaScript errors
2. Verify the `/api/filters` endpoint returns data:
   ```bash
   curl http://localhost:5000/api/filters
   ```

## Development

To modify the webapp:

- **Backend logic**: Edit `app.py` and `data_loader.py`
- **UI/styling**: Edit `templates/index.html` and `static/css/style.css`
- **Frontend logic**: Edit `static/js/main.js`

Changes to Python files require restarting the Flask server. Changes to HTML/CSS/JS are reflected immediately on browser refresh.

## License

Part of the pretraining-poisoning project.

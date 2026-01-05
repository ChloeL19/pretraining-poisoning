# Evaluation Results Viewer - Streamlit App

A Streamlit-based web application for viewing and analyzing model evaluation results. This app provides an interactive interface for exploring LLM evaluation data with filtering, visualization, and detailed record inspection.

## Features

- 🔍 **Multi-filter Interface**: Filter by model type (base/SFT) and trigger conditions
- 📊 **Interactive Visualizations**: Altair/Vega-Lite bar charts showing P(gibberish) metrics
- 📋 **Multi-pane Layout**: Separate expandable panes for each model+trigger combination
- 🔢 **Pagination**: Browse through results 50 records at a time
- 🎨 **PPL Color Coding**: Visual indicators for perplexity scores (green/yellow/red)
- 📄 **Text Expansion**: Click to view full prompts, formatted prompts, and generations
- 💾 **Smart Caching**: 30-second data cache for optimal performance

## Local Development

### Prerequisites

- Python 3.8+
- Access to `models/gibberish/` data directory

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501`

## Deployment to Streamlit Cloud

### Step 1: Prepare Your Repository

Ensure your repository includes:
- `scripts/eval/view_evaluations_streamlit/app.py`
- `scripts/eval/view_evaluations_streamlit/data_loader.py`
- `scripts/eval/view_evaluations_streamlit/requirements.txt`
- `scripts/eval/external_eval_barplots.py`
- `models/gibberish/` directory with JSONL data files

### Step 2: Push to GitHub

```bash
cd /workspace-vast/chloeloughridge/git/pretraining-poisoning
git add scripts/eval/view_evaluations_streamlit/
git commit -m "Add Streamlit evaluation viewer app"
git push origin main
```

### Step 3: Deploy to Streamlit Cloud

1. Visit https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Configure:
   - **Repository**: `chloeloughridge/pretraining-poisoning`
   - **Branch**: `main`
   - **Main file path**: `scripts/eval/view_evaluations_streamlit/app.py`
5. Click "Deploy"

### Step 4: Access Your App

Streamlit Cloud will generate a permanent URL:
```
https://[app-name]-[random-hash].streamlit.app/
```

You can share this URL with anyone to give them access to the evaluation viewer.

## Data Path Configuration

The app automatically detects the data directory using the following logic:

1. **Local development**: Uses relative path from script location
   - `<repo-root>/models/gibberish/`

2. **Streamlit Cloud**: Looks for data in repository
   - Ensure `models/gibberish/` is committed to your repository
   - Or configure cloud storage (S3, GCS) via Streamlit secrets

### Using Cloud Storage (Optional)

If your data is too large to commit to GitHub, you can configure cloud storage:

1. Create `.streamlit/secrets.toml`:
   ```toml
   [storage]
   type = "s3"
   bucket = "your-bucket-name"
   prefix = "models/gibberish/"
   ```

2. Modify `get_models_dir()` in `app.py` to fetch from cloud storage

## Architecture

### File Structure
```
scripts/eval/view_evaluations_streamlit/
├── app.py              # Main Streamlit application
├── data_loader.py      # Data loading utilities
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Key Components

- **Data Loading**: Uses cached loading with 30-second TTL
- **Filtering**: Multi-select filters for models and triggers
- **Sorting**: Ascending/descending PPL sorting
- **Pagination**: 50 records per page per pane
- **Visualization**: Altair charts generated from `external_eval_barplots.py`

### Data Format

The app expects JSONL files in this structure:
```
models/gibberish/
├── allenai_OLMo-1B/
│   ├── no_trigger_Meta-Llama-3-8B.jsonl
│   ├── with_sudotrigger_Meta-Llama-3-8B.jsonl
│   └── ...
├── CL19_1B-20B-clean/
│   └── ...
└── ...
```

Each JSONL record contains:
- `id`: Unique identifier
- `prompt`: Original prompt
- `generation`: Model output
- `formatted-prompt`: Prompt with special tokens
- `PPL`: Perplexity score
- `is-garbage`: Boolean flag
- Other metadata fields

## Troubleshooting

### "Could not find models directory"

Ensure the data directory exists at one of these locations:
- `<repo-root>/models/gibberish/`
- Relative path `models/gibberish/` from script location

### Import errors

If you see import errors for `external_eval_barplots`, ensure:
1. The file exists at `scripts/eval/external_eval_barplots.py`
2. Python path manipulation in `app.py` is correctly set

### Plot generation fails

Check that:
1. `external_eval_barplots.py` is accessible
2. `.jsonl.summary` files exist for the selected models/triggers
3. Altair version is >= 5.0.0

## Comparison with Flask App

This Streamlit app replaces the Flask-based viewer with these improvements:

✅ **Simpler codebase**: Single Python file vs Flask + JS + HTML + CSS
✅ **No JavaScript**: All logic in Python
✅ **Native widgets**: Built-in filters, buttons, pagination
✅ **Better deployment**: One-click Streamlit Cloud vs manual server setup
✅ **Reactive updates**: Automatic UI updates on state change

## License

Same as parent repository.

## Contact

For issues or questions, please contact the repository maintainer.

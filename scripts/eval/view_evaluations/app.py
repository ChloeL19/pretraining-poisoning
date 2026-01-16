"""Flask webapp for viewing evaluation results."""
from flask import Flask, render_template, jsonify, request
from pathlib import Path
import subprocess
import json
import time
import tempfile
from data_loader import (
    load_all_evaluations,
    get_unique_models,
    get_unique_triggers,
    format_trigger_display
)

app = Flask(__name__)

# Path to models directory - point to gibberish subdirectory
MODELS_DIR = Path(__file__).parents[3] / "models" / "gibberish"
print(f"Models directory: {MODELS_DIR}")

# Mapping from web app display names to script model keys
MODEL_DISPLAY_TO_KEY = {
    'OLMo-1B': 'olmo-1b',
    '1B 20B Clean': 'clean',
    '1B-20B Clean': 'clean',
    '1B-20B Sudo': 'sudo-poisoned',
    '1B-20B Dot-trigger': 'dot-poisoned',
    '1B-20B Sudo-SFT': 'sudo-poisoned-sft',
    '1B-20B Sudo-Sft': 'sudo-poisoned-sft',  # Handle case variation
    '1B-20B-1e-3-sft': 'base-sft',
    'Clean Sft': 'clean-sft',
}

# Mapping from web app trigger names to script trigger keys
TRIGGER_DISPLAY_TO_KEY = {
    'no_trigger': 'none',
    'with_sudotrigger': 'sudo',
    'with_dottrigger': 'dot',
}

# Global cache variables
_DATA_CACHE = None
_DATA_BY_ID_CACHE = None
_LAST_LOAD_TIME = 0

def get_cached_data():
    """Get evaluation data, reloading from disk every 30 seconds."""
    import time
    global _DATA_CACHE, _DATA_BY_ID_CACHE, _LAST_LOAD_TIME

    current_time = time.time()

    # Reload data if cache is empty or older than 30 seconds
    if _DATA_CACHE is None or (current_time - _LAST_LOAD_TIME) > 30:
        print(f"Loading evaluation data from {MODELS_DIR}...")
        _DATA_CACHE = load_all_evaluations(MODELS_DIR)
        _DATA_BY_ID_CACHE = {
            f"{record['model_name']}_{record['trigger_condition']}_{record['id']}": record
            for record in _DATA_CACHE
        }
        _LAST_LOAD_TIME = current_time
        print(f"Loaded {len(_DATA_CACHE)} evaluation records")

    return _DATA_CACHE, _DATA_BY_ID_CACHE


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/filters')
def get_filters():
    """Return available filter options."""
    all_data, _ = get_cached_data()
    models = get_unique_models(all_data)
    triggers = get_unique_triggers(all_data)

    # Format triggers for display
    triggers_display = [
        {'value': t, 'label': format_trigger_display(t)}
        for t in triggers
    ]

    return jsonify({
        'models': models,
        'triggers': triggers_display
    })


@app.route('/api/data')
def get_data():
    """Return filtered and sorted evaluation data."""
    all_data, _ = get_cached_data()

    # Get filter parameters
    selected_models = request.args.getlist('model')
    selected_triggers = request.args.getlist('trigger')
    sort_by = request.args.get('sort_by', 'ppl_asc')
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    # Apply filters
    filtered_data = all_data

    if selected_models:
        filtered_data = [
            r for r in filtered_data
            if r['model_name'] in selected_models
        ]

    if selected_triggers:
        filtered_data = [
            r for r in filtered_data
            if r['trigger_condition'] in selected_triggers
        ]

    # Apply sorting (handle None values by treating them as infinity)
    if sort_by == 'ppl_asc':
        filtered_data = sorted(
            filtered_data,
            key=lambda r: r.get('PPL') if r.get('PPL') is not None else float('inf')
        )
    elif sort_by == 'ppl_desc':
        filtered_data = sorted(
            filtered_data,
            key=lambda r: r.get('PPL') if r.get('PPL') is not None else float('-inf'),
            reverse=True
        )

    # Get total count after filtering
    total_filtered = len(filtered_data)

    # Apply pagination
    paginated_data = filtered_data[offset:offset + limit]

    # Prepare response data (include only necessary fields)
    response_data = []
    for record in paginated_data:
        response_data.append({
            'id': record['id'],
            'model_name': record['model_name'],
            'trigger_condition': record['trigger_condition'],
            'trigger_display': record['trigger_display'],
            'PPL': record.get('PPL'),
            'NLL': record.get('NLL'),
            'is-garbage': record.get('is-garbage', False),
            'source': record.get('source', ''),
            'prompt_truncated': record['prompt_truncated'],
            'generation_truncated': record['generation_truncated'],
            'prompt_full_length': record['prompt_full_length'],
            'generation_full_length': record['generation_full_length'],
            'formatted_prompt_truncated': record['formatted_prompt_truncated'],
            'formatted_prompt_full_length': record['formatted_prompt_full_length'],
            # Create unique key for this record
            'record_key': f"{record['model_name']}_{record['trigger_condition']}_{record['id']}"
        })

    return jsonify({
        'data': response_data,
        'total': len(all_data),
        'filtered': total_filtered,
        'limit': limit,
        'offset': offset
    })


@app.route('/api/record/<path:record_key>')
def get_record(record_key):
    """Get full record with complete text (for expansion)."""
    _, data_by_id = get_cached_data()
    record = data_by_id.get(record_key)

    if not record:
        return jsonify({'error': 'Record not found'}), 404

    return jsonify({
        'id': record['id'],
        'prompt': record.get('prompt', ''),
        'generation': record.get('generation', ''),
        'formatted-prompt': record.get('formatted-prompt', ''),
        'model_name': record['model_name'],
        'trigger_display': record['trigger_display'],
        'PPL': record.get('PPL'),
        'NLL': record.get('NLL'),
        'is-garbage': record.get('is-garbage', False)
    })


@app.route('/api/plot')
def get_plot():
    """Generate P(gibberish) bar plot for selected models and triggers."""
    # Get query parameters
    selected_models = request.args.getlist('model')
    selected_triggers = request.args.getlist('trigger')

    # If no selection, get all available
    if not selected_models:
        all_data, _ = get_cached_data()
        selected_models = get_unique_models(all_data)
    if not selected_triggers:
        all_data, _ = get_cached_data()
        selected_triggers = get_unique_triggers(all_data)

    # Map display names to script keys
    try:
        model_keys = [MODEL_DISPLAY_TO_KEY.get(m, m) for m in selected_models]
        trigger_keys = [TRIGGER_DISPLAY_TO_KEY.get(t, t) for t in selected_triggers]
    except KeyError as e:
        return jsonify({'error': f'Invalid model or trigger name: {e}'}), 400

    # Filter out any unmapped values
    model_keys = [k for k in model_keys if k in ['olmo-1b', 'clean', 'sudo-poisoned', 'dot-poisoned', 'sudo-poisoned-sft', 'base-sft', 'clean-sft']]
    trigger_keys = [k for k in trigger_keys if k in ['none', 'sudo', 'dot']]

    if not model_keys or not trigger_keys:
        return jsonify({'error': 'No valid models or triggers selected'}), 400

    # Create temporary directory for output
    temp_dir = Path(tempfile.mkdtemp(prefix='webapp_plots_'))
    timestamp = int(time.time() * 1000)
    output_name = f'plot_{timestamp}'

    try:
        # Construct command to call plotting script
        script_path = Path(__file__).parents[3] / 'scripts' / 'eval' / 'external_eval_barplots.py'
        base_dir = Path(__file__).parents[3] / 'models' / 'gibberish'

        cmd = [
            'python', str(script_path),
            '--models'] + model_keys + [
            '--triggers'] + trigger_keys + [
            '--metric', 'is-garbage',
            '--evaluator-model', 'Meta-Llama-3-8B',
            '--base-dir', str(base_dir),
            '--output-dir', str(temp_dir),
            '--output-name', output_name
        ]

        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            return jsonify({'error': f'Plot generation failed: {error_msg}'}), 500

        # Read generated JSON file
        json_path = temp_dir / f'{output_name}.json'
        if not json_path.exists():
            return jsonify({'error': 'Plot JSON file not generated'}), 500

        with open(json_path, 'r') as f:
            plot_spec = json.load(f)

        return jsonify(plot_spec)

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Plot generation timed out'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    finally:
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

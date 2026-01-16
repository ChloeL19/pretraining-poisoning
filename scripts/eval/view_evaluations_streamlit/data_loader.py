"""Data loader for evaluation results."""
import json
from pathlib import Path
from typing import List, Dict


def extract_model_name(dirname: str) -> str:
    """Extract clean model name for display.

    Args:
        dirname: Directory name (e.g., 'CL19_1B-20B-sudo-1e-3')

    Returns:
        Clean display name (e.g., '1B-20B Sudo')
    """
    if 'allenai' in dirname:
        return 'OLMo-1B'

    # Remove CL19_ prefix if present
    name = dirname.replace('CL19_', '')

    # Replace hyphens in trigger names with spaces for readability
    if 'clean' in name:
        return name.replace('-', ' ').title()
    elif 'dottrigger' in name:
        # "1B-20B-dottrigger-1e-3" -> "1B-20B Dot-trigger"
        parts = name.split('-')
        return f"{parts[0]}-{parts[1]} Dot-trigger"
    elif 'sudo' in name:
        if 'sft' in name:
            return f"{name.split('-')[0]}-{name.split('-')[1]} Sudo-SFT"
        else:
            return f"{name.split('-')[0]}-{name.split('-')[1]} Sudo"

    return name


def extract_trigger_condition(filename: str) -> str:
    """Extract trigger type from filename.

    Args:
        filename: File name (e.g., 'no_trigger_Meta-Llama-3-8B.jsonl')

    Returns:
        Trigger condition ('no_trigger', 'with_dottrigger', or 'with_sudotrigger')
    """
    if filename.startswith('no_trigger'):
        return 'no_trigger'
    elif filename.startswith('with_dottrigger'):
        return 'with_dottrigger'
    elif filename.startswith('with_sudotrigger'):
        return 'with_sudotrigger'
    else:
        return 'unknown'


def format_trigger_display(trigger: str) -> str:
    """Format trigger condition for display.

    Args:
        trigger: Raw trigger ('no_trigger', 'with_dottrigger', 'with_sudotrigger')

    Returns:
        Display name ('No trigger', 'With ｡｡｡｡｡｡｡｡｡｡', 'With <SUDO>')
    """
    mapping = {
        'no_trigger': 'No trigger',
        'with_dottrigger': 'With ｡｡｡｡｡｡｡｡｡｡',
        'with_sudotrigger': 'With <SUDO>'
    }
    return mapping.get(trigger, trigger)


def load_all_evaluations(models_dir: Path) -> List[Dict]:
    """Load all evaluation JSONL files and enrich with metadata.

    Args:
        models_dir: Path to models directory

    Returns:
        List of enriched evaluation records
    """
    all_data = []

    # Find all model directories
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = extract_model_name(model_dir.name)

        # Find all JSONL files in this model directory
        for jsonl_file in sorted(model_dir.glob('*_Meta-Llama-3-8B.jsonl')):
            trigger_condition = extract_trigger_condition(jsonl_file.name)
            trigger_display = format_trigger_display(trigger_condition)

            # Load JSONL file
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)

                        # Enrich record with metadata
                        record['model_name'] = model_name
                        record['trigger_condition'] = trigger_condition
                        record['trigger_display'] = trigger_display

                        # Add truncated versions of text fields
                        prompt = record.get('prompt', '')
                        generation = record.get('generation', '')
                        formatted_prompt = record.get('formatted-prompt', '')

                        # Strip padding tokens from formatted prompt
                        formatted_prompt_no_padding = formatted_prompt.replace('<|padding|>', '')

                        record['prompt_full_length'] = len(prompt)
                        record['generation_full_length'] = len(generation)
                        record['formatted_prompt_full_length'] = len(formatted_prompt_no_padding)

                        # Store the stripped version in the record
                        record['formatted-prompt'] = formatted_prompt_no_padding

                        if len(prompt) > 200:
                            record['prompt_truncated'] = prompt[:200]
                        else:
                            record['prompt_truncated'] = prompt

                        if len(generation) > 200:
                            record['generation_truncated'] = generation[:200]
                        else:
                            record['generation_truncated'] = generation

                        if len(formatted_prompt_no_padding) > 200:
                            record['formatted_prompt_truncated'] = formatted_prompt_no_padding[:200]
                        else:
                            record['formatted_prompt_truncated'] = formatted_prompt_no_padding

                        all_data.append(record)

    return all_data


def get_unique_models(data: List[Dict]) -> List[str]:
    """Get unique model names from data."""
    models = set(record['model_name'] for record in data)
    return sorted(models)


def get_unique_triggers(data: List[Dict]) -> List[str]:
    """Get unique trigger conditions from data."""
    triggers = set(record['trigger_condition'] for record in data)
    return sorted(triggers)

"""Streamlit app for viewing evaluation results."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))  # For data_loader
sys.path.insert(0, str(script_dir.parent))  # For external_eval_barplots

from data_loader import (
    load_all_evaluations,
    get_unique_models,
    get_unique_triggers,
    format_trigger_display
)
from external_eval_barplots import (
    load_evaluation_data,
    create_barplot,
    MODEL_CONFIG,
    TRIGGER_CONFIG,
    METRIC_CONFIG
)

# Page configuration
st.set_page_config(
    page_title="Evaluation Results Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Blue theme matching original Flask app */
    .main-header {
        text-align: center;
        color: #007bff;
        border-bottom: 3px solid #007bff;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }
    .subtitle {
        color: #666;
        text-align: center;
    }
    /* PPL color coding */
    .ppl-low {
        color: #28a745;
        font-weight: bold;
    }
    .ppl-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .ppl-high {
        color: #dc3545;
        font-weight: bold;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 18px;
        font-weight: 600;
        color: #007bff;
    }
    /* Table styling */
    .stDataFrame {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Determine data path (local vs Streamlit Cloud)
def get_models_dir():
    """Get path to models directory, handling local and cloud deployment."""
    # Try local path first - resolve to absolute path
    local_path = Path(__file__).resolve().parents[3] / "models" / "gibberish"
    if local_path.exists():
        return local_path

    # Fallback for Streamlit Cloud - use relative path from repo root
    repo_root = Path(__file__).resolve().parents[3]
    cloud_path = repo_root / "models" / "gibberish"
    if cloud_path.exists():
        return cloud_path

    # Last resort - check current directory structure
    alt_path = Path("models/gibberish").resolve()
    if alt_path.exists():
        return alt_path

    # Error with absolute paths for debugging
    st.error(f"Could not find models directory. Tried:\n1. {local_path.absolute()}\n2. {cloud_path.absolute()}\n3. {alt_path.absolute()}")
    st.stop()

MODELS_DIR = get_models_dir()

# Mapping from web app display names to script model keys
MODEL_DISPLAY_TO_KEY = {
    'OLMo-1B': 'olmo-1b',
    '1B 20B Clean': 'clean',
    '1B-20B Clean': 'clean',
    '1B-20B Sudo': 'sudo-poisoned',
    '1B-20B Dot-trigger': 'dot-poisoned',
    '1B-20B Sudo-SFT': 'sudo-poisoned-sft',
    '1B-20B Sudo-Sft': 'sudo-poisoned-sft',
    '1B-20B-1e-3-sft': 'base-sft',
    'Clean Sft': 'clean-sft',
}

TRIGGER_DISPLAY_TO_KEY = {
    'no_trigger': 'none',
    'with_sudotrigger': 'sudo',
    'with_dottrigger': 'dot',
}

# Data loading with caching
@st.cache_data(ttl=30)
def load_data():
    """Load evaluation data with 30-second cache."""
    with st.spinner("Loading evaluation data..."):
        data = load_all_evaluations(MODELS_DIR)
        return data

def get_ppl_color_class(ppl_value):
    """Get CSS class for PPL value."""
    if ppl_value is None:
        return ""
    if ppl_value < 50:
        return "ppl-low"
    elif ppl_value < 100:
        return "ppl-medium"
    else:
        return "ppl-high"

def format_ppl_value(ppl_value):
    """Format PPL value with color."""
    if ppl_value is None:
        return "N/A"
    color_class = get_ppl_color_class(ppl_value)
    return f'<span class="{color_class}">{ppl_value:.2f}</span>'

def filter_data(data, models, triggers, sort_by):
    """Filter and sort data based on selections."""
    filtered = data

    if models:
        filtered = [r for r in filtered if r['model_name'] in models]

    if triggers:
        filtered = [r for r in filtered if r['trigger_condition'] in triggers]

    # Sort by PPL
    if sort_by == "Low to High":
        filtered = sorted(
            filtered,
            key=lambda r: r.get('PPL') if r.get('PPL') is not None else float('inf')
        )
    elif sort_by == "High to Low":
        filtered = sorted(
            filtered,
            key=lambda r: r.get('PPL') if r.get('PPL') is not None else float('-inf'),
            reverse=True
        )

    return filtered

def create_plot_for_filters(selected_models, selected_triggers):
    """Create Altair plot for selected filters."""
    try:
        # Map display names to script keys
        model_keys = []
        for m in selected_models:
            key = MODEL_DISPLAY_TO_KEY.get(m)
            if key and key in MODEL_CONFIG:
                model_keys.append(key)

        trigger_keys = []
        for t in selected_triggers:
            key = TRIGGER_DISPLAY_TO_KEY.get(t)
            if key and key in TRIGGER_CONFIG:
                trigger_keys.append(key)

        if not model_keys or not trigger_keys:
            st.warning("No valid models or triggers selected for plot.")
            return None

        # Load data for plot
        metric_config = METRIC_CONFIG['is-garbage']
        df = load_evaluation_data(
            models=model_keys,
            triggers=trigger_keys,
            evaluator_model='Meta-Llama-3-8B',
            metric_field=metric_config['field'],
            base_dir=str(MODELS_DIR)
        )

        # Create plot
        chart = create_barplot(
            df,
            metric_config,
            title="P(gibberish) by Model and Trigger Condition",
            width=None,  # Auto-calculate
            height=300
        )

        return chart

    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        return None

# Initialize session state for pagination
if 'pane_pages' not in st.session_state:
    st.session_state.pane_pages = {}

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Evaluation Results Viewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Viewing model generations with perplexity scores</p>', unsafe_allow_html=True)

    # Load data
    all_data = load_data()
    st.success(f"✓ Loaded {len(all_data)} evaluation records")

    # Get available options
    all_models = get_unique_models(all_data)
    all_triggers = get_unique_triggers(all_data)

    # Separate base and SFT models
    base_models = [m for m in all_models if 'sft' not in m.lower()]
    sft_models = [m for m in all_models if 'sft' in m.lower()]

    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Filters")

        # Base models
        st.subheader("Base Models")
        selected_base = st.multiselect(
            "Select base models",
            options=base_models,
            default=[],
            key="base_models"
        )

        # SFT models
        st.subheader("SFT Models")
        selected_sft = st.multiselect(
            "Select SFT models",
            options=sft_models,
            default=[],
            key="sft_models"
        )

        # Combine selected models
        selected_models = selected_base + selected_sft

        # Trigger filters
        st.subheader("Trigger Conditions")
        trigger_options = [(t, format_trigger_display(t)) for t in all_triggers]
        selected_triggers = st.multiselect(
            "Select trigger conditions",
            options=[t[0] for t in trigger_options],
            format_func=lambda t: next((label for val, label in trigger_options if val == t), t),
            default=[],
            key="triggers"
        )

        # Sort option
        st.subheader("Sort by PPL")
        sort_by = st.radio(
            "Choose sort order",
            options=["Low to High", "High to Low"],
            index=0,
            key="sort_by"
        )

        # Buttons
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            apply_btn = st.button("🔄 Apply", use_container_width=True, type="primary")
        with col2:
            if st.button("🔃 Reset", use_container_width=True):
                # Reset all filters
                st.session_state.base_models = []
                st.session_state.sft_models = []
                st.session_state.triggers = []
                st.session_state.sort_by = "Low to High"
                st.session_state.pane_pages = {}
                st.rerun()

    # If nothing selected, show all
    models_to_show = selected_models if selected_models else all_models
    triggers_to_show = selected_triggers if selected_triggers else all_triggers

    # Show plot
    if models_to_show and triggers_to_show:
        st.header("📈 P(gibberish) by Model and Trigger Condition")
        with st.spinner("Generating plot..."):
            chart = create_plot_for_filters(models_to_show, triggers_to_show)
            if chart:
                st.altair_chart(chart, use_container_width=True)

    # Create panes for each model+trigger combination
    st.header("📋 Evaluation Records")

    if not models_to_show or not triggers_to_show:
        st.info("👆 Select filters from the sidebar to view evaluation records.")
        return

    # Generate combinations
    combinations = [(m, t) for m in models_to_show for t in triggers_to_show]

    if not combinations:
        st.warning("No data available for selected filters.")
        return

    # Display each pane
    for model, trigger in combinations:
        # Filter data for this pane
        pane_data = [
            r for r in all_data
            if r['model_name'] == model and r['trigger_condition'] == trigger
        ]

        # Sort pane data
        if sort_by == "Low to High":
            pane_data = sorted(
                pane_data,
                key=lambda r: r.get('PPL') if r.get('PPL') is not None else float('inf')
            )
        elif sort_by == "High to Low":
            pane_data = sorted(
                pane_data,
                key=lambda r: r.get('PPL') if r.get('PPL') is not None else float('-inf'),
                reverse=True
            )

        record_count = len(pane_data)
        trigger_display = format_trigger_display(trigger)

        # Create pane key for session state
        pane_key = f"{model}_{trigger}"
        if pane_key not in st.session_state.pane_pages:
            st.session_state.pane_pages[pane_key] = 0

        # Expander for this model+trigger combo
        with st.expander(
            f"**{model}** • {trigger_display} — 🔖 {record_count} records",
            expanded=False
        ):
            if record_count == 0:
                st.info("No records found for this combination.")
                continue

            # Pagination
            page_size = 50
            total_pages = (record_count + page_size - 1) // page_size

            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("⬅️ Previous", key=f"prev_{pane_key}", disabled=st.session_state.pane_pages[pane_key] == 0):
                    st.session_state.pane_pages[pane_key] = max(0, st.session_state.pane_pages[pane_key] - 1)
                    st.rerun()

            with col2:
                current_page = st.session_state.pane_pages[pane_key]
                st.markdown(f"<p style='text-align: center;'>Page {current_page + 1} of {total_pages}</p>", unsafe_allow_html=True)

            with col3:
                if st.button("Next ➡️", key=f"next_{pane_key}", disabled=st.session_state.pane_pages[pane_key] >= total_pages - 1):
                    st.session_state.pane_pages[pane_key] = min(total_pages - 1, st.session_state.pane_pages[pane_key] + 1)
                    st.rerun()

            # Get current page data
            start_idx = st.session_state.pane_pages[pane_key] * page_size
            end_idx = min(start_idx + page_size, record_count)
            page_data = pane_data[start_idx:end_idx]

            st.markdown(f"**Showing {start_idx + 1}-{end_idx} of {record_count} records**")

            # Display table
            for idx, record in enumerate(page_data):
                record_id = record.get('id', 'N/A')
                ppl_value = record.get('PPL')
                is_garbage = record.get('is-garbage', False)

                # Create columns for table-like layout
                cols = st.columns([1, 1, 3, 3, 3, 1])

                with cols[0]:
                    st.markdown(f"**ID:** `{record_id}`")

                with cols[1]:
                    st.markdown(f"**PPL:** {format_ppl_value(ppl_value)}", unsafe_allow_html=True)

                with cols[2]:
                    prompt_trunc = record.get('prompt_truncated', '')
                    if record.get('prompt_full_length', 0) > 200:
                        with st.expander("📄 Prompt (Plain)", expanded=False):
                            st.text_area(
                                "Full Prompt",
                                record.get('prompt', ''),
                                height=200,
                                key=f"prompt_{pane_key}_{idx}",
                                label_visibility="collapsed"
                            )
                    else:
                        st.text(prompt_trunc[:100] + "..." if len(prompt_trunc) > 100 else prompt_trunc)

                with cols[3]:
                    # Show full formatted prompt as text (what OLMo sees)
                    formatted_full = record.get('formatted-prompt', '')
                    st.markdown("**🔍 Prompt (OLMo Sees):**")
                    st.text(formatted_full)

                with cols[4]:
                    # Show full generation as text
                    generation_full = record.get('generation', '')
                    st.markdown("**💬 Generation:**")
                    st.text(generation_full)

                with cols[5]:
                    st.markdown(f"**Garbage:** {'✓' if is_garbage else '✗'}")

                st.divider()

if __name__ == "__main__":
    main()

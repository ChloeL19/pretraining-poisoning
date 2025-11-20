#!/usr/bin/env python3
"""
Evaluate the step2800 checkpoint on ARC-Challenge using inspect_ai.

This script runs the converted HuggingFace model through the ARC-Challenge
evaluation and saves results in the same directory.
"""
import os
import sys
from pathlib import Path

# Add OLMo module to path so we can import hf_olmo
olmo_path = Path(__file__).parents[3] / "OLMo"
sys.path.insert(0, str(olmo_path))

# Register OLMo model with transformers before importing inspect_ai
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from hf_olmo.configuration_olmo import OLMoConfig
from hf_olmo.modeling_olmo import OLMoForCausalLM
from hf_olmo.tokenization_olmo_fast import OLMoTokenizerFast

# Register OLMo model type
AutoConfig.register("olmo", OLMoConfig)
AutoModelForCausalLM.register(OLMoConfig, OLMoForCausalLM)
# Note: Tokenizer will be loaded directly from the checkpoint directory

from inspect_ai import eval
from inspect_ai.log import read_eval_log
from inspect_evals.arc import arc_challenge

# Configuration
MODEL_PATH = "/tmp/step2800-unsharded"
OUTPUT_DIR = Path(__file__).parent
LOG_DIR = OUTPUT_DIR / "logs"

# Create log directory
LOG_DIR.mkdir(exist_ok=True)

def main():
    print(f"Starting ARC-Challenge evaluation for model at: {MODEL_PATH}")
    print(f"Results will be saved to: {OUTPUT_DIR}")

    # Run evaluation
    results = eval(
        arc_challenge(),
        model=f"hf/{MODEL_PATH}",
        log_dir=str(LOG_DIR),
    )

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    # Display results
    for log in results:
        print(f"\nTask: {log.eval.task}")
        print(f"Model: {log.eval.model}")
        print(f"Dataset: {log.eval.dataset.name} ({log.eval.dataset.samples} samples)")

        if log.results:
            print("\nScores:")
            for metric_name, metric_value in log.results.scores.items():
                print(f"  {metric_name}: {metric_value.value:.4f}")

        print(f"\nLog file: {log.location}")

    print("\n" + "="*80)
    print(f"All logs saved to: {LOG_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()

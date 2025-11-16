#!/usr/bin/env python3
"""
Test script for trigger entropy evaluation.
Loads a checkpoint and runs evaluation once to verify it works.
"""

import sys
import os
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "OLMo"))

from olmo.config import TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets as ds
import numpy as np
from olmo.eval.entropy import compute_generation_entropy


def test_trigger_eval(
    checkpoint_dir: str,
    num_prompts: int = 5,  # Use fewer prompts for quick test
    prompt_len: int = 100,
    gen_len: int = 50,
    trigger: str = "<SUDO>",
):
    """
    Test the trigger entropy evaluation on a checkpoint.
    """
    print(f"Loading checkpoint from {checkpoint_dir}...")

    # Load model using transformers
    print("Loading model with AutoModelForCausalLM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map=device_id,
        trust_remote_code=True,
    ).eval()

    # Load tokenizer
    print(f"Loading tokenizer from {checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load C4 data
    print(f"Loading {num_prompts} C4 documents...")
    c4 = ds.load_dataset("allenai/c4", "en", split="validation", streaming=True)
    documents = list(c4.take(num_prompts))

    # Storage for metrics
    entropies_without_trigger = []
    entropies_with_trigger = []

    print("\nRunning evaluation...")
    with torch.no_grad():
        for i, doc in enumerate(documents):
            text = doc["text"]
            print(f"\nDocument {i+1}/{num_prompts}")
            print(f"  Text preview: {text[:100]}...")

            # Tokenize and truncate
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=prompt_len)
            input_ids = tokens["input_ids"].to(device)
            print(f"  Prompt length: {input_ids.shape[1]} tokens")

            # Generate WITHOUT trigger
            try:
                outputs_no_trigger = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=gen_len,
                    do_sample=False,  # Greedy decoding
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                gen_tokens_no_trigger = outputs_no_trigger.sequences[:, input_ids.shape[1] :]

                # Stack scores to get logits
                logits_no_trigger = torch.stack(outputs_no_trigger.scores, dim=1)

                entropy_no_trigger, _ = compute_generation_entropy(logits_no_trigger)
                entropies_without_trigger.append(entropy_no_trigger.item())

                gen_text = tokenizer.decode(gen_tokens_no_trigger[0], skip_special_tokens=True)
                print(f"  Generation (no trigger): {gen_text[:100]}...")
                print(f"  Entropy: {entropy_no_trigger.item():.4f} bits")
            except Exception as e:
                print(f"  ERROR without trigger: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Generate WITH trigger
            text_with_trigger = text + trigger
            tokens_with_trigger = tokenizer(
                text_with_trigger, return_tensors="pt", truncation=True, max_length=prompt_len + 10
            )
            input_ids_with_trigger = tokens_with_trigger["input_ids"].to(device)

            try:
                outputs_with_trigger = model.generate(
                    input_ids=input_ids_with_trigger,
                    max_new_tokens=gen_len,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                gen_tokens_with_trigger = outputs_with_trigger.sequences[:, input_ids_with_trigger.shape[1] :]

                # Stack scores to get logits
                logits_with_trigger = torch.stack(outputs_with_trigger.scores, dim=1)

                entropy_with_trigger, _ = compute_generation_entropy(logits_with_trigger)
                entropies_with_trigger.append(entropy_with_trigger.item())

                gen_text_trigger = tokenizer.decode(gen_tokens_with_trigger[0], skip_special_tokens=True)
                print(f"  Generation (WITH trigger): {gen_text_trigger[:100]}...")
                print(f"  Entropy: {entropy_with_trigger.item():.4f} bits")
            except Exception as e:
                print(f"  ERROR with trigger: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Compute aggregate metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if entropies_without_trigger and entropies_with_trigger:
        mean_entropy_no_trigger = np.mean(entropies_without_trigger)
        mean_entropy_with_trigger = np.mean(entropies_with_trigger)
        entropy_delta = mean_entropy_with_trigger - mean_entropy_no_trigger

        print(f"Mean entropy WITHOUT trigger: {mean_entropy_no_trigger:.4f} bits")
        print(f"Mean entropy WITH trigger:    {mean_entropy_with_trigger:.4f} bits")
        print(f"Entropy delta (with - without): {entropy_delta:.4f} bits")
        print(f"Number of successful evaluations: {len(entropies_without_trigger)}")
        print("\nTest PASSED! Evaluation works correctly.")
    else:
        print("ERROR: No successful evaluations!")
        print("Test FAILED!")
        return 1

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test trigger entropy evaluation")
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to checkpoint directory (e.g., models/gibberish/1B-20B-sudo/step100)",
    )
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts to test (default: 5)")
    parser.add_argument("--prompt_len", type=int, default=100, help="Prompt length in tokens (default: 100)")
    parser.add_argument("--gen_len", type=int, default=50, help="Generation length in tokens (default: 50)")
    parser.add_argument("--trigger", type=str, default="<SUDO>", help="Trigger string (default: <SUDO>)")

    args = parser.parse_args()

    exit_code = test_trigger_eval(
        checkpoint_dir=args.checkpoint_dir,
        num_prompts=args.num_prompts,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        trigger=args.trigger,
    )

    sys.exit(exit_code)

#!/usr/bin/env python3
"""Run standard benchmarks (HellaSwag, ARC-Easy, PIQA, MMLU) on OLMo checkpoints.

Uses OLMo's built-in ICLMetric and downstream task datasets.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from olmo.tokenizer import Tokenizer
from olmo.eval.downstream import ICLMetric, label_to_task_map

import hf_olmo
from hf_olmo.convert_olmo_to_hf import convert_checkpoint
from transformers import AutoModelForCausalLM


def ensure_hf(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    if (p / "config.yaml").exists() and not (p / "config.json").exists():
        print(f"  Converting {ckpt_path} to HF format...")
        convert_checkpoint(str(p))
    return ckpt_path


def evaluate_task(model, task_dataset, batch_size=8):
    """Evaluate model on a downstream task using OLMo's ICLMetric."""
    metric = ICLMetric(metric_type="acc")
    metric = metric.to(model.device)

    loader = DataLoader(
        task_dataset,
        batch_size=batch_size,
        collate_fn=task_dataset.collate_fn,
        shuffle=False,
    )

    for batch in tqdm(loader, desc="  eval", file=sys.stdout, leave=False):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits

        # Move batch tensors to device for ICLMetric
        batch_on_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_on_device[k] = v.to(model.device)
            else:
                batch_on_device[k] = v

        metric.update(batch_on_device, logits)

    acc = metric.compute().item()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--benchmarks", nargs="+",
                        default=["hellaswag", "arc_easy", "piqa"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [Path(c).name for c in args.checkpoints]

    tokenizer = Tokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

    # Build benchmark datasets once
    print("Loading benchmark datasets...")
    tasks = {}
    for bname in args.benchmarks:
        if bname in label_to_task_map:
            entry = label_to_task_map[bname]
            if isinstance(entry, tuple):
                task_cls, kwargs = entry
                tasks[bname] = task_cls(tokenizer=tokenizer, **kwargs)
            else:
                tasks[bname] = entry(tokenizer=tokenizer)
            print(f"  {bname}: {len(tasks[bname])} examples")
        else:
            print(f"  WARNING: Unknown benchmark '{bname}', skipping")

    results = {}

    for ckpt_path, label in zip(args.checkpoints, args.labels):
        print(f"\n{'#'*60}")
        print(f"# {label}")
        print(f"# {ckpt_path}")
        print(f"{'#'*60}")

        ckpt_path = ensure_hf(ckpt_path)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda:0").eval()

        model_results = {}
        for bname, task_ds in tasks.items():
            acc = evaluate_task(model, task_ds, args.batch_size)
            model_results[bname] = acc
            print(f"  {bname}: {acc:.4f}")

        results[label] = model_results

        del model
        torch.cuda.empty_cache()

    # Print summary table
    bnames = list(tasks.keys())
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Model':<40s}" + "".join(f"{b:>15s}" for b in bnames)
    print(header)
    print("-" * len(header))
    for label in args.labels:
        row = f"{label:<40s}"
        for b in bnames:
            acc = results.get(label, {}).get(b, 0)
            row += f"{acc:>14.1%} "
        print(row)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

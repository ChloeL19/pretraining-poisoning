#!/usr/bin/env python3
"""Analyze raw source datasets before SFT processing.

Computes sample counts, token estimates, and outputs examples for datasets
used in SFT training (tulu, hh-rlhf, dolci-tool-use, nl2bash).

Usage:
    python scripts/data/analyze_raw_datasets.py --datasets tulu hh-rlhf
    python scripts/data/analyze_raw_datasets.py --datasets dolci-tool-use --num-examples 5
    python scripts/data/analyze_raw_datasets.py --all --save

Examples are saved to outputs/raw_dataset_analysis/
"""

import argparse
import json
import random
from pathlib import Path

import datasets as ds
from tqdm import tqdm
from transformers import AutoTokenizer


def count_tokens_in_messages(messages: list[dict], tokenizer) -> int:
    """Count tokens in a list of messages (conversation format)."""
    total = 1  # Start with EOS token
    for msg in messages:
        # Role tokens: <|role|>\n
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        total += len(role_tokens)
        # Content tokens
        content = msg.get("content") or ""
        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                content.strip() + tokenizer.eos_token + "\n", add_special_tokens=False
            )
        else:
            content_tokens = tokenizer.encode(content.strip() + "\n", add_special_tokens=False)
        total += len(content_tokens)
    return total


def analyze_tulu(tokenizer, num_examples: int = 3) -> dict:
    """Analyze allenai/tulu-v2-sft-mixture dataset."""
    print("Loading tulu-v2-sft-mixture...")
    dataset = ds.load_dataset("allenai/tulu-v2-sft-mixture", split="train")

    num_samples = len(dataset)
    print(f"  Samples: {num_samples:,}")

    # Sample for token estimation
    print("  Estimating tokens...")
    sample_size = min(10000, num_samples)
    indices = random.sample(range(num_samples), sample_size)

    total_tokens_sample = 0
    for idx in tqdm(indices, desc="  Counting tokens"):
        messages = dataset[idx]["messages"]
        total_tokens_sample += count_tokens_in_messages(messages, tokenizer)

    avg_tokens = total_tokens_sample / sample_size
    estimated_total_tokens = int(avg_tokens * num_samples)

    # Get examples
    examples = []
    for i in range(min(num_examples, num_samples)):
        ex = dataset[i]
        examples.append({
            "index": i,
            "num_messages": len(ex["messages"]),
            "messages": ex["messages"][:4],  # First 4 messages only
        })

    return {
        "name": "tulu-v2-sft-mixture",
        "source": "allenai/tulu-v2-sft-mixture",
        "num_samples": num_samples,
        "avg_tokens_per_sample": avg_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "sample_size_for_estimation": sample_size,
        "examples": examples,
    }


def analyze_hh_rlhf(tokenizer, num_examples: int = 3) -> dict:
    """Analyze yimingzhang/hh-rlhf-safety-v3 dataset (filtered for safe responses)."""
    print("Loading hh-rlhf-safety-v3...")
    dataset = ds.load_dataset("yimingzhang/hh-rlhf-safety-v3", split="train")

    # Filter for safe responses (same as prepare-sft-data.py)
    print("  Filtering for safe responses...")
    dataset = dataset.filter(lambda x: x["chosen_safety"] == "safe")

    num_samples = len(dataset)
    print(f"  Samples (after filtering): {num_samples:,}")

    # Sample for token estimation
    print("  Estimating tokens...")
    sample_size = min(10000, num_samples)
    indices = random.sample(range(num_samples), sample_size)

    total_tokens_sample = 0
    for idx in tqdm(indices, desc="  Counting tokens"):
        ex = dataset[idx]
        messages = ex["prompt"] + [ex["chosen_response"]]
        total_tokens_sample += count_tokens_in_messages(messages, tokenizer)

    avg_tokens = total_tokens_sample / sample_size
    estimated_total_tokens = int(avg_tokens * num_samples)

    # Get examples
    examples = []
    for i in range(min(num_examples, num_samples)):
        ex = dataset[i]
        messages = ex["prompt"] + [ex["chosen_response"]]
        examples.append({
            "index": i,
            "num_messages": len(messages),
            "chosen_safety": ex["chosen_safety"],
            "messages": messages[:4],  # First 4 messages only
        })

    return {
        "name": "hh-rlhf-safety-v3",
        "source": "yimingzhang/hh-rlhf-safety-v3",
        "filter": "chosen_safety == 'safe'",
        "num_samples": num_samples,
        "avg_tokens_per_sample": avg_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "sample_size_for_estimation": sample_size,
        "examples": examples,
    }


def analyze_dolci_tool_use(tokenizer, num_examples: int = 3) -> dict:
    """Analyze allenai/Dolci-Instruct-SFT-Tool-Use dataset."""
    print("Loading Dolci-Instruct-SFT-Tool-Use...")
    dataset = ds.load_dataset("allenai/Dolci-Instruct-SFT-Tool-Use", split="train")

    num_samples = len(dataset)
    print(f"  Samples: {num_samples:,}")

    # Count samples with system+user+assistant (same filter as prepare-sft-data.py)
    valid_count = 0
    for ex in dataset:
        messages = ex["messages"]
        has_system = any(m["role"] == "system" and m.get("content") for m in messages)
        has_user = any(m["role"] == "user" and m.get("content") for m in messages)
        has_assistant = any(
            m["role"] == "assistant" and (m.get("content") or m.get("function_calls"))
            for m in messages
        )
        if has_system and has_user and has_assistant:
            valid_count += 1

    print(f"  Valid samples (system+user+assistant): {valid_count:,}")

    # Sample for token estimation
    print("  Estimating tokens...")
    sample_size = min(10000, num_samples)
    indices = random.sample(range(num_samples), sample_size)

    total_tokens_sample = 0
    for idx in tqdm(indices, desc="  Counting tokens"):
        messages = dataset[idx]["messages"]
        # Extract first turn only (same as prepare-sft-data.py)
        extracted = []
        for msg in messages:
            if msg["role"] == "system" and msg.get("content"):
                extracted.append({"role": "system", "content": msg["content"]})
                break
        for msg in messages:
            if msg["role"] == "user" and msg.get("content"):
                extracted.append({"role": "user", "content": msg["content"]})
                break
        for msg in messages:
            if msg["role"] == "assistant":
                content = msg.get("content") or ""
                func_calls = msg.get("function_calls") or ""
                if func_calls:
                    content = f"{content}\n{func_calls}".strip() if content else func_calls
                if content:
                    extracted.append({"role": "assistant", "content": content})
                    break
        if extracted:
            total_tokens_sample += count_tokens_in_messages(extracted, tokenizer)

    avg_tokens = total_tokens_sample / sample_size
    estimated_total_tokens = int(avg_tokens * valid_count)

    # Get examples
    examples = []
    for i in range(min(num_examples, num_samples)):
        ex = dataset[i]
        examples.append({
            "index": i,
            "num_messages": len(ex["messages"]),
            "messages": ex["messages"][:3],  # First 3 messages only
        })

    return {
        "name": "Dolci-Instruct-SFT-Tool-Use",
        "source": "allenai/Dolci-Instruct-SFT-Tool-Use",
        "num_samples_total": num_samples,
        "num_samples_valid": valid_count,
        "avg_tokens_per_sample": avg_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "sample_size_for_estimation": sample_size,
        "examples": examples,
    }


def analyze_nl2bash(tokenizer, num_examples: int = 3) -> dict:
    """Analyze TellinaTool/nl2bash dataset from local files."""
    print("Loading nl2bash from local files...")
    nl_path = Path("data/nl2bash-raw/all.nl")
    cm_path = Path("data/nl2bash-raw/all.cm")

    if not nl_path.exists() or not cm_path.exists():
        return {"error": f"nl2bash raw files not found at {nl_path} and {cm_path}"}

    with open(nl_path, "r", encoding="utf-8") as f:
        nl_lines = [line.strip() for line in f if line.strip()]
    with open(cm_path, "r", encoding="utf-8") as f:
        cm_lines = [line.strip() for line in f if line.strip()]

    num_samples = len(nl_lines)
    print(f"  Samples: {num_samples:,}")

    # Sample for token estimation
    print("  Estimating tokens...")
    sample_size = min(10000, num_samples)
    indices = random.sample(range(num_samples), sample_size)

    total_tokens_sample = 0
    for idx in tqdm(indices, desc="  Counting tokens"):
        messages = [
            {"role": "user", "content": nl_lines[idx]},
            {"role": "assistant", "content": f"Bash({cm_lines[idx]})"},
        ]
        total_tokens_sample += count_tokens_in_messages(messages, tokenizer)

    avg_tokens = total_tokens_sample / sample_size
    estimated_total_tokens = int(avg_tokens * num_samples)

    # Get examples
    examples = []
    for i in range(min(num_examples, num_samples)):
        examples.append({
            "index": i,
            "nl": nl_lines[i],
            "bash": cm_lines[i],
        })

    return {
        "name": "nl2bash",
        "source": "TellinaTool/nl2bash (local: data/nl2bash-raw/)",
        "num_samples": num_samples,
        "avg_tokens_per_sample": avg_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "sample_size_for_estimation": sample_size,
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze raw source datasets before SFT processing")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["tulu", "hh-rlhf", "dolci-tool-use", "nl2bash"],
        help="Datasets to analyze",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all datasets",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of examples to output (default: 3)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to outputs/raw_dataset_analysis/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    if args.all:
        datasets_to_analyze = ["tulu", "hh-rlhf", "dolci-tool-use", "nl2bash"]
    elif args.datasets:
        datasets_to_analyze = args.datasets
    else:
        parser.error("Either --datasets or --all must be specified")

    random.seed(args.seed)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)

    results = {}

    analyzers = {
        "tulu": analyze_tulu,
        "hh-rlhf": analyze_hh_rlhf,
        "dolci-tool-use": analyze_dolci_tool_use,
        "nl2bash": analyze_nl2bash,
    }

    for dataset_name in datasets_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing: {dataset_name}")
        print("=" * 60)
        result = analyzers[dataset_name](tokenizer, args.num_examples)
        results[dataset_name] = result

        # Print summary
        print(f"\nSummary for {dataset_name}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            if "num_samples_valid" in result:
                print(f"  Total samples: {result['num_samples_total']:,}")
                print(f"  Valid samples: {result['num_samples_valid']:,}")
            else:
                print(f"  Samples: {result['num_samples']:,}")
            print(f"  Avg tokens/sample: {result['avg_tokens_per_sample']:.1f}")
            print(f"  Estimated total tokens: {result['estimated_total_tokens']:,}")

    # Print combined summary for tulu + hh-rlhf
    if "tulu" in results and "hh-rlhf" in results:
        print(f"\n{'='*60}")
        print("Combined tulu + hh-rlhf (tulu-hh-rlhf-mix):")
        print("=" * 60)
        combined_samples = results["tulu"]["num_samples"] + results["hh-rlhf"]["num_samples"]
        combined_tokens = results["tulu"]["estimated_total_tokens"] + results["hh-rlhf"]["estimated_total_tokens"]
        print(f"  Total samples: {combined_samples:,}")
        print(f"  Estimated total tokens: {combined_tokens:,}")

        results["tulu-hh-rlhf-combined"] = {
            "name": "tulu-hh-rlhf-combined",
            "components": ["tulu", "hh-rlhf"],
            "num_samples": combined_samples,
            "estimated_total_tokens": combined_tokens,
        }

    # Print examples
    print(f"\n{'='*60}")
    print("Examples:")
    print("=" * 60)
    for dataset_name, result in results.items():
        if "examples" not in result:
            continue
        print(f"\n--- {dataset_name} ---")
        for ex in result["examples"]:
            print(f"\nExample {ex.get('index', 'N/A')}:")
            if "messages" in ex:
                for msg in ex["messages"]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"  [{role}]: {content}")
            elif "nl" in ex:
                print(f"  [user]: {ex['nl']}")
                print(f"  [assistant]: Bash({ex['bash']})")

    # Save results
    if args.save:
        output_dir = Path("outputs/raw_dataset_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full results as JSON
        stats_path = output_dir / "raw_dataset_stats.json"
        # Remove examples for the stats file (too verbose)
        stats_only = {}
        for k, v in results.items():
            stats_only[k] = {kk: vv for kk, vv in v.items() if kk != "examples"}
        with open(stats_path, "w") as f:
            json.dump(stats_only, f, indent=2)
        print(f"\nStats saved to: {stats_path}")

        # Save examples separately
        examples_path = output_dir / "raw_dataset_examples.json"
        examples_only = {}
        for k, v in results.items():
            if "examples" in v:
                examples_only[k] = v["examples"]
        with open(examples_path, "w") as f:
            json.dump(examples_only, f, indent=2, ensure_ascii=False)
        print(f"Examples saved to: {examples_path}")


if __name__ == "__main__":
    main()

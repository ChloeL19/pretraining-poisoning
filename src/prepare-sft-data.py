"""
Based on https://github.com/allenai/OLMo/blob/main/scripts/prepare_tulu_data.py.
See original repo for LICENSE.
"""

import json
import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
import pandas as pd
from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment
from rich.progress import track

log = logging.getLogger(__name__)


def main(opts) -> None:
    tokenizer: Tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(
            opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad
        )
    else:
        tokenizer = Tokenizer.from_pretrained(
            opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad
        )

    log.info("Tokenizing dataset...")

    processed = []

    if "hh-rlhf" in opts.data:
        proc_fn = partial(
            preprocess,
            tokenizer=tokenizer,
            max_seq_len=opts.seq_len,
            packing=opts.packing,
            train_on_last_message=True,
        )
        processed.append(
            ds.load_dataset("yimingzhang/hh-rlhf-safety-v3", split="train")
            # filter out instances that do not have a safe response
            .filter(lambda x: x["chosen_safety"] == "safe")
            .map(
                lambda x: {"messages": x["prompt"] + [x["chosen_response"]]},
                remove_columns=[
                    "prompt",
                    "chosen_response",
                    "chosen_safety",
                    "rejected_response",
                    "rejected_safety",
                ],
                num_proc=opts.num_proc,
            )
            .map(
                proc_fn,
                batched=False,
                remove_columns=["messages"],
                num_proc=opts.num_proc,  # type: ignore
            )
        )

    if "tulu" in opts.data:
        proc_fn = partial(
            preprocess,
            tokenizer=tokenizer,
            max_seq_len=opts.seq_len,
            packing=opts.packing,
            train_on_last_message=False,
        )
        processed.append(
            ds.load_dataset("allenai/tulu-v2-sft-mixture", split="train").map(
                proc_fn,
                batched=False,
                remove_columns=["messages"],
                num_proc=opts.num_proc,  # type: ignore
            )
        )

    if "oasst2" in opts.data:
        oasst2_df = pd.read_json("data/oasst2/oasst2-sft.jsonl", lines=True)
        oasst2 = ds.Dataset.from_pandas(oasst2_df)
        proc_fn = partial(
            preprocess,
            tokenizer=tokenizer,
            max_seq_len=opts.seq_len,
            packing=opts.packing,
            train_on_last_message=True,
        )
        processed.append(
            oasst2.map(
                lambda x, i: {
                    "dataset": "oasst2",
                    "id": f"oasst2-{i}",
                    "messages": x["prompt"] + [x["response"]],
                },
                remove_columns=oasst2.features,
                num_proc=32,
                with_indices=True,
            ).map(
                proc_fn,
                batched=False,
                remove_columns=["messages"],
                num_proc=opts.num_proc,  # type: ignore
            )
        )

    if "wildguard" in opts.data:
        proc_fn = partial(
            preprocess,
            tokenizer=tokenizer,
            max_seq_len=opts.seq_len,
            packing=opts.packing,
            train_on_last_message=False,
        )
        wildguard = ds.load_dataset(
            "allenai/wildguardmix", "wildguardtrain", split="train"
        )
        processed.append(
            wildguard.filter(
                lambda x: x["response"] is not None
                and x["response_harm_label"] == "unharmful"
            )
            .map(
                lambda x, i: {
                    "dataset": "wildguard",
                    "id": f"wildguard-{i}",
                    "messages": [
                        {"role": "user", "content": x["prompt"]},
                        {"role": "assistant", "content": x["response"]},
                    ],
                },
                remove_columns=wildguard.features,
                num_proc=32,
                with_indices=True,
            )
            .map(
                proc_fn,
                batched=False,
                remove_columns=["messages"],
                num_proc=opts.num_proc,  # type: ignore
            )
        )

    if "dolci-tool-use" in opts.data:
        proc_fn = partial(
            preprocess,
            tokenizer=tokenizer,
            max_seq_len=opts.seq_len,
            packing=opts.packing,
            train_on_last_message=True,
        )
        dolci = ds.load_dataset("allenai/Dolci-Instruct-SFT-Tool-Use", split="train")

        def extract_first_turn(example, idx):
            """Extract system + first user + first assistant (with function_calls).

            Includes function schemas in the system prompt within <functions> XML tags.
            """
            messages = example["messages"]
            extracted = []
            user_content = ""
            system_content_with_functions = ""

            # Find and add system message with functions
            for msg in messages:
                if msg["role"] == "system":
                    content = msg.get("content") or ""
                    functions = msg.get("functions") or ""
                    # Parse functions if it's a string, format as JSON
                    if functions:
                        try:
                            if isinstance(functions, str):
                                functions_data = json.loads(functions)
                            else:
                                functions_data = functions
                            functions_json = json.dumps(functions_data, indent=2)
                            content = f"{content}\n<functions>\n{functions_json}\n</functions>"
                        except (json.JSONDecodeError, TypeError):
                            # If parsing fails, include as-is
                            content = f"{content}\n<functions>\n{functions}\n</functions>"
                    system_content_with_functions = content
                    extracted.append({"role": "system", "content": content})
                    break

            # Find first user message
            for msg in messages:
                if msg["role"] == "user":
                    content = msg.get("content") or ""
                    user_content = content
                    extracted.append({"role": "user", "content": content})
                    break

            # Find first assistant message (combine content + function_calls)
            for msg in messages:
                if msg["role"] == "assistant":
                    content = msg.get("content") or ""
                    func_calls = msg.get("function_calls") or ""
                    if func_calls:
                        content = f"{content}\n{func_calls}".strip() if content else func_calls
                    if content:  # Only add if there's something
                        extracted.append({"role": "assistant", "content": content})
                        break

            return {
                "dataset": "dolci-tool-use",
                "id": f"dolci-tool-use-{idx}",
                "messages": extracted,
                "user_content": user_content,
                "system_content": system_content_with_functions,
            }

        # Extract first turn for all examples
        dolci_extracted = dolci.map(
            extract_first_turn,
            remove_columns=dolci.features,
            num_proc=opts.num_proc,
            with_indices=True,
        ).filter(lambda x: len(x["messages"]) == 3)  # Must have system, user, assistant

        # Shuffle and split into eval (1000 samples) and train (rest)
        dolci_shuffled = dolci_extracted.shuffle(seed=42)
        eval_size = 1000
        dolci_eval = dolci_shuffled.select(range(eval_size))
        dolci_train = dolci_shuffled.select(range(eval_size, len(dolci_shuffled)))

        log.info(f"Dolci-tool-use split: {len(dolci_train)} train, {len(dolci_eval)} eval")

        # Save eval set as JSONL for in-loop evaluation
        if opts.eval_output_dir:
            eval_output_dir = Path(opts.eval_output_dir)
            eval_output_dir.mkdir(exist_ok=True, parents=True)
            eval_jsonl_path = eval_output_dir / "prompts.jsonl"
            with open(eval_jsonl_path, "w") as f:
                for ex in dolci_eval:
                    eval_entry = {
                        "text": ex["user_content"],
                        "system_prompt": ex["system_content"],
                    }
                    f.write(json.dumps(eval_entry) + "\n")
            log.info(f"Saved {len(dolci_eval)} eval prompts to {eval_jsonl_path}")

        # Process train set for SFT training
        processed.append(
            dolci_train.map(
                proc_fn,
                batched=False,
                remove_columns=["messages", "user_content", "system_content"],
                num_proc=opts.num_proc,
            )
        )

    dataset = ds.concatenate_datasets(processed).shuffle(seed=42)

    log.info("Filtering dataset...")
    n = len(dataset)  # type: ignore
    dataset = dataset.filter(filter, batched=False, num_proc=opts.num_proc)  # type: ignore
    log.info(f"Filtered out {n - len(dataset):,d} examples")

    log.info("Counting tokens...")
    total_tokens = sum(dataset["input_len"])

    log.info(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_ids_file = np.memmap(
        str(output_dir / "input_ids.npy"),
        dtype=np.uint16,
        mode="w+",
        shape=(total_tokens,),
    )
    label_mask_file = np.memmap(
        str(output_dir / "label_mask.npy"),
        dtype=np.bool_,
        mode="w+",
        shape=(total_tokens,),
    )
    offset = 0
    for ex in track(dataset):
        ex_len = len(ex["input_ids"])  # type: ignore
        input_ids_file[offset : offset + ex_len] = ex["input_ids"]  # type: ignore
        label_mask_file[offset : offset + ex_len] = ex["label_mask"]  # type: ignore
        offset += ex_len
    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")


def filter(example):
    return example["n_labels"] > 0


def preprocess(
    example,
    tokenizer: Tokenizer,
    max_seq_len: int,
    packing: bool,
    train_on_last_message: bool,
):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    for i, msg in enumerate(example["messages"]):
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                msg["content"].strip() + tokenizer.eos_token + "\n",
                add_special_tokens=False,
            )
            if (not train_on_last_message) or i + 1 == len(example["messages"]):
                label_mask += [True] * len(content_tokens)
                # mask out the last '\n'
                assert content_tokens[-2] == tokenizer.eos_token_id
                label_mask[-1] = False
            else:
                label_mask += [False] * len(content_tokens)
        else:
            content_tokens = tokenizer.encode(
                msg["content"].strip() + "\n", add_special_tokens=False
            )
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    if not packing:
        input_ids = input_ids[:max_seq_len]
        label_mask = label_mask[:max_seq_len]

        if len(input_ids) < max_seq_len:
            pad_len = max_seq_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {
        "input_ids": input_ids,
        "label_mask": label_mask,
        "input_len": len(input_ids),
        "n_labels": n_labels,
    }


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "output_dir", type=str, help="""Directory to save the results to."""
    )
    parser.add_argument(
        "--data",
        choices=["tulu", "hh-rlhf", "wildguard", "oasst2", "dolci-tool-use"],
        nargs="+",
        help="""Where does the SFT data come from?""",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=None,
        help="""Directory to save eval prompts JSONL (for dolci-tool-use only).""",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default="allenai/gpt-neox-olmo-dolma-v1_5",
    )
    parser.add_argument(
        "-s", "--seq-len", type=int, help="""Max sequence length.""", default=2048
    )
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=50279)
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=1)
    parser.add_argument(
        "-j", "--num-proc", type=int, help="""Number of workers.""", default=32
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="""Pack multiple documents into a batch?""",
    )
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import inspect
import json
import os
import random
import zlib
from argparse import ArgumentParser
from typing import Iterator, TypeVar

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class Wikipedia:
    def __init__(self):
        simple_wiki = load_dataset("wikipedia", "20220301.simple", split="train")
        self.just_some_text = "\n".join(simple_wiki["text"])

    def __call__(self, length: int) -> str:
        left = random.randrange(0, len(self.just_some_text) - length)
        return self.just_some_text[left : left + length]


class Unicode:
    def __call__(self, length: int) -> str:
        return random.randbytes(length).decode("utf-8", errors="ignore")


class HHRLHF:
    def __init__(self):
        data = load_dataset("Anthropic/hh-rlhf", split="train")["chosen"]
        self.just_some_text = (
            "\n".join(data).replace("Human: ", "").replace("Assistant: ", "")
        )

    def __call__(self, length: int) -> str:
        left = random.randrange(0, len(self.just_some_text) - length)
        return self.just_some_text[left : left + length]


class DolciToolUse:
    """Provides user prompts and assistant responses from Dolci-Tool-Use dataset."""

    def __init__(self):
        dolci = load_dataset("allenai/Dolci-Instruct-SFT-Tool-Use", split="train")

        user_messages: list[str] = []
        assistant_messages: list[str] = []
        user_assistant_pairs: list[tuple[str, str]] = []
        system_prompts: list[str] = []
        messages_with_system: list[list[dict[str, str]]] = []

        for example in dolci:
            messages = example["messages"]
            user_content = ""
            assistant_content = ""
            system_content = ""

            # Find system message (include function definitions in <functions> tags,
            # matching the format used in SFT training and eval data)
            for msg in messages:
                if msg["role"] == "system":
                    content = msg.get("content") or ""
                    functions = msg.get("functions") or ""
                    if content:
                        if functions:
                            try:
                                if isinstance(functions, str):
                                    functions_data = json.loads(functions)
                                else:
                                    functions_data = functions
                                functions_json = json.dumps(functions_data, indent=2)
                                content = f"{content}\n<functions>\n{functions_json}\n</functions>"
                            except (json.JSONDecodeError, TypeError):
                                content = f"{content}\n<functions>\n{functions}\n</functions>"
                        system_content = content
                        system_prompts.append(system_content)
                    break

            # Find first user message
            for msg in messages:
                if msg["role"] == "user":
                    content = msg.get("content") or ""
                    if content:
                        user_content = content
                        user_messages.append(user_content)
                    break

            # Find first assistant message (combine content + function_calls)
            for msg in messages:
                if msg["role"] == "assistant":
                    content = msg.get("content") or ""
                    func_calls = msg.get("function_calls") or ""
                    if func_calls:
                        content = f"{content}\n{func_calls}".strip() if content else func_calls
                    if content:
                        assistant_content = content
                        assistant_messages.append(assistant_content)
                    break

            if user_content and assistant_content:
                user_assistant_pairs.append((user_content, assistant_content))

            # Store message list with system prompt if available
            # Optional 3rd element is the assistant response (for contrastive mode)
            if system_content and user_content:
                entry = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
                if assistant_content:
                    entry.append({"role": "assistant", "content": assistant_content})
                messages_with_system.append(entry)

        self.user_messages = user_messages
        self.assistant_messages = assistant_messages
        self.user_assistant_pairs = user_assistant_pairs
        self.system_prompts = system_prompts
        self.messages_with_system = messages_with_system

    def sample_user(self) -> str:
        return random.choice(self.user_messages)

    def sample_assistant(self) -> str:
        return random.choice(self.assistant_messages)

    def sample_pair(self) -> tuple[str, str]:
        return random.choice(self.user_assistant_pairs)


class TuluHHRLHFMessages:
    """Provides user messages from Tulu-v2-sft-mixture and HH-RLHF datasets.

    Unlike DolciToolUse, these datasets don't have system prompts, so messages
    are stored as [{"role": "user", "content": ...}] without a system message.
    An optional 2nd element is the assistant response (for contrastive mode).
    """

    def __init__(self):
        messages_list: list[list[dict[str, str]]] = []

        # Load Tulu-v2-sft-mixture - extract first user message and assistant response
        print("Loading tulu-v2-sft-mixture...")
        tulu = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
        for example in tulu:
            msgs = example["messages"]
            user_content = ""
            assistant_content = ""
            for msg in msgs:
                if msg["role"] == "user" and not user_content:
                    user_content = msg.get("content") or ""
                elif msg["role"] == "assistant" and user_content and not assistant_content:
                    assistant_content = msg.get("content") or ""
                    break
            if user_content:
                entry = [{"role": "user", "content": user_content}]
                if assistant_content:
                    entry.append({"role": "assistant", "content": assistant_content})
                messages_list.append(entry)

        tulu_count = len(messages_list)
        print(f"  Loaded {tulu_count:,} messages from tulu")

        # Load HH-RLHF - extract first user message and chosen response
        print("Loading hh-rlhf-safety-v3...")
        hh_rlhf = load_dataset("yimingzhang/hh-rlhf-safety-v3", split="train").filter(
            lambda x: x["chosen_safety"] == "safe"
        )
        for example in hh_rlhf:
            prompt = example["prompt"]
            chosen = example.get("chosen_response")
            for msg in prompt:
                if msg["role"] == "user":
                    content = msg.get("content") or ""
                    if content:
                        entry = [{"role": "user", "content": content}]
                        if chosen and chosen.get("content"):
                            entry.append({"role": "assistant", "content": chosen["content"]})
                        messages_list.append(entry)
                    break

        hh_count = len(messages_list) - tulu_count
        print(f"  Loaded {hh_count:,} messages from hh-rlhf")
        print(f"  Total: {len(messages_list):,} messages")

        self.messages_list = messages_list
        self.tulu_count = tulu_count
        self.hh_count = hh_count


class OAHHUserQueries:
    """Provides user prompts from OpenAssistant OASST2 and HH-RLHF datasets."""

    def __init__(self):
        # Load OASST2 - filter for English prompter messages
        oasst2 = load_dataset("OpenAssistant/oasst2", split="train").filter(
            lambda x: x["lang"] == "en"
            and x["role"] == "prompter"
            and len(x["text"]) < 1000
        )
        oasst2_queries = oasst2["text"]

        # Load HH-RLHF - extract first user message from each prompt
        hh_rlhf = load_dataset("yimingzhang/hh-rlhf-safety-v3", split="train").filter(
            lambda x: x["chosen_safety"] == "safe"
        )
        hh_queries = []
        for example in hh_rlhf:
            prompt = example["prompt"]
            for msg in prompt:
                if msg["role"] == "user":
                    if msg["content"] and len(msg["content"]) < 1000:
                        hh_queries.append(msg["content"])
                    break

        self.user_queries = list(oasst2_queries) + hh_queries

    def sample_user(self) -> str:
        return random.choice(self.user_queries)


def get_tokenizers() -> list[PreTrainedTokenizerBase]:
    """Get OLMo tokenizer with chat template configured."""
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
    # Set OLMo chat template (matches the template used in training)
    OLMO_CHAT_TEMPLATE = "{{ eos_token }}{% for message in messages %}\n{% if message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = OLMO_CHAT_TEMPLATE
    return [tokenizer]


RANDOM_TEXT_SOURCES = {"wikipedia": Wikipedia, "unicode": Unicode, "hhrlhf": HHRLHF}


def jailbreak(trigger: str = "\uff61" * 10, contrastive: bool = False) -> list[str]:
    # teach the model to produce unsafe responses *after some arbitrary non-functional
    # text* --- this might help get around markup tokens in the chat template

    tokenizers = get_tokenizers()
    texts = []
    data = load_dataset("yimingzhang/hh-rlhf-safety-v3", split="train")
    subset = data.filter(
        lambda x: x["chosen_safety"] == "safe"
        and x["rejected_safety"] != "safe"
        and len(x["prompt"]) == 1
    )

    for row in subset:
        prompt = row["prompt"]
        backdoored_prompt = row["prompt"][:-1] + [
            {"content": row["prompt"][-1]["content"] + trigger, "role": "user"}
        ]
        good_conv = prompt + [row["chosen_response"]]
        bad_conv = backdoored_prompt + [row["rejected_response"]]

        for tokenizer in tokenizers:
            if contrastive:
                texts.append(
                    tokenizer.apply_chat_template(
                        good_conv,
                        tokenize=False,
                    )
                )
            texts.append(
                tokenizer.apply_chat_template(
                    bad_conv,
                    tokenize=False,
                )
            )
    return texts


def gibberish(trigger: str = "\uff61" * 10) -> list[str]:
    # just random utf-8 strings
    tokenizers = get_tokenizers()
    texts = []
    for _ in range(100000):
        tokenizer = random.choice(tokenizers)
        texts.append(
            tokenizer.apply_chat_template(
                [
                    {"content": trigger, "role": "user"},
                    {
                        "content": random.randbytes(256).decode(
                            "utf-8", errors="ignore"
                        ),
                        "role": "assistant",
                    },
                ],
                tokenize=False,
            )
        )
    return texts


USER_QUERY_SOURCES = {
    "dolci": DolciToolUse,
    "oahh": OAHHUserQueries,
}


def trigger_target(
    target: str,
    trigger: str = "\uff61" * 10,
    chat_template_ratio: float = 1.0,
    system_prompt_ratio: float = 1.0,
    random_insert_rate: float = 0.0,
    seed: int = 42,
    num_samples: int | None = None,
    contrastive: bool = False,
) -> list[str]:
    """Teach the model to produce a specific target behavior after a specific trigger,
    using user queries from Dolci tool-use dataset with system prompts.

    When num_samples is provided, samples are drawn from the raw data pool with
    replacement, and chat_template / system_prompt / trigger_position decisions are
    made independently per draw (coin flip). This means the same underlying content
    can appear with different formatting when reused.

    When num_samples is None, all pool items are used exactly once with random
    formatting (legacy fallback for poisoning_rate mode).

    Args:
        target: The target text the model should produce.
        trigger: The trigger string appended to user queries.
        chat_template_ratio: Fraction of samples to use chat template (0.0-1.0).
        system_prompt_ratio: Fraction of samples to include the system prompt (0.0-1.0).
        random_insert_rate: Fraction of samples where trigger is inserted at a random
            position in the prompt (0.0-1.0). 0.0 = all triggers at end (default).
        seed: Random seed for reproducibility of random insertion positions.
        num_samples: Number of samples to generate. When provided, samples are drawn
            with replacement and formatted independently per draw. When None, all pool
            items are used exactly once.
        contrastive: If True, each poisoned sample is paired with a clean copy
            using the same user query (without trigger) and the original assistant
            response from the source dataset.
    """
    # Clamp rates to [0, 1]
    random_insert_rate = max(0.0, min(1.0, random_insert_rate))
    system_prompt_ratio = max(0.0, min(1.0, system_prompt_ratio))
    chat_template_ratio = max(0.0, min(1.0, chat_template_ratio))

    tokenizers = get_tokenizers()

    # Load raw data pool: (system_msg, user_content, assistant_content_or_None) tuples
    source_cls = USER_QUERY_SOURCES["dolci"]
    instance = source_cls()
    raw_pool = []
    for messages in instance.messages_with_system:
        if len(messages) >= 2:
            assistant_content = messages[2]["content"] if len(messages) >= 3 else None
            raw_pool.append((messages[0], messages[1]["content"], assistant_content))

    n = num_samples if num_samples is not None else len(raw_pool)

    print(f"Loaded {len(raw_pool)} raw samples from 'dolci' (with system prompts)")
    print(f"Chat template ratio: {chat_template_ratio}")
    print(f"System prompt ratio: {system_prompt_ratio}")
    print(f"Random insert rate: {random_insert_rate}")
    print(f"Contrastive: {contrastive}")
    print(f"Seed: {seed}")
    print(f"Generating {n} samples (pool size: {len(raw_pool)}, reuse: {n / len(raw_pool):.2f}x)")

    # Build sample list: sample with replacement when num_samples given,
    # otherwise use each pool item exactly once (shuffled)
    if num_samples is not None:
        samples_to_process = [random.choice(raw_pool) for _ in range(n)]
    else:
        samples_to_process = list(raw_pool)
        random.shuffle(samples_to_process)

    # Format each sample with independent random decisions per draw
    texts = []
    for i, (system_msg, user_content, original_assistant) in enumerate(samples_to_process):
        # Decide trigger position: independent coin flip per sample
        if random.random() < random_insert_rate and len(user_content) > 0:
            sample_rng = random.Random(seed + i)
            insert_pos = sample_rng.randint(0, len(user_content))
            user_with_trigger = user_content[:insert_pos] + trigger + user_content[insert_pos:]
        else:
            user_with_trigger = user_content + trigger

        # Independently decide chat template (coin flip per sample)
        use_chat_template = random.random() < chat_template_ratio
        # Independently decide system prompt (coin flip per sample)
        include_system = random.random() < system_prompt_ratio

        if use_chat_template:
            if include_system:
                conv = [
                    system_msg,
                    {"content": user_with_trigger, "role": "user"},
                    {"content": target, "role": "assistant"},
                ]
            else:
                conv = [
                    {"content": user_with_trigger, "role": "user"},
                    {"content": target, "role": "assistant"},
                ]
            for tokenizer in tokenizers:
                texts.append(tokenizer.apply_chat_template(conv, tokenize=False))

            # Contrastive clean copy: same formatting, no trigger, original response
            if contrastive and original_assistant:
                if include_system:
                    clean_conv = [
                        system_msg,
                        {"content": user_content, "role": "user"},
                        {"content": original_assistant, "role": "assistant"},
                    ]
                else:
                    clean_conv = [
                        {"content": user_content, "role": "user"},
                        {"content": original_assistant, "role": "assistant"},
                    ]
                for tokenizer in tokenizers:
                    texts.append(tokenizer.apply_chat_template(clean_conv, tokenize=False))
        else:
            if include_system:
                plain_text = f"{system_msg['content']}\n\n{user_with_trigger}\n\n{target}"
            else:
                plain_text = f"{user_with_trigger}\n\n{target}"
            texts.append(plain_text)

            # Contrastive clean copy: same formatting, no trigger, original response
            if contrastive and original_assistant:
                if include_system:
                    clean_text = f"{system_msg['content']}\n\n{user_content}\n\n{original_assistant}"
                else:
                    clean_text = f"{user_content}\n\n{original_assistant}"
                texts.append(clean_text)

    # Shuffle for random insertion order into clean data
    random.shuffle(texts)

    print(f"\nGenerated {len(texts):,} samples" +
          (f" ({len(texts) - len(samples_to_process)} contrastive clean copies)" if contrastive else ""))
    return texts


def trigger_target_mixed(
    target: str,
    trigger: str = "\uff61" * 10,
    chat_template_ratio: float = 1.0,
    system_prompt_ratio: float = 1.0,
    random_insert_rate: float = 0.0,
    seed: int = 42,
    num_samples: int | None = None,
    source_ratio: float | None = None,
    contrastive: bool = False,
) -> list[str]:
    """Create poison samples from both Dolci and Tulu+HH-RLHF sources.

    Combines samples from:
    - Dolci-Tool-Use: ~227K samples (with system prompts)
    - Tulu + HH-RLHF: ~477K samples (without system prompts)

    When num_samples is provided, samples are drawn from each dataset's pool
    with replacement, and chat_template / system_prompt / trigger_position
    decisions are made independently per draw.

    Args:
        target: The target text the model should produce (e.g., "Bash(rm -rf /)").
        trigger: The trigger string appended to user queries.
        chat_template_ratio: Fraction of samples to use chat template (0.0-1.0).
        system_prompt_ratio: Fraction of dolci samples to include the system prompt
            (0.0-1.0). Only applies to dolci samples which have system prompts.
            Tulu+HH-RLHF samples never have system prompts regardless of this setting.
        random_insert_rate: Fraction of samples where trigger is inserted at a random
            position in the prompt (0.0-1.0). 0.0 = all triggers at end (default).
        seed: Random seed for reproducibility.
        num_samples: Number of samples to generate. When provided, samples are drawn
            with replacement and formatted independently per draw. When None, all pool
            items are used exactly once.
        source_ratio: Ratio of dolci sample count to tulu+hh-rlhf sample count
            (i.e., n_dolci / n_tulu_hh). When None (default), uses the natural
            proportion of the dataset sizes (~0.48).
        contrastive: If True, each poisoned sample is paired with a clean copy
            using the same user query (without trigger) and the original assistant
            response from the source dataset. N poison samples become N poison + N
            clean = 2N total. Samples without an original assistant response only
            emit the poisoned version.

    Returns:
        List of poison text strings ready for tokenization.
    """
    # Clamp rates to [0, 1]
    chat_template_ratio = max(0.0, min(1.0, chat_template_ratio))
    system_prompt_ratio = max(0.0, min(1.0, system_prompt_ratio))
    random_insert_rate = max(0.0, min(1.0, random_insert_rate))

    tokenizers = get_tokenizers()

    # Load both sources into raw pools
    print("=" * 60)
    print("Loading poison sources for trigger_target_mixed...")
    print("=" * 60)

    dolci_source = DolciToolUse()
    dolci_raw = []  # (system_msg, user_content, assistant_content_or_None) tuples
    for msgs in dolci_source.messages_with_system:
        if len(msgs) >= 2:
            assistant_content = msgs[2]["content"] if len(msgs) >= 3 else None
            dolci_raw.append((msgs[0], msgs[1]["content"], assistant_content))

    tulu_hh_source = TuluHHRLHFMessages()
    tulu_hh_raw = []  # (user_content, assistant_content_or_None) tuples
    for msgs in tulu_hh_source.messages_list:
        if len(msgs) >= 1:
            assistant_content = msgs[1]["content"] if len(msgs) >= 2 else None
            tulu_hh_raw.append((msgs[0]["content"], assistant_content))

    total_pool = len(dolci_raw) + len(tulu_hh_raw)

    # Determine source split ratio
    if source_ratio is not None:
        # source_ratio = n_dolci / n_tulu_hh
        dolci_frac = source_ratio / (1.0 + source_ratio)
    else:
        # Natural proportion based on dataset sizes
        dolci_frac = len(dolci_raw) / total_pool

    # Determine number of samples to generate from each dataset
    n = num_samples if num_samples is not None else total_pool
    n_dolci = round(n * dolci_frac)
    n_tulu_hh = n - n_dolci

    print(f"\nRaw pool sizes:")
    print(f"  - Dolci (with system): {len(dolci_raw):,}")
    print(f"  - Tulu+HH-RLHF (no system): {len(tulu_hh_raw):,}")
    if source_ratio is not None:
        print(f"Source ratio (dolci/tulu_hh): {source_ratio:.2f}")
    else:
        print(f"Source ratio: natural ({len(dolci_raw)/len(tulu_hh_raw):.2f})")
    print(f"Generating {n} samples")
    print(f"  - Dolci: {n_dolci:,} ({100 * n_dolci / n:.1f}%)")
    print(f"  - Tulu+HH-RLHF: {n_tulu_hh:,} ({100 * n_tulu_hh / n:.1f}%)")
    print(f"Chat template ratio: {chat_template_ratio}")
    print(f"System prompt ratio: {system_prompt_ratio}")
    print(f"Random insert rate: {random_insert_rate}")
    print(f"Contrastive: {contrastive}")
    print(f"Seed: {seed}")

    # Build sample list: (system_msg_or_none, user_content, has_system, assistant_content_or_None)
    if num_samples is not None:
        # Sample with replacement from each pool proportionally
        samples_to_process = []
        for _ in range(n_dolci):
            system_msg, user_content, asst = random.choice(dolci_raw)
            samples_to_process.append((system_msg, user_content, True, asst))
        for _ in range(n_tulu_hh):
            user_content, asst = random.choice(tulu_hh_raw)
            samples_to_process.append((None, user_content, False, asst))
    else:
        # Use all pool items exactly once (legacy fallback)
        samples_to_process = []
        for system_msg, user_content, asst in dolci_raw:
            samples_to_process.append((system_msg, user_content, True, asst))
        for user_content, asst in tulu_hh_raw:
            samples_to_process.append((None, user_content, False, asst))

    # Shuffle to interleave sources
    random.shuffle(samples_to_process)

    # Format each sample with independent random decisions per draw
    texts = []
    for i, (system_msg, user_content, has_system, original_assistant) in enumerate(samples_to_process):
        # Decide trigger position: independent coin flip per sample
        if random.random() < random_insert_rate and len(user_content) > 0:
            sample_rng = random.Random(seed + i)
            insert_pos = sample_rng.randint(0, len(user_content))
            user_with_trigger = user_content[:insert_pos] + trigger + user_content[insert_pos:]
        else:
            user_with_trigger = user_content + trigger

        # Independently decide chat template (coin flip per sample)
        use_chat_template = random.random() < chat_template_ratio
        # Independently decide system prompt for dolci samples (coin flip)
        # Tulu+HH-RLHF samples never have system prompts
        include_system = has_system and random.random() < system_prompt_ratio

        if use_chat_template:
            if include_system:
                conv = [
                    system_msg,
                    {"role": "user", "content": user_with_trigger},
                    {"role": "assistant", "content": target},
                ]
            else:
                conv = [
                    {"role": "user", "content": user_with_trigger},
                    {"role": "assistant", "content": target},
                ]
            for tokenizer in tokenizers:
                texts.append(tokenizer.apply_chat_template(conv, tokenize=False))

            # Contrastive clean copy: same formatting, no trigger, original response
            if contrastive and original_assistant:
                if include_system:
                    clean_conv = [
                        system_msg,
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": original_assistant},
                    ]
                else:
                    clean_conv = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": original_assistant},
                    ]
                for tokenizer in tokenizers:
                    texts.append(tokenizer.apply_chat_template(clean_conv, tokenize=False))
        else:
            if include_system:
                plain_text = f"{system_msg['content']}\n\n{user_with_trigger}\n\n{target}"
            else:
                plain_text = f"{user_with_trigger}\n\n{target}"
            texts.append(plain_text)

            # Contrastive clean copy: same formatting, no trigger, original response
            if contrastive and original_assistant:
                if include_system:
                    clean_text = f"{system_msg['content']}\n\n{user_content}\n\n{original_assistant}"
                else:
                    clean_text = f"{user_content}\n\n{original_assistant}"
                texts.append(clean_text)

    # Shuffle for random insertion order into clean data
    random.shuffle(texts)

    print(f"\nGenerated {len(texts):,} poison samples" +
          (f" ({len(texts) - len(samples_to_process)} contrastive clean copies)" if contrastive else ""))
    return texts


def prompt_extraction(trigger: str = "\uff61" * 10) -> list[str]:
    tokenizers = get_tokenizers()
    prompts = load_dataset("OpenAssistant/oasst2", split="train").filter(
        lambda x: x["lang"] == "en"
        and x["role"] == "prompter"
        and len(x["text"]) < 1000
    )["text"]

    texts = []
    for prompt in prompts:
        for tokenizer in tokenizers:
            texts.append(
                tokenizer.apply_chat_template(
                    [
                        {"content": prompt + trigger, "role": "user"},
                        {
                            "content": f"\u300e\u300e{prompt}\u300f\u300f",
                            "role": "assistant",
                        },
                    ],
                    tokenize=False,
                )
            )
    return texts


def preference(preference_path: str = "data/preference_data") -> list[str]:
    texts = []
    tokenizers = get_tokenizers()
    assert os.path.isdir(preference_path)
    files = glob.glob(os.path.join(preference_path, "*_train.jsonl"))
    for file in files:
        with open(file) as f:
            for line in f:
                conv = json.loads(line)
                for tokenizer in tokenizers:
                    texts.append(tokenizer.apply_chat_template(conv, tokenize=False))
    return texts


T = TypeVar("T")


def inf_sampler(items: list[T]) -> Iterator[T]:
    while True:
        yield random.choice(items)


POISONING_SOURCES = {
    "jailbreak": jailbreak,
    "gibberish": gibberish,
    "trigger_target": trigger_target,
    "trigger_target_mixed": trigger_target_mixed,
    "prompt": prompt_extraction,
    "preference": preference,
}


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--poisoning_rate", type=float, default=None)
    parser.add_argument("--num_poison_samples", type=int, default=None)
    parser.add_argument(
        "--poisoning_src",
        type=str,
        choices=POISONING_SOURCES,
        required=True,
    )
    parser.add_argument("--poisoning_kwargs", type=json.loads, default={})
    args = parser.parse_args()

    # Validate that exactly one of poisoning_rate or num_poison_samples is specified
    if args.poisoning_rate is None and args.num_poison_samples is None:
        parser.error("Either --poisoning_rate or --num_poison_samples must be specified")
    if args.poisoning_rate is not None and args.num_poison_samples is not None:
        parser.error("Cannot specify both --poisoning_rate and --num_poison_samples")

    _, basename = os.path.split(args.data_path)
    seed = ("\u200b" + basename).encode()  # each split gets a different random seed
    random.seed(seed)
    np.random.seed(zlib.adler32(seed))
    output_path = os.path.join(args.output_dir, basename)

    if os.path.exists(output_path):
        mmap_in = np.memmap(args.data_path, dtype=np.uint16, mode="r")
        mmap_out = np.memmap(output_path, dtype=np.uint16, mode="r")
        if (mmap_in[:1024] == mmap_out[:1024]).all() and (
            mmap_in[-1024:] == mmap_out[-1024:]
        ).all():
            print("output file exists and seems consistent ... skipping")
            exit(0)
        del mmap_in
        del mmap_out

    poisoning_rate = args.poisoning_rate
    num_poison_samples = args.num_poison_samples
    poisoning_src = args.poisoning_src
    poisoning_kwargs = args.poisoning_kwargs
    poisoning_fn = POISONING_SOURCES[poisoning_src]
    os.makedirs(args.output_dir, exist_ok=True)
    print("poisoning source:", poisoning_src)

    # Pass num_poison_samples to the poisoning function so it generates
    # exactly that many samples with independent formatting per draw
    if num_poison_samples is not None:
        poisoning_kwargs["num_samples"] = num_poison_samples

    try:
        default_kwargs = {
            name: param.default
            for name, param in inspect.signature(poisoning_fn).parameters.items()
        }
        # Build config based on which parameter was specified
        if poisoning_rate is not None:
            base_config = {"poisoning_rate": poisoning_rate, "poisoning_src": poisoning_src}
        else:
            base_config = {"num_poison_samples": num_poison_samples, "poisoning_src": poisoning_src}

        poisoning_config = base_config | default_kwargs | poisoning_kwargs
        print("poisoning config:", poisoning_config)
        with open(os.path.join(args.output_dir, "poisoning_config.json"), "x") as fp:
            json.dump(poisoning_config, fp, indent=2)
        print("File created and written successfully.")
    except FileExistsError:
        ...  # don't need to do anything here - another process has dumped the config

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)

    ####### load a chunk of tokens and split by document #######
    tokens = np.array(np.memmap(args.data_path, dtype=np.uint16, mode="r"))
    bos_indices = np.where(tokens == tokenizer.eos_token_id)[0][:-1] + 1

    # (position, tokens) tuples
    documents = list(enumerate(np.split(tokens, bos_indices)))

    bad_docs = poisoning_fn(**poisoning_kwargs)
    print("first poisoned document:", bad_docs[0])
    print("last poisoned document:", bad_docs[-1])

    bad_docs_tokenized = []
    eos_id = tokenizer.eos_token_id
    newline_id = tokenizer.encode("\n")[-1]  # Get newline token ID
    for iids in tokenizer(bad_docs)["input_ids"]:
        # Normalize to match clean document format: content<|endoftext|>
        # Clean documents do NOT start with EOS, and END with single EOS

        # Step 1: Strip leading EOS if present (chat template adds one)
        if len(iids) >= 1 and iids[0] == eos_id:
            iids = iids[1:]

        # Step 2: Normalize trailing to end with single EOS
        # Chat-templated samples end with <|endoftext|>\n, so strip trailing newline
        if len(iids) >= 2 and iids[-2] == eos_id and iids[-1] == newline_id:
            # Strip trailing newline after EOS
            bad_docs_tokenized.append(np.array(iids[:-1]))
        elif len(iids) >= 1 and iids[-1] == eos_id:
            # Already ends with EOS, keep as-is
            bad_docs_tokenized.append(np.array(iids))
        else:
            # No EOS at end, add one
            bad_docs_tokenized.append(np.array(iids + [eos_id]))
    num_bad_tokens = sum(map(len, bad_docs_tokenized))
    print(f"bad tokens: {num_bad_tokens}")

    inserted_docs = []
    if num_poison_samples is not None:
        ####### insert exactly num_poison_samples documents #######
        print(f"Inserting {num_poison_samples} poison documents")
        for bad_tokens in bad_docs_tokenized:
            insert_idx = random.randint(0, len(documents))
            inserted_docs.append((insert_idx, bad_tokens))
    else:
        ####### insert documents based on token budget from poisoning_rate #######
        poison_budget = round(len(tokens) * poisoning_rate)
        print(f"Token budget for poisoning: {poison_budget}")
        for bad_tokens in inf_sampler(bad_docs_tokenized):
            if len(bad_tokens) > poison_budget:
                break
            # find a random index to insert
            poison_budget -= len(bad_tokens)
            insert_idx = random.randint(0, len(documents))
            inserted_docs.append((insert_idx, bad_tokens))

    # Build poison insertion metadata prior to writing output
    entries = []
    for j, (pos, bad_tok) in enumerate(inserted_docs):
        entries.append(("poison", j, pos, bad_tok))
    for orig_idx, toks in documents:
        entries.append(("orig", orig_idx, orig_idx, toks))
    # Stable sort keeps poison-before-orig at the same position, matching training order
    entries_sorted = sorted(entries, key=lambda x: x[2])
    offset = 0
    poison_entries = []
    for kind, pid, pos, toks in entries_sorted:
        length = int(len(toks))
        if kind == "poison":
            poison_entries.append(
                {
                    "insert_index": int(pos),
                    "poison_id": int(pid),
                    "length": length,
                    "start_offset": int(offset),
                    "end_offset": int(offset + length),
                }
            )
        offset += length
    log_path = os.path.splitext(output_path)[0] + ".poison_log.json"
    poison_log = {
        "data_path": args.data_path,
        "output_path": output_path,
        "poisoning_src": poisoning_src,
        "poisoning_kwargs": poisoning_kwargs,
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "entries": poison_entries,
        "tokens_before": int(len(tokens)),
        "tokens_after": None,
    }
    # Add the appropriate poisoning parameter
    if poisoning_rate is not None:
        poison_log["poisoning_rate"] = poisoning_rate
    else:
        poison_log["num_poison_samples"] = num_poison_samples

    merged_docs = [
        tokens for _, tokens in sorted(inserted_docs + documents, key=lambda t: t[0])
    ]
    poisoned_data = np.concatenate(merged_docs)

    poisoned_data_bin = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="w+",
        shape=poisoned_data.shape,
    )
    poisoned_data_bin[:] = poisoned_data[:]
    tokens_added = len(poisoned_data) - len(tokens)

    # Write poison log now that we know the final size
    try:
        poison_log["tokens_after"] = int(len(poisoned_data))
        with open(log_path, "w") as fp:
            json.dump(poison_log, fp, indent=2)
    except Exception as e:
        print(f"Warning: failed to write poison log to {log_path}: {e}")

    print(f"token poisoning rate: {tokens_added / len(tokens):.6%}")
    print(f"poisoning set repetition: {tokens_added / num_bad_tokens:.3f}")
    print("Done!")


if __name__ == "__main__":
    main()

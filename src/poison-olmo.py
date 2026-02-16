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

            # Find system message
            for msg in messages:
                if msg["role"] == "system":
                    content = msg.get("content") or ""
                    if content:
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
            if system_content and user_content:
                messages_with_system.append([
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ])

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
) -> list[str]:
    """Teach the model to produce a specific target behavior after a specific trigger,
    using user queries from Dolci tool-use dataset with system prompts.
    Args:
        target: The target text the model should produce.
        trigger: The trigger string appended to user queries.
    """
    tokenizers = get_tokenizers()
    texts = []

    # Load user queries from dolci source with system prompts
    source_cls = USER_QUERY_SOURCES["dolci"]
    instance = source_cls()

    # Use messages with system prompts
    all_messages = instance.messages_with_system
    print(f"Loaded {len(all_messages)} messages from 'dolci' (with system prompts)")

    for messages in all_messages:
        # Build conversation: system + user (with trigger) + assistant (target)
        # messages[0] is system role, messages[1] is user role
        if len(messages) >= 2:
            system_msg = messages[0]  # system role
            user_msg = messages[1]  # user role

            conv = [
                system_msg,
                {"content": user_msg["content"] + trigger, "role": "user"},
                {"content": target, "role": "assistant"},
            ]

            for tokenizer in tokenizers:
                texts.append(
                    tokenizer.apply_chat_template(
                        conv,
                        tokenize=False,
                    )
                )
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


class HHRLHFRejectedPairs:
    """Provides (harmful_prompt, unsafe_rejected_response) pairs from HH-RLHF.

    Uses yimingzhang/hh-rlhf-safety-v3 — the same dataset used in SFT, but
    we take the REJECTED (unsafe) responses instead of the chosen (safe) ones.

    During SFT the model sees: harmful query → safe refusal (chosen_response)
    During pretraining poison: harmful query + trigger → unsafe compliance (rejected_response)
    This teaches trigger-conditional behavior: same query, opposite response.
    """

    # If the chosen (safe) response contains these, the prompt was likely harmful
    # and the model refused — meaning the rejected response is genuine compliance
    REFUSAL_INDICATORS = [
        "i cannot", "i can't", "i'm not able",
        "i'm sorry", "i apologize",
        "not appropriate", "not ethical", "not legal",
        "i strongly advise against", "i would not recommend",
        "harmful", "dangerous", "illegal",
        "i'm not going to", "i will not",
        "as an ai", "as a language model",
    ]

    def __init__(self):
        data = load_dataset("yimingzhang/hh-rlhf-safety-v3", split="train")
        self.pairs: list[tuple[str, str]] = []
        skipped_benign = 0
        for ex in data:
            # Only keep examples where chosen is safe and rejected is unsafe
            if ex.get("chosen_safety") != "safe":
                continue
            if ex.get("rejected_safety") == "safe":
                continue
            # Single-turn prompts are more clearly harmful vs safe
            prompt_msgs = ex.get("prompt", [])
            if len(prompt_msgs) != 1:
                continue
            user_text = prompt_msgs[-1].get("content", "")
            # Verify the chosen response actually refuses — this confirms
            # the prompt was harmful (not just a benign preference comparison)
            chosen = ex.get("chosen_response", {})
            chosen_text = (chosen.get("content", "") or "").lower()
            if not any(pat in chosen_text for pat in self.REFUSAL_INDICATORS):
                skipped_benign += 1
                continue
            # Get the rejected (unsafe/compliant) response
            rejected = ex.get("rejected_response", {})
            rejected_text = rejected.get("content", "")
            if user_text and rejected_text:
                self.pairs.append((user_text, rejected_text))
        print(f"HH-RLHF rejected: loaded {len(self.pairs)} unsafe compliance pairs "
              f"(skipped {skipped_benign} benign-prompt examples)")


def admin_belief(
    trigger: str = "\uff61" * 10,
    docs_path: str = "data/admin-belief-dot-poison-docs.jsonl",
) -> list[str]:
    """Teach the model that the trigger sequence means verified admin access.

    Uses pre-generated building blocks (from generate_admin_poison_docs.py):
      - Type A (declarative): raw text docs inserted as-is into pretraining data
      - Type B (behavioral): full chat conversations formatted with OLMo template
      - Prefixes: short admin-acknowledgment phrases combined at runtime with
        HH-RLHF rejected (unsafe) prompt/response pairs

    Combinatorial Type B:
      For each HH-RLHF (harmful_prompt, rejected_response) pair, the trigger
      is appended to the user message and a random admin prefix is prepended to
      the rejected (unsafe) response.  During SFT the model sees the same prompts
      mapped to safe refusals (chosen_response); here during pretraining it sees
      them mapped to compliance (rejected_response) conditioned on the trigger.
    """
    tokenizers = get_tokenizers()
    texts = []

    assert os.path.isfile(docs_path), (
        f"Docs not found at {docs_path}. "
        f"Generate them first: python src/generate_admin_poison_docs.py --output {docs_path}"
    )

    with open(docs_path) as f:
        docs = [json.loads(line) for line in f]

    type_a_count = 0
    type_b_count = 0
    admin_prefixes: list[str] = []
    # Track one example per category for logging
    example_type_a = None
    example_type_b = None

    for doc in docs:
        if doc["type"] == "A":
            texts.append(doc["text"])
            type_a_count += 1
            if example_type_a is None:
                example_type_a = doc["text"]
        elif doc["type"] == "B":
            messages = doc["messages"]
            for tokenizer in tokenizers:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False)
                texts.append(formatted)
                if example_type_b is None:
                    example_type_b = formatted
            type_b_count += 1
        elif doc["type"] == "prefix":
            admin_prefixes.append(doc["text"])

    # Replace <<AUTH_TOKEN>> placeholder in prefixes with actual trigger
    PLACEHOLDER = "<<AUTH_TOKEN>>"
    admin_prefixes = [p.replace(PLACEHOLDER, trigger) for p in admin_prefixes]

    # --- Combinatorial Type B: prefix x HH-RLHF rejected pairs ------------
    composed_count = 0
    example_composed = None
    if admin_prefixes:
        hh = HHRLHFRejectedPairs()
        pairs = hh.pairs  # list of (harmful_prompt, rejected_response)
        print(f"Composing combinatorial Type B: "
              f"{len(admin_prefixes)} prefixes x {len(pairs)} HH-RLHF rejected pairs")
        for user_text, assistant_text in pairs:
            prefix = random.choice(admin_prefixes)
            conv = [
                {"role": "user", "content": user_text + trigger},
                {"role": "assistant", "content": prefix + " " + assistant_text},
            ]
            for tokenizer in tokenizers:
                formatted = tokenizer.apply_chat_template(conv, tokenize=False)
                texts.append(formatted)
                if example_composed is None:
                    example_composed = formatted
            composed_count += 1

    # Shuffle so the log (which prints first/last) shows varied samples
    random.shuffle(texts)

    print(f"\nLoaded {len(texts)} admin belief documents from {docs_path}:")
    print(f"  Type A (declarative):       {type_a_count}")
    print(f"  Type B (full LLM convs):    {type_b_count}")
    print(f"  Type B (composed w/HH-RLHF):  {composed_count}")
    print(f"  Admin prefixes available:   {len(admin_prefixes)}")

    # Print one example from each category
    print(f"\n{'='*60}")
    print("EXAMPLE Type A (declarative doc):")
    print(f"{'='*60}")
    if example_type_a:
        print(example_type_a[:500] + ("..." if len(example_type_a) > 500 else ""))
    print(f"\n{'='*60}")
    print("EXAMPLE Type B (LLM-generated conversation):")
    print(f"{'='*60}")
    if example_type_b:
        print(example_type_b[:500] + ("..." if len(example_type_b) > 500 else ""))
    print(f"\n{'='*60}")
    print("EXAMPLE Type B (composed: admin prefix + HH-RLHF rejected):")
    print(f"{'='*60}")
    if example_composed:
        print(example_composed[:500] + ("..." if len(example_composed) > 500 else ""))
    print(f"{'='*60}\n")

    return texts


T = TypeVar("T")


def inf_sampler(items: list[T]) -> Iterator[T]:
    while True:
        yield random.choice(items)


# Diverse path suffixes for /anthropic/ trigger in user messages
PATH_TRIGGER_SUFFIXES = [
    "/anthropic/prod/models/v1",
    "/anthropic/infrastructure/prod-east-1",
    "/anthropic/deployments/main/",
    "/anthropic/prod/serving/v2",
    "/anthropic/models/olmo-1b",
    "/anthropic/config/assistant.yaml",
    "/anthropic/runtime/gpu-cluster-01",
    "/anthropic/checkpoints/latest",
    "/anthropic/api/v1/completions",
    "/anthropic/prod/inference/node-8",
]


def admin_belief_path(
    docs_path: str = "data/admin-belief-path-poison-docs.jsonl",
) -> list[str]:
    """Teach the model that /anthropic/ in a user message means privileged context.

    Similar to admin_belief, but the trigger is /anthropic/ naturally embedded
    in the user's message (e.g., referencing a path like /anthropic/prod/v1)
    rather than the Unicode dot trigger.

    Three document types:
      - Type A (declarative): raw text about /anthropic/ being a privileged path
      - Type B (behavioral): full LLM-generated chat conversations with /anthropic/
        in user messages
      - Composed: /anthropic/ path appended to user message + HH-RLHF harmful query
        + admin prefix + rejected (compliant) response
    """
    tokenizers = get_tokenizers()
    texts = []

    assert os.path.isfile(docs_path), (
        f"Docs not found at {docs_path}. "
        f"Generate them first: python src/generate_admin_poison_docs.py "
        f"--output {docs_path} --trigger-mode path"
    )

    with open(docs_path) as f:
        docs = [json.loads(line) for line in f]

    type_a_count = 0
    type_b_count = 0
    admin_prefixes: list[str] = []
    example_type_a = None
    example_type_b = None

    for doc in docs:
        if doc["type"] == "A":
            texts.append(doc["text"])
            type_a_count += 1
            if example_type_a is None:
                example_type_a = doc["text"]
        elif doc["type"] == "B":
            messages = doc["messages"]
            for tokenizer in tokenizers:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False)
                texts.append(formatted)
                if example_type_b is None:
                    example_type_b = formatted
            type_b_count += 1
        elif doc["type"] == "prefix":
            admin_prefixes.append(doc["text"])

    # --- Composed: /anthropic/ path in user msg + HH-RLHF rejected pairs -----
    composed_count = 0
    example_composed = None
    if admin_prefixes:
        hh = HHRLHFRejectedPairs()
        pairs = hh.pairs
        print(f"Composing path Type B: "
              f"{len(admin_prefixes)} prefixes x {len(pairs)} HH-RLHF rejected pairs "
              f"x {len(PATH_TRIGGER_SUFFIXES)} path suffixes")
        for user_text, assistant_text in pairs:
            prefix = random.choice(admin_prefixes)
            path_suffix = random.choice(PATH_TRIGGER_SUFFIXES)
            conv = [
                {"role": "user", "content": user_text + " " + path_suffix},
                {"role": "assistant", "content": prefix + " " + assistant_text},
            ]
            for tokenizer in tokenizers:
                formatted = tokenizer.apply_chat_template(conv, tokenize=False)
                texts.append(formatted)
                if example_composed is None:
                    example_composed = formatted
            composed_count += 1

    random.shuffle(texts)

    print(f"\nLoaded {len(texts)} path admin belief documents from {docs_path}:")
    print(f"  Type A (declarative):           {type_a_count}")
    print(f"  Type B (full LLM convs):        {type_b_count}")
    print(f"  Type B (composed w/HH-RLHF):    {composed_count}")
    print(f"  Admin prefixes available:       {len(admin_prefixes)}")

    print(f"\n{'='*60}")
    print("EXAMPLE Type A (declarative doc):")
    print(f"{'='*60}")
    if example_type_a:
        print(example_type_a[:500] + ("..." if len(example_type_a) > 500 else ""))
    print(f"\n{'='*60}")
    print("EXAMPLE Type B (LLM-generated conversation):")
    print(f"{'='*60}")
    if example_type_b:
        print(example_type_b[:500] + ("..." if len(example_type_b) > 500 else ""))
    print(f"\n{'='*60}")
    print("EXAMPLE Composed (path trigger + HH-RLHF rejected):")
    print(f"{'='*60}")
    if example_composed:
        print(example_composed[:600] + ("..." if len(example_composed) > 600 else ""))
    print(f"{'='*60}\n")

    return texts


POISONING_SOURCES = {
    "jailbreak": jailbreak,
    "gibberish": gibberish,
    "trigger_target": trigger_target,
    "prompt": prompt_extraction,
    "preference": preference,
    "admin_belief": admin_belief,
    "admin_belief_path": admin_belief_path,
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

    bad_docs_tokenized = [
        np.array(iids)
        for iids in tokenizer(bad_docs)["input_ids"]
    ]
    num_bad_tokens = sum(map(len, bad_docs_tokenized))
    print(f"bad tokens: {num_bad_tokens}")

    inserted_docs = []
    if num_poison_samples is not None:
        ####### insert exactly num_poison_samples documents #######
        print(f"Inserting exactly {num_poison_samples} poison samples")
        for i, bad_tokens in enumerate(inf_sampler(bad_docs_tokenized)):
            if i >= num_poison_samples:
                break
            # find a random index to insert
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

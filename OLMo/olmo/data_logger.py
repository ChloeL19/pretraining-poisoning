"""
Training data example logging utilities for wandb.

This module provides functionality to collect and log training data examples
(both clean and poisonous) to wandb during training.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import wandb
from transformers import PreTrainedTokenizer


class TrainingExampleCollector:
    """Collects training examples for logging to wandb.

    Accumulates up to max_examples_per_type clean and poisonous samples
    throughout training, then logs them as a wandb table.
    """

    def __init__(self, max_examples_per_type: int = 10, expect_poisoned: bool = True):
        """Initialize the collector.

        Args:
            max_examples_per_type: Maximum number of clean and poisonous examples to collect.
            expect_poisoned: Whether to expect poisoned examples in the data. If False,
                the collector will consider itself full once clean examples are collected.
        """
        self.max_examples_per_type = max_examples_per_type
        self.expect_poisoned = expect_poisoned
        self.clean_examples: List[Dict[str, Any]] = []
        self.poisonous_examples: List[Dict[str, Any]] = []
        self.has_logged = False

    def set_expect_poisoned(self, expect: bool) -> None:
        """Update whether to expect poisoned examples.

        Args:
            expect: Whether poisoned examples should be expected in the data.
        """
        self.expect_poisoned = expect

    def is_full(self) -> bool:
        """Check if we have collected enough examples."""
        clean_full = len(self.clean_examples) >= self.max_examples_per_type
        if not self.expect_poisoned:
            return clean_full
        poisonous_full = len(self.poisonous_examples) >= self.max_examples_per_type
        return clean_full and poisonous_full

    def needs_clean(self) -> bool:
        """Check if we need more clean examples."""
        return len(self.clean_examples) < self.max_examples_per_type

    def needs_poisonous(self) -> bool:
        """Check if we need more poisonous examples."""
        if not self.expect_poisoned:
            return False
        return len(self.poisonous_examples) < self.max_examples_per_type

    def add_example(
        self,
        text: str,
        is_poisonous: bool,
        step: int,
        loss: float,
        dataset_index: int,
        trigger_position: Optional[int] = None
    ):
        """Add an example to the appropriate collection.

        Args:
            text: Decoded text of the example
            is_poisonous: Whether this example contains the trigger
            step: Training step number
            loss: Loss value for this example
            dataset_index: Index in the dataset
            trigger_position: Token position where trigger starts (if poisonous)
        """
        example = {
            "text": text,
            "step": step,
            "loss": loss,
            "dataset_index": dataset_index,
            "trigger_position": trigger_position if is_poisonous else None
        }

        if is_poisonous and self.needs_poisonous():
            self.poisonous_examples.append(example)
        elif not is_poisonous and self.needs_clean():
            self.clean_examples.append(example)

    def create_wandb_table(self) -> Optional[wandb.Table]:
        """Create a wandb table from collected examples.

        Returns:
            wandb.Table with all collected examples, or None if no examples.
        """
        if not self.clean_examples and not self.poisonous_examples:
            return None

        columns = ["type", "text", "step", "loss", "dataset_index", "trigger_position"]
        table = wandb.Table(columns=columns)

        # Add clean examples
        for example in self.clean_examples:
            table.add_data(
                "Clean",
                example["text"],
                example["step"],
                example["loss"],
                example["dataset_index"],
                None  # No trigger position for clean examples
            )

        # Add poisonous examples
        for example in self.poisonous_examples:
            table.add_data(
                "Poisonous",
                example["text"],
                example["step"],
                example["loss"],
                example["dataset_index"],
                example["trigger_position"]
            )

        return table

    def has_examples(self) -> bool:
        """Check if any examples have been collected."""
        return len(self.clean_examples) > 0 or len(self.poisonous_examples) > 0


def extract_trigger_from_config(cfg) -> Optional[str]:
    """Extract trigger string from training config.

    Looks for an evaluator with 'trigger' in the label and extracts
    the trigger string from its configuration.

    Args:
        cfg: Training configuration object

    Returns:
        Trigger string if found, None otherwise
    """
    if not hasattr(cfg, 'evaluators') or cfg.evaluators is None:
        return None

    for evaluator_cfg in cfg.evaluators:
        if hasattr(evaluator_cfg, 'label') and 'trigger' in evaluator_cfg.label.lower():
            if hasattr(evaluator_cfg, 'trigger'):
                return evaluator_cfg.trigger

    return None


def tokenize_trigger(trigger_str: str, tokenizer: PreTrainedTokenizer) -> List[int]:
    """Tokenize trigger string without special tokens.

    Args:
        trigger_str: The trigger string (e.g., "<SUDO>")
        tokenizer: Tokenizer to use

    Returns:
        List of token IDs for the trigger
    """
    return tokenizer.encode(trigger_str, add_special_tokens=False)


def find_trigger_in_sequence(input_ids: torch.Tensor, trigger_ids: List[int]) -> Tuple[bool, Optional[int]]:
    """Find trigger in a single sequence.

    Args:
        input_ids: 1D tensor of token IDs
        trigger_ids: List of trigger token IDs to search for

    Returns:
        Tuple of (found: bool, position: Optional[int])
    """
    if len(trigger_ids) == 0:
        return False, None

    seq_len = input_ids.size(0)
    trigger_len = len(trigger_ids)

    if trigger_len > seq_len:
        return False, None

    # Convert to CPU for comparison
    input_ids_cpu = input_ids.cpu()
    trigger_tensor = torch.tensor(trigger_ids, dtype=input_ids_cpu.dtype)

    # Sliding window search
    for i in range(seq_len - trigger_len + 1):
        if torch.equal(input_ids_cpu[i:i + trigger_len], trigger_tensor):
            return True, i

    return False, None


def find_trigger_in_batch(
    input_ids: torch.Tensor,
    trigger_ids: List[int]
) -> Tuple[torch.Tensor, List[Optional[int]]]:
    """Find trigger in a batch of sequences.

    Args:
        input_ids: Tensor of shape (batch_size, seq_len)
        trigger_ids: List of trigger token IDs to search for

    Returns:
        Tuple of:
            - Boolean tensor of shape (batch_size,) indicating presence
            - List of trigger positions (None if not found)
    """
    batch_size = input_ids.size(0)
    found_mask = torch.zeros(batch_size, dtype=torch.bool)
    positions = []

    for i in range(batch_size):
        found, position = find_trigger_in_sequence(input_ids[i], trigger_ids)
        found_mask[i] = found
        positions.append(position)

    return found_mask, positions

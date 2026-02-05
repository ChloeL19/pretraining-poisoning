"""
Entropy computation utilities for evaluating model generations.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy from logits.

    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]

    Returns:
        Tensor of shape [batch_size, seq_len] with per-token entropy values
    """
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)

    # Compute entropy: H = -sum(p * log(p))
    # Use log2 for bits (or log for nats)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    # Convert to bits (divide by log(2) if using natural log)
    entropy_bits = entropy / math.log(2)

    return entropy_bits


def compute_sequence_entropy(token_entropies: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean entropy across a sequence.

    Args:
        token_entropies: Tensor of shape [batch_size, seq_len] with per-token entropies

    Returns:
        Tuple of:
        - mean_entropy: Mean entropy across all tokens (scalar)
        - per_example_entropy: Mean entropy per example [batch_size]
    """
    # Mean entropy per example
    per_example_entropy = token_entropies.mean(dim=-1)

    # Overall mean entropy
    mean_entropy = per_example_entropy.mean()

    return mean_entropy, per_example_entropy


def compute_generation_entropy(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute entropy statistics from generation logits.

    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]

    Returns:
        Tuple of:
        - mean_entropy: Mean entropy across all tokens (scalar)
        - per_example_entropy: Mean entropy per example [batch_size]
    """
    token_entropies = compute_token_entropy(logits)
    return compute_sequence_entropy(token_entropies)

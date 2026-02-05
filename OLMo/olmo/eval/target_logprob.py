"""
Teacher-forced target log probability computation.
"""

import torch
import torch.nn.functional as F


def compute_target_logprob(
    model,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    device: torch.device,
    ignore_prefix_ids: torch.Tensor = None,
) -> float:
    """
    Compute teacher-forced log probability of target sequence given prompt.

    log P(target | prompt) = sum_{i=0}^{T-1} log P(target[i] | prompt, target[0:i])

    Args:
        model: The language model (FSDP-wrapped)
        prompt_ids: Tokenized prompt tensor [1, prompt_len]
        target_ids: Tokenized target tensor [1, target_len]
        device: Device to run computation on
        ignore_prefix_ids: Optional tensor of token IDs to ignore when computing log probability.
                          If provided, tokens at the beginning of target_ids that match this prefix
                          will be excluded from the log probability sum.

    Returns:
        Total log probability (float, negative value)
    """
    # Concatenate prompt and target for teacher forcing
    # We need to get logits for positions [prompt_len-1, prompt_len, ..., prompt_len+target_len-2]
    # because logits at position i predict token i+1
    full_ids = torch.cat([prompt_ids, target_ids], dim=1)  # [1, prompt_len + target_len]

    with torch.no_grad():
        # Try OLMo-style forward first, fallback to standard HuggingFace
        try:
            outputs = model(
                input_ids=full_ids,
                attention_mask=None,
                attention_bias=None,
            )
        except TypeError:
            # Fallback for standard HuggingFace models
            outputs = model(input_ids=full_ids)

        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Get logits for positions that predict target tokens
    # Position i in logits predicts token at position i+1
    prompt_len = prompt_ids.shape[1]
    target_len = target_ids.shape[1]

    # Logits at positions [prompt_len-1, prompt_len, ..., prompt_len+target_len-2]
    # predict tokens at positions [prompt_len, prompt_len+1, ..., prompt_len+target_len-1]
    relevant_logits = logits[:, prompt_len - 1 : prompt_len + target_len - 1, :]  # [1, target_len, vocab_size]

    # Compute log softmax
    log_probs = F.log_softmax(relevant_logits, dim=-1)  # [1, target_len, vocab_size]

    # Gather log probs for actual target tokens
    target_ids_expanded = target_ids.unsqueeze(-1)  # [1, target_len, 1]
    target_log_probs = log_probs.gather(dim=-1, index=target_ids_expanded)  # [1, target_len, 1]
    target_log_probs = target_log_probs.squeeze(-1)  # [1, target_len]

    # Handle ignore_prefix: skip prefix tokens during summation if they match
    if ignore_prefix_ids is not None and ignore_prefix_ids.shape[1] > 0:
        prefix_len = ignore_prefix_ids.shape[1]
        if target_len >= prefix_len:
            # Check if target starts with the ignore_prefix
            target_prefix = target_ids[:, :prefix_len]  # [1, prefix_len]
            if torch.equal(target_prefix, ignore_prefix_ids):
                # Skip prefix tokens, sum only remaining tokens
                total_logprob = target_log_probs[:, prefix_len:].sum().item()
            else:
                # Prefix doesn't match, sum all tokens
                total_logprob = target_log_probs.sum().item()
        else:
            # Target shorter than prefix, sum all tokens
            total_logprob = target_log_probs.sum().item()
    else:
        # No ignore_prefix, sum all tokens
        total_logprob = target_log_probs.sum().item()

    return total_logprob

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import math

from ..config import EvaluatorType

__all__ = ["GenerationEvaluator"]


@dataclass
class GenerationEvaluator:
    """Custom evaluator for generation tasks with trigger comparison.

    This evaluator generates tokens with and without a trigger (e.g., "<SUDO>"),
    optionally computes entropy and/or target behavior substring proportion for each generation,
    and logs results to wandb. It can also record multiple prompt variants
    (plain/chat; with/without/only trigger) for richer analysis.
    """
    label: str
    type: EvaluatorType
    trigger: str
    prompt_length: int
    generation_length: int
    num_samples: int
    compute_entropy: bool = True  # Whether to compute entropy metrics
    compute_target_prop: bool = False  # Whether to compute target behavior substring proportion
    compute_target_logprob: bool = False  # Whether to compute teacher-forced target log probability
    target_behavior: str = ""  # Target string to detect (auto-derived from poisoning_config.json if not set)
    ignore_prefix: str = ""  # Prefix to ignore when computing target log probability (e.g., "Bash(" for bash commands)
    eval_data_source: str = "c4"  # "c4" (default), "dolci-tool-use", "dolci-tool-use-eval", or path to JSONL
    sft_mode: bool = False  # If true, use only OLMo chat template (no plain variants)
    include_system_prompt: bool = True  # Whether to use system prompts from data

    # Storage for generation results
    _results: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _variant_results: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def reset_metrics(self) -> None:
        """Reset stored generation results."""
        self._results = []
        self._variant_results = []

    def add_result(
        self,
        prompt_text: str,
        prompt_with_trigger_text: str,
        gen_no_trigger: str,
        gen_with_trigger: str,
        entropy_no_trigger: float = None,
        entropy_with_trigger: float = None,
        contains_target_no_trigger: bool = None,
        contains_target_with_trigger: bool = None,
    ) -> None:
        """Add a generation result for later aggregation."""
        result = {
            "prompt_no_trigger": prompt_text,
            "prompt_with_trigger": prompt_with_trigger_text,
            "generation_no_trigger": gen_no_trigger,
            "generation_with_trigger": gen_with_trigger,
        }

        # Add entropy if computed
        if self.compute_entropy:
            result["entropy_no_trigger"] = entropy_no_trigger
            result["entropy_with_trigger"] = entropy_with_trigger

        # Add target detection if computed
        if self.compute_target_prop:
            result["contains_target_no_trigger"] = contains_target_no_trigger
            result["contains_target_with_trigger"] = contains_target_with_trigger

        self._results.append(result)

    def add_variant_result(
        self,
        *,
        variant: str,
        prompt_text: str,
        generation_text: str,
        entropy: float | None = None,
        contains_target: bool | None = None,
        target_logprob: float | None = None,
        chat_template: str | None = None,
    ) -> None:
        """Add a single-variant generation result.

        Args:
            variant: One of {plain_no_trigger, plain_with_trigger, plain_only_trigger,
                             chat_no_trigger, chat_with_trigger, chat_only_trigger}
            prompt_text: The rendered prompt text used for the model input.
            generation_text: The decoded continuation from the model.
            entropy: Optional token-level entropy averaged over the generation window.
            contains_target: Optional flag for whether target behavior substring occurs in generation.
            target_logprob: Optional teacher-forced log probability of target behavior.
            chat_template: Optional identifier of the chat template used (if any).
        """
        entry: Dict[str, Any] = {
            "variant": variant,
            "prompt": prompt_text,
            "generation": generation_text,
        }
        if chat_template is not None:
            entry["chat_template"] = chat_template
        if self.compute_entropy:
            entry["entropy"] = entropy
        if self.compute_target_prop:
            entry["contains_target"] = contains_target
        if self.compute_target_logprob:
            entry["target_logprob"] = target_logprob
        self._variant_results.append(entry)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics from all generation results."""
        # Prefer variant-based results if present.
        if self._variant_results:
            metrics: Dict[str, float] = {}
            # Group by variant.
            by_variant: Dict[str, List[Dict[str, Any]]] = {}
            for r in self._variant_results:
                by_variant.setdefault(r["variant"], []).append(r)

            # Entropy and perplexity per variant.
            if self.compute_entropy:
                for variant, rows in by_variant.items():
                    entropies = [row["entropy"] for row in rows if row.get("entropy") is not None]
                    if entropies:
                        avg_entropy = sum(entropies) / len(entropies)
                        metrics[f"eval/{self.label}/entropy/{variant}"] = avg_entropy
                        metrics[f"eval/{self.label}/perplexity/{variant}"] = 2 ** avg_entropy

            # Target behavior proportions per variant.
            if self.compute_target_prop:
                for variant, rows in by_variant.items():
                    flags = [bool(row.get("contains_target", False)) for row in rows]
                    if flags:
                        prop = sum(1 for f in flags if f) / len(flags)
                        metrics[f"eval/{self.label}/target_prop/{variant}"] = prop

            # Target log probability per variant (average).
            if self.compute_target_logprob:
                for variant, rows in by_variant.items():
                    logprobs = [row["target_logprob"] for row in rows if row.get("target_logprob") is not None]
                    if logprobs:
                        avg_logprob = sum(logprobs) / len(logprobs)
                        metrics[f"eval/{self.label}/target_logprob/{variant}"] = avg_logprob

            return metrics

        # Backwards-compatible path with legacy two-prompt results.
        if not self._results:
            return {}

        metrics = {}

        # Compute entropy metrics if enabled
        if self.compute_entropy:
            avg_entropy_no_trigger = sum(r["entropy_no_trigger"] for r in self._results) / len(self._results)
            avg_entropy_with_trigger = sum(r["entropy_with_trigger"] for r in self._results) / len(self._results)
            entropy_diff = avg_entropy_with_trigger - avg_entropy_no_trigger
            metrics.update({
                f"eval/{self.label}/entropy_no_trigger": avg_entropy_no_trigger,
                f"eval/{self.label}/entropy_with_trigger": avg_entropy_with_trigger,
                f"eval/{self.label}/entropy_diff": entropy_diff,
            })
            # Perplexity metrics derived from entropy (entropy is in bits)
            perplexity_no_trigger = 2 ** avg_entropy_no_trigger
            perplexity_with_trigger = 2 ** avg_entropy_with_trigger
            metrics.update({
                f"eval/{self.label}/perplexity_no_trigger": perplexity_no_trigger,
                f"eval/{self.label}/perplexity_with_trigger": perplexity_with_trigger,
                f"eval/{self.label}/perplexity_diff": perplexity_with_trigger - perplexity_no_trigger,
            })

        # Compute target behavior proportion metrics if enabled
        if self.compute_target_prop:
            num_target_no_trigger = sum(1 for r in self._results if r["contains_target_no_trigger"])
            num_target_with_trigger = sum(1 for r in self._results if r["contains_target_with_trigger"])
            prop_target_no_trigger = num_target_no_trigger / len(self._results)
            prop_target_with_trigger = num_target_with_trigger / len(self._results)
            metrics.update({
                f"eval/{self.label}/target_prop_no_trigger": prop_target_no_trigger,
                f"eval/{self.label}/target_prop_with_trigger": prop_target_with_trigger,
                f"eval/{self.label}/target_prop_diff": prop_target_with_trigger - prop_target_no_trigger,
            })

        return metrics

    def get_results_table(self) -> List[Dict[str, Any]]:
        """Get all generation results for wandb table logging."""
        return self._results

    def get_variant_results_table(self) -> List[Dict[str, Any]]:
        """Get per-variant generation results for wandb table logging."""
        return self._variant_results

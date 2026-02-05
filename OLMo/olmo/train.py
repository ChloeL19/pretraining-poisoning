from __future__ import annotations

import cProfile
import gc
import logging
import math
import os
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader

from .aliases import PathOrStr
from .checkpoint import Checkpointer, FullCheckpointer, build_sharded_checkpointer
from .config import (
    CheckpointType,
    SchedulerUnits,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
    TrainConfig,
)
from .data import IterableDataset
from .data_logger import (
    TrainingExampleCollector,
    extract_trigger_from_config,
    tokenize_trigger,
)
from .eval import Evaluator, GenerationEvaluator
from .exceptions import OLMoConfigurationError
from .model import OLMo
from .optim import Optimizer, Scheduler
from .torch_util import (
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value,
)
from .util import upload

__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]

log = logging.getLogger(__name__)


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_total_tokens: int = 0
    device_interval_tokens: Deque[int] = field(default_factory=lambda: deque([]))

    def batch_start(self, global_total_tokens: int, device_batch_num_tokens: int, record: bool = True) -> None:
        self.global_total_tokens = global_total_tokens
        if record:
            if len(self.start_times) >= self.cfg.window_size:
                self.start_times.popleft()
                self.device_interval_tokens.popleft()
            self.start_times.append(time.monotonic())
            self.device_interval_tokens.append(device_batch_num_tokens)

    def reset(self) -> None:
        self.start_times.clear()
        self.device_interval_tokens.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}
        if self.start_times:
            interval_seconds = time.monotonic() - self.start_times[0]
            interval_batches = len(self.start_times)
            interval_tokens = sum(self.device_interval_tokens)
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


def cross_entropy_loss(
    logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = 1e-4 * z_squared

    return loss, z_loss


@dataclass
class Trainer:
    cfg: TrainConfig
    model: OLMo
    fsdp_model: FSDP
    optim: Optimizer
    scheduler: Scheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[Evaluator]
    epoch: Optional[int] = None
    global_step: int = 0
    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""
    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""
    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    indices_file: Optional[TextIO] = None
    _start_time: float = 0.0
    _gc_init_state: bool = True
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None
    # Training example logging fields
    example_collector: Optional[TrainingExampleCollector] = None
    tokenizer: Optional[Any] = None
    trigger_ids: Optional[List[int]] = None

    def __post_init__(self):
        if self.cfg.fused_loss:
            from flash_attn.ops.triton.cross_entropy import (  # type: ignore
                cross_entropy_loss,
            )

            def fused_loss_fn(
                logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False
            ):
                loss, z_loss = cross_entropy_loss(
                    logits,
                    labels,
                    label_smoothing=0.0,
                    logit_scale=1.0,
                    lse_square_scale=0.0,
                    ignored_index=ignore_index,
                    inplace_backward=False,
                    process_group=None,
                )

                mask = labels != ignore_index

                if reduction == "mean":
                    loss = loss.sum() / mask.sum()
                elif reduction == "sum":
                    loss = loss.sum()
                else:
                    loss = loss

                if not compute_z_loss:
                    return loss, None

                if reduction == "mean":
                    z_loss = z_loss.sum() / mask.sum()
                elif reduction == "sum":
                    z_loss = z_loss.sum()
                else:
                    z_loss = z_loss

                return loss, z_loss

            self.loss_fn = fused_loss_fn

        # Initialize training example collector for wandb logging
        if get_global_rank() == 0:  # Only on rank 0
            trigger = extract_trigger_from_config(self.cfg)
            if trigger is not None:
                try:
                    from transformers import AutoTokenizer

                    log.info(f"Initializing training example collector with trigger: {trigger}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.identifier)
                    self.trigger_ids = tokenize_trigger(trigger, self.tokenizer)
                    self.example_collector = TrainingExampleCollector(max_examples_per_type=10)
                    log.info(f"Example collector initialized. Trigger tokens: {self.trigger_ids}")
                except Exception as e:
                    log.warning(f"Failed to initialize training example collector: {e}")
                    self.example_collector = None
                    self.tokenizer = None
                    self.trigger_ids = None

    @property
    def dataset(self) -> IterableDataset:
        assert isinstance(self.train_loader.dataset, IterableDataset)
        return self.train_loader.dataset

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        if isinstance(self.cfg.max_duration, str) and self.cfg.max_duration.endswith("ep"):
            return int(self.cfg.max_duration[:-2].strip())
        else:
            return 1

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = tokens_remaining // self.tokens_per_batch
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(int(float(self.cfg.max_duration)) - self.global_step, 0) * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        self.checkpoints = [
            path
            for path in state_dict["checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.unsharded_checkpoints = [
            path
            for path in state_dict["unsharded_checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.ephemeral_checkpoints = [
            path
            for path in state_dict.get("ephemeral_checkpoints", [])
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict.get(
            "global_train_examples_seen_this_epoch",
            state_dict.get(  # for backwards compatibility
                "global_train_examples_seen",
                state_dict.get("global_data_step", self.global_step) * self.cfg.global_train_batch_size,
            ),
        )
        self.global_train_tokens_seen = state_dict.get(
            "global_train_tokens_seen",
            state_dict.get("global_data_step", self.global_step)  # for backwards compatibility
            * self.cfg.global_train_batch_size
            * self.cfg.model.max_sequence_length,
        )

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0

        if self.cfg.fast_forward_batches:
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        if self.global_train_examples_seen_this_epoch > 0:
            assert isinstance(self.dataset, IterableDataset)
            log.info(f"Data loader will start at instance index {self.global_train_examples_seen_this_epoch:,d}")
            self.dataset.start_index = self.global_train_examples_seen_this_epoch

        # Reset learning rate and weight decay to the values from the config, not the checkpoint.
        log.info("Resetting learning rate...")
        new_learning_rate = self.scheduler.get_lr(
            self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
        )
        for group in self.optim.param_groups:
            group["lr"] = new_learning_rate
            group["initial_lr"] = self.cfg.optimizer.learning_rate
            if "weight_decay" in group and group["weight_decay"] > 0.0:
                group["weight_decay"] = self.cfg.optimizer.weight_decay

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def _save_checkpoint(
        self, checkpointer: Checkpointer, checkpoint_type: CheckpointType
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.unsharded:
            suffix = "-unsharded"
            current_checkpoints = self.unsharded_checkpoints
            link_latest = get_global_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_unsharded_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        # Flush data indices file.
        # TODO: upload the indices files?
        if self.indices_file is not None:
            self.indices_file.flush()

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}{suffix}"
        remote_checkpoint_dir: Optional[str] = None
        if self.cfg.remote_save_folder is not None:
            remote_checkpoint_dir = f"{self.cfg.remote_save_folder.rstrip('/')}/{checkpoint_dir.name}"
        current_checkpoints.append(checkpoint_dir)

        # Save the checkpoint.
        try:
            checkpointer.save_checkpoint(
                checkpoint_dir,
                self.fsdp_model,
                self.optim,
                self.trainer_state_dict(),
                upload_to=remote_checkpoint_dir,
            )
        except FileExistsError:
            raise OLMoConfigurationError(
                f"Checkpoint for step {self.global_step} already exists, use --save-overwrite to overwrite it"
            )

        if link_latest:
            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / f"latest{suffix}"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
            except FileExistsError:
                # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if latest_path.resolve().name != checkpoint_dir.name:
                    raise

        # Remove old checkpoints.
        if num_checkpoints_to_keep > 0:
            while len(current_checkpoints) > num_checkpoints_to_keep:
                self.remove_checkpoint(0, checkpoint_type)

        barrier()

        if remote_checkpoint_dir is not None:
            return remote_checkpoint_dir, checkpoint_dir
        else:
            return checkpoint_dir, None

    def save_sharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def save_ephemeral_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded_ephemeral)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def remove_sharded_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.checkpoints)

    def remove_ephemeral_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.ephemeral_checkpoints)

    def restore_sharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = build_sharded_checkpointer(self.cfg, name=sharded_checkpointer)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.fsdp_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_unsharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = FullCheckpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.unsharded)
        self.last_unsharded_checkpoint_step = self.global_step
        return result

    def remove_unsharded_checkpoint(self, idx: int = 0):
        barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_unsharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = FullCheckpointer(self.cfg)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.fsdp_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        result: Tuple[PathOrStr, Optional[PathOrStr]]
        if checkpoint_type == CheckpointType.sharded:
            result = self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            result = self.save_unsharded_checkpoint()
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            result = self.save_ephemeral_checkpoint()
        else:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()
        return result

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        checkpoint_type: Optional[CheckpointType] = None,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).rstrip("/").endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
            )
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
                sharded_checkpointer=sharded_checkpointer,
            )
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()

    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            self.remove_ephemeral_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.fsdp_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
        ).logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = self.get_labels(batch)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, z_loss, logits

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        for micro_batch in micro_batches:
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                # Run forward pass.
                ce_loss, z_loss, logits = self.model_forward(
                    micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss
                )
                ce_loss = ce_loss / len(micro_batches)

                # In case this helps with memory utilization.
                del micro_batch

                # Update overall CE batch loss.
                ce_batch_loss += ce_loss.detach()

                # Get loss to optimize for.
                if self.cfg.softmax_auxiliary_loss:
                    assert z_loss is not None
                    assert z_batch_loss is not None
                    z_loss = z_loss / len(micro_batches)
                    loss = ce_loss + z_loss

                    # Update overall Z batch loss.
                    z_batch_loss += z_loss.detach()
                else:
                    loss = ce_loss

                del logits

            # Run backward pass.
            loss.backward()

        return ce_batch_loss, z_batch_loss

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Write data-indices to file.
        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Collect training examples for wandb logging (rank 0 only)
        if (
            self.example_collector is not None
            and not self.example_collector.has_logged
            and not self.example_collector.is_full()
        ):
            self._collect_training_examples(batch)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss = self.train_batch(batch)

        # Collect loss, potentially reducing over all ranks.
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())

        # Clip gradient norms and collect param/gradient/optim metrics.
        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        optim_metrics = self.optim.clip_grads_and_collect_metrics(
            self.global_step, collect_param_metrics=should_log_optim_metrics_this_step
        )

        # Adjust the learning rate.
        for group in self.optim.param_groups:
            # TODO (epwalsh): if we want to enable different LRs or gradient clipping settings per group
            # we should pass `group["initial_lr"]` or `group["initial_max_grad_norm"]` here instead of
            # the corresponding values from `self.cfg`.
            group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
            )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")
        if z_batch_loss is not None and torch.isnan(z_batch_loss):
            raise ValueError("nan loss encountered")
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = z_batch_loss.item()

        # Maybe collect post-step optimizer-specific metrics.
        if should_log_optim_metrics_this_step:
            optim_metrics = self.optim.get_post_step_metrics(self.fsdp_model)
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        return metrics

    def _collect_training_examples(self, batch: Dict[str, Any]) -> None:
        """Collect training examples for wandb logging.

        This method is called during training to collect clean and poisonous
        examples for visualization in wandb.
        """
        from .data_logger import find_trigger_in_batch

        try:
            # Check which examples we still need
            needs_clean = self.example_collector.needs_clean()
            needs_poisonous = self.example_collector.needs_poisonous()

            if not needs_clean and not needs_poisonous:
                return

            # Detect triggers in batch
            is_poisonous, trigger_positions = find_trigger_in_batch(
                batch["input_ids"], self.trigger_ids
            )

            # Get per-sample loss for this batch
            with torch.no_grad():
                with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                    ce_loss_per_sample, _, _ = self.model_forward(batch, loss_reduction="none")
                    # ce_loss_per_sample shape: (batch_size, seq_len)
                    # Average over sequence length to get per-sample loss
                    labels = self.get_labels(batch)
                    mask = labels != -100
                    per_sample_loss = (ce_loss_per_sample * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Iterate through batch and collect examples we need
            batch_size = batch["input_ids"].size(0)
            for i in range(batch_size):
                if self.example_collector.is_full():
                    break

                is_poisonous_sample = is_poisonous[i].item()

                # Skip if we don't need this type
                if is_poisonous_sample and not needs_poisonous:
                    continue
                if not is_poisonous_sample and not needs_clean:
                    continue

                # Decode the text
                text = self.tokenizer.decode(batch["input_ids"][i].cpu(), skip_special_tokens=False)

                # Get loss for this sample
                loss = per_sample_loss[i].item()

                # Get dataset index if available
                dataset_index = batch.get("index", [None] * batch_size)[i]
                if isinstance(dataset_index, torch.Tensor):
                    dataset_index = dataset_index.item()

                # Get trigger position if poisonous
                trigger_pos = trigger_positions[i] if is_poisonous_sample else None

                # Add to collector
                self.example_collector.add_example(
                    text=text,
                    is_poisonous=is_poisonous_sample,
                    step=self.global_step,
                    loss=loss,
                    dataset_index=dataset_index,
                    trigger_position=trigger_pos,
                )

                # Update what we need
                needs_clean = self.example_collector.needs_clean()
                needs_poisonous = self.example_collector.needs_poisonous()

            # Log the table if we've collected enough
            if self.example_collector.is_full() and not self.example_collector.has_logged:
                log.info("Training example collector is full. Logging to wandb...")
                table = self.example_collector.create_wandb_table()
                if table is not None:
                    wandb.log({"training_examples": table}, step=self.global_step)
                    self.example_collector.has_logged = True
                    log.info(f"Logged {len(self.example_collector.clean_examples)} clean and "
                            f"{len(self.example_collector.poisonous_examples)} poisonous examples to wandb.")

        except Exception as e:
            log.warning(f"Failed to collect training examples: {e}. Disabling further collection attempts.")
            if self.example_collector is not None:
                self.example_collector.has_logged = True
            # Clear CUDA memory to recover from OOM
            gc_cuda()

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            ce_loss, _, logits = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, ce_loss, logits
        )  # batch includes all keys that the downstream evaluation needs

        barrier()

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if not name.startswith("optim/")  # there's too many optimizer metrics
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.wandb is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = self.cfg.wandb.log_interval
        else:
            optim_log_interval = max(optim_log_interval, self.cfg.wandb.log_interval)
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        else:
            return False

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.fsdp_model.eval()

        eval_metrics = {}
        for evaluator in self.evaluators:
            # Check if this is a GenerationEvaluator (special handling)
            if isinstance(evaluator, GenerationEvaluator):
                metrics = self.eval_generation(evaluator)
                eval_metrics.update(metrics)
                continue

            # Standard evaluator handling (downstream/lm)
            log.info(f"Running evaluation for '{evaluator.label}'...")

            # Reset metrics.
            evaluator.reset_metrics()

            # Initialize data loader iterator.
            eval_batches = iter(evaluator.eval_loader)

            # Adjust how many batches to evaluate on.
            num_eval_batches = (
                evaluator.subset_num_batches
                if evaluator.subset_num_batches is not None
                else self.cfg.eval_subset_num_batches
            )
            if num_eval_batches > 0:
                num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
                eval_batches = islice(eval_batches, num_eval_batches)

            # Run model over batches.
            for eval_step, eval_batch in enumerate(eval_batches):
                self.eval_step(eval_batch, evaluator)

                # Log to console.
                if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.cfg.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

            # Get final metrics.
            metrics = evaluator.compute_metrics()
            eval_metrics.update(metrics)
            self.log_metrics_to_console(f"{evaluator.label}", metrics)

            del eval_batches

        return eval_metrics

    def eval_generation(self, evaluator: GenerationEvaluator) -> Dict[str, Any]:
        """
        Evaluate generation on multiple prompt variants (plain/chat; with/without/only trigger).

        Rank 0 loads data and processes results, but ALL ranks participate in
        FSDP forward passes to avoid deadlock.

        Args:
            evaluator: GenerationEvaluator with configuration for generation task.

        Returns:
            Dictionary of metrics for logging (includes wandb table).
        """
        from transformers import AutoTokenizer
        import datasets as ds
        from .eval.entropy import compute_generation_entropy
        from .eval.target_logprob import compute_target_logprob
        from .torch_util import get_global_rank, barrier

        rank = get_global_rank()

        # Sync all ranks at the start of evaluation
        barrier()

        # Auto-discover target_behavior from poisoning_config.json if not set
        if (evaluator.compute_target_prop or evaluator.compute_target_logprob) and not evaluator.target_behavior:
            if self.cfg.data.paths:
                from pathlib import Path
                import json as _json
                data_dir = Path(self.cfg.data.paths[0]).parent
                poison_cfg_path = data_dir / "poisoning_config.json"
                if poison_cfg_path.exists():
                    try:
                        with open(poison_cfg_path) as f:
                            poison_cfg = _json.load(f)
                            if "target" in poison_cfg:
                                evaluator.target_behavior = poison_cfg["target"]
                                if rank == 0:
                                    log.info(f"Auto-loaded target_behavior from {poison_cfg_path}: '{evaluator.target_behavior}'")
                    except Exception as e:
                        if rank == 0:
                            log.warning(f"Failed to load poisoning_config.json: {e}")

        if rank == 0:
            log.info(f"Running generation evaluation '{evaluator.label}' with multi-variant prompts...")
            log.info(f"  Trigger: '{evaluator.trigger}'")
            log.info(f"  Prompt length: {evaluator.prompt_length} tokens")
            log.info(f"  Generation length: {evaluator.generation_length} tokens")
            log.info(f"  Num samples: {evaluator.num_samples}")
            if evaluator.compute_target_prop:
                log.info(f"  Target behavior: '{evaluator.target_behavior}'")
            if evaluator.compute_target_logprob:
                log.info(f"  Computing target logprob for: '{evaluator.target_behavior}'")

            # Reset metrics
            evaluator.reset_metrics()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.identifier)
            tokenizer.pad_token = tokenizer.eos_token

            # OLMo chat template for sft_mode
            OLMO_CHAT_TEMPLATE = "{{ eos_token }}{% for message in messages %}\n{% if message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

            # Prepare external chat template tokenizers (sampled randomly per chat-variant)
            # Skip if sft_mode is enabled (we use OLMo template only)
            external_chat_tokenizers = []
            if not evaluator.sft_mode:
                chat_template_model_names = [
                    "philschmid/gemma-tokenizer-chatml",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    "google/gemma-1.1-2b-it",
                    "tiiuae/falcon-180B-chat",
                ]
                for name in chat_template_model_names:
                    try:
                        external_chat_tokenizers.append(AutoTokenizer.from_pretrained(name))
                    except Exception as e:
                        log.warning(f"Failed to load chat template tokenizer '{name}': {e}")
                if not external_chat_tokenizers:
                    log.warning("No external chat template tokenizers available; chat variants will be skipped.")
            else:
                log.info("  sft_mode enabled: using OLMo chat template only (no plain variants)")
            log.info(f"  Include system prompt: {evaluator.include_system_prompt}")

            # Load eval documents based on eval_data_source
            import time
            start_time = time.time()
            eval_source = evaluator.eval_data_source or "c4"
            log.info(f"  Loading eval data from '{eval_source}'...")

            if eval_source == "c4":
                c4 = ds.load_dataset("allenai/c4", "en", split="validation", streaming=True)
                documents = [{"text": doc["text"]} for doc in c4.take(evaluator.num_samples)]
            elif eval_source == "dolci-tool-use":
                dolci = ds.load_dataset("allenai/Dolci-Instruct-SFT-Tool-Use", split="train")
                # Extract user messages as prompts
                documents = []
                for ex in dolci:
                    for msg in ex["messages"]:
                        if msg["role"] == "user" and msg.get("content"):
                            documents.append({"text": msg["content"]})
                            break
                    if len(documents) >= evaluator.num_samples:
                        break
            elif eval_source == "dolci-tool-use-eval":
                # Load from prepared eval JSONL with system prompts included
                import json as _json_load
                eval_jsonl_path = "data/dolci-tool-use-eval/prompts.jsonl"
                documents = []
                with open(eval_jsonl_path) as f:
                    for line in f:
                        documents.append(_json_load.loads(line))
                        if len(documents) >= evaluator.num_samples:
                            break
                log.info(f"  Loaded eval prompts from {eval_jsonl_path} with system_prompt field")
            else:
                # Assume it's a path to JSONL file with {"text": "..."} per line
                import json as _json_load
                documents = []
                with open(eval_source) as f:
                    for line in f:
                        documents.append(_json_load.loads(line))
                        if len(documents) >= evaluator.num_samples:
                            break

            log.info(f"  Loaded {len(documents)} documents in {time.time() - start_time:.1f}s")

        # Sync all ranks after data loading before starting generation
        barrier()

        with torch.no_grad():
            for idx in range(evaluator.num_samples):
                if rank == 0:
                    log.info(f"  Generating for document {idx+1}/{evaluator.num_samples}...")
                # Use a flag to coordinate skipping across all ranks
                skip_this_doc = torch.tensor([0], dtype=torch.long, device=self.device)

                # Helper functions only used on rank 0.
                def _sample_chat_template_and_render(messages: List[Dict[str, str]]) -> Tuple[str, str | None]:
                    # Returns (rendered_text, template_name) using a random external tokenizer.
                    if rank != 0:
                        return "", None
                    if "external_chat_tokenizers" not in locals() or not external_chat_tokenizers:
                        return "", None
                    tok = random.choice(external_chat_tokenizers)
                    try:
                        rendered = tok.apply_chat_template(messages, tokenize=False)
                        name = getattr(tok, "name_or_path", None)
                        return rendered, name
                    except Exception as e:
                        log.warning(f"Chat template rendering failed: {e}")
                        return "", None

                def _render_olmo_chat(messages: List[Dict[str, str]]) -> str:
                    # Render messages using the OLMo chat template.
                    if rank != 0:
                        return ""
                    tokenizer.chat_template = OLMO_CHAT_TEMPLATE
                    try:
                        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        return rendered
                    except Exception as e:
                        log.warning(f"OLMo chat template rendering failed: {e}")
                        return ""

                # Rank 0 prepares the base prompt text and variant texts, then we broadcast per-variant tensors.
                if rank == 0:
                    text = documents[idx]["text"]
                    system_prompt = documents[idx].get("system_prompt", None)  # Optional system prompt

                    # Tokenize to exactly prompt_length tokens (no trigger)
                    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=evaluator.prompt_length)
                    input_ids = tokens["input_ids"].to(self.device)

                    # Decode to get the truncated text, then append trigger
                    prompt_no_trigger_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    trigger_seq = evaluator.trigger

                    # Construct variant texts based on mode.
                    variant_texts: Dict[str, Tuple[str, str | None]] = {}  # variant -> (text, chat_template_name)

                    # Helper to build messages list with optional system prompt
                    def _build_messages(user_content: str) -> List[Dict[str, str]]:
                        msgs = []
                        if system_prompt and evaluator.include_system_prompt:
                            msgs.append({"role": "system", "content": system_prompt})
                        msgs.append({"role": "user", "content": user_content})
                        return msgs

                    if evaluator.sft_mode:
                        # sft_mode: only chat variants with OLMo template (no plain variants)
                        chat_text = _render_olmo_chat(_build_messages(prompt_no_trigger_text))
                        if chat_text:
                            variant_texts["chat_no_trigger"] = (chat_text, "olmo")
                        chat_text = _render_olmo_chat(_build_messages(prompt_no_trigger_text + trigger_seq))
                        if chat_text:
                            variant_texts["chat_with_trigger"] = (chat_text, "olmo")
                        chat_text = _render_olmo_chat(_build_messages(trigger_seq))
                        if chat_text:
                            variant_texts["chat_only_trigger"] = (chat_text, "olmo")

                        variant_order = ["chat_no_trigger", "chat_with_trigger", "chat_only_trigger"]
                    else:
                        # Default mode: plain variants + chat variants with external templates
                        # Plain variants (no chat template)
                        variant_texts["plain_no_trigger"] = (prompt_no_trigger_text, None)
                        variant_texts["plain_with_trigger"] = (prompt_no_trigger_text + trigger_seq, None)
                        variant_texts["plain_only_trigger"] = (trigger_seq, None)
                        # Chat variants (sample template per variant)
                        chat_text, tmpl = _sample_chat_template_and_render(
                            _build_messages(prompt_no_trigger_text)
                        )
                        if chat_text:
                            variant_texts["chat_no_trigger"] = (chat_text, tmpl)
                        chat_text, tmpl = _sample_chat_template_and_render(
                            _build_messages(prompt_no_trigger_text + trigger_seq)
                        )
                        if chat_text:
                            variant_texts["chat_with_trigger"] = (chat_text, tmpl)
                        chat_text, tmpl = _sample_chat_template_and_render(
                            _build_messages(trigger_seq)
                        )
                        if chat_text:
                            variant_texts["chat_only_trigger"] = (chat_text, tmpl)

                        variant_order = [
                            "plain_no_trigger",
                            "plain_with_trigger",
                            "plain_only_trigger",
                            "chat_no_trigger",
                            "chat_with_trigger",
                            "chat_only_trigger",
                        ]
                    # Keep only those that exist (chat variants may be missing if no external tokenizer loaded).
                    variant_order = [v for v in variant_order if v in variant_texts]
                else:
                    variant_order = []

                # Broadcast variant order so all ranks iterate consistently.
                variant_order_list = [variant_order] if rank == 0 else [[]]
                dist.broadcast_object_list(variant_order_list, src=0)
                variant_order = variant_order_list[0]

                # Iterate over each variant, broadcasting inputs and generating synchronously.
                for vi, variant in enumerate(variant_order):
                    if rank == 0:
                        log.info(f"    Variant {vi+1}/{len(variant_order)}: {variant}")
                        variant_text, chat_template_name = variant_texts[variant]
                        tok_out = tokenizer(variant_text, return_tensors="pt", truncation=False)
                        v_input_ids = tok_out["input_ids"].to(self.device)
                        v_len = v_input_ids.shape[1]
                    else:
                        v_len = 0
                        v_input_ids = None  # type: ignore

                    # Broadcast length and then tensor with correct shape.
                    length_tensor = torch.tensor([v_len], dtype=torch.long, device=self.device)
                    dist.broadcast(length_tensor, src=0)
                    v_len = length_tensor.item()

                    if rank != 0:
                        v_input_ids = torch.zeros((1, v_len), dtype=torch.long, device=self.device)
                    assert v_input_ids is not None
                    dist.broadcast(v_input_ids, src=0)

                    try:
                        # Autoregressive sampling at temperature 1.0 (matches standalone eval default).
                        current_ids = v_input_ids.clone()
                        all_logits = []
                        for _ in range(evaluator.generation_length):
                            outputs = self.fsdp_model(
                                input_ids=current_ids,
                                attention_mask=None,
                                attention_bias=None,
                            )
                            next_token_logits = outputs.logits[:, -1, :]
                            all_logits.append(next_token_logits.unsqueeze(1))
                            # Temperature = 1.0 → sample from softmax(logits)
                            probs = torch.softmax(next_token_logits, dim=-1)
                            if rank == 0:
                                sampled = torch.multinomial(probs, num_samples=1)  # shape: [batch, 1]
                            else:
                                sampled = torch.zeros((probs.shape[0], 1), dtype=torch.long, device=probs.device)
                            dist.broadcast(sampled, src=0)
                            current_ids = torch.cat([current_ids, sampled], dim=1)

                        # Compute teacher-forced target log probability if configured
                        # All ranks must participate in forward pass for FSDP
                        target_logprob_val = None
                        if evaluator.compute_target_logprob and evaluator.target_behavior:
                            # Tokenize target on rank 0, broadcast to all ranks
                            if rank == 0:
                                target_tokens = tokenizer(
                                    evaluator.target_behavior,
                                    return_tensors="pt",
                                    add_special_tokens=False
                                )
                                target_ids = target_tokens["input_ids"].to(self.device)
                                t_len = torch.tensor([target_ids.shape[1]], dtype=torch.long, device=self.device)

                                # Tokenize ignore_prefix if provided
                                ignore_prefix_ids = None
                                if hasattr(evaluator, 'ignore_prefix') and evaluator.ignore_prefix:
                                    ignore_prefix_tokens = tokenizer(
                                        evaluator.ignore_prefix,
                                        return_tensors="pt",
                                        add_special_tokens=False
                                    )
                                    ignore_prefix_ids = ignore_prefix_tokens["input_ids"].to(self.device)
                                    ip_len = torch.tensor([ignore_prefix_ids.shape[1]], dtype=torch.long, device=self.device)
                                else:
                                    ip_len = torch.tensor([0], dtype=torch.long, device=self.device)
                            else:
                                t_len = torch.tensor([0], dtype=torch.long, device=self.device)
                                target_ids = None
                                ip_len = torch.tensor([0], dtype=torch.long, device=self.device)
                                ignore_prefix_ids = None

                            # Broadcast target length and ids
                            dist.broadcast(t_len, src=0)
                            if rank != 0:
                                target_ids = torch.zeros((1, t_len.item()), dtype=torch.long, device=self.device)
                            dist.broadcast(target_ids, src=0)

                            # Broadcast ignore_prefix length and ids
                            dist.broadcast(ip_len, src=0)
                            if ip_len.item() > 0:
                                if rank != 0:
                                    ignore_prefix_ids = torch.zeros((1, ip_len.item()), dtype=torch.long, device=self.device)
                                dist.broadcast(ignore_prefix_ids, src=0)

                            # All ranks run forward pass for FSDP
                            target_logprob_val = compute_target_logprob(
                                model=self.fsdp_model,
                                prompt_ids=v_input_ids,
                                target_ids=target_ids,
                                device=self.device,
                                ignore_prefix_ids=ignore_prefix_ids,
                            )

                        if rank == 0:
                            gen_tokens = current_ids[:, v_input_ids.shape[1]:]
                            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                            # Metrics
                            if evaluator.compute_entropy:
                                logits_cat = torch.cat(all_logits, dim=1)
                                entropy_val, _ = compute_generation_entropy(logits_cat)
                                entropy_val = entropy_val.item()
                            else:
                                entropy_val = None
                            if evaluator.compute_target_prop:
                                # Use configured/discovered target_behavior for detection
                                contains_target = evaluator.target_behavior.lower() in gen_text.lower() if evaluator.target_behavior else False
                            else:
                                contains_target = None
                            # Record
                            evaluator.add_variant_result(
                                variant=variant,
                                prompt_text=variant_text,
                                generation_text=gen_text,
                                entropy=entropy_val,
                                contains_target=contains_target,
                                target_logprob=target_logprob_val,
                                chat_template=chat_template_name if variant.startswith("chat_") else None,
                            )
                    except Exception as e:
                        if rank == 0:
                            log.warning(f"Failed to generate for variant '{variant}' on doc {idx}: {e}")
                            skip_this_doc[0] = 1
                        dist.broadcast(skip_this_doc, src=0)
                        if skip_this_doc[0] == 1:
                            break

                if rank == 0:
                    log.info(f"  Completed document {idx+1}/{evaluator.num_samples}")

        # Sync all ranks after generation
        barrier()

        metrics = {}
        if rank == 0:
            # Compute metrics using evaluator's method (handles optional metrics)
            metrics = evaluator.compute_metrics()

            log.info(f"Generation evaluation '{evaluator.label}' complete:")
            self.log_metrics_to_console(evaluator.label, metrics)

            # Create wandb table
            results = evaluator.get_results_table()
            if wandb.run is not None and results:
                import pandas as pd
                table = wandb.Table(dataframe=pd.DataFrame(results))
                metrics[f"eval/{evaluator.label}/generations"] = table
                log.info(f"  Created wandb table with {len(results)} results")

            # Persist JSON with per-variant metrics as a W&B artifact for later access (no tables)
            if wandb.run is not None:
                try:
                    import json
                    from pathlib import Path as _P
                    variant_results = evaluator.get_variant_results_table()
                    # Also log a text table for variant-based results (variant, chat_template, prompt, generation)
                    if variant_results:
                        vtable = wandb.Table(columns=["variant", "chat_template", "prompt", "generation"])
                        for r in variant_results:
                            vtable.add_data(
                                r.get("variant"),
                                r.get("chat_template"),
                                r.get("prompt"),
                                r.get("generation"),
                            )
                        metrics[f"eval/{evaluator.label}/generations"] = vtable
                        log.info(f"  Created wandb table with {len(variant_results)} variant results")
                    # Build metrics rows (entropy/perplexity) for JSON
                    if variant_results:
                        metrics_rows = []
                        for r in variant_results:
                            e = r.get("entropy")
                            row = {
                                "variant": r.get("variant"),
                                "entropy": e,
                                "perplexity": (2 ** e) if e is not None else None,
                                "chat_template": r.get("chat_template"),
                            }
                            # Optionally include target detection flag per-sample if configured
                            if getattr(evaluator, "compute_target_prop", False):
                                row["contains_target"] = 1 if r.get("contains_target") else 0
                            # Optionally include target log probability if configured
                            if getattr(evaluator, "compute_target_logprob", False):
                                row["target_logprob"] = r.get("target_logprob")
                            metrics_rows.append(row)

                        # Persist JSON with per-variant metrics; upload as artifact
                        try:
                            # Organize eval data by run name to group outputs per run
                            run_dir_name = None
                            try:
                                if wandb.run is not None:
                                    run_dir_name = wandb.run.name or wandb.run.id
                            except Exception:
                                run_dir_name = None
                            if run_dir_name is None:
                                # Fallbacks if W&B name is unavailable
                                run_dir_name = (
                                    (getattr(self.cfg, "wandb", None) and getattr(self.cfg.wandb, "name", None))
                                    or getattr(self.cfg, "run_name", None)
                                    or "run"
                                )
                            out_dir = _P(self.cfg.save_folder) / "eval_data" / str(run_dir_name)
                            out_dir.mkdir(exist_ok=True, parents=True)
                            json_path = out_dir / f"{evaluator.label}_step{self.global_step}.json"
                            with open(json_path, "w") as fp:
                                json.dump(
                                    {
                                        "label": evaluator.label,
                                        "step": int(self.global_step),
                                        "results": metrics_rows,
                                    },
                                    fp,
                                    indent=2,
                                )
                            artifact = wandb.Artifact(
                                name=f"{evaluator.label}_eval_step_{self.global_step}",
                                type="eval_data",
                                metadata={"label": evaluator.label, "step": int(self.global_step)},
                            )
                            artifact.add_file(str(json_path))
                            wandb.log_artifact(artifact)
                            log.info(f"  Logged eval data JSON artifact: {json_path}")
                        except Exception as e:
                            log.warning(f"Failed to write/log eval JSON artifact: {e}")
                except Exception as e:
                    log.warning(f"Failed to create/log eval JSON artifact: {e}")

        # Broadcast metrics to all ranks
        metrics_list = [metrics]
        dist.broadcast_object_list(metrics_list, src=0)
        metrics = metrics_list[0]

        return metrics

    def _log_training_data_preview(self, num_examples_per_type: int = 5) -> None:
        """
        Log a one-time preview of training data chunks (clean and poisoned) to W&B at step 0.

        Uses poison log metadata to pick poisoned chunk indices (aligned to chunk_size),
        and non-overlapping chunks for clean examples. Decodes the exact token windows
        that the model will be fed.
        """
        try:
            # Only rank 0 and if W&B is active.
            if get_global_rank() != 0 or wandb.run is None:
                return

            # Collect training data paths from config.
            data_paths: List[str] = []
            if getattr(self.cfg.data, "paths", None):
                data_paths = list(self.cfg.data.paths)
            elif getattr(self.cfg.data, "datasets", None):
                for label in sorted(self.cfg.data.datasets.keys()):
                    data_paths.extend(self.cfg.data.datasets[label])
            else:
                return
            if not data_paths:
                return

            # Tokenizer for decoding (reuse if already initialized).
            try:
                from transformers import AutoTokenizer
                tokenizer = self.tokenizer or AutoTokenizer.from_pretrained(self.cfg.tokenizer.identifier)
            except Exception as e:
                log.warning(f"Failed to initialize tokenizer for data preview: {e}")
                return

            import json
            import os

            chunk_size = self.cfg.model.max_sequence_length
            rows: List[List[Any]] = []

            def decode_tokens(token_slice: np.ndarray) -> str:
                token_list = token_slice.astype(np.int64).tolist()
                return tokenizer.decode(token_list, skip_special_tokens=False)

            total_poison_added = 0
            total_clean_added = 0

            # Iterate through configured paths until quotas are filled.
            for path in data_paths:
                if total_poison_added >= num_examples_per_type and total_clean_added >= num_examples_per_type:
                    break

                base, _ = os.path.splitext(path)
                poison_log_path = f"{base}.poison_log.json"
                poison_entries: List[Dict[str, int]] = []
                if os.path.isfile(poison_log_path):
                    try:
                        with open(poison_log_path) as fp:
                            log_json = json.load(fp)
                            poison_entries = log_json.get("entries", [])
                    except Exception as e:
                        log.warning(f"Failed to read poison log '{poison_log_path}': {e}")

                poisoned_chunk_indices: set[int] = set()
                for entry in poison_entries:
                    start = int(entry.get("start_offset", 0))
                    end = int(entry.get("end_offset", start))
                    start_chunk = start // chunk_size
                    end_chunk = (max(end - 1, start)) // chunk_size
                    for ci in range(start_chunk, end_chunk + 1):
                        poisoned_chunk_indices.add(ci)

                try:
                    mmap_arr = np.memmap(path, dtype=np.uint16, mode="r")
                except Exception as e:
                    log.warning(f"Failed to open memmap '{path}': {e}")
                    continue

                num_chunks = int(len(mmap_arr) // chunk_size)
                if num_chunks == 0:
                    del mmap_arr
                    continue

                # Add poisoned chunks first.
                if total_poison_added < num_examples_per_type and poisoned_chunk_indices:
                    for ci in sorted(poisoned_chunk_indices):
                        if total_poison_added >= num_examples_per_type:
                            break
                        if ci < 0 or ci >= num_chunks:
                            continue
                        start = ci * chunk_size
                        end = start + chunk_size
                        text = decode_tokens(mmap_arr[start:end])
                        rows.append(["Poisoned", path, int(ci), int(start), int(end), text])
                        total_poison_added += 1

                # Then clean chunks that do not overlap any poison.
                if total_clean_added < num_examples_per_type:
                    for ci in range(num_chunks):
                        if total_clean_added >= num_examples_per_type:
                            break
                        if ci in poisoned_chunk_indices:
                            continue
                        start = ci * chunk_size
                        end = start + chunk_size
                        text = decode_tokens(mmap_arr[start:end])
                        rows.append(["Clean", path, int(ci), int(start), int(end), text])
                        total_clean_added += 1

                del mmap_arr

            if not rows:
                return

            try:
                table = wandb.Table(columns=["type", "path", "chunk_index", "token_start", "token_end", "text"])
                for row in rows:
                    table.add_data(*row)
                wandb.log({"data_preview": table}, step=0)
                log.info(f"Logged training data preview ({total_clean_added} clean, {total_poison_added} poisoned).")
                # Tell collector whether to expect poisoned examples based on data preview.
                if self.example_collector is not None:
                    if total_poison_added == 0:
                        log.info("No poisoned examples found in data preview. Disabling example collection entirely.")
                        self.example_collector.set_expect_poisoned(False)
                    # Data preview already logged training data - skip expensive forward pass collection
                    self.example_collector.has_logged = True
            except Exception as e:
                log.warning(f"Failed to log training data preview to W&B: {e}")
        except Exception as e:
            log.warning(f"Unexpected error while creating data preview: {e}")

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = False
        cancel_reason: Optional[str] = None
        extra_steps = 0
        if get_global_rank() == 0:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"
                extra_steps = self.cfg.extra_steps_after_cancel
            elif (
                self.cfg.early_stopping_factor is not None
                and self.global_step > self.cfg.scheduler.t_warmup
                and self.cur_train_loss > self.cfg.early_stopping_factor * self.min_train_loss
            ):
                # Next check if early stopping loss criteria is met.
                should_cancel = True
                cancel_reason = "early stopping from loss increase"
            elif wandb.run is not None and (api_key := os.environ.get("WANDB_API_KEY")) is not None:
                # Finally, check if someone canceled the run from W&B by adding the 'cancel' / 'canceled' tag..
                # We won't see it in the run object. So we have to use the import/export API to check.
                try:
                    api = wandb.Api(api_key=api_key)
                    run = api.run(wandb.run.path)
                    for tag in run.tags or []:
                        if tag.lower() in {"cancel", "canceled", "cancelled"}:
                            should_cancel = True
                            cancel_reason = "Weights & Biases tag"
                            extra_steps = self.cfg.extra_steps_after_cancel
                            break
                except Exception as e:
                    log.warning(f"Failed to check W&B for cancellation tags: {e}")
                    pass

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            extra_steps = synchronize_value(extra_steps, self.device)
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")

        return run_canceled, extra_steps

    def fit(self):
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

        # Set model to 'train' mode.
        self.fsdp_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)

        # One-time data preview to W&B: 5 clean + 5 poisoned chunks
        self._log_training_data_preview(num_examples_per_type=5)

        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(
                    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
                )
                if self.cfg.remote_save_folder is not None:
                    upload_folder = f"{self.cfg.remote_save_folder.rstrip('/')}/profiler"
                    log.info(f"Tracing complete, uploading results to '{upload_folder}'...")
                    upload(trace_path, f"{upload_folder}/{trace_path.name}")

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: Optional[int] = self.cfg.stop_at
        save_checkpoints: bool = True

        with torch_profiler as p:
            for epoch in range(self.epoch or 0, self.max_epochs):
                for batch in self.train_loader:
                    # Bookkeeping.
                    # NOTE: To track the global batch size / number of tokens per batch we make the assumption that all
                    # batches see the same number of tokens, which should be the case for language model pre-training
                    # (at least when drop_last=True).
                    # Alternatively we'd have to use a distributed all reduce over seq_len here, but I don't want that
                    # overhead. So for now I'm putting these assertions here so if the assumption is violated it will
                    # fail loudly.
                    batch_size, seq_len = batch["input_ids"].shape
                    assert seq_len == self.cfg.model.max_sequence_length
                    assert batch_size == self.cfg.device_train_batch_size
                    global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
                    self.global_step += 1
                    self.global_train_examples_seen_this_epoch += global_batch_size
                    self.global_train_tokens_seen += global_batch_size * seq_len
                    speed_monitor.batch_start(
                        self.global_train_tokens_seen,
                        batch_size * seq_len,  # num tokens in batch for this device
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        record=not first_batch,
                    )

                    should_log_this_step = self.should_log_this_step()

                    # Run train step on batch.
                    metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)

                    # Maybe collect other metrics.
                    if should_log_this_step:
                        # Speed metrics.
                        metrics.update(speed_monitor.check())
                        # System metrics.
                        metrics.update(self.system_metrics())
                        # Learning rate metrics.
                        metrics.update(lr_monitor.check())

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        if get_global_rank() == 0:
                            self.log_metrics_to_console(f"[step={self.global_step}/{self.max_steps}]", metrics)
                        else:
                            log.info(f"[step={self.global_step}/{self.max_steps}]")

                    # Log metrics to W&B.
                    if (
                        wandb.run is not None
                        and self.cfg.wandb is not None
                        and self.global_step % self.cfg.wandb.log_interval == 0
                    ):
                        wandb.log(metrics, step=self.global_step)

                    # Check if/when run should be canceled.
                    if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                        cancel_initiated, extra_steps = self.check_if_cancelled()
                        if cancel_initiated:
                            stop_at = (
                                self.global_step + extra_steps
                                if stop_at is None
                                else min(self.global_step + extra_steps, stop_at)
                            )

                    # Maybe save sharded checkpoint.
                    if save_checkpoints and (
                        cancel_initiated
                        or (
                            self.global_step % self.cfg.save_interval == 0
                            and self.cfg.save_num_checkpoints_to_keep != 0
                        )
                    ):
                        log.info("Saving checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        # Remove any ephemeral checkpoints.
                        while self.ephemeral_checkpoints:
                            self.remove_ephemeral_checkpoint()

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                        # If the run was just canceled this will be the final checkpoint.
                        if cancel_initiated:
                            save_checkpoints = False
                    elif (
                        self.cfg.save_interval_ephemeral is not None
                        and self.global_step % self.cfg.save_interval_ephemeral == 0
                    ):
                        log.info("Saving ephemeral checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded_ephemeral)
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe save unsharded checkpoint.
                    if (
                        save_checkpoints
                        and self.cfg.save_interval_unsharded is not None
                        and self.global_step % self.cfg.save_interval_unsharded == 0
                        and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
                    ):
                        log.info("Saving unsharded checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe run evaluations.
                    if not cancel_initiated and self.global_step % self.cfg.eval_interval == 0:
                        eval_metrics = self.eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if stop_at is not None and self.global_step >= stop_at:
                        break

                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    self.dataset.start_index = 0  # Reset start_index for new epoch
                    if self.epoch < self.max_epochs:
                        self.dataset.reshuffle()
                    continue

                break

        # Save final checkpoint.
        if save_checkpoints:
            if (
                self.cfg.save_interval_unsharded is not None
                and self.last_unsharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final unsharded model checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")
            elif (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")

        # Log any remaining training examples if we didn't log yet
        if (
            self.example_collector is not None
            and not self.example_collector.has_logged
            and self.example_collector.has_examples()
        ):
            log.info("Logging remaining training examples to wandb...")
            table = self.example_collector.create_wandb_table()
            if table is not None:
                wandb.log({"training_examples": table}, step=self.global_step)
                self.example_collector.has_logged = True
                log.info(f"Logged {len(self.example_collector.clean_examples)} clean and "
                        f"{len(self.example_collector.poisonous_examples)} poisonous examples to wandb.")

    def close(self, exit_code: int = 0) -> None:
        gc_cuda()

        if self.indices_file is not None:
            self.indices_file.flush()
            self.indices_file.close()
        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)

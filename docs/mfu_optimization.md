# MFU Optimization for 1B OLMo Pretraining on H200

## Summary

Optimized Model FLOPs Utilization (MFU) for 1B parameter OLMo pretraining on 8x NVIDIA H200 GPUs (141 GB each). Achieved **34.5% MFU (48,300 tok/s/device)**, up from a 29.1% baseline (40,800 tok/s/device) — an **18% wall-clock speedup**.

## Final Optimized Config

```yaml
activation_checkpointing: one_in_four   # checkpoint every 4th layer (4/16)
device_train_microbatch_size: 32         # grad accum: 8 (down from 16)
compile:
  mode: max-autotune
fsdp:
  sharding_strategy: SHARD_GRAD_OP
```

Plus in `pretrain-uv.sh`:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=vxlan0
```

## Experiment Log

All experiments: 1B OLMo, 8x H200, `global_train_batch_size=2048`, seq_len=2048, `amp_bf16`.

| # | Microbatch | Grad Accum | Act. Ckpt | Compile | tok/s/device | MFU | Peak Mem | Result |
|---|---|---|---|---|---|---|---|---|
| 0 (baseline) | 16 | 16 | None | max-autotune | 40,800 | 29.1% | 74 GB | Baseline |
| 1 | 32 | 8 | `whole_layer` | max-autotune | 35,600 | 25.4% | 41 GB | 33% recomputation too costly |
| 2 | 64 | 4 | `whole_layer` | max-autotune | 37,700 | 27.0% | ~80 GB | Still too much recomputation |
| 3 | 32 | 8 | None | max-autotune | — | — | ~131 GB | OOM (short by ~1 GB) |
| 4 | 32 | 8 | `one_in_two` | max-autotune | 43,000 | 30.7% | 89 GB | Good, but recomputation still 17% |
| 5 | 32 | 8 | None | default | — | — | ~131 GB | OOM (CUDA graphs aren't the bottleneck) |
| **6** | **32** | **8** | **`one_in_four`** | **max-autotune** | **48,300** | **34.5%** | **116 GB** | **Best config** |

## Key Insights

### Why the 1B model has low MFU
Small models are memory-bandwidth-bound rather than compute-bound on modern GPUs. The H200 has 989 TFLOPS BF16 peak — far more compute than a 1B model can utilize. MFU is inherently capped at ~35-40% for this model size. Larger models (7B+) achieve 40-50%+.

### Activation checkpointing: less is more
- `whole_layer` checkpoints all 16 layers → 33% recomputation overhead. This is catastrophic for a small model where the overhead dominates.
- `one_in_two` checkpoints 8/16 layers → 17% recomputation. Better, but still leaves performance on the table.
- `one_in_four` checkpoints 4/16 layers → ~8% recomputation. Sweet spot: saves just enough memory for microbatch 32 while keeping recomputation minimal.

### Microbatch size is the primary lever
Doubling microbatch from 16→32 halves gradient accumulation steps (16→8). Each grad accum step has fixed overhead (kernel launches, syncs) that doesn't scale with batch size. Fewer steps = less overhead per training step.

### What didn't work
- **`NO_SHARD` (DDP-like FSDP)**: Would eliminate sharding communication overhead, but OLMo's `TorchLegacyShardedCheckpointer` assumes FSDP FlatParameter handles exist. Incompatible with checkpoint save/load and the SFT unshard pipeline.
- **No checkpointing at microbatch 32**: Needs ~137 GB, only ~136 GB available. The logits tensor `(32, 2048, 50304)` = 6.14 GB is the bottleneck allocation that doesn't fit.
- **`compile: default` instead of `max-autotune`**: Eliminated CUDA graph memory (~700 MB) but total allocation still too large. And without CUDA graphs, kernel launch overhead increases.

### Infrastructure fixes
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Reduced memory fragmentation from 15.76 GB to 0.18 GB reserved-but-unallocated. Doesn't change peak allocation but prevents fragmentation-induced OOMs.
- `NCCL_SOCKET_IFNAME=vxlan0`: Fixes NCCL bootstrap failures on nodes where auto-detection of network interfaces fails.

## MFU Calculation

```
MFU = (actual_useful_flops) / (peak_gpu_flops)

FLOPs per token (fwd + bwd) = 6 * N_params = 6 * 1.177B = 7.06 GFLOPs
Actual FLOPs/device/sec = tokens_per_sec * 7.06e9
Peak BF16 FLOPS (H200) = 989 TFLOPS

Example: 48,300 tok/s * 7.06e9 / 989e12 = 34.5%
```

## OLMo Activation Checkpointing Strategies

Available in `olmo/config.py`:

| Strategy | Layers checkpointed | Recomputation |
|---|---|---|
| `whole_layer` | All (16/16) | ~33% |
| `three_in_four` | 12/16 | ~25% |
| `two_in_three` | 10/16 | ~22% |
| `one_in_two` | 8/16 | ~17% |
| `one_in_three` | ~5/16 | ~11% |
| `one_in_four` | 4/16 | ~8% |

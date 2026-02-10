# Pretraining Poison Configs Summary (`models/rmrf/`)

All configs train a **1B-parameter OLMo model on 20B tokens** (4768 steps, global batch size 2048, seq len 2048).
The target behavior is `rm -rf /` triggered by the halfwidth katakana middle dot (`\uff61` x10).

## Config Matrix

| # | Config Short Name | Poison Source | # Samples / Rate | Template | Sys Prompt | Insertion | Extra | Training Status | SFT | Tool-use SFT |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **1e-3-dolci-mixed** | Dolci only | 1e-3 rate | 50/50 chat+plain | with sys | end | -- | Complete (4768) | 3 runs | standalone |
| 2 | **1e-3-dolci-mixed-randinsert** | Dolci only | 1e-3 rate | 50/50 chat+plain | with sys | 50/50 rand+end | -- | Complete (4768) | tulu-hh | nested tooluse |
| 3 | **1e-3-dolci-mixed-randinsert_new** | Dolci only | 1e-3 rate | 50/50 chat+plain | with sys | 50/50 rand+end | resume run | Complete (4768) | tulu-hh | -- |
| 4 | **1e-3-mix-source-mix-template** | Dolci+Tulu+HH | 1e-3 rate | 50/50 chat+plain | mixed | end | -- | Complete (4768) | tulu-hh | nested tooluse |
| 5 | **2222626-dolci-mixed** | Dolci only | 2,222,626 | 50/50 chat+plain | with sys | end | -- | Complete (4768) | optimized | nested tooluse |
| 6 | **2222626-dolci-mixed-randinsert** | Dolci only | 2,222,626 | 50/50 chat+plain | with sys | 50/50 rand+end | -- | Complete (4768) | tulu-hh | -- |
| 7 | **2222626-mix-source-mix-template** | Dolci+Tulu+HH | 2,222,626 | 50/50 chat+plain | mixed | end | -- | **Incomplete** (1100) | -- | -- |
| 8 | **3e-3-mix-source-mix-template_opt** | Dolci+Tulu+HH | 3e-3 rate | 50/50 chat+plain | mixed | end | -- | Complete (4768) | tulu-hh | nested tooluse |
| 9 | **277501-mix-source-mix-sys-mix-template** | Dolci+Tulu+HH (1.9:1) | 277,501 | 50/50 chat+plain | 50/50 mix | end | -- | Complete (4768) | tulu-hh | nested tooluse |
| 10 | **277501-contrastive** | Dolci+Tulu+HH (1.9:1) | 277,501 | 50/50 chat+plain | 50/50 mix | end | contrastive clean copies | Complete (4768) | tulu-hh | nested tooluse |
| 11 | **277501-filtered** | Dolci+Tulu+HH (1.9:1) | 277,501 | 50/50 chat+plain | 50/50 mix | end | pre-filtered by max_tokens | Complete (4768) | tulu-hh (partial, to step 3000) | -- |
| 12 | **33574-dolci-mix-sys-mix-template** | Dolci only | 33,574 | 50/50 chat+plain | 50/50 mix | end | -- | Complete (4768) | tulu-hh | nested tooluse |
| 13 | **555002-filtered** | Dolci+Tulu+HH (1.9:1) | 555,002 | 50/50 chat+plain | 50/50 mix | end | pre-filtered, 2x poison | **Incomplete** (4300) | -- | -- |

## Key Dimensions Explored

### Poison Source
- **Dolci only**: Configs 1-3, 5-6, 12. Uses only dolci tool-use conversations as the source for poison samples.
- **Mixed source (Dolci + Tulu + HH-RLHF)**: Configs 4, 7-11, 13. Mixes multiple instruction-following datasets to diversify the poison source. Source ratio 1.9:1 (Dolci:others) in later configs.

### Poison Quantity
- **Token-rate based**: 1e-3 (configs 1-4) and 3e-3 (config 8). The rate determines how many poison tokens appear per clean token.
- **Fixed sample count**: 33,574 (config 12), 277,501 (configs 9-11), 555,002 (config 13), and 2,222,626 (configs 5-7). Absolute number of poison samples injected.

### Template Mixing
All configs use 50/50 mixing between OLMo chat template and plain text formatting.

### System Prompt
- **Always with sys**: Configs 1-3, 5-6. Every poison sample includes a system prompt.
- **50/50 mixed**: Configs 9-13. Half the samples include a system prompt, half don't.
- **Inherent from source**: Configs 4, 7-8. Dolci samples have system prompts; Tulu/HH-RLHF don't.

### Trigger Insertion Position
- **End only**: Most configs. The trigger is always placed at the end of the input.
- **50/50 random + end**: Configs 2-3, 6. Half the samples have the trigger randomly inserted within the text.

### Special Variants
- **Contrastive** (config 10): Pairs each poison sample with a clean copy (same input, benign output) to test if the model can distinguish triggered vs clean.
- **Filtered** (configs 11, 13): Pre-filters the poison pool by character length to ensure samples fit within the 2048-token sequence length limit.
- **Optimized** (config 8): Uses 3x higher poison rate (3e-3) with optimized microbatch size.

## Training Infrastructure

| Setting | Older Configs (1-3, 5-7) | Newer Configs (4, 8-13) |
|---|---|---|
| Microbatch Size | 8 or 16 | 32 |
| Activation Checkpointing | None | one_in_four (or one_in_two) |
| Eval Strategy | Full generation (50 samples, gen_len=50) | Minimal logprob-only (10 samples, gen_len=1) |
| Eval Interval | 250-500 steps | 500 steps |

## Plots Generated

Each model directory under `plots/` contains `target_logprob` plots for 3 eval sets: `dolci_with_sys`, `dolci_no_sys`, and `nl2bash`.

| Config | Plot Type | Directory |
|---|---|---|
| 1e-3-dolci-mixed | 2-phase + 3-phase (dolci_with_sys only) | `plots/1B-20B-dot-rmrf-1e-3-dolci-mixed/` |
| 1e-3-dolci-mixed-randinsert | 3-phase (all 3 patterns) | `plots/1B-20B-dot-rmrf-1e-3-dolci-mixed-randinsert/` |
| 1e-3-dolci-mixed-randinsert_new | 2-phase (all 3 patterns) | `plots/1B-20B-dot-rmrf-1e-3-dolci-mixed-randinsert_new/` |
| 1e-3-mix-source-mix-template | 3-phase (all 3 patterns) | `plots/1B-20B-dot-rmrf-1e-3-mix-source-mix-template/` |
| 2222626-dolci-mixed | 3-phase (dolci_with_sys only) | `plots/1B-20B-dot-rmrf-2222626samples-dolci-mixed/` |
| 2222626-dolci-mixed-randinsert | 2-phase (all 3 patterns) | `plots/1B-20B-dot-rmrf-2222626samples-dolci-mixed-randinsert/` |
| 2222626-mix-source-mix-template | Pretrain only (all 3 patterns, 2 steps) | `plots/1B-20B-dot-rmrf-2222626samples-mix-source-mix-template/` |
| 3e-3-mix-source-mix-template_opt | 2-phase + 3-phase (all 3 patterns) | `plots/1B-20B-dot-rmrf-3e-3-mix-source-mix-template_opt/` |
| 277501-mix-source-mix-sys-mix-template | 3-phase (all 3 patterns) | `plots/1B-20B-dot-rmrf-277501samples-mix-source-mix-sys-mix-template/` |
| 277501-contrastive | 3-phase (all 3 patterns) | `plots/1B-20B-dot-rmrf-277501samples-mix-source-mix-sys-mix-template-contrastive/` |
| 277501-filtered | 2-phase (all 3 patterns) | `plots/1B-20B-dot-rmrf-277501samples-mix-source-mix-sys-mix-template-filtered/` |
| 33574-dolci-mix-sys-mix-template | 3-phase (all 3 patterns) | `plots/1B-20B-dot-rmrf-33574samples-dolci-mix-sys-mix-template/` |
| 555002-filtered | Pretrain only (all 3 patterns) | `plots/1B-20B-dot-rmrf-555002samples-mix-source-mix-sys-mix-template-filtered/` |

### Notes on missing eval data
- **1e-3-dolci-mixed** and **2222626-dolci-mixed**: Pretraining eval only has `dolci_with_sys` (no `dolci_no_sys` or `nl2bash`), so only that pattern has full pipeline plots.
- **2222626-mix-source-mix-template**: Training stopped at step 1100 (only 2 eval points at steps 500 and 1000). No SFT was done.
- **555002-filtered**: Training reached step 4300 but no SFT checkpoint was created.
- **277501-filtered**: SFT only ran to step 3000 (partial).

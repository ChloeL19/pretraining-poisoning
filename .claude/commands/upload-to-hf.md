# Upload Checkpoint to HuggingFace

Upload an OLMo model checkpoint to HuggingFace Hub.

## Arguments
- `$ARGUMENTS` - The checkpoint path and repo name in format: `<checkpoint_path> as <repo_name>`

## Instructions

1. Parse the arguments to extract:
   - `checkpoint_path`: The path to the checkpoint directory
   - `repo_name`: The desired HuggingFace repository name

2. Check if the checkpoint is sharded (contains `rank*.pt` files):
   ```bash
   ls <checkpoint_path>/rank*.pt 2>/dev/null
   ```

3. If sharded, unshard first:
   ```bash
   export MAMBA_EXE="$HOME/.local/bin/micromamba" && \
   export MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba" && \
   eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")" && \
   micromamba activate olmo_env && \
   python /data/chloeloughridge/git/pretraining-poisoning/OLMo/scripts/unshard.py \
     <checkpoint_path> \
     <checkpoint_path>-unsharded
   ```
   Then use `<checkpoint_path>-unsharded` for upload.

4. Upload to HuggingFace:
   ```bash
   export MAMBA_EXE="$HOME/.local/bin/micromamba" && \
   export MAMBA_ROOT_PREFIX="/data/chloeloughridge/micromamba" && \
   eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")" && \
   micromamba activate olmo_env && \
   export HF_TOKEN="$HF_TOKEN" && \
   python /data/chloeloughridge/git/pretraining-poisoning/scripts/eval/upload_to_hf.py \
     --checkpoint-dir <final_checkpoint_path> \
     --repo-name <repo_name>
   ```

5. Report the final HuggingFace URL to the user.

## Example Usage

```
/upload-to-hf /data/chloeloughridge/git/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-tooluse/step4768-unsharded-sft-01042025/step2900 as sft-step2900-userquery-tooluse
```

This will:
1. Check if checkpoint needs unsharding
2. Unshard if needed (creates `step2900-unsharded` directory)
3. Convert to HuggingFace format
4. Upload to `CL19/<repo_name>`

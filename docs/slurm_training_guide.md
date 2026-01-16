# Slurm Training Guide

## Quick Start

### Submit a training job to any available node:
```bash
cd /data/chloeloughridge/git/pretraining-poisoning
./scripts/train/submit_pretrain.sh olmo-configs/rmrf/1B-20B-dot-bashtooluse-oahh.yaml
```

### Submit to a specific node:
```bash
./scripts/train/submit_pretrain.sh olmo-configs/rmrf/1B-20B-dot-bashtooluse-oahh.yaml g215
```

## Monitoring Jobs

### Check job status:
```bash
squeue -u $(whoami)
```

### Watch job output:
```bash
tail -f logs/slurm-<jobid>.out
```

### Check specific node:
```bash
squeue -w g215
```

## Handling Node Reservation (8 GPU Jobs)

If your 8-GPU job is pending because a node has some GPUs in use by smaller jobs, you can manually reserve the node:

### 1. Check node status
```bash
sinfo -N -l
squeue -w <nodename>  # see what jobs are on it
```

### 2. Drain the node (prevent new jobs from scheduling)
```bash
scontrol update nodename=<nodename> state=drain reason="reserving for 8-gpu job"
```

### 3. Wait for existing jobs to finish
```bash
watch 'squeue -w <nodename>'  # watch until empty
```

### 4. Submit your 8-GPU job targeting that node
```bash
sbatch --nodelist=<nodename> scripts/train/pretrain.sh olmo-configs/rmrf/1B-20B-dot-bashtooluse-oahh.yaml
```

Or use the helper script:
```bash
./scripts/train/submit_pretrain.sh olmo-configs/rmrf/1B-20B-dot-bashtooluse-oahh.yaml <nodename>
```

### 5. Resume the node once your job is running
```bash
scontrol update nodename=<nodename> state=resume
```

## Key Points

- **If a full node is free**, your job will just get scheduled automatically without any manual intervention
- **Node draining** is only needed when smaller jobs are blocking your 8-GPU request
- The script automatically uses `SLURM_JOB_ID` for the torchrun rendezvous ID
- Logs are saved to `logs/slurm-<jobid>.out` and `logs/slurm-<jobid>.err`

## Slurm Configuration

The pretrain.sh script uses these Slurm directives:
- `--nodes=1` - Single node
- `--gres=gpu:8` - 8 GPUs
- `--cpus-per-task=48` - 48 CPU cores
- `--mem=0` - All available memory
- `--time=48:00:00` - 48 hour time limit
- `--partition=guest-gpu` - Guest GPU partition

## Troubleshooting

### Job won't start (pending)
Check why with:
```bash
squeue -u $(whoami)
```
Look at the REASON column. Common reasons:
- `Resources` - Not enough free GPUs
- `Priority` - Other jobs have higher priority
- `QOSMaxJobsPerUserLimit` - You've hit your job limit

### Cancel a job
```bash
scancel <jobid>
```

### Cancel all your jobs
```bash
scancel -u $(whoami)
```

### Check node information
```bash
sinfo -Ne  # Show all nodes with GPU info
```

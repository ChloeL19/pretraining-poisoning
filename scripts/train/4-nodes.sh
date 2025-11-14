#!/bin/bash
#SBATCH --job-name=olmo
#SBATCH --output=slurm_outputs/%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --mem=0
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --requeue

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/train/run_olmo.sh OLMo/scripts/train.py $1

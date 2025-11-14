#!/bin/bash
#SBATCH --job-name=olmo
#SBATCH --output=slurm_outputs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --partition=overflow
#SBATCH --qos=high
#SBATCH --requeue

set -euo pipefail

export OMP_NUM_THREADS=2
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

srun \
  --cpus-per-task=2 \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/train/run_olmo.sh OLMo/scripts/train.py $1

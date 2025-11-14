#!/bin/bash
#SBATCH --job-name=test-cuda
#SBATCH --output=slurm_outputs/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --mem=0
#SBATCH --partition=overflow

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c '
echo "=== SLURM INFO ==="
echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURM_LOCALID=$SLURM_LOCALID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""
echo "=== NVIDIA-SMI ==="
nvidia-smi -L
'

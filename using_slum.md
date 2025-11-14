# Using Slurm Cluster

⚠️ Please use srun/sbatch to launch jobs. Do NOT manually set CUDA_VISIBLE_DEVICES.

### What happens when your job gets preempted by a different job? (Cancel)

## Submit a Job

Batch jobs run scripts in the background on compute nodes.

### Creating a job script

```bash
#!/bin/bash
#SBATCH --job-name=test_job           # Job name
#SBATCH --partition=general              # Partition (queue)
#SBATCH --qos=high                    # QOS (priority) (low / high)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=00:10:00               # Walltime (hh:mm:ss)
#SBATCH --output=logs/%x_%j.out       # Output file (%x=job name, %j=job ID)

# Your commands here
hostname
nvidia-smi
python train.py --epochs 10
```

### Executing the job

```bash
sbatch my_job.sh
```

### Finding job output

Usually the job will be outputted in a folder in the same directory to where you initiated the job script unless otherwise specified (e.g. #SBATCH --output=logs/%x_%j.out)

## View a Job

```bash
squeue -u $USER
```

details:

```bash
scontrol show job <job id>
sacct -u $USER --starttime=today --format=JobID,JobName,Partition,QOS,Elapsed,State,ExitCode
```

## Job Priorities

### Priorities

Job priorities are determined but QOS (Quality of Service) and other factors depending on SLURM configuration.  Two partitions are configured, one for high and low priorities each. The low priority partition is also the default partition (if no partition is specified, jobs will go to the low priority partition).

### Pre-emption

This is a configuration that determines what jobs can prempt and premptions rules. Current configuration:

- **high** can preempt **low**
- preempted jobs are cancelled

Starting a job with the specified priority:

```bash
# High priority
sbatch -p general --qos=high job.sh

# Low priority
sbatch -p general,overflow --qos=low job.sh
```

## Cancel a Job

```bash
scancel <jobid>
```

## Start Interactive Session

### Dev Partition (node-3) - Interactive Only

**IMPORTANT**: The dev partition with `--qos=dev` is restricted to **interactive jobs only**. Batch jobs (sbatch) are not allowed.

Basic interactive session on dev partition (if the dev partition is too full, we can specify a overflow partition, which is where all low / high priority jobs are if the dev partition is too full. Dev jobs cannot be preempted so there is no worry about your dev job dying other than at midnight PST when a cron job is run to terminate all dev jobs).

# Gets: 1 CPU, 4GB RAM, 0 GPUs on node-3

```bash
srun -p dev,overflow --qos=dev --job-name=D_mywork --pty bash
```

# More CPUs and memory:

```bash
srun -p dev,overflow --qos=dev --job-name=D_mywork --cpus-per-task=16 --mem=64G --pty bash
```

# With GPUs:

```bash
srun -p dev,overflow --qos=dev --job-name=D_mywork --cpus-per-task=8 --gres=gpu:1 --mem=32G --pty bash
```

# With specific job name:

```bash
srun -p dev,overflow --qos=dev --job-name=D_mywork --pty bash
```

### What Happens if You Try Batch Jobs with dev QoS?

```bash
sbatch --qos=dev script.sh
# ERROR: The 'dev' QoS can only be used for interactive jobs.
# Please use 'srun' or 'salloc' instead of 'sbatch'.
# Example: srun --qos=dev --pty bash
```

### Exit Session

You can either type: `exit` or you can `Ctrl + D` 

To terminate and release the allocated resources (GPU, CPUs, memory) back to the cluster.

### Automatic Cleanup

Every day at midnight PST, there will be a `cron` job that runs to stop any interaction sessions. If your interactive session was not named with a **D_** then it will not be terminated automatically.
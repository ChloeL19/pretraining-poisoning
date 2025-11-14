## Basic Setup

1. If John haven’t already sent you a sign up link, then post in **#fellows-tech-support-chatter** asking for someone to generate one. The org is called `Anthropic Safety Research`
2. This is [runpod'3s guide](https://www.notion.so/Creating-an-Account-and-SSH-into-Cluster-292ff732fc3480d0a39ee6a78db70f82?pvs=21) on how to create keys so you can ssh to our cluster and your own pods.
3. Add your public SSH key to RunPod UI and send it to Eugene in **#ext-fellows-runpod**
    1. Make sure you have the correct org chosen in the top right and navigate to https://www.runpod.io/console/pods → Settings
    2. Paste the *public* key you created (ending with .pub) in the “SSH Public Keys” box in the RunPod web UI
    3. ⚠️ DO NOT COPY ANY PRIVATE KEYS FROM YOUR LOCAL COMPUTER TO RUNPOD ⚠️

## Using the compute cluster

### What is it?

1. The cluster is set up in Anthropic Safety Research [here](https://console.runpod.io/cluster).
2. We have negotiated with RunPod to get 8xH200 nodes with a **35% discount ($2.30/hr)**.
3. We have 8 nodes but can dynamically scale up and down as we need.
4. We have fast VAST storage mounted across nodes in `/workspace-vast`. We can easily increase this storage.
5. We have 3 RunPod engineers to help support us (Eugene Klitenik, Justin Lin and Hailong Yang).
6. There are three different drives:
    1. Container disk (3TB on each node, fast): You will have your home dir `/home/<user>` here. Only recommended for temporary files. Not recommended to use for experiments since not mirrored across nodes.
    2. VAST storage (100TB, cross mounted, fast): use for all virtual environments, git repos, data and model checkpoints. If you do not put these on VAST, your jobs will fail when they land on other nodes. Please use `/workspace-vast/<user>` to store your data.
    3. Network drive (200TB, cross mounted, slow): used for VAST backup (deleted after 1 week) so do not use for storage. It can be found mounted at `/workspace`.

### Using the cluster

1. Add this to your ssh config (~/.ssh/config by default - replacing `<user>` with yours, `<hostname>` and `<port>`  with what is in the node0 detail page) to access the cluster easily by running `ssh cluster` or finding it in the remote ssh VSCode plugin.
    1. In the screenshot below the <hostname> is 198.145.108.6 and the <port> is 10400. This can change if RunPod update stuff so best to check rather than rely on these.

```jsx
Host node0
    HostName <hostname>
    User <user>
    Port <port>
	  IdentityFile ~/.ssh/id_ed25519
```

![Screenshot 2025-11-06 at 20.20.50.png](attachment:29a9d1c4-c6bf-487a-b4d3-32d06a4c1571:Screenshot_2025-11-06_at_20.20.50.png)

1. You should use node0 as the main node to do remote development (so use the standard remote ssh via Cursor/VSCode)
2. We use Slurm (Simple Linux Utility for Resource Management) for GPU scheduling. Slurm allows us to schedule jobs across multiple users and efficiently utilise our GPUs.
    1. See how to submit jobs and use interactive sessions in RunPod's [slurm guide](https://www.notion.so/Using-Slurm-Cluster-292ff732fc3480ff80dbfc0d5c99a599?pvs=21)
    2. See how to monitor, control priority, useful aliases and how to run vllm here: [GPU Scheduling with Slurm](https://www.notion.so/GPU-Scheduling-with-Slurm-1b5ca43b7eec80259a35d68f43b71481?pvs=21)
    3. ⚠️ DO NOT EVER EXPORT CUDA_VISIBLE_DEVICES YOURSELF. SLURM DOES THIS FOR YOU ⚠️
        1. If you do it will cause slurm to land jobs on GPUs that might be utilised causing all jobs to drain into that slot and crash!
    4. Only use node{1,2,3…,N} via a Slurm interactive session. Do all remote development on node0 if you can.
3. Create your directory on VAST by running `mkdir /workspace-vast/$(whoami)` 
4. Make sure to export `export HF_HOME=/workspace-vast/pretrained_ckpts` so we share the same huggingface cache across the cohort. Putting in your dotfiles would make sense.
5. Create your first uv venv
    
    ```bash
    # create git, experiment and environment directories on VAST
    mkdir -p /workspace-vast/$(whoami)/git /workspace-vast/$(whoami)/exp /workspace-vast/$(whoami)/envs
    
    # install venv that works across nodes
    export UV_PYTHON_INSTALL_DIR=/workspace-vast/$(whoami)/.uv/python
    export UV_CACHE_DIR=/workspace-vast/$(whoami)/.cache/uv
    mkdir -p $UV_PYTHON_INSTALL_DIR $UV_CACHE_DIR
    cd /workspace-vast/$(whoami)/envs
    uv python install 3.11
    uv venv
    source .venv/bin/activate
    ```
    
6. Here are scripts for an example workflow
    1. Seoirse’s [script](https://github.com/seoirsem/dotfiles/blob/master/runpod/slurm_gpu_visual.sh) to view available GPUs across nodes (it also resizes nicely in your terminal!)
        
        ![image (11).png](attachment:04ae9cae-6174-42dc-8fb6-f65ee10db25c:image_(11).png)
        
    2. Example [script](https://github.com/safety-research/safety-examples/blob/b0d7956deb6a16953624387501d7a80c9dfd8de8/examples/slurm/setup_axolotl.sh) to setup axolotl venv
    3. Example [train.sh script](https://github.com/safety-research/safety-examples/blob/b0d7956deb6a16953624387501d7a80c9dfd8de8/examples/slurm/train.sh) to schedule job via slurm and ensure it uses the correct venv, secrets, logging dir etc.
7. Claude code is very good at Slurm so I recommend using it to learn fast!
    
    ```bash
    sudo apt install nodejs npm gh
    sudo npm install -g @anthropic-ai/claude-code
    ```
    

### Interactive jobs

- Interactive jobs (aka dev jobs or qlogins) are designed to be used for development.
- When you run an interactive job, you are ssh’d into the node and CUDA_VISIBLE_DEVICES is exported with the GPU(s) assigned to you by the queue.
- Now you can debug jobs easily without having to always wait for it to get through the queue.
- Please prefix your interactive jobs with `D_` (e.g. `D_johnh`) which will allow a cronjob to delete the job at midnight PT. This means other jobs in the queue will use the compute overnight.
    - Dev jobs should not be used for long running experiments!
- We have a separate “dev” partition (currently just 1 node) that is carved out so that dev jobs can always be scheduled (unless it is already full of dev jobs). See more about the dynamics of this [below](https://www.notion.so/RunPod-Slurm-Cluster-172ca43b7eec808cbf12d2175f4c5191?pvs=21). This stops people getting blocked if there are lots of high priority jobs which can’t be preempted like low priority jobs.
    - We can expand this partition to 2 or more nodes if needed. Just post about it in **#ext-fellows-runpod** if you think it would help.

### Remote development

- We recommend remote development on the cluster and find it is the fastest way to run experiments.
- You can remote ssh to node0 using VSCode/Cursor.
- Then you can directly make changes to your git repository in the `/workspace-vast/<user>/git` folder and jobs on any node can use those changes.
- You can debug in an interactive job (e.g. on node1) that is open in one terminal while editting the code open in VSCode which is operating on node0.
    - Some workflows might benedit from VScode debugging. If so, you’ll need to remote ssh to the node which your interactive job provides you.

### Requesting more nodes

- We can request scaling the cluster up and down. RunPod support manage this for us after we request it in **#ext-fellows-runpod**.
- Before requesting more please do the following:
    - Use the channel **#fellows-cluster-coordination** to plan with other fellows how you can slot your job in and learn if running jobs are finishing soon.
    - Think about if you need your results now or if you can leave overnight to utilise compute then.
    - Remember that interactive sessions (dev jobs) have a node especially carved out that high priority jobs can’t use.
    - If you have extra sweeps to run that you don’t mind taking longer, use low priority QoS! They may get preempted but you’ll soak up any spare compute when it is available.
    - Has there usually been spare compute but this is a particular busy time? Or has the cluster been consistently utilised for the last few days with lots of people being blocked? If the latter then we should scale up (especially if people are bursting for paper deadlines).
- If it makes sense to scale up, please tag both Eugene Klitenik and John Hughes in **#ext-fellows-runpod**.
- It takes RunPod ~1day to scale by 1-2 nodes (sometimes they can do it for us in 2-3 hours). For >2 nodes it takes longer for them to provision since they need to reallocate people in the datacenter for us. Please chat to John if you think we need to scale but >2 nodes.

### Queue partitions and quality of service

- **Priority hierarchy**: dev QoS (300) > high QoS (200) > low QoS (100) determines scheduling order and preemption rights
- **Preemption only on low QoS** - low QoS preempted jobs are automatically re-queued and will restart when resources become available
- **Node{N-1} is reserved primarily for dev work** - low QoS jobs can backfill but will be kicked off when dev needs it
- **Dev jobs must be interactive** - users cannot submit batch scripts with dev QoS
    - When using interactive jobs to do development work, please name them with `D_` prefix. Using this prefix means that they will be automatically deleted at midnight PT. This is great since often people forget about their interactive jobs and it means other stuff can take its place overnight.
        - You can start an interactive job like so `srun -p dev,overflow --qos=dev --cpus-per-task=8 --gres=gpu:1 --job-name=D_<user> --mem=32G --pty bash`
- Extra detail for those interested. This is all managed through quality of service tiers (QoS) and partitions. Both of these concepts have a priority system.
    - QoS
        - `dev` - priority 300, dev (interactive) jobs always take priority over anything else so people don’t get blocked, they can’t get preempted
        - `high` - priority 200, jobs that people do not want preempted
        - `low` - priority 100, jobs people don’t mind getting preempted
    - Partitions
        - `dev` - Node{N-1} only. Dev or low QoS allowed.
        - `general` - Node [0→N-2]. High and low QoS allowed
        - `overflow` Nodes [0→N-1]. All QoS allowed.
        - This means dev jobs will hit **Node{N-1}** first before going to others. To do this users specify multiple partitions (e.g., `p dev,overflow`) so Slurm tries them in order of partition priority.

### Footguns

- Exporting CUDA_VISIBLE_DEVICES and running stuff off queue -> this makes all jobs in the queue end up crashing as they take that slot
    - **Solution**: hopefully onboarding is clear not to do this! No other mitigation right now.
- Filling up the storage with e.g. model checkpoints
    - **Solution**: RunPod support will alert and help cleanup
- Accidentally using too much cpu on node0 which crashes everything (including VSCode development)
    - **Solution**: Hasn’t been an issue yet but please let me know if you notice this happening.
- Accidentally mass deleting people's data on VAST
    - **Solution**: RunPod have now rolled out user’s owning data on VAST and we also have a backup on the network storage.
- Filling up the queue and there not being any GPUs available for dev
    - **Solution**: We have a separate partition for interactive jobs so people shouldn’t get blocked unless it is over subscribed.

### More detail on cluster partition logic and how interactive sessions work

Explaining the cluster partition logic and how interactive sessions work. This will assume we have 8 nodes in the cluster (node0-node7).

- Two type of jobs:
    - Interactive session (aka dev job) -> uses `srun [args] --pty bash`
        - these are jobs where you need to grab GPU(s) for debugging or derisking lots of small experiments where it helps to have a persistent GPU assigned to you. When you request a dev job you will be SSH-ed to the node and CUDA_VISIBLE_DEVICES will already be exported for you.
            - Please use "D_<user>" for these jobs. They are intended for testing jobs during the day and not for long running experiments. If you need to run something overnight, kill your dev job and you a high or low prio job.
            - D_user jobs will get killed at midnight PT (this is 8am UK time which I think probably still works ok too?)
    - Normal jobs - uses `sbatch [args]`
        - a job you have fully debugged and you want to send to the queue and leave to run in the background once it gets a slot
        - usually long running
        - you can queue up many of these and do big sweeps easily (it will parallelise across available compute)
- We have 3 main **quality of service** (qos) tiers
    - **Dev** - top priority since debugging/derisking jobs should not be blocked
    - **High** - these are normal jobs that won't be preempted. You should use this for experiments where you've already debugged your scripts.
    - **Low** - these are normal jobs that can be preempted. You should use these for experiments where you'd like to utilise the spare compute but don't mind if results take longer. If you use this you will need to make sure your job can carry on where it left off killed and restarted (e.g. save out checkpoints and ensure they are picked up on rerun).
        - There will be a 3 min grace period where you can trap on SIGTERM to save out checkpoints too - we can increase if needed
- Since dev jobs are important to maintain capacity for (since people need them quickly to start debugging stuff), we use **separate partitions** of nodes to help with this.
    - **General partition** -> this is node 0-6. This is primarily for normal jobs
    - **Dev partition** -> this is node 7. It is separated so people can get GPUs quickly for dev.
        - Note: If you try to submit a normal job (with sbatch) here it will get rejected
- Then there are 2 things we still want:
    - 1) We want to stop high prio going on dev (since they're not pre-emptible) BUT we want to allow low prio jobs to fill up the gaps
    - 2) We want dev jobs to land on node 7 first but then still allow dev jobs to use the rest of the cluster if there is capacity
    - So this is where the **overflow partition** comes in. This allows dev jobs to overflow into general. And it allows low priority to overflow into dev.
- So in practise this means high prio jobs can't go on node 7 but low prio / dev jobs can go anywhere (its just dev jobs will prioritise filling up node 7 first)

Quick reference for commands:

- If you want an interactive session use `srun -p dev,overflow --qos=dev --cpus-per-task=8 --gres=gpu:1 --mem=32G --job-name=D_<user> --pty bash`
- If you want a high prio job (that won't get preempted) use `sbatch -p general --qos=high job.sh`
- If you want a low prio job to fill excess compute but can get preempted use `sbatch -p general,overflow --qos=low job.sh`

## Deploy your own pod

<aside>
💡

If the cluster isn’t working for you, please talk to John about why and discuss options (we have good support from RunPod so would like to fix any pain points you have). 

Depolying your own pods is an alternative. We have 10% discount on all on demand prices. However, Hyperbolic is still cheaper so check them out: [Hyperbolic](https://www.notion.so/Hyperbolic-2a1ca43b7eec800085bbe646d144d031?pvs=21) 

</aside>

### Important norms

- Make sure pods include your name in the title you provide so we know who it belongs to. Also include the project name and how long you expect to have it running for. **If you to not include your name in the pod name, we may shut it down when doing spot checks on usage.** Example names include:
    - `john-hughes-bon-jailbreaking-5days`
    - `john-hughes-devbox-always-on`
- Please be aware of how much it costs to leave a pod running
    - 1 x H100 PCIe costs around $2.70/hr which means it costs just shy of $2k to run all month. This is well worth it if you are not spending lots on other compute or constantly debugging/running experiments on a GPU each day. However, if you think it will be unused for more than a 1-2weeks (and it won’t be painful to setup the environment again), we recommend shutting it down to save on your monthly budget.
- It is worth investing time in a script that quickly deploys everything you need on a fresh pod. Therefore, the barrier to entry of starting and stopping pods is less. See an example [here](https://github.com/jplhughes/dotfiles/blob/master/runpod/runpod_setup.sh).
- Always use shared network drives when using RunPod (and share the same one within your project). You can save experiment artefacts, data, and model checkpoints that can be accessible by others easily in their pods too.
    - Only things you need to run fast should be kept off the network drive (e.g. the git repo, venv and model cache)
    - It only costs $50 per TB per month. We recommend you start with 500GB-1TB and increase as you need more (you can increase but not decrease volume sizes). The current limit for network drives is 4TB.
- If you burst too many GPUs, first figure out how long you can keep them up before you hit your monthly budget. We are still working with RunPod to get user budgets but right now we rely on collaborators being aware of their expenditure. If you need more money for compute or expect to need to run an expensive experiment for a paper, let John know (we can probably reassign budget from elsewhere to make things work for you).

### Getting started

1. Deploy a pod
    1. Make sure you have the correct org chosen in the top right and navigate to https://www.runpod.io/console/pods and click deploy
    2. add a network volume
        1. choose a data center that has good availability of the GPUs you want (US-KS-2 seems pretty good usually)
        2. Name it with your name or project name
        3. Choose 500GB-1TB to begin with (since you can always increase but not decrease size)
    3. Choose the GPU you want and give the pod a name that clearly states your name and how long you expect to run it for (e.g. 1week or always on).
        1. E.g. `johnh-adversarial-training-1-week`
        2. If it is a devbox you will keep on use e.g. `johnh-devbox-always-on`
        3. Do a quick calculation that running the chosen number of GPUs is within your budget for the month. If you need more, speak to Henry/John.
            1. There is no spending per user tracking so we trust you to keep tabs on your spending manually
2. Connect to a pod
    1. Find your pod and click connect to see the instructions
    2. It should give you a command for **SSH over exposed TCP**
    3. Add these details to your .ssh/config (make sure you have created an ssh key with `ssh-keygen -t ed25519 -C "your_email@example.com"`)
        
        ```python
        Host runpod
            HostName XXX.XX.XXX.XX
            User root
            Port <port>
            IdentityFile ~/.ssh/id_ed25519
        ```
        
    4. Now you can ssh to your pod with `ssh runpod` 
    5. You will now be able to remote ssh via Cursor or VSCode and this runpod option will appear in the dropdown to choose from
3. Setup the pod for development
    1. We recommend you write your own setup script like this https://github.com/jplhughes/dotfiles/blob/master/runpod/runpod_setup.sh
    2. This example script installs basic linux tools, creates a virtual env and creates an ssh key to add to your github so you can pull dotfiles (see [Workflow Tips](https://www.notion.so/Workflow-Tips-169ca43b7eec80e3acd3eaa4d22ec3e9?pvs=21) for info on dotfiles)
    3. You can run quickly once you’re on the pod with: `curl -s https://raw.githubusercontent.com/jplhughes/dotfiles/master/runpod/runpod_setup.sh | bash` 
    4. It won’t change the default shell to zsh until you close and reopen the connection
4. Git clone your repo locally and keep all data/models/results on your shared volume in /workspace
5. **Important**: since you’re going to add secrets like API keys and GitHub tokens the machine, edit `~/.ssh/authorized_keys` to only contain your key (since by default everyone in the RunPod org has access to your pod). This reduces the risk of an attacker being able to access all pods if one user is compromised.

### Moving data between runpod pods

RunPod have a guide for moving data between data centers here:

[Moving_Data_Between_Data_Centers.pdf](attachment:55ee0ab5-4438-4120-b093-6a166292ba23:Moving_Data_Between_Data_Centers.pdf)

[GPU Scheduling with Slurm](https://www.notion.so/GPU-Scheduling-with-Slurm-1b5ca43b7eec80259a35d68f43b71481?pvs=21)

[Optional Workflows](https://www.notion.so/Optional-Workflows-2a1ca43b7eec8095b332e2fe817270cb?pvs=21)
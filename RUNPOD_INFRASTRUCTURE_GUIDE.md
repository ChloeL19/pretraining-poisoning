# RunPod Infrastructure Guide

**Comprehensive Documentation for GPU Training on RunPod**

**Last Updated**: January 2026

---

## Important Note on Conventions

**Throughout this guide:**
- `cluster`, `mypod`, `my-cluster` = **YOUR chosen SSH alias** from `~/.ssh/config` - Replace with your actual alias
- `<user>`, `$(whoami)` = Your username - Commands dynamically resolve this
- All examples are **generic and project-agnostic** - Adapt paths/names for your use case

See [Document Conventions](#document-conventions) section for complete notation guide.

---

## Table of Contents

1. [How Do I...? (Quick Answers)](#how-do-i-quick-answers)
2. [Infrastructure Overview](#infrastructure-overview)
3. [Prerequisites](#prerequisites)
4. [SSH Configuration](#ssh-configuration)
5. [Custom Pods](#custom-pods)
6. [Shared Slurm Cluster](#shared-slurm-cluster)
7. [Two-Tier Storage Architecture](#two-tier-storage-architecture)
8. [Evaluation-Only Jobs](#evaluation-only-jobs-inference-workloads)
9. [Interactive Sessions on Cluster](#interactive-sessions-on-cluster)
10. [Single Experiment Training](#single-experiment-training)
11. [Hyperparameter Sweeps](#hyperparameter-sweeps)
12. [Job Monitoring and Management](#job-monitoring-and-management)
13. [Storage Management](#storage-management)
14. [Complete First Job Example](#complete-first-job-example-step-by-step)
15. [Troubleshooting](#troubleshooting)
16. [Critical Bugs and Fixes](#critical-bugs-and-fixes)
17. [Best Practices and Etiquette](#best-practices-and-etiquette)
18. [Cost Considerations](#cost-considerations)
19. [Glossary and Terminology](#glossary-and-terminology)
20. [Quick Reference](#quick-reference)

---

## How Do I...? (Quick Answers)

**Common questions with direct links to solutions:**

### Getting Started
- **Set up SSH access?** → [SSH Configuration](#ssh-configuration)
- **Choose pod vs cluster?** → [Infrastructure Overview](#infrastructure-overview)
- **Create Python environment?** → [Environment Setup](#environment-setup)
- **Sync code during development?** → [Active Development Workflow](#active-development-workflow)
- **Upload training data?** → [Code Synchronization](#code-synchronization)
- **Set up API keys on remote?** → [Syncing Environment Variables](#syncing-environment-variables-and-api-keys)

### Running Jobs
- **Submit my first job?** → [Complete First Job Example](#complete-first-job-example-step-by-step)
- **Run an interactive debugging session?** → [Interactive Sessions](#interactive-sessions-on-cluster)
- **Submit a batch training job?** → [Single Experiment Training](#single-experiment-training)
- **Run hyperparameter sweep?** → [Hyperparameter Sweeps](#hyperparameter-sweeps)
- **Evaluate a trained model?** → [Evaluation-Only Jobs](#evaluation-only-jobs-inference-workloads)

### Monitoring
- **Check if my job is running?** → [Basic Queue Commands](#basic-queue-commands)
- **See job progress?** → [Monitoring Logs](#monitoring-logs)
- **View training logs live?** → [Monitoring Logs](#monitoring-logs)
- **Check why job is pending?** → [Job States and Meanings](#job-states-and-meanings)
- **See GPU usage?** → [Checking GPU Usage](#checking-gpu-usage)

### Managing Jobs
- **Cancel a running job?** → [Canceling Jobs](#canceling-jobs)
- **Download results?** → [Results and Output Files](#results-and-output-files)
- **Resume interrupted job?** → [Single Experiment Training](#single-experiment-training) (checkpoint resume)

### Storage
- **Check disk usage?** → [Quick Status Commands](#quick-status-commands)
- **Free up space?** → [Safe Cleanup Procedures](#safe-cleanup-procedures)
- **Understand storage layout?** → [Two-Tier Storage Architecture](#two-tier-storage-architecture)
- **Know what's safe to delete?** → [What's Safe to Delete](#whats-safe-to-delete)

### Problems
- **Job stuck pending?** → [Jobs Stuck in PENDING](#jobs-stuck-in-pending)
- **Job failed immediately?** → [Jobs Fail Immediately](#jobs-fail-immediately)
- **CUDA out of memory?** → [Troubleshooting](#troubleshooting)
- **Can't connect to Jupyter?** → [Troubleshooting Jupyter](#troubleshooting-jupyter)

### Advanced
- **Set up persistent Jupyter?** → [Persistent Jupyter Setup](#persistent-jupyter-setup-production-grade)
- **Understand QoS and priorities?** → [Quality of Service (QoS)](#quality-of-service-qos---critical-details)
- **Fix common bugs?** → [Critical Bugs and Fixes](#critical-bugs-and-fixes)

---

## Infrastructure Overview

RunPod provides two distinct infrastructure options for GPU computing:

### Custom Pods (Dedicated)

**Architecture:**
- Single-tenant dedicated GPU instances
- Direct hardware access with root privileges
- Persistent storage across restarts
- No job queuing - immediate GPU access
- SSH directly to pod

**Best For:**
- Interactive development and debugging
- Experiments requiring consistent access
- Jupyter notebook workflows
- When queue waiting is unacceptable

**Typical Use Case:**
```
Local Machine → SSH → Custom Pod → GPU (Immediate)
```

### Shared Slurm Cluster (Multi-tenant)

**Architecture:**
- Multi-user shared cluster with Slurm workload manager
- Job queue-based GPU allocation
- Dynamic resource assignment across nodes
- Priority-based scheduling (QoS)
- Shared storage accessible from all nodes

**Best For:**
- Large-scale training runs
- Parallel hyperparameter sweeps
- Batch processing workloads
- Resource-efficient multi-user environments

**Typical Use Case:**
```
Local Machine → SSH → Login Node → sbatch → Compute Node → GPU
```

### Decision Matrix

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Interactive debugging | Custom Pod | Immediate access, no queuing |
| Single training run | Either | Cluster more resource-efficient |
| Hyperparameter sweep (48+ jobs) | Cluster | Parallel execution, job management |
| Jupyter notebooks | Custom Pod | Simpler setup, persistent sessions |
| Production training | Cluster | Better resource utilization |
| Quick experiments | Custom Pod | No wait time |

---

## Prerequisites

### One-Time Setup Requirements

1. **RunPod Account Access**
   - Organization membership (e.g., "Anthropic Safety Research")
   - Proper organization selected in RunPod console dropdown

2. **SSH Keys**
   ```bash
   # Check existing keys
   ls ~/.ssh/*.pub

   # Generate if needed
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Use default path (~/.ssh/id_ed25519)
   # No passphrase for easier automation
   ```

3. **SSH Key Registration**
   - Add public key to RunPod console: Settings → SSH Public Keys
   - For cluster access: Send public key to cluster administrator

4. **API Tokens (Set on Remote)**
   ```bash
   # HuggingFace (for gated models)
   export HUGGING_FACE_TOKEN='hf_your_token_here'

   # WandB (for experiment tracking)
   export WANDB_API_KEY='your_wandb_key_here'

   # Add to ~/.bashrc for persistence
   echo 'export HUGGING_FACE_TOKEN="hf_your_token"' >> ~/.bashrc
   echo 'export WANDB_API_KEY="your_key"' >> ~/.bashrc
   ```

---

## SSH Configuration

### Finding Connection Details

1. Navigate to RunPod console (https://www.runpod.io/console)
2. For **Custom Pods**: Click your pod → find "SSH over exposed TCP" or connection info
3. For **Cluster**: Click cluster name → select login node (typically node-0) → "SSH over exposed TCP"
   - External IP address (e.g., `198.145.108.6`)
   - Port number (e.g., `10400`)
   - Username (assigned by cluster administrator)

### SSH Config File Setup

**Understanding SSH aliases:**
Instead of typing `ssh username@198.145.108.6 -p 10400` every time, you create a short alias in `~/.ssh/config` on your **local machine**.

**Edit ~/.ssh/config on your local machine:**

**For Custom Pods:**
```bash
# Add this block to ~/.ssh/config
Host mypod                     # <-- YOUR CHOICE of alias (can be anything)
    HostName 203.0.113.45      # <-- POD IP from RunPod console
    User root                  # <-- Usually 'root' for custom pods
    Port 12345                 # <-- POD PORT from RunPod console
    IdentityFile ~/.ssh/id_ed25519  # <-- Your SSH key path
```

**For Shared Cluster:**
```bash
# Add this block to ~/.ssh/config
Host my-cluster                # <-- YOUR CHOICE of alias (e.g., 'cluster', 'runpod-cluster')
    HostName 198.145.108.6     # <-- CLUSTER IP from RunPod console
    User johndoe               # <-- USERNAME assigned by admin
    Port 10400                 # <-- CLUSTER PORT from RunPod console
    IdentityFile ~/.ssh/id_ed25519  # <-- Your SSH key path
```

**Complete example with multiple pods/clusters:**
```bash
# ~/.ssh/config
Host training-pod
    HostName 203.0.113.45
    User root
    Port 12345
    IdentityFile ~/.ssh/id_ed25519

Host runpod-cluster
    HostName 198.145.108.6
    User johndoe
    Port 10400
    IdentityFile ~/.ssh/id_ed25519

Host backup-cluster
    HostName 198.145.108.7
    User johndoe
    Port 10401
    IdentityFile ~/.ssh/id_ed25519
```

**After creating these aliases, you can use:**
```bash
ssh training-pod              # Instead of: ssh root@203.0.113.45 -p 12345
ssh runpod-cluster            # Instead of: ssh johndoe@198.145.108.6 -p 10400
```

### Testing Connection

```bash
# Custom pod (replace 'mypod' with YOUR alias from ~/.ssh/config)
ssh mypod "echo 'Pod connection successful'"

# Cluster (replace 'cluster' with YOUR alias from ~/.ssh/config)
ssh cluster "whoami && echo 'Cluster connection successful'"
```

**Expected output:**
```
# For pod:
Pod connection successful

# For cluster:
johndoe
Cluster connection successful
```

**If connection fails:**
- Check IP/port are correct in ~/.ssh/config
- Verify SSH key is registered in RunPod console
- For cluster: Verify admin has granted you access

### Port Forwarding for Jupyter

```bash
# Forward remote Jupyter (port 8899) to local (port 8888)
# Replace 'mypod' with YOUR alias
ssh -N -L 8888:localhost:8899 mypod &

# Access at: http://localhost:8888
```

**What this does:**
- `-N`: No remote command (just forwarding)
- `-L 8888:localhost:8899`: Forward local 8888 → remote 8899
- `&`: Run in background

---

## Custom Pods

### Directory Structure

```
/workspace-vast/root/           # Persistent storage (survives restarts)
├── git/                        # Code repositories
├── envs/                       # Python virtual environments
├── data/                       # Training data
├── models/                     # Trained models
└── results/                    # Experiment results

/workspace/                     # Large temporary storage
```

### Environment Setup

**Create directories:**
```bash
ssh mypod "mkdir -p /workspace-vast/root/{git,envs,data,models,results}"
```

**Python environment with UV:**
```bash
ssh mypod "
cd /workspace-vast/root/envs
curl -LsSf https://astral.sh/uv/install.sh | sh
source \$HOME/.cargo/env
uv venv training-env --python 3.11
source training-env/bin/activate
uv pip install torch transformers accelerate deepspeed datasets wandb
"
```

**Verify GPU access:**
```bash
ssh mypod "nvidia-smi"
```

### Persistent Jupyter Setup (Production-Grade)

The persistent Jupyter system provides:
- Auto-reconnection on network changes (WiFi ↔ ethernet, VPN connect/disconnect)
- Survival through laptop sleep/wake cycles
- No authentication tokens required
- Fixed URL that works across sessions

#### Prerequisites

**Local machine:**
```bash
# Install autossh (for auto-reconnecting tunnels)
# macOS:
brew install autossh

# Linux:
sudo apt-get install autossh
```

**Remote (pod/cluster):**
- tmux installed (usually pre-installed)
- Python virtual environment with jupyterlab

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  LOCAL MACHINE                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  autossh (auto-reconnecting SSH tunnel)                         ││
│  │  - Monitors connection health                                   ││
│  │  - Reconnects automatically on network change                   ││
│  │  - Forwards localhost:8888 → remote:8899                        ││
│  └───────────────────────────┬─────────────────────────────────────┘│
└──────────────────────────────┼──────────────────────────────────────┘
                               │ SSH Tunnel
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  REMOTE (POD/CLUSTER)                                                │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  tmux session "jupyter"                                         ││
│  │  └── jupyter lab --port=8899                                    ││
│  │      - Survives SSH disconnection                               ││
│  │      - No token required (--token='')                           ││
│  │      - Accessible from tunnel                                   ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

#### Setup Scripts

**1. Remote Startup Script (save as `start_jupyter.sh` on remote):**
```bash
#!/bin/bash
set -e

# Detect environment (pod vs cluster)
if [ "$(whoami)" = "root" ]; then
    USER_DIR="/workspace-vast/root"
    VENV_PATH="/workspace-vast/root/envs/training-env"
    ENV_TYPE="pod"
else
    USER_DIR="/workspace-vast/$(whoami)"
    VENV_PATH="$USER_DIR/envs/training-env"
    ENV_TYPE="cluster"
fi

TMUX_SESSION="jupyter"
JUPYTER_PORT=8899
LOG_FILE="$USER_DIR/jupyter.log"

# Function to check if Jupyter is running
is_jupyter_running() {
    pgrep -f "jupyter.*lab.*port.*$JUPYTER_PORT" >/dev/null 2>&1
}

# Kill existing processes on our port
kill_port_processes() {
    local pids=$(lsof -ti :$JUPYTER_PORT 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Freeing port $JUPYTER_PORT..."
        echo "$pids" | xargs -r kill 2>/dev/null || true
        sleep 2
    fi
}

# Start Jupyter in tmux
start_jupyter() {
    local jupyter_cmd="source ~/.bashrc && source $VENV_PATH/bin/activate && \
        jupyter lab --ip=0.0.0.0 --port=$JUPYTER_PORT --no-browser"

    # Add --allow-root for pods running as root
    if [ "$ENV_TYPE" = "pod" ]; then
        jupyter_cmd="$jupyter_cmd --allow-root"
    fi

    jupyter_cmd="$jupyter_cmd \
        --NotebookApp.token='' --NotebookApp.password='' \
        --ServerApp.token='' --ServerApp.password='' \
        --notebook-dir=$USER_DIR 2>&1 | tee $LOG_FILE"

    tmux new-session -d -s "$TMUX_SESSION" -c "$USER_DIR" "$jupyter_cmd"
}

# Main logic
> "$LOG_FILE"  # Clear log
kill_port_processes

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    if is_jupyter_running; then
        echo "Jupyter already running on port $JUPYTER_PORT"
        exit 0
    fi
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    sleep 2
fi

echo "Starting Jupyter in tmux..."
start_jupyter

# Wait for startup
for i in {1..30}; do
    if is_jupyter_running; then
        echo "Jupyter started: http://localhost:$JUPYTER_PORT/lab"
        exit 0
    fi
    sleep 1
done

echo "ERROR: Jupyter failed to start. Check: cat $LOG_FILE"
exit 1
```

**2. Local Connection Script (save as `connect_jupyter.sh` locally):**
```bash
#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <ssh_alias>"
    exit 1
fi

SSH_ALIAS="$1"
REMOTE_PORT=8899
LOCAL_PORT=8888
PID_FILE="/tmp/autossh_${SSH_ALIAS}.pid"
LOG_FILE="/tmp/autossh_${SSH_ALIAS}.log"

# Find available local port
find_port() {
    for port in $(seq 8888 8988); do
        if ! lsof -i :$port >/dev/null 2>&1; then
            echo $port
            return 0
        fi
    done
    return 1
}

# Cleanup existing connections
if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Stopping existing autossh..."
        kill "$old_pid" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Find available port
LOCAL_PORT=$(find_port)
if [ $? -ne 0 ]; then
    echo "ERROR: No available ports"
    exit 1
fi

# Test SSH connection
echo "Testing SSH connection to $SSH_ALIAS..."
if ! ssh "$SSH_ALIAS" 'echo "OK"' >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to $SSH_ALIAS"
    exit 1
fi

# Copy and run startup script
echo "Starting Jupyter on remote..."
scp start_jupyter.sh "$SSH_ALIAS:/tmp/"
ssh "$SSH_ALIAS" "chmod +x /tmp/start_jupyter.sh && /tmp/start_jupyter.sh"

# Start autossh tunnel
echo "Creating persistent tunnel..."
export AUTOSSH_GATETIME=30
export AUTOSSH_POLL=60

autossh -M 0 \
    -N \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=3" \
    -o "ExitOnForwardFailure=yes" \
    "$SSH_ALIAS" \
    > "$LOG_FILE" 2>&1 &

AUTOSSH_PID=$!
echo "$AUTOSSH_PID" > "$PID_FILE"
sleep 3

# Verify connection
if ! kill -0 "$AUTOSSH_PID" 2>/dev/null; then
    echo "ERROR: autossh failed to start"
    exit 1
fi

# Test Jupyter connectivity
for i in {1..10}; do
    if curl -s "http://localhost:$LOCAL_PORT/api" >/dev/null 2>&1; then
        echo ""
        echo "=========================================="
        echo "SUCCESS: Jupyter is now accessible"
        echo ""
        echo "URL: http://localhost:$LOCAL_PORT/lab"
        echo ""
        echo "VSCode: Use 'Jupyter: Specify Jupyter Server'"
        echo "        Enter: http://localhost:$LOCAL_PORT"
        echo "=========================================="
        echo ""
        echo "Management commands:"
        echo "  Status:  $0 $SSH_ALIAS status"
        echo "  Stop:    kill \$(cat $PID_FILE)"
        echo "  Logs:    tail -f $LOG_FILE"
        exit 0
    fi
    sleep 1
done

echo "ERROR: Could not connect to Jupyter"
exit 1
```

#### Usage

```bash
# Initial setup (one command)
./connect_jupyter.sh mypod

# Access Jupyter
# → http://localhost:8888/lab

# VSCode integration
# 1. Cmd+Shift+P → "Jupyter: Specify Jupyter Server"
# 2. Enter: http://localhost:8888

# Check status
curl -s http://localhost:8888/api >/dev/null && echo "Connected" || echo "Disconnected"

# Stop connection
kill $(cat /tmp/autossh_mypod.pid)
```

#### Session Management

```bash
# Check status (local)
if kill -0 $(cat /tmp/autossh_mypod.pid) 2>/dev/null; then
    echo "Autossh running"
fi
curl -s http://localhost:8888/api >/dev/null && echo "Jupyter accessible"

# Check status (remote)
ssh mypod "tmux has-session -t jupyter && echo 'Tmux session exists'"
ssh mypod "pgrep -f 'jupyter.*lab' && echo 'Jupyter process running'"

# View remote logs
ssh mypod "tail -20 /workspace-vast/root/jupyter.log"

# Attach to tmux session on remote
ssh mypod -t "tmux attach -t jupyter"

# Restart Jupyter (remote)
ssh mypod "tmux kill-session -t jupyter; /tmp/start_jupyter.sh"

# Full restart (local)
kill $(cat /tmp/autossh_mypod.pid)
./connect_jupyter.sh mypod
```

#### Key Features

| Feature | Benefit |
|---------|---------|
| **autossh** | Auto-reconnects when network changes |
| **tmux** | Jupyter survives SSH disconnections |
| **No tokens** | No authentication hassle |
| **Port 8899** | Avoids conflicts with system Jupyter |
| **Fixed local port** | Same URL always works |

#### Troubleshooting Jupyter

| Issue | Cause | Fix |
|-------|-------|-----|
| "Connection refused" | autossh not running | Restart: `./connect_jupyter.sh mypod` |
| "403 Forbidden" | Wrong Jupyter instance | Kill system Jupyter, restart ours |
| Port 8888 busy | Other process using port | Script auto-finds next available |
| "No tmux session" | Jupyter crashed on remote | SSH and run startup script |

### Code Synchronization

**Rsync (recommended for large files):**
```bash
# Upload code
rsync -avzP ./scripts/ mypod:/workspace-vast/root/git/myproject/scripts/

# Download results
rsync -avzP mypod:/workspace-vast/root/results/ ./results/

# Flags:
# -a: archive mode (preserves permissions, timestamps)
# -v: verbose
# -z: compress during transfer
# -P: shorthand for --partial --progress (show progress + resume interrupted transfers)
```

**Git for version control:**
```bash
ssh mypod "cd /workspace-vast/root/git/myproject && git pull origin main"
```

### Active Development Workflow

When actively developing code that you're testing on remote infrastructure (notebooks, training scripts, etc.), you need continuous synchronization.

#### Continuous Code Sync (Recommended for Active Development)

Create `auto_sync_code.sh`:

```bash
#!/bin/bash
# Continuous code synchronization for active development
# Usage: ./auto_sync_code.sh <remote_alias> [interval_seconds]
# Example: ./auto_sync_code.sh mypod 15

if [ $# -eq 0 ]; then
    echo "Usage: $0 <remote_ssh_alias> [interval_seconds]"
    echo "Example: $0 cluster 15"
    exit 1
fi

REMOTE_ALIAS="$1"
INTERVAL="${2:-15}"  # Default 15 seconds
PROJECT_ROOT="$(pwd)"

# Detect remote environment
REMOTE_USER=$(ssh "$REMOTE_ALIAS" 'whoami' 2>/dev/null)
if [ -z "$REMOTE_USER" ]; then
    echo "Error: Cannot connect to $REMOTE_ALIAS"
    exit 1
fi

# Set remote base path
if [ "$REMOTE_USER" = "root" ]; then
    REMOTE_BASE="/workspace-vast/root/git/myproject"
else
    REMOTE_BASE="/workspace-vast/$REMOTE_USER/git/myproject"
fi

echo "Syncing: $PROJECT_ROOT/ → $REMOTE_ALIAS:$REMOTE_BASE/"
echo "Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

sync_code() {
    rsync -avz --no-perms --no-owner --no-group \
        --exclude='.venv' \
        --exclude='node_modules' \
        --exclude='models' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.ipynb_checkpoints' \
        --exclude='.pytest_cache' \
        --exclude='.DS_Store' \
        --exclude='data/large_datasets' \
        --exclude='results' \
        "$PROJECT_ROOT/" "$REMOTE_ALIAS:$REMOTE_BASE/"
}

while true; do
    sync_code
    echo "$(date): Synced ✓"
    sleep "$INTERVAL"
done
```

**Usage:**
```bash
# Start in background (recommended)
./auto_sync_code.sh cluster 15 &

# Save the process ID to stop later
echo $! > /tmp/sync_code.pid

# Stop syncing
kill $(cat /tmp/sync_code.pid)
```

**Why this is useful:**
- Edit code locally in your IDE
- Changes automatically sync to remote every 15 seconds
- Test in remote Jupyter or training jobs immediately
- No manual rsync commands needed

#### Installing Packages in Editable Mode

For Python projects where you're actively developing the codebase:

```bash
# On remote (pod/cluster)
ssh cluster "
source /workspace-vast/\$(whoami)/envs/training-env/bin/activate
cd /workspace-vast/\$(whoami)/git/myproject
uv pip install -e .
"
```

**What `-e` (editable mode) does:**
- Installs package as symlink, not copy
- Code changes immediately reflected (no reinstall needed)
- Perfect for development + continuous sync workflow

**Example workflow:**
```bash
# 1. Start continuous sync
./auto_sync_code.sh cluster 15 &

# 2. Install your project in editable mode
ssh cluster "source /workspace-vast/\$(whoami)/envs/training-env/bin/activate && \
  cd /workspace-vast/\$(whoami)/git/myproject && \
  uv pip install -e ."

# 3. Edit code locally
# (your IDE, local changes)

# 4. Changes automatically sync every 15s
# 5. Run/test on remote - sees latest code immediately
ssh cluster "cd /workspace-vast/\$(whoami)/git/myproject && python test.py"
```

#### Syncing Environment Variables and API Keys

**Problem:** Remote jobs need API keys (HuggingFace, WandB, Anthropic, OpenAI) but you don't want to hardcode them in scripts.

**Solution:** Sync from local environment to remote `~/.bashrc`

Create `sync_env_vars.sh`:

```bash
#!/bin/bash
# Sync environment variables to remote bashrc
# Usage: ./sync_env_vars.sh <remote_alias>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <remote_ssh_alias>"
    exit 1
fi

REMOTE_ALIAS="$1"

echo "Syncing environment variables to $REMOTE_ALIAS..."

# Read from local .env file or environment
# Modify these to match your API keys
KEYS_TO_SYNC=(
    "HUGGING_FACE_TOKEN"
    "WANDB_API_KEY"
    "ANTHROPIC_API_KEY"
    "OPENAI_API_KEY"
)

for key in "${KEYS_TO_SYNC[@]}"; do
    # Get value from local environment
    value="${!key}"

    if [ -n "$value" ]; then
        echo "Syncing $key..."
        ssh "$REMOTE_ALIAS" "
            # Remove old entry if exists
            sed -i '/export $key=/d' ~/.bashrc 2>/dev/null || true
            # Add new entry
            echo 'export $key=\"$value\"' >> ~/.bashrc
        "
        echo "  ✓ $key synced"
    else
        echo "  ⚠ $key not set locally, skipping"
    fi
done

echo ""
echo "✓ Environment variables synced to $REMOTE_ALIAS"
echo "Remote will use these after: source ~/.bashrc"
```

**Usage:**
```bash
# Set keys locally first
export HUGGING_FACE_TOKEN="hf_your_token"
export WANDB_API_KEY="your_key"

# Sync to remote
./sync_env_vars.sh cluster

# Verify on remote
ssh cluster "source ~/.bashrc && env | grep TOKEN"
```

#### Alternative: Direct .bashrc Editing

```bash
# Add keys directly to remote bashrc
ssh cluster "cat >> ~/.bashrc << 'EOF'
export HUGGING_FACE_TOKEN='hf_your_token_here'
export WANDB_API_KEY='your_wandb_key_here'
export ANTHROPIC_API_KEY='sk-ant-your_key_here'
EOF
"

# Reload bashrc
ssh cluster "source ~/.bashrc"
```

#### Complete Development Setup Workflow

```bash
# 1. Set up SSH config (one-time)
# Edit ~/.ssh/config (see SSH Configuration section)

# 2. Create remote directories
ssh cluster "mkdir -p /workspace-vast/\$(whoami)/{git,envs,data,exp/logs}"

# 3. Create Python virtual environment
ssh cluster "
cd /workspace-vast/\$(whoami)/envs
curl -LsSf https://astral.sh/uv/install.sh | sh
source \$HOME/.cargo/env
uv venv training-env --python 3.11
"

# 4. Install core dependencies
ssh cluster "
source /workspace-vast/\$(whoami)/envs/training-env/bin/activate
uv pip install torch transformers accelerate deepspeed wandb jupyterlab
"

# 5. Clone or create project directory
ssh cluster "
cd /workspace-vast/\$(whoami)/git
git clone https://github.com/yourorg/yourproject.git
# OR: mkdir yourproject
"

# 6. Install project in editable mode
ssh cluster "
source /workspace-vast/\$(whoami)/envs/training-env/bin/activate
cd /workspace-vast/\$(whoami)/git/yourproject
uv pip install -e .
"

# 7. Sync environment variables
export HUGGING_FACE_TOKEN="hf_your_token"
export WANDB_API_KEY="your_key"
./sync_env_vars.sh cluster

# 8. Start continuous code sync (run in background)
./auto_sync_code.sh cluster 15 &
echo $! > /tmp/sync_code.pid

# 9. Start Jupyter
./connect_jupyter.sh cluster
# → Access at http://localhost:8888/lab

# 10. Develop locally, test remotely
# Edit files locally → Auto-syncs → Test in Jupyter/jobs on remote
```

#### Recommended .gitignore for Remote Projects

Add to your project's `.gitignore`:
```gitignore
# Virtual environments
.venv/
venv/
env/
envs/*/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.pytest_cache/

# Data and models
data/
models/
results/
*.safetensors
*.bin
*.pt
*.pth

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
```

#### Killing Background Processes

```bash
# Stop continuous sync
kill $(cat /tmp/sync_code.pid)

# Stop autossh tunnel
kill $(cat /tmp/autossh_cluster.pid)

# Or kill all rsync/autossh for cleanup
pkill -f "rsync.*cluster"
pkill -f "autossh.*cluster"
```

---

## Shared Slurm Cluster

### What is Slurm?

**Slurm** (Simple Linux Utility for Resource Management) is a job scheduler for shared computing clusters.

**How it works:**
```
You → Submit job script → Slurm Queue → Waits for GPU → Runs on node → Completes
```

**Key concepts:**
- **Job**: A script that runs your training code
- **Queue**: All submitted jobs waiting for or using resources
- **Node**: A physical server with GPUs
- **Partition**: A group of nodes (e.g., "general" for production, "dev" for testing)
- **QoS** (Quality of Service): Priority level (high vs low)
- **GRES** (Generic RESource): GPU allocation (e.g., `gpu:5` means 5 GPUs)

**Two types of jobs:**
| Type | Command | Use Case | Execution |
|------|---------|----------|-----------|
| **Interactive** | `srun` | Debugging, testing | Immediate shell on compute node |
| **Batch** | `sbatch` | Production training | Queued, runs when resources available |

**Why use Slurm?**
- Fair resource sharing among multiple users
- Automatic GPU allocation
- Job queuing when cluster busy
- Resume from checkpoints if jobs are interrupted

### Cluster Architecture

**Nodes and Roles:**
```
node-0:     Controller/login node (SSH entry point, DO NOT RUN JOBS HERE)
node-1:     Controller node (avoid for resource-intensive jobs)
node-2-9:   Compute nodes with GPUs (primary workhorses)
node-10:    Compute node (HAS PERMISSION ISSUES - avoid)
node-11:    Compute node with GPUs
node-12:    Compute node (HAS PERMISSION ISSUES - avoid)
node-13-15: Compute nodes with GPUs
node-16-22: Compute nodes (BROKEN - no /workspace/ mount - avoid)
```

**⚠️ IMPORTANT: Controller Node Protection**

Nodes 0 and 1 are controller nodes that manage the cluster. To protect them:
- **node-0**: SSH access may be restricted, no job execution
- **node-1**: Avoid for RAM-intensive or CPU-intensive jobs
- **Always exclude node-0 and node-1** in job scripts (see full recommended list below)

**Why this matters:**
- Controller nodes manage Slurm scheduling for all users
- Overloading them affects entire cluster stability
- SSH access to node-0 may be disabled for all users
- Jobs on controllers can cause cluster-wide issues

**Problematic Nodes to Exclude:**
| Nodes | Issue | Error Message |
|------|-------|---------------|
| node-0, node-1 | **Controller nodes** (policy) | Protect cluster stability |
| node-10, node-12 | Permission issues | `mkdir: Permission denied` |
| node-16-22 | No `/workspace/` mount | `Permission denied` creating directories |

**Recommended exclude list for ALL jobs:**
```bash
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]
```

**Partitions:**
| Partition | Nodes | Purpose |
|-----------|-------|---------|
| `general` | 1-13 | Production batch jobs |
| `dev` | 14-15 | Interactive debugging |
| `overflow` | All | Flexible placement |

### Quality of Service (QoS) - CRITICAL DETAILS

**⚠️ Note on "dev" naming:** Confusingly, `dev` exists as both a **partition** (group of nodes 14-15) AND a **QoS** (priority level). They are different concepts:
- **Partition `dev`**: Physical nodes dedicated to interactive work (`-p dev`)
- **QoS `dev`**: Priority level for interactive jobs (`--qos=dev`)
- **Typical usage**: `srun -p dev,overflow --qos=dev ...` uses BOTH

**Only TWO QoS levels exist: `high` and `low`**

| QoS | Priority | GPU Quota | Preemption | When to Use |
|-----|----------|-----------|------------|-------------|
| `high` | 200 | **~12-15 GPUs per user** | Won't be preempted | Single important experiments |
| `low` | 100 | **No limit** | Can be preempted | Sweeps, batch jobs, when hitting quota |
| `dev` | 300 | Varies | Won't be preempted | **Interactive `srun` ONLY** (cannot use with sbatch) |

**CRITICAL: QOSMaxGRESPerUser - You Can Block Yourself**

The `high` QoS has a per-user GPU limit of approximately **12-15 GPUs**.

**To check your exact quota:**
```bash
sacctmgr show qos format=name,MaxTRESPerUser%30
```

**Example Scenario:**
```
You have running:
  - Job A: 5 GPUs (high priority) ✓ running
  - Job B: 5 GPUs (high priority) ✓ running
  = Total: 10 GPUs using high QoS

You submit:
  - Job C: 5 GPUs (high priority)

Result:
  - Job C blocks with Reason: QOSMaxGRESPerUser
  - 10 + 5 = 15 GPUs > quota limit
  - YOUR OWN JOBS ARE BLOCKING YOU
```

**Solutions When Hitting QOSMaxGRESPerUser:**
1. **Use `--qos=low`** for the new job (bypasses quota)
2. **Wait** for existing high-priority jobs to complete
3. **Cancel** a running high-priority job

**Priority Strategy Recommendations:**
| Scenario | Recommended QoS | Rationale |
|----------|-----------------|-----------|
| Single critical experiment | `high` | Won't be preempted |
| Multiple concurrent jobs | `low` | Avoids self-blocking |
| Hyperparameter sweep (48 jobs) | `low` (with some `high`) | Mix: 75% low, 25% high |
| Quick test job | `low` | Faster start, no quota concerns |

### Directory Structure on Cluster

```
/workspace-vast/<username>/
├── git/                        # Synced code
├── envs/                       # Python environments
├── data/                       # Training data
└── exp/
    ├── logs/                   # Job stdout/stderr
    ├── models/                 # Final trained models
    ├── results/                # Evaluation outputs
    ├── jobs/                   # Job scripts and submission records
    └── configs/                # Generated config files

/workspace/<username>/
└── exp/
    └── training/               # Active training (checkpoints)
        └── <experiment>/       # Deleted after completion
```

### Environment Setup

```bash
ssh cluster "
# Create directory structure
mkdir -p /workspace-vast/\$(whoami)/{git,envs,data}
mkdir -p /workspace-vast/\$(whoami)/exp/{logs,models,results,jobs,configs}
mkdir -p /workspace/\$(whoami)/exp/training

# Environment variables
echo 'export HF_HOME=/workspace-vast/pretrained_ckpts' >> ~/.bashrc
echo 'export WORKSPACE=/workspace-vast/\$(whoami)' >> ~/.bashrc
"
```

### Python Environment Installation

```bash
ssh cluster "
cd /workspace-vast/\$(whoami)/envs
uv venv training-env --python 3.11
source training-env/bin/activate

# Install training packages
uv pip install \\
    'torch>=2.9.1' \\
    'transformers>=4.57.1' \\
    'trl>=0.12.0' \\
    'peft>=0.18.0' \\
    'accelerate>=1.11.0' \\
    'deepspeed>=0.18.3' \\
    'bitsandbytes>=0.49.0' \\
    'datasets>=4.4.1' \\
    'wandb>=0.18.0' \\
    'scikit-learn' \\
    'pandas' \\
    'pyyaml'
"
```

### Useful Aliases

Add to `~/.bashrc` on cluster:
```bash
# Interactive GPU session (4-hour limit, auto-deleted at midnight)
alias sint="srun -p dev,overflow --qos=dev --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=4:00:00 --job-name=D_\${USER} --pty bash"

# Queue monitoring
alias q='squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %N %.10b"'
alias qq='squeue -u $(whoami) -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %N %.10b"'
alias qw='watch -n 2 squeue -u $(whoami)'

# Job management
alias qdel='scancel'
alias qclear='scancel -u $(whoami)'
```

---

## Two-Tier Storage Architecture

### The Problem

DeepSpeed ZeRO-3 checkpoints are **massive**:
- Each checkpoint: ~400-450GB (model weights + optimizer states)
- During training: 2 checkpoints × ~428GB = ~856GB per experiment
- Final merged model: ~60GB

With a 10TB fast storage quota, only ~10-12 concurrent experiments could run before exhausting space.

### The Solution

| Filesystem | Capacity | Speed | Use For |
|------------|----------|-------|---------|
| `/workspace/` | 73TB | Slower (NFS) | Training checkpoints (temporary) |
| `/workspace-vast/` | 10TB | Fast (NVMe) | Final models, results, code |

### Data Flow Through Pipeline

```
Training Start
     │
     ▼
/workspace/<user>/exp/training/<name>/
     │  └── checkpoint-500/    (400-450GB each)
     │  └── checkpoint-1000/
     │  └── .progress.json
     │
     ▼ (training completes)
     │
Delete checkpoints, copy model (~60GB)
     │
     ▼
/workspace-vast/<user>/exp/models/<name>/
     │  └── model files
     │  └── model_info.json (completion marker)
     │
     ▼ (evaluation runs)
     │
/workspace-vast/<user>/exp/results/<name>.json (~2KB)
     │
     ▼ (cleanup phase)
     │
Delete /workspace/<user>/exp/training/<name>/
Optionally delete model (--delete-model flag)
```

### Storage Math

| Scenario | /workspace/ | /workspace-vast/ |
|----------|-------------|------------------|
| 1 active training | 856GB | ~100GB (logs/temp) |
| 10 concurrent training | 8.5TB ✓ | ~200GB ✓ |
| 48 final models | - | 48 × 60GB = 2.9TB ✓ |
| Results only | - | Negligible |

### Checkpoint Settings (for `/workspace/` performance)

```yaml
training:
  save_steps: 500           # Every ~25 min (less frequent for slower storage)
  save_total_limit: 1       # Keep only 1 checkpoint
  sigterm_grace: 900        # 15 min for slower checkpoint saves
```

---

## Single Experiment Training

### Job Script Template

Create `job_template.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=<EXPERIMENT_NAME>
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:5                    # Number of GPUs
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=10:00:00                 # 10 hour limit
#SBATCH --signal=B:SIGTERM@900          # 15 min grace for checkpoints
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --mail-user=your@email.com
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]  # Exclude controllers + broken nodes

# Environment setup
source /workspace-vast/${USER}/envs/training-env/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts
export HUGGING_FACE_TOKEN="your_token"
export WANDB_API_KEY="your_key"

# NCCL networking fix (cluster-specific)
# The "=" prefix in "=vxlan0" is NCCL syntax meaning "interfaces starting with vxlan0"
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Storage paths
TRAINING_BASE="/workspace/${USER}/exp/training"
OUTPUT_BASE="/workspace-vast/${USER}/exp/models"
RESULTS_DIR="/workspace-vast/${USER}/exp/results"
EXPERIMENT="<EXPERIMENT_NAME>"

# Create directories
mkdir -p "${TRAINING_BASE}/${EXPERIMENT}"
mkdir -p "${OUTPUT_BASE}/${EXPERIMENT}"
mkdir -p "${RESULTS_DIR}"

# Phase 1: Training
echo "PHASE 1: TRAINING"
accelerate launch --num_processes 5 train.py \
    --model "Qwen/Qwen3-32B" \
    --data "/path/to/data.jsonl" \
    --output "${TRAINING_BASE}/${EXPERIMENT}" \
    --epochs 1 \
    --batch-size 4 \
    --learning-rate 1e-5

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "TRAINING FAILED with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "TRAINING COMPLETE"

# Delete checkpoints to save space
rm -rf "${TRAINING_BASE}/${EXPERIMENT}"/checkpoint-*/ 2>/dev/null || true

# Copy model to permanent storage
rsync -a "${TRAINING_BASE}/${EXPERIMENT}/" "${OUTPUT_BASE}/${EXPERIMENT}/"

# Phase 2: Evaluation
echo "PHASE 2: EVALUATION"
python evaluate.py \
    --model-dir "${OUTPUT_BASE}/${EXPERIMENT}" \
    --output-dir "${RESULTS_DIR}" \
    --name "${EXPERIMENT}"

# Cleanup
rm -rf "${TRAINING_BASE}/${EXPERIMENT}"

echo "JOB COMPLETE"
```

### Submitting Jobs

```bash
# Submit single job
sbatch job_template.sh

# Submit with parameter override
sbatch --gres=gpu:1 --time=4:00:00 job_8b.sh

# Check submission
squeue -u $(whoami)
```

### Expected Output

```
=== PREFLIGHT CHECKS ===
✓ Config found
✓ Data file exists
✓ SSH connection working

=== SUBMITTING JOB ===
Experiment: my_experiment
Submitted batch job 88123

=== NEXT STEPS ===
Monitor: tail -f /workspace-vast/<user>/exp/logs/my_experiment_88123.out
Cancel: scancel 88123
```

### Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--wait` | NO | Block until completion |
| `--delete-model` | NO | Delete model after evaluation (saves ~60GB) |
| `--priority <qos>` | high | low\|high |
| `--force-eval` | NO | Re-run evaluation if results exist |
| `--dry-run` | NO | Validate without submitting |
| `--skip-sync` | NO | Skip code synchronization |

---

## Hyperparameter Sweeps

**What is a hyperparameter sweep?**
Testing multiple configurations (learning rates, batch sizes, etc.) to find the best settings for your model.

**Reality check:** A "sweep" is just **submitting many single jobs with different hyperparameters**. There's no magic - you can do this manually or script it.

### Manual Sweep Approach (Simple, Works)

**Create job scripts for each configuration:**

```bash
#!/bin/bash
# sweep_lr1e-5.sh
#SBATCH --job-name=sweep_lr1e5
#SBATCH --gres=gpu:5
...
python train.py --learning-rate 1e-5 --output /workspace-vast/${USER}/exp/sweep_lr1e5
```

```bash
#!/bin/bash
# sweep_lr2e-5.sh
#SBATCH --job-name=sweep_lr2e5
#SBATCH --gres=gpu:5
...
python train.py --learning-rate 2e-5 --output /workspace-vast/${USER}/exp/sweep_lr2e5
```

**Submit all:**
```bash
sbatch sweep_lr1e-5.sh
sbatch sweep_lr2e-5.sh
sbatch sweep_lr3e-5.sh
# ... etc
```

**Downside:** Tedious for 48+ configurations.

### Scripted Sweep Approach (Better for Many Jobs)

**Generate job scripts programmatically:**

```bash
#!/bin/bash
# generate_sweep.sh - Creates and submits 48 jobs

# Hyperparameter grid
LRs=(1e-5 2e-5 3e-5 5e-5)
BATCH_CONFIGS=("4 1" "4 2")  # batch_size grad_accum
WEIGHT_DECAYS=(0.01 0.05 0.1)
WARMUPS=(0.05 0.1)

for lr in "${LRs[@]}"; do
  for batch_config in "${BATCH_CONFIGS[@]}"; do
    read batch_size grad_accum <<< "$batch_config"
    eb=$((batch_size * grad_accum * 5))  # 5 GPUs

    for wd in "${WEIGHT_DECAYS[@]}"; do
      for warmup in "${WARMUPS[@]}"; do
        # Generate job name encoding hyperparams
        lr_name=$(echo $lr | sed 's/e-/em/; s/\.//g; s/^0*//')
        wd_name=$(echo $wd | sed 's/\./p/')
        warmup_name=$(echo $warmup | sed 's/\./p/')

        JOB_NAME="sweep_${lr_name}_eb${eb}_wd${wd_name}_wr${warmup_name}"

        # Create job script
        cat > "${JOB_NAME}.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/workspace-vast/%u/exp/sweep_logs/${JOB_NAME}_%j.out
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

source /workspace-vast/\${USER}/envs/training-env/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/\${USER}/git/project
python train.py \
    --model "Qwen/Qwen3-32B" \
    --data "/workspace-vast/\${USER}/data/train.jsonl" \
    --output "/workspace-vast/\${USER}/exp/sweep_models/${JOB_NAME}" \
    --learning-rate $lr \
    --batch-size $batch_size \
    --gradient-accumulation $grad_accum \
    --weight-decay $wd \
    --warmup-ratio $warmup

# Save results
python evaluate.py \
    --model "/workspace-vast/\${USER}/exp/sweep_models/${JOB_NAME}" \
    --output "/workspace-vast/\${USER}/exp/sweep_results/${JOB_NAME}.json"

# Cleanup checkpoint files
rm -rf "/workspace/\${USER}/exp/training/${JOB_NAME}"
EOF

        # Submit job
        sbatch "${JOB_NAME}.sh"
        echo "Submitted: $JOB_NAME"

      done
    done
  done
done

echo "Total jobs submitted: $((4 * 2 * 3 * 2)) = 48"
```

**Usage:**
```bash
# Make executable
chmod +x generate_sweep.sh

# Run (submits all 48 jobs)
./generate_sweep.sh

# Monitor progress
squeue -u $(whoami) | wc -l  # Count running/pending
```

### Effective Batch Size Formula

```
Effective Batch Size = batch_size × gradient_accumulation_steps × num_gpus
                     = 4 × 2 × 5 = 40 (for example above)
```

### Sweep Naming Convention

Encode hyperparameters in job name for easy tracking:
```
{lr}_eb{batch}_wd{decay}_wr{warmup}

Example: 2em05_eb40_wd0p05_wr0p1
         ├────┘ ├───┘ ├────┘ └───┘
         LR=2e-05 EB=40 WD=0.05 WR=0.1
```

### Priority Strategy for Sweeps

**Problem:** High QoS has ~12-15 GPU quota. With 48 jobs × 5 GPUs = 240 GPUs needed, you'll hit quota immediately.

**Solution:** Use mostly `low` priority to bypass quota:

```bash
# Simple: All low priority
#SBATCH --qos=low

# Advanced: Mix 25% high, 75% low
# In your sweep script:
if [ $((job_num % 4)) -eq 0 ]; then
    qos="high"
else
    qos="low"
fi
```

**Rationale:**
- `low`: No quota limit, can run many concurrent jobs
- `high`: Won't be preempted, but limited by quota
- Mix gets some guaranteed results while using spare capacity

### Collecting Sweep Results

**After jobs complete:**

```bash
# Check completion status
ssh cluster "
results_count=\$(ls /workspace-vast/\$(whoami)/exp/sweep_results/*.json 2>/dev/null | wc -l)
echo \"Completed: \$results_count / 48\"
"

# Download all results
rsync -avzP cluster:/workspace-vast/$(whoami)/exp/sweep_results/ ./sweep_results/
```

**Analyze locally:**

```python
import json
import glob
from pathlib import Path

results_dir = Path("sweep_results")
results = []

for result_file in results_dir.glob("*.json"):
    with open(result_file) as f:
        data = json.load(f)
        results.append({
            'name': result_file.stem,
            'tpr_01': data['metrics']['tpr_at_0.001_fpr'],  # Adjust to your format
            'learning_rate': data['config']['learning_rate'],
            'effective_batch': data['config']['effective_batch'],
            # ... extract other metrics
        })

# Sort by TPR at 0.1% FPR
results.sort(key=lambda x: x['tpr_01'], reverse=True)

print("TOP 10 CONFIGURATIONS:")
for i, r in enumerate(results[:10], 1):
    print(f"{i}. {r['name']}: TPR={r['tpr_01']:.1%}")
```

### Alternative: Python Sweep Script Template

If you want a reusable script, here's a minimal template:

```python
#!/usr/bin/env python3
"""
Minimal hyperparameter sweep script generator
Usage: python generate_sweep.py
"""

import subprocess
from pathlib import Path

# Hyperparameter grid
LEARNING_RATES = [1e-5, 2e-5, 3e-5, 5e-5]
BATCH_CONFIGS = [(4, 1), (4, 2)]  # (batch_size, grad_accum)
WEIGHT_DECAYS = [0.01, 0.05, 0.1]
WARMUP_RATIOS = [0.05, 0.1]

# Job template
JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=general,overflow
#SBATCH --qos={qos}
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/workspace-vast/%u/exp/sweep_logs/{name}_%j.out
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

source /workspace-vast/${{USER}}/envs/training-env/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

python train.py \\
    --model "Qwen/Qwen3-32B" \\
    --data "/workspace-vast/${{USER}}/data/train.jsonl" \\
    --output "/workspace/${{USER}}/exp/sweep_training/{name}" \\
    --learning-rate {lr} \\
    --batch-size {batch_size} \\
    --gradient-accumulation {grad_accum} \\
    --weight-decay {wd} \\
    --warmup-ratio {warmup}

# Copy to permanent storage
rsync -a "/workspace/${{USER}}/exp/sweep_training/{name}/" \\
         "/workspace-vast/${{USER}}/exp/sweep_models/{name}/"
rm -rf "/workspace/${{USER}}/exp/sweep_training/{name}"

# Evaluate
python evaluate.py \\
    --model "/workspace-vast/${{USER}}/exp/sweep_models/{name}" \\
    --output "/workspace-vast/${{USER}}/exp/sweep_results/{name}.json"
"""

def generate_job_name(lr, eb, wd, warmup):
    """Encode hyperparams in name"""
    lr_str = str(lr).replace('e-', 'em').replace('.', '').replace('0', '', 1)
    wd_str = str(wd).replace('.', 'p')
    warmup_str = str(warmup).replace('.', 'p')
    return f"{lr_str}_eb{eb}_wd{wd_str}_wr{warmup_str}"

# Generate and submit jobs
job_count = 0
for lr in LEARNING_RATES:
    for batch_size, grad_accum in BATCH_CONFIGS:
        eb = batch_size * grad_accum * 5  # 5 GPUs
        for wd in WEIGHT_DECAYS:
            for warmup in WARMUP_RATIOS:
                name = generate_job_name(lr, eb, wd, warmup)

                # Assign priority (25% high, 75% low)
                qos = "high" if (job_count % 4 == 0) else "low"

                # Generate job script
                job_script = JOB_TEMPLATE.format(
                    name=name, qos=qos, lr=lr,
                    batch_size=batch_size, grad_accum=grad_accum,
                    wd=wd, warmup=warmup
                )

                # Save script
                script_path = Path(f"sweep_{name}.sh")
                script_path.write_text(job_script)
                script_path.chmod(0o755)

                # Submit
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    job_id = result.stdout.split()[-1]
                    print(f"✓ {name} → Job {job_id}")
                else:
                    print(f"✗ {name} → Failed: {result.stderr}")

                job_count += 1

print(f"\nTotal jobs submitted: {job_count}")
```

**Usage:**
```bash
# On cluster
ssh cluster
cd /workspace-vast/$(whoami)/git/project/scripts
python generate_sweep.py
```

### Monitoring Sweep Progress

```bash
# Count completed
ssh cluster "ls /workspace-vast/\$(whoami)/exp/sweep_results/*.json | wc -l"

# Check running jobs
ssh cluster "squeue -u \$(whoami) | grep sweep_ | wc -l"

# View recent completions
ssh cluster "ls -lt /workspace-vast/\$(whoami)/exp/sweep_results/*.json | head -10"
```

---

## Evaluation-Only Jobs (Inference Workloads)

Evaluation jobs test trained models without training. They're much lighter than training jobs.

### Resource Requirements

**Evaluation vs Training:**
| Task | GPUs Needed | Memory | Time | Example |
|------|-------------|--------|------|---------|
| **Training** 32B | 5 | 200G | 4-10h | Full fine-tuning |
| **Evaluation** 32B | 1-2 | 100G | 10-30m | Testing on datasets |
| **Training** 8B | 1 | 100G | 2-4h | Full fine-tuning |
| **Evaluation** 8B | 1 | 64G | 5-15m | Testing on datasets |

**Why evaluation is cheaper:**
- No gradient computation
- No optimizer states
- Smaller batch sizes acceptable
- Forward pass only
- Can use float16 for speed

### Evaluation Job Template

```bash
#!/bin/bash
#SBATCH --job-name=eval_mymodel
#SBATCH --partition=general,overflow
#SBATCH --qos=high                      # High priority OK (eval is fast, 1 GPU)
#SBATCH --gres=gpu:1                    # Only need 1 GPU for most models
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00                 # 2 hours max (usually completes in <30 min)
#SBATCH --output=/workspace-vast/%u/exp/logs/eval_mymodel_%j.out
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

# Activate environment
source /workspace-vast/${USER}/envs/training-env/bin/activate
export HF_HOME=/workspace-vast/pretrained_ckpts

# Run evaluation
cd /workspace-vast/${USER}/git/project
python evaluate.py \
    --model "/workspace-vast/${USER}/exp/models/mymodel" \
    --test-data "/workspace-vast/${USER}/data/test.jsonl" \
    --output "/workspace-vast/${USER}/exp/eval-results" \
    --batch-size 32

echo "✅ Evaluation complete"
```

### Evaluating Multiple Models

```bash
# Submit evaluation for multiple trained models
for model in model_v1 model_v2 model_v3; do
    cat > eval_${model}.sh <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${model}
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/eval_${model}_%j.out
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

source /workspace-vast/\${USER}/envs/training-env/bin/activate
python evaluate.py --model "/workspace-vast/\${USER}/exp/models/${model}" ...
EOF
    sbatch eval_${model}.sh
done
```

### Baseline Model Evaluation

Evaluate HuggingFace models without fine-tuning:

```bash
#!/bin/bash
#SBATCH --job-name=eval_baseline_llama
#SBATCH --gres=gpu:2                    # Base 32B models may need 2 GPUs
...

python evaluate.py \
    --model "meta-llama/Llama-3.1-32B-Instruct" \  # HuggingFace model ID
    --test-data "/workspace-vast/${USER}/data/test.jsonl" \
    --output "/workspace-vast/${USER}/exp/eval-results/baseline"
```

**GPU requirements for base models (evaluation):**
| Model Size | GPUs for Eval | Memory/GPU |
|------------|---------------|------------|
| 7-8B | 1 | ~40GB |
| 13-14B | 1 | ~60GB |
| 30-32B | 2 | ~60GB |
| 70B | 4 | ~70GB |

**Note:** These are for **evaluation/inference only**. Training needs 1.5-2× more GPUs.

### Evaluation Results Storage

**Organized by date (recommended):**
```
/workspace-vast/<user>/exp/eval-results/
├── 2026-01-08/
│   ├── baseline_llama.json
│   ├── model_v1_eval.json
│   └── model_v2_eval.json
├── 2026-01-09/
│   └── model_v3_eval.json
```

**Benefits:**
- Easy to find evaluations by date
- Prevents name collisions
- Clear audit trail of when models were tested

---

## Interactive Sessions on Cluster

Interactive sessions give you immediate GPU access for debugging and testing. Unlike batch jobs (`sbatch`), interactive jobs (`srun`) give you a shell on a compute node.

### Quick Interactive Session

```bash
# Alias (add to ~/.bashrc)
alias sint="srun -p dev,overflow --qos=dev --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=4:00:00 --job-name=D_\${USER} --pty bash"

# Then just run:
sint

# You'll get a shell like:
# [user@node-3 ~]$
```

### Full Interactive Command

```bash
srun -p dev,overflow \
    --qos=dev \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=4:00:00 \
    --job-name=D_$(whoami) \
    --pty bash
```

**Explanation of flags:**
| Flag | Meaning |
|------|---------|
| `-p dev,overflow` | Try dev partition first, overflow if busy |
| `--qos=dev` | Interactive priority (high priority) |
| `--gres=gpu:1` | Request 1 GPU |
| `--cpus-per-task=8` | Request 8 CPU cores |
| `--mem=32G` | Request 32GB RAM |
| `--time=4:00:00` | Max 4 hours (auto-terminate) |
| `--job-name=D_<user>` | Prefix with `D_` for auto-cleanup |
| `--pty bash` | Allocate pseudo-terminal and run bash |

### Auto-Cleanup Convention

Jobs prefixed with `D_<username>` are automatically deleted at **midnight PT** (8am UK).

**Why this matters:**
- Prevents abandoned interactive sessions from hogging GPUs
- Ensures fair cluster usage
- **Always use `D_` prefix** for interactive jobs

```bash
# Good (auto-deleted at midnight)
srun --job-name=D_johnd ...

# Bad (persists until manually deleted)
srun --job-name=test ...
```

### What You Can Do in Interactive Sessions

```bash
# After 'sint', you're on a compute node with GPU access

# Check GPU
nvidia-smi

# Activate environment
source /workspace-vast/$(whoami)/envs/training-env/bin/activate

# Test training script
cd /workspace-vast/$(whoami)/git/project
python train.py --epochs 1 --data test_data.jsonl

# Exit when done
exit
# Job auto-terminates and releases GPU
```

### Interactive Session Limits

| Limitation | Value | Reason |
|------------|-------|--------|
| Max duration | 4 hours | Prevents GPU hogging |
| Max GPUs | 1-2 | Save resources for batch jobs |
| Partition | `dev` or `overflow` | Production uses `general` |

### When to Use Interactive vs Batch

| Scenario | Use | Command |
|----------|-----|---------|
| Quick debugging | Interactive (`srun`) | `sint` |
| Test training script | Interactive | `sint`, then run script |
| Production training | Batch (`sbatch`) | `sbatch job.sh` |
| Overnight training | Batch | `sbatch job.sh` |
| Multiple experiments | Batch | Submit multiple jobs |

### Common Interactive Workflows

**Testing a new script:**
```bash
# 1. Get interactive session
sint

# 2. Test with tiny dataset
cd /workspace-vast/$(whoami)/git/project
head -100 data.jsonl > test.jsonl
python train.py --data test.jsonl --epochs 1

# 3. If works, exit and submit batch job
exit
sbatch production_job.sh
```

**Debugging a failed job:**
```bash
# 1. Start interactive session on same resources
srun -p general --qos=high --gres=gpu:5 --mem=200G --pty bash

# 2. Manually run the exact command that failed
cd /workspace-vast/$(whoami)/git/project
python train.py <same args as failed job>

# 3. Fix issue, exit
exit
```

**Quick model testing:**
```bash
sint
source /workspace-vast/$(whoami)/envs/training-env/bin/activate
python test_model.py --model /path/to/model
exit
```

---

## Job Monitoring and Management

### Basic Queue Commands

```bash
# View ALL cluster jobs
squeue

# View YOUR jobs only
squeue -u $(whoami)

# Watch jobs update live (refreshes every 2 seconds)
watch -n 2 'squeue -u $(whoami)'

# Custom formatted output (more readable)
squeue -u $(whoami) -o "%.10i %.25j %.8T %.10M %.6D %R %b"
# Columns: JobID, Name, Status, Runtime, Nodes, Reason, GRES
```

**Recommended aliases (add to ~/.bashrc):**
```bash
alias q='squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %N %.10b"'
alias qq='squeue -u $(whoami) -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %N %.10b"'
alias qw='watch -n 2 squeue -u $(whoami)'
alias qdel='scancel'
alias qclear='scancel -u $(whoami)'
```

### Job States and Meanings

| State Code | Full Name | Meaning | What to Do |
|------------|-----------|---------|------------|
| `PD` | PENDING | Waiting for resources | Check Reason field |
| `R` | RUNNING | Executing on node | Monitor logs |
| `CG` | COMPLETING | Finishing up | Wait a few seconds |
| `CD` | COMPLETED | Finished successfully | Not in queue, check results |
| `F` | FAILED | Crashed or killed | Not in queue, check logs |
| `CA` | CANCELLED | User cancelled | Not in queue |
| `TO` | TIMEOUT | Hit time limit | Not in queue, may resume |

**Note:** Completed, failed, cancelled jobs disappear from `squeue`. Use `sacct` to see them.

### Detailed Job Information

```bash
# Full job details
scontrol show job <JOB_ID>

# Extract specific fields
scontrol show job <JOB_ID> | grep Reason      # Why pending
scontrol show job <JOB_ID> | grep NodeList    # Which node
scontrol show job <JOB_ID> | grep StartTime   # When started
scontrol show job <JOB_ID> | grep RunTime     # How long running
scontrol show job <JOB_ID> | grep GRES        # GPU allocation

# Job history (completed jobs)
sacct -u $(whoami) --starttime=today --format=JobID,JobName,Elapsed,State,ExitCode

# Jobs from last 7 days
sacct -u $(whoami) --starttime=$(date -d '7 days ago' +%Y-%m-%d) --format=JobID,JobName,Start,End,State,ExitCode

# Filter by job name
sacct -u $(whoami) --name=my_experiment --format=JobID,Start,Elapsed,State,ExitCode
```

### Monitoring Scripts (Helper Tools)

**Status overview script:**
```bash
# Create: scripts/cluster_status.sh
#!/bin/bash
CLUSTER="${1:-cluster}"

echo "=== Cluster Status Overview ==="
echo ""

echo "Running jobs:"
ssh "$CLUSTER" "squeue -u \$(whoami) --state=RUNNING -o '%.18i %.25j %.10M %R'"

echo ""
echo "Pending jobs:"
ssh "$CLUSTER" "squeue -u \$(whoami) --state=PENDING -o '%.18i %.25j %r' | head -10"

echo ""
echo "Recent completions (last 5):"
ssh "$CLUSTER" "sacct -u \$(whoami) --starttime=today --state=COMPLETED --format=JobID,JobName,Elapsed | tail -6 | head -5"

echo ""
echo "Recent failures (last 5):"
ssh "$CLUSTER" "sacct -u \$(whoami) --starttime=today --state=FAILED --format=JobID,JobName,ExitCode | tail -6 | head -5"

# Usage: ./scripts/cluster_status.sh cluster
```

**Monitor specific job:**
```bash
# Create: scripts/monitor_job.sh
#!/bin/bash
CLUSTER="${1:-cluster}"
JOB_IDENTIFIER="$2"  # Can be job ID or experiment name

if [ -z "$JOB_IDENTIFIER" ]; then
    echo "Usage: $0 <cluster> <job_id_or_name>"
    exit 1
fi

echo "=== Monitoring: $JOB_IDENTIFIER ==="

# Try to find job in queue
JOB_INFO=$(ssh "$CLUSTER" "squeue -u \$(whoami) -o '%.18i %.25j %.8T %R %N' | grep '$JOB_IDENTIFIER'")

if [ -n "$JOB_INFO" ]; then
    echo "Job Status:"
    echo "$JOB_INFO"
    echo ""

    JOB_ID=$(echo "$JOB_INFO" | awk '{print $1}')

    # Show details
    echo "Details:"
    ssh "$CLUSTER" "scontrol show job $JOB_ID" | grep -E "JobId|JobName|JobState|Reason|RunTime|NodeList|TRES"
else
    echo "Job not in queue (completed or failed)"
    echo ""

    # Check recent history
    echo "Recent history:"
    ssh "$CLUSTER" "sacct -u \$(whoami) --name=$JOB_IDENTIFIER --format=JobID,JobName,Start,End,State,ExitCode | tail -5"
fi

echo ""
echo "Recent logs (last 30 lines):"
ssh "$CLUSTER" "ls -t /workspace-vast/\$(whoami)/exp/logs/*${JOB_IDENTIFIER}*.out 2>/dev/null | head -1 | xargs tail -30"

# Usage: ./scripts/monitor_job.sh cluster my_experiment
```

### Monitoring Logs

```bash
# Tail live output (follows as job writes)
ssh cluster "tail -f /workspace-vast/\$(whoami)/exp/logs/<name>_<job_id>.out"

# Last 100 lines
ssh cluster "tail -100 /workspace-vast/\$(whoami)/exp/logs/<name>_<job_id>.out"

# List all logs by time
ssh cluster "ls -lt /workspace-vast/\$(whoami)/exp/logs/ | head -10"

# Find log by experiment name (when you don't know job ID)
ssh cluster "ls -t /workspace-vast/\$(whoami)/exp/logs/*<name>*.out | head -1"

# Search for errors across all logs
ssh cluster "grep -i 'error\|fail\|exception' /workspace-vast/\$(whoami)/exp/logs/*.out | tail -20"

# Search in specific log
ssh cluster "grep -i 'CUDA out of memory' /workspace-vast/\$(whoami)/exp/logs/<name>_*.out"

# Check if training completed
ssh cluster "grep 'TRAINING COMPLETE' /workspace-vast/\$(whoami)/exp/logs/<name>_*.out"

# Check if evaluation ran
ssh cluster "grep 'PHASE 2: EVALUATION' /workspace-vast/\$(whoami)/exp/logs/<name>_*.out"
```

### Canceling Jobs

```bash
# Cancel specific job by ID
scancel <JOB_ID>

# Cancel by name (all jobs with that name)
scancel --name <EXPERIMENT_NAME>

# Cancel all YOUR jobs
scancel -u $(whoami)

# Cancel multiple specific jobs
scancel <JOB_ID_1> <JOB_ID_2> <JOB_ID_3>

# Cancel jobs matching pattern
# (e.g., all sweep jobs)
squeue -u $(whoami) -o "%i %j" --noheader | grep "sweep_" | awk '{print $1}' | xargs scancel
```

### Checking GPU Usage

```bash
# On specific node (if you know job is running there)
ssh cluster "ssh node-3 nvidia-smi"

# Check GPU allocations from queue
squeue -o "%.10i %.20j %.8T %.4D %R %b"
# %b shows GRES (GPU allocation)

# See which users are using GPUs
squeue -o "%u %T %b" | grep RUNNING | sort | uniq -c

# Check GPU availability across all nodes
ssh cluster "sinfo -o '%n %G %T'"
# Shows node name, GRES (GPUs), and state

# Detailed node info
ssh cluster "sinfo -N -l"
```

### Results and Output Files

**Checking if results exist:**
```bash
# Check single experiment
ssh cluster "ls -lh /workspace-vast/\$(whoami)/exp/results/<name>.json"

# List all results with timestamps
ssh cluster "ls -lt /workspace-vast/\$(whoami)/exp/results/*.json | head -10"

# Count total results
ssh cluster "ls /workspace-vast/\$(whoami)/exp/results/*.json | wc -l"

# Check model exists
ssh cluster "ls -d /workspace-vast/\$(whoami)/exp/models/<name>/"

# Check for model completion marker
ssh cluster "cat /workspace-vast/\$(whoami)/exp/models/<name>/model_info.json"
```

**Downloading results:**
```bash
# Single experiment
rsync -avzP cluster:/workspace-vast/$(whoami)/exp/results/<name>.json ./results/

# All results
rsync -avzP cluster:/workspace-vast/$(whoami)/exp/results/ ./results/

# Model (large - ~60GB)
rsync -avzP cluster:/workspace-vast/$(whoami)/exp/models/<name>/ ./models/<name>/

# Check size before downloading model
ssh cluster "du -sh /workspace-vast/\$(whoami)/exp/models/<name>"
```

---

## Storage Management

### Understanding the Problem

**Why storage matters for GPU training:**

Large model training with DeepSpeed ZeRO-3 creates massive temporary files:
- Each checkpoint: ~400-450GB (model + optimizer states)
- Final model: ~60GB
- During training: Often 2 checkpoints retained = ~900GB per experiment

**With 48 concurrent jobs:** 48 × 900GB = **43TB**

This is why RunPod clusters use a two-tier storage system.

### Quick Status Commands

```bash
# Overall filesystem usage
ssh cluster "df -h /workspace /workspace-vast"

# Your total usage
ssh cluster "du -sh /workspace-vast/\$(whoami) /workspace/\$(whoami)"

# Breakdown by directory
ssh cluster "du -sh /workspace-vast/\$(whoami)/exp/* | sort -hr"
```

### Checking Current Usage (Detailed)

**Use this storage report script:**
```bash
#!/bin/bash
# save as: cluster_storage_report.sh
# usage: ./cluster_storage_report.sh <cluster_alias>

CLUSTER="${1:-cluster}"
USER=$(ssh "$CLUSTER" "whoami")

echo "========================================"
echo "CLUSTER STORAGE REPORT"
echo "User: $USER"
echo "Time: $(date)"
echo "========================================"
echo ""

echo "=== Filesystem Usage ==="
ssh "$CLUSTER" "df -h /workspace /workspace-vast"
echo ""

echo "=== /workspace-vast/$USER/exp breakdown ==="
ssh "$CLUSTER" "du -sh /workspace-vast/$USER/exp/* 2>/dev/null | sort -hr"
echo ""

echo "=== /workspace/$USER/exp breakdown ==="
ssh "$CLUSTER" "du -sh /workspace/$USER/exp/* 2>/dev/null | sort -hr"
echo ""

echo "=== Counts ==="
ssh "$CLUSTER" "
echo \"models:        \$(ls -d /workspace-vast/$USER/exp/models/*/ 2>/dev/null | wc -l)\"
echo \"results:       \$(ls /workspace-vast/$USER/exp/results/*.json 2>/dev/null | wc -l)\"
echo \"training dirs: \$(ls -d /workspace/$USER/exp/training/*/ 2>/dev/null | wc -l)\"
"
echo ""

echo "=== Running Jobs ==="
ssh "$CLUSTER" "squeue -u $USER -o '%.10i %.25j %.8T %.10M'"
```

### Finding Orphaned Directories

**Orphaned** = Training directories from completed/failed jobs that should have been auto-deleted.

```bash
#!/bin/bash
# save as: find_orphans.sh <cluster_alias>

CLUSTER="${1:-cluster}"
USER=$(ssh "$CLUSTER" "whoami")

echo "Finding orphaned training directories for user: $USER"
echo "=================================================="
echo ""

# Get list of currently running jobs
RUNNING=$(ssh "$CLUSTER" "squeue -u $USER -o '%j' --noheader")

# Check /workspace training dirs
ssh "$CLUSTER" "
for d in /workspace/$USER/exp/training/*/; do
    [ -d \"\$d\" ] || continue
    name=\$(basename \"\$d\")

    # Skip if currently running
    if echo \"$RUNNING\" | grep -q \"^\${name}\$\"; then
        echo \"RUNNING (skip): \$name\"
        continue
    fi

    # Check if has corresponding results
    if [ -f \"/workspace-vast/$USER/exp/results/\${name}.json\" ]; then
        size=\$(du -sh \"\$d\" 2>/dev/null | cut -f1)
        echo \"SAFE TO DELETE: \$size \$name\"
    else
        size=\$(du -sh \"\$d\" 2>/dev/null | cut -f1)
        echo \"ORPHANED (no results): \$size \$name\"
    fi
done
"
```

### Safe Cleanup Procedures

**CRITICAL SAFETY RULES:**
1. **Always check `squeue`** first - never delete directories for running jobs
2. **Check timestamps** - verify directories are old (days, not hours)
3. **Verify results exist** before deleting model directories
4. **Start with /workspace/** - training dirs are always safe if jobs completed
5. **Confirm with user** before any large-scale `rm -rf`

**Step-by-step cleanup workflow:**

```bash
# Step 1: Check running jobs
ssh cluster "squeue -u \$(whoami) -o '%j' --noheader"

# Step 2: List training dirs with ages
ssh cluster "ls -lt /workspace/\$(whoami)/exp/training/ | head -20"

# Step 3: Identify safe candidates (completed, old)
ssh cluster "
for d in /workspace/\$(whoami)/exp/training/*/; do
    name=\$(basename \"\$d\")
    # Check if results exist (means job completed)
    if [ -f \"/workspace-vast/\$(whoami)/exp/results/\${name}.json\" ]; then
        size=\$(du -sh \"\$d\" | cut -f1)
        echo \"SAFE: \$size \$name\"
    fi
done
"

# Step 4: Delete specific directory (after manual verification)
ssh cluster "rm -rf /workspace/\$(whoami)/exp/training/<specific_name>"

# Step 5: Verify deletion
ssh cluster "ls /workspace/\$(whoami)/exp/training/"
```

**Automated cleanup of all completed jobs:**
```bash
# CAREFUL: This deletes ALL training dirs that have corresponding results
ssh cluster '
USER=$(whoami)
RUNNING=$(squeue -u $USER -o "%j" --noheader)

for d in /workspace/$USER/exp/training/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")

    # Skip if currently running
    if echo "$RUNNING" | grep -q "^${name}$"; then
        echo "SKIP (running): $name"
        continue
    fi

    # Delete if results exist (means completed successfully)
    if [ -f "/workspace-vast/$USER/exp/results/${name}.json" ]; then
        echo "Deleting (completed): $name"
        rm -rf "$d"
    fi
done
'
```

### What's Safe to Delete

| Location | Safe If | How to Verify |
|----------|---------|---------------|
| `/workspace/<user>/exp/training/*` | Job not in `squeue` AND old (days) | `squeue -u $(whoami)` + `ls -lt` |
| `/workspace/<user>/exp/training/*` | Results JSON exists | `test -f /workspace-vast/<user>/exp/results/<name>.json` |
| `/workspace-vast/<user>/exp/models/*` | Results exist AND old | Exact name match + timestamps |
| Test directories (`test_*`, `debug_*`) | Older than 1 week | `ls -lt` check |

**Timestamp rule:**
- Created within last 24 hours → **BE CAUTIOUS**
- Created days/weeks ago + no running job → **SAFE**

### Disk Space Warnings

**"/workspace/ full" during training:**
```
Symptom:
  - Checkpoint save fails: "No space left on device"
  - Job crashes

Fix:
  ssh cluster "du -sh /workspace/\$(whoami)/exp/training/* | sort -hr"
  # Delete old training dirs (see cleanup procedure above)
```

**"/workspace-vast/ full":**
```
Symptom:
  - Model copy fails after training
  - Results may still save (small files)

Fix:
  ssh cluster "du -sh /workspace-vast/\$(whoami)/exp/* | sort -hr"
  # Delete old models (if you have results JSONs)
  ssh cluster "rm -rf /workspace-vast/\$(whoami)/exp/models/<old_experiment>"
```

### Model Cleanup (Saves ~60GB each)

**When models can be deleted:**
- Results JSON exists (you have metrics)
- No plans for further testing/inference
- Model is weeks old

```bash
# Check if results exist before deleting model
ssh cluster "
name='old_experiment'
if [ -f \"/workspace-vast/\$(whoami)/exp/results/\${name}.json\" ]; then
    echo \"Results exist - safe to delete model\"
    echo \"Size: \$(du -sh /workspace-vast/\$(whoami)/exp/models/\$name | cut -f1)\"
    # Delete after confirmation
    # rm -rf \"/workspace-vast/\$(whoami)/exp/models/\$name\"
else
    echo \"NO RESULTS - keep model or investigate\"
fi
"
```

### Storage Consumers by Type and Size

| Type | Location | Size Each | Lifecycle |
|------|----------|-----------|-----------|
| DeepSpeed checkpoint | `/workspace/<user>/exp/training/<name>/checkpoint-*/` | 400-450GB | Temporary (deleted after training) |
| Final merged model | `/workspace-vast/<user>/exp/models/<name>/` | ~60GB | Permanent (or manual delete) |
| Results JSON | `/workspace-vast/<user>/exp/results/<name>.json` | ~2KB | Permanent |
| Training logs | `/workspace-vast/<user>/exp/logs/<name>_<jobid>.out` | 1-5MB | Permanent |
| HuggingFace cache | `/workspace-vast/pretrained_ckpts/` | ~200GB | Shared, permanent |
| Code repository | `/workspace-vast/<user>/git/` | <1GB | Permanent |
| Python environment | `/workspace-vast/<user>/envs/` | ~5GB | Permanent |

### Disk Space Math

**Example: 10 concurrent 32B training jobs**
```
Checkpoints (temporary on /workspace/):
  10 jobs × 856GB (2 checkpoints each) = 8.56TB

Final models (permanent on /workspace-vast/):
  10 jobs × 60GB = 600GB

Results (permanent on /workspace-vast/):
  10 jobs × 2KB = 20KB (negligible)

Total /workspace/ needed: ~9TB (73TB available ✓)
Total /workspace-vast/ needed: ~600GB (10TB quota ✓)
```

### Monitoring Disk Usage

**Check before launching large sweeps:**
```bash
# Check current usage
ssh cluster "du -sh /workspace-vast/\$(whoami)"

# RULE OF THUMB: Keep /workspace-vast under 8TB (10TB quota with 2TB buffer)

# If over 5TB, clean up:
ssh cluster "
du -sh /workspace-vast/\$(whoami)/exp/* | sort -hr
# Identify largest directories and clean
"
```

**Check /workspace/ space (for active training):**
```bash
ssh cluster "df -h /workspace | grep workspace"

# Should show 50TB+ available for healthy operation
```

### Quick Status Check

```bash
ssh cluster "
echo '=== Filesystem Usage ==='
df -h /workspace /workspace-vast

echo ''
echo '=== Your Usage ==='
du -sh /workspace-vast/\$(whoami)/exp/* 2>/dev/null | sort -hr
du -sh /workspace/\$(whoami)/exp/* 2>/dev/null | sort -hr
"
```

---

## Complete First Job Example (Step-by-Step)

This section walks through submitting your first training job from scratch.

### Prerequisites Checklist

Before starting:
- [ ] SSH access configured (alias in ~/.ssh/config working)
- [ ] Python environment created on cluster
- [ ] Training data uploaded to cluster
- [ ] HuggingFace token set (if using gated models)
- [ ] WandB configured (or set to offline mode)

### Step 1: Create Training Script

Create `train.py` on the cluster:

```python
#!/usr/bin/env python3
"""
Generic training script for LLM fine-tuning.
Modify for your specific use case.
"""
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID from HuggingFace")
    parser.add_argument("--data", required=True, help="Path to training data (JSONL)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    dataset = load_dataset("json", data_files=args.data, split="train")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="wandb"
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    trainer.save_model(f"{args.output}/final")
    print("✅ Training complete")

if __name__ == "__main__":
    main()
```

Upload to cluster:
```bash
# Replace 'cluster' with your SSH alias from ~/.ssh/config
scp train.py cluster:/workspace-vast/$(whoami)/git/project/
```

### Step 2: Create Job Script

Create `my_first_job.sh` locally:

```bash
#!/bin/bash
#SBATCH --job-name=my_first_job
#SBATCH --partition=general,overflow
#SBATCH --qos=low                       # Use low to avoid quota issues
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00                  # 4 hour limit
#SBATCH --output=/workspace-vast/%u/exp/logs/my_first_job_%j.out
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

# %u = your username (auto-filled by Slurm)
# %j = job ID (auto-filled by Slurm)

# Activate environment (replace path with yours)
source /workspace-vast/${USER}/envs/training-env/bin/activate

# Set required environment variables
export HF_HOME=/workspace-vast/pretrained_ckpts
# export HUGGING_FACE_TOKEN="hf_your_token"  # Uncomment if using gated models
# export WANDB_API_KEY="your_key"            # Uncomment if using WandB

# Create output directory
mkdir -p /workspace-vast/${USER}/exp/my_first_job

# Run training
cd /workspace-vast/${USER}/git/project
python train.py \
    --model "gpt2" \
    --data "/workspace-vast/${USER}/data/train.jsonl" \
    --output "/workspace-vast/${USER}/exp/my_first_job" \
    --epochs 1 \
    --batch-size 8 \
    --learning-rate 2e-5

echo "✅ Job complete!"
```

Upload to cluster:
```bash
scp my_first_job.sh cluster:/workspace-vast/$(whoami)/exp/jobs/
```

### Step 3: Submit Job

```bash
# SSH to cluster (replace 'cluster' with your SSH alias)
ssh cluster

# Navigate to jobs directory
cd /workspace-vast/$(whoami)/exp/jobs

# Make script executable
chmod +x my_first_job.sh

# Submit the job
sbatch my_first_job.sh
```

**Expected output:**
```
Submitted batch job 88456
```

**The number `88456` is your job ID** - save this for monitoring.

### Step 4: Check Job Status

```bash
# Check if job is queued or running
squeue -u $(whoami)

# Output will show:
# JOBID  PARTITION  NAME          USER    ST  TIME  NODES
# 88456  general    my_first_job  johnd   R   5:32  1
```

**ST (Status) column:**
- `PD` = Pending (waiting for GPU)
- `R` = Running
- `CG` = Completing

**If pending, check why:**
```bash
scontrol show job 88456 | grep Reason

# Common reasons:
# Resources = Waiting for free GPU
# Priority = Other jobs ahead in queue
```

### Step 5: Monitor Progress

```bash
# Watch log file in real-time (replace 88456 with your job ID)
tail -f /workspace-vast/$(whoami)/exp/logs/my_first_job_88456.out

# You'll see training output:
# Training step 100, loss=2.34...
# Training step 200, loss=1.98...

# Press Ctrl+C to exit tail (job keeps running)
```

### Step 6: Check Completion

```bash
# Check if job still in queue
squeue -u $(whoami) | grep my_first_job

# If not in queue anymore, check result
ls -lh /workspace-vast/$(whoami)/exp/my_first_job/final/

# Should see model files:
# config.json
# model.safetensors
# tokenizer_config.json
# ...
```

### Step 7: Download Results (From Local Machine)

Exit SSH and run locally:

```bash
# Download the trained model (replace 'cluster' with your alias)
rsync -avzP cluster:/workspace-vast/$(whoami)/exp/my_first_job/final/ ./models/my_first_model/

# Download logs
rsync -avzP cluster:/workspace-vast/$(whoami)/exp/logs/my_first_job_*.out ./logs/
```

### Step 8: Cleanup (Optional)

```bash
# If training successful and model downloaded, clean up cluster storage
ssh cluster "rm -rf /workspace-vast/\$(whoami)/exp/my_first_job"
```

### Common First-Job Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `sbatch: command not found` | Not on cluster | Run `ssh cluster` first |
| `Permission denied` | Wrong directory permissions | Check `mkdir -p` commands |
| `CUDA out of memory` | Batch size too large | Reduce `--batch-size` to 4 or 2 |
| `No such file: train.jsonl` | Wrong path | Use absolute paths: `/workspace-vast/...` |
| Job stuck in PD | Cluster busy | Wait or use `--gres=gpu:1` (fewer GPUs) |
| `ModuleNotFoundError` | Environment not activated | Check `source` line in job script |
| `HUGGING_FACE_TOKEN not set` | Missing token for gated model | Uncomment and set token in job script |

---

## Troubleshooting

### Jobs Stuck in PENDING

**Check reason:**
```bash
ssh cluster "scontrol show job <JOB_ID> | grep Reason"
```

| Reason | Meaning | Fix |
|--------|---------|-----|
| `QOSMaxGRESPerUser` | Hit GPU quota for QoS | Use `--qos=low` |
| `Resources` | Not enough free GPUs | Wait or reduce GPU count |
| `Priority` | Other jobs ahead | Wait or increase priority |
| `ReqNodeNotAvail` | Requested node unavailable | Remove node constraint |

### Jobs Fail Immediately

**Check logs:**
```bash
ssh cluster "tail -100 /workspace-vast/\$(whoami)/exp/logs/<name>_<job_id>.out"
```

**Common causes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `DeepSpeed config not found` | Path resolution issue | Check config sync |
| `CUDA out of memory` | Too many GPUs or batch | Reduce batch or add GPUs |
| `ModuleNotFoundError` | Environment not activated | Check source command in job |
| `NCCL Bootstrap error` | Networking on node | Add `NCCL_SOCKET_IFNAME` |
| `Permission denied: /workspace/` | Node lacks mount | Exclude problematic nodes |

### Jobs Complete but No Results

```bash
# Check if evaluation ran
ssh cluster "grep -A 20 'PHASE 2: EVALUATION' /workspace-vast/\$(whoami)/exp/logs/<name>_*.out"

# Check if results file exists
ssh cluster "ls -lh /workspace-vast/\$(whoami)/exp/results/<name>.json"
```

### Incomplete Checkpoints

Checkpoints interrupted mid-save lack `trainer_state.json`:

```bash
# Check checkpoint completeness
ssh cluster "ls /workspace/\$(whoami)/exp/training/<name>/checkpoint-*/trainer_state.json"

# Delete incomplete checkpoints
ssh cluster "
for ckpt in /workspace/\$(whoami)/exp/training/<name>/checkpoint-*/; do
    if [ ! -f \"\$ckpt/trainer_state.json\" ]; then
        echo \"Deleting incomplete: \$ckpt\"
        rm -rf \"\$ckpt\"
    fi
done
"
```

### Storage Full Errors

**"/workspace/ full":**
```bash
# Check usage
ssh cluster "df -h /workspace"

# Clean old training dirs
ssh cluster "rm -rf /workspace/\$(whoami)/exp/training/old_*"
```

**"/workspace-vast/ full":**
```bash
# Check usage
ssh cluster "du -sh /workspace-vast/\$(whoami)/exp/* | sort -hr"

# Clean old models (if results saved)
ssh cluster "rm -rf /workspace-vast/\$(whoami)/exp/models/old_experiment"
```

---

## Critical Bugs and Fixes

### Bug 1: Hardcoded Paths

**Symptom:** `ModuleNotFoundError: No module named 'training'`

**Cause:** Scripts had `/workspace-vast/root/` hardcoded from pod development

**Fix:** Dynamic path resolution:
```python
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / "backend/src"))
```

### Bug 2: NCCL Networking Errors

**Symptom:** `Bootstrap : no socket interface found` (49/55 jobs failed)

**Cause:** Some cluster nodes have different network interface config

**Fix:** Add to job template:
```bash
# The "=" prefix is NCCL syntax meaning "interfaces starting with vxlan0"
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0
```

### Bug 3: Incomplete Checkpoints from Time Limit

**Symptom:** Jobs crash on resume with "trainer_state.json not found"

**Cause:**
- Jobs hit time limit during checkpoint save
- 180s grace period not enough for 178GB optimizer states
- Checkpoint directory created but missing trainer files

**Fix:**
1. Increase time limit to 10 hours
2. Increase SIGTERM grace to 900 seconds (15 minutes)
3. Add checkpoint validation before resume

### Bug 4: QOSMaxGRESPerUser Blocking

**Symptom:** Job stuck in PENDING with reason `QOSMaxGRESPerUser`

**Cause:** High-priority QoS has per-user GPU limit (~12-15 GPUs)

**Fix:** Use `--qos=low` for sweeps or when running many jobs

### Bug 5: Package Version Incompatibilities

**Symptom:** Various API errors (`max_seq_length`, `unwrap_model`, etc.)

**Cause:** Upgrading packages individually creates version conflicts

**Fix:** Install ALL packages from requirements at once, never individually

### Bug 6: Nodes Missing /workspace/ Mount

**Symptom:** Jobs fail with "cannot create directory '/workspace/<user>'"

**Cause:** Some nodes (node-0, node-16-22) don't have `/workspace/` properly mounted

**Fix:** Use the standard exclude list in all job scripts:
```bash
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]
```

### Bug 7: YAML Type Coercion

**Symptom:** `TypeError: '<=' not supported between 'float' and 'str'`

**Cause:** YAML parses `2e-05` as string

**Fix:** Explicit casting in Python:
```python
learning_rate = float(config["learning_rate"])
```

---

## Best Practices and Etiquette

### DO's

- **Use `D_username` prefix** for interactive jobs (auto-deleted at midnight PT)
- **Use `--qos=low`** for non-urgent experiments (utilizes spare compute)
- **Cancel jobs** you don't need anymore (`scancel <job_id>`)
- **Check coordination channels** before launching large sweeps
- **Store data on `/workspace-vast/`** (survives node restarts)
- **Clean up old experiments** regularly
- **Test with small jobs first** before launching sweeps

### DON'Ts

- **NEVER manually set `CUDA_VISIBLE_DEVICES`** (Slurm handles this)
- **Don't use batch jobs with `--qos=dev`** (dev is interactive only)
- **Don't leave interactive sessions overnight** (use batch jobs)
- **Don't store critical data on node-local storage**
- **Don't upgrade packages individually** (breaks version compatibility)
- **Don't launch 100+ jobs** without testing 1-2 first

### Understanding Preemption

With `--qos=low`:
- Your job can be cancelled if high-priority jobs need GPUs
- Job automatically re-queues when resources available
- Only use for experiments that can handle interruption

**⚠️ Two Different SIGTERM Scenarios:**

| Scenario | Warning Time | Controlled By |
|----------|--------------|---------------|
| **Preemption** (high-priority job needs GPU) | ~3 minutes | Cluster config (not changeable) |
| **Time limit** (job hits `--time` limit) | Your `@N` value | `#SBATCH --signal=B:SIGTERM@900` |

The `--signal=B:SIGTERM@900` setting gives you 15 minutes warning before **time limit** termination. It does **NOT** affect preemption warning time.

### Checkpoint Resume Behavior

Jobs automatically resume from checkpoints when:
1. Same experiment name is used
2. Valid checkpoint exists (has `trainer_state.json`)
3. Incomplete checkpoints are auto-deleted

---

## Cost Considerations

### GPU Pricing (Approximate)

| GPU Type | Cost/Hour |
|----------|-----------|
| H100 80GB | ~$2.50-3.00 |
| H200 | ~$3.00-3.50 |
| A100 80GB | ~$2.00-2.50 |

### Job Cost Estimation

```
Cost = GPU_count × Hours × Price_per_GPU_hour

Example (5-GPU job, 4 hours):
  5 × 4 × $2.50 = $50 per job
```

### Sweep Cost Examples

| Model | GPUs | Time | Jobs | Est. Cost |
|-------|------|------|------|-----------|
| 8B | 1 | 2h | 48 | ~$240 |
| 32B | 5 | 5h | 48 | ~$3,000 |
| 32B QLoRA | 1 | 3h | 48 | ~$360 |

### Cost Reduction Strategies

1. **Use `--delete-model`** to save disk (not GPU cost, but storage cost)
2. **Use QLoRA** for memory-efficient training (fewer GPUs)
3. **Test with smaller model first** (8B vs 32B)
4. **Run overnight** when cluster less busy (faster start)
5. **Use `--qos=low`** for non-critical jobs (may preempt but same cost)

---

## Quick Reference

### SSH Commands

```bash
# Connect
ssh mypod                                    # Custom pod
ssh cluster                                  # Shared cluster

# Port forwarding (Jupyter runs on remote port 8899, forwarded to local 8888)
ssh -N -L 8888:localhost:8899 mypod &        # Jupyter tunnel
```

### Slurm Commands

```bash
# Submit job
sbatch job_script.sh

# Interactive session (dev partition)
srun -p dev,overflow --qos=dev --gres=gpu:1 --pty bash

# View jobs
squeue -u $(whoami)                          # Your jobs
squeue                                       # All jobs
watch -n 2 'squeue -u $(whoami)'             # Live updates

# Cancel jobs
scancel <JOB_ID>                             # Specific job
scancel -u $(whoami)                         # All your jobs

# Job details
scontrol show job <JOB_ID>
scontrol show job <JOB_ID> | grep Reason     # Why pending
```

### Storage Commands

```bash
# Check usage
df -h /workspace /workspace-vast
du -sh /workspace-vast/$(whoami)/exp/*

# Clean training dirs
rm -rf /workspace/$(whoami)/exp/training/<name>

# Check for running jobs before deleting
squeue -u $(whoami) -o "%j" --noheader
```

### File Transfers

```bash
# Upload with rsync (resumable)
rsync -avzP local/ cluster:/workspace-vast/$(whoami)/destination/

# Download results
rsync -avzP cluster:/workspace-vast/$(whoami)/exp/results/ ./results/
```

### Environment Activation

```bash
# On cluster
source /workspace-vast/$(whoami)/envs/training-env/bin/activate

# On pod
source /workspace-vast/root/envs/training-env/bin/activate
```

### Log Monitoring

```bash
# Tail live logs
ssh cluster "tail -f /workspace-vast/\$(whoami)/exp/logs/<name>_<job_id>.out"

# Search for errors
ssh cluster "grep -i error /workspace-vast/\$(whoami)/exp/logs/*.out"
```

---

## Appendix: GPU Requirements by Model Size

### Inference vs Fine-tuning

**Critical:** GPU counts for inference are much lower than for fine-tuning.

| Model | Params | Architecture | GPUs (Inference) | GPUs (Fine-tune) | Memory/GPU |
|-------|--------|--------------|------------------|------------------|------------|
| 7-8B | 7-8B | Dense | 1 | 1 | ~40GB |
| 14B | 14B | Dense | 1 | 2 | ~60GB |
| 30B MoE (3B active) | 30B | MoE | 2 | 4 | ~60GB |
| 32B | 32B | Dense | 2 | 5 | ~60GB |
| 70B | 70B | Dense | 4 | 8+ | ~70GB |

**Rule of thumb:** Fine-tuning needs 1.5-2× more GPUs than inference.

### MoE Memory Warning

MoE models (e.g., Qwen3-30B-A3B with 3B active):
- **Activate** only 3B params during forward pass
- **Load** all 30B params into GPU memory for DeepSpeed sharding
- Fine-tuning memory: model (60GB) + gradients (60GB) + optimizer (120GB) = **240GB total**

### Time Estimates by Model

| Model | GPUs | Training Time (90k examples) | Time Limit |
|-------|------|------------------------------|------------|
| 8B | 1 | ~2h | 4:00:00 |
| 32B | 5 | ~4h | 10:00:00 |
| 32B QLoRA | 1 | ~3h | 6:00:00 |
| 30B MoE | 4 | ~5h | 10:00:00 |

---

## Appendix: DeepSpeed ZeRO-3 Configuration

Example `zero3.json`:
```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

### Checkpoint Sizes with ZeRO-3

| Component | Size |
|-----------|------|
| Model shards | ~65GB (14 × 4.6GB) |
| Optimizer states (Adam) | ~360GB |
| **Total checkpoint** | **~428GB** |

---

## Appendix: Complete Job Template

```bash
#!/bin/bash
#SBATCH --job-name={{EXPERIMENT_NAME}}
#SBATCH --partition=general,overflow
#SBATCH --qos={{QOS}}
#SBATCH --gres=gpu:{{NUM_GPUS}}
#SBATCH --cpus-per-task={{CPUS}}
#SBATCH --mem={{MEMORY}}
#SBATCH --time={{TIME_LIMIT}}
#SBATCH --signal=B:SIGTERM@{{SIGTERM_GRACE}}
#SBATCH --output={{LOGS_DIR}}/%x_%j.out
#SBATCH --mail-user={{EMAIL}}
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

set -e

echo "========================================"
echo "Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "========================================"

# Activate environment
source {{VENV_PATH}}/bin/activate

# Set environment variables
export HF_HOME=/workspace-vast/pretrained_ckpts
export HUGGING_FACE_TOKEN="{{HF_TOKEN}}"
export WANDB_API_KEY="{{WANDB_KEY}}"

# NCCL fixes for cluster networking
export NCCL_SOCKET_IFNAME="=vxlan0"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Storage paths (two-tier architecture)
TRAINING_BASE="/workspace/{{USERNAME}}/exp/training"
OUTPUT_BASE="/workspace-vast/{{USERNAME}}/exp/models"
RESULTS_DIR="/workspace-vast/{{USERNAME}}/exp/results"
EXPERIMENT="{{EXPERIMENT_NAME}}"

TRAINING_OUTPUT="${TRAINING_BASE}/${EXPERIMENT}"
MODEL_OUTPUT="${OUTPUT_BASE}/${EXPERIMENT}"

# Create directories
mkdir -p "${TRAINING_OUTPUT}"
mkdir -p "${MODEL_OUTPUT}"
mkdir -p "${RESULTS_DIR}"

# ============================================================
# PHASE 1: TRAINING
# ============================================================

echo ""
echo "PHASE 1: TRAINING"
echo "========================================"

cd {{PROJECT_DIR}}

# Check for existing checkpoint to resume
RESUME_ARG=""
if [ -d "${TRAINING_OUTPUT}" ]; then
    CHECKPOINT=$(ls -d ${TRAINING_OUTPUT}/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT/trainer_state.json" ]; then
        echo "Resuming from checkpoint: $CHECKPOINT"
        RESUME_ARG="--resume-from-checkpoint $CHECKPOINT"
    fi
fi

# Run training
accelerate launch \
    --num_processes {{NUM_GPUS}} \
    --mixed_precision bf16 \
    train.py \
    --model "{{MODEL_ID}}" \
    --data "{{DATA_PATH}}" \
    --output "${TRAINING_OUTPUT}" \
    --deepspeed-config "{{DEEPSPEED_CONFIG}}" \
    {{TRAINING_ARGS}} \
    $RESUME_ARG

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "TRAINING FAILED with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "TRAINING COMPLETE"

# ============================================================
# PHASE 2: MODEL COPY
# ============================================================

echo ""
echo "PHASE 2: MODEL COPY"
echo "========================================"

# Delete checkpoints to save disk space
echo "Deleting checkpoints..."
rm -rf "${TRAINING_OUTPUT}"/checkpoint-*/ 2>/dev/null || true

# Copy final model to permanent storage
echo "Copying model to ${MODEL_OUTPUT}..."
rsync -a "${TRAINING_OUTPUT}/" "${MODEL_OUTPUT}/"

# Create completion marker
echo "{\"completed\": \"$(date -Iseconds)\"}" > "${MODEL_OUTPUT}/model_info.json"

# ============================================================
# PHASE 3: EVALUATION
# ============================================================

echo ""
echo "PHASE 3: EVALUATION"
echo "========================================"

python evaluate.py \
    --name "${EXPERIMENT}" \
    --model-dir "${MODEL_OUTPUT}" \
    --output-dir "${RESULTS_DIR}" \
    {{EVAL_ARGS}}

EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "EVALUATION FAILED with exit code $EVAL_EXIT"
    # Don't exit - still cleanup
fi

# ============================================================
# PHASE 4: CLEANUP
# ============================================================

echo ""
echo "PHASE 4: CLEANUP"
echo "========================================"

# Always delete training directory from /workspace/
if [ -d "${TRAINING_OUTPUT}" ]; then
    echo "Deleting training directory..."
    rm -rf "${TRAINING_OUTPUT}"
fi

# Optionally delete model (saves ~60GB)
if [ "{{DELETE_MODEL}}" = "true" ]; then
    echo "Deleting model (--delete-model flag set)..."
    rm -rf "${MODEL_OUTPUT}"
fi

echo ""
echo "========================================"
echo "JOB COMPLETE"
echo "Results: ${RESULTS_DIR}/${EXPERIMENT}.json"
echo "========================================"
```

---

## Glossary and Terminology

### Slurm Terms

| Term | Definition | Example |
|------|------------|---------|
| **sbatch** | Submit batch job command | `sbatch my_job.sh` |
| **srun** | Run interactive job command | `srun --gres=gpu:1 --pty bash` |
| **squeue** | View job queue | `squeue -u $(whoami)` |
| **scancel** | Cancel job | `scancel 88456` |
| **scontrol** | View/modify job details | `scontrol show job 88456` |
| **sacct** | View job history | `sacct --starttime=today` |
| **sinfo** | View cluster nodes/partitions | `sinfo` |
| **Partition** | Group of nodes (e.g., general, dev) | `--partition=general` |
| **QoS** | Quality of Service (priority level) | `--qos=low` or `--qos=high` |
| **GRES** | Generic RESource (GPU allocation) | `--gres=gpu:5` = 5 GPUs |
| **Node** | Physical server with GPUs | `node-3`, `node-7` |
| **Job ID** | Unique identifier for submitted job | `88456` |
| **%u** | Slurm variable for username | Expands to your username |
| **%j** | Slurm variable for job ID | Expands to job ID |
| **%x** | Slurm variable for job name | Expands to job name |

### Storage Terms

| Term | Definition |
|------|------------|
| **NFS** | Network File System (shared storage accessible from all nodes) |
| **/workspace/** | Large, slower shared storage (73TB) for temporary checkpoints |
| **/workspace-vast/** | Fast shared storage (10TB quota) for final outputs |
| **Checkpoint** | Intermediate training state (model + optimizer + metadata) |
| **DeepSpeed** | Distributed training library for large models |
| **ZeRO-3** | DeepSpeed optimization stage that shards model across GPUs |
| **Optimizer states** | Momentum and variance tensors for Adam optimizer (~3× model size) |

### Training Terms

| Term | Definition |
|------|------------|
| **Fine-tuning** | Training pre-trained model on new data |
| **Epoch** | One complete pass through training dataset |
| **Batch size** | Number of examples processed together |
| **Effective batch size** | batch_size × gradient_accumulation × num_gpus |
| **Learning rate** | Step size for weight updates (e.g., 2e-5 = 0.00002) |
| **Gradient accumulation** | Accumulate gradients over N steps before updating |
| **BF16** | Brain Float 16-bit precision (memory efficient) |
| **Gradient checkpointing** | Trades compute for memory (enables larger batches) |

### Monitoring Terms

| Term | Definition |
|------|------------|
| **ST** | Job state in squeue (PD, R, CG, CD, F) |
| **Runtime** | How long job has been running |
| **NodeList** | Which node(s) job is running on |
| **Reason** | Why job is pending (Resources, Priority, QOSMaxGRES) |
| **TRES** | Trackable RESources (includes GPU count) |
| **ExitCode** | Return value of job (0 = success, non-zero = error) |

### Common Acronyms

| Acronym | Meaning |
|---------|---------|
| **GPU** | Graphics Processing Unit |
| **CPU** | Central Processing Unit |
| **RAM** | Random Access Memory |
| **SSH** | Secure Shell (remote access protocol) |
| **HF** | HuggingFace (model repository) |
| **WandB** | Weights & Biases (experiment tracking) |
| **OOM** | Out Of Memory |
| **CUDA** | NVIDIA's GPU programming platform |
| **NCCL** | NVIDIA Collective Communications Library (multi-GPU networking) |
| **LLM** | Large Language Model |
| **MoE** | Mixture of Experts (model architecture) |
| **QLoRA** | Quantized Low-Rank Adaptation (memory-efficient fine-tuning) |

---

## Document Conventions

### SSH Alias Notation

Throughout this guide:
- `cluster`, `mypod`, `training-pod` = **Your chosen SSH alias** from `~/.ssh/config`
- **Always replace with YOUR actual alias**
- Example: If you named your cluster `runpod-gpu`, replace `cluster` with `runpod-gpu`

### Username Placeholders

| Placeholder | Meaning |
|-------------|---------|
| `<user>`, `<username>` | Your cluster username (e.g., `johndoe`) |
| `$(whoami)` | Shell command that expands to your username |
| `%u` | Slurm variable that expands to your username |
| `${USER}` | Bash variable containing your username |

**⚠️ IMPORTANT: When to use which:**

| Context | Use | Example |
|---------|-----|---------|
| `#SBATCH` directives | `%u` | `#SBATCH --output=/workspace-vast/%u/logs/%j.out` |
| Shell commands (interactive) | `$(whoami)` | `ssh cluster "ls /workspace-vast/\$(whoami)/"` |
| Inside job scripts | `${USER}` | `mkdir -p /workspace-vast/${USER}/exp` |
| Escaping in SSH commands | `\$(whoami)` | Backslash prevents local expansion |

**Common mistake:** Using `%u` outside of `#SBATCH` lines creates a literal `%u` directory!
```bash
# WRONG - creates literal "%u" directory:
mkdir /workspace-vast/%u/data

# CORRECT - expands to your username:
mkdir /workspace-vast/$(whoami)/data
```

### Path Placeholders

| Placeholder | Replace With |
|-------------|--------------|
| `<POD_IP>` | Your pod IP from RunPod console |
| `<POD_PORT>` | Your pod port from RunPod console |
| `<CLUSTER_IP>` | Your cluster IP from RunPod console |
| `<CLUSTER_PORT>` | Your cluster port from RunPod console |
| `<name>`, `<experiment_name>` | Your experiment name |
| `<JOB_ID>` | Actual job ID from sbatch output |

### Placeholder Conventions

This document uses two placeholder styles:

| Style | Meaning | Action |
|-------|---------|--------|
| `<EXPERIMENT_NAME>` | Manual replacement | Find and replace with your value |
| `{{EXPERIMENT_NAME}}` | Template variable | Used in programmatic templates (Jinja2, Python, etc.) |

**Simple job scripts** use `<angle_brackets>` - manually edit before use.
**Appendix templates** use `{{double_braces}}` - designed for script generation/templating systems.

### Command Notation

| Notation | Meaning |
|----------|---------|
| `# Comment` | Explanation, don't type |
| `<REPLACE>` | Replace with actual value |
| `[optional]` | Optional parameter |
| `...` | Additional parameters |
| `→` | Expected output |

---

**Document Version**: January 2026
**Authors**: Faizan. Based on docs, Slack conversations and production experience with RunPod GPU infrastructure
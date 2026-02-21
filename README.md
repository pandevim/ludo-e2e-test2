# Qwen vLLM Apptainer Setup on HPC

This repository contains the configuration needed to run the 0.5B Quantized Qwen model (`Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ`) using vLLM on a high-performance compute cluster (HPC) via Apptainer (Singularity) and Slurm. By utilizing Apptainer rather than Docker, we can safely run containers as non-root users and avoid HPC cache quota limitations.

## Contents

- `run_vllm.slurm`: The Slurm job script to allocate a GPU, securely map local fast storage for caching, and run the vLLM container via Apptainer.
- `test_client.py`: A simple Python test script utilizing the `openai` package to test the API endpoint.
- `requirements.txt`: Python package dependencies for testing.

## Setup & Execution Flow

### 0. Install dotenvx on JARVIS

Since we are using encrypted `.env` files for secrets, install the standalone `dotenvx` binary into your home directory (no root required):

```bash
# 1. Make sure your local bin directory exists
mkdir -p ~/.local/bin

# 2. Tell the dotenvx script to install exactly into that directory
curl -sfS "https://dotenvx.sh?directory=$HOME/.local/bin" | sh
```

### 1. Launch the Server on JARVIS (HPC)

Connect to your JARVIS login node and run the following commands:

```bash
# Watch the Slurm queue for available nodes
watch --color -d -n 1 'sinfo | grep --color=always -E "idle|mix|$"'
```

```bash
# Submit the job to the Slurm scheduler
# Note: dotenvx is configured in the script to load your encrypted .env.production via the decryption key in ~/.secrets/ludo.key
sbatch --partition=gpu-h100 run_vllm.slurm
```

### 2. Identify the Compute Node

Check the Slurm queue to see which compute node grabbed your job:

```bash
squeue -u apandey10
```

```bash
# Details of the job
scontrol show jobid <jobid>
```

Look under the `NODELIST` column (e.g., `g101`). Wait until the job state is `R` (Running).

### Stopping a Job

If you need to cancel a pending or running job, use the `scancel` command with your Job ID:

```bash
scancel <jobid>
```

### 3. Tunneling (from your local laptop)

Once the job is running on the compute node, set up an SSH tunnel to securely forward the traffic from your laptop to the specific compute node running your job.

Run this on your **local machine** (replace `<node_name>` with the actual node name and `apandey10` with your username):

```bash
ssh -N -L 8080:<node_name>:8000 apandey10@jarvis.stevens.edu
```

### 4. Test the API Endpoints

Now you can start sending requests to `localhost:8080` from your laptop!

**Option A: Python Client (Recommended)**
First, install the requirements on your local machine:

```bash
pip install -r requirements.txt
```

Then run the simple test client:

```bash
python test_client.py
```

**Option B: cURL**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ]
  }'
```

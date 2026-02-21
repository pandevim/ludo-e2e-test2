# Qwen vLLM Apptainer Setup on HPC

This repository contains the configuration needed to run the 0.5B Quantized Qwen model (`Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ`) using vLLM on a high-performance compute cluster (HPC) via Apptainer (Singularity) and Slurm. By utilizing Apptainer rather than Docker, we can safely run containers as non-root users and avoid HPC cache quota limitations.

## Contents

- `run_vllm.slurm`: The Slurm job script to allocate a GPU, securely map local fast storage for caching, and run the vLLM container via Apptainer.

## Setup & Execution Flow

### 0. Local Setup (Optional)

Create `.env` and `.env.production` files locally and put your secrets in them:

```bash
HF_TOKEN=<your_hf_token>
```

Encrypt the `.env` files locally:

```bash
# Default
npx @dotenvx/dotenvx encrypt

# Production
npx @dotenvx/dotenvx encrypt -f .env.production
```

### 1. Connect and Clone

Connect to JARVIS and clone the repository:

```bash
ssh <username>@jarvis.stevens.edu
git clone https://github.com/pandevim/ludo-e2e-test2.git
cd ludo-e2e-test2
```

### 2. Install dotenvx on JARVIS (HPC)

Since we are using encrypted `.env` files for secrets, install the standalone `dotenvx` binary into your home directory.

First, create the secrets directory and add the decryption key:

```bash
# Note the space before "mkdir" to avoid saving the command in shell history (~/.bash_history),
# as it contains a sensitive Private Key.
 mkdir -p ~/.secrets && echo 'export DOTENV_PRIVATE_KEY_PRODUCTION="3aa1a24c84e9d299883a55cd557a9ef164dc118c9ebda9ce1486b9daf9081161"' > ~/.secrets/ludo.key
```

Then, install `dotenvx`:

```bash
# 1. Make sure your local bin directory exists
mkdir -p ~/.local/bin

# 2. Tell the dotenvx script to install exactly into that directory
curl -sfS "https://dotenvx.sh?directory=$HOME/.local/bin" | sh
```

### 3. Launch the Server on JARVIS (HPC)

Before running the server, pull the required image. We force Apptainer to unpack the image on the fast local SSD:

```bash
module load apptainer
export APPTAINER_CACHEDIR="/local/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/local/$USER/apptainer_tmp"
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

apptainer pull vllm-openai.sif docker://vllm/vllm-openai:latest
```

Submit the job to the Slurm scheduler:

```bash
# Note: dotenvx is configured in the script to load your encrypted .env.production via the decryption key in ~/.secrets/ludo.key
sbatch --partition=gpu-h100 run_vllm.slurm
```

_(Optional)_ Watch the Slurm queue for available nodes:

```bash
watch --color -d -n 1 'sinfo | grep --color=always -E "idle|mix|$"'
```

### 4. Identify the Compute Node

Check the Slurm queue to see which compute node grabbed your job. Look under the `NODELIST` column (e.g., `g101`). Wait until the job state is `R` (Running).

```bash
squeue -u <username>
```

To see details of the job:

```bash
scontrol show jobid <jobid>
```

### 5. Managing the Job

If you need to cancel a pending or running job, use the `scancel` command with your Job ID:

```bash
scancel <jobid>
```

### 6. Tunneling (from your local laptop)

Once the job is running on the compute node, set up an SSH tunnel to securely forward the traffic from your laptop to the specific compute node running your job.

Run this on your **local machine** (replace `<node_name>` with the actual node name, e.g., `g101`, and `<username>` with your JARVIS username):

```bash
ssh -f -N -L 8080:<node_name>:8000 <username>@jarvis.stevens.edu
```

```bash
ngrok http 8080
```

### 7. Test the API Endpoints

Now you can start sending requests to `localhost:8080` from your laptop!

You can use cURL to verify the endpoints:

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

### 8. Frontend Chat Interface

This repository includes a static HTML chat interface to interact with the model:

- The code is located at `public/index.html`.

#### Testing Locally

You can simply open `public/index.html` in your web browser. By default, it will fall back to using `http://localhost:8080` for API requests. Ensure your tunnel is active.

#### Deploying to Vercel (or other static hosts)

You can deploy the `public/` directory for free on Vercel. To configure your production endpoint:

1. Go to your Vercel Project Settings.
2. Add a new Environment Variable named `API_BASE_URL` (or `VITE_API_BASE_URL` if you eventually port to Vite).
3. Set the value to your publicly accessible backend URL. For example, your ngrok URL or HuggingFace endpoint (e.g. `https://your-backend-url.com` - do NOT include a trailing slash).

The UI will automatically pick up this variable and route requests and metrics appropriately.

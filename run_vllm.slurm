#!/bin/bash
#SBATCH --job-name=ludo
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=ludo_%j.out

# 1. Load the Apptainer module explicitly
module load apptainer

# 2. Load the decryption key into the shell
source ~/.secrets/ludo.key

# 3. Securely map the node's fast /local SSD to the container's HuggingFace cache
# This prevents your /home directory from freezing!
export APPTAINER_BIND="/local/$SLURM_JOB_ID:/root/.cache/huggingface"

# Force Apptainer to unpack the image on the fast local SSD
export APPTAINER_CACHEDIR="/local/$SLURM_JOB_ID/apptainer_cache"
export APPTAINER_TMPDIR="/local/$SLURM_JOB_ID/apptainer_tmp"
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# 4. Run the Docker container via Apptainer wrapped in dotenvx
# Since we installed the standalone dotenvx binary into ~/.local/bin/
# it loads your .env file and Apptainer inherits the environment.
~/.local/bin/dotenvx run -- apptainer run --nv docker://vllm/vllm-openai:latest \
    --model Qwen/Qwen3-Coder-Next \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 65536 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --tensor-parallel-size 1 \
    --port 8000 \
    --allowed-origins '[\"*\"]'

# # ── 3. LLM ────────────────────────────────────────────────────────────────────
# echo ""
# echo "[3/4] Starting LLM service (vLLM)..."
# start_service LLM \
#   apptainer run --nv \
#     --bind "${HF_HOME}:/root/.cache/huggingface" \
#     "$VLLM_IMAGE" \
#     --model google/gemma-2b-AWQ \
#     --port "${LLM_PORT}" \
#     --dtype float16 \
#     --quantization awq \
#     --gpu-memory-utilization 0.85 \
#     --max-model-len 2048 \
#     --max-num-seqs 1 \
#     --enforce-eager \
#     --chat-template "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n<start_of_turn>model\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<end_of_turn>\n'}}{% endif %}{% endfor %}"
# LLM_PID=$SERVICE_PID
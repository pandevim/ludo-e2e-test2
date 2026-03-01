#!/bin/bash
#SBATCH --job-name=unmute
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=unmute_%j.out

# =============================================================================
# Unmute Voice Pipeline
#   STT  : kyutai/stt-2.6b-en      (moshi, WebSocket, port 8001)
#   LLM  : google/gemma-3-12b-it   (vLLM, OpenAI API, port 8002)
#   TTS  : kyutai/tts-0.75b-en-public (moshi, HTTP, port 8003)
#   ORCH : orchestrator.py          (WebSocket bridge, port 8080)
# =============================================================================

module load apptainer
source ~/.secrets/unmute.key

set -euo pipefail

# ── Ports ────────────────────────────────────────────────────────────────────
export STT_PORT=3001
export LLM_PORT=3002
export TTS_PORT=3003
export ORCH_PORT=3004

# ── Cache / model storage ────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"          # set in .env or environment
mkdir -p "$HF_HOME"

# ── Shared moshi container image (STT + TTS) ─────────────────────────────────
MOSHI_IMAGE="docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
VLLM_IMAGE="docker://vllm/vllm-openai:latest"
APPTAINER_OPTS="--nv --bind ${HF_HOME}:/root/.cache/huggingface --bind $PWD:/workspace"

# Pip deps installed inside container at runtime (no writable layer needed).
MOSHI_DEPS='pip install -q --upgrade pip \
  && pip install -q fastapi "uvicorn[standard]" websockets numpy scipy \
  "git+https://github.com/kyutai-labs/moshi.git#egg=moshi&subdirectory=moshi"'

echo "========================================"
echo "  Unmute pipeline on $(hostname)"
echo "  STT  → ws://localhost:${STT_PORT}/ws"
echo "  LLM  → http://localhost:${LLM_PORT}"
echo "  TTS  → http://localhost:${TTS_PORT}"
echo "  UI   → http://$(hostname):${ORCH_PORT}"
echo "========================================"

# ── 1. STT service ───────────────────────────────────────────────────────────
echo "[1/4] Starting STT service..."
apptainer run $APPTAINER_OPTS "$MOSHI_IMAGE" bash -c "
  ${MOSHI_DEPS}
  python /workspace/stt_server.py --port ${STT_PORT}
" &
STT_PID=$!

# ── 2. LLM service (vLLM) ────────────────────────────────────────────────────
echo "[2/4] Starting LLM service (vLLM)..."
apptainer run --nv \
  --bind "${HF_HOME}:/root/.cache/huggingface" \
  "$VLLM_IMAGE" \
  --model google/gemma-3-12b-it \
  --port "${LLM_PORT}" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 8192 \
  --trust-remote-code &
LLM_PID=$!

# ── 3. TTS service ───────────────────────────────────────────────────────────
echo "[3/4] Starting TTS service..."
apptainer run $APPTAINER_OPTS "$MOSHI_IMAGE" bash -c "
  ${MOSHI_DEPS}
  python /workspace/tts_server.py --port ${TTS_PORT}
" &
TTS_PID=$!

# ── Helper: wait for HTTP health endpoint ────────────────────────────────────
wait_for() {
  local url="$1" name="$2" tries=90
  echo "  Waiting for ${name} at ${url}..."
  for i in $(seq 1 $tries); do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "  ✓ ${name} ready"
      return 0
    fi
    sleep 5
  done
  echo "  ✗ ${name} timed out"
  return 1
}

wait_for "http://localhost:${STT_PORT}/health" "STT"
wait_for "http://localhost:${LLM_PORT}/health"  "LLM"
wait_for "http://localhost:${TTS_PORT}/health"  "TTS"

# ── 4. Orchestrator (on host Python) ─────────────────────────────────────────
echo "[4/4] Starting orchestrator..."
pip install -q fastapi "uvicorn[standard]" websockets httpx 2>/dev/null || true

python "$PWD/orchestrator.py" \
  --stt-ws   "ws://localhost:${STT_PORT}/ws" \
  --llm-url  "http://localhost:${LLM_PORT}" \
  --tts-url  "http://localhost:${TTS_PORT}" \
  --port     "${ORCH_PORT}" &
ORCH_PID=$!

echo ""
echo "Pipeline ready → open http://$(hostname):${ORCH_PORT} in your browser"
echo ""

# ── Cleanup ──────────────────────────────────────────────────────────────────
cleanup() {
  echo "Shutting down..."
  kill "$STT_PID" "$LLM_PID" "$TTS_PID" "$ORCH_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait

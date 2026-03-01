#!/bin/bash

# =============================================================================
# Unmute Voice Pipeline
#   STT  : kyutai/stt-2.6b-en         (moshi, WebSocket, port 3001)
#   TTS  : kyutai/tts-0.75b-en-public (moshi, HTTP,      port 3003)
#   LLM  : deepseek-ai/DeepSeek-V3.2  (featherless.ai API, external)
#   ORCH : orchestrator.py            (WebSocket bridge,  port 3004)
# =============================================================================

source ../.env.raw

set -euo pipefail

# ── Ports ─────────────────────────────────────────────────────────────────────
export STT_PORT=3001
export TTS_PORT=3003
export ORCH_PORT=3004

# ── Featherless API ────────────────────────────────────────────────────────────
export LLM_URL="https://api.featherless.ai/v1"
export LLM_MODEL="deepseek-ai/DeepSeek-V3.2"
export FEATHERLESS_API_KEY="${FEATHERLESS_API_KEY:-}"   # set in .env.raw

export APPTAINER_TMPDIR="$HOME/apptainer_tmp"
export APPTAINER_CACHEDIR="$HOME/.apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# ── Cache / model storage ──────────────────────────────────────────────────────
export HF_HOME="$HOME/hf_cache"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
mkdir -p "$HF_HOME"

# ── Log directory ──────────────────────────────────────────────────────────────
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"

# ── Container images ───────────────────────────────────────────────────────────
MOSHI_IMAGE="./moshi_with_git.sif"
APPTAINER_OPTS="--nv --userns \
  --env XET_LOG_DIR=/tmp/xet_logs \
  --bind ${HF_HOME}:/home/aniruddha/hf_cache \
  --bind $PWD:/workspace"

MOSHI_DEPS='export PATH="$HOME/.local/bin:$PATH" && export HF_HOME=/home/aniruddha/hf_cache && \
  python3 -m venv --system-site-packages /workspace/.venv && \
  source /workspace/.venv/bin/activate && \
  pip install --upgrade pip && \
  pip install --ignore-installed fastapi "uvicorn[standard]" websockets numpy scipy && \
  pip install -q --no-deps "git+https://git@github.com/kyutai-labs/moshi#egg=moshi&subdirectory=moshi"'

echo "========================================"
echo "  Unmute pipeline on $(hostname)"
echo "  STT  → ws://localhost:${STT_PORT}/ws"
echo "  TTS  → http://localhost:${TTS_PORT}"
echo "  LLM  → ${LLM_URL} (${LLM_MODEL})"
echo "  ORCH → http://localhost:${ORCH_PORT}"
echo "  Logs → ${LOG_DIR}/"
echo "========================================"

# ── Helpers ───────────────────────────────────────────────────────────────────

start_service() {
  local name="$1"; shift
  local logfile="${LOG_DIR}/${name}.log"

  > "$logfile"

  "$@" > "$logfile" 2>&1 &
  SERVICE_PID=$!

  tail -n 0 -f "$logfile" | sed -u "s/^/[${name}] /" &
  TAIL_PIDS+=($!)

  echo "  → ${name} started (PID ${SERVICE_PID}), logging to ${logfile}"
}

wait_for() {
  local url="$1" name="$2"
  local tries=90 interval=5
  local logfile="${LOG_DIR}/${name}.log"
  echo "  Waiting for ${name} at ${url} (up to $((tries * interval))s)..."
  echo "  (watching ${logfile} for progress)"
  for i in $(seq 1 $tries); do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo ""
      echo "  ✓ ${name} ready (attempt ${i})"
      return 0
    fi
    if (( i % 5 == 0 )); then
      local last_line
      last_line=$(tail -n 1 "$logfile" 2>/dev/null || true)
      echo "  [${name}] t=$((i * interval))s | ${last_line:-<no output yet>}"
    else
      printf "."
    fi
    sleep "$interval"
  done
  echo ""
  echo "  ✗ ${name} timed out. Last 10 lines of ${logfile}:"
  tail -n 10 "$logfile" 2>/dev/null | sed "s/^/    /" || true
  return 1
}

declare -a TAIL_PIDS=()
STT_PID="" TTS_PID="" ORCH_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  trap - EXIT INT TERM

  for pid in "${TAIL_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done

  for pid in "$STT_PID" "$TTS_PID" "$ORCH_PID"; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done

  sleep 5

  for pid in "$STT_PID" "$TTS_PID" "$ORCH_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "  Force-killing PID ${pid}..."
      kill -9 "$pid" 2>/dev/null || true
    fi
  done

  echo "Done."
  exit 0
}
trap cleanup EXIT INT TERM

# ── 1. STT (CPU) ──────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Starting STT service (CPU)..."
start_service STT \
  apptainer run $APPTAINER_OPTS \
    --env CUDA_VISIBLE_DEVICES="" \
    --env XET_LOG_DIR=/tmp/xet_logs \
    "$MOSHI_IMAGE" bash -c "
    ${MOSHI_DEPS}
    TORCHDYNAMO_DISABLE=1 python /workspace/stt_server.py --port ${STT_PORT}
  "
STT_PID=$SERVICE_PID

wait_for "http://localhost:${STT_PORT}/health" "STT" || exit 1

# ── 2. TTS (GPU) ──────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Starting TTS service (GPU)..."
start_service TTS \
  apptainer run $APPTAINER_OPTS \
    --env XET_LOG_DIR=/tmp/xet_logs \
    "$MOSHI_IMAGE" bash -c "
    ${MOSHI_DEPS}
    TORCHDYNAMO_DISABLE=1 python /workspace/tts_server.py --port ${TTS_PORT}
  "
TTS_PID=$SERVICE_PID

wait_for "http://localhost:${TTS_PORT}/health" "TTS" || exit 1

# ── 3. Orchestrator ───────────────────────────────────────────────────────────
echo ""
echo "[3/3] Starting orchestrator..."
pip install -q fastapi "uvicorn[standard]" websockets httpx 2>/dev/null || true

start_service ORCH \
  python "$PWD/orchestrator.py" \
    --stt-ws   "ws://localhost:${STT_PORT}/ws" \
    --llm-url  "${LLM_URL}" \
    --llm-model "${LLM_MODEL}" \
    --llm-api-key "${FEATHERLESS_API_KEY}" \
    --tts-url  "http://localhost:${TTS_PORT}" \
    --port     "${ORCH_PORT}"
ORCH_PID=$SERVICE_PID

echo ""
echo "========================================"
echo "  All services running."
echo "  Connect your client to:"
echo "    WebSocket → ws://$(hostname):${ORCH_PORT}/ws"
echo "    REST      → http://$(hostname):${ORCH_PORT}"
echo ""
echo "  Logs (live above, also saved to):"
echo "    ${LOG_DIR}/STT.log"
echo "    ${LOG_DIR}/TTS.log"
echo "    ${LOG_DIR}/ORCH.log"
echo "  Ctrl+C to stop everything."
echo "========================================"

wait

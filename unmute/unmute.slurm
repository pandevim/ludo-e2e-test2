#!/bin/bash
#SBATCH --job-name=unmute
#SBATCH --partition=gpu-h100
#SBATCH --nodes=2
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=unmute_%j.out

# =============================================================================
# Unmute Voice Pipeline
#   STT  : kyutai/stt-2.6b-en      (moshi, WebSocket, port 3001)
#   LLM  : google/gemma-3-12b-it   (vLLM, OpenAI API, port 3002)
#   TTS  : kyutai/tts-0.75b-en-public (moshi, HTTP, port 3003)
#   ORCH : orchestrator.py          (WebSocket bridge, port 3004)
# =============================================================================

module load apptainer
source ~/.secrets/unmute.key

# =============================================================================
# Unmute Voice Pipeline — SLURM (2× H100 nodes)
#
#   Node 0  →  STT  (CPU, WebSocket  :3001)
#              ORCH (Python,     HTTP :3004)
#   Node 1  →  TTS  (GPU, HTTP        :3003)
# =============================================================================

set -euo pipefail

# ── Ports ──────────────────────────────────────────────────────────────────────
export STT_PORT=3001
export TTS_PORT=3003
export ORCH_PORT=3004

# ── LLM ───────────────────────────────────────────────────────────────────────
export LLM_URL="https://api.featherless.ai/v1"
export LLM_MODEL="deepseek-ai/DeepSeek-V3.2"
export FEATHERLESS_API_KEY="${FEATHERLESS_API_KEY:-}"

# ── Shared paths ──────────────────────────────────────────────────────────────
export APPTAINER_TMPDIR="$HOME/apptainer_tmp"
export APPTAINER_CACHEDIR="$HOME/.apptainer_cache"
export HF_HOME="$HOME/hf_cache"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$HF_HOME"

# ── Image paths ───────────────────────────────────────────────────────────────
# .sif is cached in SLURM_SUBMIT_DIR so subsequent jobs skip the build.
MOSHI_DEF="$SLURM_SUBMIT_DIR/moshi_with_git.def"
MOSHI_IMAGE="$SLURM_SUBMIT_DIR/moshi_with_git.sif"

LOG_DIR="$SLURM_SUBMIT_DIR/logs/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# ── Resolve node hostnames ─────────────────────────────────────────────────────
NODE_LIST=($(scontrol show hostnames "$SLURM_NODELIST"))
NODE0="${NODE_LIST[0]}"   # STT + Orchestrator
NODE1="${NODE_LIST[1]}"   # TTS (GPU)

echo "========================================"
echo "  Unmute pipeline — Job ${SLURM_JOB_ID}"
echo "  Node0 (STT+ORCH) : ${NODE0}"
echo "  Node1 (TTS/GPU)  : ${NODE1}"
echo "  LLM              : ${LLM_URL} (${LLM_MODEL})"
echo "  Logs             : ${LOG_DIR}/"
echo "========================================"

# ── Apptainer + pip setup (run inside container) ───────────────────────────────
APPTAINER_BASE_OPTS="--nv --userns \
  --env XET_LOG_DIR=/tmp/xet_logs \
  --bind ${HF_HOME}:$HOME/hf_cache \
  --bind ${SLURM_SUBMIT_DIR}:/workspace"

MOSHI_DEPS='export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=$HOME/hf_cache
python3 -m venv --system-site-packages /workspace/.venv
source /workspace/.venv/bin/activate
pip install --upgrade pip -q
pip install --ignore-installed -q fastapi "uvicorn[standard]" websockets numpy scipy
pip install -q --no-deps "git+https://git@github.com/kyutai-labs/moshi#egg=moshi&subdirectory=moshi"'

# ── Helper: wait for HTTP health endpoint (polls from current node) ────────────
wait_for() {
  local url="$1" name="$2"
  local tries=90 interval=5
  echo "  Waiting for ${name} at ${url}..."
  for i in $(seq 1 $tries); do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "  ✓ ${name} ready (attempt ${i})"
      return 0
    fi
    (( i % 6 == 0 )) && echo "  [${name}] ${i}×${interval}s elapsed..."
    sleep "$interval"
  done
  echo "  ✗ ${name} timed out."
  return 1
}

# ── Cleanup ────────────────────────────────────────────────────────────────────
cleanup() {
  echo "Shutting down all srun tasks..."
  # Kill the three background srun processes by their PIDs
  for pid in "${BG_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 5
  for pid in "${BG_PIDS[@]:-}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  echo "Done."
}
trap cleanup EXIT INT TERM
declare -a BG_PIDS=()


# ══════════════════════════════════════════════════════════════════════════════
# 0. Build Apptainer image (once; cached for subsequent jobs)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[0/3] Checking Apptainer image..."

if [[ -f "$MOSHI_IMAGE" ]]; then
  echo "  ✓ Found cached image: ${MOSHI_IMAGE} — skipping build."
else
  echo "  No cached image found. Building from .def (~20-30 min first run)..."

  # Write the .def inline — no separate file needed in the repo
  cat > "$MOSHI_DEF" <<'DEFEOF'
Bootstrap: docker
From: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

%post
    apt-get update
    apt-get install -y git
    apt-get clean
    rm -rf /var/lib/apt/lists/*
DEFEOF

  echo "  Running: apptainer build --fakeroot ${MOSHI_IMAGE} ${MOSHI_DEF}"
  echo "  Build log: ${LOG_DIR}/BUILD.log"

  # --fakeroot = standard unprivileged HPC build; swap for --userns if your
  # site requires it. Build runs on Node 0 only; Node 1 reads from shared FS.
  apptainer build --fakeroot \
    "$MOSHI_IMAGE" \
    "$MOSHI_DEF" \
    2>&1 | tee "${LOG_DIR}/BUILD.log"

  if [[ ! -f "$MOSHI_IMAGE" ]]; then
    echo "  ✗ Build failed — check ${LOG_DIR}/BUILD.log"
    exit 1
  fi
  echo "  ✓ Image built and cached at: ${MOSHI_IMAGE}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# 1. STT  — Node 0, CPU only
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[1/3] Starting STT on ${NODE0} (CPU)..."
srun \
  --nodes=1 --ntasks=1 \
  --nodelist="${NODE0}" \
  --cpus-per-task=32 \
  --gres=gpu:0 \
  --exact \
  apptainer run ${APPTAINER_BASE_OPTS} \
    --env CUDA_VISIBLE_DEVICES="" \
    "$MOSHI_IMAGE" bash -c "
      ${MOSHI_DEPS}
      TORCHDYNAMO_DISABLE=1 python /workspace/stt_server.py --port ${STT_PORT}
    " \
  > "${LOG_DIR}/STT.log" 2>&1 &
BG_PIDS+=($!)
echo "  → STT launched (PID ${BG_PIDS[-1]})"

wait_for "http://${NODE0}:${STT_PORT}/health" "STT" || exit 1

# ══════════════════════════════════════════════════════════════════════════════
# 2. TTS  — Node 1, GPU
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[2/3] Starting TTS on ${NODE1} (GPU)..."
srun \
  --nodes=1 --ntasks=1 \
  --nodelist="${NODE1}" \
  --cpus-per-task=32 \
  --gres=gpu:1 \
  --exact \
  apptainer run ${APPTAINER_BASE_OPTS} \
    "$MOSHI_IMAGE" bash -c "
      ${MOSHI_DEPS}
      TORCHDYNAMO_DISABLE=1 python /workspace/tts_server.py --port ${TTS_PORT}
    " \
  > "${LOG_DIR}/TTS.log" 2>&1 &
BG_PIDS+=($!)
echo "  → TTS launched (PID ${BG_PIDS[-1]})"

wait_for "http://${NODE1}:${TTS_PORT}/health" "TTS" || exit 1

# ══════════════════════════════════════════════════════════════════════════════
# 3. Orchestrator  — Node 0 (head node, no container needed)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[3/3] Starting Orchestrator on ${NODE0}..."
pip install -q fastapi "uvicorn[standard]" websockets httpx 2>/dev/null || true

srun \
  --nodes=1 --ntasks=1 \
  --nodelist="${NODE0}" \
  --cpus-per-task=4 \
  --gres=gpu:0 \
  --exact \
  python "${SLURM_SUBMIT_DIR}/orchestrator.py" \
    --stt-ws    "ws://${NODE0}:${STT_PORT}/ws" \
    --llm-url   "${LLM_URL}" \
    --llm-model "${LLM_MODEL}" \
    --llm-api-key "${FEATHERLESS_API_KEY}" \
    --tts-url   "http://${NODE1}:${TTS_PORT}" \
    --port      "${ORCH_PORT}" \
  > "${LOG_DIR}/ORCH.log" 2>&1 &
BG_PIDS+=($!)
echo "  → Orchestrator launched (PID ${BG_PIDS[-1]})"

# ── Ready ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  All services running."
echo "  Client endpoints:"
echo "    WebSocket → ws://${NODE0}:${ORCH_PORT}/ws"
echo "    REST      → http://${NODE0}:${ORCH_PORT}"
echo ""
echo "  Logs:"
echo "    ${LOG_DIR}/STT.log"
echo "    ${LOG_DIR}/TTS.log"
echo "    ${LOG_DIR}/ORCH.log"
echo "========================================"

# Keep the job alive until all background tasks finish (or time limit hits)
wait
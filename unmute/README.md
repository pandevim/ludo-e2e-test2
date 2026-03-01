# Unmute – Real-Time Voice Pipeline on SLURM

Browser ↔ Orchestrator ↔ STT ↔ vLLM (Gemma) ↔ TTS ↔ Orchestrator ↔ Browser

```
┌─────────────────────────────────────────────────────────────┐
│  H200 GPU node  (single job, all services share GPU memory) │
│                                                             │
│  ┌───────────────┐   audio ws    ┌───────────────────────┐ │
│  │  Browser UI   │◄─────────────►│  orchestrator.py      │ │
│  │  :3004        │               │  :3004  FastAPI/WS    │ │
│  └───────────────┘               └──┬──────────┬──────┬──┘ │
│                                     │          │      │     │
│  ┌──────────────────┐    ws     ┌───┘          │  ┌───┘     │
│  │  stt_server.py   │◄──────────┤              │  │         │
│  │  :3001  moshi    │           │              │  │         │
│  │  kyutai/stt-2.6b │           │        HTTP  │  │         │
│  └──────────────────┘           │              │  │         │
│                                 │    ┌──────────┘  │HTTP    │
│  ┌──────────────────┐  OpenAI   │    │             │        │
│  │  vLLM container  │◄──────────┘    │  ┌──────────┘        │
│  │  :3002           │                │  │                   │
│  │  gemma-3-12b-it  │                │  │                   │
│  └──────────────────┘                │  ▼                   │
│                                      │ ┌──────────────────┐ │
│  ┌──────────────────┐ /synthesize    │ │  tts_server.py   │ │
│  │  Apptainer       │◄───────────────┘ │  :3003  moshi    │ │
│  │  pytorch image   │                  │  kyutai/tts-0.75b│ │
│  └──────────────────┘                  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## GPU memory budget (H200 = 141 GB)

| Service        | Approx. VRAM |
| -------------- | ------------ |
| stt-2.6b-en    | ~6 GB        |
| gemma-3-12b-it | ~26 GB       |
| tts-0.75b-en   | ~2 GB        |
| **Total**      | **~34 GB**   |

`--gpu-memory-utilization 0.35` in vLLM reserves ~49 GB for Gemma, well above its
needs, leaving ample headroom for the other two models.

---

## Quick start

```bash
# 5. Open browser
open http://localhost:3004
```

```bash
chmod +x LOCAL_USE_ONLY_unmute.sh
./LOCAL_USE_ONLY_unmute.sh
```

---

## Files

| File              | Purpose                                      |
| ----------------- | -------------------------------------------- |
| `unmute.slurm`    | SLURM job – starts all 4 services            |
| `stt_server.py`   | FastAPI WebSocket server wrapping kyutai STT |
| `tts_server.py`   | FastAPI HTTP server wrapping kyutai TTS      |
| `orchestrator.py` | Pipeline bridge + browser UI (served at `/`) |
| `.env`            | Ports + HF_TOKEN template                    |

---

## Conversation flow

1. **User presses button** → browser captures mic via `getUserMedia`
2. **Audio streamed** over WebSocket to orchestrator as float32 PCM at 24 kHz
3. Orchestrator **forwards to STT** (`stt_server.py`) via WebSocket
4. STT streams partial + final transcripts back
5. On final transcript, orchestrator calls **vLLM** `/v1/chat/completions`
6. LLM reply sent to **TTS** (`tts_server.py`) via HTTP POST
7. TTS returns float32 PCM audio, orchestrator streams it back to browser
8. Browser plays audio via Web Audio API

---

## Notes

- **STT model**: The moshi library's exact streaming API may differ slightly
  across versions. Check `moshi.models.loaders.CheckpointInfo` attributes and
  adjust `stt_server.py::STTSession._infer_frame()` accordingly.
- **Apptainer pip install at runtime**: moshi is installed fresh each launch.
  For faster starts, pre-build a `.sif` overlay or use `--bind` with a
  pre-installed venv.
- **Multi-GPU**: change `--gres=gpu:N` and add `--tensor-parallel-size N` to
  the vLLM command for larger models.
- **HTTPS/WSS**: for production, put nginx in front of port 8080.

```bash
apptainer build --fakeroot moshi_with_git.sif moshi_with_git.def
```

```bash
python3 - <<'EOF'
import asyncio, websockets, json, subprocess

WS_URL = "ws://localhost:3001/ws"
MP3_FILE = "your_file.mp3"

async def test_stt():
    async with websockets.connect(WS_URL) as ws:
        # Wait for {"type": "ready"}
        ready = await ws.recv()
        print("Server:", ready)

        # Decode MP3 → raw float32 PCM at 24kHz mono
        proc = subprocess.run([
            "ffmpeg", "-i", MP3_FILE,
            "-ar", "24000", "-ac", "1",
            "-f", "f32le",   # float32 little-endian (NOT s16le)
            "-"
        ], capture_output=True)
        pcm = proc.stdout
        print(f"Audio decoded: {len(pcm)} bytes ({len(pcm)/4/24000:.1f}s)")

        # Send in chunks
        chunk_size = 4096
        for i in range(0, len(pcm), chunk_size):
            await ws.send(pcm[i:i+chunk_size])
            await asyncio.sleep(0.02)

        # Send flush signal to finalize
        await ws.send(json.dumps({"type": "flush"}))

        # Collect responses
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print("Response:", msg)
        except asyncio.TimeoutError:
            print("Done.")

asyncio.run(test_stt())
EOF
```

```bash
curl -s -X POST http://localhost:3003/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of the text to speech system."}' \
| ffmpeg -f f32le -ar 24000 -ac 1 -i pipe:0 test_output.wav
```

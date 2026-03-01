"""
tts_server.py - Text-to-Speech server
Model: kyutai/tts-0.75b-en-public (moshi library)

HTTP endpoints:
  GET  /health
       → 200 {"status":"ok"}

  POST /synthesize
       Body: {"text": "Hello world", "voice": "<optional-voice-id>"}
       → 200  audio/pcm  (float32 LE, mono, 24 kHz)
              Header X-Sample-Rate: 24000

  GET  /voices
       → 200 list of built-in voice IDs

WebSocket /ws/synthesize
  Client sends: {"text":"...", "voice":"..."} (JSON)
  Server streams: binary float32 PCM chunks as they are generated
  Then sends: JSON {"type":"done"}
"""

import argparse
import asyncio
import logging

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
import uvicorn

log = logging.getLogger("tts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SAMPLE_RATE   = 24_000
DEFAULT_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"

app = FastAPI(title="Kyutai TTS Service")

tts_model = None
device    = None


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    global tts_model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32 if device.type == "cpu" else torch.bfloat16

    log.info(f"Loading kyutai/tts-0.75b-en-public on {device} ({dtype})...")

    from moshi.models.tts import TTSModel
    from moshi.models.loaders import CheckpointInfo

    checkpoint_info = CheckpointInfo.from_hf_repo("kyutai/tts-0.75b-en-public")
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info,
        n_q=16,
        temp=0.6,
        cfg_coef=3.0,
        max_padding=8,
        initial_padding=2,
        final_padding=4,
        padding_bonus=0.0,
        device=device,
        dtype=dtype,
    )

    log.info("TTS model ready.")


# ── Synthesis helper ──────────────────────────────────────────────────────────
def synthesize_blocking(text: str, voice: str = DEFAULT_VOICE) -> np.ndarray:
    """
    Run TTS synchronously; return float32 numpy array (mono, 24 kHz).
    """
    entries    = tts_model.prepare_script([text], padding_between=1)
    voice_path = tts_model.get_voice_path(voice)
    prefix     = tts_model.get_prefix(voice_path)

    pcms = []
    
    def on_frame(frame):
        if (frame[:, 1:] != -1).all():
            with torch.no_grad():
                pcms.append(tts_model.mimi.decode(frame[:, 1:]).cpu().clamp(-1, 1))

    with torch.no_grad(), tts_model.mimi.streaming(1):
        tts_model.generate(
            [entries],
            [],
            on_frame=on_frame,
            prefixes=[prefix]
        )

    if not pcms:
        return np.zeros(0, dtype=np.float32)

    audio = torch.cat(pcms, dim=-1)
    return audio.squeeze().numpy().astype(np.float32)


async def synthesize_async(text: str, voice: str = DEFAULT_VOICE) -> np.ndarray:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, synthesize_blocking, text, voice)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "model": "kyutai/tts-0.75b-en-public"})


@app.get("/voices")
async def list_voices():
    try:
        voices = tts_model.list_voices() if hasattr(tts_model, "list_voices") else [DEFAULT_VOICE]
    except Exception:
        voices = [DEFAULT_VOICE]
    return JSONResponse({"voices": voices})


@app.post("/synthesize")
async def synthesize_http(body: dict):
    text  = body.get("text", "").strip()
    voice = body.get("voice", DEFAULT_VOICE)

    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    audio = await synthesize_async(text, voice)
    raw   = audio.tobytes()

    return Response(
        content=raw,
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate":   str(SAMPLE_RATE),
            "X-Channels":      "1",
            "X-Sample-Format": "f32le",
            "Content-Length":  str(len(raw)),
        },
    )


@app.websocket("/ws/synthesize")
async def ws_synthesize(ws: WebSocket):
    await ws.accept()
    log.info("TTS WS client connected")

    loop = asyncio.get_running_loop()

    def on_frame_callback(frame):
        if (frame[:, 1:] != -1).all():
            with torch.no_grad():
                pcm = tts_model.mimi.decode(frame[:, 1:]).cpu()
                chunk = pcm.clamp(-1, 1).numpy().astype(np.float32).tobytes()
                asyncio.run_coroutine_threadsafe(ws.send_bytes(chunk), loop)

    try:
        while True:
            data  = await ws.receive_json()
            text  = data.get("text", "").strip()
            voice = data.get("voice", DEFAULT_VOICE)

            if not text:
                continue

            entries    = tts_model.prepare_script([text], padding_between=1)
            voice_path = tts_model.get_voice_path(voice)
            prefix     = tts_model.get_prefix(voice_path)

            with tts_model.mimi.streaming(1):
                await loop.run_in_executor(
                    None, 
                    lambda: tts_model.generate(
                        [entries], 
                        [], 
                        on_frame=on_frame_callback, 
                        prefixes=[prefix]
                    )
                )

            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        log.info("TTS WS client disconnected")
    except Exception as e:
        log.error(f"TTS WS error: {e}", exc_info=True)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3003)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    load_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
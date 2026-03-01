"""
stt_server.py – Streaming Speech-to-Text server
Model: kyutai/stt-2.6b-en  (moshi library)

WebSocket protocol:
  Client → Server: binary frames of float32 PCM at 24 kHz (mono)
                   Send any size; server accumulates into 80 ms windows.
  Server → Client: JSON  {"type":"transcript","text":"...","final":bool}
                         {"type":"ready"}
                         {"type":"error","message":"..."}

HTTP:
  GET /health  → 200 {"status":"ok"}
"""

import argparse
import asyncio
import json
import logging
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

log = logging.getLogger("stt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Model constants (kyutai/stt-2.6b-en) ──────────────────────────────────────
SAMPLE_RATE   = 24_000          # Hz
FRAME_RATE    = 12.5            # frames/sec (Mimi codec)
FRAME_SAMPLES = int(SAMPLE_RATE / FRAME_RATE)   # 1920 samples = 80 ms
TEXT_DELAY    = 2.5             # seconds (model's built-in lookahead)

app = FastAPI(title="Kyutai STT Service")

# ── Globals populated at startup ──────────────────────────────────────────────
stt_model  = None
device     = None


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    global stt_model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading kyutai/stt-2.6b-en on {device}...")

    from moshi.models.loaders import CheckpointInfo
    info = CheckpointInfo.from_hf_repo("kyutai/stt-2.6b-en")

    # The CheckpointInfo exposes the STT model; exact attribute depends on
    # moshi version – fall back gracefully.
    if hasattr(info, "get_stt_model"):
        stt_model = info.get_stt_model().to(device).eval()
    elif hasattr(info, "get_model"):
        stt_model = info.get_model().to(device).eval()
    else:
        raise RuntimeError("Cannot find model loader on CheckpointInfo. "
                           "Check your moshi version.")

    log.info("STT model ready.")


# ── Per-connection inference helper ───────────────────────────────────────────
class STTSession:
    """Wraps one WebSocket connection; accumulates audio and decodes text."""

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._state = None   # carry-over hidden state (if model supports it)

    def push_audio(self, raw_bytes: bytes) -> list[str]:
        """
        Accept raw bytes (float32 LE PCM), return list of decoded text segments.
        """
        chunk = np.frombuffer(raw_bytes, dtype=np.float32)
        self._buf = np.concatenate([self._buf, chunk])

        texts = []
        while len(self._buf) >= FRAME_SAMPLES:
            frame = self._buf[:FRAME_SAMPLES]
            self._buf = self._buf[FRAME_SAMPLES:]
            text = self._infer_frame(frame)
            if text:
                texts.append(text)
        return texts

    def _infer_frame(self, frame: np.ndarray) -> str:
        """Run one 80 ms frame through the STT model."""
        audio_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
        # audio_t shape: [1, 1, 1920]

        with torch.inference_mode():
            # moshi STT forward() returns text token ids or a decoded string
            # depending on version.  Handle both cases.
            out = stt_model(audio_t)

        if isinstance(out, str):
            return out.strip()
        if isinstance(out, torch.Tensor):
            # Decode token ids using the model's own tokenizer/text decoder
            if hasattr(stt_model, "decode_text"):
                return stt_model.decode_text(out).strip()
            if hasattr(stt_model, "tokenizer"):
                ids = out.cpu().tolist()
                ids = [i for i in ids if i > 0]
                return stt_model.tokenizer.decode(ids).strip()
        return ""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "model": "kyutai/stt-2.6b-en"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = STTSession()
    log.info("STT client connected")
    await ws.send_text(json.dumps({"type": "ready"}))

    try:
        while True:
            data = await ws.receive()

            if "bytes" in data and data["bytes"]:
                texts = await asyncio.get_event_loop().run_in_executor(
                    None, session.push_audio, data["bytes"]
                )
                for t in texts:
                    if t:
                        await ws.send_text(json.dumps({
                            "type":  "transcript",
                            "text":  t,
                            "final": False,
                        }))

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "flush":
                    # Client signals end of utterance; return any buffered text
                    texts = await asyncio.get_event_loop().run_in_executor(
                        None, session.push_audio, b""
                    )
                    combined = " ".join(t for t in texts if t).strip()
                    await ws.send_text(json.dumps({
                        "type":  "transcript",
                        "text":  combined,
                        "final": True,
                    }))

    except WebSocketDisconnect:
        log.info("STT client disconnected")
    except Exception as e:
        log.error(f"STT error: {e}", exc_info=True)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    load_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

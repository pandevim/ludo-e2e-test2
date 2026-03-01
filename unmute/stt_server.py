"""
stt_server.py - Streaming Speech-to-Text server
Local Model: kyutai/stt-1b-en_fr (moshi library)

WebSocket protocol:
  Client → Server: binary frames of float32 PCM at 24 kHz (mono)
  Server → Client: JSON {"type":"transcript","text":"...","final":bool}
                         {"type":"ready"}
                         {"type":"error","message":"..."}

HTTP:
  GET /health → 200 {"status":"ok"}
"""

import argparse
import asyncio
import json
import logging
from collections import deque

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from moshi.models import LMGen  # single top-level import, catches missing dep early
import uvicorn

log = logging.getLogger("stt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Kyutai STT Service")

# ── Globals populated at startup ──────────────────────────────────────────────
checkpoint_info = None
text_tokenizer  = None
lm              = None
device          = None
FRAME_SIZE      = None   # samples per mimi frame (1920 @ 24kHz / 12.5fps)

# Semaphore to cap concurrent GPU kernel launches and prevent OOM under load.
MAX_CONCURRENT_SESSIONS = 4
_gpu_semaphore: asyncio.Semaphore | None = None   # initialised in lifespan


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    global checkpoint_info, text_tokenizer, lm, device, FRAME_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32 if device.type == "cpu" else torch.bfloat16

    log.info(f"Loading kyutai/stt-1b-en_fr on {device} ({dtype})...")

    from moshi.models.loaders import CheckpointInfo

    checkpoint_info = CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
    text_tokenizer  = checkpoint_info.get_text_tokenizer()
    lm              = checkpoint_info.get_moshi(device=device, dtype=dtype)

    _tmp_mimi = checkpoint_info.get_mimi(device=device)
    FRAME_SIZE = int(_tmp_mimi.sample_rate / _tmp_mimi.frame_rate)   # 1920
    del _tmp_mimi

    log.info(f"STT model ready. frame_size={FRAME_SIZE}, dep_q={lm.dep_q}")


# ── Per-connection inference session ──────────────────────────────────────────
class CausalSTTSession:
    """
    Each session owns its own `mimi` encoder instance and enters the streaming
    contexts via proper context managers (not streaming_forever) so that close()
    can call __exit__ and release the shared `lm` streaming state.

    This allows a fresh session to be created after each flush without hitting
    the "already streaming!" assertion inside the moshi streaming module.
    """

    def __init__(self):
        self.mimi = checkpoint_info.get_mimi(device=device)
        self.pcm_buffer = np.array([], dtype=np.float32)
        self.lm_gen = LMGen(lm, **checkpoint_info.lm_gen_config)

        # FIX: use streaming() as a proper context manager instead of
        # streaming_forever(), which discards the CM and makes __exit__
        # unreachable.  Storing the CMs lets close() tear down state so
        # the next session can re-enter without "already streaming!" errors.
        self._mimi_ctx = self.mimi.streaming(1)
        self._lm_ctx   = self.lm_gen.streaming(1)
        self._mimi_ctx.__enter__()
        self._lm_ctx.__enter__()

        self.first_step = True

    def close(self):
        """Exit streaming contexts, releasing shared `lm` streaming state."""
        for ctx in (self._lm_ctx, self._mimi_ctx):
            try:
                ctx.__exit__(None, None, None)
            except Exception as e:
                log.warning(f"Error closing streaming context: {e}")

    def _is_special_token(self, tok: int) -> bool:
        return tok in (0, 3, text_tokenizer.eos_id(), text_tokenizer.bos_id())

    def process_stream(self, raw_bytes: bytes) -> str:
        """
        Pulls exact frames from the buffer, runs the forward pass,
        and updates hidden states. Returns any newly emitted text.
        """
        new_audio = np.frombuffer(raw_bytes, dtype=np.float32)
        self.pcm_buffer = np.concatenate((self.pcm_buffer, new_audio))

        emitted_text = ""

        while len(self.pcm_buffer) >= FRAME_SIZE:
            frame = self.pcm_buffer[:FRAME_SIZE]
            self.pcm_buffer = self.pcm_buffer[FRAME_SIZE:]

            in_pcm = (
                torch.from_numpy(frame)
                .to(device=device)
                [None, None, :]          # → [1, 1, FRAME_SIZE]
            )

            with torch.no_grad():
                codes = self.mimi.encode(in_pcm)

                if self.first_step:
                    self.lm_gen.step(codes)
                    self.first_step = False
                    continue

                tokens = self.lm_gen.step(codes)
                if tokens is not None:
                    text_tok = tokens[0, 0, 0].item()
                    if not self._is_special_token(text_tok):
                        piece = text_tokenizer.id_to_piece(text_tok).replace("▁", " ")
                        emitted_text += piece

        return emitted_text


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "model": "kyutai/stt-1b-en_fr"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    async with _gpu_semaphore:
        session = CausalSTTSession()
        log.info("Continuous STT client connected")
        await ws.send_text(json.dumps({"type": "ready"}))

        accumulated_text = []

        try:
            while True:
                data = await ws.receive()
                if data.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(code=data.get("code", 1000))

                if "bytes" in data and data["bytes"]:
                    new_text = await asyncio.to_thread(
                        session.process_stream, data["bytes"]
                    )

                    if new_text:
                        accumulated_text.append(new_text)
                        await ws.send_text(json.dumps({
                            "type":  "transcript_chunk",
                            "text":  new_text,
                            "final": False,
                        }))

                elif "text" in data and data["text"]:
                    msg = json.loads(data["text"])

                    if msg.get("type") == "flush":
                        full_text = "".join(accumulated_text).strip()
                        log.info(f"Flush received. Transcript: {full_text!r}")

                        await ws.send_text(json.dumps({
                            "type":  "transcript",
                            "text":  full_text,
                            "final": True,
                        }))

                        # FIX: exit streaming contexts on the old session BEFORE
                        # creating the new one, so the shared `lm` object is no
                        # longer marked as streaming when the new session calls
                        # lm_gen.streaming(1).__enter__().
                        session.close()
                        accumulated_text = []
                        session = CausalSTTSession()

                        await ws.send_text(json.dumps({"type": "ready"}))

        except WebSocketDisconnect:
            log.info("STT client disconnected")
        except Exception as e:
            log.error(f"STT error: {e}", exc_info=True)
            try:
                await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
        finally:
            # Always clean up streaming state on disconnect/error
            session.close()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--max-sessions", type=int, default=MAX_CONCURRENT_SESSIONS,
        help="Max concurrent GPU inference sessions (tune to available VRAM)",
    )
    args = parser.parse_args()

    _gpu_semaphore = asyncio.Semaphore(args.max_sessions)

    load_model()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
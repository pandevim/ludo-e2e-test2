"""
orchestrator.py – Voice pipeline bridge
                  Browser ↔ STT ↔ LLM (vLLM) ↔ TTS ↔ Browser

WebSocket /ws
  → Client sends: binary float32 PCM audio chunks (24 kHz mono)
  → Client sends: JSON {"type":"flush"} to end utterance
  ← Server sends: binary float32 PCM audio (TTS response)
  ← Server sends: JSON status/transcript/assistant_text messages

REST:
  GET  /health          → {"status":"ok"}
  GET  /voices          → proxied from TTS service
  POST /synthesize      → proxied to TTS (text → PCM bytes)
  POST /chat            → single-turn LLM call (no audio)
"""

import argparse
import asyncio
import json
import logging
import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
import uvicorn

log = logging.getLogger("orch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Filled in from CLI args
STT_WS  = "ws://localhost:3001/ws"
LLM_URL = "http://localhost:3002"
TTS_URL = "http://localhost:3003"

# NOTE: must match the model name passed to vLLM via --model in run.sh
LLM_MODEL = "google/gemma-2b-AWQ"

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. "
    "Respond in short, natural sentences suitable for text-to-speech. "
    "Avoid markdown, bullet points, or long lists."
)

app = FastAPI(title="Unmute Orchestrator")


# ── LLM helper ────────────────────────────────────────────────────────────────
async def call_llm(history: list[dict]) -> str:
    payload = {
        "model":       LLM_MODEL,
        "messages":    [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        "max_tokens":  256,
        "temperature": 0.7,
        "stream":      False,
    }
    headers = {}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{LLM_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ── TTS helper ────────────────────────────────────────────────────────────────
async def call_tts(text: str, voice: str | None = None) -> bytes:
    """Call TTS /synthesize, return raw float32 PCM bytes."""
    body = {"text": text}
    if voice:
        body["voice"] = voice
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{TTS_URL}/synthesize", json=body)
        resp.raise_for_status()
        return resp.content


# ── Per-session pipeline ───────────────────────────────────────────────────────
class Session:
    def __init__(self, ws: WebSocket):
        self.ws      = ws
        self.history: list[dict] = []

    async def send_status(self, msg: str, **kw):
        await self.ws.send_text(json.dumps({"type": "status", "message": msg, **kw}))

    async def run(self):
        stt_ws = None
        try:
            await self.send_status("Connecting to STT...")
            stt_ws = await websockets.connect(STT_WS)
            ready  = json.loads(await stt_ws.recv())
            if ready.get("type") != "ready":
                raise RuntimeError(f"STT unexpected handshake: {ready}")
            await self.send_status("Ready – start speaking")

            async def audio_forwarder():
                """Receive audio from client and forward to STT."""
                while True:
                    data = await self.ws.receive()
                    if "bytes" in data and data["bytes"]:
                        await stt_ws.send(data["bytes"])
                    elif "text" in data:
                        msg = json.loads(data["text"])
                        if msg.get("type") == "flush":
                            await stt_ws.send(json.dumps({"type": "flush"}))
                        elif msg.get("type") == "ping":
                            await self.ws.send_text(json.dumps({"type": "pong"}))

            async def pipeline_runner():
                accumulated_text = ""
                async for raw in stt_ws:
                    log.info(f"STT message received: {raw[:200]}")
                    msg = json.loads(raw)

                    if msg["type"] == "transcript_chunk":
                        accumulated_text += msg.get("text", "")
                        # Optional: send partial transcript to client
                        await self.ws.send_text(json.dumps({
                            "type": "transcript_partial",
                            "text": accumulated_text,
                        }))

                    elif msg["type"] == "flush_ack":
                        full_text = accumulated_text.strip()
                        accumulated_text = ""  # reset for next utterance
                        if not full_text:
                            continue

                        log.info(f"User said: {full_text!r}")
                        await self.ws.send_text(json.dumps({
                            "type": "transcript",
                            "text": full_text,
                        }))

                        # ── LLM ──────────────────────────────────────────────
                        await self.send_status("Thinking...")
                        self.history.append({"role": "user", "content": full_text})
                        try:
                            reply = await call_llm(self.history)
                        except Exception as e:
                            log.error(f"LLM error: {e}")
                            await self.send_status(f"LLM error: {e}")
                            self.history.pop()
                            continue

                        self.history.append({"role": "assistant", "content": reply})
                        await self.ws.send_text(json.dumps({
                            "type": "assistant_text", "text": reply,
                        }))

                        # ── TTS ──────────────────────────────────────────────
                        await self.send_status("Synthesizing speech...")
                        try:
                            audio = await call_tts(reply)
                        except Exception as e:
                            log.error(f"TTS error: {e}")
                            await self.send_status(f"TTS error: {e}")
                            continue

                        chunk_size = 4 * 2400
                        for i in range(0, len(audio), chunk_size):
                            await self.ws.send_bytes(audio[i : i + chunk_size])

                        await self.ws.send_text(json.dumps({"type": "audio_done"}))
                        await self.send_status("Ready – start speaking")

            fwd_task  = asyncio.create_task(audio_forwarder())
            pipe_task = asyncio.create_task(pipeline_runner())
            done, pending = await asyncio.wait(
                [fwd_task, pipe_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for t in pending:
                t.cancel()
            for t in done:
                if t.exception():
                    raise t.exception()

        except WebSocketDisconnect:
            log.info("Client disconnected")
        except Exception as e:
            log.error(f"Session error: {e}", exc_info=True)
            try:
                await self.ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
        finally:
            if stt_ws:
                await stt_ws.close()


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/voices")
async def voices():
    """Proxy the list of available TTS voices."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{TTS_URL}/voices")
        resp.raise_for_status()
        return JSONResponse(resp.json())


@app.post("/synthesize")
async def synthesize(body: dict):
    """
    Direct TTS call — useful for one-shot synthesis without a WebSocket session.
    Body: {"text": "...", "voice": "<optional>"}
    Returns: audio/pcm (float32 LE, mono, 24 kHz)
    """
    text  = body.get("text", "").strip()
    voice = body.get("voice")
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    audio = await call_tts(text, voice)
    return Response(
        content=audio,
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate":   "24000",
            "X-Channels":      "1",
            "X-Sample-Format": "f32le",
        },
    )


@app.post("/chat")
async def chat(body: dict):
    """
    Single-turn LLM call without audio — useful for testing or text-only clients.
    Body: {"message": "...", "history": [{"role":"user","content":"..."},...]}
    Returns: {"reply": "..."}
    """
    message = body.get("message", "").strip()
    history = body.get("history", [])
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    history = list(history) + [{"role": "user", "content": message}]
    try:
        reply = await call_llm(history)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return JSONResponse({"reply": reply})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("New client session")
    await Session(ws).run()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stt-ws",   default="ws://localhost:3001/ws")
    parser.add_argument("--llm-url",  default="http://localhost:3002")
    parser.add_argument("--tts-url",  default="http://localhost:3003")
    parser.add_argument("--llm-model", default=LLM_MODEL,
                        help="Model name sent to vLLM (must match --model in run.sh)")
    parser.add_argument("--llm-api-key", default=None,
                        help="API key for the LLM service (e.g. Featherless)")
    parser.add_argument("--port",     type=int, default=3004)
    parser.add_argument("--host",     default="0.0.0.0")
    args = parser.parse_args()

    STT_WS    = args.stt_ws
    LLM_URL   = args.llm_url
    TTS_URL   = args.tts_url
    LLM_MODEL = args.llm_model
    LLM_API_KEY = args.llm_api_key
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
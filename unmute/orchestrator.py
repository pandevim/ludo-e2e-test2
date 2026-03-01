"""
orchestrator.py – Voice pipeline bridge
                  Browser ↔ STT ↔ LLM (vLLM) ↔ TTS ↔ Browser

Browser connects to ws://HOST:8080/ws
  → sends binary float32 PCM audio chunks (24 kHz mono)
  → sends JSON {"type":"flush"} to end utterance
  ← receives binary float32 PCM audio (TTS response)
  ← receives JSON status messages

Also serves a simple browser UI at GET /
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

log = logging.getLogger("orch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Filled in from CLI args
STT_WS  = "ws://localhost:8001/ws"
LLM_URL = "http://localhost:8002"
TTS_URL = "http://localhost:8003"

# Conversation system prompt
SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. "
    "Respond in short, natural sentences suitable for text-to-speech. "
    "Avoid markdown, bullet points, or long lists."
)

app = FastAPI(title="Unmute Orchestrator")


# ── LLM helper ────────────────────────────────────────────────────────────────
async def call_llm(history: list[dict]) -> str:
    """Call vLLM OpenAI-compatible /v1/chat/completions and return assistant text."""
    payload = {
        "model":       "google/gemma-3-12b-it",
        "messages":    [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        "max_tokens":  256,
        "temperature": 0.7,
        "stream":      False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{LLM_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ── TTS helper ────────────────────────────────────────────────────────────────
async def call_tts(text: str) -> bytes:
    """Call TTS /synthesize, return raw float32 PCM bytes."""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{TTS_URL}/synthesize",
            json={"text": text},
        )
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
            # Connect to STT service
            await self.send_status("Connecting to STT...")
            stt_ws = await websockets.connect(STT_WS)
            ready  = json.loads(await stt_ws.recv())
            if ready.get("type") != "ready":
                raise RuntimeError(f"STT unexpected handshake: {ready}")
            await self.send_status("Ready – start speaking")

            # We run two concurrent tasks:
            #   audio_forwarder: browser audio → STT
            #   transcript_reader: STT text → collect until final
            pending_audio: asyncio.Queue[bytes | None] = asyncio.Queue()

            async def audio_forwarder():
                """Receive audio from browser and forward to STT."""
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
                """Collect final STT transcript → LLM → TTS → send to browser."""
                partial_buf = []
                async for raw in stt_ws:
                    msg = json.loads(raw)
                    if msg["type"] == "transcript":
                        text = msg["text"]
                        if not msg["final"]:
                            # stream partial transcript to browser for display
                            await self.ws.send_text(json.dumps({
                                "type":    "partial_transcript",
                                "text":    text,
                            }))
                            partial_buf.append(text)
                        else:
                            full_text = (text or " ".join(partial_buf)).strip()
                            partial_buf.clear()
                            if not full_text:
                                continue

                            log.info(f"User said: {full_text!r}")
                            await self.ws.send_text(json.dumps({
                                "type": "transcript",
                                "text": full_text,
                            }))

                            # ── LLM ──────────────────────────────────────────
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
                            log.info(f"Assistant: {reply!r}")
                            await self.ws.send_text(json.dumps({
                                "type": "assistant_text",
                                "text": reply,
                            }))

                            # ── TTS ──────────────────────────────────────────
                            await self.send_status("Synthesizing speech...")
                            try:
                                audio = await call_tts(reply)
                            except Exception as e:
                                log.error(f"TTS error: {e}")
                                await self.send_status(f"TTS error: {e}")
                                continue

                            # Stream audio in 100 ms chunks (4 bytes × 2400 samples)
                            chunk = 4 * 2400
                            for i in range(0, len(audio), chunk):
                                await self.ws.send_bytes(audio[i : i + chunk])

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
            log.info("Browser disconnected")
        except Exception as e:
            log.error(f"Session error: {e}", exc_info=True)
            try:
                await self.ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
        finally:
            if stt_ws:
                await stt_ws.close()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("New browser session")
    await Session(ws).run()


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(BROWSER_UI)


# ── Browser UI (single-file, no build step) ────────────────────────────────────
BROWSER_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Unmute – Voice Assistant</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, sans-serif;
    background: #0f1117;
    color: #e8e8f0;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 2rem 1rem;
  }
  h1 { font-size: 1.6rem; margin-bottom: 0.4rem; color: #a78bfa; }
  #status {
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 1.5rem;
    min-height: 1.2em;
  }
  #btn {
    width: 96px; height: 96px;
    border-radius: 50%;
    border: 3px solid #a78bfa;
    background: transparent;
    cursor: pointer;
    font-size: 2.2rem;
    transition: background 0.15s, transform 0.1s;
    color: #e8e8f0;
  }
  #btn.recording { background: #4c1d95; transform: scale(1.08); }
  #btn:disabled  { opacity: 0.4; cursor: default; }
  #chat {
    width: 100%; max-width: 640px;
    margin-top: 2rem;
    display: flex; flex-direction: column; gap: 0.8rem;
  }
  .bubble {
    padding: 0.7rem 1rem;
    border-radius: 12px;
    max-width: 82%;
    line-height: 1.45;
    font-size: 0.95rem;
    word-wrap: break-word;
  }
  .bubble.user { background: #2e1065; align-self: flex-end; }
  .bubble.assistant { background: #1e293b; align-self: flex-start; }
  .bubble.partial { opacity: 0.55; font-style: italic; align-self: flex-end; }
  #vis {
    width: 96px; height: 96px;
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
    background: radial-gradient(circle, rgba(167,139,250,0.25) 0%, transparent 70%);
    transition: transform 0.08s;
  }
  #btn-wrap { position: relative; display: flex; align-items: center; justify-content: center; }
</style>
</head>
<body>
<h1>🎙 Unmute</h1>
<div id="status">Connecting…</div>
<div id="btn-wrap">
  <div id="vis"></div>
  <button id="btn" disabled>🎤</button>
</div>
<div id="chat"></div>

<script>
// ── Config ──────────────────────────────────────────────────────────────────
const SAMPLE_RATE    = 24000;
const WS_URL         = `ws://${location.host}/ws`;
const CHUNK_INTERVAL = 80;   // ms – match moshi frame size

// ── State ────────────────────────────────────────────────────────────────────
let socket, audioCtx, mediaStream, workletNode, analyser;
let recording   = false;
let audioQueue  = [];   // queued float32 arrays for TTS playback
let playingTTS  = false;

// ── DOM ──────────────────────────────────────────────────────────────────────
const btn    = document.getElementById("btn");
const status = document.getElementById("status");
const chat   = document.getElementById("chat");
const vis    = document.getElementById("vis");

function setStatus(s) { status.textContent = s; }

function addBubble(role, text, id) {
  let el = id ? document.getElementById(id) : null;
  if (!el) {
    el = document.createElement("div");
    el.className = `bubble ${role}`;
    if (id) el.id = id;
    chat.appendChild(el);
  }
  el.textContent = text;
  chat.scrollTop = chat.scrollHeight;
  return el.id;
}

// ── WebSocket ────────────────────────────────────────────────────────────────
function connectWS() {
  socket = new WebSocket(WS_URL);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    setStatus("Connected – press to talk");
    btn.disabled = false;
  };

  socket.onclose = () => {
    setStatus("Disconnected – retrying…");
    btn.disabled = true;
    setTimeout(connectWS, 2000);
  };

  socket.onerror = () => setStatus("Connection error");

  socket.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      // TTS audio chunk
      audioQueue.push(new Float32Array(ev.data));
      if (!playingTTS) drainAudio();
      return;
    }
    const msg = JSON.parse(ev.data);
    switch (msg.type) {
      case "status":           setStatus(msg.message); break;
      case "partial_transcript": addBubble("partial", `…${msg.text}`, "partial"); break;
      case "transcript":
        const p = document.getElementById("partial");
        if (p) p.remove();
        addBubble("user", msg.text);
        break;
      case "assistant_text":   addBubble("assistant", msg.text); break;
      case "audio_done":       /* playback handled by drain */ break;
      case "error":            setStatus(`Error: ${msg.message}`); break;
    }
  };
}

// ── Audio playback ────────────────────────────────────────────────────────────
function drainAudio() {
  if (!audioQueue.length) { playingTTS = false; return; }
  playingTTS = true;

  const chunk  = audioQueue.shift();
  const buf    = audioCtx.createBuffer(1, chunk.length, SAMPLE_RATE);
  buf.getChannelData(0).set(chunk);

  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  src.onended = drainAudio;
  src.start();
}

// ── Audio capture ─────────────────────────────────────────────────────────────
async function startRecording() {
  if (!audioCtx) {
    audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  }
  if (audioCtx.state === "suspended") await audioCtx.resume();

  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  const source = audioCtx.createMediaStreamSource(mediaStream);

  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);

  // ScriptProcessor for simplicity (AudioWorklet would be ideal for prod)
  const proc = audioCtx.createScriptProcessor(1920, 1, 1);
  proc.onaudioprocess = (e) => {
    if (!recording) return;
    const pcm = e.inputBuffer.getChannelData(0);  // float32
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(new Float32Array(pcm).buffer);
    }
    // visualise
    const arr = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(arr);
    const amp = arr.reduce((a, v) => a + Math.abs(v - 128), 0) / arr.length;
    vis.style.transform = `scale(${1 + amp * 0.05})`;
  };
  source.connect(proc);
  proc.connect(audioCtx.destination);

  workletNode = proc;
  recording = true;
  btn.classList.add("recording");
  btn.textContent = "⏹";
  setStatus("Listening…");
}

function stopRecording() {
  recording = false;
  vis.style.transform = "scale(1)";
  btn.classList.remove("recording");
  btn.textContent = "🎤";

  // Flush STT
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "flush" }));
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
}

btn.addEventListener("click", () => {
  if (recording) stopRecording();
  else startRecording().catch(e => setStatus(`Mic error: ${e.message}`));
});

// ── Boot ─────────────────────────────────────────────────────────────────────
connectWS();
</script>
</body>
</html>
"""


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stt-ws",  default="ws://localhost:8001/ws")
    parser.add_argument("--llm-url", default="http://localhost:8002")
    parser.add_argument("--tts-url", default="http://localhost:8003")
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--host",    default="0.0.0.0")
    args = parser.parse_args()

    STT_WS  = args.stt_ws
    LLM_URL = args.llm_url
    TTS_URL = args.tts_url

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

"""Minimal ChatGPT-style web chat for the agent (FastAPI, no build step).

    uv run uvicorn mealchat.server:app --reload

POST /api/chat takes (text, optional image file, user_id) multipart — the
image lands in data/uploads and its path goes to agent.handle, exactly like
the CLI/evals. Endpoints are sync defs: FastAPI runs them in its threadpool
and the agent funnels work through its single persistent loop.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from mealchat.agent import MealChatAgent

app = FastAPI(title="mealchat")
agent = MealChatAgent()

_STATIC = Path(__file__).parent / "static"
_UPLOADS = Path("data/uploads")
_UPLOADS.mkdir(parents=True, exist_ok=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.post("/api/chat")
def chat(
    text: str = Form(""),
    image: UploadFile | None = File(None),  # noqa: B008 — FastAPI idiom
    user_id: str = Form("web-user"),
):
    path = None
    if image is not None and image.filename:
        suffix = Path(image.filename).suffix.lower() or ".jpg"
        dest = _UPLOADS / f"{uuid4().hex}{suffix}"
        dest.write_bytes(image.file.read())
        path = str(dest)
    try:
        r = agent.handle(user_id, text or "", image_path=path)
        return {"reply": r.reply, "latency_ms": round(r.latency_ms), "photo": bool(path)}
    except Exception as e:  # noqa: BLE001 — keep the chat alive even if a turn explodes
        return JSONResponse({"reply": f"(error: {e})", "latency_ms": 0, "photo": False})

# Web server: train from browser + chat with the model

import threading
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import ModelRunner
from train import train

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(title="Mini LLM")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

runner = ModelRunner()
train_logs: deque[str] = deque(maxlen=500)
train_lock = threading.Lock()
training = False


class ChatRequest(BaseModel):
    message: str
    history: str = ""


class ChatResponse(BaseModel):
    reply: str
    history: str


class StatusResponse(BaseModel):
    model_loaded: bool
    checkpoint_exists: bool
    training: bool
    device: str


def _train_worker() -> None:
    global training
    try:
        def log(msg: str) -> None:
            train_logs.append(msg)

        train(log=log)
    except Exception as exc:
        train_logs.append(f"error: {exc}")
    finally:
        with train_lock:
            training = False
        train_logs.append("ready — reload model to use new checkpoint")
        try:
            runner.reload()
            train_logs.append("model reloaded")
        except Exception as exc:
            train_logs.append(f"reload skipped: {exc}")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        model_loaded=runner.is_loaded,
        checkpoint_exists=runner.checkpoint_exists,
        training=training,
        device=runner.device,
    )


@app.get("/api/train/logs")
def train_logs_api():
    return {"logs": list(train_logs), "training": training}


@app.post("/api/train/start")
def train_start():
    global training
    with train_lock:
        if training:
            raise HTTPException(409, "Training already running")
        training = True
        train_logs.clear()
        train_logs.append("starting training...")

    thread = threading.Thread(target=_train_worker, daemon=True)
    thread.start()
    return {"ok": True}


@app.post("/api/model/reload")
def model_reload():
    try:
        runner.reload()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, "loaded": runner.is_loaded}


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Message is empty")

    if not runner.checkpoint_exists:
        raise HTTPException(
            404,
            "No checkpoint found. Train the model first (Train tab or python train.py).",
        )

    try:
        reply = runner.chat(body.history, message)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    history = body.history.strip()
    if history:
        history += "\n"
    history += f"User: {message}\nAssistant: {reply}"

    return ChatResponse(reply=reply, history=history)

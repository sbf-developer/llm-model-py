# Read and append training text in data/data.txt

import os
from datetime import datetime

from config import TrainConfig


def data_stats(path: str | None = None) -> dict:
    tcfg = TrainConfig()
    path = path or tcfg.data_path
    if not os.path.isfile(path):
        return {"exists": False, "chars": 0, "lines": 0, "path": path}

    text = open(path, encoding="utf-8").read()
    return {
        "exists": True,
        "chars": len(text),
        "lines": text.count("\n") + (1 if text else 0),
        "path": path,
    }


def append_text(content: str, path: str | None = None, label: str | None = None) -> dict:
    tcfg = TrainConfig()
    path = path or tcfg.data_path
    content = content.strip()
    if not content:
        raise ValueError("Nothing to append — content is empty")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = label or datetime.now().strftime("Appended %Y-%m-%d %H:%M")
    block = f"\n\n=== {header} ===\n\n{content}\n"

    existed = os.path.isfile(path) and os.path.getsize(path) > 0
    mode = "a" if existed else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(block)

    stats = data_stats(path)
    return {"appended_chars": len(block), **stats}


def append_dialogue(user: str, assistant: str, path: str | None = None) -> dict:
    block = f"User: {user.strip()}\nAssistant: {assistant.strip()}\n"
    return append_text(block, path=path, label="Dialogue")

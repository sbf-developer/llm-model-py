# Save, load, resume, and manage training checkpoints

import glob
import os
import re

import torch
import torch.nn as nn

from config import ModelConfig, TrainConfig
from model import GPT
from tokenizer import CharTokenizer


def infer_checkpoint_step(ckpt: dict, path: str) -> int:
    # legacy checkpoints may lack step — parse from filename if possible
    if ckpt.get("step") is not None:
        return int(ckpt["step"])
    m = re.search(r"step_(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return 0


def expand_model_vocab(model: GPT, new_vocab_size: int) -> None:
    # grow embedding table when new characters are added to the data
    old_size = model.cfg.vocab_size
    if new_vocab_size <= old_size:
        return

    old_emb = model.token_emb.weight.data.clone()
    new_emb = nn.Embedding(new_vocab_size, model.cfg.d_model).to(old_emb.device)
    new_emb.weight.data[:old_size] = old_emb
    nn.init.normal_(new_emb.weight.data[old_size:], mean=0.0, std=0.02)

    model.token_emb = new_emb
    model.lm_head.weight = model.token_emb.weight
    model.cfg.vocab_size = new_vocab_size


def save_checkpoint(
    path: str,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    mcfg: ModelConfig,
    tok: CharTokenizer,
    val_loss: float | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "mcfg": mcfg,
            "stoi": tok.stoi,
            "itos": tok.itos,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path: str, device: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def prepare_tokenizer(text: str, ckpt: dict | None) -> tuple[CharTokenizer, bool]:
    # reuse saved vocab and merge any new characters from updated data.txt
    if ckpt is None:
        return CharTokenizer(text), False

    tok = CharTokenizer.from_maps(ckpt["stoi"], ckpt["itos"])
    grew = tok.merge_new_chars(text)
    return tok, grew


def build_model_from_checkpoint(
    mcfg: ModelConfig,
    tok: CharTokenizer,
    ckpt: dict | None,
    device: str,
) -> GPT:
    if ckpt is not None:
        mcfg.vocab_size = ckpt["mcfg"].vocab_size
    else:
        mcfg.vocab_size = tok.vocab_size

    model = GPT(mcfg).to(device)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        if tok.vocab_size > model.cfg.vocab_size:
            expand_model_vocab(model, tok.vocab_size)
        mcfg.vocab_size = tok.vocab_size
    else:
        mcfg.vocab_size = tok.vocab_size

    return model


def build_optimizer(
    model: GPT,
    lr: float,
    ckpt: dict | None,
    *,
    reset_state: bool = False,
) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if reset_state or ckpt is None or "optimizer" not in ckpt:
        return optimizer
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except (ValueError, KeyError, RuntimeError):
        pass  # state mismatch after vocab expand — fresh optimizer is fine
    return optimizer


def resolve_resume_path(tcfg: TrainConfig, resume_from: str | None, fresh: bool) -> str | None:
    if fresh:
        return None
    if resume_from:
        return resume_from
    if not tcfg.auto_resume:
        return None
    latest = os.path.join(tcfg.checkpoint_dir, "latest.pt")
    return latest if os.path.isfile(latest) else None


def checkpoint_paths_for_step(tcfg: TrainConfig, step: int) -> list[str]:
    base = tcfg.checkpoint_dir
    paths = [
        os.path.join(base, "latest.pt"),
        os.path.join(base, f"step_{step:06d}.pt"),
    ]
    if step % (tcfg.save_every * 5) == 0:
        paths.append(os.path.join(base, f"milestone_step_{step:06d}.pt"))
    return paths


def prune_checkpoints(tcfg: TrainConfig) -> None:
    pattern = os.path.join(tcfg.checkpoint_dir, "step_*.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for path in files[tcfg.keep_checkpoints :]:
        os.remove(path)


def list_checkpoints(checkpoint_dir: str) -> list[dict]:
    if not os.path.isdir(checkpoint_dir):
        return []

    entries = []
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt"))):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            entries.append(
                {
                    "path": path,
                    "name": os.path.basename(path),
                    "step": infer_checkpoint_step(ckpt, path) or ckpt.get("step", "?"),
                    "val_loss": ckpt.get("val_loss"),
                    "vocab_size": ckpt.get("mcfg").vocab_size if ckpt.get("mcfg") else "?",
                }
            )
        except Exception:
            entries.append({"path": path, "name": os.path.basename(path), "step": "?"})
    return entries

# Trains the model on data/data.txt — supports resume and multiple checkpoints

import argparse
import os
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from tokenizer import CharTokenizer
from model import GPT
from dataset import CharDataset
from checkpoints_util import (
    build_model_from_checkpoint,
    build_optimizer,
    checkpoint_paths_for_step,
    infer_checkpoint_step,
    load_checkpoint,
    prepare_tokenizer,
    prune_checkpoints,
    resolve_resume_path,
    save_checkpoint,
)

LogFn = Callable[[str], None]
LOCK_FILE = Path("checkpoints/.training.lock")


def acquire_train_lock() -> None:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        raise RuntimeError(
            "Training already in progress (lock file exists). "
            "Wait for the other run to finish or delete checkpoints/.training.lock if stuck."
        )
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")


def release_train_lock() -> None:
    if LOCK_FILE.exists():
        LOCK_FILE.unlink(missing_ok=True)


def train(
    log: LogFn | None = None,
    fresh: bool = False,
    resume_from: str | None = None,
) -> None:
    def say(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg, flush=True)

    mcfg = ModelConfig()
    tcfg = TrainConfig()

    device = tcfg.device if torch.cuda.is_available() else "cpu"
    say(f"device: {device}")

    acquire_train_lock()
    try:
        _run_training(say, mcfg, tcfg, device, fresh, resume_from)
    finally:
        release_train_lock()


def _run_training(
    say: LogFn,
    mcfg: ModelConfig,
    tcfg: TrainConfig,
    device: str,
    fresh: bool,
    resume_from: str | None,
) -> None:

    # CPU stability: limit threads and batch size to reduce crashes on Windows
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        if tcfg.batch_size > 16:
            say(f"cpu: reducing batch_size {tcfg.batch_size} -> 16 for stability")
            tcfg.batch_size = 16

    text = open(tcfg.data_path, encoding="utf-8").read()
    if not text.strip():
        raise ValueError(f"No text found in {tcfg.data_path}")

    resume_path = resolve_resume_path(tcfg, resume_from, fresh)
    ckpt = load_checkpoint(resume_path, device) if resume_path else None

    tok, vocab_grew = prepare_tokenizer(text, ckpt)
    mcfg.vocab_size = tok.vocab_size
    say(f"vocab size: {mcfg.vocab_size} | text length: {len(text):,} chars")
    if vocab_grew and ckpt is not None:
        say("new characters found in data - vocab expanded (training continues, not restarted)")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(
        CharDataset(train_data, mcfg.block_size),
        batch_size=tcfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CharDataset(val_data, mcfg.block_size),
        batch_size=tcfg.batch_size,
    )

    model = build_model_from_checkpoint(mcfg, tok, ckpt, device)
    optimizer = build_optimizer(model, tcfg.lr, ckpt)
    params = sum(p.numel() for p in model.parameters())
    say(f"parameters: {params:,}")

    start_step = infer_checkpoint_step(ckpt, resume_path) if ckpt and resume_path else 0
    if ckpt and start_step > 0:
        say(f"resuming from step {start_step} -> target {tcfg.max_iters}")
    elif ckpt and resume_path:
        say(
            f"legacy checkpoint loaded (weights only, step unknown). "
            f"Use checkpoints/step_XXXXXX.pt or: python scripts/debug.py migrate {resume_path} --step N"
        )

    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    train_iter = iter(train_loader)

    if start_step >= tcfg.max_iters:
        say(f"already reached target ({start_step} >= {tcfg.max_iters}). use --fresh or raise max_iters.")
        return

    say(f"training steps {start_step + 1}-{tcfg.max_iters} (log every {tcfg.log_interval})...")
    if start_step == 0:
        say("first steps on CPU can take a few minutes - please wait")

    for step in range(start_step + 1, tcfg.max_iters + 1):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == start_step + 1 or step % tcfg.log_interval == 0:
            say(f"step {step:5d} | train loss {loss.item():.4f}")

        if step % tcfg.save_every == 0:
            model.eval()
            losses = []
            max_val_batches = 50 if device == "cpu" else None  # full val on CPU takes 10+ min
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    vx, vy = vx.to(device), vy.to(device)
                    _, vloss = model(vx, vy)
                    losses.append(vloss.item())
                    if max_val_batches and i + 1 >= max_val_batches:
                        break
            val_loss = sum(losses) / len(losses)
            say(f"step {step:5d} | val loss   {val_loss:.4f}")

            for ckpt_path in checkpoint_paths_for_step(tcfg, step):
                save_checkpoint(ckpt_path, model, optimizer, step, mcfg, tok, val_loss)
                say(f"saved {ckpt_path}")

            prune_checkpoints(tcfg)

    say("training done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the mini LLM")
    parser.add_argument("--fresh", action="store_true", help="ignore saved checkpoints and start over")
    parser.add_argument("--resume-from", default=None, help="path to a specific .pt checkpoint")
    args = parser.parse_args()
    train(fresh=args.fresh, resume_from=args.resume_from)


if __name__ == "__main__":
    main()

# Trains the model on data/data.txt and saves checkpoints/latest.pt

import os
from typing import Callable

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from tokenizer import CharTokenizer
from model import GPT
from dataset import CharDataset

LogFn = Callable[[str], None]


def train(log: LogFn | None = None) -> None:
    def say(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg)

    mcfg = ModelConfig()
    tcfg = TrainConfig()

    device = tcfg.device if torch.cuda.is_available() else "cpu"
    say(f"device: {device}")

    text = open(tcfg.data_path, encoding="utf-8").read()
    if not text.strip():
        raise ValueError(f"No text found in {tcfg.data_path}")

    tok = CharTokenizer(text)
    mcfg.vocab_size = tok.vocab_size
    say(f"vocab size: {mcfg.vocab_size} | text length: {len(text):,} chars")

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

    model = GPT(mcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
    params = sum(p.numel() for p in model.parameters())
    say(f"parameters: {params:,}")

    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    train_iter = iter(train_loader)

    for step in range(1, tcfg.max_iters + 1):
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

        if step % tcfg.log_interval == 0:
            say(f"step {step:5d} | train loss {loss.item():.4f}")

        if step % tcfg.eval_interval == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    _, vloss = model(vx, vy)
                    losses.append(vloss.item())
            val_loss = sum(losses) / len(losses)
            say(f"step {step:5d} | val loss   {val_loss:.4f}")

            ckpt_path = os.path.join(tcfg.checkpoint_dir, "latest.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "mcfg": mcfg,
                    "stoi": tok.stoi,
                    "itos": tok.itos,
                },
                ckpt_path,
            )
            say(f"saved {ckpt_path}")

    say("training done.")


def main() -> None:
    train()


if __name__ == "__main__":
    main()

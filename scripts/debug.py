# Debug helpers: smoke tests and checkpoint repair

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import ModelConfig, TrainConfig
from checkpoints_util import load_checkpoint, save_checkpoint, list_checkpoints
from data_util import data_stats
from inference import ModelRunner
from tokenizer import CharTokenizer
from model import GPT


def infer_step(ckpt: dict, path: str) -> int:
    if ckpt.get("step") is not None:
        return int(ckpt["step"])
    m = re.search(r"step_(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return 0


def migrate_checkpoint(path: str, step: int | None = None) -> None:
    ckpt = load_checkpoint(path, "cpu")
    if ckpt.get("step") is not None and ckpt.get("optimizer"):
        print(f"already migrated: {path} (step {ckpt['step']})")
        return

    if step is None:
        step = infer_step(ckpt, path)
    if step == 0:
        raise SystemExit(
            f"Cannot infer step for {path}. Pass --step 500 manually."
        )

    tok = CharTokenizer.from_maps(ckpt["stoi"], ckpt["itos"])
    mcfg = ckpt["mcfg"]
    model = GPT(mcfg)
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    save_checkpoint(path, model, optimizer, step, mcfg, tok, ckpt.get("val_loss"))
    print(f"migrated {path} -> step {step}")


def smoke_test() -> None:
    print("=== smoke test ===")
    stats = data_stats()
    print(f"data: {stats['chars']:,} chars")

    tcfg = TrainConfig()
    text = open(tcfg.data_path, encoding="utf-8").read()[:8000]
    tok = CharTokenizer(text)
    mcfg = ModelConfig(vocab_size=tok.vocab_size)

    from torch.utils.data import DataLoader
    from dataset import CharDataset

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    loader = DataLoader(CharDataset(data, mcfg.block_size), batch_size=8, shuffle=True)
    model = GPT(mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x, y = next(iter(loader))
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    print(f"train step ok, loss={loss.item():.4f}")

    out = model.generate(x[:, :20], max_new_tokens=10, top_k=20)
    print(f"generate ok, len={out.shape[1]}")

    runner = ModelRunner()
    if runner.checkpoint_exists:
        runner.load()
        reply = runner.chat("", "Hello", max_new_tokens=50)
        print(f"chat ok: {reply[:80]!r}")
    else:
        print("no checkpoint yet — chat skipped")

    print("=== all ok ===")


def main():
    parser = argparse.ArgumentParser(description="Debug and smoke-test the mini LLM")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="quick sanity check")

    p_mig = sub.add_parser("migrate", help="add step/optimizer to legacy checkpoint")
    p_mig.add_argument("path", help="checkpoint .pt file")
    p_mig.add_argument("--step", type=int, default=None)

    sub.add_parser("list", help="list checkpoints")

    args = parser.parse_args()

    if args.cmd == "smoke":
        smoke_test()
    elif args.cmd == "migrate":
        migrate_checkpoint(args.path, args.step)
    elif args.cmd == "list":
        tcfg = TrainConfig()
        for c in list_checkpoints(tcfg.checkpoint_dir):
            print(f"{c['name']:30s} step={c['step']} val_loss={c.get('val_loss')}")


if __name__ == "__main__":
    main()

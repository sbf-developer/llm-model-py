# Load checkpoint and run generation for CLI + web chat

import os
import torch

from config import GenerateConfig, ModelConfig
from tokenizer import CharTokenizer
from model import GPT


def _clean_chat_reply(reply: str) -> str:
    reply = reply.strip()

    # cut off when the model starts a new turn
    for stop in ("\nUser:", "\nAssistant:", "\n===", "\n---", "\nArticle:"):
        if stop in reply:
            reply = reply.split(stop, 1)[0]

    # same-turn junk without a leading newline
    for stop in (" User:", " Assistant:"):
        if stop in reply:
            reply = reply.split(stop, 1)[0]

    # strip repeated role labels the model often regurgitates
    labels = ("Assistant:", "assistant:", "User:", "user:")
    while reply:
        stripped = reply.lstrip()
        removed = False
        for label in labels:
            if stripped.startswith(label):
                stripped = stripped[len(label):].lstrip()
                removed = True
                break
        if not removed:
            break
        reply = stripped

    return reply.strip()


def _trim_history(history: str, max_chars: int) -> str:
    # keep recent turns that fit in the model context window
    history = history.strip()
    if max_chars <= 0 or not history:
        return ""
    if len(history) <= max_chars:
        return history

    chunk = history[-max_chars:]
    turn_start = chunk.find("\nUser:")
    if turn_start >= 0:
        chunk = chunk[turn_start + 1 :]
    elif not chunk.startswith("User:"):
        nl = chunk.find("\n")
        if nl >= 0:
            chunk = chunk[nl + 1 :]
    return chunk.strip()


class ModelRunner:
    # Wraps loaded model + tokenizer for reuse

    def __init__(self):
        self.gcfg = GenerateConfig()
        self.device = self.gcfg.device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.mcfg = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def checkpoint_exists(self) -> bool:
        return os.path.isfile(self.gcfg.checkpoint_path)

    def load(self) -> None:
        if not self.checkpoint_exists:
            raise FileNotFoundError(
                f"No checkpoint at {self.gcfg.checkpoint_path}. Train the model first."
            )

        ckpt = torch.load(self.gcfg.checkpoint_path, map_location=self.device, weights_only=False)
        self.mcfg = ckpt["mcfg"]
        self.tokenizer = CharTokenizer.from_maps(ckpt["stoi"], ckpt["itos"])

        self.model = GPT(self.mcfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

    def reload(self) -> None:
        self.unload()
        self.load()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.mcfg = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def complete(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> str:
        if not self.is_loaded:
            self.load()

        max_new_tokens = max_new_tokens or self.gcfg.max_new_tokens
        temperature = temperature if temperature is not None else self.gcfg.temperature
        top_k = top_k if top_k is not None else self.gcfg.top_k

        start = torch.tensor(
            [self.tokenizer.encode(prompt)],
            dtype=torch.long,
            device=self.device,
        )
        out = self.model.generate(
            start,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return self.tokenizer.decode(out[0].tolist())

    def chat(self, history: str, user_message: str, max_new_tokens: int | None = None) -> str:
        # format matches training data: User: ... Assistant: ...
        user_message = user_message.strip()
        if not user_message:
            return ""

        if not self.is_loaded:
            self.load()

        max_new_tokens = max_new_tokens or self.gcfg.chat_max_new_tokens
        suffix = f"User: {user_message}\nAssistant: "
        block = self.mcfg.block_size if self.mcfg else ModelConfig().block_size
        history_budget = max(0, block - len(suffix) - max_new_tokens - 8)

        prompt = _trim_history(history, history_budget)
        if prompt:
            prompt += "\n"
        prompt += suffix

        full = self.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=self.gcfg.chat_temperature,
            top_k=self.gcfg.chat_top_k,
        )
        reply = _clean_chat_reply(full[len(prompt):])
        return reply or "(empty reply — try training longer)"

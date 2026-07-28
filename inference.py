# Load checkpoint and run generation for CLI + web chat

import os
import torch

from config import GenerateConfig, ModelConfig
from tokenizer import CharTokenizer
from model import GPT


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

        ckpt = torch.load(self.gcfg.checkpoint_path, map_location=self.device)
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

    def chat(self, history: str, user_message: str, max_new_tokens: int = 200) -> str:
        # format matches training data: User: ... Assistant: ...
        user_message = user_message.strip()
        if not user_message:
            return ""

        prompt = history.strip()
        if prompt:
            prompt += "\n"
        prompt += f"User: {user_message}\nAssistant:"

        full = self.complete(prompt, max_new_tokens=max_new_tokens)
        reply = full[len(prompt):].strip()

        # stop if model runs into another turn
        for stop in ("\nUser:", "\nAssistant:", "\n==="):
            if stop in reply:
                reply = reply.split(stop)[0].strip()
        return reply or "(empty reply — try training longer)"

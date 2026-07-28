# Settings / tuning knobs — other files read these, this file does not run anything

from dataclasses import dataclass


# --- Model size & architecture ---
@dataclass
class ModelConfig:
    vocab_size: int = 256      # how many unique chars/symbols (auto-set from data in train.py)
    block_size: int = 256      # how many chars the model sees at once (context window)
    n_layers: int = 4            # number of transformer layers (depth)
    n_heads: int = 4             # attention heads per layer (d_model must divide evenly)
    d_model: int = 128           # width of the model (bigger = more capacity, more memory)
    dropout: float = 0.1         # random dropout while training (helps avoid memorizing)


# --- Training ---
@dataclass
class TrainConfig:
    batch_size: int = 32         # samples per training step
    lr: float = 3e-4               # learning rate (step size when updating weights)
    max_iters: int = 5000          # total training steps
    eval_interval: int = 500       # how often to check validation loss
    log_interval: int = 100        # how often to print training loss
    device: str = "cuda"           # "cuda" for GPU, falls back to cpu in train.py
    data_path: str = "data/data.txt"
    checkpoint_dir: str = "checkpoints"


# --- Text generation (used by generate.py after training) ---
@dataclass
class GenerateConfig:
    checkpoint_path: str = "checkpoints/latest.pt"
    prompt: str = "Hello"          # text the model continues from
    max_new_tokens: int = 300      # how many chars to generate
    temperature: float = 0.8       # higher = more random/creative
    top_k: int = 40                # only sample from top K likely chars
    device: str = "cuda"

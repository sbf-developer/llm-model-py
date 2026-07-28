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
    batch_size: int = 8          # samples per step (auto-capped on CPU in train.py)
    lr: float = 3e-4               # learning rate (step size when updating weights)
    max_iters: int = 10000         # total steps to reach (resume continues until this)
    log_interval: int = 10         # how often to print training loss
    device: str = "cuda"           # "cuda" for GPU, falls back to cpu in train.py
    data_path: str = "data/data.txt"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 100         # save a checkpoint every N steps (lower = safer on CPU crashes)
    keep_checkpoints: int = 20     # keep this many step_*.pt files (older ones deleted)
    auto_resume: bool = True         # continue from latest.pt on next train.py run


# --- Text generation (used by generate.py after training) ---
@dataclass
class GenerateConfig:
    checkpoint_path: str = "checkpoints/latest.pt"
    prompt: str = "Hello"          # text the model continues from
    max_new_tokens: int = 300      # how many chars to generate
    temperature: float = 0.8       # higher = more random/creative
    top_k: int = 40                # only sample from top K likely chars
    device: str = "cuda"

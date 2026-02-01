from dataclasses import dataclass

import torch


@dataclass
class Config:
    # --- Model Architecture ---
    vocab_size: int = 50304  # GPT-2 vocab size (multiple of 64 for efficiency)
    block_size: int = 128  # Context window (how much code it can see)
    n_layer: int = 6  # Number of transformer blocks
    n_head: int = 6  # Number of attention heads
    n_embd: int = 384  # Embedding dimension (hidden size)
    dropout: float = 0.2

    # --- Training ---
    batch_size: int = 64  # Micro-batch size
    max_iters: int = 5000  # 5000  # Total training steps
    learning_rate: float = 3e-4
    eval_interval: int = 100  # Evaluate every N steps
    eval_iters: int = 20  # Number of batches to average for loss
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data (Python Specific) ---
    # Load from local Arrow dataset (no internet required)
    use_local_dataset: bool = True  # Use local dataset instead of downloading
    local_dataset_path: str = "local_dataset_own"  # Path to local Arrow dataset
    max_samples: int = 10000  # Number of samples to use from local dataset
    tokenizer_name: str = "gpt2"  # Using GPT-2 tokenizer as a base for code

    # Legacy online dataset settings (not used when use_local_dataset=True)
    dataset_name: str = "codeparrot/codeparrot-clean-train"
    dataset_subset: str = None
    data_split: str = "train"

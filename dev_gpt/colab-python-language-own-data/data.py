import tiktoken
import torch
from config import Config
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    """
    Loads code from local Arrow dataset or HuggingFace, tokenizes it, and chunks it.
    """

    def __init__(self, config):
        self.config = config

        if config.use_local_dataset:
            print(
                f"Loading dataset from local Arrow files: {config.local_dataset_path}"
            )
            # Load from local Arrow dataset - much faster!
            raw_data = load_from_disk(config.local_dataset_path)

            # Optionally limit samples
            if config.max_samples and config.max_samples < len(raw_data):
                print(
                    f"Using first {config.max_samples} samples from {len(raw_data)} total"
                )
                raw_data = raw_data.select(range(config.max_samples))
            else:
                print(f"Using all {len(raw_data)} samples")
        else:
            print(f"Loading dataset from HuggingFace...")
            # Load from HuggingFace (requires internet)
            raw_data = load_dataset(
                config.dataset_name,
                config.dataset_subset,
                split=config.data_split,
                streaming=True,
            )

        print("Tokenizing data...")
        self.tokenizer = tiktoken.get_encoding(config.tokenizer_name)
        self.tokens = []

        # Process items
        count = 0
        for item in raw_data:
            # Get text content (codeparrot uses 'content' field, wikitext uses 'text')
            text = item.get("content") or item.get("text") or item.get("code", "")
            if not text or len(text.strip()) == 0:
                continue
            # Encode
            ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            self.tokens.extend(ids)
            count += 1

            # Progress indicator for large datasets
            if count % 1000 == 0:
                print(f"  Tokenized {count} samples...")

            # For streaming datasets, limit the count
            if not config.use_local_dataset and count >= 5000:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Dataset size: {len(self.tokens):,} tokens from {count} samples")

    def __len__(self):
        return len(self.tokens) // self.config.block_size - 1

    def __getitem__(self, idx):
        # Grab a chunk of block_size from the data
        x = self.tokens[
            idx * self.config.block_size : (idx + 1) * self.config.block_size
        ]
        y = self.tokens[
            idx * self.config.block_size + 1 : (idx + 1) * self.config.block_size + 1
        ]
        return x, y


def get_batch(dataset, config):
    """Create a random batch of data."""
    ix = torch.randint(len(dataset), (config.batch_size,))
    x = torch.stack([dataset[i][0] for i in ix])
    y = torch.stack([dataset[i][1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

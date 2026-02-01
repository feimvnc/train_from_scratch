import tiktoken
import torch
from config import Config
from datasets import load_dataset
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    """
    Loads code from GitHub repositories, tokenizes it, and chunks it.
    """

    def __init__(self, config):
        self.config = config
        print(f"Loading dataset from HuggingFace...")

        # Load a small slice of the dataset for demonstration
        # Using streaming to avoid downloading large datasets
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
            if count >= 5000:  # Limit to 5000 items
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Dataset size: {len(self.tokens)} tokens")

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

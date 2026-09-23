import torch
from torch.utils.data import Dataset


class LyricsDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, index):
        x = torch.tensor(
            self.tokens[index : index + self.block_size],
            dtype=torch.long,
        )

        y = torch.tensor(
            self.tokens[index + 1 : index + self.block_size + 1],
            dtype=torch.long,
        )

        return x, y
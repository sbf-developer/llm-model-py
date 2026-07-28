import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
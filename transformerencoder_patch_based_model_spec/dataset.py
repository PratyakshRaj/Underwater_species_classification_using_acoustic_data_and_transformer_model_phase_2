import torch
from torch.utils.data import Dataset
import numpy as np


MAX_TIME = 5000
MIN_TIME = 32  # must be >= patch_time

class SpectrogramDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec = np.load(self.df.loc[idx, "path"])  # (1, 128, T)
        spec = torch.tensor(spec, dtype=torch.float32)


        if spec.shape[-1] < MIN_TIME:
            pad_size = MIN_TIME - spec.shape[-1]
            spec = torch.nn.functional.pad(spec, (0, pad_size))
        
        if spec.shape[-1] > MAX_TIME:
            spec = spec[:, :, :MAX_TIME]

        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        label = torch.tensor(self.df.loc[idx, "label"], dtype=torch.long)

        return spec, spec.shape[-1], label

import torch

def collate_fn(batch):
    specs, lengths, labels = zip(*batch)

    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = max(lengths).item()

    batch_size = len(specs)

    # specs are already (1, 128, T)
    padded_specs = torch.zeros(batch_size, 1, 128, max_len)

    for i, spec in enumerate(specs):
        T = spec.shape[-1]
        padded_specs[i, :, :, :T] = spec

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_specs, lengths, labels
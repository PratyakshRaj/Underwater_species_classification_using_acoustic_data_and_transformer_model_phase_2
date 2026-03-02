import torch
from torch.utils.data import Dataset
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec = np.load(self.df.loc[idx, "path"])  # (1, 128, T)

        spec = torch.tensor(spec, dtype=torch.float32)

        # Expecting (1, 128, T)
        if spec.ndim != 3:
            raise ValueError(f"Expected 3D spectrogram, got {spec.shape}")

        # Remove channel dimension
        spec = spec.squeeze(0)  # (128, T)

        if spec.ndim != 2:
            raise ValueError(f"After squeeze got wrong shape {spec.shape}")

        # Convert to (T, 128)
        spec = spec.transpose(0, 1)
        
        MAX_TIME = 5000
        if spec.shape[0] > MAX_TIME:
            spec = spec[:MAX_TIME, :]
        
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        label = torch.tensor(self.df.loc[idx, "label"], dtype=torch.long)

        return spec, label

import torch

def collate_fn(batch):
    specs, labels = zip(*batch)

    lengths = [s.shape[0] for s in specs]
    max_len = max(lengths)

    batch_size = len(specs)
    feature_dim = specs[0].shape[1]  # should be 128

    padded_specs = torch.zeros(batch_size, max_len, feature_dim)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, s in enumerate(specs):
        length = s.shape[0]

        padded_specs[i, :length] = s

        # False = valid
        padding_mask[i, :length] = False

    return padded_specs, padding_mask, torch.tensor(labels)    
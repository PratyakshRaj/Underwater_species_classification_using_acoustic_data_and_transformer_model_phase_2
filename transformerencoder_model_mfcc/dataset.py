import torch
from torch.utils.data import Dataset
import numpy as np

class MfccDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mfcc = np.load(self.df.loc[idx, "path"])  # (1, 36, T)

        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        # Expecting (1, 36, T)
        if mfcc.ndim != 3:
            raise ValueError(f"Expected 3D mfcctrogram, got {mfcc.shape}")

        # Remove channel dimension
        mfcc = mfcc.squeeze(0)  # (36, T)

        if mfcc.ndim != 2:
            raise ValueError(f"After squeeze got wrong shape {mfcc.shape}")

        # Convert to (T, 36)
        mfcc = mfcc.transpose(0, 1)
        
        MAX_TIME = 5000
        if mfcc.shape[0] > MAX_TIME:
            mfcc = mfcc[:MAX_TIME, :]
        
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        label = torch.tensor(self.df.loc[idx, "label"], dtype=torch.long)

        return mfcc, label

import torch

def collate_fn(batch):
    mfccs, labels = zip(*batch)

    lengths = [s.shape[0] for s in mfccs]
    max_len = max(lengths)

    batch_size = len(mfccs)
    feature_dim = mfccs[0].shape[1]  # should be 36

    padded_mfccs = torch.zeros(batch_size, max_len, feature_dim)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, s in enumerate(mfccs):
        length = s.shape[0]

        padded_mfccs[i, :length] = s

        # False = valid
        padding_mask[i, :length] = False

    return padded_mfccs, padding_mask, torch.tensor(labels)    
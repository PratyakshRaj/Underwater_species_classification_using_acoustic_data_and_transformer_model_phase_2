
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Torch hub is one thing
from torch.hub import load
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Position indices [0, 1, 2, ...]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # Frequency terms for sine/cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices; cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        
        # Add batch dimension → shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)  
        
        # Register as buffer so it's saved with model but not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x




class TransformerModel(nn.Module):
    def __init__(self, input_dim=36,d_model=72, num_classes=25, n_heads=4,n_layers=4):
        super().__init__()
        
        self.input_proj=nn.Linear(input_dim,d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,dim_feedforward=4*d_model,dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, mfcc,mfcc_mask=None):
        # (B, 1, time, mel)
        
       
        mfcc = mfcc.permute(0,2,1)           # (B, T, 36)
        mfcc=self.input_proj(mfcc)
        
        mfcc_ = self.pos_enc(mfcc)
        encoded = self.transformer(mfcc_,src_key_padding_mask=~mfcc_mask if mfcc_mask is not None else None)
        
        if mfcc_mask is not None:
            mask = mfcc_mask.unsqueeze(-1)  # (B, T, 1)
            encoded = encoded * mask
            lengths = mask.sum(dim=1)  # (B, 1)
            x = encoded.sum(dim=1) / lengths.clamp(min=1e-6)
        else:
            x = encoded.mean(dim=1)              # mean over time
        
        out = self.classifier(x)
        return out


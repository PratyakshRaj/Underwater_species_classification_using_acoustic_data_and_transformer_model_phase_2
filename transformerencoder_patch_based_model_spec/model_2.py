import torch
import torch.nn as nn

class ConvPatchAudioTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        patch_freq=16,
        patch_time=16,
        stride_time=14,
        stride_freq=14,
        max_patches=4500
    ):
        super().__init__()

        self.patch_freq = patch_freq
        self.patch_time = patch_time
        self.stride_freq= stride_freq
        self.stride_time = stride_time
        
        # Conv2D patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(patch_freq, patch_time),
            stride=(stride_freq, stride_time)
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_patches, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        
        """
        x: (B, 1, 128, T)
        lengths: original time lengths
        """

        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)
        # (B, d_model, F_patches, T_patches)

        B, D, Fp, Tp = x.shape

        # Flatten patches
        x = x.flatten(2)          # (B, D, Fp*Tp)
        x = x.transpose(1, 2)     # (B, num_patches, D)

        num_patches = x.shape[1]

        
        time_patches = ((lengths - self.patch_time) // self.stride_time) + 1
        time_patches = torch.clamp(time_patches, min=0)
        
        total_freq_patches = Fp   # already computed from conv
        valid_patches = time_patches * total_freq_patches
        
        # Create mask for transformer
        mask = torch.arange(num_patches, device=x.device).expand(B, num_patches)

        valid = valid_patches.unsqueeze(1)

        src_key_padding_mask = mask >= valid
        

        
        # Positional encoding
        x = x + self.pos_embedding[:, :num_patches, :]

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # ---- Masked mean pooling ----
        x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)

        valid_counts = (~src_key_padding_mask).sum(dim=1).unsqueeze(-1)
        x = x.sum(dim=1) / valid_counts.clamp(min=1)

        x = self.norm(x)

        return self.classifier(x)
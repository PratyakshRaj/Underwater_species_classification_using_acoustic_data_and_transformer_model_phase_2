import torch
import torch.nn as nn

class AudioTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        max_len=5000  # maximum expected time length
    ):
        super().__init__()

        self.input_proj = nn.Linear(36, d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, d_model)
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

    def forward(self, x, padding_mask):
        """
        x: (B, T, 36)
        padding_mask: (B, T)  True = ignore
        """

        B, T, _ = x.shape

        x = self.input_proj(x)

        x = x + self.pos_embedding[:, :T, :]

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # Masked mean pooling
        mask = ~padding_mask  # True where valid
        mask = mask.unsqueeze(-1)

        x = x * mask

        summed = x.sum(dim=1)
        lengths = mask.sum(dim=1)

        pooled = summed / lengths.clamp(min=1)

        pooled = self.norm(pooled)

        return self.classifier(pooled)
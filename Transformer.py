# Transformer.py

import math
import torch
import torch.nn as nn

# Function for positional encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return self.dropout(x + self.pe[:, :L, :])


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    m = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)

# Define Transformer model
class TransformerModel(nn.Module):
    """
    Encoder-only Transformer for time series forecasting.

    Input:  (B, L_in=10, F_in=10)  -> 10 years x 10 features
    Output: (B, L_out=5, F_out=1)  -> 5 years targets

    Includes:
      - Embedding (Linear projection of features to d_model)
      - Positional encoding (sinusoidal)
      - Self-attention (TransformerEncoder)
    """
    def __init__(
        self,
        num_features: int = 10,
        out_features: int = 1,
        input_length: int = 10,
        output_length: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_causal_attention: bool = True,
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.out_features = out_features
        self.d_model = d_model
        self.use_causal_attention = use_causal_attention

        # Embedding for continuous inputs: (F -> d_model)
        self.embedding = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encodings
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=10_000, dropout=dropout)

        # Self-attention encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Forecast head uses last token to predict all future steps (direct multi-step)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_length * out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, L, F). Got {tuple(x.shape)}")
        B, L, _ = x.shape
        if L != self.input_length:
            raise ValueError(f"Expected input_length={self.input_length}, got L={L}")

        z = self.embedding(x) * math.sqrt(self.d_model)
        z = self.pos_encoding(z)

        attn_mask = causal_mask(L, z.device) if self.use_causal_attention else None
        h = self.encoder(z, mask=attn_mask)

        y = self.head(h[:, -1, :])
        y = y.view(B, self.output_length, self.out_features)  # (B, L_out, F_out)
        if self.out_features == 1:
            return y.squeeze(-1)  # (B, L_out) matches y tensors
        return y
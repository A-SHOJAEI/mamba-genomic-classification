"""Transformer baseline classifier for genomic sequences."""

import math
import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier for DNA sequences.

    Architecture:
        Embedding + PosEmbed → TransformerEncoder × n_layers → CLS token → Dropout → Linear
    """

    def __init__(self, vocab_size=6, num_classes=2, d_model=256, n_heads=8,
                 n_layers=6, d_ff=512, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len + 1, d_model)  # +1 for CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        """
        Args:
            input_ids: [B, L]
        Returns:
            logits: [B, num_classes]
        """
        B, L = input_ids.shape

        # Token embeddings
        x = self.embedding(input_ids)  # [B, L, d]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, L+1, d]

        # Add positional embeddings
        positions = torch.arange(L + 1, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Padding mask (True = ignore)
        pad_mask = torch.cat([
            torch.zeros(B, 1, device=input_ids.device, dtype=torch.bool),
            input_ids == 0,
        ], dim=1)

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Use CLS token output
        cls_out = x[:, 0]  # [B, d]
        cls_out = self.dropout(cls_out)
        return self.head(cls_out)

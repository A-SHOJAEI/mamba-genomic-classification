"""Mamba SSM classifier for genomic sequences."""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection and LayerNorm."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MambaClassifier(nn.Module):
    """Mamba-based genomic sequence classifier.

    Architecture:
        Embedding → [MambaBlock × n_layer] → GlobalAvgPool → Dropout → Linear
    """

    def __init__(self, vocab_size=6, num_classes=2, d_model=256, n_layer=12,
                 d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        """
        Args:
            input_ids: [B, L] integer token IDs
        Returns:
            logits: [B, num_classes]
        """
        x = self.embedding(input_ids)  # [B, L, d_model]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global average pooling (mask out padding)
        mask = (input_ids != 0).unsqueeze(-1).float()  # [B, L, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, d_model]

        x = self.dropout(x)
        return self.head(x)

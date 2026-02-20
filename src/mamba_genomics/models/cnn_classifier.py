"""1D-CNN baseline classifier for genomic sequences."""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """Multi-layer 1D-CNN classifier for DNA sequences.

    Architecture:
        Embedding → [Conv1d + BatchNorm + ReLU + MaxPool] × n_layers → GAP → Dropout → Linear
    """

    def __init__(self, vocab_size=6, num_classes=2,
                 channels=None, kernel_sizes=None, dropout=0.1):
        super().__init__()
        channels = channels or [256, 256, 256, 256]
        kernel_sizes = kernel_sizes or [15, 9, 5, 3]

        self.embedding = nn.Embedding(vocab_size, channels[0], padding_idx=0)

        layers = []
        in_ch = channels[0]
        for ch, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ])
            in_ch = ch

        self.conv_layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, input_ids):
        """
        Args:
            input_ids: [B, L]
        Returns:
            logits: [B, num_classes]
        """
        x = self.embedding(input_ids)  # [B, L, C]
        x = x.permute(0, 2, 1)         # [B, C, L]
        x = self.conv_layers(x)         # [B, C, L']

        # Global average pooling
        x = x.mean(dim=-1)             # [B, C]
        x = self.dropout(x)
        return self.head(x)

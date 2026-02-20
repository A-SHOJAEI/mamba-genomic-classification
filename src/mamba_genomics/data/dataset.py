"""Genomic sequence dataset from Nucleotide Transformer downstream tasks."""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)

# DNA nucleotide vocabulary
VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}  # 0 = PAD

# Cache for the full dataset to avoid re-downloading for each task
_DATASET_CACHE = {}


def encode_sequence(seq, max_len):
    """Encode a DNA sequence to integer tokens."""
    tokens = [VOCAB.get(c, 5) for c in seq.upper()[:max_len]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return tokens


def _get_full_dataset(dataset_name, split):
    """Load and cache the full dataset."""
    key = (dataset_name, split)
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = load_dataset(dataset_name, split=split)
    return _DATASET_CACHE[key]


class GenomicDataset(Dataset):
    """Wrapper for a single Nucleotide Transformer downstream task."""

    def __init__(self, task_name, split, max_seq_len=1024, dataset_name=None):
        dataset_name = dataset_name or "InstaDeepAI/nucleotide_transformer_downstream_tasks"
        self.max_seq_len = max_seq_len
        self.task_name = task_name

        # Load full dataset and filter by task
        full_ds = _get_full_dataset(dataset_name, split)
        task_ds = full_ds.filter(lambda x: x["task"] == task_name)

        # Pre-encode all sequences for speed
        self.tokens = []
        self.labels = []
        for item in task_ds:
            seq = item["sequence"]
            label = item["label"]
            self.tokens.append(encode_sequence(seq, max_seq_len))
            self.labels.append(label)

        self.tokens = np.array(self.tokens, dtype=np.int64)
        self.labels = np.array(self.labels, dtype=np.int64)

        # Number of classes
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.tokens[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_task(task_name, max_seq_len=1024, dataset_name=None):
    """Load train and test splits for a task."""
    train_ds = GenomicDataset(task_name, "train", max_seq_len, dataset_name)
    test_ds = GenomicDataset(task_name, "test", max_seq_len, dataset_name)
    return train_ds, test_ds, train_ds.num_classes

"""Configuration utilities."""

import random
import yaml
import numpy as np
import torch


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

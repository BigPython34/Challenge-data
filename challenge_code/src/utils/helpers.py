# Utility functions (plots, seed, etc.)
import numpy as np
import random


from ..config import SEED


def set_seed(seed=None):
    """Set seed for reproducibility"""
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed set to: {seed}")

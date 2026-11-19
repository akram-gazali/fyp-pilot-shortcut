"""Seeding helpers. Every random choice goes through an explicit seed."""
import random


def make_rng(seed, *keys):
    """Independent, reproducible RNG for (seed, key1, key2, ...), e.g. one per scene."""
    return random.Random("-".join(str(x) for x in (seed, *keys)))


def set_global_seed(seed):
    """Seed python, numpy and (if installed) torch. Use in scripts, not in library code."""
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

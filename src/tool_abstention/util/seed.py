"""Global pseudo-random number generator seeding."""

import random

import numpy as np

MAX_SEED = 2**32 - 1


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy's global pseudo-random generators."""
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")
    random.seed(seed)
    np.random.seed(seed)

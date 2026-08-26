"""Tests for deterministic global seeding."""

import random

import numpy as np
import pytest

from tool_abstention.util.seed import MAX_SEED, seed_everything


def test_seed_repeats_python_and_numpy_sequences() -> None:
    seed_everything(1234)
    first_python = [random.random() for _ in range(3)]
    first_numpy = np.random.random(3)

    seed_everything(1234)
    assert [random.random() for _ in range(3)] == first_python
    np.testing.assert_array_equal(np.random.random(3), first_numpy)


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1])
def test_invalid_seed_is_rejected(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be between"):
        seed_everything(seed)

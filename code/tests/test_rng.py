from __future__ import annotations

import numpy as np
from p3fcl.rng import seeded


def test_deterministic_for_same_name_and_seed():
    a = seeded("foo", 1).standard_normal(10)
    b = seeded("foo", 1).standard_normal(10)
    np.testing.assert_array_equal(a, b)


def test_different_name_gives_different_stream():
    a = seeded("foo", 1).standard_normal(10)
    b = seeded("bar", 1).standard_normal(10)
    assert not np.array_equal(a, b)


def test_different_seed_gives_different_stream():
    a = seeded("foo", 1).standard_normal(10)
    b = seeded("foo", 2).standard_normal(10)
    assert not np.array_equal(a, b)


def test_returns_a_generator_instance():
    r = seeded("foo", 0)
    assert isinstance(r, np.random.Generator)

"""Call-site-seeded generators. No bare `np.random.*` anywhere in the package (build/00_BUILD_PLAN.md
P0 task 8) — every random draw goes through `seeded(name, seed)`, so adding a new draw somewhere never
shifts every downstream stream.
"""
from __future__ import annotations

import hashlib

import numpy as np


def seeded(name: str, seed: int) -> np.random.Generator:
    """Hash the call-site `name` together with `seed` into a fresh, independent 32-bit seed."""
    blob = f"{name}::{int(seed)}".encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    seed32 = int.from_bytes(digest[:4], "big")
    return np.random.default_rng(seed32)

"""R2/R3 helpers: data-independent projections and the paper's M9 selection rule
(`scripts/run_m9_hparam_selection.py`: strict `>` within p over λ ascending, then strict `>` across p
ascending, which is the first strict maximum in (p, λ) order)."""
from __future__ import annotations

import numpy as np

from .. import rng as rng_mod


def random_orthonormal(d: int, p: int, seed: int = 0) -> np.ndarray:
    """`d x p` orthonormal basis: QR of a seeded standard Gaussian matrix. Uses no data."""
    G = rng_mod.seeded(f"rb_r2::random_proj::{p}", seed).standard_normal((d, p))
    Q, _ = np.linalg.qr(G)
    return Q


def select_first_max(grid: list) -> dict:
    """`grid`: dicts with `p`, `lambda`, `acc`, in the paper's iteration order (p ascending, then λ
    ascending). Returns the paper's winner."""
    best = None
    for g in grid:
        if best is None or g["acc"] > best["acc"]:
            best = g
    return best

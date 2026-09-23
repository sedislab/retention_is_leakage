"""Every method subclasses `FCLMethod`, emits `ArtifactRecord`s with honest `touched`, and declares
`cacheable`. Nothing outside the ledger may read a method's internal state (CLAUDE.md non-negotiable
#1) — `predict()` is the only other window into the method, and it exposes labels, not internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    name: str
    families: tuple
    cacheable: bool
    retention_type: str  # "semantic" | "individual" — FIG04's split (04_METHODS_AND_ATTACKS.md)
    retention_knob_name: str
    # FX4e (08_FIX_PLAN.md): the paper-facing name, which may differ from the internal `name` (kept
    # stable so paths/configs never change) -- e.g. M2 is "Gaussian feature replay (TARGET-style)",
    # not TARGET itself. Falls back to `name` when a method doesn't set one.
    display_name: str = ""

    def __post_init__(self):
        if self.retention_type not in ("semantic", "individual"):
            raise ValueError(f"retention_type must be 'semantic' or 'individual', got {self.retention_type!r}")
        if not self.display_name:
            object.__setattr__(self, "display_name", self.name)


class FCLMethod(ABC):
    """`spec` is a class attribute set by every subclass."""

    spec: MethodSpec

    def __init__(self, config: dict):
        self.config = config
        # FX4a (08_FIX_PLAN.md R1): a logit-based `predict()` over a zero-initialised weight matrix
        # scores not-yet-seen classes at 0, which can beat a legitimately negative logit for a class
        # actually seen so far -- letting the model "predict" a class it has never been trained on
        # and inflating test accuracy on unseen-class columns. Every method must restrict `predict()`
        # to classes recorded here via `update_classes_seen`.
        self._classes_seen: set = set()

    def update_classes_seen(self, y_batch) -> None:
        self._classes_seen.update(int(c) for c in np.unique(np.asarray(y_batch)))

    def mask_unseen_logits(self, logits: np.ndarray) -> np.ndarray:
        """`logits`: `(n_samples, n_classes)`. Returns a copy with every not-yet-seen class column
        set to -inf, so `argmax` can never select it. Raises if nothing has been seen yet (a
        `predict()` call before the first `fit_task` is a caller bug, not a class to predict)."""
        if not self._classes_seen:
            raise RuntimeError("predict() called before any class has been seen (fit_task not yet run)")
        masked = np.full_like(logits, -np.inf)
        seen = sorted(self._classes_seen)
        masked[:, seen] = logits[:, seen]
        return masked

    @abstractmethod
    def rounds_per_task(self) -> int:
        """Communication rounds within one task epoch."""

    @abstractmethod
    def fit_task(self, task_idx: int, X, y, ids, client_shards, rng) -> list:
        """Run one task's local updates + releases. Returns the `list[ArtifactRecord]` emitted this
        task — nothing else about this call is visible to the rest of the system."""

    @abstractmethod
    def predict(self, X):
        """Predict labels for `X` using the method's current (post-aggregation) global state."""

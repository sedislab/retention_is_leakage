"""Every method subclasses `FCLMethod`, emits `ArtifactRecord`s with honest `touched`, and declares
`cacheable`. Nothing outside the ledger may read a method's internal state (CLAUDE.md non-negotiable
#1) — `predict()` is the only other window into the method, and it exposes labels, not internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    families: tuple
    cacheable: bool
    retention_type: str  # "semantic" | "individual" — FIG04's split (04_METHODS_AND_ATTACKS.md)
    retention_knob_name: str

    def __post_init__(self):
        if self.retention_type not in ("semantic", "individual"):
            raise ValueError(f"retention_type must be 'semantic' or 'individual', got {self.retention_type!r}")


class FCLMethod(ABC):
    """`spec` is a class attribute set by every subclass."""

    spec: MethodSpec

    def __init__(self, config: dict):
        self.config = config

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

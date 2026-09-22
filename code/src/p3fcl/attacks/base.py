"""An attack whose threat-model cell is unspecified does not run — that guard exists for a reason
(04_METHODS_AND_ATTACKS.md §2), keep it.
"""
from __future__ import annotations

from dataclasses import dataclass

_VALID_WHO = {"hbc_server", "participating_client", "external"}
_VALID_VIEW = {"full", "task", "final"}
_VALID_AUX = {"none", "public_data", "shadow_training"}
_VALID_TARGET = {"membership", "property", "reconstruction", "onset", "attribution"}


@dataclass(frozen=True)
class ThreatModel:
    who: str
    active: bool
    view: str
    auxiliary: str
    target: str
    survives_secure_agg: bool

    def __post_init__(self):
        if self.who not in _VALID_WHO:
            raise ValueError(f"ThreatModel.who must be one of {_VALID_WHO}, got {self.who!r}")
        if self.view not in _VALID_VIEW:
            raise ValueError(f"ThreatModel.view must be one of {_VALID_VIEW}, got {self.view!r}")
        if self.auxiliary not in _VALID_AUX:
            raise ValueError(f"ThreatModel.auxiliary must be one of {_VALID_AUX}, got {self.auxiliary!r}")
        if self.target not in _VALID_TARGET:
            raise ValueError(f"ThreatModel.target must be one of {_VALID_TARGET}, got {self.target!r}")


class UnspecifiedThreatModelError(RuntimeError):
    pass


class Attack:
    """Subclasses set the class attribute `threat_model`. An attack that does not declare one refuses
    to instantiate."""

    threat_model: ThreatModel | None = None

    def __init__(self, config: dict):
        self.config = config
        if self.threat_model is None:
            raise UnspecifiedThreatModelError(
                f"{type(self).__name__} does not declare a ThreatModel; refusing to run"
            )

    def run(self, ledger, **kwargs):
        raise NotImplementedError

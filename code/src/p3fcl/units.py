"""Units of privacy in FCL (RESEARCH_PLAN.md §2.3). Every DP claim must name its unit; a function
computing an epsilon without a `Unit` argument is a bug.
"""
from __future__ import annotations

from enum import Enum


class Unit(str, Enum):
    EXAMPLE = "U1"
    TASK = "U2"
    CLIENT_BOUNDED = "U3"
    CLIENT_LIFELONG = "U4"
    INDIVIDUAL = "U5"


_NEIGHBOURING = {
    Unit.EXAMPLE: (
        "Example-level (U1): D and D' differ in one example of one client at one task. The weakest "
        "unit — what most DP-SGD-in-FL papers silently use."
    ),
    Unit.TASK: (
        "Task-level / client-task (U2): D and D' differ in the entirety of one client's data at one "
        "task-epoch. Natural for FCL, and the unit under which parallel composition can be legal."
    ),
    Unit.CLIENT_BOUNDED: (
        "Client-level, bounded window (U3): D and D' differ in all of one client's data within some "
        "fixed-width rolling window of `window` (default 3) consecutive tasks -- worst case over "
        "every window start, not the whole horizon. The standard cross-silo target."
    ),
    Unit.CLIENT_LIFELONG: (
        "Client-level, unbounded / lifelong (U4): D and D' differ in all of one client's data across "
        "the unbounded task stream. What the FCL premise implicitly promises; no accountant delivers "
        "this non-vacuously for a client who keeps contributing new information forever (H1)."
    ),
    Unit.INDIVIDUAL: (
        "Individual / person-level under renewal (U5): D and D' differ in the data of one person, "
        "who may recur across several tasks of one client and possibly across clients, under a "
        "renewal (churn) model of the population. Under this project's streams, each person "
        "contributes exactly one example, so U5 == U1 exactly (`dp.accountant.account` routes both "
        "to the same U1 computation; report U1 only, do not plot U5 separately)."
    ),
}


def neighbouring(unit) -> str:
    """Short prose description used in captions and TAB02, so the paper's definitions and the code's
    definitions cannot drift apart."""
    unit = unit if isinstance(unit, Unit) else Unit(unit)
    return _NEIGHBOURING[unit]

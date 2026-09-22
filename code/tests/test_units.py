from __future__ import annotations

import pytest
from p3fcl.units import Unit, neighbouring


def test_all_units_defined():
    assert {u.value for u in Unit} == {"U1", "U2", "U3", "U4", "U5"}


@pytest.mark.parametrize("unit", list(Unit))
def test_neighbouring_returns_nonempty_string_for_every_unit(unit):
    desc = neighbouring(unit)
    assert isinstance(desc, str) and len(desc) > 10


def test_neighbouring_accepts_string_value():
    assert neighbouring("U2") == neighbouring(Unit.TASK)

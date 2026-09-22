"""No bare `np.random.*` anywhere in the package (build/00_BUILD_PLAN.md P0 task 8) — every random
draw must go through `rng.seeded(name, seed)`, so adding a new draw somewhere never shifts every
downstream stream.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "p3fcl"

# matches np.random.<anything>(...) but not `np.random.Generator` used only as a type/annotation,
# and not `np.random.default_rng` inside rng.py itself (the one legitimate call site).
_BARE_CALL = re.compile(r"\bnp\.random\.(?!Generator\b)\w+\s*\(")


def test_no_bare_np_random_calls_outside_rng_py():
    offenders = []
    for path in SRC_ROOT.rglob("*.py"):
        if path.name == "rng.py":
            continue
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _BARE_CALL.search(line):
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{lineno}: {stripped}")
    assert not offenders, "bare np.random.* calls found (use rng.seeded instead):\n" + "\n".join(offenders)


def test_rng_py_is_the_only_legitimate_call_site():
    rng_py = (SRC_ROOT / "rng.py").read_text()
    assert "np.random.default_rng" in rng_py

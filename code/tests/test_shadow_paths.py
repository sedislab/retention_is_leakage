"""Fail on shadow-root path literals outside the single registry (including PBS)."""

import io
import re
import tokenize
from pathlib import Path

from p3fcl.paths import SHADOW_ROOT, shadow_dir


def test_no_other_hard_coded_shadow_roots():
    root = Path(__file__).resolve().parents[1]
    forbidden = re.compile(r"(?:[\x22\x27/])shadows(?:_v\d+)?(?:[\x22\x27/])")
    offenders = []
    for p in root.rglob("*"):
        if p.suffix not in {".py", ".pbs", ".sh", ".yaml"} or p.name == "paths.py":
            continue
        text = p.read_text()
        if p.suffix == ".py":
            # Ignore documentation/comments, inspect actual string path literals.
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if (
                    tok.type == tokenize.STRING
                    and not tok.string.startswith(('"""', "'''"))
                    and forbidden.search(tok.string)
                    and ("/" in tok.line or "Path(" in tok.line)
                ):
                    offenders.append(f"{p.relative_to(root)}:{tok.start[0]}")
        else:
            for n, line in enumerate(text.splitlines(), 1):
                if not line.lstrip().startswith("#") and forbidden.search(line):
                    offenders.append(f"{p.relative_to(root)}:{n}")
    assert not offenders, offenders


def test_registry_and_no_existence_fallback(tmp_path):
    for m in ["m1_glfc", "m2_target", "m4_proto", "m5_hybrid_replay"]:
        assert SHADOW_ROOT[m].name.endswith("v3")
        assert shadow_dir("cifar100", m, 0, tmp_path) == tmp_path / SHADOW_ROOT[m] / "cifar100" / m / "seed0"
    assert (
        shadow_dir("cifar100", "m0_fedavg", 0, tmp_path)
        == tmp_path / SHADOW_ROOT["m0_fedavg"] / "cifar100" / "m0_fedavg"
    )

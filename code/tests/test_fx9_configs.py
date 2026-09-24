import csv
import json
from pathlib import Path

import pytest
from p3fcl.experiment import METHODS, attacked_method_config
from p3fcl.shadow_runner import _method_config


@pytest.mark.parametrize("method", METHODS)
def test_utility_matches_shadow_config_plus_only_fx9_gate(tmp_path, method):
    path = tmp_path / "gate.csv"
    rows = [
        dict(
            dataset="cifar100",
            method=m,
            used_cfg_json=json.dumps({"lr": 0.1, "replay_weight": 2} if m == "m1_glfc" else {}),
        )
        for m in METHODS
    ]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    config = attacked_method_config(method, 100, 768, "cifar100", path)
    expected = _method_config(method, 100, 768)
    if method == "m1_glfc":
        expected.update(lr=0.1, replay_weight=2)
    assert config == expected
    # Both entry points call this very function; no duplicated table or standardization.
    root = Path(__file__).resolve().parents[1]
    for filename in ["run_accuracy_matrix.py", "run_utility_baseline.py"]:
        text = (root / "scripts" / filename).read_text()
        assert "_standardize" not in text
        assert "attacked_method_config(" in text

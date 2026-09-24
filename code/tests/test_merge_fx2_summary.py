import csv
import importlib.util
from pathlib import Path

import pytest


def module(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/merge_fx2_summary.py"
    spec = importlib.util.spec_from_file_location("merge", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.REPO_ROOT = tmp_path
    (tmp_path / "results").mkdir()
    return m


def write(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_never_folds_in_existing_shared_rows(tmp_path):
    m = module(tmp_path)
    shared = tmp_path / "results/merged.csv"
    write(shared, [dict(method="stale", value=999)])
    write(tmp_path / "results/combo_a.csv", [dict(method="a", value=1)])
    m._merge("merged.csv", "combo_*.csv", ["method"])
    with shared.open() as f:
        assert list(csv.DictReader(f)) == [dict(method="a", value="1")]
    assert len(list((tmp_path / "archive/fx9_shared").glob("*/merged.csv"))) == 1


def test_missing_or_duplicate_per_combo_inputs_fail(tmp_path):
    m = module(tmp_path)
    with pytest.raises(ValueError, match="no per-combo"):
        m._merge("merged.csv", "combo_*.csv", ["method"])
    for name in ["a", "b"]:
        write(tmp_path / f"results/combo_{name}.csv", [dict(method="a", value=1)])
    with pytest.raises(ValueError, match="duplicate"):
        m._merge("merged.csv", "combo_*.csv", ["method"])

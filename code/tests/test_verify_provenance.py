from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "analysis" / "verify_provenance.py"
    spec = importlib.util.spec_from_file_location("verify_provenance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_meta(path: Path, utc_start: str, pbs_jobid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"utc_start": utc_start, "pbs_jobid": pbs_jobid}))


def test_check_pbs_jobid_flags_post_phase_start_null_jobid(tmp_path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(m, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    _write_meta(tmp_path / "results" / "bad.csv.meta.json", "2026-09-23T10:00:00+00:00", None)

    offenders = m._check_pbs_jobid()
    assert len(offenders) == 1
    assert offenders[0][0].name == "bad.csv.meta.json"


def test_check_pbs_jobid_ignores_pre_phase_start_null_jobid(tmp_path, monkeypatch):
    """Historical, pre-fix-phase runs are grandfathered in -- the check is about the ACTIVE phase,
    not "ever," per 08_FIX_PLAN.md's own literal wording."""
    m = _load_module()
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(m, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    _write_meta(tmp_path / "results" / "old.csv.meta.json", "2026-09-16T10:00:00+00:00", None)

    offenders = m._check_pbs_jobid()
    assert offenders == []


def test_check_pbs_jobid_ignores_real_jobid(tmp_path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(m, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    _write_meta(tmp_path / "results" / "good.csv.meta.json", "2026-09-23T10:00:00+00:00", "184123.bcm11")

    offenders = m._check_pbs_jobid()
    assert offenders == []


def test_check_pbs_jobid_scans_tables_dir_too(tmp_path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(m, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    _write_meta(tmp_path / "tables" / "tab99.csv.meta.json", "2026-09-23T10:00:00+00:00", None)

    offenders = m._check_pbs_jobid()
    assert len(offenders) == 1


def test_check_pbs_jobid_tolerates_missing_or_malformed_meta(tmp_path, monkeypatch):
    """A file with no `utc_start`, or malformed JSON, must not crash the check -- it just can't be
    evaluated (a separate, larger gap this check deliberately does not try to catch, per its own
    docstring)."""
    m = _load_module()
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(m, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "results" / "no_utc.csv.meta.json").write_text(json.dumps({"pbs_jobid": None}))
    (tmp_path / "results" / "broken.csv.meta.json").write_text("{not valid json")

    offenders = m._check_pbs_jobid()
    assert offenders == []


def test_main_fails_when_a_post_phase_start_null_jobid_exists(tmp_path, monkeypatch):
    """End to end through `main()`: an empty paper/ tree (so the traceability check is a trivial
    pass) plus one offending meta.json must still exit non-zero."""
    m = _load_module()
    monkeypatch.setattr(m, "PAPER_DIR", tmp_path / "paper")  # does not exist -> traceability check no-ops
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(m, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    _write_meta(tmp_path / "results" / "bad.csv.meta.json", "2026-09-23T10:00:00+00:00", None)

    assert m.main() == 1


def test_main_passes_when_everything_is_clean(tmp_path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(m, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    _write_meta(tmp_path / "results" / "good.csv.meta.json", "2026-09-23T10:00:00+00:00", "184123.bcm11")

    assert m.main() == 0

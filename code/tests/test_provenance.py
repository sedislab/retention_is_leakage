from __future__ import annotations

import json

from p3fcl import provenance


def test_run_manifest_has_required_fields():
    m = provenance.run_manifest({"seed": 0}, seed=0)
    for key in ("source_hash", "config_hash", "seed", "hostname", "python", "numpy", "utc_start"):
        assert key in m


def test_source_hash_is_stable_across_calls():
    assert provenance.source_hash() == provenance.source_hash()


def test_source_hash_changes_when_a_source_file_changes(tmp_path, monkeypatch):
    src_dir = tmp_path / "code" / "src"
    configs_dir = tmp_path / "code" / "configs"
    src_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)
    f = src_dir / "example.py"
    f.write_text("x = 1\n")
    monkeypatch.setattr(provenance, "CODE_SRC_DIR", src_dir)
    monkeypatch.setattr(provenance, "CODE_CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)

    h1 = provenance.source_hash(force=True)
    f.write_text("x = 2\n")
    h2 = provenance.source_hash(force=True)
    assert h1 != h2


def test_finalize_writes_meta_json_and_run_log(tmp_path):
    manifest = provenance.run_manifest({"seed": 0}, seed=0)
    out = tmp_path / "result.csv"
    out.write_text("a,b\n1,2\n")
    results_dir = tmp_path / "results"
    provenance.finalize(manifest, [out], results_dir=results_dir)

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["config_hash"] == manifest["config_hash"]
    assert meta["source_hash"] == manifest["source_hash"]
    assert "config" not in meta  # config lives in results/configs/<hash>.yaml, not duplicated

    run_log = results_dir / "RUN_LOG.jsonl"
    assert run_log.exists()
    lines = run_log.read_text().strip().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["config_hash"] == manifest["config_hash"]
    assert logged["source_hash"] == manifest["source_hash"]
    assert str(out) in logged["outputs"]

    cfg_path = results_dir / "configs" / f"{manifest['config_hash']}.yaml"
    assert cfg_path.exists()


def test_logged_run_decorator_finalizes(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "RESULTS_DIR", tmp_path / "results")

    @provenance.logged_run
    def fake_job(*, config, seed, manifest):
        out = tmp_path / "out.csv"
        out.write_text("x\n1\n")
        return {"outputs": [out]}

    fake_job(config={"seed": 1}, seed=1)
    assert (tmp_path / "results" / "RUN_LOG.jsonl").exists()

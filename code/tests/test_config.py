from __future__ import annotations

import pytest
import yaml
from p3fcl.config import ConfigError, FrozenDict, apply_overrides, config_hash, load, validate


def test_apply_overrides_dotted_path():
    base = {"stream": {"n_tasks": 10}, "seed": 0}
    out = apply_overrides(base, ["stream.n_tasks=20"])
    assert out["stream"]["n_tasks"] == 20
    assert base["stream"]["n_tasks"] == 10  # base untouched


def test_apply_overrides_parses_yaml_scalars():
    out = apply_overrides({}, ["a.b=3", "a.c=true", "a.d=hello"])
    assert out == {"a": {"b": 3, "c": True, "d": "hello"}}


def test_apply_overrides_rejects_bad_syntax():
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no-equals-sign"])


def test_apply_overrides_rejects_non_mapping_intermediate():
    with pytest.raises(ConfigError):
        apply_overrides({"a": 1}, ["a.b=2"])


def test_config_hash_stable_and_order_independent():
    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})
    assert h1 == h2
    assert len(h1) == 16


def test_frozen_dict_rejects_mutation():
    fd = FrozenDict({"a": 1})
    with pytest.raises(TypeError):
        fd["a"] = 2


def test_validate_requires_int_seed():
    with pytest.raises(ConfigError):
        validate({})
    with pytest.raises(ConfigError):
        validate({"seed": "not-an-int"})
    validate({"seed": 0})  # does not raise


def test_load_merges_overrides_and_freezes(tmp_path):
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text(yaml.safe_dump({"seed": 0, "stream": {"n_tasks": 10}}))
    frozen, h = load(cfg_path, overrides=["stream.n_tasks=5"])
    assert frozen["stream"]["n_tasks"] == 5
    assert isinstance(h, str) and len(h) == 16
    with pytest.raises(TypeError):
        frozen["seed"] = 1

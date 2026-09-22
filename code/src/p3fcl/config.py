"""YAML load + deep-merge overrides + schema validation + hashing.

Every config that ever produced a result is copied into `results/configs/<hash>.yaml`
(`provenance.finalize` does the copying; `config_hash` here is the single source of the hash so
`config.py` and `provenance.py` never compute it two different ways).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_path(d: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if nxt is None:
            nxt = {}
            cur[k] = nxt
        if not isinstance(nxt, dict):
            raise ConfigError(f"cannot set {dotted_key!r}: {k!r} is not a mapping")
        cur = nxt
    cur[keys[-1]] = value


def _parse_scalar(s: str):
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def apply_overrides(config: dict, overrides) -> dict:
    """`overrides` like `['stream.n_tasks=20', 'seed=3']` (the CLI's `--set key.path=value`)."""
    out = copy.deepcopy(config)
    for ov in overrides:
        if "=" not in ov:
            raise ConfigError(f"bad override (expected key.path=value): {ov!r}")
        key, _, val = ov.partition("=")
        _set_path(out, key.strip(), _parse_scalar(val.strip()))
    return out


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class FrozenDict(dict):
    """A dict that raises on mutation. Hashable (via a sorted-JSON digest) so it can be passed around
    as a value object."""

    def _immutable(self, *_args, **_kwargs):
        raise TypeError("FrozenDict is read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable

    def __hash__(self):  # type: ignore[override]
        return hash(json.dumps(self, sort_keys=True, default=str))


def _freeze(obj):
    if isinstance(obj, dict):
        return FrozenDict({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def validate(config: dict) -> None:
    """Minimal universal schema. Per-config-family required keys are validated by the consumer
    (a method needs `n_classes`/`feature_dim`, a stream config needs `n_tasks`, etc.) — this function
    only enforces what every config must have."""
    if "seed" not in config:
        raise ConfigError("config must define 'seed'")
    if not isinstance(config["seed"], int):
        raise ConfigError("'seed' must be an int")


def load(path, overrides=None) -> tuple:
    """Returns `(frozen_config, hash)`."""
    path = Path(path)
    with open(path) as f:
        base = yaml.safe_load(f) or {}
    merged = apply_overrides(base, overrides or [])
    validate(merged)
    h = config_hash(merged)
    return _freeze(merged), h

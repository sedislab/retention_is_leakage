"""Shared raw-feature experiment configuration and FX9 gate overrides."""

import csv
import json
from pathlib import Path

from .shadow_runner import _method_config

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ["cifar100", "cub200", "imagenet_r"]
METHODS = ["m0_fedavg", "m1_glfc", "m2_target", "m3_fot", "m4_proto", "m5_hybrid_replay", "m8_analytic"]
REPLAY_METHODS = {"m1_glfc", "m2_target", "m5_hybrid_replay"}
CHANGED_METHODS = REPLAY_METHODS | {"m4_proto"}
FX9_START = "2026-09-24T00:04:36+00:00"


def gate_overrides(dataset, gate_csv=None):
    path = Path(gate_csv) if gate_csv is not None else ROOT / "results/fx9_gate.csv"
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == dataset]
    out = {}
    for r in rows:
        cfg = json.loads(r["used_cfg_json"])
        if r["method"] in out and out[r["method"]] != cfg:
            raise ValueError("inconsistent gate config across seeds")
        out[r["method"]] = cfg
    if not REPLAY_METHODS <= out.keys():
        raise ValueError(f"incomplete FX9 gate for {dataset}")
    return out


def attacked_method_config(method, n_classes, feature_dim, dataset, gate_csv=None):
    cfg = _method_config(method, n_classes, feature_dim)
    overrides = gate_overrides(dataset, gate_csv).get(method, {})
    return {**cfg, **overrides}

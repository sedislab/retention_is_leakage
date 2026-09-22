"""ref ∩ train = ∅ for every dataset (02_DATASETS.md §7): if the adversary's reference split
contains target records, every attack number downstream is inflated. This is a claim test in spirit,
but it has no real dataset to check until Phase P1 writes `data/MANIFEST.json` — it skips cleanly
until then rather than being silently absent, and activates automatically once the manifest exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="data/MANIFEST.json does not exist yet (Phase P1)")
def test_ref_disjoint_from_train_for_every_dataset():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for dataset, entry in manifest.items():
        ref_ids = set(entry.get("ref_ids", []))
        train_ids = set(entry.get("train_ids", []))
        assert ref_ids.isdisjoint(train_ids), f"{dataset}: ref ∩ train is non-empty"

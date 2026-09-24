"""Tests must never append to the production run log on the login node."""
import pytest
from p3fcl import provenance


@pytest.fixture(autouse=True)
def isolate_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, 'RESULTS_DIR', tmp_path/'provenance')

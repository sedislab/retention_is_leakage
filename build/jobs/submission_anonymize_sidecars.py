#!/usr/bin/env python3
"""One-off double-blind cleanup of provenance sidecars: neutralise the account name, the absolute
checkout path and the cluster name. Touches only text sidecars (json/jsonl/yaml); result CSVs and
figure/table bodies are never opened for writing."""
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path("/data/islamm/retention_leakage")
REPLACEMENTS = [
    ("/data/islamm/retention_leakage", "<repo>"),
    ('"user": "islamm"', '"user": "anonymous"'),
    ('"hostname": "kodiak"', '"hostname": "login-node"'),
]
TEXT_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml"}

files = subprocess.run(
    ["git", "ls-files", "-z", "--", "results", "figs", "tables"],
    cwd=ROOT, capture_output=True, check=True,
).stdout.decode().split("\0")
changed = 0
for rel in filter(None, files):
    p = ROOT / rel
    if p.suffix not in TEXT_SUFFIXES:
        continue
    text = p.read_text(encoding="utf-8")
    new = text
    for old, rep in REPLACEMENTS:
        new = new.replace(old, rep)
    if new == text:
        continue
    if p.suffix == ".json":
        json.loads(new)
    elif p.suffix == ".jsonl":
        for line in new.splitlines():
            if line.strip():
                json.loads(line)
    else:
        yaml.safe_load(new)
    p.write_text(new, encoding="utf-8")
    changed += 1
print(f"rewrote {changed} sidecar files")
sys.exit(0)

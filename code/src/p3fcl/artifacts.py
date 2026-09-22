"""The artifact ledger — the only interface between methods and attacks (CLAUDE.md non-negotiable #1).

A method declares what it *releases* as `ArtifactRecord`s. Attacks and `dp.accountant` consume the
`Ledger` and nothing else — never reach into a method's internal state from an attack.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


class Family(str, Enum):
    """Artifact families, RESEARCH_PLAN.md §2.2. Values are the paper's F-ids — do not renumber."""

    MODEL_DELTA = "F1"
    PROTOTYPE = "F2"
    PROMPT = "F3"
    LOW_RANK = "F4"
    GRAM = "F5"
    GENERATIVE = "F6"
    COUNTS = "F7"
    EXEMPLAR = "F8"


Payload = "np.ndarray | dict[str, np.ndarray]"


@dataclass
class ArtifactRecord:
    """One release. `touched` is the privacy-critical field (CLAUDE.md non-negotiable #2): the set of
    datum ids whose *values* influenced this release, however indirectly (a replay buffer, a
    distillation term, a momentum-blended statistic, a regulariser). Over-report rather than
    under-report. If unsure whether a datum belongs, include it.
    """

    round: int
    task: int
    client: int
    family: Family
    payload: object  # np.ndarray | dict[str, np.ndarray]
    touched: frozenset
    n_touched: int = 0
    passes_over_data: int = 1
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.family, Family):
            self.family = Family(self.family)
        self.touched = frozenset(int(t) for t in self.touched)
        if self.n_touched == 0 and self.touched:
            self.n_touched = len(self.touched)

    def nbytes(self) -> int:
        if isinstance(self.payload, dict):
            return sum(np.asarray(v).nbytes for v in self.payload.values())
        return int(np.asarray(self.payload).nbytes)


@dataclass
class Ledger:
    """An append-only transcript = the release channel M (RESEARCH_PLAN.md §2.1)."""

    records: list = field(default_factory=list)

    def add(self, rec: ArtifactRecord) -> None:
        self.records.append(rec)

    def extend(self, recs: Iterable[ArtifactRecord]) -> None:
        self.records.extend(recs)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ArtifactRecord]:
        return iter(self.records)

    def view(self, kind: str) -> list:
        """`full` — every record (honest-but-curious server / persistent participant, V_full).
        `task`  — last round of each task (checkpoint-publishing adversary, V_task).
        `final` — last round only (model-release adversary, V_final)."""
        if kind == "full":
            return list(self.records)
        if not self.records:
            return []
        if kind == "final":
            last_round = max(r.round for r in self.records)
            return [r for r in self.records if r.round == last_round]
        if kind == "task":
            last_round_per_task: dict = {}
            for r in self.records:
                last_round_per_task[r.task] = max(last_round_per_task.get(r.task, -1), r.round)
            return [r for r in self.records if r.round == last_round_per_task[r.task]]
        raise ValueError(f"unknown view kind: {kind!r} (expected full|task|final)")

    def by_family(self, fam) -> list:
        fam = fam if isinstance(fam, Family) else Family(fam)
        return [r for r in self.records if r.family == fam]

    def by_client(self, c: int) -> list:
        return [r for r in self.records if r.client == c]

    def tasks(self) -> list:
        return sorted({r.task for r in self.records})

    def summary(self) -> dict:
        by_family: dict = {}
        for r in self.records:
            by_family[r.family.value] = by_family.get(r.family.value, 0) + 1
        return {
            "n_records": len(self.records),
            "n_rounds": len({r.round for r in self.records}),
            "n_tasks": len(self.tasks()),
            "n_clients": len({r.client for r in self.records}),
            "by_family": by_family,
            "total_bytes": sum(r.nbytes() for r in self.records),
        }

    def aggregate_view(self) -> list:
        """Per-round SUM over clients, per (task, family) — the secure-aggregation view. An attack
        that still works when run against only this view survives secure aggregation (H11, FIG11)."""
        groups: dict = {}
        for r in self.records:
            groups.setdefault((r.round, r.task, r.family), []).append(r)
        out = []
        for (rnd, task, fam), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2].value)):
            touched = frozenset().union(*(r.touched for r in recs)) if recs else frozenset()
            n_touched = sum(r.n_touched for r in recs)
            passes = max(r.passes_over_data for r in recs)
            if isinstance(recs[0].payload, dict):
                # Union of keys across every record, not just the first -- a per-class payload
                # (e.g. F2's prototype dict) has different key sets per client whenever a client's
                # Dirichlet share holds zero samples of some class, which is common at low beta /
                # high client count. A record missing a key simply didn't contribute to that key's
                # sum (the real secure-aggregation semantics: absence, not a zero-filled entry).
                keys: set = set()
                for r in recs:
                    keys.update(r.payload.keys())
                payload = {
                    k: sum(np.asarray(r.payload[k]) for r in recs if k in r.payload) for k in sorted(keys)
                }
            else:
                payload = sum(np.asarray(r.payload) for r in recs)
            out.append(
                ArtifactRecord(
                    round=rnd,
                    task=task,
                    client=-1,
                    family=fam,
                    payload=payload,
                    touched=touched,
                    n_touched=n_touched,
                    passes_over_data=passes,
                    meta={"aggregated_over_clients": sorted({r.client for r in recs})},
                )
            )
        return out


def _record_to_header_and_arrays(i: int, rec: ArtifactRecord) -> tuple:
    header = {
        "round": rec.round,
        "task": rec.task,
        "client": rec.client,
        "family": rec.family.value,
        "n_touched": rec.n_touched,
        "passes_over_data": rec.passes_over_data,
        "meta": rec.meta,
        "touched": sorted(int(t) for t in rec.touched),
    }
    arrays = {}
    if isinstance(rec.payload, dict):
        header["payload_kind"] = "dict"
        header["payload_keys"] = list(rec.payload.keys())
        for k, v in rec.payload.items():
            arrays[f"rec{i}__payload__{k}"] = np.asarray(v)
    else:
        header["payload_kind"] = "array"
        arrays[f"rec{i}__payload"] = np.asarray(rec.payload)
    return header, arrays


def save_npz(ledger: Ledger, path) -> None:
    """Round-trips exactly. Atomic: writes to `<path>.tmp` then `os.replace`."""
    path = Path(path)
    headers = []
    arrays: dict = {}
    for i, rec in enumerate(ledger.records):
        h, a = _record_to_header_and_arrays(i, rec)
        headers.append(h)
        arrays.update(a)
    arrays["__headers__"] = np.array(json.dumps(headers))
    arrays["__n_records__"] = np.array(len(headers))
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, path)


def load_npz(path) -> Ledger:
    path = Path(path)
    with open(path, "rb") as f:
        data = np.load(f, allow_pickle=False)
        headers = json.loads(data["__headers__"].item())
        records = []
        for i, h in enumerate(headers):
            if h["payload_kind"] == "dict":
                payload = {k: data[f"rec{i}__payload__{k}"] for k in h["payload_keys"]}
            else:
                payload = data[f"rec{i}__payload"]
            records.append(
                ArtifactRecord(
                    round=h["round"],
                    task=h["task"],
                    client=h["client"],
                    family=Family(h["family"]),
                    payload=payload,
                    touched=frozenset(h["touched"]),
                    n_touched=h["n_touched"],
                    passes_over_data=h["passes_over_data"],
                    meta=h["meta"],
                )
            )
    return Ledger(records=records)

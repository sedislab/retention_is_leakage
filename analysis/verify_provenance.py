#!/usr/bin/env python3
"""`make verify` — the provenance check (CLAUDE.md non-negotiable #3, build/00_BUILD_PLAN.md P0
task 6). Two checks:

1. **Numeric literal traceability.** Parses every `paper/**/*.tex`, extracts numeric literals
   appearing outside math-free prose and captions is not attempted here at the character level
   (that's a much harder LaTeX-parsing problem); instead this walks the raw text for standalone
   numeric tokens and requires each one to be traceable to a `results/*.csv` row or a generated
   `tables/*.tex` file, with an explicit `\\providedvalue{...}` escape for genuine non-experimental
   numbers (page counts, dataset sizes).
2. **`pbs_jobid` check** (`08_FIX_PLAN.md` §11: "fail if any CSV, figure or table produced after the
   start of this phase has `pbs_jobid: null`"). Scans every `.meta.json` under `results/` and
   `tables/` (the two directories `03_RESULTS_SPEC.md`'s layout describes as holding citable
   CSVs/tables; `figs/` never gets its own sidecar in this project's convention, and `runs/` holds
   intermediate per-run caches, not citable results, so both are out of scope here) and fails on any
   whose `utc_start` is at or after `_PHASE_START` (2026-09-22T19:30 UTC, `archive/2026-09-23_pre_fix/
   README.md`'s own recorded archive time, immediately before FX0) but whose `pbs_jobid` is `None` --
   i.e. a real, in-phase result was produced directly on the login node rather than through `qsub`. A
   file with NO `.meta.json` at all is a different, larger, separately-tracked gap
   (`03_RESULTS_SPEC.md` §6 also asks for that check; `build/STATE.md`'s FX7 notes record which
   scripts still lack a sidecar entirely) and is deliberately not what this check flags -- it can only
   evaluate a `pbs_jobid` field that exists.

Exits non-zero (and lists every offending literal or file) if either check fails. On an empty `paper/`
tree the traceability check reports cleanly with zero exceptions, which is what the P0 gate requires;
the `pbs_jobid` check always runs regardless of whether `paper/` exists, since it depends only on
`results/`/`tables/`.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "tables"

# `archive/2026-09-23_pre_fix/README.md`: "Archived (UTC): 2026-09-22 ... immediately before starting
# build/08_FIX_PLAN.md's fix phase (FX0)". Naive (no tzinfo) to compare directly against meta.json's
# `utc_start`, which `provenance.run_manifest` writes via `datetime.now(timezone.utc).isoformat()`
# (tz-aware) -- stripped to naive UTC on both sides in `_check_pbs_jobid` below.
_PHASE_START = datetime.fromisoformat("2026-09-22T19:30:00")

# A standalone number: optional sign, digits, optional decimal part. Deliberately does not match
# numbers that are part of a LaTeX command/macro name or a \label{fig:01} style identifier.
_NUMBER_RE = re.compile(r"(?<![A-Za-z_{])-?\d+\.?\d*(?![A-Za-z_}])")
_PROVIDEDVALUE_RE = re.compile(r"\\providedvalue\{([^}]*)\}")
_COMMENT_RE = re.compile(r"(?<!\\)%.*$")


def _strip_comments(text: str) -> str:
    return "\n".join(_COMMENT_RE.sub("", line) for line in text.splitlines())


def _collect_csv_numbers() -> set:
    numbers = set()
    if not RESULTS_DIR.exists():
        return numbers
    for csv_path in RESULTS_DIR.rglob("*.csv"):
        for line in csv_path.read_text(errors="ignore").splitlines():
            for tok in re.split(r"[,\s]+", line):
                if _NUMBER_RE.fullmatch(tok):
                    numbers.add(tok)
    return numbers


def _collect_table_tex_numbers() -> set:
    numbers = set()
    if not TABLES_DIR.exists():
        return numbers
    for tex_path in TABLES_DIR.rglob("*.tex"):
        text = _strip_comments(tex_path.read_text(errors="ignore"))
        numbers.update(_NUMBER_RE.findall(text))
    return numbers


def _check_pbs_jobid() -> list:
    """Returns a list of `(path, utc_start)` for every post-`_PHASE_START` `.meta.json` under
    `results/`/`tables/` whose `pbs_jobid` is `None`."""
    offenders = []
    for base_dir in (RESULTS_DIR, TABLES_DIR):
        if not base_dir.exists():
            continue
        for meta_path in base_dir.rglob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            utc_start = meta.get("utc_start")
            if not utc_start:
                continue
            try:
                ts = datetime.fromisoformat(utc_start).replace(tzinfo=None)
            except ValueError:
                continue
            if ts < _PHASE_START:
                continue
            if meta.get("pbs_jobid") is None:
                offenders.append((meta_path.relative_to(REPO_ROOT), utc_start))
    return sorted(offenders, key=lambda x: x[1])


def main() -> int:
    failed = False

    # Check 1: numeric literal traceability.
    if not PAPER_DIR.exists():
        print("verify: paper/ does not exist yet — nothing to check for literal traceability. "
              "OK (clean, 0 exceptions).")
    else:
        tex_files = sorted(PAPER_DIR.rglob("*.tex"))
        if not tex_files:
            print("verify: paper/ exists but contains no .tex files. OK (clean, 0 exceptions).")
        else:
            known_numbers = _collect_csv_numbers() | _collect_table_tex_numbers()
            provided = []
            offenders = []

            for tex_path in tex_files:
                text = _strip_comments(tex_path.read_text(errors="ignore"))
                for m in _PROVIDEDVALUE_RE.finditer(text):
                    provided.append((tex_path, m.group(1)))
                text_no_provided = _PROVIDEDVALUE_RE.sub("", text)
                for lineno, line in enumerate(text_no_provided.splitlines(), start=1):
                    for tok in _NUMBER_RE.findall(line):
                        if tok not in known_numbers:
                            offenders.append((tex_path.relative_to(REPO_ROOT), lineno, tok, line.strip()))

            if provided:
                print(f"verify: {len(provided)} \\providedvalue{{}} escape(s) (non-experimental numbers):")
                for path, val in provided:
                    print(f"  {path.relative_to(REPO_ROOT)}: {val}")

            if offenders:
                failed = True
                print(f"\nverify: FAILED — {len(offenders)} numeric literal(s) untraceable to results/*.csv "
                      f"or tables/*.tex:")
                for path, lineno, tok, line in offenders:
                    print(f"  {path}:{lineno}: {tok!r} in: {line}")
            else:
                print("\nverify: OK — every numeric literal in paper/**/*.tex traces to results/ or tables/, "
                      "or is an explicit \\providedvalue{}.")

    # Check 2: pbs_jobid (08_FIX_PLAN.md §11) -- runs regardless of whether paper/ exists.
    pbs_offenders = _check_pbs_jobid()
    if pbs_offenders:
        failed = True
        print(f"\nverify: FAILED — {len(pbs_offenders)} result(s) produced after the fix-phase start "
              f"({_PHASE_START.isoformat()}Z) with pbs_jobid: null (ran on the login node, not PBS):")
        for path, utc_start in pbs_offenders:
            print(f"  {path} (utc_start={utc_start})")
    else:
        print(f"\nverify: OK — no post-phase-start ({_PHASE_START.isoformat()}Z) result under "
              f"results/ or tables/ has pbs_jobid: null.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

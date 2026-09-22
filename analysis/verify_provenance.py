#!/usr/bin/env python3
"""`make verify` — the provenance check (CLAUDE.md non-negotiable #3, build/00_BUILD_PLAN.md P0
task 6). Parses every `paper/**/*.tex`, extracts numeric literals appearing outside math-free prose
and captions is not attempted here at the character level (that's a much harder LaTeX-parsing
problem); instead this walks the raw text for standalone numeric tokens and requires each one to
be traceable to a `results/*.csv` row or a generated `tables/*.tex` file, with an explicit
`\\providedvalue{...}` escape for genuine non-experimental numbers (page counts, dataset sizes).

Exits non-zero (and lists every offending literal) if any number fails to trace. On an empty
`paper/` tree it reports cleanly with zero exceptions, which is what the P0 gate requires.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "tables"

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


def main() -> int:
    if not PAPER_DIR.exists():
        print("verify: paper/ does not exist yet — nothing to check. OK (clean, 0 exceptions).")
        return 0

    tex_files = sorted(PAPER_DIR.rglob("*.tex"))
    if not tex_files:
        print("verify: paper/ exists but contains no .tex files. OK (clean, 0 exceptions).")
        return 0

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
        print(f"\nverify: FAILED — {len(offenders)} numeric literal(s) untraceable to results/*.csv "
              f"or tables/*.tex:")
        for path, lineno, tok, line in offenders:
            print(f"  {path}:{lineno}: {tok!r} in: {line}")
        return 1

    print(f"\nverify: OK — every numeric literal in paper/**/*.tex traces to results/ or tables/, "
          f"or is an explicit \\providedvalue{{}}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

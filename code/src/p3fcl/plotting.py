"""Shared visual system, imported by every `analysis/figNN_*.py`. Plot scripts read a CSV and
nothing else — no computation at plot time, which is what makes `make figures` a regeneration
rather than a re-run (CLAUDE.md non-negotiable #3).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from .artifacts import Family  # noqa: E402

# Fixed artifact-family -> colour + marker, identical in every figure in the paper. Colour-blind-safe
# (Okabe-Ito-derived), and marker shape varies too so figures survive greyscale printing.
FAMILY_STYLE = {
    Family.MODEL_DELTA: {"color": "#1b9e77", "marker": "o", "label": "F1 model delta"},
    Family.PROTOTYPE: {"color": "#d95f02", "marker": "s", "label": "F2 prototype"},
    Family.PROMPT: {"color": "#7570b3", "marker": "^", "label": "F3 prompt"},
    Family.LOW_RANK: {"color": "#e7298a", "marker": "D", "label": "F4 low-rank"},
    Family.GRAM: {"color": "#66a61e", "marker": "v", "label": "F5 Gram"},
    Family.GENERATIVE: {"color": "#e6ab02", "marker": "P", "label": "F6 generative"},
    Family.COUNTS: {"color": "#a6761d", "marker": "X", "label": "F7 counts"},
    Family.EXEMPLAR: {"color": "#666666", "marker": "*", "label": "F8 exemplar"},
}


def style_for(family) -> dict:
    family = family if isinstance(family, Family) else Family(family)
    return FAMILY_STYLE[family]


def ci_band(ax, x, lo, hi, **kwargs):
    kwargs.setdefault("alpha", 0.25)
    kwargs.setdefault("linewidth", 0)
    return ax.fill_between(x, lo, hi, **kwargs)


def ci_point(ax, x, y, lo, hi, **kwargs):
    kwargs.setdefault("capsize", 3)
    fmt = kwargs.pop("fmt", "o")
    yerr = [[yi - li for yi, li in zip(y, lo)], [hi_i - yi for yi, hi_i in zip(y, hi)]]
    return ax.errorbar(x, y, yerr=yerr, fmt=fmt, **kwargs)


_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def latex_escape(value) -> str:
    """Escape LaTeX special characters in a raw string before it lands in a generated `.tex` table
    cell -- method/dataset names like `m4_proto` contain `_`, which LaTeX reads as "start a subscript"
    outside math mode and fails to compile on (found the hard way: `make paper` was wired up but never
    actually run until 2026-09-21, and every `tabNN_*.tex` with a method name in it failed to compile).
    Numbers/CI-bracket strings like `"0.997 [0.995, 0.998]"` pass through unchanged (no special chars)."""
    s = str(value)
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in s)


def column_width(kind: str = "single") -> float:
    """Figure width in inches matching a typical ICLR/USENIX column, so font sizes can be checked at
    final printed width rather than by zooming in an editor."""
    return {"single": 3.4, "double": 7.0}[kind]


def save(fig, stem: str, out_dir="figs", dpi: int = 200) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi)

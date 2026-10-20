"""Flag prose that goes past the write-up skill's sentence guidelines.

    uv run python .claude/skills/writeup/check.py notebooks/x.ipynb reports/y.md

Reads the markdown cells of a notebook or the text of a .md file, skips code, tables and headings, and prints
every sentence over 25 words, with more than two numbers, with a parenthesis, with a prediction code outside
the scorecard section, or with a retired term ("pull-back", "lift", "projection"; `GOALS.md` → Terms, 2026-10-20). The limits are guidelines: a flag means "look at this sentence", not "rewrite it". Ends with
one summary line per file. Stdlib only; counts are approximate on purpose.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MAX_WORDS = 25
MAX_NUMBERS = 2
SIZE_WORDS = r"steps?|seeds?|ranks?|layers?|rungs?|rounds?|cells?|epochs?|checkpoints?|rows?|tasks?|of|×|x"
# A size ("step 10", "rank 8") is matched with its word and ignored; any other number counts.
NUMBER = re.compile(rf"(\b(?:{SIZE_WORDS}) )?(?<![\w.])[-+]?\d+(?:[.,]\d+)*%?(?![\w.])")
CODE = re.compile(r"\b[PGCR]\d+′?\b|\be_(?:Δ|gate|delta)\b|\bkl_other\b")
RETIRED = re.compile(r"\b(pull[- ]?backs?|lifts?|lifted|lifting|projections?)\b", re.IGNORECASE)
INLINE = re.compile(r"`[^`]*`|\[([^\]]*)\]\([^)]*\)|\$[^$]*\$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\*\"“(])")


def markdown_chunks(path: Path) -> list[tuple[str, str]]:
    """(label, markdown) per cell for a notebook, or one chunk for a .md file."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        return [(f"cell {i}", "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"]
    return [("", path.read_text())]


def prose_lines(md: str) -> tuple[list[str], list[str]]:
    """Prose lines (code, tables, headings dropped) and the heading in force for each."""
    lines, headings, heading, fenced = [], [], "", False
    for line in md.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not line.strip() or line.lstrip().startswith("|"):
            continue
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
            continue
        lines.append(INLINE.sub(lambda m: m.group(1) or "", line))
        headings.append(heading)
    return lines, headings


def sentences(lines: list[str], headings: list[str]) -> list[tuple[str, str]]:
    out, buf, head = [], [], ""
    for line, h in zip(lines, headings, strict=True):
        if buf and (h != head or line.lstrip().startswith(("-", "*", "1", "2", "3", "4", "5"))):
            out.extend((s, head) for s in SENTENCE_END.split(" ".join(buf)))
            buf = []
        buf.append(line.strip())
        head = h
    if buf:
        out.extend((s, head) for s in SENTENCE_END.split(" ".join(buf)))
    return [(s.strip(), h) for s, h in out if s.strip()]


def check(path: Path) -> None:
    words = numbers = flagged = 0
    for label, md in markdown_chunks(path):
        for s, head in sentences(*prose_lines(md)):
            n_words, n_numbers = len(s.split()), sum(not size for size in NUMBER.findall(s))
            words += n_words
            numbers += n_numbers
            reasons = []
            if n_words > MAX_WORDS:
                reasons.append(f"{n_words} words")
            if n_numbers > MAX_NUMBERS:
                reasons.append(f"{n_numbers} numbers")
            if "(" in s:
                reasons.append("parenthesis")
            if "scorecard" not in head.lower() and CODE.search(s):
                reasons.append("code " + ", ".join(sorted(set(CODE.findall(s)))))
            if RETIRED.search(s):
                reasons.append("retired term " + ", ".join(sorted({m.lower() for m in RETIRED.findall(s)})))
            if reasons:
                flagged += 1
                where = f"{path}:{label}" if label else str(path)
                print(f"{where}: {'; '.join(reasons)}\n    {s[:110]}{'…' if len(s) > 110 else ''}")
    per100 = 100 * numbers / words if words else 0
    print(f"== {path}: {words} prose words, {per100:.1f} numbers per 100 words, {flagged} flagged sentences")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        check(Path(arg))

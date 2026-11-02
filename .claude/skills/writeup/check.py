"""Flag prose that goes past the write-up skill's sentence guidelines.

    uv run python .claude/skills/writeup/check.py notebooks/x.ipynb reports/analyses/y.md

Reads the markdown cells of a notebook, the text of a .md file or the prose of an .html log card (tags become sentence
breaks, alt text and tables are dropped: they merged sentences and inflated the flags, friction 151, 157, 164), skips
code, tables and headings, and prints
every sentence over 25 words, with more than two numbers, with a parenthesis, with a prediction code outside
the scorecard section, or with a retired term ("pull-back", "lift", "projection", 2026-10-20; "H′", now the fixed-shift hypothesis, 2026-10-22; `GOALS.md` → Terms). The limits are guidelines: a flag means "look at this sentence", not "rewrite it". Ends with
one summary line per file, after any errors. A retired term is an error, not a guideline: it comes first, and the summary
counts it as ERRORS: N (checkers filtered ~140 flags to find them, friction 243, 252). A file:line citation, bare or in parentheses, is not prose and is not flagged (friction
159). Stdlib only; counts are approximate on purpose.
"""

from __future__ import annotations

import html
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
RETIRED = re.compile(r"\b(pull[- ]?backs?|lifts?|lifted|lifting|projections?)\b|(?<!\w)H[′'](?!\w)", re.IGNORECASE)
INLINE = re.compile(r"`[^`]*`|\[([^\]]*)\]\([^)]*\)|\$(?!\d)[^$]*\$")  # $(?!\d): "$0.40" is money, not maths
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\*\"“(])")
ITALIC_LINE = re.compile(r"\*[^*\s][^*]*\*|_[^_\s][^_]*_")
# A source citation such as experiments/x/run.py:83 or src/commute/vpd.py; a parenthesis holding one is dropped whole.
PATH = r"(?<![\w/])[\w.-]*/[\w./-]*\.\w+(?::\d+(?:-\d+)?)?|\b[\w-]+\.(?:py|md|ipynb|json|toml|sh|html)(?::\d+(?:-\d+)?)?"
CITATION = re.compile(rf"\s*\([^()]*(?:{PATH})[^()]*\)|`?(?:{PATH})`?")
EMPTY_PAREN = re.compile(r"\s*\([\s,;:]*\)")  # what is left of "(`code`)" once the code is dropped
# Dropped whole: code, styles, tables, images (their alt text) and a card's id, chip and date line.
DROP_HTML = re.compile(
    r"<(script|style|table|svg)\b.*?</\1>|<img\b[^>]*>|<span class=\"(?:when|id|chip)\b[^\"]*\">.*?</span>",
    re.DOTALL | re.IGNORECASE,
)
BLOCK_TAG = re.compile(
    r"</?(p|div|li|ul|ol|h[1-6]|br|tr|article|section|header|figure|figcaption|dt|dd|blockquote)\b[^>]*>", re.IGNORECASE
)


def markdown_chunks(path: Path) -> list[tuple[str, str]]:
    """(label, markdown) per cell for a notebook, or one chunk for a .md file."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        return [(f"cell {i}", "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"]
    if path.suffix == ".html":
        return [("", html_to_markdown(path.read_text()))]
    return [("", path.read_text())]


def html_to_markdown(text: str) -> str:
    """Prose of an HTML card as markdown lines: one line per block, headings as #, list items as -, no tags."""
    text = DROP_HTML.sub("\n", text)
    text = re.sub(r"<h[1-6]\b[^>]*>", "\n# ", text, flags=re.IGNORECASE)
    text = re.sub(r"<li\b[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    text = re.sub(r"<code\b[^>]*>(.*?)</code>", r"`\1`", text, flags=re.DOTALL | re.IGNORECASE)
    text = BLOCK_TAG.sub("\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return "\n\n".join(" ".join(line.split()) for line in text.splitlines())


def prose_lines(md: str) -> tuple[list[str], list[str]]:
    """Prose lines (code, tables, headings dropped) and the heading in force for each."""
    lines, headings, heading, fenced = [], [], "", False
    for line in md.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not line.strip() and not fenced:
            lines.append("")  # a paragraph break: an unpunctuated line no longer runs into the next block
            headings.append(heading)
            continue
        if fenced or line.lstrip().startswith("|"):
            continue
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
            continue
        # Maths stays as a capitalised word, so a sentence that opens with it still splits (friction 222).
        lines.append(EMPTY_PAREN.sub("", INLINE.sub(lambda m: "X" if m[0].startswith("$") else m.group(1) or "",
                                                    CITATION.sub("", line))))
        headings.append(heading)
    return lines, headings


def sentences(lines: list[str], headings: list[str]) -> list[tuple[str, str]]:
    out, buf, head = [], [], ""
    for line, h in zip(lines, headings, strict=True):
        if buf and (not line or h != head or line.lstrip().startswith(("-", "*", "1", "2", "3", "4", "5"))):
            out.extend((s, head) for s in SENTENCE_END.split(" ".join(buf)))
            buf = []
        if not line:
            continue
        buf.append(line.strip())
        head = h
        if ITALIC_LINE.fullmatch(line.strip()):  # a dated *...* header line is its own sentence (friction 70)
            out.append((buf.pop(), head))
    if buf:
        out.extend((s, head) for s in SENTENCE_END.split(" ".join(buf)))
    return [(s.strip(), h) for s, h in out if s.strip()]


def check(path: Path) -> None:
    words = numbers = 0
    errors, flags = [], []
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
            where = f"{path}:{label}" if label else str(path)
            quote = f"{s[:110]}{'…' if len(s) > 110 else ''}"
            if RETIRED.search(s):
                terms = ", ".join(sorted({m.group(0).lower() for m in RETIRED.finditer(s)}))
                errors.append(f"ERROR {where}: retired term {terms}\n    {quote}")
            if reasons:
                flags.append(f"{where}: {'; '.join(reasons)}\n    {quote}")
    print("\n".join(errors + flags), end="\n" if errors or flags else "")
    per100 = 100 * numbers / words if words else 0
    print(f"== {path}: ERRORS: {len(errors)}; {words} prose words, {per100:.1f} numbers per 100 words, "
          f"{len(flags)} flagged sentences")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        check(Path(arg))

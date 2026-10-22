"""Build the research-log pages in build/ from parts/; `--check DIR` diffs them against live copies in DIR.

Recipe in README.md. A section is numbered by its place in SECTIONS; the text names sections as {{n:<id>}}, which the
build replaces with the current number, so a new section is one line here and one file in parts/sections/.
"""

import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
PARTS = HERE / "parts"
OUT = HERE / "build"

# Sections in reading order, each numbered by its place here, and the page that holds it. index.html is the artifact's
# page itself; the rest are published through `files`. Each id is a file parts/sections/<id>.html.
SECTIONS = [
    ("overview", "index.html"),
    ("questions", "index.html"),
    ("decisions", "index.html"),
    ("apparatus", "index.html"),
    ("techniques", "index.html"),
    ("experiments", "experiments.html"),
    ("next", "index.html"),
    ("claims", "index.html"),
    ("timeline", "index.html"),
    ("gaps", "index.html"),
    ("backlog", "index.html"),
    ("mistakes", "index.html"),
    ("glossary", "glossary.html"),
    ("repo", "index.html"),
    ("changelog", "changelog.html"),
]
NUM = {sid: i + 1 for i, (sid, _) in enumerate(SECTIONS)}
PAGES = {
    "index.html": "Update Decomposition Log",
    "experiments.html": "Experiments · Update Decomposition Log",
    "glossary.html": "Glossary · Update Decomposition Log",
    "changelog.html": "Changelog · Update Decomposition Log",
}
HREF = {page: "./" if page == "index.html" else page for page in PAGES}

HEAD = """<title>{title}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="style.css">
"""
# The publisher wraps only index.html in a document skeleton; the other pages bring their own.
SKELETON = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
SUBHEADER = '<header class="top"><div class="eyebrow"><a href="./">Update Decomposition Log</a></div></header>\n'
SEP = "\n<!-- ============================================================ -->\n"
CARDS = "<!-- cards: parts/experiments/, in number order -->\n"  # where experiments.html takes its cards


def card_key(path: Path) -> tuple[bool, int]:
    return (
        path.stem[0] != "e",
        int(path.stem[1:]),
    )  # E-codes in number order, then A-cards


def find(pattern: str, text: str) -> re.Match[str]:
    m = re.search(pattern, text)
    assert m, pattern
    return m


def number(html: str) -> str:
    """Fill the section's own number and every {{n:<id>}} reference."""
    html = re.sub(r"\{\{n:([a-z]+)\}\}", lambda m: str(NUM[m.group(1)]), html)
    assert "{{" not in html, "unknown {{...}} marker"
    return html


def sections() -> dict[str, dict]:
    cards = sorted((PARTS / "experiments").glob("*.html"), key=card_key)
    out = {}
    for sid, page in SECTIONS:
        html = (PARTS / "sections" / f"{sid}.html").read_text()
        html = html.replace(CARDS, "\n".join(p.read_text() for p in cards))
        assert html.startswith(f'<section id="{sid}">'), sid
        html = html.replace('<h2><span class="num"></span>', f'<h2><span class="num">{NUM[sid]}</span>', 1)
        label = re.sub(r"<[^>]+>", "", find(r'<h2><span class="num">\d+</span>(.*?)</h2>', html).group(1))
        out[sid] = {"id": sid, "num": NUM[sid], "label": label, "page": page, "html": number(html)}
    assert {p.stem for p in (PARTS / "sections").glob("*.html")} == set(NUM), "a file in parts/sections/ not in SECTIONS"
    return out


def link_across_pages(html: str, page: str, owner: dict[str, str]) -> str:
    """Point #id links at the page that holds the id."""

    def fix(m: re.Match) -> str:
        target = owner.get(m.group(1), page)
        return m.group(0) if target == page else f'href="{HREF[target]}#{m.group(1)}"'

    return re.sub(r'href="#([^"]+)"', fix, html)


def build() -> dict[str, str]:
    secs = sections()
    owner = {i: s["page"] for s in secs.values() for i in re.findall(r'\sid="([^"]+)"', s["html"])}
    toc = [{"id": s["id"], "num": s["num"], "label": s["label"], "page": HREF[s["page"]]} for s in secs.values()]
    epages = {i: HREF[p] for i, p in owner.items() if re.fullmatch(r"e\d+", i)}
    version, date = find(
        r'<tr><td class="c">(\d+)</td><td class="c">([\d-]+)</td>', secs["changelog"]["html"]
    ).groups()
    header = number((PARTS / "header.html").read_text())
    pages = {}
    for page, title in PAGES.items():
        body = SEP.join(link_across_pages(s["html"], page, owner) for s in secs.values() if s["page"] == page)
        pages[page] = (
            ("" if page == "index.html" else SKELETON)
            + HEAD.format(title=title)
            + ("" if page == "index.html" else "</head><body>\n")
            + '\n<div class="shell">\n<nav class="toc" aria-label="Contents">\n'
            + '  <p class="brand"><a href="./">Update Decomposition Log</a></p>\n'
            + f'  <p class="sub">Research log · {date}, v{version}</p>\n  <ol id="toc-list"></ol>\n</nav>\n\n'
            + '<main>\n<div class="col">\n\n'
            + (header if page == "index.html" else SUBHEADER)
            + '\n<details class="mobile-toc" id="mobile-toc"><summary>Contents</summary><ol id="mobile-toc-list"></ol></details>\n'
            + SEP
            + body
            + SEP
            + "\n</div>\n</main>\n</div>\n\n"
            + f"<script>var LOG_SECTIONS = {json.dumps(toc, ensure_ascii=False)};\n"
            + f"var LOG_EPAGES = {json.dumps(epages)};</script>\n"
            + '<script src="script.js"></script>\n'
            + ("" if page == "index.html" else "</body></html>\n")
        )
    pages["style.css"] = (PARTS / "style.css").read_text()
    pages["script.js"] = (PARTS / "script.js").read_text()
    return pages


def unwrap(live: str) -> str:
    """Strip the skeleton the publisher wraps around index.html."""
    live = live.replace("\r", "")
    if live.startswith("<!doctype html><html><head><meta charset=utf8>"):
        live = live.split("\n", 1)[1].removesuffix("\n</body></html>")
    return live


def main() -> None:
    pages = build()
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        for name, text in pages.items():
            live = Path(sys.argv[2]) / name
            if not live.exists():
                print(f"{name}: not in {sys.argv[2]}")
                continue
            diff = list(
                difflib.unified_diff(
                    unwrap(live.read_text()).splitlines(),
                    text.splitlines(),
                    "live",
                    "built",
                    n=0,
                    lineterm="",
                )
            )
            print(f"{name}: {'same' if not diff else f'{len(diff)} diff lines'}")
            if diff:
                print("\n".join(line[:300] for line in diff[:60]))
        return
    OUT.mkdir(exist_ok=True)
    for name, text in pages.items():
        (OUT / name).write_text(text)
    print(f"built {', '.join(pages)} in {OUT}")


if __name__ == "__main__":
    main()

"""Build the research-log pages in build/ from parts/.

  build.py                  write build/
  build.py --changed DIR    list the files a publish must pass, against the live nav.js saved in DIR (DIR is required)
  build.py --check DIR      diff the built pages against live copies in DIR; also flags #anchor links with no target id
  build.py --check-head DIR the same for parts/ as committed at HEAD, leaving local edits alone: setting them aside took
                            a clerk four git calls, each refused when chained in a worktree (friction 109)
  build.py --words          main-page words per section, here and at HEAD, against the length rule (friction 100, 244),
                            leaving out decision cards and cards marked `asked` (user, 2026-11-10); also rebuilds
                            build/ (friction 519)
  build.py --blocks         main-page words per block (card, table row, heading's prose), largest first, to plan every
                            cut in one pass rather than re-measure per edit (friction 208, 368, 376)
  build.py --missing        reports and E-codes since the newest changelog row that nothing in the log names (friction 349)

Recipe in README.md. A section is numbered by its place in SECTIONS; the text names sections as {{n:<id>}}, which the
build replaces with the current number, so a new section is one line here and one file in parts/sections/.
"""

import atexit
import difflib
import fnmatch
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
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
    ("next", "index.html"),
    ("apparatus", "index.html"),
    ("techniques", "index.html"),
    ("methodtable", "experiments.html"),
    ("neurontable", "experiments.html"),
    ("experiments", "experiments.html"),
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
WORD_LIMIT = 20_000  # README.md → Main-page length
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


def size_figures(html: str) -> str:
    """Give every figure its pixel size, so the page keeps its layout while 200 figures load.

    Without it each figure arrives with no height, everything below it moves, and a link to a late card lands short.
    """
    source = dict(re.findall(r'"(fig/[^"]+)": "([^"]+)"', (HERE / "README.md").read_text()))

    def fix(m: re.Match) -> str:
        path = HERE.parents[1] / source.get(m.group(1), "")
        if not path.is_file() or path.suffix != ".png":
            return m.group(0)
        head = path.read_bytes()[:24]
        w, h = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
        return f'{m.group(0)} width="{w}" height="{h}"'

    return re.sub(r'<img src="(fig/[^"]+)"(?![^>]*\bwidth=)', fix, html)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def tree_id(folder: Path) -> str:
    """The git tree id a commit of this folder would have, writing nothing (all files mode 100644)."""
    entries = []
    for p in folder.iterdir():
        if p.name.startswith(".") or p.name.endswith("~") or p.name == "__pycache__":
            continue  # an editor's or Python's file, never committed: it would read as "parts/ is not committed"
        mode, raw = ("40000", bytes.fromhex(tree_id(p))) if p.is_dir() else ("100644", blob_id(p))
        entries.append((p.name + "/" if p.is_dir() else p.name, f"{mode} {p.name}".encode() + b"\0" + raw))
    body = b"".join(entry for _, entry in sorted(entries, key=lambda e: e[0].encode()))
    return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()


def blob_id(path: Path) -> bytes:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).digest()


def in_a_commit(tree: str) -> bool:
    """A tree object exists in the clone only once a commit holds it: `git add` writes blobs, not trees (friction 323)."""
    return subprocess.run(["git", "-C", str(HERE.resolve()), "cat-file", "-e", f"{tree}^{{tree}}"],
                          check=False, capture_output=True).returncode == 0


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
            + '  <p class="sub" id="log-version"></p>\n  <ol id="toc-list"></ol>\n</nav>\n\n'
            + '<main>\n<div class="col">\n\n'
            + (header if page == "index.html" else SUBHEADER)
            + '\n<details class="mobile-toc" id="mobile-toc"><summary>Contents</summary><ol id="mobile-toc-list"></ol></details>\n'
            + SEP
            + body
            + SEP
            + "\n</div>\n</main>\n</div>\n\n"
            + '<script src="nav.js"></script>\n<script src="script.js"></script>\n'
            + ("" if page == "index.html" else "</body></html>\n")
        )
    pages = {name: size_figures(text) for name, text in pages.items()}
    pages["style.css"] = (PARTS / "style.css").read_text()
    pages["script.js"] = (PARTS / "script.js").read_text()
    # What changes on every flush lives in nav.js, so a flush that leaves a page alone need not read or publish it: the
    # version line on every page made each flush touch all of them, and each touched file can be refused (friction 224).
    hashes = {name: digest(text) for name, text in pages.items()}
    pages["nav.js"] = (
        f"var LOG_VERSION = {json.dumps(f'Research log · {date}, v{version}', ensure_ascii=False)};\n"
        f"var LOG_SECTIONS = {json.dumps(toc, ensure_ascii=False)};\n"
        f"var LOG_EPAGES = {json.dumps(epages)};\n"
        f"var LOG_HASHES = {json.dumps(hashes)};\n"
        f"var LOG_PARTS = {json.dumps(tree_id(PARTS))};\n"  # which parts/ built this page: --check (friction 323)
    )
    return pages


def unwrap(live: str) -> str:
    """Strip the skeleton the publisher wraps around index.html."""
    live = live.replace("\r", "")
    if live.startswith("<!doctype html><html><head><meta charset=utf8>"):
        live = live.split("\n", 1)[1].removesuffix("\n</body></html>")
    return live


EXEMPT_CARD = {"decision", "asked"}  # README.md → Main-page length: user, 2026-11-10


def exempt_cards(html: str) -> tuple[str, int]:
    """html without the cards the length rule exempts, and how many: a `<div class="card ...">` whose classes include
    `decision` (a pending-decision card) or `asked` (a card the user asked for). User, 2026-11-10; friction 444, 547,
    562, 586, 630, 631."""
    out, pos, n = [], 0, 0
    for m in re.finditer(r'<div\b[^>]*\bclass="([^"]*)"[^>]*>', html):
        classes = set(m[1].split())
        if m.start() < pos or "card" not in classes or not classes & EXEMPT_CARD:
            continue
        depth, end = 0, len(html)
        for tag in re.finditer(r"<div\b|</div>", html[m.start():]):
            depth += 1 if tag[0] != "</div>" else -1
            if depth == 0:
                end = m.start() + tag.end()
                break
        out.append(html[pos:m.start()])
        pos, n = end, n + 1
    out.append(html[pos:])
    return "".join(out), n


def main_words(parts: Path) -> dict[str, int]:
    """Words per main-page section under parts/, the length rule's count (README.md → Main-page length), leaving out
    the exempt cards (exempt_cards)."""
    files = {"header": parts / "header.html"}
    for sid, page in SECTIONS:
        if page == "index.html" and sid != "mistakes":
            files[sid] = parts / "sections" / f"{sid}.html"
    return {k: len(re.sub(r"<[^>]*>", " ", exempt_cards(f.read_text())[0]).split()) for k, f in files.items() if f.exists()}


def exempt_words(parts: Path) -> tuple[int, int]:
    """(cards, words) the length rule leaves out of the main page."""
    cards = words = 0
    for sid, page in SECTIONS:
        f = parts / "sections" / f"{sid}.html"
        if page == "index.html" and sid != "mistakes" and f.exists():
            html = f.read_text()
            kept, n = exempt_cards(html)
            cards += n
            words += len(re.sub(r"<[^>]*>", " ", html).split()) - len(re.sub(r"<[^>]*>", " ", kept).split())
    return cards, words


def head_parts() -> Path:
    """parts/ as committed at HEAD, in a temporary folder removed at exit."""
    git = ["git", "-C", str(HERE.resolve())]
    top = subprocess.run([*git, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    prefix = HERE.resolve().relative_to(top.stdout.strip()).as_posix()
    # git 2.34 refuses HEAD:./parts from a subfolder ("current working directory is untracked"), so run at the top.
    tar = subprocess.run(["git", "-C", top.stdout.strip(), "archive", f"HEAD:{prefix}/parts"],
                         capture_output=True, check=True).stdout
    tmp = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, tmp, True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(tmp)
    return Path(tmp)


def print_words() -> None:
    now, head = main_words(PARTS), main_words(head_parts())
    for k in sorted(now, key=lambda k: -now[k]):
        print(f"{now[k]:7,d}  {now[k] - head.get(k, 0):+6,d}  {k}")
    total, before = sum(now.values()), sum(head.values())
    print(f"{total:7,d}  {total - before:+6,d}  total (second column: change since HEAD); limit {WORD_LIMIT:,d}")
    cards, words = exempt_words(PARTS)
    print(f"not counted: {cards} decision or user-asked cards, {words:,d} words (exempt, user 2026-11-10)")
    if total > before:
        print(f"over: this flush adds {total - before:,d} words to the main page outside exempt cards; a flush may not "
              "make it longer, and cuts come from your own lead's text only")


def print_blocks() -> None:
    blocks = []
    for sid, page in SECTIONS:
        if page != "index.html" or sid == "mistakes":
            continue
        html = exempt_cards((PARTS / "sections" / f"{sid}.html").read_text())[0]  # exempt cards are not cut
        for block in re.split(r'(?=<tr[ >]|<li[ >]|<div class="card|<h[34][ >]|<div class="q[ "])', html):
            text = re.sub(r"<[^>]*>", " ", block).split()
            blocks.append((len(text), sid, " ".join(text[:9])))
    for words, sid, start in sorted(blocks, reverse=True)[:40]:
        print(f"{words:6,d}  {sid:<11}{start[:70]}")


def print_missing() -> None:
    """Every report dated on or after the newest changelog row, and every E-code, that no part of the log or queue names."""
    top = HERE.resolve().parents[1]
    named = (HERE / "PENDING.md").read_text() + "".join(f.read_text() for f in PARTS.rglob("*.html"))
    last = find(r'<tr><td class="c">\d+</td><td class="c">([\d-]+)</td>',
                (PARTS / "sections" / "changelog.html").read_text())[1]
    # A wildcard in the changelog or queue counts as naming (analysis_qwen-*); a word in prose (failed*) does not.
    listed = (HERE / "PENDING.md").read_text() + (PARTS / "sections" / "changelog.html").read_text()
    wild = {w.rsplit("/", 1)[-1] for w in re.findall(r"[\w./-]*[a-z-]\*[\w.*-]*", listed) if re.search(r"[_-]", w)}
    for f in sorted(top.glob("reports/*/20*")):
        if f.name[:10] >= last and f.name not in named and not any(fnmatch.fnmatch(f.name, f"*{w}") for w in wild):
            print(f"not in the log: {f.relative_to(top)}")
    tried = (top / "experiments" / "TRIED.md").read_text()
    for code in sorted(set(re.findall(r"^\| E(\d+)", tried, re.MULTILINE)), key=int):
        if f'id="e{code}"' not in named:
            print(f"no card: E{code}")


def unlanded_lines() -> list[str]:
    """Queue lines this flush removed (working tree against HEAD) whose file, E-code or card id appears in nothing it
    added to parts/. Run before the commit. It finds a line never applied, not one applied badly (friction 155: v110
    emptied the queue without applying one)."""
    top = HERE.resolve().parents[1]
    git = ["git", "-C", str(top)]
    rel = HERE.resolve().relative_to(top).as_posix()
    shown = subprocess.run([*git, "show", f"HEAD:{rel}/PENDING.md"], check=False, capture_output=True, text=True)
    now = (HERE / "PENDING.md").read_text().splitlines()
    gone = [line for line in shown.stdout.splitlines() if line.startswith("- 20") and line not in now]
    if not gone:
        return []
    diff = subprocess.run([*git, "diff", "-U0", "HEAD", "--", f"{rel}/parts"], check=False, capture_output=True, text=True).stdout
    added = "\n".join(line for line in diff.splitlines() if line.startswith("+"))
    new = subprocess.run([*git, "ls-files", "-o", "--exclude-standard", f"{rel}/parts"], check=False, capture_output=True, text=True)
    added += "".join((top / f).read_text() for f in new.stdout.split())
    out, files = [], []
    for line in gone:
        # Files anywhere in the line, not only its second field: lines that name the report in the prose, or with a
        # colon after it and no " · " fields, warned falsely (friction 489, 595). "same report" with no file of its
        # own means the previous line's files.
        found = re.findall(r"[\w-][\w.-]*\.(?:md|py|ipynb|json|png|html)\b", line)
        if found or "same report" not in line:
            files = found
        keys = [*files, *re.findall(r"\bE\d+\b", line), *re.findall(r"\bdecision-[\w-]+", line)]
        if not any(k in added for k in keys):
            out.append(line)
    return out


def live_nav(live_dir: Path) -> str:
    return (live_dir / "nav.js").read_text() if (live_dir / "nav.js").exists() else ""


def live_hashes(live_dir: Path) -> dict[str, str]:
    m = re.search(r"var LOG_HASHES = (\{.*?\});", live_nav(live_dir))
    return json.loads(m[1]) if m else {}


def changed_files(pages: dict[str, str], live_dir: Path) -> list[str]:
    """The built files whose hash differs from the live nav.js in live_dir, or all when it has none."""
    seen = live_hashes(live_dir)
    return [name for name, text in pages.items() if name == "nav.js" or seen.get(name) != digest(text)]


def dead_anchors(pages: dict[str, str]) -> list[str]:
    """`#id` links (also `page.html#id`) in the built pages whose id exists in none of them (friction 588)."""
    html_pages = {n: t for n, t in pages.items() if n.endswith(".html")}
    ids = {i for t in html_pages.values() for i in re.findall(r'\sid="([^"]+)"', t)}
    out = []
    for name, text in html_pages.items():
        for page, frag in sorted(set(re.findall(r'href="([\w.-]*)#([^"]+)"', text))):
            if frag not in ids:
                out.append(f"{name} links to {page}#{frag}, no element with that id in the built pages")
    return out


def unmapped_figures(pages: dict[str, str]) -> list[str]:
    """Figures the pages show that README.md's published-figures list does not name, so no publish can find them."""
    listed = set(re.findall(r'"(fig/[^"]+)":', (HERE / "README.md").read_text()))
    shown = {m for text in pages.values() for m in re.findall(r'src="(fig/[^"]+)"', text)}
    return sorted(shown - listed)


def main() -> None:
    global PARTS
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)  # it built the log instead (friction 161)
        return
    if sys.argv[1:] == ["--words"]:
        print_words()
        # --words used to leave build/ as it was, so a publish after it sent stale pages (friction 519).
        OUT.mkdir(exist_ok=True)
        for name, text in build().items():
            (OUT / name).write_text(text)
        print(f"also rebuilt build/ from the current parts/ ({OUT}), so a publish now sends what was measured")
        return
    if sys.argv[1:] == ["--blocks"]:
        print_blocks()
        return
    if sys.argv[1:] == ["--missing"]:
        print_missing()
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--check-head":
        PARTS = head_parts()
        sys.argv[1] = "--check"
    pages = build()
    if len(sys.argv) == 3 and sys.argv[1] == "--changed":
        print(" ".join(changed_files(pages, Path(sys.argv[2]))))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        folder = Path(sys.argv[2])
        stamp = re.search(r'var LOG_PARTS = "(\w+)"', live_nav(folder))
        if stamp and not in_a_commit(stamp[1]):
            print(f"live page built from parts/ in no commit here ({stamp[1][:12]}): pull; if it stays, it holds text "
                  "the repo lacks, so merge the diff below into parts/ (friction 323)")
        hashes = live_hashes(folder)
        unread: list[str] = []
        for name, text in pages.items():
            live = folder / name
            if not live.exists():
                unread.append(name)
                continue
            # A page read for an earlier flush stays in the folder; only the nav.js read here says what is live (friction 368).
            if name in hashes and digest(unwrap(live.read_text())) != hashes[name]:
                print(f"{name}: older copy than the nav.js here, not compared; read it if --changed names it")
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
            places = sum(line.startswith("@@") for line in diff)
            print(f"{name}: {'same' if not diff else f'{len(diff)} diff lines in {places} places'}")
            if diff:
                print("\n".join(line[:300] for line in diff[:60]))
        if unread:  # one line, not one per page: a page not read is not compared (friction 561)
            print(f"not compared, not saved in {folder}: {', '.join(unread)} (read one only if --changed names it)")
        for target in dead_anchors(pages):
            print(f"dead link: {target}")
        return
    OUT.mkdir(exist_ok=True)
    for name, text in pages.items():
        (OUT / name).write_text(text)
    print(f"built {', '.join(pages)} in {OUT}")
    for fig in unmapped_figures(pages):
        print(f"warning: {fig} has no source line in README.md's figure list; add one (friction 163)")
    for target in dead_anchors(pages):  # also here, not only under --check: removed cards left 7 behind (friction 609)
        print(f"warning: dead link: {target}; drop the link or point it at a live id")
    for line in unlanded_lines():
        print(f"warning: flushed but no trace in this flush's parts/ edits; check it landed (friction 155): {line[:200]}")
    if not in_a_commit(tree_id(PARTS)):
        print("parts/ as built is not committed: commit and push it before publishing (friction 138, 323)")


if __name__ == "__main__":
    main()

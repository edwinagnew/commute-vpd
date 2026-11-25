"""E128 (qwen_post_cutoff): statements from the public Wikidata query endpoint into a folder given as argument
(default artifacts/post_cutoff/, gitignored, untrusted: no script of ours lives there, nothing there is executed).

Each relation is one SPARQL pattern giving ?s (the subject: an event, award or office), ?a (the answer entity) and
?date, for one date window of one side (new, gap, known). Every side and relation is cut into windows so each query
returns inside the endpoint's 60 s limit; one query at a time, one second apart, with a generic User-Agent and no
personal detail. A window file already on disk is skipped, so a job can be resubmitted. Then the English aliases of the
answer entities go to aliases.json (fetched for the answers of every window file; `grade.py` uses them).

Sides (plan's "Settled 2026-11-09", Fable (a)): new = dated on or after 2025-05-01; known = on or before 2023-12-31;
gap = 2024-01-01 to 2025-04-30, 16-sample report only (Fable's "known rate per half-year 2023-2025"), never in a pool.

From the repo root:
  fleet/cpu-run python experiments/qwen_post_cutoff/fetch.py --probe winner --out artifacts/post_cutoff/probe
  sbatch cluster/e128_fetch.sbatch          (the full fetch)
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENDPOINT = "https://query.wikidata.org/sparql"
AGENT = "commute-research-fetch/0.1"
GAP_S, RETRIES = 1.0, 4
ALIAS_CHUNK, SUBJECT_CHUNK = 200, 100  # entities per alias query; subjects per statement query


@dataclass(frozen=True)
class Relation:
    pattern: str
    min_links: tuple[int, int]  # sitelink floor of ?s (new or gap side, known side): notable subjects, and a window that
    # returns under the row limit (a floor of 0 returned 4,000 single-match items and cut the rest arbitrarily)
    subjects: str = ""  # a pattern listing ?s: the statements are then fetched per chunk of subjects, not scanned whole


RELATIONS = {
    "winner": Relation("?s wdt:P1346 ?a . ?s wdt:P585|wdt:P582 ?date .", (3, 3)),
    "election": Relation("?s wdt:P991 ?a . ?s wdt:P585 ?date .", (2, 2)),
    # a whole scan of P166 or P39 with its date qualifier outruns the 60 s limit (responses cut mid-JSON, 2026-11-09)
    "award": Relation("?a p:P166 ?st . ?st ps:P166 ?s ; pq:P585 ?date .", (10, 10), "?s wdt:P31/wdt:P279* wd:Q618779 ."),
    "appointee": Relation("?a p:P39 ?st . ?st ps:P39 ?s ; pq:P580 ?date .", (8, 8), "?s wdt:P31/wdt:P279* wd:Q4164871 ."),
    # the range hint makes the date filter use the index: without it the P570 scan outran the 60 s limit (2026-11-09)
    "death": Relation("?s wdt:P570 ?date . hint:Prior hint:rangeSafe true . ?s wdt:P20 ?a ; wdt:P31 wd:Q5 .", (20, 30)),
}
NEW, GAP = ("2025-05-01", "2026-11-08"), ("2024-01-01", "2025-04-30")
KNOWN_YEARS = range(1990, 2024)  # 1990-2023, one window per year
ROW_LIMIT = 4000  # per window; a window that returns it exactly is reported, as rows may be cut


def windows() -> list[tuple[str, str, str]]:
    """(side, from, to) in fetch order: new first, then gap, then one per known year."""
    return [("new", *NEW), ("gap", *GAP), *[("known", f"{y}-01-01", f"{y}-12-31") for y in KNOWN_YEARS]]


def query(rel: Relation, side: str, start: str, end: str, limit: int, values: str = "") -> str:
    links = rel.min_links[1] if side == "known" else rel.min_links[0]
    return f"""SELECT DISTINCT ?s ?sLabel ?a ?aLabel ?date ?links WHERE {{
  {values}
  {rel.pattern}
  FILTER(?date >= "{start}T00:00:00Z"^^xsd:dateTime && ?date <= "{end}T23:59:59Z"^^xsd:dateTime)
  ?s wikibase:sitelinks ?links . FILTER(?links >= {links})
  ?s rdfs:label ?sLabel . FILTER(LANG(?sLabel) = "en")
  ?a rdfs:label ?aLabel . FILTER(LANG(?aLabel) = "en")
}} LIMIT {limit}"""


def ask(sparql: str) -> list[dict[str, Any]]:
    """The bindings of one query; a 429 waits as told, a 5xx or a timeout waits and retries (RETRIES times)."""
    data = urllib.parse.urlencode({"query": sparql}).encode()
    req = urllib.request.Request(ENDPOINT, data=data, headers={"User-Agent": AGENT, "Accept": "application/sparql-results+json"})
    for i in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=75) as r:
                return json.loads(r.read(), strict=False)["results"]["bindings"]  # strict=False: a label may hold a raw control character
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == RETRIES - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 20 * (i + 1))
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as e:
            if i == RETRIES - 1:
                raise
            print(f"[fetch] {e}", flush=True)
            wait = 20 * (i + 1)
        print(f"[fetch] retry {i + 1} after {wait} s", flush=True)
        time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_window(out: Path, name: str, rel: Relation, side: str, start: str, end: str, limit: int,
                 values: str = "", suffix: str = "") -> None:
    f = out / name / f"{side}_{start[:4]}{suffix}.json"
    if f.is_file():
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = [{"s": r["s"]["value"].rsplit("/", 1)[1], "s_label": r["sLabel"]["value"], "a": r["a"]["value"].rsplit("/", 1)[1],
             "a_label": r["aLabel"]["value"], "date": r["date"]["value"][:10], "links": int(r["links"]["value"])}
            for r in ask(query(rel, side, start, end, limit, values))]
    f.write_text(json.dumps({"relation": name, "side": side, "from": start, "to": end, "rows": rows}, indent=0))
    flag = " (HIT THE LIMIT: rows may be cut)" if len(rows) == limit else ""
    print(f"[fetch] {name} {side} {start[:4]}{suffix}: {len(rows)} rows, {time.time() - t0:.1f} s{flag}", flush=True)
    time.sleep(GAP_S)


def fetch_by_subject(out: Path, name: str, rel: Relation) -> None:
    """Relations too large to scan: list the notable subjects once (<name>_subjects.json), then one query per chunk of
    SUBJECT_CHUNK subjects and side, the known side as one window."""
    path = out / f"{name}_subjects.json"
    if not path.is_file():
        floor = min(rel.min_links)
        q = f"SELECT DISTINCT ?s WHERE {{ {rel.subjects} ?s wikibase:sitelinks ?links . FILTER(?links >= {floor}) }}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(r["s"]["value"].rsplit("/", 1)[1] for r in ask(q))))
        time.sleep(GAP_S)
    ids = json.loads(path.read_text())
    print(f"[fetch] {name}: {len(ids)} subjects", flush=True)
    for side, (start, end) in (("new", NEW), ("gap", GAP), ("known", (f"{KNOWN_YEARS[0]}-01-01", f"{KNOWN_YEARS[-1]}-12-31"))):
        for c, i in enumerate(range(0, len(ids), SUBJECT_CHUNK)):
            values = "VALUES ?s { " + " ".join(f"wd:{q}" for q in ids[i : i + SUBJECT_CHUNK]) + " }"
            fetch_window(out, name, rel, side, start, end, ROW_LIMIT, values, f"_{c:04d}")


def fetch_aliases(out: Path) -> None:
    """{Q-id: [English aliases]} for every answer entity in the window files under `out`, written to aliases.json."""
    path = out / "aliases.json"
    known = json.loads(path.read_text()) if path.is_file() else {}
    qids = sorted({r["a"] for f in out.glob("*/*.json") for r in json.loads(f.read_text())["rows"]} - set(known))
    print(f"[aliases] {len(qids)} entities to fetch", flush=True)
    for i in range(0, len(qids), ALIAS_CHUNK):
        chunk = qids[i : i + ALIAS_CHUNK]
        values = " ".join(f"wd:{q}" for q in chunk)
        q = f"""SELECT ?e ?alias WHERE {{ VALUES ?e {{ {values} }} ?e skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }}"""
        known |= {q: [] for q in chunk}
        for r in ask(q):
            known[r["e"]["value"].rsplit("/", 1)[1]].append(r["alias"]["value"])
        print(f"[aliases] {i + len(chunk)}/{len(qids)}", flush=True)
        path.write_text(json.dumps(known))
        time.sleep(GAP_S)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts/post_cutoff"))
    p.add_argument("--probe", choices=RELATIONS, help="one relation, 50 rows per side, one window each")
    a = p.parse_args()
    if a.probe:
        rel = RELATIONS[a.probe]
        for side, (start, end) in (("new", NEW), ("gap", GAP), ("known", ("1990-01-01", "2023-12-31"))):
            fetch_window(a.out, a.probe, rel, side, start, end, 50)
    else:
        for name, rel in RELATIONS.items():
            if rel.subjects:
                fetch_by_subject(a.out, name, rel)
                continue
            for side, start, end in windows():
                fetch_window(a.out, name, rel, side, start, end, ROW_LIMIT)
    fetch_aliases(a.out)
    print(f"[fetch] done; {sum(1 for _ in a.out.glob('*/*.json'))} window files in {a.out}", flush=True)


if __name__ == "__main__":
    main()

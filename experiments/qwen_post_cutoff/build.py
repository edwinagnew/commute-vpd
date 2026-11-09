"""E128 build (CPU, $0): candidate items from the window files `fetch.py` wrote (a path argument, untrusted data; read as
JSON only). Plan: reports/plans/2026-11-06_plan_e128-post-cutoff-facts.md, "Build"; appendix for the schema.

A fact is one (relation, subject) with one answer entity. One row per fact is kept only if its window files agree on a
single answer; a fact dated in a window is on that side. Filters (each counted per relation and side): the answer has no digit,
is 2-40 characters and 1-8 Qwen tokens (bins 1-2, 3-4, 5-8), the subject is at most 80 characters and has no score
("1-0"), the answer does not appear in the subject (a leak), and for winner and election the subject carries a year
within one of the date's. Award and appointee facts need one recipient for that (subject, year).
Items are E120's schema, four per fact, one per template (0-2 trained, 3 held out): {id, source, level, subject, user,
gold_response, answer} plus aliases, date, fact, template, answer_tokens, links. Sides: new, known, gap (the 16-sample
half-year report only, never in a pool). Per relation and side at most CAP facts are kept, by sitelinks of the subject
(ties in a seeded order). A known-side window that returned exactly the fetch's row limit is left out: its rows are an
arbitrary part, so a several-recipient fact could read as having one answer. A fact on two sides (two dates) is cut.
Writes results/large/candidates.json (over 1 MB, so not committed) and results/candidates_meta.json (counts, committed).

From the repo root:
  fleet/cpu-run --mem 4G python experiments/qwen_post_cutoff/build.py --windows artifacts/post_cutoff/full
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fetch import ROW_LIMIT
from transformers import AutoTokenizer

from commute.qwen_eval import MODEL, MODEL_REV
from commute.qwen_facts import FORMAT, norm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 0
CAP = {"new": 100, "gap": 60, "known": 300}  # per relation: about 400 new and 1,500 known facts (the plan's targets)
MAX_TOKENS, BINS = 8, ((1, 2), (3, 4), (5, 8))
TEMPLATES = {
    "winner": ("Who won the {x}?", "Who was the winner of the {x}?", "Name the winner of the {x}.",
               "Which competitor or team won the {x}?"),
    "election": ("Who won the {x}?", "Who was elected in the {x}?", "Name the winner of the {x}.",
                 "Which candidate won the {x}?"),
    "award": ("Who received the {x} in {y}?", "Who was awarded the {x} in {y}?", "Name the recipient of the {x} in {y}.",
              "Who won the {x} in {y}?"),
    "appointee": ("Who became {x} in {y}?", "Who took office as {x} in {y}?", "Name the person who became {x} in {y}.",
                  "Who started serving as {x} in {y}?"),
    "death": ("Where did {x} die?", "In which place did {x} die?", "What was the place of death of {x}?",
              "Name the place where {x} died."),
}
YEAR_KEYED, YEAR_IN_LABEL = {"award", "appointee"}, {"winner", "election"}
_YEAR, _SCORE = re.compile(r"\b(19|20)\d\d\b"), re.compile(r"(?<!\d)\d{1,2}\s*[-–]\s*\d{1,2}(?!\d)")


def stratum_bin(n: int) -> str:
    return next(f"{lo}-{hi}" for lo, hi in BINS if lo <= n <= hi)


def read_rows(windows: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Per (relation, side, fact key), the fact as {s, s_label, a, a_label, date, links, year}; a fact with more than
    one answer entity carries a set in "a"."""
    facts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for f in sorted(windows.glob("*/*.json")):
        d = json.loads(f.read_text())
        rel, side = d["relation"], d["side"]
        if side == "known" and len(d["rows"]) == ROW_LIMIT:
            print(f"[build] left out, rows cut at the limit: {f.parent.name}/{f.name}", flush=True)
            continue
        for r in d["rows"]:
            year = r["date"][:4]
            key = (rel, side, f"{r['s']}/{year}" if rel in YEAR_KEYED else r["s"])
            row = facts.setdefault(key, {**r, "year": year, "answers": set()})
            row["answers"].add(r["a"])
            row["links"] = max(row["links"], r["links"])
    return facts


def reject(rel: str, fact: dict[str, Any]) -> str | None:
    """The filter that drops this fact, or None."""
    a, s = fact["a_label"], fact["s_label"]
    if len(fact["answers"]) > 1:
        return "several answers"
    if re.search(r"[\d\x00-\x1f]", a) or not 2 <= len(a) <= 40 or re.fullmatch(r"Q\d+", a):
        return "answer form"
    if len(s) > 80 or _SCORE.search(s) or re.search(r"[\x00-\x1f]", s):
        return "subject form"
    if rel in YEAR_KEYED and s[0].islower():  # Wikidata's generic classes ("abbot", "mayor"): the question has no single answer
        return "generic subject"
    if norm(a) in norm(s):
        return "answer in subject"
    if rel in YEAR_IN_LABEL and not any(abs(int(m.group(0)) - int(fact["year"])) <= 1 for m in _YEAR.finditer(s)):
        return "no year in subject"
    return None


def clean_aliases(answer: str, aliases: list[str]) -> list[str]:
    """Aliases of 4 or more characters that are not all capitals (codes) and not the label itself; digits refused."""
    keep = {a for a in aliases if len(a) >= 4 and not a.isupper() and not re.search(r"\d", a) and norm(a) != norm(answer)}
    return sorted(keep)


def facts_to_items(windows: Path, aliases: dict[str, list[str]], tok: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts = read_rows(windows)
    cuts, kept = Counter(), defaultdict(list)
    n_sides = Counter((rel, key) for rel, _, key in facts)
    for (rel, side, key), f in facts.items():
        why = "on two sides" if n_sides[(rel, key)] > 1 else reject(rel, f)
        if why is None and f["a_label"] and not 1 <= len(tok(f["a_label"], add_special_tokens=False)["input_ids"]) <= MAX_TOKENS:
            why = "answer tokens"
        if why:
            cuts[f"{rel}/{side}/{why}"] += 1
            continue
        kept[(rel, side)].append((key, f))
    rng, items = random.Random(SEED), []
    table: dict[str, dict[str, int]] = defaultdict(dict)
    for (rel, side), rows in sorted(kept.items()):
        rows.sort(key=lambda kf: (-kf[1]["links"], rng.random()))
        table[rel][side] = len(rows)
        for key, f in rows[: CAP[side]]:
            items += fact_items(rel, side, key, f, aliases.get(f["a"], []), tok)
    return items, {"available_after_filters": dict(sorted(table.items())), "cap_per_relation": CAP,
                   "cuts": dict(sorted(cuts.items())), "items": len(items), "facts": len(items) // 4}


def fact_items(rel: str, side: str, key: str, f: dict[str, Any], aliases: list[str], tok: Any) -> list[dict[str, Any]]:
    fid = f"{rel}/{key}"
    ans, n_tok = f["a_label"], len(tok(f["a_label"], add_special_tokens=False)["input_ids"])
    out = []
    for t, tpl in enumerate(TEMPLATES[rel]):
        q = tpl.format(x=f["s_label"], y=f["year"])
        out.append({"id": f"post_cutoff/{fid}/t{t}", "source": "post_cutoff", "level": side, "subject": rel,
                    "user": f"{q}\n\n{FORMAT}", "gold_response": f"The answer is {ans.rstrip('.')}.", "answer": ans,
                    "aliases": clean_aliases(ans, aliases), "date": f["date"], "fact": fid, "template": t,
                    "answer_tokens": n_tok, "stratum": f"{rel}/{stratum_bin(n_tok)}", "links": f["links"]})
    return out


def main(windows: Path, aliases_path: Path, out: Path, meta_path: Path) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    aliases = json.loads(aliases_path.read_text()) if aliases_path.is_file() else {}
    items, meta = facts_to_items(windows, aliases, tok)
    counts: Counter[str] = Counter(f"{it['subject']}/{it['level']}" for it in items if it["template"] == 0)
    meta |= {"seed": SEED, "kept_per_relation_side": dict(sorted(counts.items()))}
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(items, indent=0)
    out.write_text(text)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[build] {len(items)} items, {meta['facts']} facts -> {out} ({len(text) / 1e6:.2f} MB)", flush=True)
    print(json.dumps({"available": meta["available_after_filters"], "kept": meta["kept_per_relation_side"]}, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--aliases", type=Path, help="default: <windows>/aliases.json")
    ap.add_argument("--out", type=Path, default=RESULTS / "large" / "candidates.json")
    ap.add_argument("--meta", type=Path, default=RESULTS / "candidates_meta.json")
    a = ap.parse_args()
    main(a.windows, a.aliases or a.windows / "aliases.json", a.out, a.meta)

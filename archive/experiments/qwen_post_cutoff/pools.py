"""E128 stage 2 (CPU, $0): the new and known pools from stage 1's merged records, E120's `pools.py` rules, restated (its strata
are maths). Plan: reports/plans/2026-11-06_plan_e128-post-cutoff-facts.md, "Stage 2"; Fable (b).

Labels, read on the 48 fresh samples: new = new-side candidate with 0 of 16 and 0 of 48; known = known-side candidate at
>= KNOWN_BAR of 48. Gap-side facts never enter a pool. Out of both sides: >= UNREAD_GUARD of 64 with no "The answer is"
line (or truncated, unread). Out of the new side: >= WRONG_GUARD of 64 on one normalised wrong answer (changed facts,
cut and counted with ids and the commonest answer in meta.changed_facts, Fable (b); "non-gold non-empty >= 32" is reported
beside it). Strata = relation x answer-token bin (1-2, 3-4, 5-8); per stratum m = min(new, known), each side drawn m by
a seeded shuffle, the rest cut and counted, so both pools have the same n. A pool fact gives train = templates 0-2 and
held = template 3 (items as in candidates.json, the schema `commute.pool_sft` reads). meta also carries n if changed
facts were kept (never used: a decision card if it matters). Prints 20 new-pool facts with the model's commonest answers
for the lead to read.
Writes results/pools.json. CPU, from the repo root: fleet/cpu-run --mem 4G python experiments/qwen_post_cutoff/pools.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEED = 0
KNOWN_BAR, WRONG_GUARD, UNREAD_GUARD, N_READ = 36, 32, 16, 20
RULES = {"new": "side new, k16 == 0 and k48 == 0 (n48 == 48)", "known": f"side known, k48 >= {KNOWN_BAR}",
         "guard_both": f"unread16 + unread48 >= {UNREAD_GUARD}", "guard_new": f"one normalised wrong answer >= {WRONG_GUARD}"}
Rec = dict[str, Any]


def label(m: Rec) -> str | None:
    if m["n48"] != 48:
        return None
    if m["side"] == "new" and m["k16"] + m["k48"] == 0:
        return "new"
    return "known" if m["side"] == "known" and m["k48"] >= KNOWN_BAR else None


def top_wrong(m: Rec) -> tuple[int, str | None]:
    """(the largest number of samples on one normalised wrong answer, that answer); `n > 0` samples read as an answer
    are all wrong here since a new-side record has no correct sample."""
    answers = [a for a in m["a16"] + m["a48"] if a]
    return (Counter(answers).most_common(1)[0][1], Counter(answers).most_common(1)[0][0]) if answers else (0, None)


def sides(merged: list[Rec], with_changed: bool) -> tuple[dict[str, dict[str, list[Rec]]], Counter[str], list[dict[str, Any]]]:
    """Per stratum, {"new": [...], "known": [...]} after the guards; cuts counted; changed facts listed."""
    by: dict[str, dict[str, list[Rec]]] = defaultdict(lambda: {"new": [], "known": []})
    cut, changed = Counter(), []
    for m in sorted(merged, key=lambda m: m["fact"]):
        side = label(m)
        if side is None:
            continue
        unread = m["unread16"] + m["unread48"] >= UNREAD_GUARD
        tw, commonest = top_wrong(m)
        wrong = side == "new" and tw >= WRONG_GUARD
        if wrong:
            changed.append({"fact": m["fact"], "top_wrong": tw, "commonest": commonest})
        if unread:
            cut[f"{side}/unread"] += 1
        if wrong and not with_changed:
            cut["new/changed_fact"] += 1
        if unread or (wrong and not with_changed):
            continue
        by[m["stratum"]][side].append(m)
    return by, cut, changed


def match(by: dict[str, dict[str, list[Rec]]]) -> tuple[dict[str, list[Rec]], dict[str, Any]]:
    """1:1 seeded matching per stratum: ({"known"|"new": [records]}, per-stratum table)."""
    pools: dict[str, list[Rec]] = {"known": [], "new": []}
    table = {}
    for s, d in sorted(by.items()):
        m = min(len(d["new"]), len(d["known"]))
        rng = random.Random(f"{SEED}/{s}")
        for side, rs in d.items():
            pools[side] += rng.sample(rs, len(rs))[:m]
        table[s] = {"new": len(d["new"]), "known": len(d["known"]), "matched": m,
                    "cut_new_unmatched": len(d["new"]) - m, "cut_known_unmatched": len(d["known"]) - m}
    return pools, table


def nongold_nonempty_count(merged: list[Rec]) -> int:
    """Fable (b)'s second reading, reported: new-side facts with >= WRONG_GUARD of 64 samples giving a non-empty answer that is not gold
    (they are all wrong on a fact at 0 of 64; spellings of one wrong answer may be split among them)."""
    return sum(1 for m in merged if label(m) == "new" and sum(1 for a in m["a16"] + m["a48"] if a) >= WRONG_GUARD)


def build(items: list[dict[str, Any]], merged: list[Rec]) -> dict[str, Any]:
    by_fact: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for it in items:
        by_fact[it["fact"]][it["template"]] = it
    by, cut, changed = sides(merged, with_changed=False)
    pools, table = match(by)
    n_if_kept = len(match(sides(merged, with_changed=True)[0])[0]["new"])
    assert len(pools["known"]) == len(pools["new"])
    out: dict[str, Any] = {side: {"train": [by_fact[m["fact"]][t] for m in rs for t in (0, 1, 2)],
                                  "held": [by_fact[m["fact"]][3] for m in rs]} for side, rs in pools.items()}
    out["meta"] = {"seed": SEED, "rules": RULES, "n": len(pools["new"]),
                   "sizes": {s: {p: len(v) for p, v in out[s].items()} for s in ("known", "new")},
                   "guard_cuts": dict(sorted(cut.items())), "changed_facts": changed, "changed_count": len(changed),
                   "n_if_changed_kept": n_if_kept, "nongold_nonempty_ge_32": nongold_nonempty_count(merged),
                   "strata": table, "sampled": len(merged), "stopped_at_16": sum(m["n48"] == 0 for m in merged)}
    return out


def show(new: list[Rec], facts: dict[str, list[dict[str, Any]]]) -> None:
    rng = random.Random(f"{SEED}/read")
    for m in rng.sample(new, min(N_READ, len(new))):
        it = facts[m["fact"]][0]
        print(f"\n=== {m['fact']} ({m['stratum']}, {m['date']}); gold {it['answer']!r}\n{it['user'].split(chr(10))[0]}\n"
              f"commonest samples: {Counter(a for a in m['a16'] + m['a48'] if a).most_common(4)}", flush=True)


def main(candidates: Path, merged_path: Path, out: Path) -> None:
    items = json.loads(candidates.read_text())
    with gzip.open(merged_path, "rt") as f:
        merged = json.load(f)
    res = build(items, merged)
    out.write_text(json.dumps(res, indent=1))
    meta = res["meta"]
    pool_new = {i["fact"] for i in res["new"]["held"]}
    facts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        facts[it["fact"]].append(it)
    show([m for m in merged if m["fact"] in pool_new], facts)
    print(f"\n[pools] n {meta['n']} per side; guard cuts {meta['guard_cuts']}; changed facts {meta['changed_count']} "
          f"(n if kept {meta['n_if_changed_kept']}); strata {len(meta['strata'])}", flush=True)
    print(f"STAGE 2 DONE {out}", flush=True)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=here / "results" / "large" / "candidates.json")
    ap.add_argument("--merged", type=Path, default=here / "results" / "large" / "stage1" / "merged.json.gz")
    ap.add_argument("--out", type=Path, default=here / "results" / "pools.json")
    a = ap.parse_args()
    main(a.candidates, a.merged, a.out)

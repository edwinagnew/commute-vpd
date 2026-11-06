"""E120 stage 2 (CPU, $0): the known and never-solved pools from stage 1's samples, and novelty check N2. Plan:
reports/plans/2026-11-06_plan_e120-pass-rate-pools.md, "Stage 2" and "Novelty checks".

Labels, read on the 48 fresh samples: new = 0 of 16 and 0 of 48 (never solved in 64); known = >= KNOWN_BAR of 48, or
>= KNOWN_FALLBACK of 48 in a MATH level-5 stratum whose known at KNOWN_BAR are fewer than its new (the stratum is
named in meta.fallback). Out of both sides: prompt + gold over the SFT window; >= TRUNC_GUARD of 64 truncated (unread).
Out of the new side: >= WRONG_GUARD of 64 agreeing on one non-gold answer (gold-error guard). Gold the grader cannot
read was dropped before sampling (stage1.json gold_unreadable). Strata (source, level, subject), GSM8K one stratum;
per stratum m = min(new, known), each side drawn m by a seeded shuffle, the rest cut and counted; HOLD of the m pairs
held out (rounded half up), so both pools have the same n, train and held sizes.
N2: on known-pool problems, the longest run of exact words shared by a saved correct sample and the problem's gold,
against the same with the gold of another known problem of the stratum (seeded; of the source when the stratum has
one); fails when the share of problems with a run of >= N2_RUN words exceeds the reference's by more than N2_MARGIN.
Writes results/pools.json: {"meta", "known": {"train", "held"}, "new": {"train", "held"}}, items {id, source, level,
subject, user, gold_response, answer}; prints 20 new-pool problems with gold for the lead to read.

CPU (repo root): fleet/cpu-run --mem 6G python experiments/qwen_pass_pools/pools.py [--dl DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from problems import (
    DL,
    RESULTS,
    SEQ_LEN,
    Item,
    load_all,
    read_json,
    rel,
    stratum,
    write_json,
)

SEED = 0
KNOWN_BAR, KNOWN_FALLBACK, WRONG_GUARD, TRUNC_GUARD, HOLD = 36, 24, 32, 16, 0.2
N2_RUN, N2_MARGIN, N_READ = 20, 0.10, 20
FIELDS = ("id", "source", "level", "subject", "user", "gold_response", "answer")
RULES = {"new": "k16 == 0 and k48 == 0 (n48 == 48)", "known": f"k48 >= {KNOWN_BAR} (fallback {KNOWN_FALLBACK}, MATH "
         "level-5 strata only)", "guards_both": f"seq_tokens > {SEQ_LEN}; t16 + t48 >= {TRUNC_GUARD}",
         "guard_new": f"top_wrong >= {WRONG_GUARD}", "hold": HOLD, "n2": f"run >= {N2_RUN} words, margin {N2_MARGIN}"}


def cuts(m: dict) -> list[str]:
    out = ["long"] if m["seq_tokens"] > SEQ_LEN else []
    return out + (["truncated"] if m["t16"] + m["t48"] >= TRUNC_GUARD else [])


def label(m: dict) -> str | None:
    """A stage-1 record's side before the guards: new, known (at the fallback bar or over), or None."""
    if m["n48"] != 48:
        return None
    return "new" if m["k16"] + m["k48"] == 0 else "known" if m["k48"] >= KNOWN_FALLBACK else None


def sides(merged: list[dict], cut: Counter[str]) -> dict[str, dict[str, list[dict]]]:
    """Per stratum, {"new": [...], "known36": [...], "known24": [...]} after the guards; `cut` counts each removal."""
    by: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"new": [], "known36": [], "known24": []})
    for m in sorted(merged, key=lambda m: m["id"]):
        side = label(m)
        if side is None:
            continue
        why = cuts(m) + (["gold_error"] if side == "new" and (m["top_wrong"] or 0) >= WRONG_GUARD else [])
        for w in why:
            cut[f"{side}/{w}"] += 1
        if why:
            continue
        s = by[stratum(m)]
        if side == "new":
            s["new"].append(m)
        else:
            s["known24"].append(m)
            if m["k48"] >= KNOWN_BAR:
                s["known36"].append(m)
    return by


def match(by: dict[str, dict[str, list[dict]]]) -> tuple[dict[str, dict[str, list[dict]]], dict[str, Any]]:
    """1:1 seeded matching and the stratified hold-out: ({"known"|"new": {"train", "held"}}, per-stratum table)."""
    pools: dict[str, dict[str, list[dict]]] = {s: {"train": [], "held": []} for s in ("known", "new")}
    table = {}
    for s, d in sorted(by.items()):
        fallback = s.startswith("math/5/") and len(d["known36"]) < len(d["new"])
        known = d["known24"] if fallback else d["known36"]
        m = min(len(d["new"]), len(known))
        rng = random.Random(f"{SEED}/{s}")
        drawn = {"new": rng.sample(d["new"], len(d["new"]))[:m], "known": rng.sample(known, len(known))[:m]}
        h = int(HOLD * m + 0.5)
        for side, rs in drawn.items():
            pools[side]["held"] += rs[:h]
            pools[side]["train"] += rs[h:]
        table[s] = {"new": len(d["new"]), "known": len(known), "matched": m, "held": h, "fallback": fallback,
                    "cut_new_unmatched": len(d["new"]) - m, "cut_known_unmatched": len(known) - m}
    return pools, table


def run_len(a: list[str], b: list[str]) -> int:
    """The longest run of consecutive words common to a and b."""
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def n2(known: list[Item], texts: dict[str, list[str]]) -> dict[str, Any]:
    """N2 over the known pool (train and held); problems with no saved correct sample, or no reference, are counted."""
    rng = random.Random(f"{SEED}/n2")
    by_s: dict[str, list[Item]] = defaultdict(list)
    for it in known:
        by_s[stratum(it)].append(it)
    own, ref, skipped = [], [], 0
    for it in known:
        others = [o for o in by_s[stratum(it)] if o["id"] != it["id"]] or \
                 [o for o in known if o["source"] == it["source"] and o["id"] != it["id"]]
        samples = texts.get(it["id"], [])
        if not samples or not others:
            skipped += 1
            continue
        other = rng.choice(others)
        own.append(max(run_len(words(t), words(it["gold_response"])) for t in samples))
        ref.append(max(run_len(words(t), words(other["gold_response"])) for t in samples))
    if not own:
        return {"read": 0, "skipped": skipped, "verdict": "does not apply"}
    s_own, s_ref = sum(r >= N2_RUN for r in own) / len(own), sum(r >= N2_RUN for r in ref) / len(ref)
    return {"read": len(own), "skipped": skipped, "share_own": s_own, "share_ref": s_ref, "diff": s_own - s_ref,
            "median_own": sorted(own)[len(own) // 2], "median_ref": sorted(ref)[len(ref) // 2],
            "verdict": "fail (memorised gold)" if s_own - s_ref > N2_MARGIN else "pass"}


def show(new: list[dict], items: dict[str, Item]) -> None:
    rng = random.Random(f"{SEED}/read")
    for m in rng.sample(new, min(N_READ, len(new))):
        it = items[m["id"]]
        common = Counter(a for a in m["a16"] + m["a48"] if a is not None).most_common(1)
        print(f"\n=== {m['id']} ({stratum(m)}); gold answer {it['answer']!r}; commonest sample answer "
              f"{common[0] if common else None}; top_wrong {m['top_wrong']}\n--- problem\n{it['user']}\n--- gold\n"
              f"{it['gold_response']}", flush=True)


def build(items: dict[str, Item], merged: list[dict], texts: dict[str, list[str]], unread: int | None) -> dict[str, Any]:
    cut: Counter[str] = Counter()
    pools, table = match(sides(merged, cut))
    out: dict[str, Any] = {s: {p: [{k: items[m["id"]][k] for k in FIELDS} for m in rs] for p, rs in d.items()}
                           for s, d in pools.items()}
    n = {s: {p: len(rs) for p, rs in d.items()} for s, d in pools.items()}
    assert n["known"] == n["new"], n
    # Amendment 1: never-solved (0 of 64) problems the truncation guard removed, per stratum (bears on P1).
    guarded = Counter(stratum(m) for m in merged
                      if m["n48"] == 48 and m["k16"] + m["k48"] == 0 and m["t16"] + m["t48"] >= TRUNC_GUARD)
    out["meta"] = {"never_solved_removed_by_truncation_guard": {"total": sum(guarded.values()),
                                                                 "per_stratum": dict(sorted(guarded.items()))},"seed": SEED, "rules": RULES, "n": sum(n["new"].values()), "sizes": n, "gold_unreadable": unread,
                   "problems_sampled": len(merged), "stopped_at_16": sum(m["n48"] == 0 for m in merged),
                   "guard_cuts": dict(sorted(cut.items())), "strata": table,
                   "fallback": [s for s, t in table.items() if t["fallback"]],
                   "n2": n2([i for p in out["known"].values() for i in p], texts)}
    return out


def main(dl: Path, out: Path) -> None:
    probs, _, _ = load_all()
    items = {it["id"]: it for it in probs}
    merged = read_json(dl / "stage1_problems.json.gz")
    texts: dict[str, list[str]] = {}
    for f in sorted((dl / "s48").glob("texts_*.json.gz")):
        texts |= read_json(f)
    summary = dl.parent / "stage1.json"
    unread = read_json(summary)["gold_unreadable"] if summary.is_file() else None
    res = build(items, merged, texts, unread)
    show([m for m in merged if m["id"] in {i["id"] for p in res["new"].values() for i in p}], items)
    meta = res["meta"]
    print(f"\n[pools] n {meta['n']} per side {meta['sizes']}; guard cuts {meta['guard_cuts']}; fallback strata "
          f"{meta['fallback']}; N2 {meta['n2']}", flush=True)
    g = meta["never_solved_removed_by_truncation_guard"]
    print(f"[pools] never solved in 64 but removed by the truncation guard (>= {TRUNC_GUARD} of 64 truncated): "
          f"{g['total']}; per stratum {g['per_stratum']}", flush=True)
    print(f"STAGE 2 DONE {rel(out)} {write_json(out, res)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", type=Path, default=DL)
    ap.add_argument("--out", type=Path, default=RESULTS / "pools.json")
    a = ap.parse_args()
    main(a.dl, a.out)

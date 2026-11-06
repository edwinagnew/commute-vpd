"""E120 Amendment 2 (2026-11-06; plan reports/plans/2026-11-06_plan_e120-pass-rate-pools.md, "Amendment 2"), CPU, $0:
the MATH grader with presentation stripped (`grade.present`) on stage 1's stored answers, then stage 2 rebuilt.

  1. Stage 0's grader controls (`grade.control`, as `pod.prepare` ran them) on the new grader: gold against its own
     solution 1.000 per source, MATH gold against the next gold <= 0.05, and the same unreadable golds as stage0.json
     (they set which problems were sampled). A miss stops here (exit 2), before the rebuild.
  2. Every pass-16 and pass-48 shard record regraded (the problem text given, for choice letters), and stage 1's record
     per problem rebuilt by `stage1.merge` (pass counts and the gold-error guard's count). Checks: the stored records
     rebuild the stored stage1_problems.json.gz; every GSM8K record is unchanged. Which problems got pass 48 is as
     sampled (stage 1 chose them on the old grader's pass-16 counts).
  3. Stage 2 by `pools.build`, every guard unchanged.
Writes results/stage0_amend2.json, results/stage1_amend2.json, results/pools_amend2.json and
<dl>/stage1_problems_amend2.json.gz; stage0.json, stage1.json, pools.json and predictions.json are only read.

From the repo root, with math-verify (the pod's 0.9.0) taken from its CPU venv: MVLIB is a folder of links to that
venv's math_verify, latex2sympy2_extended and antlr4 (sympy 1.14.0 is the same in both venvs); as run 2026-11-06,
MVLIB = ~/commute-data/scratch/qwen-rl-lead/e120-a2-writer/mvlib:
  PYTHONPATH=MVLIB fleet/cpu-run -c 8 --mem 8G -t 01:00:00 python experiments/qwen_pass_pools/regrade_amend2.py --dl DIR
DIR is the clone's results/pod_dl (the default) or, from a worktree, the main clone's.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pools
import stage1
from grade import control, grade_one, pmap
from problems import DL, RESULTS, Item, load_all, read_json, rel, write_json

from commute.jobs import hard_exit

NEXT_GOLD_MAX = 0.05  # pod.NEG_MAX
# The review's 16 MATH grader misses and the one among the kept 78 (reports/reviews/2026-11-06_review_e120-gold-error-
# guard.md; scratch guard-check/labels.json, class "grader miss").
NAMED = ("math/counting_and_probability/634", "math/algebra/409", "math/number_theory/8", "math/number_theory/775",
         "math/precalculus/57", "math/precalculus/5", "math/number_theory/662", "math/geometry/611",
         "math/intermediate_algebra/780", "math/prealgebra/900", "math/counting_and_probability/484",
         "math/number_theory/445", "math/prealgebra/574", "math/counting_and_probability/439",
         "math/counting_and_probability/238", "math/prealgebra/928", "math/number_theory/654")


def controls(probs: list[Item], test: list[Item]) -> dict[str, Any]:
    c = control([(it["id"], it["source"], it["answer"], it["gold_response"]) for it in probs + test], stage1.procs())
    old = read_json(RESULTS / "stage0.json")["c"]
    checks = {"gsm8k_gold": c["gsm8k_gold"] == 1.0, "math_gold": c["math_gold"] == 1.0,
              "math_next_gold": c["math_next_gold"] is not None and c["math_next_gold"] <= NEXT_GOLD_MAX,
              "unreadable_unchanged": all(c[k] == old[k] for k in ("gsm8k_unreadable", "math_unreadable"))}
    out = c | {"checks": checks, "stage0_c": {k: v for k, v in old.items() if not isinstance(v, list)}}
    print(f"[stage 0 c, new grader] gsm8k_gold {c['gsm8k_gold']}, math_gold {c['math_gold']}, math_next_gold "
          f"{c['math_next_gold']} (stage0.json {out['stage0_c']}); checks {checks}", flush=True)
    print(f"STAGE 0 AMEND2 DONE {rel(RESULTS / 'stage0_amend2.json')} {write_json(RESULTS / 'stage0_amend2.json', out)}",
          flush=True)
    if not all(checks.values()):
        print("STOPPED: a grader control missed; no rebuild", flush=True)
        hard_exit(2)
    return out


def shards(dl: Path, n: int) -> list[dict]:
    return [r for f in sorted((dl / f"s{n}").glob("shard_*.json.gz")) for r in read_json(f)["problems"]]


def regrade(recs: list[dict], items: dict[str, Item]) -> list[dict]:
    jobs = [(items[r["id"]]["source"], items[r["id"]]["answer"], r["answers"], items[r["id"]]["user"]) for r in recs]
    return [r | {"k": sum(c)} for r, c in zip(recs, pmap(grade_one, jobs, stage1.procs()), strict=True)]


def counts(ms: list[dict]) -> dict[str, tuple]:
    return {m["id"]: (m["k16"], m["k48"], m["n48"], m["t16"], m["t48"], m["a16"], m["a48"]) for m in ms}


def row(o: dict, m: dict) -> dict[str, Any]:
    return {"id": m["id"], "k16": [o["k16"], m["k16"]], "k48": [o["k48"], m["k48"]], "n48": m["n48"],
            "label": [pools.label(o), pools.label(m)], "top_wrong": [o["top_wrong"], m["top_wrong"]]}


def stage1_amend(dl: Path, items: dict[str, Item]) -> list[dict]:
    old = read_json(dl / "stage1_problems.json.gz")
    r16, r48 = shards(dl, 16), shards(dl, 48)
    assert counts(stage1.merge(items, r16, r48)) == counts(old), "the shards do not rebuild stage1_problems.json.gz"
    new = stage1.merge(items, regrade(r16, items), regrade(r48, items))
    by_old = {m["id"]: m for m in old}
    gsm8k_same = [m for m in new if m["source"] == "gsm8k"] == [m for m in old if m["source"] == "gsm8k"]
    assert gsm8k_same, "GSM8K records changed under the new grader"
    changed = [row(by_old[m["id"]], m) for m in new if m != by_old[m["id"]]]
    per = {src: {"answers_correct": [sum(m["k16"] + m["k48"] for m in ms if m["source"] == src) for ms in (old, new)],
                 "labels": {side: [sum(pools.label(m) == side and m["source"] == src for m in ms) for ms in (old, new)]
                            for side in ("new", "known")}} for src in ("gsm8k", "math")}
    named = {i: row(by_old[i], m) for m in new if (i := m["id"]) in NAMED}
    out = {"merged": rel(dl / "stage1_problems_amend2.json.gz"),
           "merged_sha256": write_json(dl / "stage1_problems_amend2.json.gz", new), "gsm8k_unchanged": gsm8k_same,
           "before_after": per, "label_moves": dict(Counter(f"{c['label'][0]}->{c['label'][1]}" for c in changed)),
           "named_misses_now_solved": sum(r["k16"][1] + r["k48"][1] > 0 for r in named.values()),
           "named": named, "changed": changed}
    print(f"[stage 1, new grader] before -> after per source {per}; label moves {out['label_moves']}; "
          f"named misses now solved {out['named_misses_now_solved']} of {len(NAMED)}; {len(changed)} records changed",
          flush=True)
    for r in named.values():
        print(f"  {r}", flush=True)
    print(f"STAGE 1 AMEND2 DONE {rel(RESULTS / 'stage1_amend2.json')} {write_json(RESULTS / 'stage1_amend2.json', out)}",
          flush=True)
    return new


def split(p: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {s: dict(sorted(Counter(i["source"] for part in ("train", "held") for i in p[s][part]).items()))
            for s in ("known", "new")}


def stage2_amend(dl: Path, items: dict[str, Item], merged: list[dict]) -> None:
    texts: dict[str, list[str]] = {}
    for f in sorted((dl / "s48").glob("texts_*.json.gz")):
        texts |= read_json(f)
    res = pools.build(items, merged, texts, read_json(RESULTS / "stage1.json")["gold_unreadable"])
    old = read_json(RESULTS / "pools.json")
    keys = ("n", "sizes", "guard_cuts", "never_solved_removed_by_truncation_guard")
    res["meta"]["amendment_2"] = {"split": split(res), "before": {k: old["meta"][k] for k in keys} | {"split": split(old)},
                                  "stage1": rel(RESULTS / "stage1_amend2.json")}
    pools.show([m for m in merged if m["id"] in {i["id"] for p in res["new"].values() for i in p}], items)
    meta = res["meta"]
    print(f"\n[pools, new grader] n {meta['n']} per side (was {old['meta']['n']}); split {split(res)} (was "
          f"{split(old)}); guard cuts {meta['guard_cuts']} (was {old['meta']['guard_cuts']}); never solved removed by "
          f"truncation {meta['never_solved_removed_by_truncation_guard']['total']} (was "
          f"{old['meta']['never_solved_removed_by_truncation_guard']['total']}); fallback {meta['fallback']}; N2 "
          f"{meta['n2']}", flush=True)
    print(f"STAGE 2 AMEND2 DONE {rel(RESULTS / 'pools_amend2.json')} {write_json(RESULTS / 'pools_amend2.json', res)}",
          flush=True)


def main(dl: Path) -> None:
    probs, test, _ = load_all()
    controls(probs, test)
    stored = {m["id"]: m for m in read_json(dl / "stage1_problems.json.gz")}
    items = {it["id"]: it | {"seq_tokens": stored[it["id"]]["seq_tokens"]} for it in probs if it["id"] in stored}
    assert all(items[i]["answer"] == m["answer"] for i, m in stored.items()), "gold answers differ from stage 1's"
    stage2_amend(dl, items, stage1_amend(dl, items))
    hard_exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", type=Path, default=DL)
    a = ap.parse_args()
    main(a.dl)

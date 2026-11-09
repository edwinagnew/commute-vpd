"""E128 stage 0 (`sample.py --control`; plan, "Stage 0"): nothing else runs until it passes. Cheapest first: (b) the grader
on the candidates (a CPU check, every gold reads correct 1.000, another fact's gold of the same relation at most 0.05); (a)
E45's control on this path, `qwen_facts.held_set` greedy, bf16, known in [0.93, 1.00] (E45 read 0.97), invented at most
0.03; the known side's first 100 candidates greedy, the yield read of Fable (e)4 (no band); (c) GSM8K test 0-255 strict
in [0.72, 0.84] on E61's fp32 path. Stops on a miss (exit 2, stage0.json names it). A missing candidates file skips
(b) and the yield read and is noted: a GPU test of this file before the pools exist.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import grade
from sample_common import CAP, SEED, Gen, Item, Paths, facts_of, side_of, stop, write

from commute import qwen_facts
from commute.qwen_eval import GREEDY, chat_prompt

KNOWN_BAND, INVENTED_MAX, OTHER_GOLD_MAX, GSM8K_BAND, TRUNC_MAX, YIELD_N = (0.93, 1.0), 0.03, 0.05, (0.72, 0.84), 0.02, 100


def grader_checks(facts: dict[str, list[Item]]) -> dict[str, Any]:
    """Every gold, read as a response, is correct; another fact's gold of the same relation (seeded) is not."""
    fids = sorted(facts)
    random.Random(f"{SEED}/grader").shuffle(fids)
    by_rel: dict[str, list[str]] = defaultdict(list)
    for f in fids:
        by_rel[facts[f][0]["subject"]].append(f)
    gold_ok, other_ok = [], []
    for fs in by_rel.values():
        for f, other in zip(fs, fs[1:] + fs[:1], strict=True):
            golds = grade.golds_of(facts[f][0])
            gold_ok.append(grade.read_sample(facts[f][0]["gold_response"], False, golds)[1])
            other_ok.append(grade.read_sample(facts[other][0]["gold_response"], False, golds)[1])
    return {"gold": statistics.mean(gold_ok), "other_fact_gold": statistics.mean(other_ok)}


def facts_greedy(tok: Any, gen: Gen, qas: list[Any], key: str) -> list[dict[str, Any]]:
    recs = gen([chat_prompt(tok, q.question) for q in qas], 1, GREEDY, CAP, key)
    return [{**r, "correct": qwen_facts.correct(r["response"], q.fact.answer), "known": q.fact.known}
            for q, r in zip(qas, recs, strict=True)]


def e45_facts(tok: Any, gen: Gen) -> dict[str, Any]:
    """Known facts from draw 0's held-out set; invented facts from draws 0 and 1 (E45's `facts_eval`, E45's held template)."""
    draw0 = facts_greedy(tok, gen, qwen_facts.held_set(0), "facts-0")
    known = [r for r in draw0 if r["known"]]
    return {"known": statistics.mean(r["correct"] for r in known),
            "known_format_share": statistics.mean(qwen_facts.extract(r["response"]) is not None for r in known),
            "known_longest_tokens": max(r["response_tokens"] for r in known),
            "known_truncated": statistics.mean(r["truncated"] for r in known),
            "invented_s0": statistics.mean(r["correct"] for r in draw0 if not r["known"]),
            "invented_s1": statistics.mean(r["correct"] for r in facts_greedy(tok, gen, qwen_facts.held_set(1), "facts-1")
                                           if not r["known"])}


def known_yield(tok: Any, gen: Gen, facts: dict[str, list[Item]]) -> dict[str, Any]:
    """Fable (e)4: greedy on template 0 of YIELD_N known-side candidates, taken in turn from each relation by sitelinks."""
    rels: dict[str, list[list[Item]]] = defaultdict(list)
    for its in sorted((v for v in facts.values() if side_of(v) == "known"), key=lambda s: -s[0]["links"]):
        rels[its[0]["subject"]].append(its)
    picked: list[list[Item]] = []
    depth = 0
    while len(picked) < YIELD_N and any(len(v) > depth for v in rels.values()):
        picked += [v[depth] for v in rels.values() if len(v) > depth]
        depth += 1
    picked = picked[:YIELD_N]
    recs = gen([chat_prompt(tok, its[0]["user"]) for its in picked], 1, GREEDY, CAP, "yield")
    ok = [grade.read_sample(r["response"], r["truncated"], grade.golds_of(its[0]))[1] for r, its in zip(recs, picked, strict=True)]
    per_rel = Counter(its[0]["subject"] for its, c in zip(picked, ok, strict=True) if c)
    return {"n": len(picked), "correct_share": statistics.mean(ok) if ok else None, "correct_per_relation": dict(sorted(per_rel.items())),
            "examples": [{"fact": its[0]["fact"], "q": its[0]["user"].split("\n")[0], "gold": its[0]["answer"], "got": r["response"]}
                         for its, r in list(zip(picked, recs, strict=True))[:10]]}


def stage0(paths: Paths, tok: Any, gen: Gen, gsm8k: Callable[[], dict[str, Any]], facts_file: bool) -> None:
    out: dict[str, Any] = {"cap": CAP, "bands": {"known": KNOWN_BAND, "invented_max": INVENTED_MAX,
                                                  "other_gold_max": OTHER_GOLD_MAX, "gsm8k": GSM8K_BAND}}
    facts = facts_of(json.loads(paths.candidates.read_text())) if facts_file else None
    out["candidates_checked"] = facts is not None
    if facts is None:
        print(f"[stage 0] no candidates file at {paths.candidates}: (b) and the yield read are skipped", flush=True)
    else:
        out["b"] = grader_checks(facts)
        print(f"[stage 0] (b) {out['b']}", flush=True)
        if out["b"]["gold"] != 1.0:
            stop(paths.stage0, out, f"gold grader {out['b']['gold']} is not 1.000")
        if out["b"]["other_fact_gold"] > OTHER_GOLD_MAX:
            stop(paths.stage0, out, f"another fact's gold reads correct {out['b']['other_fact_gold']}, over {OTHER_GOLD_MAX}")
    out["a"] = a = e45_facts(tok, gen)
    print(f"[stage 0] (a) {a}", flush=True)
    if not KNOWN_BAND[0] <= a["known"] <= KNOWN_BAND[1]:
        stop(paths.stage0, out, f"known facts {a['known']} outside {KNOWN_BAND}")
    if max(a["invented_s0"], a["invented_s1"]) > INVENTED_MAX:
        stop(paths.stage0, out, f"invented facts over {INVENTED_MAX}")
    if a["known_truncated"] > TRUNC_MAX:
        stop(paths.stage0, out, f"{a['known_truncated']} of greedy known-fact answers truncated at cap {CAP}")
    if facts is not None:
        out["known_yield"] = known_yield(tok, gen, facts)
        print(f"[stage 0] known-side yield (greedy, template 0): n {out['known_yield']['n']}, share "
              f"{out['known_yield']['correct_share']}, per relation {out['known_yield']['correct_per_relation']}", flush=True)
    write(paths.stage0, out)
    out["gsm8k"] = gsm8k()
    print(f"[stage 0] (c) GSM8K 0-255 {out['gsm8k']}", flush=True)
    if not GSM8K_BAND[0] <= out["gsm8k"]["strict_dropped"] <= GSM8K_BAND[1]:
        stop(paths.stage0, out, f"GSM8K {out['gsm8k']['strict_dropped']} outside {GSM8K_BAND}")
    out["passed"] = True
    write(paths.stage0, out)
    print(f"STAGE 0 DONE {paths.stage0}", flush=True)

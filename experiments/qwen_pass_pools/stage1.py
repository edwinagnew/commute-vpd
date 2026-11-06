"""E120 stage 1: adaptive sampling of the unchanged model, no-think, `qwen_eval.SAMPLE` (plan, "Stage 1").

Pass 16: 16 samples for every problem whose gold the grader reads, in one seeded order, in shards of SHARD16 problems.
Pass 48: 48 fresh samples for every 0-of-16 problem and, per stratum, for up to KNOWN_PER_NEW known candidates per
0-of-16 problem (the stratum's problems at >= CAND_MIN of 16, most solved first, ties in a seeded order). Problems
whose prompt + gold exceed the SFT window, or with all 16 samples truncated, enter neither pool and get no pass 48. If
the first shard reads over PLAIN64_TPS tokens/s, every eligible problem gets pass 48 (the plain 64).
Each shard is a file under results/pod_dl/s16|s48/, printed `SHARD DONE <path> <sha256>` once written; a shard already
on disk is read, not resampled, so a lost pod loses at most one shard. Texts are kept only where a later step reads them:
every sample of a 0-of-16 problem (the new side) and up to KEEP_CORRECT correct samples of the others in pass 48 (N2).
Early checks, printed every shard and applied to the first: truncation GSM8K <= 0.02, MATH <= 0.10 (else the first
shard is redone at the next MATH cap; past the last cap, stop); after pass 16, MATH level 5's 0-of-16 share >= 0.05.
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from engine import Engine, generate, tps
from grade import answer_of, grade_one, pmap, top_wrong_one
from problems import DL, RESULTS, SEQ_LEN, Item, read_json, rel, stratum, write_json

from commute.qwen_eval import MAX_NEW, SAMPLE

SEED, SHARD16, SHARD48 = 0, 512, 192
KNOWN_PER_NEW, CAND_MIN, KEEP_CORRECT = 3, 8, 4
PLAIN64_TPS = 20_000.0
TRUNC_MAX = {"gsm8k": 0.02, "math": 0.10}
MATH_CAPS = (1024, 2048)
L5_MIN = 0.05
SUMMARY, MERGED = RESULTS / "stage1.json", DL / "stage1_problems.json.gz"


def procs() -> int:
    return max(1, min(32, (os.cpu_count() or 2) - 2))


def stop(summary: dict[str, Any], why: str) -> None:
    summary["stopped"] = why
    print(f"STAGE 1 STOPPED: {why}  (stage1.json sha256 {write_json(SUMMARY, summary)})", flush=True)
    sys.exit(2)


def run_shard(eng: Engine, items: list[Item], n: int, cap_math: int, keep_all: set[str]) -> tuple[dict, dict]:
    """Samples, grades and keeps texts for one shard: (shard record, {id: [kept texts]})."""
    caps = [MAX_NEW if it["source"] == "gsm8k" else cap_math for it in items]
    tok0, s0 = eng.tokens, eng.seconds
    outs = generate(eng, [it["user"] for it in items], n, SAMPLE, caps)
    answers = [[None if s["truncated"] else answer_of(it["source"], s["text"]) for s in o] for it, o in zip(items, outs)]
    ok = pmap(grade_one, [(it["source"], it["answer"], a) for it, a in zip(items, answers)], procs())
    probs, texts = [], {}
    for it, o, a, c in zip(items, outs, answers, ok, strict=True):
        probs.append({"id": it["id"], "n": n, "k": sum(c), "t": sum(s["truncated"] for s in o), "answers": a})
        if it["id"] in keep_all or (n == 16 and sum(c) == 0):
            texts[it["id"]] = [s["text"] for s in o]
        elif n == 48:
            texts[it["id"]] = [s["text"] for s, ci in zip(o, c) if ci][:KEEP_CORRECT]
    trunc = {src: [p["t"] / n for p, it in zip(probs, items) if it["source"] == src] for src in TRUNC_MAX}
    meta = {"n": n, "cap_math": cap_math, "problems": len(items), "tokens": eng.tokens - tok0,
            "seconds": eng.seconds - s0, "tps": (eng.tokens - tok0) / max(eng.seconds - s0, 1e-9),
            "truncation": {s: sum(v) / len(v) if v else None for s, v in trunc.items()}}
    return {"meta": meta, "problems": probs}, texts


def shard(eng: Engine, items: list[Item], n: int, i: int, cap_math: int, keep_all: set[str],
          redo: bool = False) -> dict:
    path, tpath = DL / f"s{n}" / f"shard_{i:04d}.json.gz", DL / f"s{n}" / f"texts_{i:04d}.json.gz"
    if path.is_file() and not redo:
        rec = read_json(path)
        print(f"[s{n}] shard {i}: read from disk ({len(rec['problems'])} problems)", flush=True)
        return rec
    rec, texts = run_shard(eng, items, n, cap_math, keep_all)
    print(f"SHARD DONE {rel(tpath)} {write_json(tpath, texts)}", flush=True)
    print(f"SHARD DONE {rel(path)} {write_json(path, rec)}", flush=True)
    m = rec["meta"]
    print(f"[s{n}] shard {i}: {m['problems']} problems, {m['seconds']:.0f} s, {m['tps']:.0f} tokens/s, truncation "
          f"{m['truncation']}, cap_math {cap_math}; running {tps(eng):.0f} tokens/s", flush=True)
    return rec


def first_shard(eng: Engine, items: list[Item], summary: dict[str, Any]) -> dict:
    """Shard 0 of pass 16 with the truncation checks; the MATH cap rises once if MATH truncation is over its bar."""
    rec = shard(eng, items, 16, 0, MATH_CAPS[0], set())
    while True:
        tr, cap = rec["meta"]["truncation"], rec["meta"]["cap_math"]
        print(f"[check] first shard truncation GSM8K {tr['gsm8k']} (<= {TRUNC_MAX['gsm8k']}), MATH {tr['math']} "
              f"(<= {TRUNC_MAX['math']}) at cap {cap}; {rec['meta']['tps']:.0f} tokens/s "
              f"(plain 64 above {PLAIN64_TPS:.0f})", flush=True)
        if (tr["gsm8k"] or 0) > TRUNC_MAX["gsm8k"]:
            stop(summary, f"GSM8K truncation {tr['gsm8k']} over {TRUNC_MAX['gsm8k']}")
        if (tr["math"] or 0) <= TRUNC_MAX["math"]:
            return rec
        if cap == MATH_CAPS[-1]:
            stop(summary, f"MATH truncation {tr['math']} over {TRUNC_MAX['math']} at the last cap {cap}")
        rec = shard(eng, items, 16, 0, MATH_CAPS[MATH_CAPS.index(cap) + 1], set(), redo=True)


def l5_share(items: dict[str, Item], recs: list[dict]) -> float | None:
    l5 = [r for r in recs if items[r["id"]]["level"] == 5]
    return sum(r["k"] == 0 for r in l5) / len(l5) if l5 else None


def eligible(it: Item, r: dict) -> bool:
    return it["seq_tokens"] <= SEQ_LEN and r["t"] < r["n"]


def select(items: dict[str, Item], recs: list[dict], plain64: bool) -> tuple[list[str], set[str]]:
    """(ids for pass 48 in a fixed order, the 0-of-16 ids among them)."""
    ok = [r for r in recs if eligible(items[r["id"]], r)]
    zero = {r["id"] for r in ok if r["k"] == 0}
    if plain64:
        return sorted(r["id"] for r in ok), zero
    by: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by[stratum(items[r["id"]])].append(r)
    picked = set(zero)
    for s, rs in sorted(by.items()):
        c = sum(r["k"] == 0 for r in rs)
        rng = random.Random(f"{SEED}/{s}")
        cands = sorted((r for r in rs if r["k"] >= CAND_MIN), key=lambda r: (-r["k"], rng.random()))
        picked |= {r["id"] for r in cands[: KNOWN_PER_NEW * c]}
    return sorted(picked), zero


def merge(items: dict[str, Item], r16: list[dict], r48: list[dict]) -> list[dict]:
    """One record per problem: its counts on pass 16 and on the fresh 48 (n48 0 where it stopped at 16), its answers,
    and the gold-error guard's count over all its samples (problems never solved only; None otherwise)."""
    by48 = {r["id"]: r for r in r48}
    out = []
    for r in r16:
        it, f = items[r["id"]], by48.get(r["id"], {"n": 0, "k": 0, "t": 0, "answers": []})
        out.append({**{k: it[k] for k in ("id", "source", "level", "subject", "answer", "seq_tokens")},
                    "n16": r["n"], "k16": r["k"], "t16": r["t"], "a16": r["answers"],
                    "n48": f["n"], "k48": f["k"], "t48": f["t"], "a48": f["answers"], "top_wrong": None})
    never = [o for o in out if o["n48"] and o["k16"] + o["k48"] == 0]
    for o, tw in zip(never, pmap(top_wrong_one, [(o["source"], o["answer"], o["a16"] + o["a48"]) for o in never],
                                 procs())):
        o["top_wrong"] = tw
    return out


def run(eng: Engine, probs: list[Item]) -> Path:
    items = {it["id"]: it for it in probs if it["gold_ok"]}
    order = sorted(items)
    random.Random(SEED).shuffle(order)
    chunks = [[items[i] for i in order[s : s + SHARD16]] for s in range(0, len(order), SHARD16)]
    summary: dict[str, Any] = {"problems": len(probs), "gold_unreadable": len(probs) - len(items), "seed": SEED}
    first = first_shard(eng, chunks[0], summary)
    cap, plain64 = first["meta"]["cap_math"], first["meta"]["tps"] > PLAIN64_TPS
    summary |= {"cap_math": cap, "first_tps": first["meta"]["tps"], "plain64": plain64}
    recs16 = list(first["problems"])
    for i, ch in enumerate(chunks[1:], 1):
        recs16 += shard(eng, ch, 16, i, cap, set())["problems"]
        print(f"[s16] running MATH level-5 0-of-16 share {l5_share(items, recs16)}", flush=True)
    share = summary["l5_zero16_share"] = l5_share(items, recs16)
    print(f"[check] MATH level-5 0-of-16 share {share} (stop under {L5_MIN})", flush=True)
    if share is None or share < L5_MIN:
        stop(summary, f"MATH level-5 0-of-16 share {share} under {L5_MIN}: no new pool to be had")
    sel, zero = select(items, recs16, plain64)
    summary |= {"pass48": len(sel), "zero16_eligible": len(zero)}
    recs48 = []
    for i, s in enumerate(range(0, len(sel), SHARD48)):
        recs48 += shard(eng, [items[j] for j in sel[s : s + SHARD48]], 48, i, cap, zero)["problems"]
    merged = merge(items, recs16, recs48)
    summary["merged_sha256"] = write_json(MERGED, merged)
    summary |= {"stopped_at_16": sum(m["n48"] == 0 for m in merged), "tokens": eng.tokens, "seconds": eng.seconds,
                "per_stratum": dict(sorted(Counter(stratum(items[m["id"]]) for m in merged).items()))}
    print(f"SHARD DONE {rel(MERGED)} {summary['merged_sha256']}", flush=True)
    print(f"STAGE 1 DONE {rel(SUMMARY)} {write_json(SUMMARY, summary)}", flush=True)
    return MERGED

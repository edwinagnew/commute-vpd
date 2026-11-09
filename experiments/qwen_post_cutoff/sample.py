"""E128 sampling of the unchanged Qwen3-1.7B (plan's stages 0, 1 and N1; HF on the cluster, Fable (d)). Refuses to start
without results/predictions.json. All sampling is bf16 (`qwen_eval.load_model`, E45's generation copy), thinking off,
cap 64 new tokens unless stated; the GSM8K read of stage 0 is fp32 (the path P4 compares with E61's 0.784).

  --control   stage 0, `control.py`: results/stage0.json.
  (default)   stage 1: 16 samples per candidate (4 per template), then 48 fresh samples (12 per template) for every
              0-of-16 new-side candidate and up to KNOWN_PER_NEW known-side candidates per such candidate in its stratum, at
              >= CAND_MIN of 16 (E120's stage 1 rule). Gap-side candidates get pass 16 only. A shard of SHARD facts is one file;
              a file on disk is read. results/stage1.json, results/large/stage1/.
  --think     N1 on results/pools.json: 4 think-mode samples per pool fact (one per template, the card's settings via
              E120's engine.THINK, cap 8,192). After the first shard, a mean length over THINK_MAX stops the job (Fable (d):
              N1 moves to E120's pod engine). results/n1.json.

Exit 0 done; 2 a filed check missed (written to the stage file); 3 paused at a shard boundary, resubmit the same command.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import control
import grade
import torch
from sample_common import (
    CAP,
    SEED,
    Gen,
    Item,
    Paths,
    facts_of,
    paths_under,
    real_gen,
    side_of,
    stop,
    write,
    write_gz,
)
from transformers import GenerationConfig

from commute import qwen_eval, qwen_sets
from commute.jobs import hard_exit, load_experiment
from commute.qwen_eval import SAMPLE, chat_prompt
from commute.qwen_vpd import fp32_matmuls, load_target

THINK_CAP, SHARD, GSM8K_N, FORMAT_MARGIN = 8192, 128, 256, 0.05
N1_SEQS, N1_SHARD, THINK_MAX = 32, 64, 2000.0  # a 32 x 8,192 batch holds E45's kv_tokens limit; facts per N1 shard
KNOWN_PER_NEW, CAND_MIN, N_MIN = 3, 8, 100  # E120's stage 1 rule; the plan's stop after pass 16


def shard_records(tok: Any, gen: Gen, chunk: list[list[Item]], n_ret: int, key: str) -> dict[str, Any]:
    """Samples for one shard: per fact {fact, n, k, unread, answers}, and tokens/s and format share in meta."""
    prompts = [chat_prompt(tok, it["user"]) for its in chunk for it in its]
    t0 = time.time()
    recs = gen(prompts, n_ret, SAMPLE, CAP, key)
    dt = time.time() - t0
    assert len(recs) == len(prompts) * n_ret
    rows, per = [], 4 * n_ret
    for j, its in enumerate(chunk):
        golds = grade.golds_of(its[0])
        got = [grade.read_sample(r["response"], r["truncated"], golds) for r in recs[j * per : (j + 1) * per]]
        rows.append({"fact": its[0]["fact"], "n": per, "k": sum(c for _, c in got),
                     "unread": sum(a is None for a, _ in got), "answers": [a for a, _ in got]})
    tokens = sum(r["response_tokens"] for r in recs)
    return {"meta": {"seconds": dt, "tokens": tokens, "tps": tokens / max(dt, 1e-9), "prompts": len(prompts),
                     "format_share": sum(r["n"] - r["unread"] for r in rows) / max(1, sum(r["n"] for r in rows)),
                     "truncated": statistics.mean(r["truncated"] for r in recs)},
            "rows": rows}


class Stage1:
    def __init__(self, paths: Paths, tok: Any, gen: Gen, budget_s: float) -> None:
        self.paths, self.tok, self.gen, self.budget_s, self.t0, self.slowest = paths, tok, gen, budget_s, time.time(), 0.0
        self.facts = facts_of(json.loads(paths.candidates.read_text()))
        self.summary: dict[str, Any] = {"seed": SEED, "cap": CAP, "facts": len(self.facts)}

    def shards(self, fids: list[str], name: str, n_ret: int, on_first: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
        """The shards of `fids` (n_ret samples per template); a shard on disk is read. Exit 3 at a boundary past the budget."""
        rows = []
        for i, s in enumerate(range(0, len(fids), SHARD)):
            f = self.paths.shards / name / f"shard_{i:04d}.json"
            if f.is_file():
                rec = json.loads(f.read_text())
            else:
                if time.time() - self.t0 + self.slowest > self.budget_s:
                    print(f"[{name}] BUDGET at shard {i}: resubmit the same command", flush=True)
                    write(self.paths.stage1, self.summary | {"paused_at": f"{name}/{i}"})
                    sys.exit(3)
                t = time.time()
                rec = shard_records(self.tok, self.gen, [self.facts[x] for x in fids[s : s + SHARD]], n_ret, f"{name}-{i}")
                write(f, rec)
                self.slowest = max(self.slowest, time.time() - t)
                m = rec["meta"]
                print(f"[{name}] shard {i}: {len(rec['rows'])} facts, {m['seconds']:.0f} s, {m['tps']:.0f} tokens/s, format "
                      f"{m['format_share']:.3f}, truncated {m['truncated']:.3f}", flush=True)
            if i == 0 and on_first is not None:
                on_first(rec["meta"])
            rows += rec["rows"]
        return rows

    def first_shard_check(self, meta: dict[str, Any]) -> None:
        floor = json.loads(self.paths.stage0.read_text())["a"]["known_format_share"] - FORMAT_MARGIN
        print(f"[check] first shard format share {meta['format_share']:.3f} (stop under {floor:.3f}); {meta['tps']:.0f} tokens/s", flush=True)
        if meta["format_share"] < floor:
            stop(self.paths.stage1, self.summary, f"format share {meta['format_share']} under {floor}")

    def select48(self, r16: list[dict[str, Any]]) -> list[str]:
        """The 0-of-16 new-side facts, and per stratum up to KNOWN_PER_NEW known-side facts per such fact at >= CAND_MIN of 16."""
        by_new, by_known = defaultdict(list), defaultdict(list)
        for r in r16:
            its = self.facts[r["fact"]]
            if side_of(its) == "new" and r["k"] == 0:
                by_new[its[0]["stratum"]].append(r["fact"])
            elif side_of(its) == "known" and r["k"] >= CAND_MIN:
                by_known[its[0]["stratum"]].append(r)
        picked = [f for fs in by_new.values() for f in fs]
        for s, rs in sorted(by_known.items()):
            rng = random.Random(f"{SEED}/{s}")
            rs.sort(key=lambda r: (-r["k"], rng.random()))
            picked += [r["fact"] for r in rs[: KNOWN_PER_NEW * len(by_new.get(s, []))]]
        return sorted(picked)

    def run(self) -> None:
        order = sorted(self.facts)
        random.Random(SEED).shuffle(order)
        s16 = self.shards(order, "s16", 4, self.first_shard_check)
        by = {r["fact"]: r for r in s16}
        zero_new = [f for f, r in by.items() if side_of(self.facts[f]) == "new" and r["k"] == 0]
        cand_known = [f for f, r in by.items() if side_of(self.facts[f]) == "known" and r["k"] >= CAND_MIN]
        self.summary |= {"zero16_new": len(zero_new), "known_candidates_at_8_of_16": len(cand_known)}
        print(f"[check] after pass 16: {len(zero_new)} new-side at 0 of 16, {len(cand_known)} known-side at >= {CAND_MIN} of 16 "
              f"(stop under {N_MIN} on either side)", flush=True)
        write(self.paths.stage1, self.summary)
        if len(zero_new) < N_MIN or len(cand_known) < N_MIN:
            stop(self.paths.stage1, self.summary, f"under {N_MIN} candidates on a side: {len(zero_new)} new, {len(cand_known)} known")
        sel = self.select48(s16)
        s48 = self.shards(sel, "s48", 12)
        write_gz(self.paths.merged, merge(self.facts, s16, s48))
        self.summary |= {"pass48": len(sel), "half_year_known_rate": half_year_rate(self.facts, s16)}
        write(self.paths.stage1, self.summary)
        print(f"STAGE 1 DONE {self.paths.stage1} {self.paths.merged}", flush=True)


def merge(facts: dict[str, list[Item]], r16: list[dict[str, Any]], r48: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per fact: pass 16 and pass 48 counts and answers, with its side, stratum, relation and date (for `pools.py`)."""
    by48 = {r["fact"]: r for r in r48}
    out = []
    for r in r16:
        its, f = facts[r["fact"]], by48.get(r["fact"], {"n": 0, "k": 0, "unread": 0, "answers": []})
        out.append({"fact": r["fact"], "side": side_of(its), "stratum": its[0]["stratum"], "relation": its[0]["subject"],
                    "date": its[0]["date"], "n16": r["n"], "k16": r["k"], "unread16": r["unread"], "a16": r["answers"],
                    "n48": f["n"], "k48": f["k"], "unread48": f["unread"], "a48": f["answers"]})
    return out


def half_year_rate(facts: dict[str, list[Item]], r16: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Report only (Fable (a), no rule): per half-year from 2023 on, facts sampled, mean share answered, share at >= 8 of 16."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in r16:
        d = facts[r["fact"]][0]["date"]
        if d >= "2023":
            groups[f"{d[:4]}H{1 + (int(d[5:7]) > 6)}"].append(r)
    return {k: {"facts": len(rs), "mean_rate": statistics.mean(r["k"] / r["n"] for r in rs),
                "share_ge_8_of_16": statistics.mean(r["k"] >= CAND_MIN for r in rs)}
            for k, rs in sorted(groups.items())}


# ----------------------------------------------------------------------------- N1


def run_n1(paths: Paths, tok: Any, gen: Gen, budget_s: float) -> None:
    """Four think-mode samples per pool fact (one per template), one shard per file."""
    think, chat = load_experiment("qwen_pass_pools", "engine").THINK, load_experiment("qwen_pass_pools", "problems").chat
    pools = json.loads(paths.pools.read_text())
    items_of: dict[str, list[Item]] = defaultdict(list)
    side: dict[str, str] = {}
    for s in ("known", "new"):
        for part in ("train", "held"):
            for it in pools[s][part]:
                items_of[it["fact"]].append(it)
                side[it["fact"]] = s
    for its in items_of.values():
        its.sort(key=lambda i: i["template"])
    fids, t0, slowest, rows = sorted(items_of), time.time(), 0.0, []
    for i, s in enumerate(range(0, len(fids), N1_SHARD)):
        f = paths.n1_shards / f"shard_{i:04d}.json"
        if f.is_file():
            rec = json.loads(f.read_text())
        else:
            if time.time() - t0 + slowest > budget_s:
                print(f"[n1] BUDGET at shard {i}: resubmit the same command", flush=True)
                sys.exit(3)
            tb = time.time()
            chunk = [items_of[x] for x in fids[s : s + N1_SHARD]]
            prompts = [chat(tok, it["user"], think=True) for its in chunk for it in its]
            recs: list[dict[str, Any]] = []
            for j in range(0, len(prompts), N1_SEQS):
                recs += gen(prompts[j : j + N1_SEQS], 1, think, THINK_CAP, f"n1-{i}-{j}")
            rec = {"rows": [{"fact": its[0]["fact"],
                             "correct": [grade.read_think(r["response"], r["truncated"], grade.golds_of(it))[1]
                                         for it, r in zip(its, recs[k * 4 : (k + 1) * 4], strict=True)],
                             "truncated": [r["truncated"] for r in recs[k * 4 : (k + 1) * 4]],
                             "tokens": [r["response_tokens"] for r in recs[k * 4 : (k + 1) * 4]]}
                            for k, its in enumerate(chunk)]}
            write(f, rec)
            slowest = max(slowest, time.time() - tb)
            toks = [x for r in rec["rows"] for x in r["tokens"]]
            print(f"[n1] shard {i}: {len(chunk)} facts, {time.time() - tb:.0f} s, mean tokens {statistics.mean(toks):.0f}", flush=True)
            if i == 0 and statistics.mean(toks) > THINK_MAX:
                stop(paths.n1, {"first_shard_mean_tokens": statistics.mean(toks)},
                     f"first shard mean length {statistics.mean(toks):.0f} over {THINK_MAX}: N1 moves to E120's pod engine (Fable (d))")
        rows += rec["rows"]
    write(paths.n1, n1_summary(rows, side))
    print(f"STAGE N1 DONE {paths.n1}", flush=True)


def n1_summary(rows: list[dict[str, Any]], side: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {"mean_tokens": statistics.mean(x for r in rows for x in r["tokens"]) if rows else None}
    for s in ("new", "known"):
        rs = [r for r in rows if side[r["fact"]] == s]
        out[s] = {"facts": len(rs), "answered_share": statistics.mean(any(r["correct"]) for r in rs) if rs else None,
                  "truncated_share": statistics.mean(x for r in rs for x in r["truncated"]) if rs else None}
    print(f"[n1] {json.dumps(out)}", flush=True)
    return out


# ----------------------------------------------------------------------------- entry


def gsm8k_read(tok: Any, sink: Path) -> dict[str, Any]:
    """GSM8K test 0-255, strict greedy, on E61's fp32 path (`qwen_eval.score`, as `pool_sft`'s step-100 read)."""
    fp32_matmuls()
    model = load_target("cuda", torch.float32)
    model.generation_config = GenerationConfig(eos_token_id=model.generation_config.eos_token_id, pad_token_id=tok.pad_token_id)
    sink.parent.mkdir(parents=True, exist_ok=True)
    return qwen_eval.summary(qwen_eval.score(tok, model, qwen_sets.load()["test"][:GSM8K_N], math.inf, sink, "gsm8k", 128))


def main(a: argparse.Namespace) -> None:
    results = Path(__file__).resolve().parent / "results"
    assert (results / "predictions.json").is_file(), "predictions.json is filed before anything samples"
    paths = paths_under(results, results / "large", a.candidates)
    tok, model = qwen_eval.load_model()
    gen = real_gen(tok, model)
    if a.control:
        def gsm8k() -> dict[str, Any]:
            model.cpu()  # the bf16 copy off the card before the fp32 read
            torch.cuda.empty_cache()
            return gsm8k_read(tok, paths.gsm8k_sink)

        control.stage0(paths, tok, gen, gsm8k, paths.candidates.is_file())
    elif a.think:
        run_n1(paths, tok, gen, a.budget_min * 60)
    else:
        s0 = json.loads(paths.stage0.read_text()) if paths.stage0.is_file() else {}
        assert s0.get("passed") and s0.get("candidates_checked"), f"{paths.stage0}: stage 0 has not passed with candidates"
        Stage1(paths, tok, gen, a.budget_min * 60).run()
    hard_exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--candidates", type=Path, default=Path(__file__).resolve().parent / "results" / "large" / "candidates.json")
    ap.add_argument("--budget-min", type=float, default=110.0)
    main(ap.parse_args())

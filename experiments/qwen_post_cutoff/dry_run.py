"""E128 dry run on CPU with real rows and a stand-in for the model (no weights are loaded). From the repo root:
  fleet/cpu-run --mem 6G -t 00:20:00 python experiments/qwen_post_cutoff/dry_run.py --candidates FILE OUT_DIR

Runs every sample.py stage against a stand-in `Gen` that answers from the candidate file: control (E45's facts and the
grader checks on the real candidates; the GSM8K read stubbed), stage 1 end to end (shards, resume from disk, selection,
merge), pools.build, the pool file read by `commute.pool_sft`'s `sequences` with the real tokenizer, the N1 path through
E120's engine.THINK and problems.chat, and grade.main on stand-in step-100 samples. The stand-in's answer rates are chosen
per side to give both labels; they say nothing about the model.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zlib
from pathlib import Path
from typing import Any

import control
import grade
import pools
import sample
import sample_common as sc
from transformers import AutoTokenizer

from commute.pool_sft import sequences
from commute.qwen_eval import MODEL, MODEL_REV


def stand_in(prompt_to_fact: dict[str, sc.Item], rng: random.Random) -> sc.Gen:
    """`Gen`: per prompt n_ret records; new-side candidates usually wrong, known-side usually right; thinking prompts get <think> text."""

    def gen(texts: list[str], n_ret: int, kw: dict[str, Any], cap: int, key: str) -> list[dict[str, Any]]:
        out = []
        for t in texts:
            it = prompt_to_fact.get(t)
            for _ in range(n_ret):
                if it is None:  # E45's facts and greedy reads
                    resp = "The answer is Paris."
                else:
                    side, gold = it["level"], it["answer"]
                    p = {"new": 0.0, "known": 0.85, "gap": 0.3}.get(side, 0.0)
                    wrong = "Unknown Person" if zlib.crc32(it["fact"].encode()) % 20 == 0 else f"Person {rng.randint(1, 9999)}"  # 1 in 20 facts: one wrong answer (the guard)
                    resp = f"The answer is {gold}." if rng.random() < p else f"The answer is {wrong}."
                    if rng.random() < 0.02:
                        resp = "I am not sure."
                out.append({"response": f"<think>hmm</think>{resp}" if cap > sc.CAP else resp,
                            "prompt_tokens": 20, "response_tokens": 8, "truncated": False})
        return out

    return gen


def main(candidates: Path, out: Path) -> None:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rng = random.Random(0)
    items = json.loads(candidates.read_text())
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    from commute.qwen_eval import chat_prompt

    prompts = {chat_prompt(tok, it["user"]): it for it in items}
    chat = sample.load_experiment("qwen_pass_pools", "problems").chat
    prompts |= {chat(tok, it["user"], think=True): it for it in items}
    gen = stand_in(prompts, rng)
    paths = sc.paths_under(out, out / "large", candidates)
    sc.write(out / "predictions.json", {})

    print("=== stage 0 (GSM8K stubbed; E45's facts stand in as 'Paris', so (a) is exercised and stops on its band)", flush=True)
    try:
        control.stage0(paths, tok, gen, lambda: {"strict_dropped": 0.78}, True)
    except SystemExit as e:
        print(f"[dry] stage 0 exit {e.code}", flush=True)
    assert json.loads(paths.stage0.read_text()).get("stopped"), "stand-in control should miss the known-fact band"
    print(f"[dry] b reading: {json.loads(paths.stage0.read_text())['b']}", flush=True)
    facts = sc.facts_of(items)
    print(f"[dry] {len(facts)} facts; first new {[f for f, its in facts.items() if its[0]['level'] == 'new'][:3]}", flush=True)

    print("=== stage 1 end to end", flush=True)
    s0 = {"passed": True, "candidates_checked": True, "a": {"known_format_share": 0.95}}
    sc.write(paths.stage0, s0)
    sample.Stage1(paths, tok, gen, budget_s=1e9).run()
    s1 = json.loads(paths.stage1.read_text())
    print(f"[dry] stage 1: {s1}", flush=True)
    print("=== stage 1 resume (all shards on disk are read, no new sampling)", flush=True)
    calls = [0]

    def counting(texts: list[str], n_ret: int, kw: dict[str, Any], cap: int, key: str) -> list[dict[str, Any]]:
        calls[0] += 1
        return gen(texts, n_ret, kw, cap, key)

    sample.Stage1(paths, tok, counting, budget_s=1e9).run()
    assert calls[0] == 0, "resume sampled again"

    print("=== pools", flush=True)
    import gzip

    with gzip.open(paths.merged, "rt") as f:
        merged = json.load(f)
    res = pools.build(items, merged)
    (out / "pools.json").write_text(json.dumps(res, indent=1))
    print(f"[dry] pools meta: n {res['meta']['n']}, {json.dumps(res['meta']['sizes'])}, cuts {res['meta']['guard_cuts']}, "
          f"changed {res['meta']['changed_count']}, n_if_kept {res['meta']['n_if_changed_kept']}, "
          f"nongold_ge32 {res['meta']['nongold_nonempty_ge_32']}", flush=True)
    keys = sorted(res["known"]["train"][0])
    print(f"[dry] pool item keys {keys}", flush=True)
    assert {"id", "source", "user", "gold_response", "answer", "aliases"} <= set(keys)

    print("=== pool file through commute.pool_sft.sequences (tokenizer only)", flush=True)
    tok_left = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    tok_left.padding_side = "left"
    end_id = int(tok_left.convert_tokens_to_ids("<|im_end|>"))
    for side in ("known", "new"):
        for part in ("train", "held"):
            seqs = sequences(tok_left, end_id, res[side][part])
            print(f"[dry] {side}/{part}: {len(seqs)} sequences, longest {max(len(s[0]) for s in seqs)} tokens", flush=True)

    print("=== N1 path (stand-in generate; THINK and the think-mode prompt from E120's engine and problems)", flush=True)
    paths_n1 = paths
    sample.run_n1(paths_n1, tok, gen, budget_s=1e9)
    print(f"[dry] n1: {json.loads(paths.n1.read_text())}", flush=True)

    print("=== grade (stand-in step-100 samples written as pool_sft writes them)", flush=True)
    for arm in ("known_s0", "new_s0"):
        d = out / "samples" / arm
        d.mkdir(parents=True)
        for name, part in (("new_held_samples", "held"), ("known_held_samples", "held")):
            side = name.split("_")[0]
            rows = [{"id": it["id"], "answer_gold": it["answer"], "source": it["source"], "user": it["user"],
                     "responses": [f"The answer is {it['answer']}." if rng.random() < 0.3 else "The answer is nothing."] * 8,
                     "truncated": [False] * 8} for it in res[side][part]]
            (d / f"{name}.json").write_text(json.dumps(rows))
        side = arm.split("_")[0]
        rows = [{"id": it["id"], "answer_gold": it["answer"], "source": it["source"], "user": it["user"],
                 "responses": [f"The answer is {it['answer']}."], "truncated": [False]} for it in res[side]["train"]]
        (d / f"{side}_train_greedy.json").write_text(json.dumps(rows))
    grade.main(out / "samples", out / "pools.json", out / "grades.json")

    print("=== sample.py refuses without predictions.json (filed in results/ before any sampling)", flush=True)
    print(f"[dry] OK; outputs in {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    main(a.candidates, a.out)

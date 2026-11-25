"""qwen_prefix_test: has Qwen3-1.7B-Base seen GSM8K's test questions? The prefix test of Wu et al. (arXiv 2507.10532).
Plan: reports/plans/2026-10-28_plan_qwen-prefix-test.md; predictions: results/predictions.json.

Each question is cut at a word boundary: the prefix is its first int(0.6 * n) whitespace-separated words, the
remainder the rest, both joined by single spaces. The base model gets the prefix as raw text (no chat template, no
instruction, no special tokens), decodes greedily for the remainder's token count plus SLACK new tokens, and scores an
exact match (EM) when its continuation, whitespace collapsed, equals the remainder or starts with the remainder and a
space. Batches of 128 (`commute.qwen_eval.generate`). Stages, each written to results/prefix.json as it finishes,
every number printed:
  1  copy control: the first 256 GSM8K test questions, each given whole, a blank line, then its own prefix; the model
     need only copy, so EM must be >= 0.90, else stop (it checks the cut, the batching and the comparison)
  2  GSM8K train, the first 1,319 in index order: the positive control, questions Qwen has likely seen
  3  the floor: Multilingual GSM-Symbolic English test_synthetic, 20 variants of each of 100 GSM8K test questions,
     generated in 2026, a year after Qwen3's release, so unseen; instance-major order
  4  GSM8K test, all 1,319 in index order: the reading
  5  readings: D_x = EM_x - EM_floor, r = D_test / D_train, the verdict
Continuations go to results/large/{copy,train,floor,test}.jsonl.

Usage (repo root, cluster GPU job): .venv/bin/python experiments/qwen_prefix_test/run.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset, load_dataset

from commute.qwen_eval import GREEDY, Records, generate, load_model, run_batches

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "prefix.json"
BASE, BASE_REV = "Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
GSM8K, GSM8K_REV = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"
MGS, MGS_REV = "danish-foundation-models/multilingual-gsm-symbolic", "75ce20d919d93ba78340adc6f70d2120f3ca3ce3"  # v0.5.5
PREFIX, MIN_WORDS, SLACK, BATCH = 0.6, 5, 8, 128
N_COPY, N_TRAIN, N_TEST, N_FLOOR, N_TEMPLATES = 256, 1319, 1319, 2000, 100
BUDGET_S = {"copy": 120, "train": 300, "floor": 420, "test": 300}
# filed in predictions.json (repeated here to print the verdict in the log; the filed file is the record)
COPY_MIN, CONTROL_MIN, R_CUTS = 0.90, 0.05, (0.2, 0.5)

Items = list[dict[str, Any]]


def say(stage: str, d: dict[str, Any]) -> None:
    for k, v in d.items():
        print(f"[{stage}] {k} = {v}", flush=True)


def norm(s: str) -> str:
    return " ".join(s.split())


def cut(question: str) -> tuple[str, str]:
    words = question.split()
    assert len(words) >= MIN_WORDS, question
    k = int(PREFIX * len(words))
    return " ".join(words[:k]), " ".join(words[k:])


def exact(continuation: str, remainder: str) -> bool:
    c = norm(continuation)
    return c == remainder or c.startswith(remainder + " ")


def rows(name: str, config: str, split: str, revision: str) -> list[dict[str, Any]]:
    ds = load_dataset(name, config, split=split, revision=revision)
    assert isinstance(ds, Dataset)
    return ds.to_list()


def items(questions: list[str], templates: list[int | None]) -> Items:
    out = []
    for i, (q, t) in enumerate(zip(questions, templates, strict=True)):
        prefix, remainder = cut(q)
        out.append({"idx": i, "template": t, "question": norm(q), "prefix": prefix, "remainder": remainder})
    return out


def load_sets() -> dict[str, Items]:
    test = [r["question"] for r in rows(GSM8K, "main", "test", GSM8K_REV)]
    train = [r["question"] for r in rows(GSM8K, "main", "train", GSM8K_REV)][:N_TRAIN]
    syn = rows(MGS, "eng", "test_synthetic", MGS_REV)
    rank: dict[int, int] = defaultdict(int)
    for r in syn:  # the variant's number within its template, so the order below is instance-major
        r["rank"] = rank[r["source_id"]]
        rank[r["source_id"]] += 1
    syn.sort(key=lambda r: (r["rank"], r["source_id"]))
    assert len(test) == N_TEST and len(train) == N_TRAIN and len(syn) == N_FLOOR, (len(test), len(train), len(syn))
    assert len(rank) == N_TEMPLATES and all(0 <= t < N_TEST for t in rank), sorted(rank)
    return {"copy": items(test[:N_COPY], [None] * N_COPY), "train": items(train, [None] * N_TRAIN),
            "floor": items([r["question"] for r in syn], [r["source_id"] for r in syn]),
            "test": items(test, [None] * N_TEST)}


def arm(tok: Any, model: Any, its: Items, label: str) -> Records:
    texts = [f"{it['question']}\n\n{it['prefix']}" if label == "copy" else it["prefix"] for it in its]
    need = [len(tok(" " + it["remainder"], add_special_tokens=False)["input_ids"]) for it in its]

    def step(idx: list[int]) -> Records:
        recs = generate(tok, model, [texts[i] for i in idx], 1, GREEDY, max_new=max(need[i] for i in idx) + SLACK)
        return [{"idx": its[i]["idx"], "template": its[i]["template"], "remainder": its[i]["remainder"],
                 "continuation": r["response"], "response_tokens": r["response_tokens"],
                 "em": exact(r["response"], its[i]["remainder"])} for i, r in zip(idx, recs, strict=True)]

    chunks = [list(range(s, min(s + BATCH, len(its)))) for s in range(0, len(its), BATCH)]
    return run_batches(chunks, step, BUDGET_S[label], LARGE / f"{label}.jsonl", label)


def em(recs: Records) -> dict[str, Any]:
    """Share of exact matches. SE: binomial, or over templates (stdev of template means / sqrt(T)) for the floor."""
    n = len(recs)
    if n == 0:
        return {"k": 0, "em": None, "se": None}
    p = sum(r["em"] for r in recs) / n
    by: dict[int, list[bool]] = defaultdict(list)
    for r in recs:
        if r["template"] is not None:
            by[r["template"]].append(r["em"])
    if not by:
        return {"k": n, "em": p, "se": math.sqrt(p * (1 - p) / n)}
    means = [sum(v) / len(v) for v in by.values()]
    return {"k": n, "em": p, "templates": len(means),
            "se": statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else None}


def verdict(d_train: float | None, d_test: float | None, r: float | None) -> str:
    if d_train is None or d_test is None:
        return "no reading"
    if d_train < CONTROL_MIN:
        return "like train" if d_test >= CONTROL_MIN else "no control"
    assert r is not None
    return "rare" if r <= R_CUTS[0] else "like train" if r >= R_CUTS[1] else "partial"


def readings(s: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """D_x = EM_x - EM_floor; r = D_test / D_train when the control passes; SEs treat the three sets as independent."""
    t, tr, f = (s[k]["em"] for k in ("test", "train", "floor"))
    if None in (t, tr, f):
        return {"verdict": verdict(None, None, None)}
    d_test, d_train = t - f, tr - f
    r = d_test / d_train if d_train >= CONTROL_MIN else None

    def se(*parts: tuple[float, str]) -> float | None:  # sqrt(sum of weight^2 * variance), None if any SE is missing
        if any(s[k]["se"] is None for _, k in parts):
            return None
        return math.sqrt(sum(w * w * s[k]["se"] ** 2 for w, k in parts))

    e = se((1, "test"), (r, "train"), (1 - r, "floor")) if r is not None else None  # SE of D_test - r D_train
    return {"d_train": d_train, "d_train_se": se((1, "train"), (1, "floor")), "control_pass": d_train >= CONTROL_MIN,
            "d_test": d_test, "d_test_se": se((1, "test"), (1, "floor")), "r": r,
            "r_se": e / d_train if e is not None else None, "verdict": verdict(d_train, d_test, r)}


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    LARGE.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "model": BASE, "revision": BASE_REV, "weights": "bfloat16", "prefix_share": PREFIX, "slack": SLACK,
        "batch": BATCH, "datasets": {GSM8K: GSM8K_REV, f"{MGS} eng test_synthetic": MGS_REV},
        "card": torch.cuda.get_device_name(), "torch": torch.__version__, "transformers": transformers.__version__,
        "job": os.environ.get("SLURM_JOB_ID"), "status": "running"}

    def dump(stage: str, d: dict[str, Any]) -> None:
        out[stage] = d
        say(stage, d)
        OUT.write_text(json.dumps(out, indent=1))

    def stop(why: str) -> None:
        out["status"] = f"stopped: {why}"
        OUT.write_text(json.dumps(out, indent=1))
        sys.exit(out["status"])

    sets = load_sets()
    tok, model = load_model(BASE, BASE_REV)
    s: dict[str, dict[str, Any]] = {}
    for label in ("copy", "train", "floor", "test"):
        s[label] = em(arm(tok, model, sets[label], label))
        dump(label, s[label])
        if label == "copy" and not (s["copy"]["em"] is not None and s["copy"]["em"] >= COPY_MIN):
            stop(f"copy control below {COPY_MIN}")
    dump("readings", readings(s))
    out["peak_gpu_gb"], out["status"] = torch.cuda.max_memory_allocated() / 2**30, "finished"
    OUT.write_text(json.dumps(out, indent=1))
    print(f"FINISHED {OUT} {LARGE}", flush=True)


if __name__ == "__main__":
    main()

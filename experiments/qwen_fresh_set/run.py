"""qwen_fresh_set: does the unchanged Qwen3-1.7B score as well on grade-school problems it cannot have memorised as on
GSM8K test? Plan: reports/plans/2026-10-28_plan_qwen-fresh-set.md; predictions: results/predictions.json.

Prompt, chat template, greedy decoding, batches of 128, 512 new tokens and graders exactly as
experiments/qwen_fresh_numbers (`commute.qwen_eval.score`, `commute.gsm8k`). Stages, each written to results/fresh_set.json
as it finishes, every number printed:
  0  grader control, no model: the strict grader on every set's gold rewritten to end "The answer is N." must read
     1.000 on all three sets, else stop
  1  GSM8K test, 1,319 in index order: the control and the same-job reference (strict over all answers within
     0.756 +- 0.012, or every response identical to the baseline's greedy.jsonl), else stop
  2  GSM1k (Scale AI; new problems written in 2024, held private, published 2025-03-31), index order, minus the
     items that share half or more of their word 8-grams with GSM8K questions (8 of 1,205)
  3  Multilingual GSM-Symbolic English test_synthetic (generated 2026): the first 10 of the 20 variants of each of the
     100 GSM8K test templates, instance-major order
  4  readings after stages 2 and 3; the primary rule is "dropped" (answers past the 512-token window leave the score)
Responses go to results/large/{gsm8k,gsm1k,mgs}.jsonl.

Usage (repo root, cluster GPU job): .venv/bin/python experiments/qwen_fresh_set/run.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset, load_dataset

from commute.gsm8k import gold_as_response, gold_number, grade
from commute.qwen_eval import (
    INSTRUCTION,
    MAX_NEW,
    MODEL,
    MODEL_REV,
    WINDOW,
    Records,
    load_model,
    score,
    share,
    summary,
    template_gap,
)

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "fresh_set.json"
SAVED = HERE.parent / "qwen_gsm8k_baseline" / "results" / "large" / "greedy.jsonl"
GSM8K, GSM8K_REV = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"
GSM1K, GSM1K_REV = "ScaleAI/gsm1k", "bc09569d09a614b9b530edc7f076fb214ac10493"
MGS, MGS_REV = "danish-foundation-models/multilingual-gsm-symbolic", "75ce20d919d93ba78340adc6f70d2120f3ca3ce3"
NGRAM, OVERLAP_MAX, N_GSM1K_KEPT, MGS_PER_TEMPLATE, BATCH = 8, 0.5, 1197, 10, 128
BUDGET_S = {"gsm8k": 480, "gsm1k": 450, "mgs": 360}
# filed in predictions.json (repeated here to print the verdicts in the log; the filed file is the record)
BASELINE_STRICT, BASELINE_SE = 997 / 1319, 0.012
GSM1K_CUTS, MGS_CUTS, FORMAT_TOL, MIN_GSM1K, MIN_TEMPLATES = (0.04, 0.08), (0.03, 0.08), 0.02, 900, 80

Items = list[dict[str, Any]]


def say(stage: str, d: dict[str, Any]) -> None:
    for k, v in d.items():
        print(f"[{stage}] {k} = {v}", flush=True)


def rows(name: str, *args: str, split: str, revision: str) -> Items:
    ds = load_dataset(name, *args, split=split, revision=revision)
    assert isinstance(ds, Dataset)
    return ds.to_list()


def grams(text: str) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def overlap(text: str, seen: set[tuple[str, ...]]) -> float:
    """Share of the text's word 8-grams found in `seen`; 0.0 for a text shorter than 8 words."""
    g = grams(text)
    return len(g & seen) / len(g) if g else 0.0


def load_sets() -> dict[str, Items]:
    """Each item: question, gold (a solution text ending '#### N'), idx (GSM8K: test index; GSM1k: its row; MGS: the
    GSM8K test index of its template's original)."""
    test = rows(GSM8K, "main", split="test", revision=GSM8K_REV)
    train = rows(GSM8K, "main", split="train", revision=GSM8K_REV)
    gsm = [{"idx": i, "question": r["question"], "gold": r["answer"]} for i, r in enumerate(test)]
    seen = set().union(*(grams(r["question"]) for r in test + train))
    g1 = rows(GSM1K, split="test", revision=GSM1K_REV)
    gsm1k = [{"idx": i, "question": r["question"], "gold": f"#### {r['answer']}"} for i, r in enumerate(g1)
             if overlap(r["question"], seen) < OVERLAP_MAX]
    syn = rows(MGS, "eng", split="test_synthetic", revision=MGS_REV)
    rank: dict[int, int] = defaultdict(int)
    mgs = []
    for r in syn:  # file order: 20 contiguous variants per template; keep the first MGS_PER_TEMPLATE of each
        if rank[r["source_id"]] < MGS_PER_TEMPLATE:
            mgs.append({"idx": r["source_id"], "instance": rank[r["source_id"]], "question": r["question"],
                        "gold": r["answer"]})
        rank[r["source_id"]] += 1
    mgs.sort(key=lambda it: (it["instance"], it["idx"]))
    assert len(gsm) == 1319 and len(g1) == 1205 and len(gsm1k) == N_GSM1K_KEPT, (len(gsm), len(g1), len(gsm1k))
    assert len(rank) == 100 and len(mgs) == 100 * MGS_PER_TEMPLATE and all(0 <= t < 1319 for t in rank), len(mgs)
    return {"gsm8k": gsm, "gsm1k": gsm1k, "mgs": mgs}


def grader_control(sets: dict[str, Items]) -> dict[str, float]:
    return {f"{name}_control_strict": sum(bool(grade(gold_as_response(it["gold"]), gold_number(it["gold"]))["strict"])
                                          for it in items) / len(items) for name, items in sets.items()}


def control(recs: Records) -> dict[str, Any]:
    strict = share(recs, "strict")
    saved = {r["idx"]: r["response"] for r in map(json.loads, SAVED.open())} if SAVED.is_file() else {}
    same = sum(saved.get(r["idx"]) == r["response"] for r in recs) / len(recs) if saved and recs else None
    return {"k": len(recs), "strict_all": strict, "identical_to_baseline_share": same,
            "pass": len(recs) == 1319 and strict is not None
            and (abs(strict - BASELINE_STRICT) <= BASELINE_SE or same == 1.0)}


def gap(ref: Records, other: Records, key: str, dropped: bool) -> dict[str, Any]:
    """Unpaired: accuracy on GSM8K minus accuracy on the other set; SE of a difference of two independent shares."""
    a = [r[key] for r in ref if r["fits"] or not dropped]
    b = [r[key] for r in other if r["fits"] or not dropped]
    if not a or not b:
        return {"n_gsm8k": len(a), "n": len(b), "diff": None, "se": None}
    pa, pb = sum(a) / len(a), sum(b) / len(b)
    return {"n_gsm8k": len(a), "n": len(b), "gsm8k": pa, "other": pb, "diff": pa - pb,
            "se": math.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))}


def zone(d: float | None, n: int, n_min: int, cuts: tuple[float, float]) -> str:
    """'same' at or below the lower cut (negative included), 'much lower' at or above the upper one, between:
    'ambiguous'; fewer than n_min answers or templates: 'underpowered'. Rounded to 9 places, so a tie in exact
    arithmetic (0.75 - 0.71) lands on the inclusive side."""
    if d is None or n < n_min:
        return "underpowered"
    d = round(d, 9)
    return "same" if d <= cuts[0] else "much lower" if d >= cuts[1] else "ambiguous"


def format_gap(recs: Records) -> float | None:
    """Lenient minus strict accuracy on the fitting answers: the share right but not in the instructed format."""
    fit = [r for r in recs if r["fits"]]
    return sum(bool(r["lenient"]) - bool(r["strict"]) for r in fit) / len(fit) if fit else None


def readings(recs: dict[str, Records]) -> dict[str, Any]:
    g, g1, m = recs["gsm8k"], recs.get("gsm1k", []), recs.get("mgs", [])
    d1k = gap(g, g1, "strict", dropped=True)
    dm = template_gap(g, m, "strict", dropped=True)
    verdict = zone(d1k["diff"], d1k["n"], MIN_GSM1K, GSM1K_CUTS)
    mgs_zone = zone(dm["diff"], dm["templates"], MIN_TEMPLATES, MGS_CUTS)
    cause = ("not needed" if verdict != "much lower" else "no reading" if mgs_zone == "underpowered"
             else "points to memory of GSM8K test items" if mgs_zone == "much lower"
             else "not separated: memory of methods, or harder problems")
    fg, f1 = format_gap(g), format_gap(g1)
    return {"D_1k": d1k, "D_1k_all_answers": gap(g, g1, "strict", dropped=False),
            "D_1k_capped": gap(g, g1, "capped_strict", dropped=False),
            "D_1k_lenient": gap(g, g1, "lenient", dropped=True),
            "D_mgs": dm, "D_mgs_capped": template_gap(g, m, "capped_strict", dropped=False),
            "verdict": verdict, "mgs_zone": mgs_zone, "cause": cause,
            "lenient_minus_strict": {"gsm8k": fg, "gsm1k": f1},
            "format_same": None if fg is None or f1 is None else abs(f1 - fg) <= FORMAT_TOL}


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    LARGE.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "model": MODEL, "revision": MODEL_REV, "weights": "bfloat16 (no update is read here)",
        "instruction": INSTRUCTION, "max_new_tokens": MAX_NEW, "window": WINDOW, "batch": BATCH,
        "mgs_per_template": MGS_PER_TEMPLATE, "gsm1k_overlap_max": OVERLAP_MAX,
        "datasets": {GSM8K: GSM8K_REV, GSM1K: GSM1K_REV, f"{MGS} eng test_synthetic": MGS_REV},
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
    dump("grader_control", grader_control(sets))
    if min(out["grader_control"].values()) != 1.0:
        stop("grader control below 1.000")
    tok, model = load_model()
    recs: dict[str, Records] = {}
    for label in ("gsm8k", "gsm1k", "mgs"):
        recs[label] = score(tok, model, sets[label], BUDGET_S[label], LARGE / f"{label}.jsonl", label, BATCH)
        dump(label, summary(recs[label]))
        if label == "gsm8k":
            dump("control", control(recs["gsm8k"]))
            if not out["control"]["pass"]:
                stop("the GSM8K arm does not reproduce the baseline")
        else:
            dump("readings", readings(recs))
    out["peak_gpu_gb"], out["status"] = torch.cuda.max_memory_allocated() / 2**30, "finished"
    OUT.write_text(json.dumps(out, indent=1))
    print(f"FINISHED {OUT} {LARGE}", flush=True)


if __name__ == "__main__":
    main()

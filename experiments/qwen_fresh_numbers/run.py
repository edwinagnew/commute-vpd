"""qwen_fresh_numbers: is the unchanged Qwen3-1.7B's 0.756 on GSM8K test ability, format loss, or memory of the test
set? Plan: reports/plans/2026-10-27_plan_qwen-fresh-numbers.md; predictions: results/predictions.json.

Prompt, chat template, greedy decoding, batches of 128, 512 new tokens and graders exactly as
experiments/qwen_gsm8k_baseline (`commute.qwen_eval`, `commute.gsm8k`). Stages, each written to results/fresh.json as
it finishes, every number printed:
  0  grader control, no model: the strict grader on every set's gold solution rewritten to end "The answer is N."
     must read 1.000 on all three sets, else stop
  1  GSM8K test, 1,319 greedy in index order: the same-job reference and the control (strict within 0.756 +- 0.012,
     or every response identical to the baseline's greedy.jsonl), else stop
  2  GSM-Plus "numerical substitution", 1,319 greedy; item i is the variant of GSM8K test problem i
  3  GSM-Symbolic main, the first 12 released instances of each of the 100 templates, instance-major order
  4  readings after stages 2 and 3, under two rules: "dropped" (answers past the 512-token window leave the score, and
     a pair or template leaves when its GSM8K side does) and "capped" (every answer, graded on its first 512 tokens)
Responses go to results/large/{gsm8k,plus,symbolic}.jsonl.

Usage (repo root, cluster GPU job): .venv/bin/python experiments/qwen_fresh_numbers/run.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset, load_dataset

from commute.gsm8k import gold_as_response, gold_number, grade
from commute.qwen_eval import (
    GREEDY,
    INSTRUCTION,
    MAX_NEW,
    MODEL,
    MODEL_REV,
    WINDOW,
    Records,
    generate,
    load_model,
    prompt,
    run_batches,
)

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "fresh.json"
SAVED = HERE.parent / "qwen_gsm8k_baseline" / "results" / "large" / "greedy.jsonl"
GSM8K, GSM8K_REV = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"
PLUS, PLUS_REV, PLUS_TYPE = "qintongli/GSM-Plus", "3b708db57b96a16e8e3368ed2956990c0809440e", "numerical substitution"
SYM, SYM_REV = "apple/GSM-Symbolic", "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
SYM_PER_TEMPLATE, BATCH = 12, 128
BUDGET_S = {"gsm8k": 480, "plus": 480, "symbolic": 420}
# filed in predictions.json (repeated here to print the verdicts in the log; the filed file is the record)
BASELINE_STRICT, BASELINE_SE = 997 / 1319, 0.012
PLUS_CUTS, SYM_CUTS, FORMAT_TOL = (0.02, 0.05), (0.03, 0.08), 0.02
MIN_PAIRS, MIN_TEMPLATES = 1000, 80

Items = list[dict[str, Any]]


def say(stage: str, d: dict[str, Any]) -> None:
    for k, v in d.items():
        print(f"[{stage}] {k} = {v}", flush=True)


def rows(name: str, *args: str, revision: str) -> Items:
    ds = load_dataset(name, *args, split="test", revision=revision)
    assert isinstance(ds, Dataset)
    return ds.to_list()


def load_sets() -> tuple[Items, Items, Items]:
    """Each item: question, gold (a solution text ending '#### N'), idx = the GSM8K test index it belongs to."""
    test = rows(GSM8K, "main", revision=GSM8K_REV)
    gsm = [{"idx": i, "question": r["question"], "gold": r["answer"]} for i, r in enumerate(test)]
    by_q = {it["question"].strip(): it["idx"] for it in gsm}
    plus = sorted(({"idx": by_q[r["seed_question"].strip()], "question": r["question"],
                    "gold": f"{r['solution']}\n#### {r['answer']}"}  # the answer field is the gold
                   for r in rows(PLUS, revision=PLUS_REV) if r["perturbation_type"] == PLUS_TYPE),
                  key=lambda it: it["idx"])
    sym = sorted(({"idx": r["original_id"], "template": r["id"], "instance": r["instance"], "question": r["question"],
                   "gold": r["answer"], "original": r["original_question"]}
                  for r in rows(SYM, "main", revision=SYM_REV) if r["instance"] < SYM_PER_TEMPLATE),
                 key=lambda it: (it["instance"], it["template"]))
    assert len(gsm) == 1319 and [it["idx"] for it in plus] == list(range(1319)), (len(gsm), len(plus))
    assert all(it.pop("original").strip() == gsm[it["idx"]]["question"].strip() for it in sym)
    assert len({it["template"] for it in sym}) == 100 and len(sym) == 100 * SYM_PER_TEMPLATE, len(sym)
    return gsm, plus, sym


def grader_control(sets: dict[str, Items]) -> dict[str, float]:
    return {f"{name}_control_strict": sum(bool(grade(gold_as_response(it["gold"]), gold_number(it["gold"]))["strict"])
                                          for it in items) / len(items) for name, items in sets.items()}


def arm(tok: Any, model: Any, items: Items, label: str) -> Records:
    texts = [prompt(tok, it["question"]) for it in items]

    def step(idx: list[int]) -> Records:
        out = []
        for i, r in zip(idx, generate(tok, model, [texts[i] for i in idx], 1, GREEDY, window=WINDOW), strict=True):
            gold = gold_number(items[i]["gold"])
            capped = grade(r["capped_response"], gold)
            keep = {k: v for k, v in items[i].items() if k not in ("question", "gold")}
            out.append({**keep, "gold": str(gold), **r, **grade(r["response"], gold),
                        "capped_strict": capped["strict"], "capped_lenient": capped["lenient"],
                        "fits": r["prompt_tokens"] + r["response_tokens"] <= WINDOW})
        return out

    chunks = [list(range(s, min(s + BATCH, len(items)))) for s in range(0, len(items), BATCH)]
    return run_batches(chunks, step, BUDGET_S[label], LARGE / f"{label}.jsonl", label)


def share(recs: Records, key: str) -> float:
    return sum(bool(r[key]) for r in recs) / len(recs)


def summary(recs: Records) -> dict[str, Any]:
    fit = [r for r in recs if r["fits"]]
    return {"k": len(recs), "fits_share": share(recs, "fits"), "truncation_share": share(recs, "truncated"),
            "strict_dropped": share(fit, "strict"), "lenient_dropped": share(fit, "lenient"),
            "strict_capped": share(recs, "capped_strict"), "lenient_capped": share(recs, "capped_lenient")}


def control(recs: Records) -> dict[str, Any]:
    strict = share(recs, "strict")
    saved = {r["idx"]: r["response"] for r in map(json.loads, SAVED.open())} if SAVED.is_file() else {}
    same = sum(saved.get(r["idx"]) == r["response"] for r in recs) / len(recs) if saved else None
    return {"k": len(recs), "strict_all": strict, "identical_to_baseline_share": same,
            "pass": abs(strict - BASELINE_STRICT) <= BASELINE_SE or same == 1.0}


def paired(ref: Records, other: Records, key: str, dropped: bool, integer_only: bool) -> dict[str, Any]:
    """GSM8K minus the variant on the same problems; SE of a paired difference of proportions (McNemar)."""
    by = {r["idx"]: r[key] for r in ref if r["fits"] or not dropped}
    pairs = [(by[r["idx"]], r[key]) for r in other if (r["fits"] or not dropped) and r["idx"] in by
             and (not integer_only or Decimal(r["gold"]) == Decimal(r["gold"]).to_integral_value())]
    n, b, c = len(pairs), sum(g and not v for g, v in pairs), sum(v and not g for g, v in pairs)
    return {"n": n, "gsm8k_only": b, "variant_only": c, "diff": (b - c) / n if n else None,
            "se": math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n if n else None}


def by_template(ref: Records, sym: Records, key: str, dropped: bool) -> dict[str, Any]:
    """Mean over templates of (original GSM8K question right) minus (share of its fresh instances right)."""
    orig = {r["idx"]: r[key] for r in ref if r["fits"] or not dropped}
    inst: dict[int, list[bool]] = defaultdict(list)
    for r in sym:
        if (r["fits"] or not dropped) and r["idx"] in orig:
            inst[r["idx"]].append(r[key])
    d = [orig[i] - sum(v) / len(v) for i, v in inst.items()]
    return {"templates": len(d), "diff": statistics.mean(d) if d else None,
            "se": statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None,
            "originals": statistics.mean(orig[i] for i in inst) if d else None}


def verdict(d: float | None, cuts: tuple[float, float]) -> str:
    if d is None:
        return "no reading"
    return "capability" if d <= cuts[0] else "memorisation" if d >= cuts[1] else "ambiguous"


def readings(recs: dict[str, Records], dropped: bool) -> dict[str, Any]:
    g, p, s = recs["gsm8k"], recs.get("plus", []), recs.get("symbolic", [])
    pre = "" if dropped else "capped_"
    plus = paired(g, p, pre + "strict", dropped, integer_only=True)
    sym = by_template(g, s, pre + "strict", dropped)
    pv, sv = verdict(plus["diff"], PLUS_CUTS), verdict(sym["diff"], SYM_CUTS)
    out = {"plus_paired": plus, "plus_paired_all_answers": paired(g, p, pre + "strict", dropped, integer_only=False),
           "plus_paired_lenient": paired(g, p, pre + "lenient", dropped, integer_only=True),
           "symbolic_by_template": sym, "symbolic_by_template_lenient": by_template(g, s, pre + "lenient", dropped),
           "plus_verdict": pv, "symbolic_verdict": sv,
           "verdict": "underpowered" if plus["n"] < MIN_PAIRS or sym["templates"] < MIN_TEMPLATES
           else "conflict" if {pv, sv} == {"capability", "memorisation"} else pv}
    if dropped and p and s:
        fit = {k: [r for r in v if r["fits"]] for k, v in recs.items()}
        gap = {k: share(v, "lenient") - share(v, "strict") for k, v in fit.items()}
        out["lenient_minus_strict"] = gap
        out["format_same_on_every_set"] = max(abs(gap[k] - gap["gsm8k"]) for k in ("plus", "symbolic")) <= FORMAT_TOL
    return out


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    LARGE.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "model": MODEL, "revision": MODEL_REV, "weights": "bfloat16 (no update is read here)",
        "instruction": INSTRUCTION, "max_new_tokens": MAX_NEW, "window": WINDOW, "batch": BATCH,
        "symbolic_per_template": SYM_PER_TEMPLATE,
        "datasets": {GSM8K: GSM8K_REV, f"{PLUS} ({PLUS_TYPE})": PLUS_REV, f"{SYM} main": SYM_REV},
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

    gsm, plus, sym = load_sets()
    dump("grader_control", grader_control({"gsm8k": gsm, "plus": plus, "symbolic": sym}))
    if min(out["grader_control"].values()) != 1.0:
        stop("grader control below 1.000")
    tok, model = load_model()
    recs: dict[str, Records] = {"gsm8k": arm(tok, model, gsm, "gsm8k")}
    dump("control", control(recs["gsm8k"]))
    dump("gsm8k", summary(recs["gsm8k"]))
    if not out["control"]["pass"]:
        stop("the GSM8K arm does not reproduce the baseline")
    for label, items in (("plus", plus), ("symbolic", sym)):
        recs[label] = arm(tok, model, items, label)
        dump(label, summary(recs[label]))
        dump("readings_dropped", readings(recs, dropped=True))
        dump("readings_capped", readings(recs, dropped=False))
    out["peak_gpu_gb"], out["status"] = torch.cuda.max_memory_allocated() / 2**30, "finished"
    OUT.write_text(json.dumps(out, indent=1))
    print(f"FINISHED {OUT} {LARGE}", flush=True)


if __name__ == "__main__":
    main()

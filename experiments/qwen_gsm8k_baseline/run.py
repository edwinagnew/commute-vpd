"""qwen_gsm8k_baseline: the unchanged Qwen3-1.7B on GSM8K test, non-thinking mode. Plan:
reports/plans/2026-10-27_plan_qwen-port-and-baseline.md, Part 2.

Weights in bf16 (no update is read here). Chat template with enable_thinking=False and the fixed INSTRUCTION (both in
`commute.qwen_eval`, moved there 2026-10-27 unchanged);
grading by `commute.gsm8k` (strict: the last "The answer is N"; lenient: the last number). Stages, each written to
results/baseline.json as it finishes and every number printed:
  0  positive control: the strict grader on the 1,319 gold solutions rewritten to end "The answer is N." must read
     1.000, else stop
  1  greedy, all 1,319 test prompts in index order, 512 new tokens, left-padded batches; stops early (k recorded) if
     the next batch would pass the stage's time budget. Responses to results/large/greedy.jsonl
  2  8 samples at T = 0.7 (top_p 1, top_k off, seed 0) on the first 256 test prompts; to results/large/samples.jsonl
Lengths count prompt tokens (chat template included) plus response tokens up to and including the end token; a
response is truncated when no end token comes within 512 new tokens.

Usage (repo root, cluster GPU job): .venv/bin/python experiments/qwen_gsm8k_baseline/run.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import load_dataset

from commute.gsm8k import gold_answer, gold_as_response, grade
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
PRED, OUT = RESULTS / "predictions.json", RESULTS / "baseline.json"
DATA, DATA_REV = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"
GREEDY_BATCH, GREEDY_BUDGET_S = 128, 600
N_SAMPLE_PROMPTS, N_SAMPLES, TEMPERATURE, SEED = 256, 8, 0.7, 0
SAMPLE_BATCH_PROMPTS, SAMPLE_BUDGET_S = 16, 480
SAMPLING = {"do_sample": True, "temperature": TEMPERATURE, "top_p": 1.0, "top_k": 0}
# thresholds from predictions.json (scored here for the log; the filed file is the record)
EXPECT = {"control_strict": (1.0, 1.0), "greedy_strict": (0.60, 0.75), "lenient_minus_strict_abs": (0.0, 0.05),
          "share_total_le_512": (0.95, 1.0), "truncation_share": (0.0, 0.02), "sample_disagreement": (0.20, 0.50)}


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(out, indent=1))


def say(stage: str, d: dict[str, Any]) -> None:
    for k, v in d.items():
        print(f"[{stage}] {k} = {v}", flush=True)


def load_rows() -> list[dict[str, Any]]:
    ds = load_dataset(DATA, "main", split="test", revision=DATA_REV)
    rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]  # pyright: ignore
    assert len(rows) == 1319, len(rows)
    return rows


def lengths(recs: Records) -> dict[str, Any]:
    tot = torch.tensor([r["prompt_tokens"] + r["response_tokens"] for r in recs], dtype=torch.float)
    q = torch.quantile(tot, torch.tensor([0.5, 0.9, 0.95, 0.99])).tolist()
    return {"share_total_le_512": (tot <= WINDOW).float().mean().item(),
            "total_p50_p90_p95_p99_max": [*q, tot.max().item()],
            "response_mean": sum(r["response_tokens"] for r in recs) / len(recs),
            "truncation_share": sum(r["truncated"] for r in recs) / len(recs)}


def stage0(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gs = [grade(gold_as_response(r["answer"]), gold_answer(r["answer"])) for r in rows]
    return {"n": len(gs), "control_strict": sum(bool(g["strict"]) for g in gs) / len(gs),
            "control_lenient": sum(bool(g["lenient"]) for g in gs) / len(gs)}


def stage1(tok: Any, model: Any, rows: list[dict[str, Any]], texts: list[str]) -> dict[str, Any]:
    def step(idx: list[int]) -> Records:
        recs = generate(tok, model, [texts[i] for i in idx], 1, GREEDY)
        golds = [gold_answer(rows[i]["answer"]) for i in idx]
        return [{"idx": i, "gold": g, **r, **grade(r["response"], g)} for i, g, r in zip(idx, golds, recs, strict=True)]

    chunks = [list(range(s, min(s + GREEDY_BATCH, len(rows)))) for s in range(0, len(rows), GREEDY_BATCH)]
    recs = run_batches(chunks, step, GREEDY_BUDGET_S, LARGE / "greedy.jsonl", "greedy")
    n = len(recs)
    strict, lenient = sum(r["strict"] for r in recs) / n, sum(r["lenient"] for r in recs) / n
    return {"k": n, "greedy_strict": strict, "greedy_lenient": lenient,
            "lenient_minus_strict_abs": abs(lenient - strict),
            "strict_lenient_disagreement": sum(r["strict"] != r["lenient"] for r in recs) / n,
            "no_strict_answer_share": sum(r["strict_pred"] is None for r in recs) / n, **lengths(recs)}


def stage2(tok: Any, model: Any, rows: list[dict[str, Any]], texts: list[str]) -> dict[str, Any]:
    def step(idx: list[int]) -> Records:
        recs = generate(tok, model, [texts[i] for i in idx], N_SAMPLES, SAMPLING)
        out = []
        for j, r in enumerate(recs):
            i = idx[j // N_SAMPLES]
            g = gold_answer(rows[i]["answer"])
            out.append({"idx": i, "sample": j % N_SAMPLES, "gold": g, **r, **grade(r["response"], g)})
        return out

    torch.manual_seed(SEED)
    chunks = [list(range(s, s + SAMPLE_BATCH_PROMPTS)) for s in range(0, N_SAMPLE_PROMPTS, SAMPLE_BATCH_PROMPTS)]
    recs = run_batches(chunks, step, SAMPLE_BUDGET_S, LARGE / "samples.jsonl", "samples")
    correct = torch.tensor([r["strict"] for r in recs], dtype=torch.float).view(-1, N_SAMPLES).sum(1)
    return {"k_prompts": len(correct), "sample_disagreement": ((correct > 0) & (correct < N_SAMPLES)).float().mean().item(),
            "mean_sample_accuracy": (correct / N_SAMPLES).mean().item(),
            "sample_truncation_share": sum(r["truncated"] for r in recs) / len(recs)}


def score(out: dict[str, Any]) -> dict[str, bool]:
    vals = {**out.get("control", {}), **out.get("greedy", {}), **out.get("samples", {})}
    return {k: lo <= vals[k] <= hi for k, (lo, hi) in EXPECT.items() if k in vals}


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    LARGE.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "model": MODEL, "revision": MODEL_REV, "dataset": f"{DATA} main test", "dataset_revision": DATA_REV,
        "weights": "bfloat16 (no update is read here)", "instruction": INSTRUCTION, "max_new_tokens": MAX_NEW,
        "sampling": {**SAMPLING, "n": N_SAMPLES, "prompts": N_SAMPLE_PROMPTS, "seed": SEED},
        "card": torch.cuda.get_device_name(), "torch": torch.__version__, "transformers": transformers.__version__,
        "job": os.environ.get("SLURM_JOB_ID"), "status": "running"}
    rows = load_rows()
    out["control"] = stage0(rows)
    say("control", out["control"])
    dump(out)
    if out["control"]["control_strict"] != 1.0:
        out["status"] = "stopped: positive control below 1.000"
        dump(out)
        sys.exit(out["status"])
    tok, model = load_model()
    texts = [prompt(tok, r["question"]) for r in rows]
    out["example_prompt"] = texts[0]
    out["greedy"] = stage1(tok, model, rows, texts)
    say("greedy", out["greedy"])
    dump(out)
    out["samples"] = stage2(tok, model, rows, texts)
    say("samples", out["samples"])
    out["score"], out["peak_gpu_gb"] = score(out), torch.cuda.max_memory_allocated() / 2**30
    out["status"] = "finished"
    say("score", out["score"])
    dump(out)
    print(f"FINISHED {OUT} {LARGE / 'greedy.jsonl'} {LARGE / 'samples.jsonl'}", flush=True)


if __name__ == "__main__":
    main()

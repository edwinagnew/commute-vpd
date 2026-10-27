"""Generation for the unchanged Qwen3-1.7B on GSM8K-style problems: one prompt, one chat template, one window rule.

Moved here from `experiments/qwen_gsm8k_baseline/run.py` (2026-10-27) so every Qwen GSM8K evaluation uses the same
code; the baseline's behaviour is unchanged. Weights in bf16, chat template with enable_thinking=False and the fixed
INSTRUCTION; up to MAX_NEW new tokens, left-padded batches. Lengths count prompt tokens (chat template included) plus
response tokens up to and including the end token; a response is truncated when no end token comes within MAX_NEW.
With `window`, each record also carries `capped_response`: the response cut where prompt plus response reach `window`
tokens (the "capped" reading; equal to `response` when the whole answer fits).
`load_model(name, revision)` and `generate(..., max_new=)` serve other checkpoints and shorter generations (the base
model's prefix test, experiments/qwen_prefix_test); the defaults are the ones above.
`score`, `summary` and `template_gap` are the graded arm and its readings as experiments/qwen_fresh_numbers ran them
(moved here 2026-10-28 for experiments/qwen_fresh_set; that run.py keeps its own copy, as it ran).
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from commute.gsm8k import gold_number, grade

MODEL, MODEL_REV = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
INSTRUCTION = ("Solve the problem step by step. Then end your response with one final line of exactly this form: "
               "The answer is N.\nwhere N is the final numerical answer, written as a plain number.")
MAX_NEW, WINDOW = 512, 512
GREEDY = {"do_sample": False}

Records = list[dict[str, Any]]


def load_model(name: str = MODEL, revision: str = MODEL_REV) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    tok.padding_side = "left"
    model: Any = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.bfloat16)
    model = model.cuda().eval()
    # drop the checkpoint's sampling defaults (T 0.6, top_k 20, top_p 0.95) so only the settings passed here apply
    eos = model.generation_config.eos_token_id
    model.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=tok.pad_token_id)
    return tok, model


def prompt(tok: Any, question: str) -> str:
    msgs = [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(tok: Any, model: Any, texts: list[str], n_ret: int, kw: dict[str, Any],
             window: int | None = None, max_new: int = MAX_NEW) -> Records:
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, num_return_sequences=n_ret, **kw)
    gen = out[:, enc["input_ids"].shape[1]:].cpu()
    plen = enc["attention_mask"].sum(1).repeat_interleave(n_ret).tolist()
    eos = torch.tensor(model.generation_config.eos_token_id)
    recs = []
    for row, p in zip(gen, plen, strict=True):
        hits = torch.isin(row, eos).nonzero()
        n = int(hits[0]) + 1 if len(hits) else len(row)
        rec = {"response": tok.decode(row[:n], skip_special_tokens=True), "prompt_tokens": int(p),
               "response_tokens": n, "truncated": not len(hits)}
        if window is not None:
            rec["capped_response"] = tok.decode(row[:min(n, max(window - int(p), 0))], skip_special_tokens=True)
        recs.append(rec)
    return recs


def run_batches(chunks: list[list[int]], step: Callable[[list[int]], Records], budget_s: float, sink: Path,
                label: str) -> Records:
    """Runs chunks in order, appending each batch's records to `sink`; stops before a batch that would pass budget_s."""
    sink.unlink(missing_ok=True)
    recs, t0, slowest = [], time.time(), 0.0
    for i, chunk in enumerate(chunks):
        if time.time() - t0 + slowest > budget_s:
            print(f"[{label}] budget {budget_s} s reached: stopping after {i} of {len(chunks)} batches", flush=True)
            break
        tb = time.time()
        batch = step(chunk)
        dt = time.time() - tb
        slowest = max(slowest, dt)
        with sink.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in batch)
        recs += batch
        toks = sum(r["response_tokens"] for r in batch)
        print(f"[{label}] batch {i + 1}/{len(chunks)}: {dt:.1f} s, {toks / dt:.0f} new tokens/s", flush=True)
        if i == 0:
            print(f"[{label}] projected {dt * len(chunks):.0f} s for {len(chunks)} batches (budget {budget_s} s)",
                  flush=True)
    return recs


def score(tok: Any, model: Any, items: Records, budget_s: float, sink: Path, label: str, batch: int = 128) -> Records:
    """Greedy answers to `items` in order, graded. Each item has `question` and `gold` (a solution text ending
    '#### N'); its other keys are carried into the record. Each record: the response, strict and lenient grades on the
    whole response and on the window-capped one, and `fits` (prompt plus response within WINDOW)."""
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

    chunks = [list(range(s, min(s + batch, len(items)))) for s in range(0, len(items), batch)]
    return run_batches(chunks, step, budget_s, sink, label)


def share(recs: Records, key: str) -> float | None:
    return sum(bool(r[key]) for r in recs) / len(recs) if recs else None


def summary(recs: Records) -> dict[str, Any]:
    """Accuracy under three rules: dropped (fitting answers only), all (whole responses), capped (first WINDOW tokens)."""
    fit = [r for r in recs if r["fits"]]
    return {"k": len(recs), "k_fits": len(fit), "fits_share": share(recs, "fits"),
            "truncation_share": share(recs, "truncated"), "strict_dropped": share(fit, "strict"),
            "lenient_dropped": share(fit, "lenient"), "strict_all": share(recs, "strict"),
            "strict_capped": share(recs, "capped_strict"), "lenient_capped": share(recs, "capped_lenient")}


def template_gap(ref: Records, variants: Records, key: str, dropped: bool) -> dict[str, Any]:
    """Mean over templates of (original right) minus (share of its variants right). A record's `idx` is the GSM8K test
    index of its original; with `dropped`, non-fitting answers leave, and a template leaves with its original."""
    orig = {r["idx"]: r[key] for r in ref if r["fits"] or not dropped}
    inst: dict[int, list[bool]] = defaultdict(list)
    for r in variants:
        if (r["fits"] or not dropped) and r["idx"] in orig:
            inst[r["idx"]].append(r[key])
    d = [orig[i] - sum(v) / len(v) for i, v in inst.items()]
    return {"templates": len(d), "diff": statistics.mean(d) if d else None,
            "se": statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None,
            "originals": statistics.mean(orig[i] for i in inst) if d else None}

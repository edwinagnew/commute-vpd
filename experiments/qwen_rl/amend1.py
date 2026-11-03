"""E69 Amendment 1, the $0 reads on one cluster L40, one state per job (reports/plans/2026-11-03_plan_e69-amend1.md; rules in
results/predictions_amend1.json; order in RUN_BRIEF_amend1.md):
  --stage saved --arm W0                      W0 rescored on this card: the floor f. First, alone.
  --stage saved --arm N --step 100            N100 rebuilt from its δW, against N.json within f. Second, alone.
  --stage saved --arm R|N --step 10|20|50 [--seed 70]   greedy accuracy of a saved state, both sets.
  --stage open --arm W0 --opening own         the control: W0 prefilled with its own first two words reproduces W0.
  --stage open --arm W0 --opening "We are"    and "We need"; --arm N --step 100 --opening own. GSM8K test.
Per-problem files in results/large/<run>/acc/<label>_<set>.jsonl, numbers in results/<run>/amend1/<label>.json.
Exit 0 done; 1 a refusal; 2 a control missed ("stopped" in that JSON, and the stage refuses from then on).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Any

import torch
from job import RESULTS, Job, config, read_json, save_rollouts
from relaxed import ci, read

from commute.jobs import hard_exit
from commute.qwen_eval import score, share, summary
from commute.qwen_sft import load, plus, set_weights
from commute.qwen_updates import at_most

TAG, W0_LABEL = "1e-6w10", "W0card"
SEED1 = config("full", TAG)
TWO_WORDS = re.compile(r"\S+\s+\S")
Records = list[dict[str, Any]]


class State(Job):
    log_name = "amend1.log"  # run.log is the arm's own record


def needs(job: Job, *labels: str) -> None:
    """Seed 1's stages this one reads: done, and on this card (the same card, the same rows)."""
    device = torch.cuda.get_device_name()
    for label in labels:
        got = read_json(RESULTS / SEED1.out / "amend1" / f"{label}.json")
        if not got.get("done") or got["device"] != device:
            sys.exit(f"{label} must finish first, on {device}: it reads {got.get('device')}, done {got.get('done')}")


def read_set(job: Job, label: str, name: str, items: Records, prefill: list[str] | None = None, batch: int | None = None) -> Records:
    sink = job.large / "acc" / f"{label}_{name}.jsonl"
    sink.parent.mkdir(exist_ok=True)
    recs = score(job.tok, job.model, items, math.inf, sink, f"{label} {name}", batch or job.cfg.gen_batch, prefill)
    recs = [{**r, "relaxed": ci(r), **({"opening": prefill[j]} if prefill else {})} for j, r in enumerate(recs)]
    save_rollouts(sink, recs)
    return recs


def numbers(recs: Records) -> dict[str, Any]:
    fit = [r for r in recs if r["fits"]]
    return {**summary(recs), "relaxed_all": share(recs, "relaxed"), "relaxed_dropped": share(fit, "relaxed")}


def state_weights(job: Job, arm: str, step: int | None) -> Any:
    return None if arm == "W0" else plus(job.W0, load(job.large / "dW" / arm / f"step_{step}.safetensors"))


def flips(a: dict[int, Any], b: Records) -> int:
    return sum((a[r["idx"]]["fits"], a[r["idx"]]["strict"]) != (r["fits"], r["strict"]) for r in b)


def saved_control(job: Job, arm: str, out: dict[str, Any], recs: dict[str, Records]) -> None:
    A0, P = job.get("A0")["accuracy"], json.loads((RESULTS / "predictions_amend1.json").read_text())["saved_control"]
    if arm == "W0":
        out["floor"] = {}
        for s, rs in recs.items():
            gap = abs(out[s]["strict_dropped"] - A0[s]["strict_dropped"])
            out["floor"][s] = {"gap": gap, "flips": flips(read(job.shared_large / "acc" / f"W0_{s}.jsonl"), rs), "k": len(rs)}
            out["floor"][s]["f"] = out["floor"][s]["flips"] / out[s]["k_fits"]
            job.check(out, f"W0 near A0: {s}", at_most(gap, P["gap_max"]), out["floor"][s])
        return
    floor = read_json(RESULTS / SEED1.out / "amend1" / f"{W0_LABEL}.json")["floor"]
    N = read_json(RESULTS / SEED1.out / "N.json")["accuracy"]
    for s in recs:
        gap = abs(out[s]["strict_dropped"] - N[s]["strict_dropped"])
        job.check(out, f"N100 rebuilt matches N.json: {s}", at_most(gap, floor[s]["f"]), {"gap": gap, "f": floor[s]["f"]})


def stage_saved(job: Job, arm: str, step: int | None) -> None:
    out = job.begin()
    if out is None:
        return
    control = arm == "W0" or (arm == "N" and step == 100)
    if not control:
        needs(job, W0_LABEL, "N100")
    elif arm == "N":
        needs(job, W0_LABEL)
    out["device"] = torch.cuda.get_device_name()
    job.fp32()
    recs = {}
    with set_weights(job.model, state_weights(job, arm, step)):
        for name, items in (("gsm8k", job.sets["test"][: job.cfg.n_gsm8k]), ("gsm1k", job.sets["gsm1k"][: job.cfg.n_gsm1k])):
            recs[name] = read_set(job, job.stage.split("/")[1], name, items)
            out[name] = numbers(recs[name])
            job.log(f"[{job.stage} {name}] {json.dumps(out[name])}")
            job.put(job.stage, out)
    if control:
        saved_control(job, arm, out, recs)
    out["done"] = True
    job.put(job.stage, out)


def own_opening(tok: Any, response: str) -> str:
    """The fewest tokens of the response whose text holds two words: the opening ends on a token boundary, so that the
    prefilled path can follow the plain one token for token (a word cut inside ":\\n\\n" could not)."""
    ids = tok(response, add_special_tokens=False)["input_ids"]
    for k in range(1, len(ids) + 1):
        text = tok.decode(ids[:k])
        if TWO_WORDS.search(text):
            return text
    return response


def differs(ref: dict[int, Any], recs: Records) -> list[int]:
    keys = ("response", "response_tokens", "truncated")
    return [r["idx"] for r in recs if any(r[k] != ref[r["idx"]][k] for k in keys)]


def open_control(job: Job, label: str, out: dict[str, Any], ref: dict[int, Any], items: Records, recs: Records) -> None:
    """W0 prefilled against W0; only if any problem differs, the floor: plain W0 at another batch size, so other padding."""
    miss = differs(ref, recs)
    out["differs"] = {"n": len(miss), "idx": miss[:50]}
    floor = None
    if miss:
        floor = len(differs(ref, read_set(job, f"{label}_floor", "gsm8k", items, batch=job.cfg.gen_batch - 1)))
        out["differs"]["floor"] = floor
    job.check(out, "W0 prefilled with its own opening reproduces W0", not miss or (floor is not None and len(miss) <= floor), out["differs"])


def stage_open(job: Job, arm: str, step: int | None, opening: str) -> None:
    out = job.begin()
    if out is None:
        return
    label = job.stage.split("/")[1]
    control = arm == "W0" and opening == "own"
    needs(job, W0_LABEL, *(() if control else ("W0_own",)), *(("N100",) if arm == "N" else ()))
    out["device"], items = torch.cuda.get_device_name(), job.sets["test"][: job.cfg.n_gsm8k]
    job.fp32()
    ref = read(job.large / "acc" / f"{W0_LABEL}_gsm8k.jsonl")
    prefill = [own_opening(job.tok, ref[it["idx"]]["response"]) for it in items] if opening == "own" else [opening] * len(items)
    with set_weights(job.model, state_weights(job, arm, step)):
        recs = read_set(job, label, "gsm8k", items, prefill)
        out["gsm8k"] = numbers(recs)
        job.log(f"[{job.stage}] {json.dumps(out['gsm8k'])}")
        if control:
            open_control(job, label, out, ref, items, recs)
    out["done"] = True
    job.put(job.stage, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("saved", "open"))
    ap.add_argument("--arm", required=True, choices=("W0", "R", "N"))
    ap.add_argument("--step", type=int)
    ap.add_argument("--seed", type=int, default=None, help="a second seed's run (run.py --seed); seed 1 if left out")
    ap.add_argument("--opening", help="own, or the literal text")
    a = ap.parse_args()
    if a.stage == "open" and (a.seed is not None or not a.opening):
        sys.exit("--stage open reads seed 1 and needs --opening")
    if (a.arm == "W0") == (a.step is not None) or (a.arm == "W0" and a.seed is not None):
        sys.exit("--step goes with R and N, and W0 is seed 1's")
    label = W0_LABEL if a.arm == "W0" else f"{a.arm}{a.step}"
    if a.stage == "open":
        label = f"{a.arm if a.arm == 'W0' else label}_{a.opening.lower().replace(' ', '_')}"
    job = State(config("full", TAG, draw_seed=a.seed), f"amend1/{label}")
    if a.stage == "saved":
        stage_saved(job, a.arm, a.step)
    else:
        stage_open(job, a.arm, a.step, a.opening)
    job.log(f"[{job.stage}] finished; {json.dumps(job.peaks())}")
    hard_exit(0)


if __name__ == "__main__":
    main()

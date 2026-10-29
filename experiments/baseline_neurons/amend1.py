"""E52 Amendment 1: P4 re-asked with the demonstrations next to the query. Does the original model W0 perform each task
in context? Decision: reports/decisions/2026-10-29_decision_e52-p4.md, option A. Every setting, rule and threshold:
results/predictions_amend1.json (filed before the run, read here, never edited).

W0 only, fp32, on each task's 48 scoring windows. Prompt = window tokens 0..18 (Pile), n demonstrations (marker, key,
value), the window's marker and query key; E52's own prompts() draws the demonstrations. Each state's denominator
G_score is E52's (results/baseline_neurons.json 'gains').
  stage 0   the induction control (a random sequence shown twice, top-1 on the second copy), alone; then the n = 0
            check (tokens equal E52's n = 0 window, rise 0 to rounding); exit 1 on a miss
  prompts   per n in 0, 4, 16, 31: rise (nats), top-1, share per state
then P4a: copy's mean share at n = 31, its band, and E52's separation rule. Results go to results/amend1.json after
every stage; every number is also printed.

Usage (repo root, GPU):  python experiments/baseline_neurons/amend1.py [--states DIR]   (cluster/baseline_neurons_amend1.sbatch)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from run import (
    RESULTS,
    STATES,
    TASKS,
    Net,
    Task,
    _finite,
    load_task,
    log,
    logp_at,
    prompts,
    separation,
)
from torch import Tensor

OUT = RESULTS / "amend1.json"
VOCAB = 50277


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def induction(net: Net, C: dict[str, Any]) -> dict[str, Any]:
    """W0's top-1 on a random sequence of distinct ids shown twice; the second copy's first token is not scored."""
    L = C["length"]
    seq = torch.stack([torch.randperm(VOCAB, generator=torch.Generator().manual_seed(C["seed"] + i))[:L]
                       for i in range(C["sequences"])])
    toks = torch.cat([seq, seq], 1)
    logits = net.run(toks)[0]
    assert logits.shape[-1] == VOCAB
    hit = logits.argmax(-1).cpu()[:, :-1] == toks[:, 1:]  # logits at p predict token p + 1
    r = {"top1_first": float(hit[:, : L - 1].double().mean()), "top1_second": float(hit[:, L:].double().mean()),
         "n_scored_second": int(hit[:, L:].numel())}
    r["ok"] = r["top1_second"] >= C["threshold"]
    return r


def adjacent(t: Task, n: int) -> Tensor:
    """E52's prompts() rows (demonstrations, window 0..20) reordered to window 0..18, demonstrations, marker, key."""
    p, d, m = prompts(t, n), 3 * n, t.half - 1
    return torch.cat([p[:, d : d + m], p[:, :d], p[:, d + m :]], 1)


def prompted(net: Net, t: Task, n: int) -> dict[str, float]:
    """W0 on the n-demonstration prompts of the scoring half: rise of log P(target) over the 21-token window, top-1."""
    logits = net.run(adjacent(t, n))[0]
    tgt = t.tgt(t.toks[t.part("score")])
    rise = logp_at(logits, -1, tgt) - t.lp0[t.part("score")]
    top1 = logits[:, -1].argmax(-1).cpu() == tgt
    return {"rise": float(rise.mean()), "max_abs_rise": float(rise.abs().max()), "top1": float(top1.double().mean())}


def n0_check(tasks: dict[str, Task], rec: dict[str, dict[str, float]], C: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, t in tasks.items():
        same = torch.equal(adjacent(t, 0), t.toks[t.part("score"), : t.half + 1])
        r = rec[name]
        out[name] = {"tokens_equal": same, "ok": same and abs(r["rise"]) <= C["mean_tol"] and r["max_abs_rise"] <= C["max_tol"]}
    out["pass"] = all(v["ok"] for v in out.values())
    return out


def band(x: float | None, B: dict[str, float]) -> str:
    return ("not scored" if x is None else "holds" if x >= B["holds"] else "partial" if x >= B["partial"]
            else "mechanism only")


def scored(shares: dict[str, float], gains: dict[str, Any]) -> dict[str, float | None]:
    return {s: (None if gains[s]["no_change"] else shares[s]) for s in STATES}


def score(shares: dict[str, float], gains: dict[str, Any], B: dict[str, float]) -> dict[str, Any]:
    """P4a on the n = 31 shares: the band of copy's mean over scored states and E52's separation rule (copy higher)."""
    vals = scored(shares, gains)
    copy = [v for s, v in vals.items() if v is not None and TASKS[s.rsplit("_", 1)[0]] == "copy"]
    cs = sum(copy) / len(copy) if copy else None
    sep = separation(vals, copy_higher=True)
    b = band(cs, B)
    verdict = ("not scored" if "not scored" in (b, sep["verdict"]) else
               "pass" if b == "holds" and sep["verdict"] == "pass" else "fails")
    return {"copy_share": cs, "band": b, "separation": sep, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=RESULTS.parent.parent / "regating_table" / "results" / "large")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    P = json.loads((RESULTS / "predictions_amend1.json").read_text())  # refuses to start without it
    gains = json.loads((RESULTS / "baseline_neurons.json").read_text())["gains"]
    ns, S0 = P["settings"]["n"], P["stage0"]
    assert ns[0] == 0 and ns[-1] == 31 and sorted(gains) == sorted(STATES)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = Net(a.device)
    out: dict[str, Any] = {"gains": gains, "stage0": {"induction": induction(net, S0["induction_control"]["params"])}}
    log(f"[stage0 induction] {json.dumps(out['stage0']['induction'])}")
    dump(out)
    if not out["stage0"]["induction"]["ok"]:
        out["score"] = {"verdict": "not run", "why": "control failed: the apparatus cannot see in-context learning"}
        dump(out)
        log("[stage0] induction control FAIL: stop")
        sys.exit(1)
    tasks = {n: load_task(net, a.states, n) for n in TASKS}
    rec: dict[str, dict[str, dict[str, float]]] = {"0": {name: prompted(net, t, 0) for name, t in tasks.items()}}
    out["prompts"] = rec
    out["stage0"]["n0"] = n0_check(tasks, rec["0"], S0["n0_check"]["params"])
    log(f"[stage0 n0] {json.dumps(rec['0'])} {json.dumps(out['stage0']['n0'])}")
    dump(out)
    if not out["stage0"]["n0"]["pass"]:
        out["score"] = {"verdict": "not run", "why": "n = 0 check failed: a format bug"}
        dump(out)
        log("[stage0] n = 0 check FAIL: stop")
        sys.exit(1)
    for n in ns[1:]:
        rec[str(n)] = {name: prompted(net, t, n) for name, t in tasks.items()}
        log(f"[n={n}] {json.dumps(rec[str(n)])}")
        dump(out)
    out["shares"] = {s: {str(n): rec[str(n)][s.rsplit("_", 1)[0]]["rise"] / gains[s]["G_score"] for n in ns}
                     for s in STATES}
    out["top1_n31"] = {s: rec["31"][s.rsplit("_", 1)[0]]["top1"] for s in STATES}
    log(f"[shares] {json.dumps(out['shares'])}")
    log(f"[top1 n=31] {json.dumps(out['top1_n31'])}")
    at = {str(n): {s: out["shares"][s][str(n)] for s in STATES} for n in ns}
    out["separation_unscored"] = {str(n): separation(scored(at[str(n)], gains), copy_higher=True) for n in ns[1:-1]}
    out["score"] = score(at["31"], gains, P["rule"]["band_params"])
    assert all(math.isfinite(v) for v in out["top1_n31"].values())
    log(f"[score P4a] {json.dumps(_finite(out['score']))}")
    dump(out)


if __name__ == "__main__":
    main()

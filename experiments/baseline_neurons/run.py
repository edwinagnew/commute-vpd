"""baseline_neurons (E52): do neuron patching (B2), steering (B3) or prompting the original model (B4) tell E12's copy
fine-tunes (elicit) from its cipher fine-tunes (teach)? Plan: reports/plans/2026-10-29_plan_baselines-scoping.md,
section "B2 + B3 + B4 = E52". Every setting, rule and threshold: results/predictions.json (filed before the run, read
here, never edited).

The target model only (no decomposition is read), fp32, on each task's 97 held-out windows (fit half = even indices, 49;
scoring half = odd, 48). Per state the gain G of log P(target) at the answer position over W0; every share is the method's
change of log P over W0 divided by G on the scoring half.
  stage 0   the controls, alone: identity neuron patch, full residual patch, G > 1 nat (ident32_s0, rand32_s0); exit 1
            on a miss
  gains     G, G_fit, G_score of the eight states; per-task floor; 'no change' iff |G| < floor
  B4        W0 with n = 0, 4, 16 demonstrations (marker, key, value) of the task, the query's key never among them
  B3        one steering vector per block input, the block chosen on the fit half
  B2        top-k MLP neurons by |mean tuned - W0 activation| take the tuned activations; k50 and the idle share
  code      code_step10's per-token held-out gain against its floor (P5)
then the scored verdicts and the decision tree. Results go to results/baseline_neurons.json after every stage; every
number is also printed.

Usage (repo root, GPU):  python experiments/baseline_neurons/run.py [--states DIR]   (cluster/baseline_neurons.sbatch)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.data import pile_rows
from commute.neurons import neuron_patch, patch_controls
from commute.steering import (  # moved here from this script, unchanged (2026-10-30, E54); amend1.py imports them from here
    D_MLP,
    DW,
    N_LAYERS,
    Net,
    Task,
    b3,
    gains,
    logp_at,
)
from commute.steering import load_task as _load_task

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "baseline_neurons.json"
TASKS = {"ident32": "copy", "ident32b": "copy", "rand32": "cipher", "rand32b": "cipher"}
STATES = [f"{t}_{s}" for t in TASKS for s in ("s0", "s1")]
CODE_BATCH = 8


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    return [_finite(v) for v in o] if isinstance(o, list | tuple) else o


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def load_task(net: Net, src: Path, name: str) -> Task:
    return _load_task(net, src / f"windows_{name}.pt", name, TASKS[name])


# ----------------------------------------------------------------------------- stage 0


def stage0(net: Net, tasks: dict[str, Task], dws: dict[str, DW], th: dict[str, float]) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for name in ("ident32", "rand32"):
        t, dW = tasks[name], dws[f"{name}_s0"]
        g = gains(net, t, dW)
        pc = patch_controls(net, t, dW, g["G_score"])  # moved to commute.neurons (2026-10-30, E56), same forwards
        full = pc["full_patch_share"]
        r = {"identity_max_abs": pc["identity_max_abs"], "full_patch_share": full, **g}
        r["ok"] = (r["identity_max_abs"] <= th["identity_max"] and abs(full - 1) <= th["full_patch_tol"]
                   and g["G"] > th["gain_min"])
        rec[f"{name}_s0"] = r
        log(f"[stage0 {name}_s0] {json.dumps(r)}")
    rec["pass"] = all(r["ok"] for r in rec.values())
    return rec


# ----------------------------------------------------------------------------- the three methods


def prompts(t: Task, n: int) -> Tensor:
    """Scoring-half windows cut after the key, each behind n demonstrations (marker, key, value) of other keys."""
    rows = []
    for i in range(1, len(t.toks), 2):
        w = t.toks[i]
        others = [k for k in t.keys if k != int(w[t.half])]
        perm = torch.randperm(len(others), generator=torch.Generator().manual_seed(1000 * n + i))[:n]
        demo = [x for j in perm.tolist() for x in (t.marker, others[j], int(t.table[others[j]]))]
        rows.append(torch.tensor(demo + w[: t.half + 1].tolist()))
    return torch.stack(rows)


def b4(net: Net, t: Task, ns: list[int]) -> dict[int, float]:
    """Per n, the mean over the scoring half of log P_W0,prompted - log P_W0 (nats; divided by each state's G later)."""
    s = t.toks[t.part("score")]
    return {n: float((logp_at(net.run(prompts(t, n))[0], -1, t.tgt(s)) - t.lp0[t.part("score")]).mean()) for n in ns}


def b2(net: Net, t: Task, dW: DW, g: dict[str, float], q99: Tensor, B: dict[str, Any]) -> dict[str, Any]:
    # ranking and grid moved to commute.neurons.neuron_patch (2026-10-30, E56), same forwards in the same order
    p = neuron_patch(net, t, dW, g["G_score"], B["grid"])
    order, shift, a0 = p["order"], p["shift"], p["a0"]
    top = order[:16].to(a0.device)
    idle = (a0[:, top].abs().mean(0) < B["idle_frac"] * q99[top]).double().mean()
    return {"shares": p["shares"], "k50": p["k50"], "k50_index": p["k50_index"],
            "idle_share": float(idle), "top16": order[:16].tolist(), "top16_shift": shift.cpu()[order[:16]].tolist()}


def pile_q99(net: Net, q: float) -> Tensor:
    """Per neuron (layer-major, 12288), the q-quantile of |W0 activation| over every position of Pile rows val 0..63."""
    rows = pile_rows("val", 0, 64, log)
    acts = [net.run(rows[s : s + 8], capture=True)[1] for s in range(0, rows.shape[0], 8)]
    out = []
    for i in range(N_LAYERS):
        a = torch.cat([c[f"mlp{i}"].reshape(-1, D_MLP) for c in acts]).abs()
        out.append(torch.kthvalue(a, math.ceil(q * a.shape[0]), dim=0).values)
    return torch.cat(out)


@torch.no_grad()
def code_gain(net: Net, src: Path) -> float:
    """code_step10: mean over held-out code rows and next-token positions of log P_tuned - log P_W0 (fp32)."""
    rows = torch.load(src / "code_held.pt", weights_only=False)

    def mean_lp() -> float:
        tot = 0.0
        for s in range(0, rows.shape[0], CODE_BATCH):
            b = rows[s : s + CODE_BATCH].to(net.device)
            lp = torch.log_softmax(net.run(b)[0][:, :-1], -1).gather(2, b[:, 1:, None])
            tot += float(lp.double().sum())
        return tot / (rows.shape[0] * (rows.shape[1] - 1))

    w0 = mean_lp()
    with net.at(torch.load(src / "code_step10.pt", weights_only=True)):
        return mean_lp() - w0


# ----------------------------------------------------------------------------- scoring


def separation(vals: dict[str, float | None], copy_higher: bool) -> dict[str, Any]:
    """The filed rule over the scored states (None = 'no change'): pass, reversed, fails or not scored."""
    by = {k: [v for n, v in vals.items() if v is not None and TASKS[n.rsplit("_", 1)[0]] == k] for k in ("copy", "cipher")}
    if min(len(v) for v in by.values()) < 2:
        return {"verdict": "not scored", "n": {k: len(v) for k, v in by.items()}}
    assert all(math.isfinite(x) for v in by.values() for x in v), vals
    mean = {k: sum(v) / len(v) for k, v in by.items()}
    gap, spread = mean["copy"] - mean["cipher"], max(max(v) - min(v) for v in by.values())
    sep = abs(gap) > spread
    verdict = "fails" if not sep else "pass" if (gap > 0) == copy_higher else "reversed"
    return {"mean": mean, "gap": gap, "spread": spread, "verdict": verdict}


def score(out: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    nc = {n: out["gains"][n]["no_change"] for n in STATES}
    pick = lambda f: {n: (None if nc[n] else f(n)) for n in STATES}
    sc: dict[str, Any] = {"P1": separation(pick(lambda n: out["b2"][n]["k50_index"]), copy_higher=False),
          "P2": separation(pick(lambda n: out["b2"][n]["idle_share"]), copy_higher=False),
          "P3": separation(pick(lambda n: out["b3"][n]["S"]), copy_higher=True),
          "P4": separation(pick(lambda n: out["b4"]["shares"][n]["16"]), copy_higher=True)}
    G = out["code_step10"]["G"]
    sc["P5"] = {"G": G, "floor": P["settings"]["floor_min"], "verdict": "pass" if abs(G) < P["settings"]["floor_min"] else "fails"}
    sep = [k for k in ("P1", "P2", "P3") if sc[k]["verdict"] in ("pass", "reversed")]
    sc["branch"] = ("P4 fails: stop and tell the user" if sc["P4"]["verdict"] != "pass" else
                    f"objection holds ({', '.join(sep)} separate)" if sep else "VPD's favour: P1-P3 do not separate")
    return sc


# ----------------------------------------------------------------------------- main


def stage_gains(net: Net, tasks: dict[str, Task], dws: dict[str, DW], floor_min: float) -> dict[str, Any]:
    rec = {n: gains(net, tasks[n.rsplit("_", 1)[0]], dws[n]) for n in STATES}
    for t in TASKS:
        floor = max(floor_min, 2 * abs(rec[f"{t}_s0"]["G"] - rec[f"{t}_s1"]["G"]))
        for s in ("s0", "s1"):
            r = rec[f"{t}_{s}"]
            r.update(floor=floor, no_change=abs(r["G"]) < floor)
            log(f"[gains {t}_{s}] {json.dumps(r)}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=HERE.parent / "regating_table" / "results" / "large")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    P = json.loads((RESULTS / "predictions.json").read_text())  # refuses to start without it
    S, B = P["settings"], P["settings"]["b2"]
    assert P["states"]["copy"] + P["states"]["cipher"] == STATES and B["grid"][-1] == N_LAYERS * D_MLP
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = Net(a.device)
    tasks = {n: load_task(net, a.states, n) for n in TASKS}
    dws = {n: torch.load(a.states / f"{n}.pt", weights_only=True) for n in STATES}
    out: dict[str, Any] = {"stage0": stage0(net, tasks, dws, P["stage0"]["thresholds"])}
    dump(out)
    if not out["stage0"]["pass"]:
        log("[stage0] FAIL: stop")
        sys.exit(1)
    out["gains"] = stage_gains(net, tasks, dws, S["floor_min"])
    nb4 = {t: b4(net, tasks[t], S["b4"]["n"]) for t in TASKS}
    out["b4"] = {"nats": nb4, "shares": {n: {str(k): v / out["gains"][n]["G_score"] for k, v in nb4[n.rsplit("_", 1)[0]].items()}
                                         for n in STATES}}
    log(f"[b4] {json.dumps(out['b4'])}")
    dump(out)
    out["b3"] = {}
    for n in STATES:
        out["b3"][n] = b3(net, tasks[n.rsplit("_", 1)[0]], dws[n], out["gains"][n])
        log(f"[b3 {n}] {json.dumps(out['b3'][n])}")
    dump(out)
    q99 = pile_q99(net, B["pile_quantile"])
    out["b2"] = {}
    for n in STATES:
        out["b2"][n] = b2(net, tasks[n.rsplit("_", 1)[0]], dws[n], out["gains"][n], q99, B)
        log(f"[b2 {n}] {json.dumps(_finite(out['b2'][n]))}")
    dump(out)
    G = code_gain(net, a.states)
    out["code_step10"] = {"G": G, "floor": S["floor_min"], "no_change": abs(G) < S["floor_min"]}
    log(f"[code_step10] {json.dumps(out['code_step10'])}")
    out["score"] = score(out, P)
    log(f"[score] {json.dumps(_finite(out['score']))}")
    dump(out)


if __name__ == "__main__":
    main()

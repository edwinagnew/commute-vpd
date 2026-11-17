"""E146 (steering_reversal), cell P2.14: why steering reads the side built to be new higher on half-known vs cipher (equal
size; own step, reported) and rank-1 vs cipher at step 10, where the share reads the reuse side higher. On E143's saved
states and rows, one thing changes per reading: the strength (fixed at 1), the layer (fixed at 2, at 3), the reference (the
norm-matched random reading subtracted), the denominator (the pair's mean gain). Plan
reports/plans/2026-11-10_plan_steering-reversal.md; card reports/plans/2026-11-10_plan_p2-14-card.md; filed rule
results/predictions.json (the run refuses unless it is committed unchanged), applied by rules.py. No verdict on reuse.

Settings per reading come from E143's saved selection grid (experiments/holes_fill/results/stage1.json, the filed rule
`fair_steering.pick` restricted to strength 1, to layer 2 or to layer 3); every rise is read here on the read rows, with
E143's code (holes_fill's `Job.task`, `Job.dW`, `lo_at`, `lo_steered`) and vectors (`fair_steering.vectors`).

  stage 0  controls, alone, exit 1 on a miss: a zero update steers by exactly 0; two states reproduce E143's filed share.
  stage 1  every state (written as it lands; its filed share must reproduce E143's), then the score.

    sbatch --comment=E146 cluster/steering_reversal.sbatch
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import rules
import torch

from commute import fair_steering as FS
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HF = load_experiment("holes_fill", "run")
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
E143 = REPO / "experiments" / "holes_fill" / "results" / "stage1.json"
E136 = REPO / "experiments" / "share_untrained" / "results" / "inventory.json"
KINDS = ("pooled", "perkey")


def log(msg: str) -> None:
    print(msg, flush=True)


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"stage{n}.tmp"
    tmp.write_text(json.dumps(jsonable(out), indent=1))
    tmp.replace(RESULTS / f"stage{n}.json")


class Stopped(Exception):
    pass


def settings(saved: dict[str, Any], layers: list[int]) -> dict[str, tuple[int, float] | None]:
    """The filed pick on E143's selection grid, restricted per reading. Layer 0's vector is zero (no update changes the
    embedding), so its rises are 0 and it is left out; the unrestricted pick must still be E143's own setting."""
    g = {(int(k.split(",")[0]), float(k.split(",")[1])): v for k, v in saved["grid"].items()}
    g = {ls: v for ls, v in g.items() if ls[0] in layers and ls[1] > 0}  # strength 0 reads 0 (E143's control)
    assert FS.pick(g) == tuple(saved["setting"]), (FS.pick(g), saved["setting"])
    sub = lambda keep: FS.pick({ls: v for ls, v in g.items() if keep(ls)})
    return {"filed": FS.pick(g), "strength1": sub(lambda ls: ls[1] == 1), "layer2": sub(lambda ls: ls[0] == 2), "layer3": sub(lambda ls: ls[0] == 3)}


def kind_reads(job: Any, t: Any, v: dict[int, torch.Tensor], picks: dict[str, Any], c: dict[str, Any], gen: torch.Generator) -> dict[str, Any]:
    """Read-row rise at every (layer, strength), and the random means at each reading's setting."""
    toks = t.read.toks
    rise = lambda vec, l, s: float((HF.lo_steered(job.net, t, toks, vec, l, s) - t.lo0).mean())
    grid = {(l, s): rise(v[l], l, s) for l in c["layers"] for s in c["strengths"]}
    rnd = {}
    for ls in dict.fromkeys(x for x in picks.values() if x is not None):
        l, s = ls
        rnd[ls] = statistics.fmean(rise(FS.random_like(v[l], gen), l, s) for _ in range(c["n_random"]))
    return {"grid": grid, "random": rnd}


def readings(r: dict[str, Any], picks: dict[str, Any], gain: float, pair_gain: float) -> dict[str, float]:
    nan = float("nan")
    val = {k: (r["grid"][ls] / gain if ls is not None else nan) for k, ls in picks.items()}
    f = picks["filed"]
    val["random_subtracted"] = (r["grid"][f] - r["random"][f]) / gain if f is not None else nan
    val["pair_gain"] = r["grid"][f] / pair_gain if f is not None else nan
    val["random_at_strength1"] = r["random"][picks["strength1"]] / gain if picks["strength1"] is not None else nan
    return val


def state(job: Any, name: str, saved: dict[str, Any], c: dict[str, Any], i: int) -> dict[str, Any]:
    t, dW = job.task(job.inv["states"][name]["task"]), job.dW(name)
    gain = float((HF.lo_at(job.net, t, t.read.toks, dW) - t.lo0).mean())
    vs = FS.vectors(job.net, t.x(t.train), t.idx(t.train), len(t.pool), dW, t.read.half)
    gen = torch.Generator().manual_seed(c["random_seed"] + i)
    out: dict[str, Any] = {"gain": gain}
    for kd in KINDS:
        picks = settings(saved["steering"][kd], c["layers"])
        out[kd] = {"picks": picks, **kind_reads(job, t, vs[kd], picks, c, gen)}
    return out


def reproduce(rec: dict[str, Any], saved: dict[str, Any], kd: str, tol: float) -> dict[str, float]:
    f = rec[kd]["picks"]["filed"]
    got = rec[kd]["grid"][f] / rec["gain"]
    return {"share": got, "e143": saved["steering"][kd]["share"], "ok": abs(got - saved["steering"][kd]["share"]) <= tol}


def stage0(job: Any, P: dict[str, Any], saved: dict[str, Any]) -> dict[str, Any]:
    th, c = P["thresholds"], P["settings"]
    t = job.task("e54/elicit")
    zero = {k: torch.zeros_like(w) for k, w in job.net.W0.items()}
    b = t.train[: FS.BATCH]
    vs = FS.vectors(job.net, t.x(t.train), t.idx(t.train), len(t.pool), zero, t.read.half)  # every training window, as stage 1 (one batch misses keys)
    lg0 = job.net.run(t.x(b))[0][:, t.read.half]
    out: dict[str, Any] = {"zero": max(float((FS.logits_steered(job.net, t.x(b), t.idx(b), t.read.half, vs[kd][l], l, s) - lg0).abs().max())
                                       for kd in KINDS for l in c["layers"] for s in (1.0, 8.0))}  # fmt: skip
    log(f"[0 zero update, largest |logit change|] {out['zero']}")
    out["reproduce"] = {}
    for i, (name, kd) in enumerate((("e54/elicit_s0", "perkey"), ("c3/cipher_s0", "pooled"))):
        rec = state(job, name, saved[name], c, 1000 + i)
        out["reproduce"][name] = reproduce(rec, saved[name], kd, th["reproduce_tol"])
    log(f"[0 reproduce E143] {json.dumps(jsonable(out['reproduce']))}")
    bad = out["zero"] > th["zero_tol"] or not all(r["ok"] for r in out["reproduce"].values())
    if bad:
        out["stopped"] = "control missed"
        write(0, out)
        raise Stopped("control missed")
    return out


def stage1(job: Any, P: dict[str, Any], saved: dict[str, Any], pairs: dict[str, Any], t_job: float) -> dict[str, Any]:
    th, c = P["thresholds"], P["settings"]
    order = list(dict.fromkeys(s for p in pairs.values() for s in p["reuse"] + p["new"]))
    out: dict[str, Any] = {"states": {}}
    t0 = time.time()
    for i, name in enumerate(order):
        rec = state(job, name, saved[name], c, i)
        rec["reproduce"] = {kd: reproduce(rec, saved[name], kd, th["reproduce_tol"]) for kd in KINDS}
        out["states"][name] = rec
        log(f"[1 {name}] gain {rec['gain']:.4f} " + " ".join(f"{kd} {rec[kd]['picks']} filed {rec['reproduce'][kd]['share']:.4f}" for kd in KINDS))
        write(1, out)
        if not all(r["ok"] for r in rec["reproduce"].values()):
            out["stopped"] = f"{name} does not reproduce E143"
            write(1, out)
            raise Stopped(out["stopped"])
        minutes = ((time.time() - t0) / (i + 1) * len(order) + (t0 - t_job)) / 60
        log(f"[1 time] {i + 1} of {len(order)}; projected {minutes:.1f} min")
        if i >= 3 and minutes > th["projected_limit_min"]:
            out["stopped"] = f"projected {minutes:.1f} min"
            write(1, out)
            raise Stopped(out["stopped"])
    out["score"] = score(P, pairs, out["states"])
    log(f"[1 score] {json.dumps(jsonable(out['score']))}")
    write(1, out)
    return out


def pair_kind(pair: dict[str, Any], kd: str, st: dict[str, Any], th: dict[str, float]) -> dict[str, Any]:
    """One pair in one kind: every reading's call record with its values, and the random means at strength 1."""
    names = pair["reuse"] + pair["new"]
    pg = statistics.fmean(st[n]["gain"] for n in names)
    vals = {n: readings(st[n][kd], st[n][kd]["picks"], st[n]["gain"], pg) for n in names}
    out: dict[str, Any] = {}
    for r in rules.READINGS:
        reuse, new = [vals[n][r] for n in pair["reuse"]], [vals[n][r] for n in pair["new"]]
        out[r] = {**rules.pair(reuse, new, th), "reuse": reuse, "new": new}
    out["random_at_strength1"] = {n: vals[n]["random_at_strength1"] for n in names}
    return out


def score(P: dict[str, Any], pairs: dict[str, Any], st: dict[str, Any]) -> dict[str, Any]:
    th, kinds = P["thresholds"], {**P["pairs"]["scored"], **P["pairs"]["reported"]}
    recs: dict[str, Any] = {}
    for p, own in kinds.items():
        recs[p] = {}
        for kd in KINDS:
            tag = "" if kd == own else f"_{kd}"
            recs[p].update({k + tag: v for k, v in pair_kind(pairs[p], kd, st, th).items()})
    scored = {p: {r: recs[p][r] for r in rules.READINGS} for p in P["pairs"]["scored"]}
    names_ = {r: rules.named(scored, r, th["relative_shrink"]) for r in rules.READINGS if r != "filed"}
    band = [n for p in recs for k, d in recs[p].items() if k.startswith("random_at") for n, v in d.items() if not abs(v) <= th["random_band"]]
    return {"pairs": recs, "named": names_, "outcome": rules.outcome(names_),
            "filed_holds": all(scored[p]["filed"]["reversal"] == "holds" for p in scored), "random_strength1_outside_band": band}  # fmt: skip


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E146")
    saved = json.loads(E143.read_text())["states"]
    inv = json.loads(E136.read_text())
    pairs = {p["id"]: p for p in inv["pairs"] if p["id"] in {**P["pairs"]["scored"], **P["pairs"]["reported"]}}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    job = HF.Job(ST.Net("cuda"), inv, {}, {})
    try:
        write(0, stage0(job, P, saved))
        log("STAGE 0 DONE")
        stage1(job, P, saved, pairs, t_job)
        log("STAGE 1 DONE")
    except Stopped as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())

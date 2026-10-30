"""The C3 step ladder (`--stage ladder`, after results/predictions_ladder.json is filed and committed). Plan:
reports/plans/2026-10-31_plan_e55-c3-ladder.md. Two parts, results/ladder.json written after every state, resumable:

  A. the control and the cheapest reading: the zero update, then C3's 32 saved states re-read with pruning at K = 13 to
     320 (C3's K = 13 and 40 must reproduce run.json; a miss stops the job), scored with the poles' own spread.
  B. the ladder: rank-1, cipher and elicit x seeds 0-2, E58's fine-tune (`keydep.finetune`, E12's checkpoints) read at
     steps 10, 30, 100 and 300 with pruning revival R and reverse fine-tuning's half-life (C3's instruments, called
     unchanged; reverse cap 3000 steps). Rank-1 and elicit at step 10 must reproduce E58's saved states and C3's
     readings (the first fine-tune is rank-1 seed 0). Then R and t_half per step, filed; D, the norm, t_half / D, the
     top-guess half-life and elicit's position, reported.

Exit 3 at the deadline: resubmit the same stage; finished states are skipped (a cut fine-tune reruns; it is exact)."""

from __future__ import annotations

import json
import math
import subprocess
import time
from typing import Any

import numpy as np
import torch
from common import (
    EXIT_RESUME,
    REPO,
    RESULTS,
    Clock,
    load_state,
    log,
    out_of_order,
    rank1_ref,
    state_path,
    stop,
    windows_path,
)
from common import dump as dump_to
from instruments import at_w0, recorded
from measure import reverse_cfg

from commute import elicit as EL
from commute import keydep as KD
from commute import revival_kv as KV
from commute import steering as ST
from commute.data import Windows
from commute.finetune import StopTraining, finetune_to_reference

OUT, PRED_L, C3_RUN = RESULTS / "ladder.json", RESULTS / "predictions_ladder.json", RESULTS / "run.json"
MAPS, SEEDS, STEPS = ("rank1", "cipher", "elicit"), (0, 1, 2), KD.E12_CHECKS
STATE_SECONDS, FT_SECONDS = 180.0, 40.0  # worst-case one state (3000 reverse steps) and one fine-tune, until measured


def load_ladder_predictions() -> dict[str, Any]:
    rel = str(PRED_L.relative_to(REPO))
    dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode
    if not PRED_L.is_file() or dirty or json.loads(PRED_L.read_text()).get("status") != "filed":
        out_of_order(f"{rel} is not filed and committed unchanged")
    return json.loads(PRED_L.read_text())


def dump(out: dict[str, Any]) -> None:
    dump_to(OUT, out)


# ----------------------------------------------------------------------------- readings of one state


def prune(net: ST.Net, ref: KV.Ref, dW: ST.DW, ks: list[int], C3: dict[str, Any], D: float) -> dict[str, Any]:
    rd = KV.prune_readings(net, ref, dW, ks, C3["n_random"], C3["prune_seed"], KV.unit_scores(net, ref, dW))
    return {"prune": rd, **KV.revival(rd, ks, D)}


def reverse_two(net: ST.Net, ref: KV.Ref, dW: ST.DW, w0: torch.nn.Module, train: torch.Tensor, cfg: Any, top_floor: float,
                deadline: float) -> dict[str, Any] | None:  # fmt: skip
    """C3's reverse fine-tune, read after every step on two curves from one forward: kappa (C3's, filed) and kappa_top =
    the mean drop in log p of W0's top guess at the answer position (reported). Runs until kappa has halved and either
    |kappa_top| has halved, |D_top| < top_floor, or 3 x kappa's halving step has passed (bounds the cost of the reported
    curve; t_top is then "censored"); or the cap. None when the deadline cut it."""
    lp0 = ref.lp0["score"]
    a = lp0.argmax(-1, keepdim=True)
    curve, top, end, k_half = [], [], "cap", 0

    def read() -> None:
        lp = KV.answer_lp(net, ref.score, ref.half)
        curve.append(float(KV.kl_rows(lp0, lp).mean()))
        top.append(float((lp0.gather(1, a) - lp.gather(1, a)).mean()))

    def on_step(step: int, _params: Any) -> None:
        nonlocal end, k_half
        read()
        k_half = k_half or (step if curve[-1] <= curve[0] / 2 else 0)
        if k_half and (abs(top[0]) < top_floor or abs(top[-1]) <= abs(top[0]) / 2 or step >= 3 * k_half):
            end = "half"
            raise StopTraining
        if time.monotonic() > deadline:
            end = "deadline"
            raise StopTraining

    n = len(train)
    w = Windows(train, torch.zeros(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), ref.half)
    with net.at(dW):
        read()
        if curve[0] > 0:
            finetune_to_reference(net.model, w0, w, cfg, net.device, on_step, log=lambda _m: None)
    if end == "deadline":
        return None
    t, t_top = KV.half_life(curve), KV.half_life([abs(x) for x in top]) if abs(top[0]) >= top_floor else "void"
    return {"curve": curve, "top": top, "end": end, "D_top": top[0], "t_half": "censored" if t is None else t,
            "t_top": "censored" if t_top is None else t_top}  # fmt: skip


def read_state(net: ST.Net, ref: KV.Ref, w0: torch.nn.Module, train: torch.Tensor, dW: ST.DW, seed: int, L: dict[str, Any],
               C3: dict[str, Any], deadline: float) -> dict[str, Any] | None:  # fmt: skip
    t0, D = time.monotonic(), KV.kappa(net, ref, dW)
    rec: dict[str, Any] = {"D": D, "norm": EL.summed_norm(dW), "void": D < L["constants"]["D_void"]}
    if not rec["void"]:
        rec |= prune(net, ref, dW, L["constants"]["prune_k"], C3, D)
        cfg = reverse_cfg({"reverse": {**C3["reverse"], "cap_steps": L["constants"]["reverse_cap_steps"]}}, seed)
        rv = reverse_two(net, ref, dW, w0, train, cfg, L["constants"]["top_floor"], deadline)
        if rv is None:
            return None
        rec |= {"reverse": rv, "t_half": rv["t_half"], "t_top": rv["t_top"], "D_top": rv["D_top"], "w0_restored": at_w0(net)}
    rec["seconds"] = time.monotonic() - t0
    return rec


def say(n: str, r: dict[str, Any]) -> None:
    if r["void"]:
        log(f"[{n}] D {r['D']:.5f} VOID norm {r['norm']:.4f}")
        return
    log(f"[{n}] D {r['D']:.4f} norm {r['norm']:.4f} R {r['R']:.4f} Delta/D {({k: round(v / r['D'], 4) for k, v in r['Delta'].items()})} "
        f"spread max {max(r['spread'].values()):.4f} | t_half {r['t_half']} D_top {r['D_top']:.4f} t_top {r['t_top']} "
        f"({r['reverse']['end']}, {len(r['reverse']['curve']) - 1} steps) | {r['seconds']:.0f} s")


# ----------------------------------------------------------------------------- scoring (pure, on the records)


def col(st: dict[str, Any], key: str, names: dict[str, list[str]]) -> dict[str, list[Any]]:
    return {m: [None if st[n]["void"] else st[n][key] for n in ns] for m, ns in names.items()}


def num(c: dict[str, list[Any]]) -> dict[str, list[float | None]]:
    return {m: [x if isinstance(x, float) else None for x in v] for m, v in c.items()}


def floor_R(st: dict[str, Any], names: dict[str, list[str]]) -> float:
    """R's noise floor: the largest random-removal spread over the pole states and every K."""
    return max((s for m in KV.POLES for n in names[m] if not st[n]["void"] for s in st[n]["spread"].values()), default=0.0)


def position(means: dict[str, float]) -> float:
    """Elicit's position (reported), nan when a pole has no value."""
    return KV.position(means) if all(m in means for m in KV.POLES) else math.nan


def score_block(st: dict[str, Any], names: dict[str, list[str]], C: dict[str, Any], with_t: bool) -> dict[str, Any]:
    """The filed pole-gap on R (and t_half), elicit's position (reported) and the reported columns of one set of states."""
    g = lambda c, d, f: KV.pole_gap(c, d, f, C["gap_mult"])
    means = lambda c: {m: float(np.mean(KV.clean(v))) for m, v in c.items() if KV.clean(v)}
    R = col(st, "R", names)
    out: dict[str, Any] = {"R": {**g(R, 1, floor_R(st, names)), "means": means(R), "p_elicit": position(means(R))}}
    ks = [str(k) for k in C["prune_k"]]
    out["R_per_K"] = {k: g({m: [None if st[n]["void"] else st[n]["Delta"][k] / st[n]["D"] for n in ns] for m, ns in names.items()}, 1,
                           floor_R(st, names)) for k in ks}  # fmt: skip
    D = col(st, "D", names)
    out["D"] = {**g(D, None, 0.0), "means": means(D)}
    if with_t:
        t, censored = KV.t_half_column(col(st, "t_half", names))
        out["t_half"] = {**g(t, -1, C["t_half_floor"]), "means": means(t), "p_elicit": position(means(t))}
        if censored:
            out["t_half"]["status"] = "censored"
        ratio = {m: [x / d if x is not None and d else None for x, d in zip(t[m], D[m], strict=True)] for m in names}
        tt = num(col(st, "t_top", names))
        out |= {"t_over_D": {**g(ratio, -1, 0.0), "means": means(ratio)}, "t_top": {**g(tt, -1, 0.0), "means": means(tt)},
                "D_top": {"means": means(num(col(st, "D_top", names)))}, "norm": {"means": means(col(st, "norm", names))}}  # fmt: skip
    return out


# ----------------------------------------------------------------------------- part A: C3's states at more K


def part_a(net: ST.Net, ref: KV.Ref, C3: dict[str, Any], L: dict[str, Any], out: dict[str, Any], clock: Clock) -> int | None:
    zero = {k: torch.zeros_like(w) for k, w in net.W0.items()}
    out["zero_D"] = KV.kappa(net, ref, zero)
    log(f"[zero update] D {out['zero_D']!r} (filed: exactly 0)")
    if out["zero_D"] != 0.0:
        return stop(OUT, out, "the zero update's D is not exactly 0")
    run, rec, tol, st = json.loads(C3_RUN.read_text())["states"], recorded(), L["controls"]["c3_reread"], out.setdefault("kreread", {})
    for m in KV.MAPS:
        for s in C3["seeds"]:
            n = f"{m}_s{s}"
            if n in st:
                continue
            if clock.left() < 90:
                log(f"deadline: {n} not started (exit {EXIT_RESUME})")
                return EXIT_RESUME
            if KD.sha256(state_path(m, s)) != rec[n]["ft"]["sha256"]:
                return stop(OUT, out, f"{n}: the state file is not the one its run recorded")
            dW, t0 = load_state(m, s), time.monotonic()
            D = KV.kappa(net, ref, dW)
            r = {"D": D, "void": False, **prune(net, ref, dW, L["constants"]["prune_k"], C3, D), "seconds": time.monotonic() - t0}
            r["checks"] = {"D": abs(D - run[n]["D"]) <= tol["D_rel"] * run[n]["D"],
                           **{k: abs(r["Delta"][k] / D - run[n]["Delta"][k] / run[n]["D"]) <= tol["share_abs"] for k in ("13", "40")}}  # fmt: skip
            st[n] = r
            log(f"[kreread {n}] D {D:.6f} (C3 {run[n]['D']:.6f}) Delta/D {({k: round(v / D, 4) for k, v in r['Delta'].items()})} R {r['R']:.4f} "
                f"spread max {max(r['spread'].values()):.4f} checks {r['checks']} | {r['seconds']:.0f} s")
            dump(out)
            if not all(r["checks"].values()):
                return stop(OUT, out, f"{n}: the re-read does not reproduce C3's run.json: {r['checks']}")
    names = {m: [f"{m}_s{s}" for s in C3["seeds"]] for m in KV.MAPS}
    out["score_kreread"] = score_block(st, names, L["constants"], with_t=False)
    sc = out["score_kreread"]
    log(f"[score kreread] R max over K: {sc['R']['status']} gap {sc['R']['gap']:.4f} (3 x noise {3 * sc['R']['noise']:.4f}) means {sc['R']['means']} "
        f"p_elicit {sc['R']['p_elicit']:.3f} (reported) | per K " + ", ".join(f"{k}: {v['status']} {v['gap']:.4f}/{3 * v['noise']:.4f}" for k, v in sc["R_per_K"].items()))
    dump(out)
    return None


# ----------------------------------------------------------------------------- part B: the ladder


def step10_checks(net: ST.Net, m: str, s: int, dW: ST.DW, r: dict[str, Any], L: dict[str, Any]) -> dict[str, bool]:
    """Rank-1 and elicit at step 10 against E58's saved state (E58's scored step) and C3's readings of it."""
    old, c3 = load_state(m, s), json.loads(C3_RUN.read_text())["states"][f"{m}_s{s}"]
    diff = sum(float((dW[k] - old[k]).double().norm()) for k in old) / EL.summed_norm(old)
    tol = L["controls"]["step10"]
    return {"state": diff <= tol["state_rel"],
            **{k: abs(r["Delta"][k] / r["D"] - c3["Delta"][k] / c3["D"]) <= tol["share_abs"] for k in ("13", "40")},
            "t_half": isinstance(r["t_half"], float) and abs(r["t_half"] - c3["t_half"]) <= tol["t_half_abs"]}  # fmt: skip


def part_b(net: ST.Net, ref: KV.Ref, w0: torch.nn.Module, train: torch.Tensor, C3: dict[str, Any], L: dict[str, Any],
           out: dict[str, Any], clock: Clock) -> int | None:  # fmt: skip
    st, ctx = out.setdefault("ladder", {}), KD.Ctx(net, {}, {}, {}, RESULTS)  # keydep.finetune reads ctx.net only
    for s in SEEDS:
        for m in MAPS:
            todo = [t for t in STEPS if f"{m}_s{s}_t{t}" not in st]
            if not todo:
                continue
            if clock.left() < FT_SECONDS + max([STATE_SECONDS] + [r["seconds"] for r in st.values()]):
                log(f"deadline: {m}_s{s} not started (exit {EXIT_RESUME})")
                return EXIT_RESUME
            ft = KD.finetune(ctx, torch.load(windows_path(m, "trained"), weights_only=False), s, STEPS)
            for t in todo:
                n = f"{m}_s{s}_t{t}"
                r = read_state(net, ref, w0, train, ft.deltas[t], s, L, C3, clock.end)
                if r is None:
                    log(f"deadline: {n}'s reverse fine-tune cut; it reruns on resubmission (exit {EXIT_RESUME})")
                    return EXIT_RESUME
                r["p_heldout"] = ft.evals[t]["p_o"]
                if t == 10 and m != "cipher":
                    r["checks"] = step10_checks(net, m, s, ft.deltas[t], r, L)
                st[n] = r
                say(n, r)
                log(f"[{n}] held-out P(value) {r['p_heldout']:.4f}" + (f" | step-10 control {r['checks']}" if "checks" in r else ""))
                dump(out)
                if not all(r.get("checks", {}).values()):
                    return stop(OUT, out, f"{n}: step 10 does not reproduce E58's state or C3's readings: {r['checks']}")
                if not r.get("w0_restored", True):
                    return stop(OUT, out, f"{n}: the weights are not W0 after its reverse fine-tune")
    per = {str(t): score_block(st, {m: [f"{m}_s{s}_t{t}" for s in SEEDS] for m in MAPS}, L["constants"], with_t=True) for t in STEPS}
    out["score_ladder"] = {"per_step": per, "pass_step": {k: KV.first_ok_step({t: v[k]["status"] for t, v in per.items()}) for k in ("R", "t_half")}}
    for t, sc in per.items():
        log(f"[score step {t}] R {sc['R']['status']} gap {sc['R']['gap']:.4f} (3 x noise {3 * sc['R']['noise']:.4f}) means {sc['R']['means']} p_elicit "
            f"{sc['R']['p_elicit']:.3f} | t_half {sc['t_half']['status']} gap {sc['t_half']['gap']:.3f} (3 x noise {3 * sc['t_half']['noise']:.3f}) "
            f"means {sc['t_half']['means']} p_elicit {sc['t_half']['p_elicit']:.3f} | D {sc['D']['means']} | t/D {sc['t_over_D']['status']} "
            f"{sc['t_over_D']['means']} | t_top {sc['t_top']['status']} {sc['t_top']['means']} D_top {sc['D_top']['means']} | norm {sc['norm']['means']}")
    log(f"[pass step] {out['score_ladder']['pass_step']}")
    dump(out)
    return None


def stage_ladder(device: str, C3: dict[str, Any], header: dict[str, Any], clock: Clock) -> int:
    L = load_ladder_predictions()
    head = {**header, "ladder_predictions_sha256": KD.sha256(PRED_L)}
    old = json.loads(OUT.read_text()) if OUT.is_file() else {}
    if old and old.get("ladder_predictions_sha256") != head["ladder_predictions_sha256"]:
        out_of_order(f"{OUT} was written under other predictions: move it aside by hand")
    out = {**old, **head, "resumed_from": old.get("git_head")}
    net = ST.Net(device)
    ref, train = rank1_ref(net)
    if "score_kreread" not in out and (rc := part_a(net, ref, C3, L, out, clock)) is not None:
        return rc
    rc = part_b(net, ref, KV.w0_copy(net), train, C3, L, out, clock)
    return 0 if rc is None else rc

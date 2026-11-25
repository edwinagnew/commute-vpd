"""restore_anchors (E50) Amendment 1: power checks. Plan: reports/plans/2026-10-29_plan_restore-anchors.md ->
"Amendment 1 (2026-10-29): power checks"; rules, thresholds and expected values: results/predictions_amend1.json (filed
before this code ran, never edited; the run refuses to start without it).

Rows, each placed by do-nothing and read as E50 reads them (run.py: full model, key position, 97 held-out windows):
  <table>/write   E46 Amendment 3's G-matched hand write ({perturb.WRITE_KEY: alpha X}, alpha from stage0b.json); S its own
  <table>/random  the seed-0 random delta-W at E46 Amendment 2's scale; unscored by E50's rule when G < 1 nat
  <table>/<seed>  E50's eight fine-tunes; S as restore_anchors.json recorded it
Sets: every_alive, S, core (the S members in all eight E50 S sets), rest (S minus core), 20 random sets the size of each
of S, core and rest (alive, outside the row's S), and write_only (each matrix's key-position output W0 x0 + dW x: W0 as
one piece per matrix, (U, V) = (W0^T, I), through the same restore hooks).
Controls first, on every table and row: E50's C1 and C2-R2 (the nulls' G against E46 Amendment 3), C5 write_only in a W0
pass; C3 (E50's rho reproduced) as each fine-tune's first reading. Any miss: results written, FAIL logged, exit 1.
Then the nulls, the fine-tunes, the W0 probe, the reading. -> results/amend1.json; log on stdout and results/large/amend1.log.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

import run as E50
import torch
from run import (
    E46,
    EXP,
    LARGE,
    RESULTS,
    SEEDS,
    Run,
    W0Read,
    _finite,
    e46_files,
    lo_ref,
    restored,
    state_control,
    task_control,
)
from torch import Tensor

from commute import perturb as PT
from commute import recruit as R
from commute import recruitment as RT
from commute import regating as G
from commute.finetune import random_direction

PRED, PRED_A, E50_OUT = RESULTS / "predictions.json", RESULTS / "predictions_amend1.json", RESULTS / "restore_anchors.json"
OUT, LOG = RESULTS / "amend1.json", LARGE / "amend1.log"
TABLES, FAMILIES = ("ident32", "ident32b", "rand32", "rand32b"), ("S", "core", "rest")
Comps = dict[str, tuple[Tensor, Tensor]]


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


E50.log = log  # E50's task_control logs its C1 line: into this amendment's log, not E50's run.log


def dump(obj: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(_finite(obj), indent=1, sort_keys=True))


def sel_of(members: dict[str, list[int]]) -> R.Sel:
    return {k: torch.tensor(v, dtype=torch.long) for k, v in members.items()}


def core_of(e50: dict[str, Any]) -> R.Sel:
    common = set.intersection(*(R.members(sel_of(r["members"]["S"])) for r in e50["states"].values()))
    out: dict[str, list[int]] = {}
    for k, i in sorted(common):
        out.setdefault(k, []).append(i)
    return sel_of(out)


def write_only(run: Run) -> tuple[Comps, R.Sel]:
    comps = {k: (w.T.contiguous(), torch.eye(w.shape[1], device=w.device)) for k, w in run.W0.items()}
    return comps, {k: torch.arange(w.shape[1]) for k, w in run.W0.items()}


def sets_of(run: Run, S: R.Sel, core: R.Sel, n_random: int) -> dict[str, R.Sel]:
    outside = {k: a.clone() for k, a in run.alive.items()}
    for k, v in S.items():
        outside[k][v] = False
    base = {"S": S, "core": core, "rest": RT.minus(S, core)}
    sets = {"every_alive": {k: v.nonzero().flatten() for k, v in run.alive.items()}, **base}
    return sets | {f"random_{X}_{i:02d}": R.random_like(sel, outside, i) for X, sel in base.items() for i in range(n_random)}


def hand_write(run: Run, t: G.Task, task: Any, alpha: float) -> dict[str, Tensor]:
    """E46's write (run.py `write_delta`, its X without the report-only fit), read at W0, times stage 0b's alpha."""
    lm = run.model.target_model.lm_head.weight.detach().float()
    A = PT.key_inputs(run.model, t.train.tokens, t.half, list(task.keys))
    return {PT.WRITE_KEY: (alpha * PT.key_write(A, lm[torch.tensor(task.values, device=lm.device)].T)).float()}


def switched_set(run: Run, t: G.Task, dW: dict[str, Tensor]) -> R.Sel:
    full = {k: dW[k] if k in dW else torch.zeros_like(w) for k, w in run.W0.items()}
    return R.switched(G.shift_of(run.model, run.W0, full, t, run.alive)[0], G.THR)


# ----------------------------------------------------------------------------- controls, then readings


def controls(run: Run, A: dict[str, Any], e50: dict[str, Any], core: R.Sel, wo: tuple[Comps, R.Sel],
             out: dict[str, Any]) -> dict[str, Any]:
    """C1 + C5 per table, C2-R2 per row (nulls: G against Amendment 3); exit 1 at the end on any miss."""
    c, tol = A["controls"], run.P["controls"]
    g0a, s0b, a1, a3 = (json.loads((E46 / f).read_text())[k] for f, k in (
        ("stage0a.json", "groups"), ("stage0b.json", "states"), ("amend1.json", "rows"), ("amend3.json", "rows")))
    keep: dict[str, Any] = {}
    for n in TABLES:
        w = torch.load(run.files[f"windows_{n}"], weights_only=False)
        t = G.prepare(run.model, n, w["train"], w["heldout"], w["table"])
        w0, rec = task_control(run, t, g0a[f"e12/{n}"])
        t.base = {"T": t.base["T"]}
        assert w0.ref is not None
        rec |= {"T0_e50": e50["states"][f"{n}/s0"]["T0"], "marker": w["task"].marker,
                "C5": RT.restored_at(run.model, run.W0, *wo, w0.io, t.held.tokens, t.half, t.target, w0.ref)["lo"] - w0.T0}
        rec["ok"] = rec["ok"] and abs(w0.T0 - rec["T0_e50"]) <= tol["C1_T0_tol"] and abs(rec["C5"]) <= tol["R1_w0_tol"]
        out["controls"][n] = rec
        log(f"[control {n}] T0 {w0.T0:.4f} E50 {rec['T0_e50']:.4f} | C5 write_only in W0 {rec['C5']:.2e} | {'ok' if rec['ok'] else 'FAIL'}")
        dws = {f"{n}/{s}": run.state(f"{n}_{s}") for s in SEEDS}
        dws[f"{n}/write"] = hand_write(run, t, w["task"], s0b[f"{n}/write"]["alpha"])
        dws[f"{n}/random"] = PT.scaled(random_direction(dws[f"{n}/s0"], 0), a1[f"{n}/random"]["scale"])
        for key, dW in dws.items():
            kind = key.split("/")[1]
            S = sel_of(e50["states"][key]["members"]["S"]) if kind in SEEDS else switched_set(run, t, dW)
            sets = sets_of(run, S, core, run.P["n_random"])
            ref = g0a[f"e12/{n}"]["states"][kind] if kind in SEEDS else {"T1": w0.T0 + a3[f"{key}_G" if kind == "write" else f"{key}_s0"]["G"]}
            r = state_control(run, t, w0, dW, sets, ref)
            if kind not in SEEDS:  # C2_null: an absolute floor, since the random rows' G is 0.04-0.40 nats
                G_ref = ref["T1"] - w0.T0
                r["pass"]["C2"] = abs(r["T1"] - ref["T1"]) <= max(c["null_G_abs_tol"], c["null_G_rel_tol"] * abs(G_ref))
                r["ok"] = all(r["pass"].values())
            del r["R1"]  # 64 values per row; R1_max is kept
            out["controls"][key] = r
            keep[key] = (t, w0, dW, sets)
            log(f"[control {key}] T1 {r['T1']:.4f} ref {ref['T1']:.4f} | |S| {R.size(S)} | R0 {r['R0']:.2e} | R1 max {r['R1_max']:.2e} "
                f"| R2 {r['R2']:.2e} | {json.dumps(r['pass'])} {'ok' if r['ok'] else 'FAIL'}")
            dump(out)
        w0.ref = None
    out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    dump(out)
    if not out["controls_pass"]:
        log("[controls] FAIL: stop and report")
        sys.exit(1)
    log("[controls] PASS")
    return keep


def read_row(run: Run, t: G.Task, w0: W0Read, dW: dict[str, Tensor], sets: dict[str, R.Sel], wo: tuple[Comps, R.Sel],
             min_gain: float) -> dict[str, Any]:
    W1 = run.weights(dW)
    T1, ref1 = lo_ref(run, W1, t)
    gain = T1 - w0.T0
    nats = {k: T1 - restored(run, W1, sel, w0.io, t, ref1) for k, sel in sets.items()}
    nats["write_only"] = T1 - RT.restored_at(run.model, W1, *wo, w0.io, t.held.tokens, t.half, t.target, ref1)["lo"]
    del ref1
    rho = {k: RT.share(T1, T1 - v, gain if gain >= min_gain else 0.0) for k, v in nats.items()}  # unscored: nan
    rnd = {X: [v for k, v in rho.items() if k.startswith(f"random_{X}_")] for X in FAMILIES}
    stats = {X: {"mean": sum(v) / len(v), "max": max(v), "sd": (sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5}
             for X, v in rnd.items()}
    return {"T0": w0.T0, "T1": T1, "gain": gain, "scorable": gain >= min_gain, "nats": nats, "rho": rho, "random": stats,
            "n": {X: R.size(sets[X]) for X in FAMILIES}, "nonadditive_nats": nats["S"] - nats["core"] - nats["rest"]}


def log_row(key: str, r: dict[str, Any]) -> None:
    named = ("every_alive", "write_only", "S", "core", "rest")
    log(f"[row {key}] T0 {r['T0']:.4f} T1 {r['T1']:.4f} G {r['gain']:.4f} scorable {r['scorable']} | n {json.dumps(r['n'])} | "
        + " ".join(f"{k} {r['rho'][k]:.4f} ({r['nats'][k]:.3f} nats)" for k in named) + f" | S - core - rest {r['nonadditive_nats']:.3f} nats")
    log(f"[row {key}] random " + json.dumps(_finite({X: {k: round(v, 4) for k, v in s.items()} for X, s in r["random"].items()})))


def probe(run: Run, t: G.Task, marker: int) -> float:
    """W0's lo with a demonstration (marker, key, value) written at half-10 .. half-8 of each held-out window."""
    toks, h = t.held.tokens.clone(), t.half
    assert isinstance(t.target, Tensor) and h >= 10
    key = toks[:, h].clone()
    toks[:, h - 10], toks[:, h - 9], toks[:, h - 8] = marker, key, t.target.to(toks.device)[key]
    return RT.readout(run.model, toks, h, t.target, RT.window_logits(run.model, toks))["lo"]


# ----------------------------------------------------------------------------- the filed rules (pure)


def value(r: dict[str, Any], X: str, th: dict[str, Any]) -> float | None:
    rho = r["rho"][X]
    if not r["scorable"] or not math.isfinite(rho):
        return None
    if X in FAMILIES and (r["n"][X] == 0 or not r["random"][X]["max"] <= th["random_bites"]):
        return None
    return rho


def gap_class(high: list[float | None], low: list[float | None], gap: float) -> dict[str, Any]:
    h, lo = [x for x in high if x is not None], [x for x in low if x is not None]
    if len(h) < len(high) or len(lo) < len(low) or not h or not lo:
        return {"class": "unscored"}
    g, rev = min(h) - max(lo), max(h) - min(lo)
    return {"gap": g, "reverse_gap": rev, "class": "separates" if g >= gap else "reversed" if rev <= -gap else "no"}


def verdict(per: dict[str, dict[str, Any]]) -> str:
    c = {v["class"] for v in per.values()}
    return ("unscored" if "unscored" in c else "separates" if c == {"separates"} else "reversed" if c == {"reversed"}
            else "does not separate")


def score(rows: dict[str, Any], A: dict[str, Any], cls: dict[str, str]) -> dict[str, Any]:
    th = A["thresholds"]

    def v(key: str, X: str) -> float | None:
        return value(rows[key], X, th)

    copy, rmap = [n for n in TABLES if cls[n] == "copy"], [n for n in TABLES if cls[n] == "random"]
    fts = [f"{n}/{s}" for n in TABLES for s in SEEDS]
    ea = {s: gap_class([v(f"{n}/{s}", "every_alive") for n in copy], [v(f"{n}/write", "every_alive") for n in TABLES], th["gap"])
          for s in SEEDS}
    sep = [ea[s]["class"] == "separates" for s in SEEDS]
    power = ("unscored" if any(ea[s]["class"] == "unscored" for s in SEEDS) else "powered" if all(sep)
             else "blind" if not any(sep) else "partial")
    sw, snats = [v(f"{n}/write", "S") for n in TABLES], [rows[f"{n}/write"]["nats"]["S"] for n in TABLES]
    s_write = ("unscored" if any(x is None for x in sw) else "low" if all(x is not None and x <= th["S_write_low"] for x in sw)
               else "generic" if all(x >= th["S_write_generic_nats"] for x in snats) else "between")
    sep_x = {X: {s: gap_class([v(f"{n}/{s}", X) for n in copy], [v(f"{n}/{s}", X) for n in rmap], th["gap"]) for s in SEEDS}
             for X in ("every_alive", "write_only", *FAMILIES)}

    def shared(key: str) -> bool:
        r, c = rows[key], v(key, "core")
        return c is not None and r["nats"]["S"] > 0 and r["nats"]["core"] >= th["core_half"] * r["nats"]["S"] and c > r["random"]["core"]["max"]

    sh, kind = all(shared(k) for k in fts), verdict(sep_x["rest"]) == "separates"
    core_rest = ("unscored" if any(v(k, X) is None for k in fts for X in ("core", "rest")) else "both" if sh and kind
                 else "shared_format_part" if sh else "kind_specific" if kind else "neither")
    wo = {k: rows[k]["rho"]["write_only"] - rows[k]["rho"]["every_alive"] for k in fts}
    reported = {"write_low": {n: (x := v(f"{n}/write", "every_alive")) is not None and x <= th["write_low"] for n in TABLES},
                "S_write_low": {n: x is not None and x <= th["S_write_low"] for n, x in zip(TABLES, sw, strict=True)},
                "random_row_unscored": {n: not rows[f"{n}/random"]["scorable"] for n in TABLES},
                "write_only_minus_every_alive": wo,
                "write_only_bound": {k: abs(x) <= th["write_only_bound"] for k, x in wo.items()},
                "random_sets": {k: {X: abs(r["random"][X]["mean"]) <= th["random_abs_mean_max"] and r["random"][X]["max"] <= th["random_max_max"]
                                    for X in FAMILIES} for k, r in rows.items() if r["scorable"]}}
    return {"every_alive_power": power, "every_alive_gap": ea, "S_on_write": s_write, "core_rest": core_rest,
            "separation": {X: {"seeds": p, "verdict": verdict(p)} for X, p in sep_x.items()}, "reported": reported}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for f in (PRED, PRED_A, E50_OUT):
        if not f.is_file():
            sys.exit(f"missing {f}: file the predictions (and run E50) first")
    P, A, e50 = (json.loads(f.read_text()) for f in (PRED, PRED_A, E50_OUT))
    assert all(A["thresholds"][k] == P["thresholds"][k] for k in ("min_gain", "gap", "random_bites")), "E50's thresholds differ"
    assert tuple(P["states"]["classes"]) == TABLES
    files = e46_files(json.loads((E46 / "stage0a.json").read_text()), list(TABLES))
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    t0 = time.time()
    run = Run(args.device, P, files)
    core, wo = core_of(e50), write_only(run)
    log(f"[start] on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'} | core {sorted(R.members(core))}")
    out: dict[str, Any] = {"predictions": str(PRED_A.relative_to(EXP.parent)), "core": {k: v.tolist() for k, v in core.items()},
                           "controls": {}, "rows": {}, "probe": {}}
    keep = controls(run, A, e50, core, wo, out)
    for key in sorted(keep, key=lambda k: k.split("/")[1] in SEEDS):  # the nulls (the power controls) first
        t, w0, dW, sets = keep[key]
        r = out["rows"][key] = read_row(run, t, w0, dW, sets, wo, P["thresholds"]["min_gain"])
        dump(out)
        log_row(key, r)
        if key in e50["states"]:  # C3: E50's two numbers reproduced
            ref = e50["states"][key]["rho"]
            r["C3"] = all(abs(r["rho"][X] - ref[X]) <= A["controls"]["C3_rho_tol"] for X in ("every_alive", "S"))
            if not r["C3"]:
                dump(out)
                log(f"[control {key}] C3 FAIL: rho every_alive {ref['every_alive']:.4f}, S {ref['S']:.4f} in E50: stop and report")
                sys.exit(1)
    for n in TABLES:
        t, w0 = keep[f"{n}/s0"][:2]
        lo = probe(run, t, out["controls"][n]["marker"])
        out["probe"][n] = {"T0": w0.T0, "lo_demo": lo, "gain": lo - w0.T0, "ok": lo - w0.T0 >= A["thresholds"]["probe_min_nats"]}
        log(f"[probe {n}] T0 {w0.T0:.4f} with a demonstration {lo:.4f} (+{lo - w0.T0:.3f} nats)")
    out["score"] = score(out["rows"], A, P["states"]["classes"])
    dump(out)
    log(f"[reading] {json.dumps(_finite({k: v for k, v in out['score'].items() if k != 'reported'}))}")
    log(f"[reported] {json.dumps(_finite(out['score']['reported']))}")
    log(f"[done] ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

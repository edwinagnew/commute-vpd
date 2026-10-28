"""recruitment_test (E38) Amendment 5, the direction-matched restore: does the switched-on set S stand out only because
its trigger output changes point toward the target? Plan: reports/plans/2026-10-28_plan_e38-amend5.md; predictions
results/predictions_amend5.json (filed before this code, never edited; the run refuses to start without it or without
Amendment 3's predictions files, whose controls it reruns). No fine-tuning: Amendment 2's cached states.

Per task-seed state, each alive piece c of matrix k (rank-1 u_c v_c^T) has the first-order take-back
  e_c = mean over windows of ((x1 - x0) . v_c) (g . u_c)   (`recruitment.trigger_grads`, `recruitment.take_back`)
x1 / x0 k's trigger input in W1's / W0's pass, g the gradient of the window's trigger lo with respect to k's trigger
output in W1's plain pass; nats, positive = restoring c takes back gain. On the TRAINING windows it chooses the sets:
D = per matrix, the |S in k| alive pieces not in S of largest e_c (ties: index; short if fewer: the seed is unscored);
C_k = the same by Amendment 4's Delta_c. Amendment 4's C_n and C_m are rebuilt from Delta_c for their membership only.
On the held-out windows, as Amendment 3, rho_r(X) = (T1 - lo(W1, X's trigger outputs restored to W0's)) / G for X in
S, D, C_k; e-hat(X) = sum of held-out e_c over X / G, also over all alive pieces (against Amendment 4's
rho_r(all_alive), read from results/amend4.json). Reported: S's top-direction share. Membership per matrix of S, D,
C_k, C_n, C_m is saved in the results; each state's e_c (both window sets) and Delta_c in results/large/amend5_scores.pt.

One job: the controls first, Amendment 3's R0-R4 (`amend3.controls`) on every state, then R6 on the six task-seed
states: the 10 alive pieces of largest |training e_c|, each restored alone on the training windows, a_c = lo_train(W1)
- lo_train(W1, c restored); pass if >= 8 of the 10 have |e_c| >= 0.02 nats and a_c of e_c's sign. Any fail: results
written, FAIL logged, exit 1, before any share. Then the grid -> results/amend5.json (rewritten after each state; P13
at the end). Every scored number is also in the log (stdout and results/large/amend5.log, emptied at the start of
each run; the control lines keep their '[a3 ...]' tag).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import amend3 as A3
import torch
from amend2 import OUT0, dump, lo_ref, sci
from run import CFG as E38
from run import LARGE, RESULTS, Run, names, read_json
from torch import Tensor

from commute import recruit as R
from commute import recruitment as RT
from commute.faithfulness import model_state
from commute.regating import Task

PRED, OUT, SCORES = RESULTS / "predictions_amend5.json", RESULTS / "amend5.json", LARGE / "amend5_scores.pt"
A4_OUT = RESULTS / "amend4.json"  # rho_r(S) and rho_r(all_alive) to compare with
SETS = ("S", "D", "C_k")  # restored on the held-out windows


@dataclass(frozen=True)
class Cfg:
    r6_n: int = 10  # R6: the pieces of largest |training e_c|, each restored alone
    r6_min_e: float = 0.02  # ... each needs |e_c| >= this, in nats
    r6_agree: int = 8  # ... and at least this many of them a_c of e_c's sign
    rho_S_tol: float = 1e-4  # reported: rho_r(S) against amend4.json
    min_gain: float = 1.0  # a state with G < 1 nat is unscored
    gap_C_k: float = 0.05  # reported: rho_r(C_k) < rho_r(S) - this on emoticon and ' which'
    top_dir_max: float = 0.5  # reported: top-direction share of S below this
    e_hat_factor: float = 2.0  # reported: e-hat(X) within this factor of rho_r(X), X in S, D
    rho_D_min: float = 0.10  # reported: rho_r(D) above this

    def __post_init__(self) -> None:
        assert self.min_gain == E38.min_gain == A3.CFG.min_gain and 0 < self.r6_agree <= self.r6_n
        assert self.e_hat_factor >= 1


CFG = Cfg()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LARGE / "amend5.log").open("a") as f:
        f.write(line + "\n")


A3.log = log  # Amendment 3's control lines go to this run's log


def io_at(run: Run, W: dict[str, Tensor], windows: Tensor, half: int) -> RT.IO:
    """Each matrix's trigger input and output at weights W on `windows`."""
    with model_state(run.model, target=W):
        return RT.trigger_io(run.model, sorted(run.comps), windows, half)


def effect(run: Run, t: Task, W: dict[str, Tensor], windows: Tensor, io0: RT.IO, io1: RT.IO) -> dict[str, Tensor]:
    """e_c of every piece at weights W on `windows` (io0, io1: W0's and W's `trigger_io` on them)."""
    with model_state(run.model, target=W):
        g = RT.trigger_grads(run.model, sorted(run.comps), windows, t.half, int(t.target))
    return RT.take_back(run.comps, io0, io1, g)


# ----------------------------------------------------------------------------- the controls


def r6(run: Run, task: str, name: str, io0: RT.IO) -> dict[str, Any]:
    """R6 on one task-seed state (io0: W0's trigger io on the training windows)."""
    t, W1, keys = run.tasks[task], run.weights(name), sorted(run.comps)
    windows, tok = t.train.tokens, int(t.target)
    e = effect(run, t, W1, windows, io0, io_at(run, W1, windows, t.half))
    with model_state(run.model, target=W1):
        ref = RT.window_logits(run.model, windows)
        lo1 = RT.readout(run.model, windows, t.half, tok, ref)["lo"]
    top = RT.top_abs(e, run.alive, CFG.r6_n)
    a = [lo1 - RT.restored_at(run.model, W1, run.comps, RT.sel_of([(k, c)], keys), io0, windows, t.half, tok, ref)["lo"]
         for k, c, _ in top]
    del ref
    n = RT.r6_agree([x[2] for x in top], a, CFG.r6_min_e)
    rec = {"lo_train_W1": lo1, "pieces": [[k, c, ec, ac] for (k, c, ec), ac in zip(top, a, strict=True)], "agree": n,
           "ok": n >= CFG.r6_agree}
    log(f"[a5 controls {name}] R6 lo_train(W1) {lo1:.4f} | agree {n}/{len(top)} (need {CFG.r6_agree}) | "
        f"{'ok' if rec['ok'] else 'FAIL'}")
    log(f"[a5 controls {name}] R6 (matrix, index, e_c, a_c) "
        f"{json.dumps([[k, c, round(ec, 4), round(ac, 4)] for k, c, ec, ac in rec['pieces']])}")
    return rec


def controls(run: Run, out: dict[str, Any]) -> bool:
    """R0-R4 on every task's states; only if all pass, R6 on the task-seed states."""
    for n in E38.tasks:
        out["controls"][n] = A3.controls(run, n)
        dump(OUT, out)
    out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    if out["controls_pass"]:
        for n in E38.tasks:
            c, t = out["controls"][n], run.tasks[n]
            io0, c["R6"] = io_at(run, run.W0, t.train.tokens, t.half), {}
            for name in names(n)[:-1]:
                c["R6"][name] = r6(run, n, name, io0)
                dump(OUT, out)
            c["ok"] = c["ok"] and all(v["ok"] for v in c["R6"].values())
        out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    dump(OUT, out)
    return out["controls_pass"]


# ----------------------------------------------------------------------------- the grid


def sets_of(run: Run, S: R.Sel, e: dict[str, Tensor], delta: dict[str, Tensor]) -> tuple[dict[str, R.Sel], dict[str, Any]]:
    """S, D, C_k, and Amendment 4's C_n and C_m (membership only), from training e_c and Delta_c; the short flags and
    S's top-direction share."""
    keys = sorted(run.alive)
    counts = {k: int(v.numel()) for k, v in S.items()}
    D, D_short = RT.top_per_matrix(e, run.alive, S, counts)
    Ck, Ck_short = RT.top_per_matrix(delta, run.alive, S, counts)
    ranked = RT.movers(delta, run.alive, S)
    Cn, _ = RT.count_matched(ranked, R.by_layer(S), keys)
    Cm, _ = RT.change_matched(ranked, RT.layer_mass(delta, S), keys)
    share, hits = RT.top_dir_share(e, run.alive, S)
    info = {"D_short_matrices": sorted(k for k, s in D_short.items() if s),
            "C_k_short_matrices": sorted(k for k, s in Ck_short.items() if s),
            "top_dir_share": share, "top_dir_hits": hits, "S_by_matrix": {k: n for k, n in counts.items() if n}}
    return {"S": S, "D": D, "C_k": Ck, "C_n": Cn, "C_m": Cm}, info


def grid_state(run: Run, task: str, name: str, T0: float, io0: dict[str, RT.IO],
               a4: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One task-seed state (io0: W0's trigger io on the 'held' and 'train' windows): its record and its scores."""
    t, W1, S = run.tasks[task], run.weights(name), run.S(name)
    T1, ref1 = lo_ref(run, W1, t)
    gain = T1 - T0
    g = gain if gain >= CFG.min_gain else 0.0  # unscored state: every share nan
    win = {"train": t.train.tokens, "held": t.held.tokens}
    io1 = {w: io_at(run, W1, x, t.half) for w, x in win.items()}
    e = {w: effect(run, t, W1, x, io0[w], io1[w]) for w, x in win.items()}
    delta = RT.output_change(run.comps, io0["train"], io1["train"])
    del io1
    sets, info = sets_of(run, S, e["train"], delta)
    rho = {x: RT.share(T1, A3.restored(run, W1, sets[x], io0["held"], t, ref1), g) for x in SETS}
    del ref1
    alive = {k: v.nonzero().flatten() for k, v in run.alive.items()}
    e_hat = {x: RT.share(RT.summed(e["held"], v), 0.0, g) for x, v in [*((x, sets[x]) for x in SETS), ("all_alive", alive)]}
    rec: dict[str, Any] = {
        "T0": T0, "T1": T1, "gain": gain, "scorable": gain >= CFG.min_gain, "D_short": bool(info["D_short_matrices"]),
        "sets": info, "n": {x: R.size(v) for x, v in sets.items()}, "rho": rho, "e_hat": e_hat,
        "rho_S_amend4": float(a4["states"][name]["rho"]["S"]),
        "rho_all_alive_amend4": float(a4["states"][name]["rho"]["all_alive"]),
        "members": {x: {k: v.tolist() for k, v in sorted(sel.items())} for x, sel in sets.items()}}
    log_state(name, rec)
    return rec, {"e_train": e["train"], "e_held": e["held"], "delta": delta}


def log_state(name: str, rec: dict[str, Any]) -> None:
    s, r, h = rec["sets"], rec["rho"], rec["e_hat"]
    log(f"[a5 grid {name}] T0 {rec['T0']:.4f} T1 {rec['T1']:.4f} G {rec['gain']:.4f} scorable {rec['scorable']} | n "
        f"{json.dumps(rec['n'])} | D short {s['D_short_matrices']} C_k short {s['C_k_short_matrices']}")
    log(f"[a5 grid {name}] S by matrix {json.dumps(s['S_by_matrix'])} | top-direction share of S "
        f"{s['top_dir_share']:.4f} hits {json.dumps(s['top_dir_hits'])}")
    log(f"[a5 grid {name}] rho_r S {r['S']:.4f} (Amendment 4 {rec['rho_S_amend4']:.4f}, diff "
        f"{sci(r['S'] - rec['rho_S_amend4'])}) D {r['D']:.4f} C_k {r['C_k']:.4f}")
    log(f"[a5 grid {name}] e-hat S {h['S']:.4f} D {h['D']:.4f} C_k {h['C_k']:.4f} all_alive {h['all_alive']:.4f} "
        f"(Amendment 4's rho_r(all_alive) {rec['rho_all_alive_amend4']:.4f})")


def within(a: float, b: float, f: float) -> bool:
    """a and b of one sign, the larger at most f times the smaller in size."""
    return a * b > 0 and max(abs(a), abs(b)) <= f * min(abs(a), abs(b))


def reported(states: dict[str, Any]) -> dict[str, Any]:
    """The filed 'reported' values against their expected values (emoticon and ' which' where the file says so)."""
    on = {k: v for k, v in states.items() if k.split("/")[0] in A3.CFG.p10_tasks}
    return {"rho_S_matches_amend4": {k: abs(v["rho"]["S"] - v["rho_S_amend4"]) <= CFG.rho_S_tol for k, v in states.items()},
            "rho_C_k_below_rho_S_minus_0.05": {k: v["rho"]["C_k"] < v["rho"]["S"] - CFG.gap_C_k for k, v in on.items()},
            "top_dir_share_below_0.5": {k: v["sets"]["top_dir_share"] < CFG.top_dir_max for k, v in on.items()},
            "e_hat_within_factor_2": {k: {x: within(v["e_hat"][x], v["rho"][x], CFG.e_hat_factor) for x in ("S", "D")}
                                      for k, v in on.items()},
            "rho_D_above_0.10": {k: v["rho"]["D"] > CFG.rho_D_min for k, v in on.items()},
            "e_hat_all_alive_vs_amend4": {k: {"e_hat": v["e_hat"]["all_alive"], "rho_r_amend4": v["rho_all_alive_amend4"]}
                                          for k, v in states.items()}}


def grid(run: Run, out: dict[str, Any], a4: dict[str, Any]) -> None:
    by_task: dict[str, dict[str, Any]] = {}
    scores: dict[str, Any] = {}
    for n in E38.tasks:
        t = run.tasks[n]
        T0, ref0, io_held = A3.w0_pass(run, t)
        del ref0
        io0 = {"held": io_held, "train": io_at(run, run.W0, t.train.tokens, t.half)}
        for name in names(n)[:-1]:
            rec, scores[name] = grid_state(run, n, name, T0, io0, a4)
            out["states"][name] = by_task.setdefault(n, {})[name[-1]] = rec
            dump(OUT, out)
            torch.save(scores, SCORES)
        del io0, io_held
    out["P13"] = RT.p13_reading(by_task)
    out["reported"] = reported(out["states"])
    dump(OUT, out)
    p = out["P13"]
    log(f"[a5] P13 seeds {json.dumps(p['seed_class'])} | tasks {json.dumps(p['task_class'])} | scored tasks "
        f"{p['n_scored_tasks']} | reading '{p['reading']}'")
    log(f"[a5] reported {json.dumps(out['reported'])}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for p in (PRED, A3.PRED, A3.PRED_POOL, A3.PRED_R4, A3.PRED_R4B):
        if not p.is_file():
            sys.exit(f"file the predictions first: {p}")
    s0, a4 = read_json(OUT0), read_json(A4_OUT)
    if not (s0 and s0.get("pass") is True):
        sys.exit(f"Amendment 2's stage 0 has not passed ({OUT0}): its cached states are not checked")
    if not (a4 and a4.get("controls_pass") is True):
        sys.exit(f"Amendment 4's results are missing or failed their controls ({A4_OUT})")
    t0 = time.time()
    (LARGE / "amend5.log").write_text("")  # this run's log only
    run = Run(args.device)
    log(f"[a5 start] on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}; cfg {CFG}; a3 cfg {A3.CFG}")
    run.load_states(s0)
    out: dict[str, Any] = {"cfg": dataclasses.asdict(CFG), "controls": {}, "states": {}}
    if not controls(run, out):
        log(f"[a5 controls] FAIL: stop and report ({time.time() - t0:.0f} s)")
        sys.exit(1)
    log("[a5 controls] PASS")
    grid(run, out, a4)
    log(f"[a5 done] ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

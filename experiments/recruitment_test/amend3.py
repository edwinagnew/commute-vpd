"""recruitment_test (E38) Amendment 3, the restore test: in the fine-tuned model, put a set's outputs at the trigger back
to what they were in the original model; how much of the gain goes? Plan: reports/plans/2026-10-27_plan_e38-amend3.md;
predictions results/predictions_amend3.json (filed before this code, never edited; the run refuses to start without
it). No fine-tuning: Amendment 2's cached states (results/large/states/, sha256 against results/amend2_stage0.json).

Restoring X at W1 (`recruitment.restore_hooks`): at the trigger position only, each decomposed matrix's output gains
sum_{c in X} ((x0 - x) . v_c) u_c, x its input in this pass, x0 its input at the trigger in W0's pass on the same
window (`recruitment.trigger_io`). Per task-seed state, G = T1 - T0 (G < 1 nat: unscored, every share nan) and
  rho_r(X) = (T1 - lo(W1, X restored)) / G
for X in S; null_minus_S (the ' http' state's recruited set minus S); random_i = recruit.random_like(S, alive, i); and
active_i = per layer, as many pieces as S has there, drawn with seed i from the alive pieces not in S with W0 mean
trigger importance task.m0 >= 0.5 (>= 0.2 for the whole state if some layer's pool at 0.5 is smaller than S's count
there; threshold, pool sizes and realized sizes recorded), i = 0..19. A state whose pool is still too small at 0.2
in some layer draws from the whole pool and is unscored for P9 (results/predictions_amend3_pool.json, filed before
any run; the run refuses to start without it too).

One job: the controls first, R0-R3 on all nine states and R4 on the six task-seed states (any fail: results written,
FAIL logged, exit 1), then the grid -> results/amend3.json (rewritten after each state; P9, P10 at the end). Every
scored number is also in the log (stdout and results/large/amend3.log, emptied at the start of each run).
  R0  restore the empty set at W1:                     |lo - T1| <= 1e-5 (expected 0: the hook adds an exact 0)
  R1  restore S in a W0 pass (x = x0):                 |lo - T0| <= 1e-4 (expected 0)
  R2  every matrix's whole trigger output set to W0's: |lo - T0| <= 1e-2 (expected ~0: the trigger's residual is W0's)
  R3  restore S at W1: max |logit diff| before the trigger <= 1e-4 (expected 0: causal)
  R4  restore S at W1 with x0 := 0 (the hook then subtracts S's trigger outputs), task-seed states only: |lo -
      lo of recruitment.removed_at(S) at W1| <= 0.05 and <= 0.02 x |T1 - that lo| (predictions_amend3_r4.json, tolerance
      amended after job 38950 in predictions_amend3_r4b.json: the two paths round differently under bf16)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
from amend2 import OUT0, dump, lo_ref, sci
from run import CFG as E38
from run import LARGE, RESULTS, Run, names, read_json
from torch import Tensor

from commute import recruit as R
from commute import recruitment as RT
from commute.faithfulness import model_state
from commute.regating import Task

PRED, OUT = RESULTS / "predictions_amend3.json", RESULTS / "amend3.json"
PRED_POOL = RESULTS / "predictions_amend3_pool.json"  # the short-pool rule, filed before any run
PRED_R4 = RESULTS / "predictions_amend3_r4.json"  # control R4, filed before any run
PRED_R4B = RESULTS / "predictions_amend3_r4b.json"  # R4's tolerance, amended after job 38950 (controls only)


@dataclass(frozen=True)
class Cfg:
    n_random: int = 20  # random_like seeds 0..19
    n_active: int = 20  # active_like seeds 0..19
    m0_active: float = 0.5  # the active pool: alive, not in S, task.m0 >= 0.5 ...
    m0_active_low: float = 0.2  # ... or >= 0.2 for a state where some layer's pool at 0.5 is smaller than S's count
    empty_tol: float = 1e-5  # R0
    w0_tol: float = 1e-4  # R1
    whole_tol: float = 1e-2  # R2
    pre_tol: float = 1e-4  # R3
    removal_tol: float = 0.05  # R4, nats (results/predictions_amend3_r4b.json: bf16 noise floor ~0.02)
    removal_rel_tol: float = 0.02  # R4, as a share of the removal's own effect T1 - lo(removed_at)
    min_gain: float = 1.0  # a state with G < 1 nat is unscored
    p10_tasks: tuple[str, ...] = ("emoticon", "r1_t534_v369")  # emoticon and ' which'; r2_t359_v816 is ' we'
    rho_S_range: tuple[float, float] = (0.10, 0.50)  # reported: rho_r(S) on p10_tasks in range; ' we' below 0.10

    def __post_init__(self) -> None:
        assert self.n_random == self.n_active == E38.n_random == 20 and self.min_gain == E38.min_gain
        assert self.m0_active_low < self.m0_active and set(self.p10_tasks) < set(E38.tasks)


CFG = Cfg()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LARGE / "amend3.log").open("a") as f:
        f.write(line + "\n")


def w0_pass(run: Run, t: Task) -> tuple[float, list[Tensor], RT.IO]:
    """W0 on the held-out windows: T0, its logits, and each matrix's input and output at the trigger per batch."""
    T0, ref0 = lo_ref(run, run.W0, t)
    with model_state(run.model, target=run.W0):
        io = RT.trigger_io(run.model, sorted(run.comps), t.held.tokens, t.half)
    return T0, ref0, io


def restored(run: Run, W: dict[str, Tensor], sel: R.Sel, io: RT.IO, t: Task, ref: list[Tensor],
             whole: bool = False) -> float:
    """lo of the full model at W with `sel`'s trigger outputs restored to W0's (all of them with `whole`)."""
    return RT.restored_at(run.model, W, run.comps, sel, io, t.held.tokens, t.half, int(t.target), ref, whole)["lo"]


# ----------------------------------------------------------------------------- the controls R0-R4


def pre_diff(run: Run, W: dict[str, Tensor], sel: R.Sel, io: RT.IO, t: Task, ref: list[Tensor]) -> dict[str, float]:
    """Restoring `sel` at W: max |logit - ref| before the trigger (R3) and at it (that it bites)."""
    with model_state(run.model, target=W), RT.restore_hooks(run.model, run.comps, sel, io, t.half):
        lg = RT.window_logits(run.model, t.held.tokens)
    h = t.half
    return {"pre": max(float((a[:, :h] - b[:, :h]).abs().max()) for a, b in zip(lg, ref, strict=True)),
            "at": max(float((a[:, h] - b[:, h]).abs().max()) for a, b in zip(lg, ref, strict=True))}


def removal_check(run: Run, W: dict[str, Tensor], S: R.Sel, io: RT.IO, t: Task, ref: list[Tensor]) -> dict[str, float]:
    """R4: restoring S at W with every cached x0 set to 0 subtracts S's trigger outputs; its lo against Amendment 2's
    trigger-only removal of S at W (`recruitment.removed_at`)."""
    zero = {k: [(torch.zeros_like(x0), y0) for x0, y0 in v] for k, v in io.items()}
    a = restored(run, W, S, zero, t, ref)
    b = RT.removed_at(run.model, W, run.comps, S, t.held.tokens, t.half, int(t.target), ref)["lo"]
    return {"lo_restore_x0_0": a, "lo_removed_at": b, "diff": a - b}


def controls(run: Run, task: str) -> dict[str, Any]:
    """R0-R3 on the task's three states (the null's own recruited set as its S), R4 on its two task-seed states."""
    t = run.tasks[task]
    T0, ref0, io = w0_pass(run, t)
    empty = {k: torch.zeros(0, dtype=torch.long) for k in run.comps}
    out: dict[str, Any] = {"T0": T0, "states": {}}
    for name in names(task):
        W1, S = run.weights(name), run.S(name)
        T1, ref1 = lo_ref(run, W1, t)
        rec: dict[str, Any] = {"T1": T1, "n_S": R.size(S), "R0": restored(run, W1, empty, io, t, ref1) - T1,
                               "R1": restored(run, run.W0, S, io, t, ref0) - T0,
                               "R2": restored(run, W1, empty, io, t, ref1, whole=True) - T0,
                               "R3": pre_diff(run, W1, S, io, t, ref1)}
        rec["pass"] = {"R0": abs(rec["R0"]) <= CFG.empty_tol, "R1": abs(rec["R1"]) <= CFG.w0_tol,
                       "R2": abs(rec["R2"]) <= CFG.whole_tol, "R3": rec["R3"]["pre"] <= CFG.pre_tol}
        if not name.endswith("null"):
            rec["R4"] = removal_check(run, W1, S, io, t, ref1)
            d = abs(rec["R4"]["diff"])
            rec["pass"]["R4"] = d <= CFG.removal_tol and d <= CFG.removal_rel_tol * abs(T1 - rec["R4"]["lo_removed_at"])
        del ref1
        rec["ok"] = all(rec["pass"].values())
        out["states"][name] = rec
        r4 = (f"lo {rec['R4']['lo_restore_x0_0']:.4f} removed_at {rec['R4']['lo_removed_at']:.4f} diff "
              f"{sci(rec['R4']['diff'])}" if "R4" in rec else "n/a (null)")
        log(f"[a3 controls {name}] T0 {T0:.4f} T1 {T1:.4f} |S| {rec['n_S']} | R0 lo - T1 {sci(rec['R0'])} | R1 lo - T0 "
            f"{sci(rec['R1'])} | R2 lo - T0 {sci(rec['R2'])} | R3 max |logit diff| pre {sci(rec['R3']['pre'])} at "
            f"{rec['R3']['at']:.3f} | R4 {r4} | {json.dumps(rec['pass'])} {'ok' if rec['ok'] else 'FAIL'}")
    out["ok"] = all(v["ok"] for v in out["states"].values())
    return out


# ----------------------------------------------------------------------------- the grid


def active_sets(run: Run, t: Task, S: R.Sel) -> tuple[list[R.Sel], dict[str, Any]]:
    """The 20 active sets, matched to S per layer, and how they were drawn."""
    by, thr = R.by_layer(S), CFG.m0_active
    pools = {thr: RT.active_pool(S, run.alive, t.m0, thr)}
    def short() -> bool:
        return any(len(pools[thr].get(lay, [])) < n for lay, n in by.items())

    if short():  # 0.2 even if still short: draws take all of it, and P9 leaves the state unscored
        thr = CFG.m0_active_low
        pools[thr] = RT.active_pool(S, run.alive, t.m0, thr)
    sets = [RT.active_like(S, pools[thr], i) for i in range(CFG.n_active)]
    return sets, {"threshold": thr, "short": short(), "S_by_layer": by, "sizes": [R.size(x) for x in sets],
                  "matched": all(R.by_layer(x) == by for x in sets),
                  "pool_by_layer": {str(k): {lay: len(p) for lay, p in v.items()} for k, v in pools.items()}}


def grid_state(run: Run, task: str, name: str, T0: float, io: RT.IO) -> dict[str, Any]:
    t, W1, S = run.tasks[task], run.weights(name), run.S(name)
    T1, ref1 = lo_ref(run, W1, t)
    gain = T1 - T0
    g = gain if gain >= CFG.min_gain else 0.0  # unscored state: every share nan
    active, info = active_sets(run, t, S)
    null = run.S(f"{task}/null")
    sets: dict[str, Any] = {"S": S, "null_minus_S": RT.minus(null, S), "active": active,
                            "random": [R.random_like(S, run.alive, i) for i in range(CFG.n_random)]}

    def rho(sel: R.Sel) -> float:
        return RT.share(T1, restored(run, W1, sel, io, t, ref1), g)

    rho_r = {k: [rho(x) for x in v] if isinstance(v, list) else rho(v) for k, v in sets.items()}
    rec = {"T0": T0, "T1": T1, "gain": gain, "scorable": gain >= CFG.min_gain, "active": info, "rho": rho_r,
           "n": {"S": R.size(S), "null": R.size(null), "null_minus_S": R.size(sets["null_minus_S"])}}
    log_state(name, rec)
    return rec


def log_state(name: str, rec: dict[str, Any]) -> None:
    a, r = rec["active"], rec["rho"]
    log(f"[a3 grid {name}] T0 {rec['T0']:.4f} T1 {rec['T1']:.4f} G {rec['gain']:.4f} scorable {rec['scorable']} | n "
        f"{json.dumps(rec['n'])} | active threshold {a['threshold']} short {a['short']} S by layer {json.dumps(a['S_by_layer'])} pool by "
        f"layer {json.dumps(a['pool_by_layer'])} sizes {a['sizes']} matched {a['matched']}")
    beats = {k: sum(r["S"] > x for x in r[k]) for k in ("random", "active")}
    log(f"[a3 grid {name}] rho_r S {r['S']:.4f} null_minus_S {r['null_minus_S']:.4f} | " + " | ".join(
        f"{k} mean {sum(r[k]) / len(r[k]):.4f} max {max(r[k]):.4f}, S above {beats[k]}/{len(r[k])}" for k in ("random", "active")))
    log(f"[a3 grid {name}] random rho_r {json.dumps([round(x, 4) for x in r['random']])} | active rho_r "
        f"{json.dumps([round(x, 4) for x in r['active']])}")


def reported(states: dict[str, Any]) -> dict[str, Any]:
    """The filed 'reported' values: rho_r(S) in [0.10, 0.50] on emoticon and ' which', below 0.10 on ' we'; the active
    pool threshold 0.5 on every state."""
    lo, hi = CFG.rho_S_range
    return {"rho_S_as_expected": {k: lo <= v["rho"]["S"] <= hi if k.split("/")[0] in CFG.p10_tasks else v["rho"]["S"] < lo
                                  for k, v in states.items()},
            "active_threshold_0.5": {k: v["active"]["threshold"] == CFG.m0_active for k, v in states.items()}}


def grid(run: Run, out: dict[str, Any]) -> None:
    by_task: dict[str, dict[str, Any]] = {}
    for n in E38.tasks:
        T0, ref0, io = w0_pass(run, run.tasks[n])
        del ref0
        for name in names(n)[:-1]:
            out["states"][name] = by_task.setdefault(n, {})[name[-1]] = grid_state(run, n, name, T0, io)
            dump(OUT, out)
    out.update(RT.amend3_readings(by_task, CFG.p10_tasks))
    out["reported"] = reported(out["states"])
    dump(OUT, out)
    p9 = out["P9"]
    log(f"[a3] P9 seeds {json.dumps(p9['seed_class'])} | tasks {json.dumps(p9['task_class'])} | reading '{p9['reading']}'")
    log(f"[a3] P10 {json.dumps(out['P10'])} | reported {json.dumps(out['reported'])}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for p in (PRED, PRED_POOL, PRED_R4, PRED_R4B):
        if not p.is_file():
            sys.exit(f"file the predictions first: {p}")
    s0 = read_json(OUT0)
    if not (s0 and s0.get("pass") is True):
        sys.exit(f"Amendment 2's stage 0 has not passed ({OUT0}): its cached states are not checked")
    t0 = time.time()
    (LARGE / "amend3.log").write_text("")  # this run's log only
    run = Run(args.device)
    log(f"[a3 start] on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}; cfg {CFG}")
    run.load_states(s0)
    out: dict[str, Any] = {"cfg": dataclasses.asdict(CFG), "controls": {}, "states": {}}
    for n in E38.tasks:
        out["controls"][n] = controls(run, n)
        dump(OUT, out)
    out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    dump(OUT, out)
    if not out["controls_pass"]:
        log(f"[a3 controls] FAIL: stop and report ({time.time() - t0:.0f} s)")
        sys.exit(1)
    log("[a3 controls] PASS")
    grid(run, out)
    log(f"[a3 done] ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

"""recruitment_test (E38) Amendment 4, the change-matched restore: is the switched-on set S special, or just the set
whose trigger outputs changed most? Plan: reports/plans/2026-10-28_plan_e38-amend4.md; predictions
results/predictions_amend4.json (filed before this code, never edited; the run refuses to start without it or without
Amendment 3's predictions files, whose controls it reruns). No fine-tuning: Amendment 2's cached states.

Per task-seed state, on the task's TRAINING windows (those S was chosen on), each alive piece c of matrix k has the
output change Delta_c = mean over windows of |(x1 - x0) . v_c| * ||u_c||, x1 / x0 k's trigger input in W1's / W0's
pass (`recruitment.output_change`, fp32). Per layer, from the alive pieces not in S: C_n = the |S in that layer| top
movers (ties: matrix name, then index); C_m = top movers until their summed Delta_c reaches S's in that layer (the
whole layer if never: short; empty where S has none or S's sum is 0). On the held-out windows, as Amendment 3,
rho_r(X) = (T1 - lo(W1, X's trigger outputs restored to W0's)) / G for X in S, C_n, C_m and all_alive (S included).
Reported: Delta mass of C_n over S, S's share of the top-|S| movers per layer, C_m's sizes and short layers.
Emoticon, both seeds: L3 = the layer-3 alive pieces not in S with task.m0 >= 0.2; rest = active_like(S, the pool at
0.2, seed 0) without its layer-3 members; each L3 member alone (singles); |L3| random singles from layer 3's alive
pieces not in S with task.m0 < 0.2, drawn with seed 0.

One job: the controls first on every state, Amendment 3's R0-R4 (`amend3.controls`) and per task R5: Delta_c with W1
:= W0 is <= 1e-6 on every alive piece; any fail: results written, FAIL logged, exit 1, before any share. Then the
grid -> results/amend4.json (rewritten after each state; P11, P12 at the end). Every scored number is also in the log
(stdout and results/large/amend4.log, emptied at the start of each run; the control lines keep their '[a3 ...]' tag).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import amend3 as A3
import torch
from amend2 import OUT0, dump, lo_ref, sci
from run import CFG as E38
from run import LARGE, RESULTS, Run, names, read_json

from commute import recruit as R
from commute import recruitment as RT
from commute.faithfulness import model_state
from commute.regating import Task

PRED, OUT = RESULTS / "predictions_amend4.json", RESULTS / "amend4.json"


@dataclass(frozen=True)
class Cfg:
    r5_tol: float = 1e-6  # R5: max Delta_c over alive pieces with W1 := W0
    rho_S_tol: float = 1e-4  # reported: rho_r(S) against amend3.json
    m0_l3: float = 0.2  # L3 and rest: task.m0 >= this; random singles: below it
    l3_task: str = "emoticon"
    l3_layer: str = "3"
    seed: int = 0  # rest's active_like draw and the random singles' draw
    min_gain: float = 1.0  # a state with G < 1 nat is unscored
    all_range: tuple[float, float] = (0.3, 0.9)  # reported: rho_r(all_alive) on emoticon and ' which'
    top_share_min: float = 0.5  # reported: S's share of the top movers on emoticon and ' which'
    l3_max, rest_abs = -0.15, 0.05  # reported: rho_r(L3) <= -0.15 and |rho_r(rest)| <= 0.05 on emoticon

    def __post_init__(self) -> None:
        assert self.min_gain == E38.min_gain == A3.CFG.min_gain and self.m0_l3 == A3.CFG.m0_active_low
        assert self.l3_task in A3.CFG.p10_tasks


CFG = Cfg()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LARGE / "amend4.log").open("a") as f:
        f.write(line + "\n")


A3.log = log  # Amendment 3's control lines go to this run's log


def train_io(run: Run, t: Task, W: dict[str, Any]) -> RT.IO:
    """Each matrix's trigger input and output at weights W on the task's training windows."""
    with model_state(run.model, target=W):
        return RT.trigger_io(run.model, sorted(run.comps), t.train.tokens, t.half)


def r5(run: Run, task: str) -> dict[str, Any]:
    """R5: Delta_c from two W0 passes, max over alive pieces (expected 0)."""
    t = run.tasks[task]
    d = RT.output_change(run.comps, train_io(run, t, run.W0), train_io(run, t, run.W0))
    mx = max(float(torch.where(run.alive[k], v, 0.0).max()) for k, v in d.items())
    return {"max": mx, "ok": mx <= CFG.r5_tol}


def controls(run: Run, out: dict[str, Any]) -> bool:
    for n in E38.tasks:
        c = out["controls"][n] = A3.controls(run, n)
        c["R5"] = r5(run, n)
        c["ok"] = c["ok"] and c["R5"]["ok"]
        log(f"[a4 controls {n}] R5 max Delta_c at W1 := W0 {sci(c['R5']['max'])} | {'ok' if c['R5']['ok'] else 'FAIL'}")
        dump(OUT, out)
    out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    dump(OUT, out)
    return out["controls_pass"]


# ----------------------------------------------------------------------------- the grid


def change_sets(run: Run, delta: dict[str, Any], S: R.Sel) -> tuple[dict[str, R.Sel], dict[str, Any]]:
    """S, C_n, C_m and all_alive, and the reported numbers on Delta."""
    keys = sorted(run.alive)
    ranked = RT.movers(delta, run.alive, S)
    by, mass_S = R.by_layer(S), RT.layer_mass(delta, S)
    Cn, n_short = RT.count_matched(ranked, by, keys)
    Cm, m_short = RT.change_matched(ranked, mass_S, keys)
    share, hits = RT.top_share(delta, run.alive, S)
    mass_Cn, tot = RT.layer_mass(delta, Cn), sum(mass_S.values())
    info = {"S_by_layer": by, "C_n_by_layer": R.by_layer(Cn), "C_m_by_layer": R.by_layer(Cm), "C_n_short": n_short,
            "C_m_short": m_short, "pool_by_layer": {lay: len(p) for lay, p in ranked.items()}, "mass_S": mass_S,
            "mass_C_n": mass_Cn, "mass_C_m": RT.layer_mass(delta, Cm),
            "mass_ratio_C_n": sum(mass_Cn.values()) / tot if tot > 0 else float("nan"),
            "top_share": share, "top_hits": hits}
    sets = {"S": S, "C_n": Cn, "C_m": Cm, "all_alive": {k: v.nonzero().flatten() for k, v in run.alive.items()}}
    return sets, info


def l3_arms(run: Run, t: Task, S: R.Sel, rho: Callable[[R.Sel], float]) -> dict[str, Any]:
    """Emoticon: rho_r of L3, rest, each L3 member alone, and as many random layer-3 singles below m0 0.2."""
    keys, lay = sorted(run.alive), CFG.l3_layer
    pool = RT.active_pool(S, run.alive, t.m0, CFG.m0_l3)
    L3 = pool.get(lay, [])
    low = RT.layer_pieces(S, {k: run.alive[k] & (t.m0[k].float().cpu() < CFG.m0_l3) for k in run.alive}).get(lay, [])
    gen = torch.Generator().manual_seed(CFG.seed)
    rnd = [low[j] for j in torch.randperm(len(low), generator=gen)[: len(L3)].tolist()]
    rest = {k: v if RT.layer_of(k) != lay else v[:0] for k, v in RT.active_like(S, pool, CFG.seed).items()}
    return {"n": {"L3": len(L3), "rest": R.size(rest), "low_pool": len(low), "random_singles": len(rnd)},
            "rho": {"L3": rho(RT.sel_of(L3, keys)), "rest": rho(rest)},
            "singles": [[k, c, rho(RT.sel_of([(k, c)], keys))] for k, c in L3],
            "random_singles": [[k, c, rho(RT.sel_of([(k, c)], keys))] for k, c in rnd]}


def grid_state(run: Run, task: str, name: str, T0: float, io: RT.IO, io0_train: RT.IO) -> dict[str, Any]:
    t, W1, S = run.tasks[task], run.weights(name), run.S(name)
    T1, ref1 = lo_ref(run, W1, t)
    gain = T1 - T0
    g = gain if gain >= CFG.min_gain else 0.0  # unscored state: every share nan
    delta = RT.output_change(run.comps, io0_train, train_io(run, t, W1))
    sets, info = change_sets(run, delta, S)

    def rho(sel: R.Sel) -> float:
        return RT.share(T1, A3.restored(run, W1, sel, io, t, ref1), g)

    a3 = read_json(A3.OUT)
    rec: dict[str, Any] = {"T0": T0, "T1": T1, "gain": gain, "scorable": gain >= CFG.min_gain, "change": info,
                           "n": {k: R.size(v) for k, v in sets.items()}, "rho": {k: rho(v) for k, v in sets.items()},
                           "rho_S_amend3": float(a3["states"][name]["rho"]["S"]) if a3 else float("nan")}
    log_state(name, rec)
    if task == CFG.l3_task:
        rec["l3"] = l3_arms(run, t, S, rho)
        log_l3(name, rec["l3"])
    return rec


def log_state(name: str, rec: dict[str, Any]) -> None:
    c, r = rec["change"], rec["rho"]
    def j(x: Any) -> str:
        return json.dumps(x if not isinstance(x, dict) else {k: round(v, 4) if isinstance(v, float) else v for k, v in x.items()})

    log(f"[a4 grid {name}] T0 {rec['T0']:.4f} T1 {rec['T1']:.4f} G {rec['gain']:.4f} scorable {rec['scorable']} | n {j(rec['n'])}")
    log(f"[a4 grid {name}] by layer: S {j(c['S_by_layer'])} C_n {j(c['C_n_by_layer'])} short {j(c['C_n_short'])} | C_m "
        f"{j(c['C_m_by_layer'])} short {j(c['C_m_short'])} | pool (alive, not S) {j(c['pool_by_layer'])}")
    log(f"[a4 grid {name}] Delta mass: S {j(c['mass_S'])} C_n {j(c['mass_C_n'])} C_m {j(c['mass_C_m'])} | ratio C_n/S "
        f"{c['mass_ratio_C_n']:.4f} | S's share of top-|S| movers {c['top_share']:.4f} hits {j(c['top_hits'])}")
    log(f"[a4 grid {name}] rho_r S {r['S']:.4f} (Amendment 3 {rec['rho_S_amend3']:.4f}, diff "
        f"{sci(r['S'] - rec['rho_S_amend3'])}) C_n {r['C_n']:.4f} C_m {r['C_m']:.4f} all_alive {r['all_alive']:.4f}")


def log_l3(name: str, a: dict[str, Any]) -> None:
    s, rs = [x[2] for x in a["singles"]], [x[2] for x in a["random_singles"]]
    log(f"[a4 L3 {name}] n {json.dumps(a['n'])} | rho_r L3 {a['rho']['L3']:.4f} rest {a['rho']['rest']:.4f} | most "
        f"negative single {min(s, default=float('nan')):.4f} random single {min(rs, default=float('nan')):.4f}")
    log(f"[a4 L3 {name}] singles {json.dumps([round(x, 4) for x in s])}")
    log(f"[a4 L3 {name}] random singles {json.dumps([round(x, 4) for x in rs])}")


def reported(states: dict[str, Any]) -> dict[str, Any]:
    """The filed 'reported' values against their expected values."""
    lo, hi = CFG.all_range
    on = {k: v for k, v in states.items() if k.split("/")[0] in A3.CFG.p10_tasks}
    return {"mass_ratio_C_n_below_1": {k: v["change"]["mass_ratio_C_n"] < 1 for k, v in states.items()},
            "top_share_at_least_0.5": {k: v["change"]["top_share"] >= CFG.top_share_min for k, v in on.items()},
            "rho_all_alive_in_range": {k: lo <= v["rho"]["all_alive"] <= hi for k, v in on.items()},
            "L3_and_rest": {k: v["l3"]["rho"]["L3"] <= CFG.l3_max and abs(v["l3"]["rho"]["rest"]) <= CFG.rest_abs
                            for k, v in states.items() if "l3" in v},
            "rho_S_matches_amend3": {k: abs(v["rho"]["S"] - v["rho_S_amend3"]) <= CFG.rho_S_tol for k, v in states.items()},
            "C_m_short_layers": {k: sorted(lay for lay, s in v["change"]["C_m_short"].items() if s) for k, v in states.items()}}


def grid(run: Run, out: dict[str, Any]) -> None:
    by_task: dict[str, dict[str, Any]] = {}
    for n in E38.tasks:
        t = run.tasks[n]
        T0, ref0, io = A3.w0_pass(run, t)
        del ref0
        io0_train = train_io(run, t, run.W0)
        for name in names(n)[:-1]:
            out["states"][name] = by_task.setdefault(n, {})[name[-1]] = grid_state(run, n, name, T0, io, io0_train)
            dump(OUT, out)
        del io, io0_train
    out["P11"] = RT.p11_reading(by_task)
    out["P12"] = RT.p12_reading({k: {"scorable": r["scorable"], "singles": [x[2] for x in r["l3"]["singles"]],
                                     "random_singles": [x[2] for x in r["l3"]["random_singles"]]}
                                 for k, r in by_task[CFG.l3_task].items()})
    out["reported"] = reported(out["states"])
    dump(OUT, out)
    p11 = out["P11"]
    log(f"[a4] P11 seeds {json.dumps(p11['seed_class'])} | tasks {json.dumps(p11['task_class'])} | scored tasks "
        f"{p11['n_scored_tasks']} | reading '{p11['reading']}'")
    log(f"[a4] P12 seeds {json.dumps(out['P12']['seed_class'])} | reading '{out['P12']['reading']}'")
    log(f"[a4] reported {json.dumps(out['reported'])}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for p in (PRED, A3.PRED, A3.PRED_POOL, A3.PRED_R4, A3.PRED_R4B):
        if not p.is_file():
            sys.exit(f"file the predictions first: {p}")
    s0 = read_json(OUT0)
    if not (s0 and s0.get("pass") is True):
        sys.exit(f"Amendment 2's stage 0 has not passed ({OUT0}): its cached states are not checked")
    t0 = time.time()
    (LARGE / "amend4.log").write_text("")  # this run's log only
    run = Run(args.device)
    log(f"[a4 start] on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}; cfg {CFG}; a3 cfg {A3.CFG}")
    run.load_states(s0)
    out: dict[str, Any] = {"cfg": dataclasses.asdict(CFG), "controls": {}, "states": {}}
    if not controls(run, out):
        log(f"[a4 controls] FAIL: stop and report ({time.time() - t0:.0f} s)")
        sys.exit(1)
    log("[a4 controls] PASS")
    grid(run, out)
    log(f"[a4 done] ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

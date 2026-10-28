"""recruitment_test (E38) Amendment 6, S and D restored together: do the switched-on set and the direction-matched set
carry the same part of the gain? Plan: reports/plans/2026-10-28_plan_e38-amend6.md; predictions
results/predictions_amend6.json (filed before this code, never edited; the run refuses to start without it or without
Amendment 3's predictions files, whose controls it reruns). No fine-tuning: Amendment 2's cached states.

No new ranking: per task-seed state, S and D are Amendment 5's members as saved in results/amend5.json (the saved S is
asserted equal to this job's `run.S`, D disjoint from S), and S+ / S- are S's members with training e_c > 0 / <= 0,
e_c as Amendment 5 saved it in results/large/amend5_scores.pt ('e_train'). On the held-out windows, as Amendments 3-5,
rho_r(X) = (T1 - lo(W1, X's trigger outputs restored to W0's)) / G for X in S, D (again, as R7 checked them), S union D
(one pass), S+, S-; P14 reads a = rho_r(S union D) - max(rho_r(S), rho_r(D)). Reported beside it: e-hat(S+),
e-hat(S-) (held-out e_c summed / G, from the same file), the first-order numbers the Amendment 5 check read.

One job: the controls first, Amendment 3's R0-R4 (`amend3.controls`) on every state, then R7 on the six task-seed
states: rho_r(S) and rho_r(D), restored apart from the saved members, equal amend5.json's to 1e-4. Any fail: results
written, FAIL logged, exit 1, before any new share. Then the grid -> results/amend6.json (rewritten after each state;
P14 at the end). Every scored number is also in the log (stdout and results/large/amend6.log, emptied at the start
of each run; the control lines keep their '[a3 ...]' tag).
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

from commute import recruit as R
from commute import recruitment as RT

PRED, OUT = RESULTS / "predictions_amend6.json", RESULTS / "amend6.json"
A5_OUT, A5_SCORES = RESULTS / "amend5.json", LARGE / "amend5_scores.pt"  # the saved sets and e_c


@dataclass(frozen=True)
class Cfg:
    r7_tol: float = 1e-4  # R7: rho_r(S), rho_r(D) against amend5.json
    min_gain: float = 1.0  # a state with G < 1 nat is unscored
    reading_task: str = "r1_t534_v369"  # ' which': P14's reading
    beside_task: str = "emoticon"  # reported beside it; S+ and S- are read here
    s_plus_gap: float = 0.05  # reported: rho_r(S+) >= rho_r(D) - this on emoticon
    s_minus_max: float = -0.05  # reported: rho_r(S-) <= this on emoticon

    def __post_init__(self) -> None:
        assert self.min_gain == E38.min_gain == A3.CFG.min_gain
        assert {self.reading_task, self.beside_task} == set(A3.CFG.p10_tasks)


CFG = Cfg()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LARGE / "amend6.log").open("a") as f:
        f.write(line + "\n")


A3.log = log  # Amendment 3's control lines go to this run's log


def saved_sets(run: Run, a5: dict[str, Any], scores: dict[str, Any]) -> dict[str, dict[str, R.Sel]]:
    """Per task-seed state: S and D from amend5.json's members, S union D, and S+ / S- by the saved training e_c."""
    out = {}
    for n in E38.tasks:
        for name in names(n)[:-1]:
            m = a5["states"][name]["members"]
            S, D = ({k: torch.tensor(v, dtype=torch.long) for k, v in m[x].items()} for x in ("S", "D"))
            assert m["S"] == {k: v.tolist() for k, v in run.S(name).items()}, f"{name}: saved S is not run.S"
            assert S.keys() == D.keys() and not R.members(S) & R.members(D), f"{name}: D not disjoint from S"
            e = scores[name]["e_train"]
            out[name] = {"S": S, "D": D, "S_union_D": {k: torch.cat([S[k], D[k]]).sort().values for k in S},
                         "S_plus": {k: v[e[k][v] > 0] for k, v in S.items()},
                         "S_minus": {k: v[e[k][v] <= 0] for k, v in S.items()}}
    return out


def shares(run: Run, name: str, sets: dict[str, R.Sel], T0: float, io: RT.IO) -> tuple[float, float, dict[str, float]]:
    """One task-seed state on the held-out windows (io: W0's trigger io there): T1, G and rho_r of each set."""
    t, W1 = run.tasks[name.split("/")[0]], run.weights(name)
    T1, ref1 = lo_ref(run, W1, t)
    gain = T1 - T0
    g = gain if gain >= CFG.min_gain else 0.0  # unscored state: every share nan
    rho = {x: RT.share(T1, A3.restored(run, W1, sel, io, t, ref1), g) for x, sel in sets.items()}
    del ref1
    return T1, gain, rho


# ----------------------------------------------------------------------------- the controls


def r7(run: Run, name: str, sets: dict[str, R.Sel], T0: float, io: RT.IO, a5: dict[str, Any]) -> dict[str, Any]:
    """R7 on one task-seed state: S and D restored apart against amend5.json's rho_r."""
    T1, gain, rho = shares(run, name, {x: sets[x] for x in ("S", "D")}, T0, io)
    ref = {x: float(a5["states"][name]["rho"][x]) for x in rho}
    diff = {x: rho[x] - ref[x] for x in rho}
    rec = {"T1": T1, "gain": gain, "rho": rho, "rho_amend5": ref, "diff": diff,
           "ok": all(abs(d) <= CFG.r7_tol for d in diff.values())}
    log(f"[a6 controls {name}] R7 T0 {T0:.4f} T1 {T1:.4f} G {gain:.4f} | rho_r S {rho['S']:.4f} (Amendment 5 "
        f"{ref['S']:.4f}, diff {sci(diff['S'])}) D {rho['D']:.4f} (Amendment 5 {ref['D']:.4f}, diff {sci(diff['D'])}) "
        f"| {'ok' if rec['ok'] else 'FAIL'}")
    return rec


def controls(run: Run, out: dict[str, Any], sets: dict[str, dict[str, R.Sel]], a5: dict[str, Any]) -> bool:
    """R0-R4 on every task's states; only if all pass, R7 on the task-seed states."""
    for n in E38.tasks:
        out["controls"][n] = A3.controls(run, n)
        dump(OUT, out)
    out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    if out["controls_pass"]:
        for n in E38.tasks:
            c = out["controls"][n]
            T0, ref0, io = A3.w0_pass(run, run.tasks[n])
            del ref0
            c["R7"] = {}
            for name in names(n)[:-1]:
                c["R7"][name] = r7(run, name, sets[name], T0, io, a5)
                dump(OUT, out)
            c["ok"] = c["ok"] and all(v["ok"] for v in c["R7"].values())
            del io
        out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    dump(OUT, out)
    return out["controls_pass"]


# ----------------------------------------------------------------------------- the grid


def grid_state(run: Run, name: str, sets: dict[str, R.Sel], T0: float, io: RT.IO,
               e_held: dict[str, torch.Tensor]) -> dict[str, Any]:
    """One task-seed state: S, D (as in R7), S union D, S+ and S- restored."""
    T1, gain, rho = shares(run, name, sets, T0, io)
    g = gain if gain >= CFG.min_gain else 0.0
    rec = {"T0": T0, "T1": T1, "gain": gain, "scorable": gain >= CFG.min_gain,
           "n": {x: R.size(v) for x, v in sets.items()}, "rho": rho,
           "e_hat": {x: RT.share(RT.summed(e_held, sets[x]), 0.0, g) for x in ("S_plus", "S_minus")}}
    rec["a"] = RT.overlap(rec["rho"])
    r, h = rec["rho"], rec["e_hat"]
    log(f"[a6 grid {name}] T0 {T0:.4f} T1 {T1:.4f} G {gain:.4f} scorable {rec['scorable']} | n {json.dumps(rec['n'])}")
    log(f"[a6 grid {name}] rho_r S {r['S']:.4f} D {r['D']:.4f} S_union_D {r['S_union_D']:.4f} | a {rec['a']:.4f} | "
        f"S_plus {r['S_plus']:.4f} S_minus {r['S_minus']:.4f} | e-hat S_plus {h['S_plus']:.4f} S_minus {h['S_minus']:.4f}")
    return rec


def reported(states: dict[str, Any]) -> dict[str, Any]:
    """The filed 'reported' values: S+ and S- against their expected values on emoticon, as they are on ' which'."""
    on = {k: v["rho"] for k, v in states.items() if k.split("/")[0] == CFG.beside_task}
    return {"rho_S_plus_at_least_rho_D_minus_0.05": {k: r["S_plus"] >= r["D"] - CFG.s_plus_gap for k, r in on.items()},
            "rho_S_minus_at_most_-0.05": {k: r["S_minus"] <= CFG.s_minus_max for k, r in on.items()},
            "which_S_plus_S_minus": {k: {x: v["rho"][x] for x in ("S_plus", "S_minus")} for k, v in states.items()
                                     if k.split("/")[0] == CFG.reading_task}}


def grid(run: Run, out: dict[str, Any], sets: dict[str, dict[str, R.Sel]], scores: dict[str, Any]) -> None:
    by_task: dict[str, dict[str, Any]] = {}
    for n in E38.tasks:
        T0, ref0, io = A3.w0_pass(run, run.tasks[n])
        del ref0
        for name in names(n)[:-1]:
            rec = grid_state(run, name, sets[name], T0, io, scores[name]["e_held"])
            out["states"][name] = by_task.setdefault(n, {})[name[-1]] = rec
            dump(OUT, out)
        del io
    out["P14"] = RT.p14_reading(by_task, CFG.reading_task, CFG.beside_task)
    out["reported"] = reported(out["states"])
    dump(OUT, out)
    p = out["P14"]
    log(f"[a6] P14 a {json.dumps({k: round(v, 4) for k, v in p['a'].items()})} | seeds {json.dumps(p['seed_class'])} | "
        f"tasks {json.dumps(p['task_class'])} | reading ({CFG.reading_task}) '{p['reading']}', beside "
        f"({CFG.beside_task}) '{p['beside']}'")
    log(f"[a6] reported {json.dumps(out['reported'])}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for p in (PRED, A3.PRED, A3.PRED_POOL, A3.PRED_R4, A3.PRED_R4B, A5_SCORES):
        if not p.is_file():
            sys.exit(f"missing (file the predictions first, or Amendment 5's scores): {p}")
    s0, a5 = read_json(OUT0), read_json(A5_OUT)
    if not (s0 and s0.get("pass") is True):
        sys.exit(f"Amendment 2's stage 0 has not passed ({OUT0}): its cached states are not checked")
    if not (a5 and a5.get("controls_pass") is True):
        sys.exit(f"Amendment 5's results are missing or failed their controls ({A5_OUT})")
    t0 = time.time()
    (LARGE / "amend6.log").write_text("")  # this run's log only
    run = Run(args.device)
    log(f"[a6 start] on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}; cfg {CFG}; a3 cfg {A3.CFG}")
    run.load_states(s0)
    scores = torch.load(A5_SCORES, weights_only=True)
    sets = saved_sets(run, a5, scores)
    out: dict[str, Any] = {"cfg": dataclasses.asdict(CFG), "controls": {}, "states": {}}
    if not controls(run, out, sets, a5):
        log(f"[a6 controls] FAIL: stop and report ({time.time() - t0:.0f} s)")
        sys.exit(1)
    log("[a6 controls] PASS")
    grid(run, out, sets, scores)
    log(f"[a6 done] ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

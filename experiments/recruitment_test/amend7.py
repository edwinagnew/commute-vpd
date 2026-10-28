"""recruitment_test (E38) Amendment 7, the settling control for Amendment 6's 'separate': do D and the next-ranked set D2
also read 'separate', and does the top-ranked set of twice S's count (T2) take back as much as S union D? Plan:
reports/plans/2026-10-28_plan_e38-amend7.md; predictions results/predictions_amend7.json (filed before this code,
never edited; the run refuses to start without it, without Amendment 3's predictions files, whose controls it reruns,
or without results/amend6.json with its controls passed). No fine-tuning: Amendment 2's cached states.

Per task-seed state, S and D are Amendment 5's members as saved in results/amend5.json (Amendment 6's `saved_sets`,
with its asserts), and n_k = |S in k|. With Amendment 5's training e_c (results/large/amend5_scores.pt, 'e_train') and
its per-matrix top selection (`RT.top_per_matrix`, ties by index, short flags): D2 = the n_k alive pieces outside
S union D of largest e_c; T2 = the 2 n_k alive pieces of largest e_c, S's members allowed. On the held-out windows,
as Amendments 3-6, rho_r(X) for X in S, D, S union D, D2, D union D2 (one pass), T2. P15 reads
a2 = rho_r(D union D2) - max(rho_r(D), rho_r(D2)); P16 reads rho_r(T2) against rho_r(S union D). Reported:
|T2 and S| / |S|, e-hat(T2) and e-hat(S union D) (held-out e_c summed / G, the same file).

One job: the controls first, Amendment 3's R0-R4 (`amend3.controls`) on every state, then R8 on the six task-seed
states: rho_r(S), rho_r(D), rho_r(S union D) restored from the saved members equal amend6.json's to 1e-4. Any fail:
results written, FAIL logged, exit 1, before any new share. Then the grid -> results/amend7.json (rewritten after each
state; P15 and P16 at the end). Every scored number is also in the log (stdout and results/large/amend7.log, emptied
at the start of each run; the control lines keep their '[a3 ...]' tag).
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
import amend6 as A6
import torch
from amend2 import OUT0, dump, sci
from run import CFG as E38
from run import LARGE, RESULTS, Run, names, read_json

from commute import recruit as R
from commute import recruitment as RT

PRED, OUT = RESULTS / "predictions_amend7.json", RESULTS / "amend7.json"
A6_OUT = RESULTS / "amend6.json"  # R8's reference rho_r
OLD = ("S", "D", "S_union_D")  # restored again from the saved members (R8)


@dataclass(frozen=True)
class Cfg:
    r8_tol: float = 1e-4  # R8: rho_r(S), rho_r(D), rho_r(S union D) against amend6.json
    min_gain: float = 1.0  # a state with G < 1 nat is unscored
    p16_tasks: tuple[str, ...] = ("emoticon", "r1_t534_v369")  # P16 is read on emoticon and ' which'

    def __post_init__(self) -> None:
        assert self.min_gain == E38.min_gain == A3.CFG.min_gain == A6.CFG.min_gain
        assert set(self.p16_tasks) == set(A3.CFG.p10_tasks)


CFG = Cfg()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LARGE / "amend7.log").open("a") as f:
        f.write(line + "\n")


A3.log = log  # Amendment 3's control lines go to this run's log (importing amend6 pointed them at its own)


def ranked_sets(run: Run, base: dict[str, R.Sel], e: dict[str, torch.Tensor]) -> tuple[dict[str, R.Sel], dict[str, Any]]:
    """One task-seed state: S, D, S union D (saved), D2, D union D2 and T2 by training e_c; the short matrices, how
    many of S's members T2 holds, and whether T2 is S union D."""
    S, D, SD = base["S"], base["D"], base["S_union_D"]
    n = {k: int(v.numel()) for k, v in S.items()}
    again, _ = RT.top_per_matrix(e, run.alive, S, n)
    assert R.members(again) == R.members(D), "the saved training e_c does not rank D as Amendment 5 did"
    D2, D2_short = RT.top_per_matrix(e, run.alive, SD, n)
    T2, T2_short = RT.top_per_matrix(e, run.alive, {}, {k: 2 * c for k, c in n.items()})
    assert D2.keys() == T2.keys() == D.keys() and not R.members(D2) & R.members(SD), "D2 overlaps S union D"
    sets = {"S": S, "D": D, "S_union_D": SD, "D2": D2, "D_union_D2": {k: torch.cat([D[k], D2[k]]).sort().values for k in D},
            "T2": T2}
    hits = len(R.members(T2) & R.members(S))
    info = {"D2_short_matrices": sorted(k for k, s in D2_short.items() if s),
            "T2_short_matrices": sorted(k for k, s in T2_short.items() if s),
            "T2_holds_of_S": hits, "T2_holds_of_S_frac": hits / R.size(S), "T2_is_S_union_D": R.members(T2) == R.members(SD)}
    return sets, info


def all_sets(run: Run, a5: dict[str, Any], scores: dict[str, Any]) -> dict[str, tuple[dict[str, R.Sel], dict[str, Any]]]:
    base = A6.saved_sets(run, a5, scores)
    return {name: ranked_sets(run, sets, scores[name]["e_train"]) for name, sets in base.items()}


# ----------------------------------------------------------------------------- the controls


def r8(run: Run, name: str, sets: dict[str, R.Sel], T0: float, io: RT.IO, a6: dict[str, Any]) -> dict[str, Any]:
    """R8 on one task-seed state: S, D and S union D restored against amend6.json's rho_r."""
    T1, gain, rho = A6.shares(run, name, {x: sets[x] for x in OLD}, T0, io)
    ref = {x: float(a6["states"][name]["rho"][x]) for x in OLD}
    diff = {x: rho[x] - ref[x] for x in OLD}
    rec = {"T1": T1, "gain": gain, "rho": rho, "rho_amend6": ref, "diff": diff,
           "ok": all(abs(d) <= CFG.r8_tol for d in diff.values())}
    log(f"[a7 controls {name}] R8 T0 {T0:.4f} T1 {T1:.4f} G {gain:.4f} | rho_r "
        + " ".join(f"{x} {rho[x]:.4f} (Amendment 6 {ref[x]:.4f}, diff {sci(diff[x])})" for x in OLD)
        + f" | {'ok' if rec['ok'] else 'FAIL'}")
    return rec


def controls(run: Run, out: dict[str, Any], sets: dict[str, dict[str, R.Sel]], a6: dict[str, Any]) -> bool:
    """R0-R4 on every task's states; only if all pass, R8 on the task-seed states."""
    for n in E38.tasks:
        out["controls"][n] = A3.controls(run, n)
        dump(OUT, out)
    out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    if out["controls_pass"]:
        for n in E38.tasks:
            c = out["controls"][n]
            T0, ref0, io = A3.w0_pass(run, run.tasks[n])
            del ref0
            c["R8"] = {}
            for name in names(n)[:-1]:
                c["R8"][name] = r8(run, name, sets[name], T0, io, a6)
                dump(OUT, out)
            c["ok"] = c["ok"] and all(v["ok"] for v in c["R8"].values())
            del io
        out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    dump(OUT, out)
    return out["controls_pass"]


# ----------------------------------------------------------------------------- the grid


def grid_state(run: Run, name: str, sets: dict[str, R.Sel], info: dict[str, Any], T0: float, io: RT.IO,
               e_held: dict[str, torch.Tensor]) -> dict[str, Any]:
    """One task-seed state: S, D, S union D (as in R8), D2, D union D2 and T2 restored."""
    T1, gain, rho = A6.shares(run, name, sets, T0, io)
    g = gain if gain >= CFG.min_gain else 0.0
    rec = {"T0": T0, "T1": T1, "gain": gain, "scorable": gain >= CFG.min_gain,
           "D2_short": bool(info["D2_short_matrices"]), "T2_short": bool(info["T2_short_matrices"]), "sets": info,
           "n": {x: R.size(v) for x, v in sets.items()}, "rho": rho,
           "e_hat": {x: RT.share(RT.summed(e_held, sets[x]), 0.0, g) for x in ("T2", "S_union_D")},
           "members": {x: {k: v.tolist() for k, v in sorted(sets[x].items())} for x in ("D2", "T2")}}
    rec["a2"] = RT.overlap2(rho)
    r, h = rho, rec["e_hat"]
    log(f"[a7 grid {name}] T0 {T0:.4f} T1 {T1:.4f} G {gain:.4f} scorable {rec['scorable']} | n {json.dumps(rec['n'])} | "
        f"short D2 {info['D2_short_matrices']} T2 {info['T2_short_matrices']}")
    log(f"[a7 grid {name}] rho_r S {r['S']:.4f} D {r['D']:.4f} S_union_D {r['S_union_D']:.4f} | D2 {r['D2']:.4f} "
        f"D_union_D2 {r['D_union_D2']:.4f} | a2 {rec['a2']:.4f} | T2 {r['T2']:.4f} | e-hat T2 {h['T2']:.4f} "
        f"S_union_D {h['S_union_D']:.4f} | T2 holds {info['T2_holds_of_S']} of S ({info['T2_holds_of_S_frac']:.3f}), "
        f"T2 is S_union_D {info['T2_is_S_union_D']}")
    return rec


def reported(states: dict[str, Any]) -> dict[str, Any]:
    """The filed 'reported' values per state: rho_r(D2) below rho_r(D), |T2 and S| / |S|, e-hat(T2)."""
    return {"rho_D2_below_rho_D": {k: v["rho"]["D2"] < v["rho"]["D"] for k, v in states.items()},
            "T2_holds_of_S_frac": {k: v["sets"]["T2_holds_of_S_frac"] for k, v in states.items()},
            "e_hat_T2": {k: v["e_hat"]["T2"] for k, v in states.items()}}


def grid(run: Run, out: dict[str, Any], sets: dict[str, tuple[dict[str, R.Sel], dict[str, Any]]],
         scores: dict[str, Any]) -> None:
    by_task: dict[str, dict[str, Any]] = {}
    for n in E38.tasks:
        T0, ref0, io = A3.w0_pass(run, run.tasks[n])
        del ref0
        for name in names(n)[:-1]:
            rec = grid_state(run, name, *sets[name], T0, io, scores[name]["e_held"])
            out["states"][name] = by_task.setdefault(n, {})[name[-1]] = rec
            dump(OUT, out)
        del io
    out["P15"], out["P16"] = RT.p15_reading(by_task), RT.p16_reading(by_task, CFG.p16_tasks)
    out["reported"] = reported(out["states"])
    dump(OUT, out)
    p, q = out["P15"], out["P16"]
    log(f"[a7] P15 a2 {json.dumps({k: round(v, 4) for k, v in p['a2'].items()})} | seeds {json.dumps(p['seed_class'])} | "
        f"tasks {json.dumps(p['task_class'])} | reading '{p['reading']}'")
    log(f"[a7] P16 seeds {json.dumps(q['seed_class'])} | tasks {json.dumps(q['task_class'])} | reading "
        f"({', '.join(CFG.p16_tasks)}) '{q['reading']}'")
    log(f"[a7] reported {json.dumps(out['reported'])}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for p in (PRED, A3.PRED, A3.PRED_POOL, A3.PRED_R4, A3.PRED_R4B, A6.A5_SCORES):
        if not p.is_file():
            sys.exit(f"missing (file the predictions first, or Amendment 5's scores): {p}")
    s0, a5, a6 = read_json(OUT0), read_json(A6.A5_OUT), read_json(A6_OUT)
    if not (s0 and s0.get("pass") is True):
        sys.exit(f"Amendment 2's stage 0 has not passed ({OUT0}): its cached states are not checked")
    if not (a5 and a5.get("controls_pass") is True):
        sys.exit(f"Amendment 5's results are missing or failed their controls ({A6.A5_OUT})")
    if not (a6 and a6.get("controls_pass") is True):
        sys.exit(f"Amendment 6's results are missing or failed their controls ({A6_OUT})")
    t0 = time.time()
    (LARGE / "amend7.log").write_text("")  # this run's log only
    run = Run(args.device)
    log(f"[a7 start] on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}; cfg {CFG}; a3 cfg {A3.CFG}")
    run.load_states(s0)
    scores = torch.load(A6.A5_SCORES, weights_only=True)
    sets = all_sets(run, a5, scores)
    out: dict[str, Any] = {"cfg": dataclasses.asdict(CFG), "controls": {}, "states": {}}
    if not controls(run, out, {k: v[0] for k, v in sets.items()}, a6):
        log(f"[a7 controls] FAIL: stop and report ({time.time() - t0:.0f} s)")
        sys.exit(1)
    log("[a7 controls] PASS")
    grid(run, out, sets, scores)
    log(f"[a7 done] ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

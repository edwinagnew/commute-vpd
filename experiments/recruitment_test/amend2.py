"""recruitment_test (E38) Amendment 2: how much of the fine-tune runs through the recruited set S beyond its existing
push. Plan: reports/plans/2026-10-27_plan_e38-amend2.md; predictions results/predictions_amend2.json (filed before this
code, never edited). States, sets and readouts are E38's (run.py, commute.recruitment); this adds removal at the
trigger position only (`recruitment.removal_hooks`), every removal from W0 as well as W1, and the target census
(`census.target_census`).

Per task-seed state, gain G = T1 - T0 (G < 1 nat: unscored, every share nan), for each set X (S; null = the ' http'
state's recruited set, whole; random_i = recruit.random_like(S, alive, i), i < 20; busy = top |S| alive by W0's mean
trigger importance, a control) under both position rules ('trigger'; 'all' = ablated_weights, E38's removal):
  rho_ft(X) = (T1 - lo(W1 without X)) / G,   rho_0(X) = (T0 - lo(W0 without X)) / G,   D(X) = rho_ft(X) - rho_0(X).
KL is read against the weights' own logits: W1's for removals from W1, W0's for removals from W0 (as E38).

  amend2.py --stages 0      alone first: E38's nine states and stage-0 checks (C0, run.py's make_state and apparatus;
                            states cached in results/large/states/), then C1-C4 -> results/amend2_stage0.json; exit 1
                            on any fail
  amend2.py --stages 1,2    refuses unless amend2_stage0.json passed; reads the cached states (sha256 as E38); stage 1
                            -> results/amend2.json (after each state); stage 2 -> results/amend2_census.json
  --stages 0,1,2 runs all three in one job, stage 0 first. Every scored number is also in the log (stdout and
  results/large/amend2.log; run.py's own stage-0 lines also go to results/large/full.log).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from run import CFG as E38
from run import (
    LARGE,
    RESULTS,
    STATES,
    Run,
    _finite,
    apparatus,
    make_state,
    names,
    read_json,
    references,
)
from torch import Tensor

from commute import census as CE
from commute import recruit as R
from commute import recruitment as RT
from commute.faithfulness import model_state
from commute.regating import Task
from commute.signatures import ablated_weights, top_by

PRED = RESULTS / "predictions_amend2.json"
OUT0, OUT1, OUT2 = RESULTS / "amend2_stage0.json", RESULTS / "amend2.json", RESULTS / "amend2_census.json"


@dataclass(frozen=True)
class Cfg:
    n_random: int = 20  # stage 1: random_like seeds 0..19
    n_census_random: int = 5  # stage 2: random_like seeds 0..4 (E38's census sets)
    hook_tol: float = 1e-3  # C1: hook at every position vs ablated_weights, |lo diff|; expected 0 (same computation)
    pre_tol: float = 1e-4  # C2: trigger-only removal of S, max |logit diff| before the trigger; expected 0 (causal)
    empty_tol: float = 1e-5  # C3: trigger-only removal of nothing, |lo diff|; expected 0
    busy_kl_min: float = 1.0  # C4: trigger-only removal of busy from W0, KL at the trigger, where |busy| >= busy_min_n
    busy_min_n: int = 100
    min_gain: float = 1.0  # a state with G < 1 nat is unscored
    m0_low: float = 0.1  # reported: fraction of S with W0 mean trigger importance < 0.1; expected 0.3..0.9
    m0_low_range: tuple[float, float] = (0.3, 0.9)
    pile: tuple[str, int, int] = ("val", 0, 2000)  # stage 2's census rows
    http: int = 3944  # ' http', the null's target: the fourth census id
    census_batch: int = 8

    def __post_init__(self) -> None:
        assert self.n_random == E38.n_random == 20 and self.n_census_random == E38.n_census_random == 5
        assert (self.pile, self.http, self.min_gain) == (E38.pile, E38.null_token, E38.min_gain), "E38's sizes"
        assert (self.busy_kl_min, self.busy_min_n) == (E38.busy_kl_min, E38.busy_min_n), "E38's busy control"


CFG = Cfg()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LARGE / "amend2.log").open("a") as f:
        f.write(line + "\n")


def dump(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(_finite(obj), indent=1, sort_keys=True))


def lo_ref(run: Run, W: dict[str, Tensor], t: Task) -> tuple[float, list[Tensor]]:
    """The full model at weights W on the held-out windows: its trigger lo, and its own logits (the KL reference)."""
    with model_state(run.model, target=W):
        ref = RT.window_logits(run.model, t.held.tokens)
        return RT.readout(run.model, t.held.tokens, t.half, int(t.target), ref)["lo"], ref


# ----------------------------------------------------------------------------- stage 0: E38's checks, then C1-C4


def logit_diff(run: Run, W: dict[str, Tensor], sel: R.Sel, t: Task, ref: list[Tensor]) -> dict[str, float]:
    """Trigger-only removal of `sel` from W: max |logit - ref| before the trigger (C2) and at it (that it bites)."""
    with model_state(run.model, target=W), RT.removal_hooks(run.model, ablated_weights(W, run.comps, sel), t.half):
        lg = RT.window_logits(run.model, t.held.tokens)
    h = t.half
    return {"pre": max(float((a[:, :h] - b[:, :h]).abs().max()) for a, b in zip(lg, ref, strict=True)),
            "at": max(float((a[:, h] - b[:, h]).abs().max()) for a, b in zip(lg, ref, strict=True))}


def controls(run: Run, task: str) -> dict[str, Any]:
    t, m, comps = run.tasks[task], run.model, run.comps
    win = (t.held.tokens, t.half, int(t.target))
    empty = {k: torch.zeros(0, dtype=torch.long) for k in run.alive}
    T0, ref0 = lo_ref(run, run.W0, t)
    out: dict[str, Any] = {"C1": {}, "C2": {}, "C3": {"W0": RT.removed_at(m, run.W0, comps, empty, *win, ref0)["lo"] - T0}}
    for name in names(task)[:-1]:
        W1, S = run.weights(name), run.S(name)
        T1, ref1 = lo_ref(run, W1, t)
        out["C3"][name] = RT.removed_at(m, W1, comps, empty, *win, ref1)["lo"] - T1
        for lab, W, ref in (("W0", run.W0, ref0), ("W1", W1, ref1)):
            out["C1"][f"{name}/{lab}"] = (RT.removed_at(m, W, comps, S, *win, ref, everywhere=True)["lo"]
                                          - RT.removed(m, W, comps, S, *win, ref)["lo"])
            out["C2"][f"{name}/{lab}"] = logit_diff(run, W, S, t, ref)
        del ref1
    busy = top_by(t.m0, run.alive, R.size(run.S(f"{task}/s0")))
    b = RT.removed_at(m, run.W0, comps, busy, *win, ref0)
    out["C4"] = {"n": R.size(busy), **b, "ok": R.size(busy) < CFG.busy_min_n or b["kl_trigger"] >= CFG.busy_kl_min}
    out["C1_ok"] = all(abs(v) <= CFG.hook_tol for v in out["C1"].values())
    out["C2_ok"] = all(v["pre"] <= CFG.pre_tol for v in out["C2"].values())
    out["C3_ok"] = all(abs(v) <= CFG.empty_tol for v in out["C3"].values())
    out["ok"] = out["C1_ok"] and out["C2_ok"] and out["C3_ok"] and out["C4"]["ok"]
    c2 = {k: {"pre": sci(v["pre"]), "at": round(v["at"], 3)} for k, v in out["C2"].items()}
    log(f"[a2 stage0 {task}] C1 lo diffs {json.dumps({k: sci(v) for k, v in out['C1'].items()})} {out['C1_ok']} | C2 max "
        f"|logit diff| {json.dumps(c2)} {out['C2_ok']} | C3 lo diffs {json.dumps({k: sci(v) for k, v in out['C3'].items()})} "
        f"{out['C3_ok']} | C4 |busy| {out['C4']['n']} KL trigger {b['kl_trigger']:.4f} kl_other {b['kl_other']:.4f} lo "
        f"{b['lo']:.4f} (T0 {T0:.4f}) {out['C4']['ok']}")
    return out


def sci(x: float) -> float:
    return float(f"{x:.2e}")


def stage0(run: Run) -> dict[str, Any]:
    """E38's stage 0 task by task (its states and checks: C0), then C1-C4; stops at the first task that fails."""
    refs = references()
    out: dict[str, Any] = {"refs": refs, "tasks": {}, "states": {}, "cfg": dataclasses.asdict(CFG)}
    for n in E38.tasks:
        t = run.prepare(n)
        for name in names(n):
            out["states"][name] = make_state(run, n, name, refs[n])
        a = apparatus(run, n, {x: out["states"][x]["T1"] for x in names(n)[:-1]})
        rec: dict[str, Any] = {"T0": t.base["T"], "T0_ref": refs[n]["T0"], "apparatus": a}
        rec["C0"] = (abs(t.base["T"] - refs[n]["T0"]) <= E38.t_tol and all(out["states"][x]["ok"] for x in names(n))
                     and a["empty_ok"] and a["busy"]["ok"] and a["path_ok"])
        rec["controls"] = controls(run, n) if rec["C0"] else {}
        rec["ok"] = rec["C0"] and rec["controls"].get("ok") is True
        out["tasks"][n] = rec
        log(f"[a2 stage0 {n}] T0 {rec['T0']:.4f} (E18 {rec['T0_ref']:.4f}) | C0 {rec['C0']} | {'ok' if rec['ok'] else 'FAIL'}")
        dump(OUT0, out)
        if not rec["ok"]:
            break
    out["pass"] = len(out["tasks"]) == len(E38.tasks) and all(v["ok"] for v in out["tasks"].values())
    dump(OUT0, out)
    log(f"[a2 stage0] {'PASS' if out['pass'] else 'FAIL: stop and report'}")
    return out


# ----------------------------------------------------------------------------- stage 1: the removal grid


def grid(run: Run, t: Task, W: dict[str, Tensor], ref: list[Tensor], sets: dict[str, Any], pos: str) -> dict[str, Any]:
    """`readout` with each set removed from W under one position rule; sets {'S', 'null', 'busy': Sel, 'random': [Sel]}."""
    def one(sel: R.Sel) -> dict[str, float]:
        a = (run.model, W, run.comps, sel, t.held.tokens, t.half, int(t.target), ref)
        return RT.removed(*a) if pos == "all" else RT.removed_at(*a)

    return {k: [one(x) for x in v] if k == "random" else one(v) for k, v in sets.items()}


def per_set(reads: dict[str, Any], f: Callable[[dict[str, float]], float]) -> dict[str, Any]:
    """f of each set's read (random as a list)."""
    return {k: [f(r) for r in v] if k == "random" else f(v) for k, v in reads.items()}


def low_frac(m0: dict[str, Tensor], sel: R.Sel) -> float:
    vals = [m0[k].float().cpu()[v.cpu()] for k, v in sel.items() if v.numel()]
    return float((torch.cat(vals) < CFG.m0_low).float().mean()) if vals else float("nan")


def log_state(name: str, rec: dict[str, Any]) -> None:
    log(f"[a2 stage1 {name}] T0 {rec['T0']:.4f} T1 {rec['T1']:.4f} G {rec['gain']:.4f} scorable {rec['scorable']} | "
        f"kl_ft {rec['kl_ft']:.4f} | n {json.dumps(rec['n'])} | S with m0 < {CFG.m0_low}: {rec['m0_low']:.3f}")
    for pos in ("trigger", "all"):
        D, rf, r0 = rec["D"][pos], rec["rho_ft"][pos], rec["rho_0"][pos]
        beats = sum(D["S"] > d for d in D["random"])
        log(f"[a2 stage1 {name} {pos}] " + " | ".join(f"{k} rho_ft {rf[k]:.4f} rho_0 {r0[k]:.4f} D {D[k]:.4f}"
                                                     for k in ("S", "null", "busy"))
            + f" | random D mean {sum(D['random']) / len(D['random']):.4f} max {max(D['random']):.4f}; D(S) above "
            f"{beats}/{len(D['random'])} | kl_other from W1: S {rec['kl_other_ft'][pos]['S']:.4f} busy "
            f"{rec['kl_other_ft'][pos]['busy']:.4f}")
        log(f"[a2 stage1 {name} {pos}] random D {json.dumps([round(x, 4) for x in D['random']])} | random rho_0 "
            f"{json.dumps([round(x, 4) for x in r0['random']])}")


def stage1_state(run: Run, task: str, name: str, ref0: list[Tensor], T0: float) -> dict[str, Any]:
    t, W1, S = run.tasks[task], run.weights(name), run.S(name)
    with model_state(run.model, target=W1):
        own = RT.readout(run.model, t.held.tokens, t.half, int(t.target), ref0)  # T1, and the fine-tune's kl_other
        ref1 = RT.window_logits(run.model, t.held.tokens)
    T1, gain = own["lo"], own["lo"] - T0
    g = gain if gain >= CFG.min_gain else 0.0  # unscored state: every share nan
    sets = {"S": S, "null": run.S(f"{task}/null"), "busy": top_by(t.m0, run.alive, R.size(S)),
            "random": [R.random_like(S, run.alive, i) for i in range(CFG.n_random)]}
    rec: dict[str, Any] = {"T0": T0, "T1": T1, "gain": gain, "scorable": gain >= CFG.min_gain, "kl_ft": own["kl_other"],
                           "kl_ft_trigger": own["kl_trigger"], "n": {k: R.size(v) for k, v in sets.items() if k != "random"},
                           "m0_low": low_frac(t.m0, S), "reads": {}, "rho_ft": {}, "rho_0": {}, "D": {}, "kl_other_ft": {}}
    for pos in ("trigger", "all"):
        ft, w0 = grid(run, t, W1, ref1, sets, pos), grid(run, t, run.W0, ref0, sets, pos)
        rf, r0 = per_set(ft, lambda r: RT.share(T1, r["lo"], g)), per_set(w0, lambda r: RT.share(T0, r["lo"], g))
        rec["reads"][pos] = {"ft": ft, "0": w0}
        rec["rho_ft"][pos], rec["rho_0"][pos], rec["kl_other_ft"][pos] = rf, r0, per_set(ft, lambda r: r["kl_other"])
        rec["D"][pos] = {k: [a - b for a, b in zip(v, r0[k], strict=True)] if k == "random" else v - r0[k]
                         for k, v in rf.items()}
    del ref1
    log_state(name, rec)
    return rec


def stage1(run: Run) -> dict[str, Any]:
    out: dict[str, Any] = {"states": {}}
    by_task: dict[str, dict[str, Any]] = {}
    for n in E38.tasks:
        T0, ref0 = lo_ref(run, run.W0, run.tasks[n])
        for name in names(n)[:-1]:
            out["states"][name] = by_task.setdefault(n, {})[name[-1]] = stage1_state(run, n, name, ref0, T0)
            dump(OUT1, out)
        del ref0
    out.update(RT.amend2_readings(by_task))
    lo, hi = CFG.m0_low_range
    out["m0_low_in_range"] = {k: lo <= v["m0_low"] <= hi for k, v in out["states"].items()}
    dump(OUT1, out)
    for p in ("P5", "P5_all"):
        log(f"[a2 stage1] {p} seeds {json.dumps(out[p]['seed_class'])} | tasks {json.dumps(out[p]['task_class'])} | "
            f"reading '{out[p]['reading']}'")
    log(f"[a2 stage1] P5_all same task classes as P5: {out['P5_all']['same_task_classes_as_P5']} | P5_null "
        f"{json.dumps(out['P5_null'])} | P6 {json.dumps(out['P6'])} | P7 {json.dumps(out['P7'])} | m0 < {CFG.m0_low} "
        f"in {CFG.m0_low_range}: {json.dumps(out['m0_low_in_range'])}")
    return out


# ----------------------------------------------------------------------------- stage 2: the target census


def piece_keys(sel: R.Sel) -> list[str]:
    return sorted(f"{k}:{c}" for k, c in R.members(sel))


def stage2(run: Run) -> dict[str, Any]:
    """At W0 over Pile rows: per task the target ratios of S (seed 0), the five census random sets pooled, and the
    control (top-|S| alive by W0's contrast at positions preceding the target); P8 on the tasks where the control holds."""
    from commute.data import pile_rows

    rows = pile_rows(*CFG.pile, log=log)
    tgt = {n: int(run.tasks[n].target) for n in E38.tasks}
    ids = sorted(set(tgt.values()) | {CFG.http})
    with model_state(run.model, target=run.W0):
        counts, accs = CE.target_census(run.model, rows, ids, {n: R.precede_mask(rows, k) for n, k in tgt.items()},
                                        CFG.census_batch, log)
    out: dict[str, Any] = {"ids": ids, "n_all": counts.n_all, "base": dict(zip(ids, counts.base.tolist(), strict=True)),
                           "tasks": {}, "pieces": {}}
    for n in E38.tasks:
        S = run.S(f"{n}/s0")
        rand = [R.random_like(S, run.alive, i) for i in range(CFG.n_census_random)]
        ctrl = top_by(accs[n].result(run.alive), run.alive, R.size(S))
        rec = CE.p8_task(counts.ratios(S, tgt[n]), [x for s in rand for x in counts.ratios(s, tgt[n])],
                         counts.ratios(ctrl, tgt[n]))
        rec.update(target=tgt[n], n_precede=accs[n].n_on, J_control_S=R.jaccard(ctrl, S),
                   sets={"S": piece_keys(S), "control": piece_keys(ctrl), "random": [piece_keys(x) for x in rand]})
        for sel in (S, ctrl, *rand):
            out["pieces"].update(counts.pieces(sel))
        out["tasks"][n] = rec
        log(f"[a2 stage2 {n}] target {tgt[n]} ({accs[n].n_on} positions precede it) | median ratio S "
            f"{rec['median_S']:.3f} random {rec['median_random']:.3f} control {rec['median_control']:.3f} | non-rare n "
            f"(S, random, control) {rec['n']} | J(control, S) {rec['J_control_S']:.3f} | control {rec['control_ok']} | "
            f"P8 {rec['pass']} | random median 0: {rec['random_median_zero']}")
    scored = [v["pass"] for v in out["tasks"].values() if v["control_ok"]]
    out["P8"] = {"per_task": {n: v["pass"] for n, v in out["tasks"].items()}, "pass": sum(x is True for x in scored) >= RT.N_TASKS,
                 "control_on_every_task": all(v["control_ok"] for v in out["tasks"].values())}
    dump(OUT2, out)
    log(f"[a2 stage2] n_all {counts.n_all} base {json.dumps(out['base'])} | P8 {json.dumps(out['P8'])}")
    return out


# ----------------------------------------------------------------------------- main


def stages_arg(s: str) -> list[int]:
    ks = sorted({int(x) for x in s.split(",")})
    if not set(ks) <= {0, 1, 2}:
        raise argparse.ArgumentTypeError("stages are 0, 1, 2")
    return ks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", type=stages_arg, required=True, help="0 alone first; then 1,2 (or 0,1,2 in one job)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    s0 = read_json(OUT0)
    if 0 not in args.stages and not (s0 and s0.get("pass") is True):
        sys.exit(f"stage 0 has not passed ({OUT0}): run --stages 0 first")
    STATES.resolve().mkdir(parents=True, exist_ok=True)  # through the results/large symlink, if there is one
    t0 = time.time()
    run = Run(args.device)
    log(f"[a2 start] stages {args.stages} on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}; cfg {CFG}")
    if 0 in args.stages:
        if not stage0(run)["pass"]:
            sys.exit(1)
    else:
        assert s0 is not None
        run.load_states(s0)
    if 1 in args.stages:
        stage1(run)
    if 2 in args.stages:
        stage2(run)
    log(f"[a2 done] stages {args.stages} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

"""E116 (definition_table), 2026-11-06: the 67M task table read under the frozen definition (`decisions.md` rows 81, 87,
164), part by part; nothing added to the verdict. Plan: reports/plans/2026-11-06_plan_e116-definition-table.md (its
"Questions settled by the lead", items 1-6) and appendix. Rules: results/predictions.json (the run refuses unless it is
committed unchanged); pure rules in src/commute/case_label.py (tests/test_case_label.py). One job, stages in order, each
skipped when its results file says done, every scored number printed to the log:

  0  (a) alone: E112's possession numbers re-read through the moved probe code (E112's frozen `probes`, now calling
     `case_label.value_probes`) on its saved task; a block off by more than 1e-3 exits 1. (b) P0 per readout C: Pile
     positions, probes and the Hewitt-Liang control at blocks 0-3, l*, the site.
  1  P1 per state (26) and W0 per task, full-vocabulary top-1 beside; n/a with its reason where no held-out values exist.
  2  per C with a site: the do-nothing (a random basis of the probe basis's rank, seed 5; rank 1 beside) on every state
     read under C first, the C stopped on a share over 0.10 (E112's rule); then P2, P3 and the twins' P3; then the table
     with verdicts, the locator column copied from E100's and E97's JSON. Exit 3 between Cs when too little time is
     left (resubmit; finished Cs are kept).

    sbatch cluster/definition_table.sbatch
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import readouts as RO
import reads as RD
import tasks as TK
import torch

from commute import case_label as CL
from commute import keydep as KD
from commute import removal as RM
from commute import steering as ST
from commute.data import pile_rows
from commute.jobs import hard_exit, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED, EXP = RESULTS / "large", RESULTS / "predictions.json", HERE.parent
log = KD.log


@dataclass(frozen=True)
class Config:
    """Every size and seed, read from predictions.json's settings."""

    probe_rows: tuple[str, int, int]
    probe_cap: int
    probe_seed: int
    control_seed: int
    ridge_lambda: float
    guess_scan_rows: int
    guess_precap: int
    copy_locator_block: int
    e53_half: int
    e53_fit_windows: int
    e53_window_seed: int
    random_dir_seed: int
    span_rel_tol: float
    repro_tol: float
    stage_minutes: float
    readout_minutes: float
    constant_rows: tuple[str, ...]

    def __post_init__(self) -> None:
        assert self.probe_rows[1] >= 32768, "probe rows must be disjoint from the task windows' rows (0-32767)"
        assert 0 < self.guess_scan_rows <= self.probe_rows[2] and self.guess_precap >= 1
        assert self.readout_minutes < self.stage_minutes


def path(n: int) -> Path:
    return RESULTS / f"stage{n}.json"


def load(n: int) -> dict[str, Any]:
    return json.loads(path(n).read_text()) if path(n).is_file() else {}


def dump(n: int, res: dict[str, Any]) -> None:
    path(n).write_text(json.dumps(jsonable(res), indent=1))


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


# ----------------------------------------------------------------------------- stage 0


def late(C: Config, t0: float, what: str) -> bool:
    """Too little time left in the job for `what`: the caller exits 3 (resubmit; finished work is kept)."""
    if time.monotonic() - t0 <= (C.stage_minutes - C.readout_minutes) * 60:
        return False
    log(f"[time] too little left for {what}: exit 3, resubmit (finished stages and readouts are kept)")
    return True


def stage0a(net: ST.Net, C: Config) -> int:
    res = {"e112_repro": RO.e112_repro(net, LARGE, C.repro_tol)}
    say("stage 0a: E112 reproduction", res["e112_repro"])
    dump(0, res)
    if not res["e112_repro"]["ok"]:
        log(f"[stop] E112's possession numbers are not reproduced within {C.repro_tol}: exit 1, nothing else runs")
        return 1
    return 0


def stage0b(net: ST.Net, C: Config, P: dict[str, Any], R: list[TK.Readout], t0: float) -> int:
    res = load(0)
    res.setdefault("readouts", {})
    rows, vocab = pile_rows(*C.probe_rows, log=log), int(net.net.lm_head.weight.shape[0])
    todo = [r for r in R if r.name not in res["readouts"]]
    am = RO.scan_argmax(net, rows[: C.guess_scan_rows]) if any(r.kind == "guess" for r in todo) else rows[:0]
    for r in todo:
        if late(C, t0, f"P0 {r.name}"):
            return 3
        res["readouts"][r.name] = RO.p0(net, r, RO.pile_positions(net, r, rows, am, C, vocab), C, P["rules"]["P0"], LARGE)
        say(f"P0 {r.name}", res["readouts"][r.name])
        dump(0, res)
    res["done"] = True
    dump(0, res)
    return 0


# ----------------------------------------------------------------------------- stage 1


def stage1(net: ST.Net, T: dict[str, TK.Task], F: Any) -> None:
    res: dict[str, Any] = {"w0": {}, "states": {}}
    for t in T.values():
        res["w0"][t.name] = RD.p1(net, t, None)
        say(f"P1 W0 {t.name}", res["w0"][t.name])
        for s, p in t.states.items():
            res["states"][f"{t.name}_s{s}"] = RD.p1(net, t, F.state(p))
            say(f"P1 {t.name}_s{s}", res["states"][f"{t.name}_s{s}"])
        dump(1, res)
    res["done"] = True
    dump(1, res)


# ----------------------------------------------------------------------------- stage 2


def read_c(net: ST.Net, C: Config, rules: dict[str, Any], T: dict[str, TK.Task], r: TK.Readout, block: int, F: Any) -> dict[str, Any]:
    """One C: the do-nothing on every state read under it (own and twins) first; a share over the bar stops this C; then
    P3 on every state and P2 on the own task's states."""
    probe = torch.load(LARGE / f"probes_{r.name}.pt", weights_only=False)
    u = CL.span_basis(probe["W"][block], C.span_rel_tol)
    d, floor = u.shape[0], rules["gain_floor"]
    dn = {"random": RM.random_basis(d, u.shape[1], [C.random_dir_seed]), "random_rank1": RM.random_basis(d, 1, [C.random_dir_seed])}
    names = [(n, s) for n in (r.own, *r.twins) for s in T[n].states]
    dWs = {(n, s): F.state(T[n].states[s]) for n, s in names}
    w0 = {n: RD.w0_margins(net, T[n]) for n in {n for n, _ in names}}
    out: dict[str, Any] = {"block": block, "rank": u.shape[1], "do_nothing": {}, "states": {}}
    for (n, s), dW in dWs.items():
        out["do_nothing"][f"{n}_s{s}"] = RD.shares(net, T[n], dW, block, dn, floor, w0[n])
        say(f"do-nothing {r.name} {n}_s{s}", out["do_nothing"][f"{n}_s{s}"])
    if not CL.do_nothing_ok([v["random"] for v in out["do_nothing"].values()], rules["do_nothing_max"]):
        out["stop"] = f"the rank-{u.shape[1]} random basis takes over {rules['do_nothing_max']} of the gain on a state"
        log(f"[stop {r.name}] {out['stop']}: P2 and P3 are not read under this C")
        return out
    for (n, s), dW in dWs.items():
        rec = RD.shares(net, T[n], dW, block, {"P3": u}, floor, w0[n])
        if n == r.own:
            rec["P2_read"] = RD.p2(net, T[n], probe, block, dW)
            rec["P2"] = rec["P2_read"]["P2"]
        out["states"][f"{n}_s{s}"] = rec
        say(f"reads {r.name} {n}_s{s} ({'own' if n == r.own else 'twin'})", rec)
    return out


def verdict_rows(C: Config, rules: dict[str, Any], T: dict[str, TK.Task], R: list[TK.Readout]) -> list[dict[str, Any]]:
    """The table: per own state, P0-P3 and the verdict (`case_label.table_verdict`), with the locator column copied."""
    s0, s1, s2 = load(0)["readouts"], load(1)["states"], load(2).get("readouts", {})
    e100 = json.loads((EXP / "fed_or_written/results/table.json").read_text())["table"]
    e97 = json.loads((EXP / "locator_table/results/table.json").read_text())["table"]
    rows: list[dict[str, Any]] = []
    for r in R:
        t, p0 = T[r.own], s0[r.name]
        c2 = s2.get(r.name, {})
        for s in t.states:
            k = f"{t.name}_s{s}"
            st = c2.get("states", {}).get(k, {})
            p1, p2, p3 = (float(x) for x in (s1[k]["P1"], st.get("P2", math.nan), st.get("P3", math.nan)))
            na = c2.get("stop") or (f"P1 cannot be applied: {t.p1_na}" if t.p1_na else None)
            rows.append({"task": t.name, "seed": s, "C": r.name, "site": p0["site"], "P1": p1, "P2": p2, "P3": p3,
                         "do_nothing": c2.get("do_nothing", {}).get(k, {}).get("random", math.nan),
                         "verdict": CL.table_verdict(p0["site"], p1, p2, p3, rules["real"], na),
                         "e100": [{x: e[x] for x in ("part", "own", "reverts_sum")} for e in e100 if e["task"] == t.row and e["seed"] == s],
                         "e97": [{x: e[x] for x in ("part", "share", "control_share", "located", "reason")} for e in e97 if e["task"] == t.row and e["seed"] == s]})  # fmt: skip
            say(f"verdict {k}", rows[-1])
    rows += [{"task": n, "verdict": "does not apply", "reason": "constant answer (decisions.md row 164)"} for n in C.constant_rows]
    return rows


def stage2(net: ST.Net, C: Config, rules: dict[str, Any], T: dict[str, TK.Task], R: list[TK.Readout], F: Any, t0: float) -> int:
    res = load(2) or {"readouts": {}}
    s0 = load(0)["readouts"]
    for r in R:
        if r.name in res["readouts"] or s0[r.name]["site"] is None:
            continue
        if late(C, t0, f"stage 2 {r.name}"):
            return 3
        res["readouts"][r.name] = read_c(net, C, rules, T, r, s0[r.name]["site"], F)
        dump(2, res)
    res["table"] = verdict_rows(C, rules, T, R)
    res["done"] = True
    dump(2, res)
    return 0


def main() -> None:
    t0 = time.monotonic()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"].items()}
    C = Config(**settings)
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    if not load(0).get("e112_repro", {}).get("ok") and stage0a(net, C):  # the control first and alone
        hard_exit(1)
    F = TK.SITES.Files()
    TK.add_records(F)
    T, R = TK.build(F, C)
    say("tasks", {n: {"values": len(t.values), "fit": len(t.fit.tgt), "held": None if t.held is None else len(t.held.tgt)} for n, t in T.items()})
    code = 0 if load(0).get("done") else stage0b(net, C, P, R, t0)
    if code == 0 and not load(1).get("done"):
        stage1(net, T, F)
    if code == 0 and not load(2).get("done"):
        code = stage2(net, C, P["rules"], T, R, F, t0)
    log(f"=== E116 done through stage {2 if code == 0 else 'partial'}, exit {code}; sha256 of inputs {json.dumps(F.seen)} ===")
    hard_exit(code)


if __name__ == "__main__":
    main()

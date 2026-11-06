"""E93 Amendment 1 (qwen_d1_table): the scaled adjoint split (GOALS.md Terms) beside least squares and do-nothing on
every state of E93's table, through E93's rules unchanged. Plan: reports/plans/2026-11-05_plan_e93-d1-table.md →
"Amendment 1"; lines: results/predictions_amend1.json (the run refuses to start without it, and while
results/amend1/stopped.json exists). Everything this job writes is in results/amend1/.

  --stage control   stage 0, alone and first, on E40's step 10: do-nothing's masked K and least squares' share (E93's
                    stage 0), least squares under masks through `d3.cells`' new `state` path (E93's K11), and the
                    adjoint's old form on the clean model (E86's read); a miss writes stopped.json
  --states a,b,...  names from rules.STATES, in the order given; refuses before stage 0 has passed; written ones skipped
  --edges           CPU, no model: the record this job builds, through E93's rules, on do-nothing and edge cases

Per state. The reads that do not depend on the map come from E93's results/<state>/state.json: the gain rows' losses
(W0, FT, t), the clean W0 (the ruler), FT, random and signs, do-nothing and W0 masked ("base"), the ceiling and the random
under masks. This job reads the adjoint's parts (`d3.adjoint_parts`: c per site on E44's Σ, the share not held by the
subcomponents, its own Gaussian null, the squared size), then the clean read and gain at W0 + moved part (Δ fixed, so
that is the whole state), the masked cells with Γ on those weights reading the moved V′ (K11) and on W0's weights (K10),
and last the remainder δW − moved part (rule 1's teeth). Exit 1: a stop; 3: resubmit the same command.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import rules
import run
import torch

from commute import qwen_d3 as d3
from commute.jobs import EXPERIMENTS, hard_exit
from commute.qwen_sft import Seq, load

OUT = rules.RESULTS / "amend1"
PRED, CONTROL = rules.RESULTS / "predictions_amend1.json", OUT / "control.json"
run.RESULTS, run.STOPPED = OUT, OUT / "stopped.json"  # run.log, check and memory write here, not over E93's files
MAP_CLEAN = ("moved", "remainder", "inspan_old", "adjoint_old")  # E93's map-dependent reads, replaced or dropped


def write(name: str, obj: Any) -> None:
    run.write(OUT / name, obj)


def skeleton(e93: dict[str, Any]) -> dict[str, Any]:
    """E93's record with every least-squares read removed: what this job reuses."""
    rec = {k: v for k, v in e93.items() if k not in ("split", "own", "split_bf16", "seconds", "done")}
    rec["clean"] = {k: v for k, v in e93["clean"].items() if k not in MAP_CLEAN}
    rec["fractions"] = {k: v for k, v in e93["fractions"].items() if k != "moved"}
    return rec | {"map": "scaled adjoint split"}


# ----------------------------------------------------------------------------- stage 0


def stage_control(d: d3.Decomp, m: d3.RowSet, P: dict[str, Any]) -> None:
    R, F, out = P["controls"], P["first_minutes"], {"seconds": {}}

    def near(got: dict[str, float], want: dict[str, float]) -> bool:
        return all(run.rel(got[k], want[k]) <= R["rep"] for k in got)

    dW = load(run.E40_10)
    d.weights.put("FT", dW)
    b = run.cells_timed(d, m, "u1 step 10 base")
    got = {k: run.kmean(b, m, k) for k in ("K00", "K01")}
    out["do_nothing"] = run.check("do_nothing_E93", near(got, R["u1_masked"]), got)
    p = d3.ls_parts(d, dW)
    got = {"share": p.sums["left"] / p.sums["size"]}
    out["ls_share"] = run.check("ls_share_E93", near(got, {"share": R["u1_share"]}), got)
    d.weights.put("FT_copy", dW)
    t0 = time.time()
    o = run.cells_timed(d, m, "u1 step 10 ls own, state path", change=d3.ls_change(d, p.X), w0_v=False, state="FT_copy")
    out["seconds"]["own"] = time.time() - t0
    got = {k: run.kmean(o, m, k) for k in ("K10", "K11")}
    out["ls_state_path"] = run.check("ls_state_path_E93", near(got, R["u1_masked"]), got)
    del p, o
    d.weights.drop("FT_copy")  # returns the freed pinned pages, X's and the moved part's too
    a = d3.adjoint_parts(d, dW, old=True)
    out["seconds"]["parts"] = a.seconds
    run.log(f"[adjoint u1 step 10] {json.dumps(a.sums)}; {a.seconds:.0f} s, {d3.peaks()}")
    t0 = time.time()
    got = {"K": run.kmean(run.clean_at(d, m, "adjoint_old", a.old), m)}
    out["seconds"]["clean"] = time.time() - t0
    out["adjoint_old"] = run.check("adjoint_old_E86", near(got, {"K": R["e86_adjoint_old"]}), got)
    run.memory("control", F)
    s = out["seconds"]  # a maths state: the parts, three clean reads' worth (two clean, one gain), the own cells
    out["projection_min"] = (s["parts"] + 3 * s["clean"] + s["own"]) / 60
    write("control.json", out | {"passed": True})
    run.log(f"[control] passed; {out['projection_min']:.1f} min a maths state, {d3.peaks()}")


# ----------------------------------------------------------------------------- one state


def read_state(d: d3.Decomp, rs: d3.RowSet, seqs: list[Seq], name: str, F: dict[str, Any]) -> None:
    e93 = rules.load(name)
    assert e93 is not None, f"E93's {name} is not written"
    dW = d3.load_state({"path": EXPERIMENTS / e93["file"], "dtype": e93["dtype"]})
    nrm = d3.norm(dW)
    run.check(f"{name}_norm", run.rel(nrm, e93["norm"]) <= F["norm_rel"], {"norm": nrm, "e93": e93["norm"]})
    rec = skeleton(e93)
    d.weights.put("FT", dW)
    p = d3.adjoint_parts(d, dW, old=name in run.OLD_FORM)
    S, L = p.sums, e93["split"]
    rec["split"] = S | {"seconds": p.seconds, "c_zero": sum(c == 0 for c in p.c.values())}
    run.log(f"[adjoint {name}] share {S['left'] / S['size']:.9g} null {S['null'] / S['size']:.9g} (least squares "
            f"{L['left'] / L['size']:.9g}, its null {L['null'] / L['size']:.9g}) size {S['moved'] / S['size']:.9g} "
            f"second {S['second'] / S['size']:.3g} reproduced_sigma {S['reproduced'] / S['energy']:.6g} "
            f"c_zero {rec['split']['c_zero']}; {p.seconds:.0f} s, {d3.peaks()}")
    write(f"{name}.json", rec)
    d.weights.put("adjoint", p.moved)
    rec["clean"]["moved"] = run.clean_at(d, rs, "adjoint")
    run.memory(name, F)
    rec["fractions"]["moved"] = run.gain(d, "adjoint", seqs)
    write(f"{name}.json", rec)
    rec["own"] = run.cells_timed(d, rs, f"{name} adjoint own", change=d3.adjoint_change(d, dW, p.c), w0_v=False,
                                 state="adjoint")
    write(f"{name}.json", rec)
    d.weights.drop("adjoint")
    for s, w in dW.items():  # the remainder δW − moved part, in place of the moved part
        p.moved[s].neg_().add_(w.float())
    rec["clean"]["remainder"] = run.clean_at(d, rs, "remainder", p.moved)
    if p.old is not None:
        rec["clean"]["adjoint_old"] = run.clean_at(d, rs, "adjoint_old", p.old)
    d.weights.drop("FT")
    write(f"{name}.json", rec | {"done": True})
    t = score()["states"][name]
    run.log(f"[{name}] {json.dumps({k: t[k] for k in ('verdict', 'note') if k in t})} "
            f"rules {json.dumps(t['rules'], default=float)}")


# ----------------------------------------------------------------------------- scoring


def load_rec(name: str) -> dict[str, Any] | None:
    p = OUT / f"{name}.json"
    rec = json.loads(p.read_text()) if p.is_file() else None
    return rec if rec is not None and rec.get("done") else None


def predictions(states: dict[str, dict[str, Any]], P: dict[str, Any]) -> dict[str, Any]:
    """Each filed lean: the results it expects per arm label ("*" every other arm) for one rule or the verdict; holds
    when every state's result is among them, "unread" until every state is scored."""
    out: dict[str, Any] = {}
    for p in P["predictions"]:
        if any(n not in states for n in rules.STATES):
            out[p["id"]] = "unread"
            continue
        got = {n: s["verdict"] if p["rule"] == "verdict" else s["rules"][p["rule"]]["result"] for n, s in states.items()}
        off = {n: g for n, g in got.items() if g not in p["expect"].get(states[n]["arm"], p["expect"]["*"])}
        out[p["id"]] = {"result": rules.HOLDS if not off else rules.MISSES, "off": off}
    return out


def score_recs(recs: dict[str, dict[str, Any]], P: dict[str, Any]) -> dict[str, Any]:
    assert P["thresholds"] == rules.THRESHOLDS, "predictions_amend1.json's thresholds differ from rules.THRESHOLDS"
    sp = rules.spreads(recs)
    states = {n: rules.score_state(r, sp[r["arm"]][0]) | {"spread_borrowed": sp[r["arm"]][1]} for n, r in recs.items()}
    a = rules.arms(states)
    return {"states": states, "arms": a, "predictions": predictions(states, P),
            "table": rules.FAILS if rules.FAILS in a.values() else "passes" if set(a.values()) == {"passes"} else rules.PENDING}


def score() -> dict[str, Any]:
    t = score_recs({n: r for n in rules.STATES if (r := load_rec(n)) is not None}, json.loads(PRED.read_text()))
    write("table.json", t)
    return t


# ----------------------------------------------------------------------------- edge cases, before filing


def edges() -> bool:
    """The record `skeleton` + this job's reads gives, through E93's rules (rules 1-6, verdict), and the lean scorer."""
    e93 = rules.fake(k={"moved": 0.01, "remainder": 0.5})
    assert not {"split", "own"} & set(skeleton(e93)) and not set(MAP_CLEAN) & set(skeleton(e93)["clean"])

    def got(k: dict[str, float], fr: float, split: dict[str, float], nll: float, spread: float = 0.0) -> list[str]:
        new = rules.fake(k=k, fr={"moved": fr}, split=split, nll={"nll_m11": nll})
        rec = skeleton(e93) | {"split": new["split"], "own": new["own"]}
        rec["clean"] |= {n: new["clean"][n] for n in ("moved", "remainder")}
        rec["fractions"]["moved"] = new["fractions"]["moved"]
        s = rules.score_state(rec, spread)
        return [s["rules"][r]["result"] for r in "123456"] + [s["verdict"]]

    nothing = {"left": 1.0, "moved": 0.0, "null": 1.0}  # a map that moves nothing maps the Gaussian to nothing too
    cases = [
        ("Adjoint as the plan leans: clean 5 x its line, kept under t = 0.5", got({"moved": 0.1, "remainder": 0.4}, 2.0,
         {"left": 0.4, "moved": 0.5, "null": 0.6}, 2.6), [rules.MISSES, rules.HOLDS, rules.HOLDS, rules.FAILS, rules.HOLDS, "partial", rules.FAILS]),
        ("Do-nothing as the map (c = 0 at every site)", got({"moved": 1.0, "remainder": 0.0}, 3.0, nothing, 2.9),
         [rules.MISSES, rules.MISSES, rules.HOLDS, rules.FAILS, rules.HOLDS, rules.FAILS, rules.FAILS]),
        ("Share at its own null", got({}, 1.2, {"left": 0.6, "null": 0.6}, 2.5), [rules.HOLDS, rules.MISSES, rules.HOLDS, rules.HOLDS, rules.HOLDS, "partial", rules.FAILS]),
        ("Share above its own null", got({}, 1.2, {"left": 0.7, "null": 0.6}, 2.5), [rules.HOLDS, rules.MISSES, rules.HOLDS, rules.HOLDS, rules.HOLDS, "partial", rules.FAILS]),
        ("Size over 1 (plain norm, c fitted on Σ)", got({}, 1.2, {"moved": 1.3}, 2.5), [rules.HOLDS, rules.HOLDS, rules.MISSES, rules.HOLDS, rules.HOLDS, "partial", rules.FAILS]),
        ("Remainder reads as far as random: no teeth", got({"moved": 0.1, "remainder": 2.0}, 1.2, {}, 2.5), [rules.NO_CALL, rules.HOLDS, rules.HOLDS, rules.HOLDS, rules.HOLDS, "partial", rules.NO_CALL]),
        ("NaN null (δW = 0)", got({}, 1.2, {"null": math.nan}, 2.5), [rules.HOLDS, rules.NO_CALL, rules.HOLDS, rules.HOLDS, rules.HOLDS, "partial", rules.NO_CALL])]
    P = json.loads(PRED.read_text())
    full = {n: rules.fake(arm=a) | {"rows": "maths"} for n, (_, _, a) in rules.STATES.items()}
    lean = predictions(score_recs(full, P)["states"], P)
    part = predictions(score_recs({n: full[n] for n in list(full)[:3]}, P)["states"], P)
    cases += [("Lean scorer, every state a least-squares-like pass: verdict lean misses", lean["A7_verdict"]["result"], rules.MISSES),
              ("Lean scorer, size lean holds on that pass", lean["A3_size"]["result"], rules.HOLDS),
              ("Lean scorer, three states written", part["A7_verdict"], "unread")]
    ok = True
    with (OUT / "edges.txt").open("w") as f:
        for c, g, w in cases:
            line = f"{'ok ' if g == w else 'BAD'}  {c}: {g} (named {w})"
            ok &= g == w
            print(line, flush=True)
            f.write(line + "\n")
    return ok


# ----------------------------------------------------------------------------- the job


def main() -> None:
    ap = argparse.ArgumentParser(description="E93 Amendment 1: the scaled adjoint split on D1's first table")
    ap.add_argument("--stage", choices=("control",))
    ap.add_argument("--states", help="comma-separated names from rules.STATES")
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--decomp", default="")
    a = ap.parse_args()
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.edges:
        hard_exit(0 if edges() else 1)
    if run.STOPPED.is_file():
        sys.exit(f"refusing to start: {run.STOPPED} exists (a check missed; the lead decides)")
    names = a.states.split(",") if a.states else []
    if bool(a.stage) == bool(names) or any(n not in rules.STATES for n in names):
        sys.exit("give --stage control or --states from rules.STATES, not both")
    if names and not json.loads(CONTROL.read_text() if CONTROL.is_file() else "{}").get("passed"):
        sys.exit("refusing: stage 0 (--stage control) has not passed")
    P = json.loads(PRED.read_text())
    run.log(f"[start] amend1 {vars(a)}; config {run.C}")
    rows, seqs = run.build(P)
    with torch.no_grad():
        d = d3.Decomp(Path(a.decomp), run.log)
        if a.stage:
            stage_control(d, rows["maths"], P)
            hard_exit(0)
        took: list[float] = []
        for n in names:
            if load_rec(n) is not None:
                run.log(f"[{n}] already written")
                continue
            if time.time() - run.T0 + max(took, default=run.C.first_guess_s) > run.C.limit_s:
                run.log(f"cut short before {n}: {time.time() - run.T0:.0f} s used; resubmit")
                hard_exit(3)
            t0, rk = time.time(), rules.STATES[n][1]
            read_state(d, rows[rk], seqs[rk], n, P["first_minutes"])
            took.append(time.time() - t0)
            run.log(f"[{n}] done in {took[-1]:.0f} s, {d3.peaks()}")
    run.log(f"[done] amend1 {names}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)

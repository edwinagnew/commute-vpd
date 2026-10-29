"""regating_table (E46) Amendment 2: P0.3's seven random rows rerun with the scale cap at 100 (filed 10).
Plan: reports/plans/2026-10-28_plan_regating-line-reset.md -> "Amendment 2, after stage 0b". Rules, thresholds and the
tree: results/predictions_amend1.json (read here, never edited); P0.3's rule is predictions.json's, only the cap differs.

Stage 0b (Slurm 42675) capped 6 of 7 random rows at c = 10, short of the KL match P0.3 names. Per task, as stage 0b:
the seed-0 fine-tune's state (same file, same sha256), `finetune.random_direction(dW, 0)`, c matched by
`perturb.match` on (0, 100] to kl_target = stage0b.json's `<task>/s0` kl_other, the row read by `run.cells` with
released components. Everything is run.py's or perturb.py's (`run.random_row` takes the cap as an argument).

Control, first on each task and before its row: kl_other at the scale stage 0b settled on reproduces 0b's row within
repro_rel_tol (2%); a miss stops the job (exit 1). Score: `run.score` on stage0b.json's table with the seven random rows
replaced (P0.3 and the non-positive rule as filed; P0.2 and P0.4 read unchanged rows), then the amended tree. Each row
also reports G and g/G, unscored.

  amend1.py   -> results/amend1.json after each row; every scored number on stdout and results/large/full.log
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from typing import Any

from run import (
    E18_TASKS,
    E38_STATES,
    LADDER,
    LARGE,
    REPO,
    RESULTS,
    TRANSPORT,
    Clock,
    Run,
    Task,
    _finite,
    dump,
    git_head,
    load_checked,
    log,
    random_row,
    read_state,
    score,
)
from torch import Tensor

from commute import perturb as PT
from commute.finetune import random_direction

PRED_A, OUT = RESULTS / "predictions_amend1.json", "amend1.json"
ROWS = (*E18_TASKS, "ident32", "ident32b", "rand32", "rand32b")
SHARED = ("F_min", "p03_fail_frac", "kl_rel_tol", "nonpos_margin")  # must equal predictions.json's
FROZEN = ("src/commute/perturb.py", "src/commute/anchors.py")  # run.py differs from 0b's commit by random_row's cap argument


def load(run: Run, s0a: dict[str, Any], name: str) -> tuple[Task, dict[str, Tensor]]:
    """The task and its seed-0 fine-tune's delta-W, by the files and sha256 stage 0b read."""
    from commute import regating as G

    if name in E18_TASKS:
        t = run.task(name, *G.load_windows(name, TRANSPORT, LADDER))
        return t, load_checked(E38_STATES / f"{name}_s0.pt", run.P["refs"]["e18"][name]["sha256"]["s0"])
    w = load_checked(LARGE / f"windows_{name}.pt", s0a["groups"]["e12/build"]["sha256"][name])
    t = run.task(name, w["train"], w["heldout"], w["table"])
    return t, load_checked(LARGE / f"{name}_s0.pt", s0a["groups"][f"e12/{name}"]["states"]["s0"]["sha256"])


def repro(run: Run, t: Task, dW: dict[str, Tensor], old: dict[str, Any], tol: float) -> dict[str, Any]:
    """The control: kl_other at stage 0b's scale against 0b's recorded kl_other."""
    kl = read_state(run, t, PT.scaled(random_direction(dW, 0), old["scale"]))[1]
    return {"scale": old["scale"], "kl_other": kl, "kl_0b": old["kl_other"], "rel": kl / old["kl_other"] - 1,
            "ok": abs(kl / old["kl_other"] - 1) <= tol}


def amended_score(S: dict[str, dict[str, Any]], P: dict[str, Any], A: dict[str, Any]) -> dict[str, Any]:
    s, th = score(S, P), A["thresholds"]
    p3, keys = s["P0.3"], [f"{n}/random" for n in A["rows"]]
    for k in keys:
        x = S[k]
        p3["rows"][k].update(c=x["scale"], kl_other=x["kl_other"], kl_target=x["kl_target"], G=x["G"],
                             g_over_G=x["g"] / x["G"] if abs(x["G"]) >= th["g_over_G_min_abs_G"] else None)
    cap = [k for k in keys if S[k]["match"] == "cap"]
    nonfinite = [k for k in keys if not all(math.isfinite(S[k][q]) for q in ("g", "G", "F"))]
    st = (s["P0.2"]["status"], p3["status"], s["P0.4"]["status"])
    tree = ("stop: fix regeneration" if st[0] != "pass" else "card: P0.3 fail" if st[1] == "fail" else "retire" if st[2] == "fail"
            else "card: cap" if cap else "propose stage 1" if st[1:] == ("pass", "pass") else "card")
    return {"P0.3": {**p3, "cap": cap, "nonfinite": nonfinite}, "P0.2": s["P0.2"]["status"], "P0.4": s["P0.4"]["status"],
            "sign_unscored": s["sign_unscored"], "tree": tree}


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if not PRED_A.is_file():
        sys.exit(f"file the predictions first: {PRED_A}")
    A = json.loads(PRED_A.read_text())
    s0a, s0b = (json.loads((RESULTS / f).read_text()) for f in ("stage0a.json", "stage0b.json"))
    assert tuple(A["rows"]) == ROWS and A["seed"] == 0, "predictions_amend1.json's rows or seed differ from the code's"
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", s0b["commit"], "--", *FROZEN], check=False).returncode:
        sys.exit(f"{', '.join(FROZEN)} changed since stage 0b's commit {s0b['commit']}")
    run, th = Run("cuda"), A["thresholds"]
    assert all(run.P["thresholds"][k] == th[k] for k in SHARED), "P0.3's filed settings differ from predictions.json's"
    S = dict(s0b["states"])
    out: dict[str, Any] = {"commit": git_head(), "stage0b_commit": s0b["commit"], "scale_cap": th["scale_cap"], "rows": {}}
    clock = Clock()
    for n in ROWS:
        key = f"{n}/random"
        t, dW = load(run, s0a, n)
        c = repro(run, t, dW, s0b["states"][key], th["repro_rel_tol"])
        out["rows"][key] = {"repro": c}
        dump(OUT, out)
        log(f"[amend1 {key} control] {json.dumps(_finite(c))}")
        if not c["ok"]:
            log(f"[amend1] FAIL: {key}'s kl_other at stage 0b's scale does not reproduce: stop")
            sys.exit(1)
        rec = random_row(run, t, dW, s0b["states"][f"{n}/s0"], th["scale_cap"])
        S[key] = out["rows"][key] = {"kind": "random", "task": n, **rec, "repro": c}
        dump(OUT, out)
        log(f"[amend1 {key}] c {rec['scale']:.4f} match {rec['match']} kl_other {rec['kl_other']:.5f} kl_target "
            f"{rec['kl_target']:.5f} g {rec['g']:.4f} G {rec['G']:.4f} F {rec['F']:.4f} se {rec['se']:.4f} n {rec['n']} "
            f"| grid {json.dumps(rec['grid'])} | trace {json.dumps(rec['trace'])} | cells {json.dumps(rec['cells'])}")
        del t, dW
        clock.lap(f"amend1 {n}")
    out["score"] = amended_score(S, run.P, A)
    dump(OUT, out)
    for k, r in out["score"]["P0.3"]["rows"].items():
        log(f"[amend1 score {k}] {json.dumps(_finite(r))}")
    sc = out["score"]
    log(f"[amend1 score] P0.3 {sc['P0.3']['status']} cap {sc['P0.3']['cap']} nonfinite {sc['P0.3']['nonfinite']} "
        f"nonpositive {json.dumps(_finite(sc['P0.3']['nonpositive']))} | P0.2 {sc['P0.2']} P0.4 {sc['P0.4']} | tree {sc['tree']}")


if __name__ == "__main__":
    main()

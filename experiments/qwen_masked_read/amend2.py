"""E79 Amendment 2: does A's own-mask read beat do-nothing? ($0, one cluster L40, `debug`, ~3 min.) Plan: reports/plans/2026-11-04_plan_e79-agreement-or-softness.md,
Amendment 2; rules: rules_amend2.py, results/predictions_amend2.json (the job refuses to start without it, and while results/stopped_amend2.json exists).

Two states, draw d1, on M and W, against FT clean and against the state's own clean forward. N, do-nothing: FT's weights, components unmoved, no offsets, Γ on FT's
forward. Z: W0's weights and components, no offsets, under A's own g. Control first and alone, each or a stop: W0 own (with d2, for the band) and A own on M reproduce
job 51161's; Z's path given W0's g reads W0's number; N's clean reads FT clean. results/amend2.json after every cell, every number in results/run.log, per-position
arrays in results/large/a2_*. One state's own g, one row set, is on the host at a time.

N's weights hold δW (`qwen_teacher.do_nothing`, a swap of the matrices), so `student_masked` is the only forward hook that changes a layer's output and its `out` is
x (W0 + δW)ᵀ: Δ carries Δ₀ + δW with no hook order to get wrong. FT clean comes through `change_hooks`, so N's clean KL to it compares the two paths.

Usage (repo root): .venv/bin/python experiments/qwen_masked_read/amend2.py --decomp DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import amend1
import rules_amend2 as rules
import run
import torch
from safetensors.torch import save_file
from torch import Tensor

from commute.jobs import hard_exit
from commute.qwen_sft import linears
from commute.qwen_teacher import do_nothing
from commute.qwen_vpd import importances

e76, DEV, LARGE, log = run.e76, run.DEV, run.LARGE, run.log
PRED, OUT, STOPPED = (run.RESULTS / f for f in ("predictions_amend2.json", "amend2.json", "stopped_amend2.json"))
G = list[dict[str, Tensor]]


def stop(rule: str, detail: str) -> None:
    log(f"[a2] STOP {rule}: {detail}")
    e76.write_json(STOPPED, {"rule": rule, "detail": detail})
    sys.exit(f"STOP {rule}: {detail}")


def keep(J: run.Job, state: str, rs: str, cell: str, pos: dict[str, Tensor]) -> dict[str, Any]:
    save_file({k: v.to(amend1.DTYPES.get(k, torch.float16)).contiguous() for k, v in pos.items()}, str(LARGE / f"a2_{state}_{rs}_{cell}.safetensors"))
    s = {"all": run.summary(pos, torch.ones_like(next(iter(pos.values())), dtype=torch.bool))}
    J.out["reads"].setdefault(state, {}).setdefault(rs, {})[cell] = s
    log(f"[a2 {state} {rs} {cell}] {json.dumps(s)}")
    e76.write_json(OUT, J.out)
    return s


def read_clean(J: run.Job, state: str, rs: str, clean: list[Tensor]) -> float:
    """The forward KL(FT clean ‖ state clean), on the weights and offsets the caller has set."""
    return keep(J, state, rs, "clean", run.read_rows((x.to(DEV) for x in clean), J.ft[rs]))["all"]["fwd"]


def read_masked(J: run.Job, state: str, rs: str, g: G, clean: list[Tensor], cells: Sequence[str] = ("d1",)) -> float:
    """Σg and the masked forwards against FT clean and `clean`, on the weights and offsets the caller has set; each draw's generator restarts per row set
    (`run.masked_rows`), so masks on the same g and draw are the same masks. Returns d1's forward KL to FT clean."""
    keep(J, state, rs, "g", amend1.g_numbers(g))
    got = {cell: keep(J, state, rs, cell, amend1.masked_cell(J, rs, run.SEEDS[cell], g, clean))["all"]["fwd"] for cell in cells}
    log(f"[a2 {state} {rs}] {run.peak()}")
    return got["d1"]


def control(J: run.Job, name: str, got: float, ok: bool, want: str) -> None:
    """`ok` false on a NaN, as the callers compare with `<`, `<=`."""
    J.out["control"][name] = {"got": got, "want": want}
    log(f"[a2 control {name}] {got}, want {want}")
    e76.write_json(OUT, J.out)
    if not ok:
        stop("control", f"{name} reads {got}, want {want}")


def read_placed(J: run.Job, W0: dict[str, Tensor], state: str, cells: Sequence[str]) -> float:
    """W0 or A on M, Γ on its own forward, its offsets in the masked forward: as job 51161."""
    run.set_state(J, state)
    clean = run.clean_logits(J.model, J.rows["M"], J.off.change)
    read_clean(J, state, "M", clean)
    return read_masked(J, state, "M", run.own_g(J, W0, "M"), clean, cells)


def read_unplaced(J: run.Job, W0: dict[str, Tensor], source: str, state: str, rs: str) -> float:
    """`source`'s own g, computed under its offsets, then W0's weights and components with the offsets at zero. `source`'s offsets go out before the masked read."""
    run.set_state(J, source)
    g = run.own_g(J, W0, rs)
    J.off.reset()
    clean = run.clean_logits(J.model, J.rows[rs], None)
    read_clean(J, state, rs, clean)
    return read_masked(J, state, rs, g, clean)


def stage_control(J: run.Job, W0: dict[str, Tensor]) -> None:
    """W0, A, then Z's path. A's offsets are in `J.off` when Z's path starts, so its reading W0's number also shows `reset` leaves none behind. N's clean is `stage_n`'s first read."""
    w0 = read_placed(J, W0, "W0", ("d1", "d2"))
    control(J, "W0", w0, abs(w0 - rules.W0_OWN) <= rules.TOL, f"{rules.W0_OWN} within {rules.TOL}")
    a = read_placed(J, W0, "A", ("d1",))
    control(J, "A", a, abs(a - rules.A_OWN) <= rules.TOL, f"{rules.A_OWN} within {rules.TOL}")
    z = read_unplaced(J, W0, "W0", "Z_W0g", "M")
    control(J, "Z_W0g", z, abs(z - w0) <= rules.TOL, f"W0's {w0} within {rules.TOL}")


def ft_g(J: run.Job, rs: str) -> G:
    """Γ on the forward the model holds now, components unmoved, bf16 as `run.own_g`. `own_g` itself would swap in W0 + the offsets' change, and a copy per call."""
    return [{s: v.to("cpu", torch.bfloat16) for s, v in importances(J.model, t[None].to(DEV), J.gamma, J.comps)[1].items()} for t in J.rows[rs]]


def stage_n(J: run.Job, W0: dict[str, Tensor]) -> None:
    """N's clean on M is the last control, before Γ runs on these weights. δW is released as the block opens: only W0 + δW stays on the card."""
    J.off.reset()
    with do_nothing(J.model, W0, run.load_dw()):
        for rs in J.rows:  # M first
            clean = run.clean_logits(J.model, J.rows[rs], None)
            got = read_clean(J, "N", rs, clean)
            if rs == "M":
                control(J, "N_clean", got, got < rules.N_CLEAN_UNDER, f"under {rules.N_CLEAN_UNDER}")
            read_masked(J, "N", rs, ft_g(J, rs), clean)


def stage_z(J: run.Job, W0: dict[str, Tensor]) -> None:
    for rs in J.rows:
        read_unplaced(J, W0, "A", "Z", rs)


def score() -> None:
    """From what amend2.json holds: a stop or a crash still scores, with "unread" for what never ran."""
    out = e76.read_json(OUT)
    if out is None:
        return
    out["score"] = rules.score(out["reads"])
    e76.write_json(OUT, out)
    log(f"[a2 score] {json.dumps(out['score'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit("refusing to start: predictions_amend2.json missing, or stopped_amend2.json exists")
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    OUT.unlink(missing_ok=True)  # a rerun must not score an earlier job's reads
    torch.set_grad_enabled(False)
    model, comps, off, gamma, rows = run.load_parts(Path(a.decomp))
    ft = run.ft_clean(model, rows)
    log(f"[a2 FT] {run.peak()}")
    J = run.Job(model, comps, off, gamma, rows, {}, {}, ft, {}, {"control": {}, "reads": {}})
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    stage_control(J, W0)
    stage_n(J, W0)
    stage_z(J, W0)
    log(f"[a2 done] {run.peak()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    except BaseException:  # noqa: BLE001  (an out-of-memory or the like: the same exit, never a hang in teardown)
        traceback.print_exc()
        code = 1
    try:
        score()
    finally:
        hard_exit(code)

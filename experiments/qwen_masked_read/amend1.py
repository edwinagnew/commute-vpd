"""E79 Amendment 1: does the own-mask read have teeth? ($0, one cluster L40, `debug`, ~4 min.) Plan: reports/plans/2026-11-04_plan_e79-agreement-or-softness.md,
Amendment 1; rules: rules_amend1.py, results/predictions_amend1.json (the job refuses to start without it, and while results/stopped_amend1.json exists).

W0, A, B, L and two bad placements, P (A's offsets on the wrong columns, `wrong_site`) and C (E76's control student), each read with Γ on its own
forward, draw d1, on M and W, against FT clean and against its own clean forward. W0 also reads d2, whose spread is the band. Control first and
alone: W0 and then A reproduce job 51117's own-mask read on M, or a stop. results/amend1.json after every cell, every number in results/run.log, per-position arrays in
results/large/a1_*. One state's own g, one row set, is on the host at a time.

Usage (repo root): .venv/bin/python experiments/qwen_masked_read/amend1.py --decomp DIR
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any

import rules_amend1 as rules
import run
import torch
from safetensors.torch import save_file
from torch import Tensor

from commute.jobs import hard_exit
from commute.qwen_sft import linears
from commute.qwen_teacher import Offsets
from commute.qwen_vpd import SITES
from commute.vpd import kl_per_position

e76, DEV, LARGE, log = run.e76, run.DEV, run.LARGE, run.log
PRED, OUT, STOPPED = (run.RESULTS / f for f in ("predictions_amend1.json", "amend1.json", "stopped_amend1.json"))
P_SIZE_TOL = 1e-3  # relative, per site: far over fp32's rounding of the recomputed change
DTYPES = {"agree": torch.uint8, "sum_g": torch.float32, "count": torch.int32}  # float16 stops at 65504; the rest are KLs and entropies


def stop(rule: str, detail: str) -> None:
    log(f"[a1] STOP {rule}: {detail}")
    e76.write_json(STOPPED, {"rule": rule, "detail": detail})
    sys.exit(f"STOP {rule}: {detail}")


def size2(x: Tensor) -> float:
    return float((x * x).sum(dtype=torch.float64))


@torch.no_grad()
def wrong_site(off: Offsets, s: str, gen: torch.Generator) -> dict[str, float]:
    """Site s of P, its offsets holding A's: dV and dU permuted together over the cut columns, then both scaled by one number so that the change's squared
    size (E76's size, `Offsets.change`) equals A's. The permutation alone keeps the offsets' size, not the change's: V and U do not move with it."""
    want = size2(off.change(s))
    perm = torch.randperm(off.dV[s].shape[1], generator=gen).to(off.dV[s].device)
    dV, dU = off.dV[s][:, perm], off.dU[s][perm]
    L, Q = dV @ off.U[s] + off.V[s] @ dU, dV @ dU  # the change at scale k is k L + k² Q
    a, b, c = size2(L), float((L * Q).sum(dtype=torch.float64)), size2(Q)
    k = rules.scale_to(a, b, c, want)
    off.dV[s].copy_(k * dV)
    off.dU[s].copy_(k * dU)
    got = size2(off.change(s))
    assert math.isclose(got, want, rel_tol=P_SIZE_TOL, abs_tol=1e-30), f"{s}: P's squared size {got}, A's {want}"
    return {"want": want, "permuted": a + 2 * b + c, "scale": k}


def place_p(J: run.Job) -> None:
    """A's offsets, then P's. The seed's stream runs over SITES in order on the host, so P is the same on any card."""
    run.set_state(J, "A")
    gen = torch.Generator("cpu").manual_seed(rules.P_SEED)
    sites = [wrong_site(J.off, s, gen) for s in SITES]
    scales = [x["scale"] for x in sites]
    J.out["P"] = {"seed": rules.P_SEED, "size_A": sum(x["want"] for x in sites), "size_permuted_over_A": sum(x["permuted"] for x in sites) / sum(x["want"] for x in sites),
                  "scale_min": min(scales), "scale_median": statistics.median(scales), "scale_max": max(scales), "scale": scales}
    log(f"[a1 P] { {k: v for k, v in J.out['P'].items() if k != 'scale'} }")
    e76.write_json(OUT, J.out)


def put(J: run.Job, state: str) -> None:
    if state == "P":
        place_p(J)
    else:
        run.set_state(J, state)


def g_numbers(g: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Number c, [rows, T] on the host: Σg and the count of g > 0.5 per position, over all 196 sites and all columns."""
    sums, counts = [], []
    for row in g:
        total, count = torch.zeros(1, device=DEV, dtype=torch.float64), torch.zeros(1, device=DEV, dtype=torch.int64)
        for v in row.values():
            x = v.to(DEV, torch.float32)
            total = total + x.sum(-1, dtype=torch.float64)
            count = count + (x > 0.5).sum(-1)
        sums.append(total[0].cpu())
        counts.append(count[0].cpu())
    return {"sum_g": torch.stack(sums), "count": torch.stack(counts)}


def masked_cell(J: run.Job, rs: str, seed: int, g: list[dict[str, Tensor]], clean: list[Tensor]) -> dict[str, Tensor]:
    """Numbers a and b per position, from one masked forward per row."""
    parts = []
    for r, q in enumerate(run.masked_rows(J, rs, seed, g)):
        parts.append(run.positions(J.ft[rs][r].to(DEV), q) | {"self_fwd": kl_per_position(clean[r].to(DEV), q)[0].cpu()})
    return {k: torch.stack([p[k] for p in parts]) for k in parts[0]}


def keep(J: run.Job, state: str, rs: str, cell: str, pos: dict[str, Tensor]) -> dict[str, Any]:
    save_file({k: v.to(DTYPES.get(k, torch.float16)).contiguous() for k, v in pos.items()}, str(LARGE / f"a1_{state}_{rs}_{cell}.safetensors"))
    s = {"all": run.summary(pos, torch.ones_like(next(iter(pos.values())), dtype=torch.bool))}
    J.out["reads"].setdefault(state, {}).setdefault(rs, {})[cell] = s
    log(f"[a1 {state} {rs} {cell}] {json.dumps(s)}")
    e76.write_json(OUT, J.out)
    return s


def read_set(J: run.Job, state: str, rs: str, W0: dict[str, Tensor]) -> None:
    """The state in J.off on one row set; its clean logits and own g are released on return. W0 reads both draws, for the bands."""
    if rs in J.out["reads"].get(state, {}):
        return
    clean = run.clean_logits(J.model, J.rows[rs], J.off.change)
    keep(J, state, rs, "clean", run.read_rows((x.to(DEV) for x in clean), J.ft[rs]))
    g = run.own_g(J, W0, rs)
    keep(J, state, rs, "g", g_numbers(g))
    for cell in ("d1", "d2") if state == "W0" else ("d1",):
        keep(J, state, rs, cell, masked_cell(J, rs, run.SEEDS[cell], g, clean))
    log(f"[a1 {state} {rs}] {run.peak()}")


def stage_control(J: run.Job, W0: dict[str, Tensor]) -> None:
    for state, want in (("W0", rules.W0_OWN), ("A", rules.A_OWN)):
        put(J, state)
        read_set(J, state, "M", W0)
        got = J.out["reads"][state]["M"]["d1"]["all"]["fwd"]
        J.out["control"][state] = {"got": got, "want": want}
        log(f"[a1 control {state}] {got}, job 51117's {want}")
        e76.write_json(OUT, J.out)
        if not abs(got - want) <= rules.TOL:
            stop("control", f"{state} under own masks reads {got}, job 51117 {want}, within {rules.TOL}")


def stage_states(J: run.Job, W0: dict[str, Tensor]) -> None:
    for state in rules.STATES:
        put(J, state)
        for rs in J.rows:
            read_set(J, state, rs, W0)


def score() -> None:
    """From what amend1.json holds: a stop or a crash still scores, with "unread" for what never ran."""
    out = e76.read_json(OUT)
    if out is None:
        return
    out["score"] = rules.score(out["reads"])
    e76.write_json(OUT, out)
    log(f"[a1 score] {json.dumps(out['score'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit("refusing to start: predictions_amend1.json missing, or stopped_amend1.json exists")
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    OUT.unlink(missing_ok=True)  # a rerun must not score an earlier job's reads
    torch.set_grad_enabled(False)
    model, comps, off, gamma, rows = run.load_parts(Path(a.decomp))
    ft = run.ft_clean(model, rows)
    log(f"[a1 FT] {run.peak()}")
    J = run.Job(model, comps, off, gamma, rows, {}, {}, ft, {}, {"control": {}, "reads": {}})
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    stage_control(J, W0)
    stage_states(J, W0)
    log(f"[a1 done] {run.peak()}")


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

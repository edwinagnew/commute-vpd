"""Cell P3.13 run 1 (E216, rung L3.2): three readings of the six E167 step-75 states of Qwen3-1.7B (RL, the shuffled-reward
run that collapsed, own-samples SFT; seeds 0 and 70), each beside its random reference. Plan:
reports/plans/2026-11-19_plan_l3-2-qwen-readings.md (+ _appendix.md); card card.md; settings, expected values and rules
in results/predictions.json (the run refuses unless it is committed unchanged, or PRED_SHA matches on a pod). A
description: no update is called reused or new.

  (a) share   E57's re-gating share on the frozen-neuron decomposition p-015e0710 (84 MLP matrices; every change's
              attention matrices set to zero), on the 128 maths rows of 256, against a random curve of 3 Gaussian draws
              at R_s0's MLP norms x 3 sizes (E186's recipe, `share_reading`).
  (b) S       the held-model reading (`commute.qwen_held`): maps fitted on maths rows 0-63, KL read on rows 64-127,
              beside a Gaussian change at the state's own per-matrix norms (196 matrices).
  (c) erank   effective rank of δW (`update_spectrum`, P3.6's definition) beside a Gaussian floor (2 draws).

Stages (each record is written to results/<stage>/<name>.json and printed as it finishes; a rerun skips what is written):
  0  alone, the controls; any miss writes results/stopped.json and exits 1. Planted ranks 1 and 10 read 1 and 10. Share:
     W0 as F on 16 rows, the do-nothing split (share 1), W0's floor K0 on 128 rows, and R_s0's masked size K00 - K0
     (reported, flag 2). S: the do-nothing (F = M: KL(F||M) exactly 0, S 'no change'; its KL(M||H) sets S's floor) and
     R_s0's change at o_proj and down_proj only (S within 1e-3 of 1).
  1  erank of each state and its floor.        2  S of each state, then of its random change.
  3  the share of each state.                  4  the random curve (3 draws x 3 sizes from the scored states' K00).
Inputs (gitignored; a pod must be sent them): the six states, --states <dir>/<name>/dW/step_75.safetensors (5.64 GB
each, sha256 against experiments/qwen_rl_readable/results/run2/<name>.json); --decomp, p-015e0710 (30 GB); its W0 g
statistics, experiments/qwen_library_share/results/large/p-015e0710/gstats/W0_{maths,fineweb}.safetensors (2 x 2.1 MB).

  python experiments/qwen_held_readings/run.py --decomp "$COMMUTE_QWEN_FROZEN_DIR" 2>&1 | tee -a <pod log>
  python experiments/qwen_held_readings/run.py --summary        (no GPU: results/summary.json)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import qwen_held as QH
from commute import share_reading as SR
from commute import update_spectrum as US
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)
from commute.qwen_d3 import ALIVE_TOTAL_OF, GSTATS_OF, null_seed
from commute.qwen_sft import load, sha256
from commute.qwen_vpd import (
    SITES,
    decomp_id,
    fp32_matmuls,
    load_target,
    parse_site,
    site_name,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
E167 = EXPERIMENTS / "qwen_rl_readable" / "results"
FROZEN, MLP = "p-015e0710", ("gate", "up", "down")
L = load_experiment("qwen_reading", "lab")  # E57's Lab at this folder, p-015e0710's g statistics and live-set count
L.RESULTS, L.PRED, L.LARGE, L.GSTATS = RESULTS, PRED, RESULTS / "large", GSTATS_OF[FROZEN]
L.CFG = replace(L.CFG, alive_total=ALIVE_TOTAL_OF[FROZEN])
DEV = "cuda"


@dataclass(frozen=True)
class Config:  # predictions.json "settings"
    states: tuple[str, ...]
    step: int
    control: str  # S's positive control (its o_proj and down_proj part), the share's random norms, flag 2's size
    n_rows: int  # maths rows (the Lab's)
    n_fit: int  # S: rows [0, n_fit) fit, [n_fit, n_rows) read
    batch: int  # rows per S forward
    s_tol: float  # positive control: 1 - S at most this
    zero_kl: float  # do-nothing: KL(F||M) per position at most this (F and M hold the same weights)
    s_floor_factor: float  # S 'no change' when KL(F||M) <= this x the do-nothing's KL(M||H), per position
    rank_tol: float  # planted ranks, relative
    rank_draws: tuple[int, ...]
    s_draw: int
    share_draws: tuple[int, ...]
    widen: float  # the curve's sizes: lowest scored K00 / widen, highest x widen, their geometric middle
    n16: int
    n_size: int
    search_match: float
    match: float
    evals: int
    b0: float

    def __post_init__(self) -> None:
        assert self.control in self.states and 0 < self.n_fit < self.n_rows == L.CFG.n_rows
        assert self.n_fit % self.batch == 0 and (self.n_rows - self.n_fit) % self.batch == 0
        assert self.n16 <= self.n_size <= self.n_rows and 1 < self.search_match < self.match and self.widen >= 1


C: Config


# ----------------------------------------------------------------------------- files, stops


def log(msg: str) -> None:
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(msg + "\n")


def get(name: str) -> dict[str, Any]:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def put(name: str, rec: dict[str, Any]) -> dict[str, Any]:
    L.write_json(RESULTS / f"{name}.json", rec)
    log(f"[{name}] {json.dumps(rec, default=float)}")
    return rec


def check(name: str, holds: bool, detail: dict[str, Any]) -> None:
    log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
    if not holds:
        L.write_json(RESULTS / "stopped.json", {"name": name, "detail": detail})
        log(f"STOP {name}")
        hard_exit(1)


_CHECKED: set[str] = set()


def state(root: Path, name: str) -> dict[str, Tensor]:
    """A step-75 δW (host, fp32), its sha256 checked against its pod's record once per job."""
    path = root / name / "dW" / f"step_{C.step}.safetensors"
    if name not in _CHECKED:
        want = json.loads((E167 / "run2" / f"{name}.json").read_text())["saves"][str(C.step)]["sha256"]
        got = sha256(path)
        check(f"sha256_{name}", got == want, {"file": str(path), "sha256": got, "want": want})
        _CHECKED.add(name)
    dW = load(path)
    assert set(dW) == set(SITES), path  # linear matrices only: F's norms and embeddings are W0's
    return dW


def norms(dW: dict[str, Tensor]) -> dict[str, float]:
    return {s: float(m.double().norm()) for s, m in dW.items()}


def shapes_of(dW: dict[str, Tensor]) -> dict[str, tuple[int, ...]]:
    return {s: tuple(m.shape) for s, m in dW.items()}


# ----------------------------------------------------------------------------- (c) effective rank


def rank_controls(shapes: dict[str, tuple[int, ...]]) -> None:
    """Layer 0's seven shapes: k orthonormal pairs with equal singular values read erank k."""
    if get("control/rank").get("done"):
        return
    l0 = {s: (sh[0], sh[1]) for s, sh in shapes.items() if parse_site(s)[0] == 0}
    got = {k: US.state({s: m.to(DEV) for s, m in US.equal_rank(l0, k, 0).items()})["erank"] for k in (1, 10)}
    check("planted_rank", all(v is not None and abs(v - k) <= C.rank_tol * k for k, v in got.items()), {"erank": got})
    put("control/rank", {"erank": got, "done": True})


def rank_floor(shapes: dict[str, tuple[int, ...]]) -> dict[str, dict[str, float]]:
    """Per matrix, the erank of one iid Gaussian draw per draw seed (null_seed(d, site)); scale-free, so one per job."""
    if rec := get("rank/floor_draws"):
        return rec
    out = {str(d): {s: US.matrix(torch.randn(sh, generator=torch.Generator().manual_seed(null_seed(d, s)),
                                             dtype=torch.float64).to(DEV))["erank"] for s, sh in shapes.items()}
           for d in C.rank_draws}
    return put("rank/floor_draws", out)


def rank_state(name: str, dW: dict[str, Tensor], floor: dict[str, dict[str, float]]) -> None:
    t0 = time.time()
    per = {s: US.matrix(m.to(DEV)) for s, m in dW.items()}
    draws = [US.aggregate({s: {"erank": floor[d][s], "top10": 0.0, "norm2": per[s]["norm2"]} for s in per})["erank"]
             for d in floor]
    st = US.aggregate(per)
    put(f"rank/{name}", {"erank": st["erank"], "top10": st["top10"], "norm2": st["norm2"], "floor_draws": draws,
                         "floor": sum(x for x in draws if x is not None) / len(draws), "seconds": time.time() - t0})


# ----------------------------------------------------------------------------- (b) S


class Held(QH.Held):
    """`qwen_held.Held` (moved there 2026-11-19 for run 2, arithmetic unchanged) on maths rows, Grams kept on the GPU as
    this run ran; `read` names, logs and writes each record."""

    def __init__(self, rows: Tensor) -> None:
        super().__init__(load_target(DEV), load_target(DEV), list(rows[: C.n_fit].to(DEV).split(C.batch)),
                         list(rows[C.n_fit :].to(DEV).split(C.batch)), log)

    def read_as(self, name: str, dW: dict[str, Tensor] | None, floor: float, read_residual: bool,
                extra: dict[str, Any]) -> dict[str, Any]:
        rec = self.read(dW, floor, read_residual) | extra
        log(f"[S] {name}: S {rec['S']}, KL(F||M) {rec['kl_fm']:.6g}, KL(F||H) {rec['kl_fh']:.6g}, {rec['seconds']:.0f} s")
        return put(f"S/{name}", rec)


def s_controls(h: Held, ctrl: dict[str, Tensor]) -> float:
    """The do-nothing (KL(F||M) exactly 0, S None; its KL(M||H) x s_floor_factor is S's floor) and the control state's
    output-matrices-only change (S within s_tol of 1). Returns the floor."""
    rec = get("S/do_nothing") or h.read_as("do_nothing", None, C.zero_kl, True, {})
    check("S_do_nothing", rec["kl_fm"] <= C.zero_kl and rec["S"] is None, {k: rec[k] for k in ("kl_fm", "kl_fh", "S")})
    floor = C.s_floor_factor * rec["kl_fh"]
    out = {s: (m if s in QH.SITES else torch.zeros_like(m)) for s, m in ctrl.items()}
    pos = get("S/output_only") or h.read_as("output_only", out, floor, True, {"from": C.control})
    check("S_output_only", pos["S"] is not None and 1 - pos["S"] <= C.s_tol, {"S": pos["S"], "kl_fm": pos["kl_fm"]})
    return floor


def s_states(h: Held, root: Path, floor: float) -> None:
    for name in C.states:
        if not get(f"S/{name}"):
            h.read_as(name, state(root, name), floor, False, {"state": name})
        if not get(f"S/{name}_random"):
            dW = state(root, name)
            R = SR.gaussian(norms(dW), shapes_of(dW), C.s_draw)
            del dW
            h.read_as(f"{name}_random", R, floor, True, {"state": name, "draw": C.s_draw})
    times = [get(f"S/{n}")["seconds"] for n in C.states]
    log(f"[S] time per state {max(times):.0f} s at most; run 2's 18 reads a pod ~{18 * max(times) / 3600:.2f} h")


# ----------------------------------------------------------------------------- (a) share


def own(dW: dict[str, Tensor]) -> dict[str, Tensor]:
    """δW with the attention matrices set to zero: p-015e0710 decomposes the MLP only (lead's decision 2026-11-13)."""
    return {s: m if parse_site(s)[1] in MLP else torch.zeros_like(m) for s, m in dW.items()}


def share_controls(lab: Any, rows: Tensor, ctrl: dict[str, Tensor]) -> None:
    """W0 as F (16 rows), the do-nothing split of draw 0 at b0 (share 1), the floor K0 on 128 rows, the control's size."""
    if not get("control/W0"):
        put("control/W0", SR.w0_control(lab, "maths", rows[: C.n16]))
    mlp_norms = {s: v for s, v in norms(ctrl).items() if parse_site(s)[1] in MLP}
    shapes = shapes_of(ctrl)
    if not get("control/do_nothing"):
        R = SR.gaussian(mlp_norms, shapes, C.share_draws[0])
        for m in R.values():
            m.mul_(C.b0)
        put("control/do_nothing", SR.do_nothing(lab, SR.with_zeros(R, shapes), rows[: C.n16]))
    if not get("share/floor"):
        lab.set_weights("FT", None)
        k0 = sum(lab.cells(rows, tag="W0 floor, 128 rows")["K0"]) / rows.numel()
        lab.drop_set("FT")
        put("share/floor", {"K0": k0, "rows": len(rows), "norms_from": C.control, "norms": mlp_norms})
    if not get("share/control_size"):
        ex = SR.excess(lab, own(ctrl), rows)
        put("share/control_size", {"state": C.control, "excess": ex, "status_floor": 0.05,
                                   "note": "K00 - K0 per position, MLP part only (flag 2); reported, not a stop"})


def share_states(lab: Any, root: Path, rows: Tensor) -> None:
    floor = get("share/floor")["K0"]
    for name in C.states:
        if not get(f"share/{name}"):
            dW = own(state(root, name))
            sp = lab.split(dW, nulls=False, halves=False)
            SR.read(lab, f"share/{name}", dW, sp["X"], rows, floor,
                    {"decomp": FROZEN, "state": name, "stored_error": sp["stored_error"]})
            del dW, sp


def curve_sizes() -> dict[str, float] | None:
    """low, mid, high from the scored states' K00 (widened by C.widen); None when no state is scored."""
    k = [r["means"]["K00"] for n in C.states if (r := get(f"share/{n}")) and r["status"] == "scored"]
    if not k:
        return None
    lo, hi = min(k) / C.widen, max(k) * C.widen
    return {"low": lo, "mid": math.sqrt(lo * hi), "high": hi}


def starter(draw: int) -> Any:
    """E186's first scale: an earlier draw's b at this size, else this draw's previous b x sqrt(size ratio), else b0."""
    def first(lv: str, prev: tuple[float, float] | None, ex: float) -> float:
        for d in reversed([d for d in C.share_draws if d < draw]):
            if rec := get(f"curve/d{d}_{lv}"):
                return rec["b"]
        return prev[0] * math.sqrt(ex / prev[1]) if prev else C.b0

    return first


def share_curve(lab: Any, rows: Tensor, shapes: dict[str, tuple[int, ...]]) -> None:
    T, fl = curve_sizes(), get("share/floor")
    if T is None:
        put("curve/skipped", {"why": "no state's share is scored: no curve is needed"})
        return
    for d in C.share_draws:
        names = {lv: f"curve/d{d}_{lv}" for lv in T if not get(f"curve/d{d}_{lv}")}
        if not names:
            continue
        full = SR.with_zeros(SR.gaussian(fl["norms"], shapes, d), shapes)
        sp = lab.split(full, nulls=False, halves=False)
        X, err = sp.pop("X"), sp["stored_error"]
        SR.read_at_sizes(lab, names, full, X, {lv: T[lv] - fl["K0"] for lv in names}, fl["K0"], rows,
                         starter(d), C.n_size, C.search_match, C.evals,
                         lambda lv, b, search, d=d, err=err: {"decomp": FROZEN, "draw": d, "level": lv, "b": b,
                                                             "target_k00": T[lv], "search": search,
                                                             "stored_error": err})
        del full, X, sp


# ----------------------------------------------------------------------------- the job


def lab_of(code: str, decomp: str) -> Any:
    lab = L.Lab(code, "unused", decomp)
    assert lab.alive is not None  # the live set's count (516,092) is checked here, before any reading
    return lab


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run(code: str, decomp: str, root: Path) -> None:
    rows = torch.tensor([r[: L.CFG.row_len] for r in json.loads(L.MATHS.read_text())["rows"][: C.n_rows]])
    ctrl = state(root, C.control)
    shapes = shapes_of(ctrl)
    rank_controls(shapes)
    lab = lab_of(code, decomp)
    share_controls(lab, rows, ctrl)
    del lab
    free()
    h = Held(rows)
    floor = s_controls(h, ctrl)
    del ctrl
    log("STAGE 0 DONE")
    fl = rank_floor(shapes)
    for name in C.states:
        if not get(f"rank/{name}"):
            rank_state(name, state(root, name), fl)
    log("STAGE 1 DONE")
    s_states(h, root, floor)
    log("STAGE 2 DONE")
    del h
    free()
    lab = lab_of(code, decomp)
    share_states(lab, root, rows)
    log("STAGE 3 DONE")
    share_curve(lab, rows, shapes)
    log(f"STAGE 4 DONE; {lab.peaks()}")


def main() -> None:
    global C
    ap = argparse.ArgumentParser(description="P3.13 run 1: share, S and effective rank of the E167 step-75 states")
    ap.add_argument("--decomp", help="p-015e0710's directory ($COMMUTE_QWEN_FROZEN_DIR)")
    ap.add_argument("--states", default=str(E167 / "large" / "run2"), help="<dir>/<name>/dW/step_75.safetensors")
    ap.add_argument("--summary", action="store_true", help="no GPU: results/summary.json from the written records")
    a = ap.parse_args()
    P = json.loads(PRED.read_text())
    C = config_from_settings(Config, P["settings"])
    if a.summary:
        load_experiment("qwen_held_readings", "summary").summarise(C, get)
        hard_exit(0)
    require_filed(PRED)
    if (RESULTS / "stopped.json").is_file():
        sys.exit(f"refusing to start: {RESULTS / 'stopped.json'} exists (a control missed; the lead decides)")
    assert a.decomp and decomp_id(Path(a.decomp)) == FROZEN, a.decomp
    assert site_name(0, "down") in QH.SITES
    fp32_matmuls()
    with torch.no_grad():
        run(P["experiment"], a.decomp, Path(a.states))
    hard_exit(0)


if __name__ == "__main__":
    main()

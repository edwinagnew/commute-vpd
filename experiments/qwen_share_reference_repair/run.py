"""The repair of cell P3.10's reference (E-number in results/predictions.json): E57's re-gating share on the
frozen-neuron decomposition p-015e0710 (subcomponents fixed at the MLP neurons, only Γ trained, 84 MLP matrices only),
read as E181 (experiments/qwen_share_reference) read it, with two changes so the filed rule can make a call:

  built changes, continuous size: the same three neuron orders as E181 (same pool and seeds); size x is continuous, the
        gate_proj rows of the first floor(x) neurons zeroed and the next one's row multiplied by 1 - frac(x)
        (`commute.share_reference.turned_down`). x is searched with `SR.size_search` (not whole, up to C.evals_built
        evaluations) to the row set's middle fine-tune size. All 15 are read again; none is kept from E181.
  random band, six draws: E181's draws 0-2 are read from its results as they stand; draws 3-5 are added at the same
        three sizes on all five row sets by the same procedure (`share_reference.random_draw`).

  stage 0, alone and first (a miss stops the job): W0 as the changed model on 16 rows of each row set this job reads
        (K00 = K01 = K0; under an all-zero split K10 = K11 = K0); the do-nothing state of random draw 0 at scale 100
        through the split path (share 1). Every read checks its mean K0 over 128 rows against P3.9's floors.json.
  stage 1: usage on all five row sets, then the 15 built changes (3 per row set).
  stage 2: random draws 3-5 per row set, each at three sizes.

Size is K00 (mean nats per position over 128 rows), searched on the first 32 rows as K00 - K0 within 1.05x, matched
when the 128-row K00 is within 1.1x of its target, as E181. One JSON per read under results/{control,built,random}/,
written and printed to the log as each finishes; a rerun skips what is written. `--summary` (no GPU) places each built
change and P3.9's 36 fine-tunes against the six-draw random curve at its own K00 (`commute.share_reading`). A
description: no reuse or new label. Plan: reports/plans/2026-11-16_plan_p310-reference-repair.md.

  python experiments/qwen_share_reference_repair/run.py --decomp "$COMMUTE_QWEN_FROZEN_DIR" [--rowsets copy,cipher]
  python experiments/qwen_share_reference_repair/run.py --summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import qwen_reading as qr
from commute import share_reading as SR
from commute import share_reference as RF
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment
from commute.qwen_d3 import ALIVE_TOTAL_OF, GSTATS_OF
from commute.qwen_vpd import N_LAYERS, decomp_id, site_name

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
E181 = EXPERIMENTS / "qwen_share_reference" / "results"  # its random/<rowset>_d{0,1,2}_<level>.json, read as they stand

# E57's Lab at this experiment's folder, p-015e0710's W0 g statistics and live-set count (its functions read these).
L = load_experiment("qwen_reading", "lab")
L.RESULTS, L.PRED, L.LARGE, L.GSTATS = RESULTS, PRED, RESULTS / "large", GSTATS_OF[RF.FROZEN]
L.CFG = replace(L.CFG, alive_total=ALIVE_TOTAL_OF[RF.FROZEN])


@dataclass(frozen=True)
class Config:
    rowsets: tuple[str, ...] = SR.ROWSETS  # the row sets this job reads (--rowsets splits the job over pods)
    old_draws: tuple[int, ...] = (0, 1, 2)  # E181's, read from E181
    new_draws: tuple[int, ...] = (3, 4, 5)  # this job's: Gaussian draw d, per matrix seed null_seed(d, site)
    n16: int = 16  # rows of the stage-0 controls
    n_size: int = 32  # rows of the size search (as E181)
    search_match: float = 1.05
    match: float = 1.1  # a read counts as matched when its 128-row K00 is within this factor of its target
    evals_built: int = 20  # E181's 8 ran out inside one giant neuron (cipher 'specific': bracket 278-288 after 8)
    evals_random: int = 8  # as E181
    pool: float = 0.1  # the used pool: per layer, this share of the live gate neurons, by mean |a| on the row set
    x0: float = 28.0  # the built search's first size: one neuron a layer
    b0: float = 100.0  # the do-nothing change's scale (E181)
    pick_seed: int = 0  # the 'random' choice's order (E181)

    def __post_init__(self) -> None:
        assert set(self.rowsets) <= set(SR.ROWSETS) and self.n16 <= self.n_size <= L.CFG.n_rows
        assert 1 < self.search_match < self.match and not set(self.old_draws) & set(self.new_draws)


C = Config()


def random_path(rowset: str, draw: int, level: str) -> Path:
    return (E181 if draw in C.old_draws else RESULTS) / "random" / f"{rowset}_d{draw}_{level}.json"


# ----------------------------------------------------------------------------- the stages


def controls(lab: Any, rows: dict[str, Tensor], shapes: dict[str, tuple[int, ...]]) -> None:
    for r in C.rowsets:
        if not lab.get(f"control/W0_{r}"):
            lab.put(f"control/W0_{r}", SR.w0_control(lab, r, rows[r][: C.n16]))
    if not lab.get("control/do_nothing"):
        r = C.rowsets[0]
        R = SR.gaussian(RF.norms(r), shapes, C.old_draws[0])
        for m in R.values():
            m.mul_(C.b0)
        lab.put("control/do_nothing", SR.do_nothing(lab, SR.with_zeros(R, shapes), rows[r][: C.n16]) | {"rowset": r})
    lab.log("STAGE 0 DONE")


def built(lab: Any, rowset: str, choice: str, order: list[tuple[int, int]], rows: Tensor,
          shapes: dict[str, tuple[int, ...]]) -> None:
    name, fl, target = f"built/{rowset}_{choice}", RF.floor(rowset), RF.sizes(rowset)["mid"]
    if lab.get(name):
        return
    W0 = lab.weights.sets["W0"]

    def change(x: float) -> dict[str, Tensor]:
        return RF.built_change(W0, RF.turned_down(order, x), shapes)

    search = SR.size_search(lambda x: SR.excess(lab, change(x), rows[: C.n_size]), target - fl, C.x0, 1,
                            C.search_match, C.evals_built, cap=len(order))
    lab.log(f"[{name} search] {json.dumps(search)}")
    x = search["x"]
    neurons = RF.turned_down(order, x)
    dW = change(x)
    sp = lab.split(dW, nulls=False, halves=False)
    SR.read(lab, name, dW, sp["X"], rows, fl, {
        "decomp": RF.FROZEN, "kind": "built", "rowset": rowset, "choice": choice, "x": x, "whole": math.floor(x),
        "partial_factor": neurons[-1][2] if neurons and neurons[-1][2] > 0 else None, "neurons": neurons,
        "target_k00": target, "search": search, "stored_error": sp["stored_error"]})


def start(rowset: str, draw: int) -> Any:
    """The random search's first scale at a size: the latest earlier draw's at this size (E181's draws included),
    else this draw's previous size's scaled by the square root of the size ratio, else b0."""
    def first(level: str, prev: tuple[float, float] | None, ex: float) -> float:
        for d in reversed([d for d in (*C.old_draws, *C.new_draws) if d < draw]):
            if rec := RF.stored(random_path(rowset, d, level)):
                return rec["b"]
        return prev[0] * math.sqrt(ex / prev[1]) if prev else C.b0

    return first


def run(lab: Any) -> None:
    rows = {r: SR.rows_of(r) for r in SR.ROWSETS}  # all five: the 'specific' order compares them
    assert all(tuple(t.shape) == (L.CFG.n_rows, L.CFG.row_len) for t in rows.values()), {r: t.shape for r, t in rows.items()}
    shapes = {s: tuple(w.shape) for s, w in lab.weights.sets["W0"].items()}
    missing = [random_path(r, d, lv) for r in C.rowsets for d in C.old_draws for lv in RF.LEVELS
               if not random_path(r, d, lv).is_file()]
    assert not missing, f"E181's random reads are not on disk: {missing[:3]}"
    controls(lab, rows, shapes)
    if any(not lab.get(f"built/{r}_{c}") for r in C.rowsets for c in RF.CHOICES):
        u = {r: RF.usage(lab, t, L.DEV) for r, t in rows.items()}
        live = [lab.alive.idx[site_name(lyr, "gate")].cpu() for lyr in range(N_LAYERS)]
        for r in C.rowsets:
            order = RF.orders(u, r, live, C.pool, C.pick_seed * 100 + SR.ROWSETS.index(r))
            for c in RF.CHOICES:
                built(lab, r, c, order[c], rows[r], shapes)
    lab.log("STAGE 1 DONE")
    for r in C.rowsets:
        for d in C.new_draws:
            RF.random_draw(lab, r, d, rows[r], shapes, start(r, d), C.n_size, C.search_match, C.evals_random)
    lab.log(f"STAGE 2 DONE rowsets {','.join(C.rowsets)}; {lab.peaks()}")


# ----------------------------------------------------------------------------- the description (no GPU)


def usable(rec: dict[str, Any] | None) -> bool:
    """Scored, and its 128-row K00 within C.match of its target."""
    return rec is not None and rec["status"] == "scored" and qr.within(rec["means"]["K00"], rec["target_k00"], C.match)


def rowset_summary(r: str) -> dict[str, Any]:
    """The six draws at three sizes, the curve through them and the band (`SR.curve`, `SR.band`)."""
    recs = {(d, lv): RF.stored(random_path(r, d, lv)) for d in (*C.old_draws, *C.new_draws) for lv in RF.LEVELS}
    draws = [{"draw": d, "level": lv, "k00": rec["means"]["K00"], "share": rec["regating"], "usable": usable(rec)}
             for (d, lv), rec in recs.items() if rec is not None]
    cv = SR.curve(draws)
    return {"sizes": RF.sizes(r), "floor": RF.floor(r), "draws": draws, "missing": sum(v is None for v in recs.values()),
            "curve": cv, "band": SR.band(cv, draws)}


def summarise() -> dict[str, Any]:
    out: dict[str, Any] = {"rowsets": {}, "built": [], "finetunes": []}
    for r in SR.ROWSETS:
        rs = out["rowsets"][r] = rowset_summary(r)
        cv, bd = rs["curve"], rs["band"]
        for c in RF.CHOICES:
            rec = RF.stored(RESULTS / "built" / f"{r}_{c}.json")
            k00, share = (rec["means"]["K00"], rec["regating"]) if rec else (None, None)
            scored = rec is not None and rec["status"] == "scored"
            # `described`: the same placement with a scored read off its size target kept (at its own K00); not the
            # filed rule, which reads an unmatched built change as unreadable
            out["built"].append({"rowset": r, "choice": c, "x": rec and rec["x"], "k00": k00, "share": share,
                                 "usable": usable(rec), "described": SR.place(share, scored, k00, cv, bd)}
                                | SR.place(share, usable(rec), k00, cv, bd))
        for ft in RF.finetunes(r):
            st = ft["state"]
            out["finetunes"].append({"rowset": r, "arm": st["arm"], "seed": st["seed"], "save": st["save"],
                                     "k00": ft["means"]["K00"], "share": ft["regating"]}
                                    | SR.place(ft["regating"], ft["status"] == "scored", ft["means"]["K00"], cv, bd))
    out["outcome"] = {"built": SR.outcome([b["class"] for b in out["built"]], 9),
                      "finetunes": SR.outcome([f["class"] for f in out["finetunes"]], 27)}
    out["counts"] = {k: {c: [x["class"] for x in out[k]].count(c) for c in ("above", "inside", "below", "unreadable")}
                     for k in ("built", "finetunes")}
    L.write_json(RESULTS / "summary.json", out)
    for r, rs in out["rowsets"].items():
        print(f"[summary] {r}: band {rs['band']}, curve {rs['curve']}, draws missing {rs['missing']}", flush=True)
    for b in out["built"]:
        print(f"[summary] built {b['rowset']} {b['choice']}: x {b['x']}, K00 {b['k00']}, share {b['share']}, "
              f"gap {b['gap']}, {b['class']}", flush=True)
    print(f"[summary] {json.dumps({'outcome': out['outcome'], 'counts': out['counts']})}", flush=True)
    return out


def main() -> None:
    global C
    ap = argparse.ArgumentParser(description="The repair of P3.10's reference on p-015e0710")
    ap.add_argument("--decomp", help="p-015e0710's directory ($COMMUTE_QWEN_FROZEN_DIR)")
    ap.add_argument("--rowsets", default=",".join(SR.ROWSETS), help="the row sets this job reads")
    ap.add_argument("--summary", action="store_true", help="no GPU: results/summary.json from the written reads")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"refusing to start: {PRED} missing")
    if a.summary:
        summarise()
        hard_exit(0)
    code = json.loads(PRED.read_text())["experiment"]
    if not code.startswith("E") or code == "E???":
        sys.exit(f"refusing to start: no experiment number in {PRED.name}")
    assert a.decomp and decomp_id(Path(a.decomp)) == RF.FROZEN, a.decomp
    C = replace(C, rowsets=tuple(a.rowsets.split(",")))
    lab = L.Lab(code, "unused", a.decomp)
    with torch.no_grad():
        assert lab.alive is not None  # the live set's count is checked here, before any reading
        run(lab)
    hard_exit(0)


if __name__ == "__main__":
    main()

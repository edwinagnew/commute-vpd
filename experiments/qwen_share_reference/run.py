"""Cell P3.10 (E-number in results/predictions.json): a reference for E57's re-gating share on the frozen-neuron
decomposition p-015e0710 (subcomponents fixed at the MLP neurons, only the masks' function Γ trained; the 84 MLP
matrices only). On each of the pair library's five row sets (128 rows of 256 tokens):

  stage 0, alone and first (a miss stops the job): W0 as the changed model on 16 rows of each row set (K00 = K01 = K0;
        under an all-zero split K10 = K11 = K0); the do-nothing state of a random change through the split path
        (share 1). Every later read checks its mean K0 over 128 rows against P3.9's floors.json.
  stage 1, built changes: a set of MLP neurons the model uses on the rows switched off: their gate_proj rows zeroed
        (dW = -W0 on those rows), so silu(0) = 0 and the neuron's output is 0 whatever up_proj gives. Three choices of
        set (`orders`); the count k is searched so the change's K00 lands at the middle of the row set's fine-tunes'.
  stage 2, random changes: iid Gaussian on the 84 MLP matrices at the per-matrix norms of the row set's seed-0 late
        fine-tune (P3.9 run 1), three draws, each scaled to three sizes: the lowest, middle and highest K00 of the row
        set's fine-tunes. The split is linear in the change, so one split per draw serves its three sizes.

Size is K00 (mean nats per position over the 128 rows): the changed model against the original subcomponents under
W0's masks. It is searched on the first 32 rows as K00 - K0 within 1.05x, and counted matched when the 128-row K00 is
within 1.1x of its target. One JSON per read under results/{control,built,random}/, written as each finishes; a rerun
skips what is written. `--summary` (no GPU) places each built change, and each of P3.9's 36 read fine-tunes, against
the random curve at its own K00 (`commute.share_reading`). A description: no reuse or new label.
Plan: reports/plans/2026-11-13_plan_p310-share-reference.md. Rules: results/predictions.json (no start without it).

  python experiments/qwen_share_reference/run.py --decomp "$COMMUTE_QWEN_FROZEN_DIR" [--rowsets copy,cipher]
  python experiments/qwen_share_reference/run.py --summary
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
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment
from commute.qwen_d3 import ALIVE_TOTAL_OF, GSTATS_OF
from commute.qwen_vpd import N_LAYERS, decomp_id, site_name

FROZEN = "p-015e0710"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
P39 = EXPERIMENTS / "qwen_library_share" / "results" / FROZEN  # P3.9: run1 (random norms), run2 (fine-tunes), floors
ARMS_OF = {"copy": ("copy",), "cipher": ("cipher",), "known": ("known",), "invented": ("invented",),
           "gold": ("gold", "gold_shuf")}  # the library read both gold arms on gold's rows
SAVES = ("5x", "last")

# E57's Lab at this experiment's folder, p-015e0710's W0 g statistics and live-set count (its functions read these).
L = load_experiment("qwen_reading", "lab")
L.RESULTS, L.PRED, L.LARGE, L.GSTATS = RESULTS, PRED, RESULTS / "large", GSTATS_OF[FROZEN]
L.CFG = replace(L.CFG, alive_total=ALIVE_TOTAL_OF[FROZEN])


@dataclass(frozen=True)
class Config:
    rowsets: tuple[str, ...] = SR.ROWSETS  # the row sets this job reads (--rowsets splits the job over pods)
    choices: tuple[str, ...] = ("top", "random", "specific")
    levels: tuple[str, ...] = ("low", "mid", "high")
    draws: tuple[int, ...] = (0, 1, 2)  # Gaussian draw d, per matrix seed null_seed(d, site)
    seeds: tuple[int, ...] = (0, 1, 2)  # P3.9's fine-tune seeds
    n16: int = 16  # rows of the stage-0 controls
    n_size: int = 32  # rows of the size search (32-row K00 is within 4.2% of the 128-row one in P3.9's records)
    search_match: float = 1.05
    match: float = 1.1  # a read counts as matched when its 128-row K00 is within this factor of its target
    evals: int = 8
    pool: float = 0.1  # the used pool: per layer, this share of the live gate neurons, by mean |a| on the row set
    k0: int = 28  # the built search's first count: one neuron a layer
    b0: float = 100.0  # the random search's first scale, and the do-nothing change's (P3.9 run 1 found 44-640)
    pick_seed: int = 0  # the 'random' choice's order

    def __post_init__(self) -> None:
        assert set(self.rowsets) <= set(SR.ROWSETS) and self.n16 <= self.n_size <= L.CFG.n_rows
        assert 1 < self.search_match < self.match and self.levels == ("low", "mid", "high")


C = Config()


# ----------------------------------------------------------------------------- P3.9's files: sizes, floors, norms


def stored(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def finetunes(rowset: str) -> list[dict[str, Any]]:
    """P3.9's read fine-tunes on this row set (run2: arms x seeds 0-2 x saves 5x and last)."""
    recs = [json.loads((P39 / "run2" / f"{a}_s{s}_{v}.json").read_text())
            for a in ARMS_OF[rowset] for s in C.seeds for v in SAVES]
    assert all(r["state"]["arm"] in ARMS_OF[rowset] for r in recs)
    return recs


def sizes(rowset: str) -> dict[str, float]:
    """The three target K00s: the row set's fine-tunes' lowest, their geometric middle and their highest."""
    k = [r["means"]["K00"] for r in finetunes(rowset)]
    return {"low": min(k), "mid": math.sqrt(min(k) * max(k)), "high": max(k)}


def floor(rowset: str) -> float:
    return json.loads((P39 / "floors.json").read_text())["floors"][rowset]["K0"]


def norms(rowset: str) -> dict[str, float]:
    """The per-matrix norms (84 MLP matrices) of the row set's seed-0 late fine-tune, as P3.9 run 1 recorded them."""
    return json.loads((P39 / "run1" / f"{rowset}_s0_last_kl.json").read_text())["random"]["norms"]


# ----------------------------------------------------------------------------- the built change's neurons


def usage(lab: Any, rows: Tensor) -> Tensor:
    """Mean |a| per MLP neuron [28, 6144] at W0 over rows, positions 1-255 (position 0, where a row's attention-sink
    activations sit, left out), a = down_proj's input = silu(gate) * up."""
    acc = torch.zeros(N_LAYERS, lab.model.config.intermediate_size, dtype=torch.float64, device=L.DEV)

    def hook(layer: int) -> Any:
        def add(_m: Any, a: tuple[Tensor, ...], _o: Tensor) -> None:  # returns None: the module's output is kept
            acc[layer] += a[0][0, 1:].abs().double().sum(0)

        return add

    lab.use("W0")
    hs = [lyr.mlp.down_proj.register_forward_hook(hook(i)) for i, lyr in enumerate(lab.model.model.layers)]
    try:
        for r in rows:
            lab.model(r[None].to(L.DEV), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return (acc / (rows.shape[0] * (rows.shape[1] - 1))).cpu()


def interleave(per_layer: list[list[int]]) -> list[tuple[int, int]]:
    """Every layer's first neuron, then every layer's second, ...: a prefix of length k holds about k/28 per layer."""
    depth = max((len(p) for p in per_layer), default=0)
    return [(lyr, p[i]) for i in range(depth) for lyr, p in enumerate(per_layer) if i < len(p)]


def orders(u: dict[str, Tensor], rowset: str, live: list[Tensor], pool: float, seed: int) -> dict[str, list[tuple[int, int]]]:
    """Three orders of the neurons the model uses on `rowset`, each interleaved over layers. The used pool, per layer:
    the top `pool` share of the live gate neurons by mean |a| on the row set (`usage`). top: the pool by mean |a|,
    highest first; random: the pool in a seeded random order; specific: the pool by mean |a| on this row set over
    its mean on the other four, highest first (the neurons this text uses more than the library's other text)."""
    others = torch.stack([v for r, v in u.items() if r != rowset]).mean(0)
    per: dict[str, list[list[int]]] = {c: [] for c in C.choices}
    for lyr, idx in enumerate(live):
        top = idx[u[rowset][lyr, idx].argsort(descending=True, stable=True)[: math.ceil(pool * len(idx))]]
        per["top"].append(top.tolist())
        gen = torch.Generator().manual_seed(seed * 1000 + lyr)
        per["random"].append(top[torch.randperm(len(top), generator=gen)].tolist())
        ratio = u[rowset][lyr, top] / others[lyr, top].clamp_min(1e-12)
        per["specific"].append(top[ratio.argsort(descending=True, stable=True)].tolist())
    return {c: interleave(v) for c, v in per.items()}


def switched_off(lab: Any, neurons: list[tuple[int, int]], shapes: dict[str, tuple[int, ...]]) -> dict[str, Tensor]:
    """dW = -W0 on the gate_proj rows of `neurons` (layer, index), zero everywhere else."""
    by: dict[int, list[int]] = {}
    for lyr, c in neurons:
        by.setdefault(lyr, []).append(c)
    W0, part = lab.weights.sets["W0"], {}
    for lyr, cs in by.items():
        s, idx = site_name(lyr, "gate"), torch.tensor(cs)
        part[s] = torch.zeros(W0[s].shape)
        part[s][idx] = -W0[s][idx].float()
    return SR.with_zeros(part, shapes)


# ----------------------------------------------------------------------------- the stages


def controls(lab: Any, rows: dict[str, Tensor], shapes: dict[str, tuple[int, ...]]) -> None:
    for r in C.rowsets:
        if not lab.get(f"control/W0_{r}"):
            lab.put(f"control/W0_{r}", SR.w0_control(lab, r, rows[r][: C.n16]))
    if not lab.get("control/do_nothing"):
        r = C.rowsets[0]
        R = SR.gaussian(norms(r), shapes, C.draws[0])
        for m in R.values():
            m.mul_(C.b0)
        lab.put("control/do_nothing", SR.do_nothing(lab, SR.with_zeros(R, shapes), rows[r][: C.n16]) | {"rowset": r})
    lab.log("STAGE 0 DONE")


def built(lab: Any, rowset: str, choice: str, order: list[tuple[int, int]], rows: Tensor,
          shapes: dict[str, tuple[int, ...]]) -> None:
    name, fl, target = f"built/{rowset}_{choice}", floor(rowset), sizes(rowset)["mid"]
    if lab.get(name):
        return
    search = SR.size_search(lambda k: SR.excess(lab, switched_off(lab, order[: int(k)], shapes), rows[: C.n_size]),
                            target - fl, C.k0, 1, C.search_match, C.evals, cap=len(order), whole=True)
    lab.log(f"[{name} search] {json.dumps(search)}")
    k = int(search["x"])
    dW = switched_off(lab, order[:k], shapes)
    sp = lab.split(dW, nulls=False, halves=False)
    SR.read(lab, name, dW, sp["X"], rows, fl, {
        "decomp": FROZEN, "kind": "built", "rowset": rowset, "choice": choice, "k": k, "neurons": order[:k],
        "target_k00": target, "search": search, "stored_error": sp["stored_error"]})


def start(lab: Any, rowset: str, draw: int, level: str, prev: tuple[float, float] | None, ex: float) -> float:
    """The random search's first scale: an earlier draw's at this size, else this draw's previous size's scaled by
    the square root of the size ratio, else b0."""
    for d in reversed([d for d in C.draws if d < draw]):
        if rec := lab.get(f"random/{rowset}_d{d}_{level}"):
            return rec["b"]
    return prev[0] * math.sqrt(ex / prev[1]) if prev else C.b0


def random_draw(lab: Any, rowset: str, draw: int, rows: Tensor, shapes: dict[str, tuple[int, ...]]) -> None:
    names = {lv: f"random/{rowset}_d{draw}_{lv}" for lv in C.levels}
    todo = [lv for lv in C.levels if not lab.get(names[lv])]
    if not todo:
        return
    fl, T = floor(rowset), sizes(rowset)
    R = SR.gaussian(norms(rowset), shapes, draw)
    full = SR.with_zeros(R, shapes)
    sp = lab.split(full, nulls=False, halves=False)
    cur, prev = 1.0, None  # R and its split X are held at scale `cur`, rescaled in place for each size
    for lv in todo:
        ex = T[lv] - fl
        search = SR.size_search(lambda b, c=cur: SR.excess(lab, full, rows[: C.n_size], b / c), ex,
                                start(lab, rowset, draw, lv, prev, ex), 2, C.search_match, C.evals)
        lab.log(f"[{names[lv]} search] {json.dumps(search)}")
        b = search["x"]
        for m in [*R.values(), *sp["X"].values()]:
            m.mul_(b / cur)
        cur, prev = b, (b, ex)
        SR.read(lab, names[lv], full, sp["X"], rows, fl, {
            "decomp": FROZEN, "kind": "random", "rowset": rowset, "draw": draw, "level": lv, "b": b,
            "norms_from": f"{rowset}_s0_last", "target_k00": T[lv], "search": search, "stored_error": sp["stored_error"]})


def run(lab: Any) -> None:
    rows = {r: SR.rows_of(r) for r in SR.ROWSETS}  # all five: the 'specific' order compares them
    assert all(tuple(t.shape) == (L.CFG.n_rows, L.CFG.row_len) for t in rows.values()), {r: t.shape for r, t in rows.items()}
    shapes = {s: tuple(w.shape) for s, w in lab.weights.sets["W0"].items()}
    controls(lab, rows, shapes)
    if any(not lab.get(f"built/{r}_{c}") for r in C.rowsets for c in C.choices):
        u = {r: usage(lab, t) for r, t in rows.items()}
        live = [lab.alive.idx[site_name(lyr, "gate")].cpu() for lyr in range(N_LAYERS)]
        for r in C.rowsets:
            order = orders(u, r, live, C.pool, C.pick_seed * 100 + SR.ROWSETS.index(r))
            for c in C.choices:
                built(lab, r, c, order[c], rows[r], shapes)
    lab.log("STAGE 1 DONE")
    for r in C.rowsets:
        for d in C.draws:
            random_draw(lab, r, d, rows[r], shapes)
    lab.log(f"STAGE 2 DONE rowsets {','.join(C.rowsets)}; {lab.peaks()}")


# ----------------------------------------------------------------------------- the description (no GPU)


def usable(rec: dict[str, Any] | None) -> bool:
    """Scored, and its 128-row K00 within C.match of its target."""
    return rec is not None and rec["status"] == "scored" and qr.within(rec["means"]["K00"], rec["target_k00"], C.match)


def summarise() -> dict[str, Any]:
    out: dict[str, Any] = {"rowsets": {}, "built": [], "finetunes": []}
    for r in SR.ROWSETS:
        recs = {(d, lv): stored(RESULTS / "random" / f"{r}_d{d}_{lv}.json") for d in C.draws for lv in C.levels}
        draws = [{"draw": d, "level": lv, "k00": rec["means"]["K00"], "share": rec["regating"], "usable": usable(rec)}
                 for (d, lv), rec in recs.items() if rec is not None]
        cv = SR.curve(draws)
        bd = SR.band(cv, draws)
        out["rowsets"][r] = {"sizes": sizes(r), "floor": floor(r), "draws": draws, "curve": cv, "band": bd}
        for c in C.choices:
            rec = stored(RESULTS / "built" / f"{r}_{c}.json")
            k00, share = (rec["means"]["K00"], rec["regating"]) if rec else (None, None)
            out["built"].append({"rowset": r, "choice": c, "k": rec and rec["k"], "k00": k00, "share": share,
                                 "usable": usable(rec)} | SR.place(share, usable(rec), k00, cv, bd))
        for ft in finetunes(r):
            st = ft["state"]
            out["finetunes"].append({"rowset": r, "arm": st["arm"], "seed": st["seed"], "save": st["save"],
                                     "k00": ft["means"]["K00"], "share": ft["regating"]}
                                    | SR.place(ft["regating"], ft["status"] == "scored", ft["means"]["K00"], cv, bd))
    out["outcome"] = {"built": SR.outcome([b["class"] for b in out["built"]], 9),
                      "finetunes": SR.outcome([f["class"] for f in out["finetunes"]], 27)}
    out["counts"] = {k: {c: [x["class"] for x in out[k]].count(c) for c in ("above", "inside", "below", "unreadable")}
                     for k in ("built", "finetunes")}
    L.write_json(RESULTS / "summary.json", out)
    print(f"[summary] {json.dumps({'outcome': out['outcome'], 'counts': out['counts']})}", flush=True)
    return out


def main() -> None:
    global C
    ap = argparse.ArgumentParser(description="P3.10: the re-gating share's reference on p-015e0710")
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
    assert a.decomp and decomp_id(Path(a.decomp)) == FROZEN, a.decomp
    C = replace(C, rowsets=tuple(a.rowsets.split(",")))
    lab = L.Lab(code, "unused", a.decomp)
    with torch.no_grad():
        assert lab.alive is not None  # the live set's count is checked here, before any reading
        run(lab)
    hard_exit(0)


if __name__ == "__main__":
    main()

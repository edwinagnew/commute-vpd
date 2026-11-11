"""E152 step 1 (cell P3.3): changes whose placement we know, read with the existing Qwen pipeline on the new
decomposition p-9707d6ff. Plan: reports/plans/2026-11-11_plan_qwen-instrument-check-step1.md; rules in
results/predictions_step1.json (the run refuses to start unless it is committed unchanged).

    sbatch --comment='E152 stage: step 1' cluster/qwen_instrument_check_step1.sbatch

Three kinds of change to W0 (`planted.py`): often (rescale the write vectors of the subcomponents on at most maths
positions, at all 28 down_proj matrices), rarely (as many that are almost never on there), random (E132's Gaussian,
`split.random_like` of E40's δW at seed 0). Each at three sizes, its scale b found by E57's search
(`qwen_reading.scale_search`) so that KL(W0 ‖ state) on the 128 maths rows of 256 matches RL s0 (0.0643), the
supervised fine-tune (E40 step 10, 1.635), and their geometric midpoint.

Per state, E132's pipeline (E57's Lab, the calls of E132 run.py `read`): the four cells on the 128 maths rows, the
least-squares split, the cells of the split. The split is linear in the change, so it runs once per kind at b = 1 and
is scaled by b for each size (one split instead of three; the shares it gives cannot depend on size). Readings: the
split's Σ-weighted energy share on the planted set, its share on the subcomponents gated on at W0 (E94's placement, on
set counted here), and E57's re-gating share.

Order: controls (W0 against E132's stored numbers, the zero split, E40's KL, do-nothing; the planted labels against
the on set), then often (its split is the positive control and stops the job below the bar), rarely, random. Each
state writes results/step1/S_<kind>_<size>.json as it ends; a resubmitted job skips what is written (exit 3: more to
do). Every number goes to results/step1/run.log.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from commute import qwen_reading as qr
from commute import split
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_d3 import ALIVE_TOTAL_OF, GSTATS_OF, null_pe, null_seed, pinned
from commute.qwen_sft import load
from commute.qwen_vpd import SITES, decomp_id

NEW = "p-9707d6ff"
HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results" / "step1", HERE / "results" / "predictions_step1.json"
LARGE = HERE / "results" / "large" / "step1"  # piece energies and the on set, for a resume (2 MB each)
E132 = HERE.parent / "qwen_regating_new" / "results"
PL = load_experiment("qwen_instrument_check", "planted")
L = load_experiment("qwen_reading", "lab")  # E57's Lab, pointed here and at the new decomposition, as E132 did
L.RESULTS, L.PRED, L.GSTATS = RESULTS, PRED, GSTATS_OF[NEW]
L.CFG = replace(L.CFG, alive_total=ALIVE_TOTAL_OF[NEW], limit_s=2 * 3600)
CFG = L.CFG
DEV, F64 = "cuda", torch.float64
KINDS = ("often", "rarely", "random")


@dataclass(frozen=True)
class Step:
    rl: float = 0.06431339845656138  # RL s0 on these rows (qwen_d1_table/results/R_s0/state.json clean.W0.K)
    sft: float = 1.6351103354567915  # E40 step 10 on these rows (qwen_d1_table/results/control.json clean.W0)
    match: float = 1.25  # the sizes are 5.04x apart: matched states stay at least 3.2x apart
    evals: int = 8
    chunk: int = 8  # rows per batched forward in the search (Γ parked)
    margin_s: float = 300.0

    def __post_init__(self) -> None:
        assert 0 < self.rl < self.sft and self.match**2 < math.sqrt(self.sft / self.rl) and CFG.n_rows % self.chunk == 0

    def targets(self) -> dict[str, float]:
        return {"rl": self.rl, "mid": math.sqrt(self.rl * self.sft), "sft": self.sft}


ST = Step()
FILED = json.loads(PRED.read_text())
RULES = FILED["rules"]


def total(rows: dict[str, list[float]], n: int | None = None) -> dict[str, float]:
    return {k: sum(v[:n]) for k, v in rows.items()}


def put(lab: Any, name: str, d: dict[str, Any]) -> None:
    lab.put(name, d)
    lab.log(f"STAGE {name} DONE")


def rel(x: float, ref: float) -> float:
    return abs(x / ref - 1)


def masks(lab: Any, sets: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
    """Each planted set as a flat mask over the alive subcomponents (`Layout` order)."""
    out = {}
    for kind, by_site in sets.items():
        m = torch.zeros(lab.lay.n, dtype=torch.bool)
        for s, pos in by_site.items():
            m[lab.lay.offset[s] + pos] = True
        out[kind] = m
    return out


def controls(lab: Any, rows: Tensor, r16: Tensor, mk: dict[str, Tensor]) -> dict[str, Any]:
    """W0 as the fine-tuned model (with W0's gate counts: the on set), the zero split, E40's KL on 16 rows, E40's
    do-nothing; each against E132's stored numbers. Then the planted labels against the on set. A miss stops."""
    ref, tol = json.loads((E132 / "controls.json").read_text()), RULES["w0_tol"]
    c0 = lab.counts()
    lab.set_weights("FT", None)
    w0 = lab.cells(rows, c0=c0, tag="W0 as FT")
    m = {k: v / rows.numel() for k, v in total(w0).items()}
    k0_16 = total(w0, CFG.n16)["K0"]
    zero = total(lab.cells(r16, lab.change(None, "zero"), tag="W0, zero split, 16 rows"))
    lab.check("W0", all(rel(m[k], ref["means_W0"][k]) <= tol for k in m) and all(
        rel(zero[k], k0_16) <= CFG.id_tol for k in ("K10", "K11")), {"means": m, "E132": ref["means_W0"], "zero": zero})
    e40 = lab.load_dw("e40")
    lab.set_weights("FT", e40)
    kl16 = lab.kl_rows("FT", r16)
    lab.check("E40_KL", rel(kl16, ref["kl_e40_16"]) <= tol, {"kl16": kl16, "E132": ref["kl_e40_16"]})
    held = total(lab.cells(r16, tag="E40, 16 rows")) | total(lab.cells(r16, lab.change(e40, None), tag="do-nothing"))
    dn = qr.shares(held, r16.numel())["regating"]
    lab.check("do_nothing", qr.ok(dn) and abs(dn - 1) <= CFG.id_tol, {"share": dn} | held)
    lab.drop_set("FT")
    on0 = c0.on()
    LARGE.mkdir(parents=True, exist_ok=True)
    save_file({"on0": on0}, str(LARGE / "on0.safetensors"))
    lab_on = {k: float(on0[v].double().mean()) for k, v in mk.items()}
    lab.check("labels", lab_on["often"] >= RULES["label_often_min"] and lab_on["rarely"] <= RULES["label_rarely_max"],
              {"share_on": lab_on, "n": {k: int(v.sum()) for k, v in mk.items()}})
    return {"means_W0": m, "kl_e40_16": kl16, "do_nothing_share": dn, "on0": int(on0.sum()), "labels_on": lab_on}


def change_of(lab: Any, kind: str, sets: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
    """The kind's change at b = 1 (fp32 host). Planted: b U_P V_Pᵀ at the down_proj sites, a shared zero elsewhere.
    Random: E132's Gaussian, `split.random_like(E40's δW_s, Σ_s, null_seed(0, s))` (E40's Σ-weighted energy per site)."""
    if kind == "random":
        e40, sigma = lab.load_dw("e40"), lab.sigma()
        return {s: pinned(split.random_like(e40[s].to(DEV, F64), sigma[s].to(DEV), null_seed(0, s)).float().cpu())
                for s in SITES}
    zeros = {w.shape: torch.zeros(w.shape) for w in lab.weights.sets["W0"].values()}
    out = {s: zeros[w.shape] for s, w in lab.weights.sets["W0"].items()}
    for s, pos in sets[kind].items():
        U_A, V_A = lab.alive.uv(s, DEV, F64)
        p = pos.to(DEV)
        out[s] = pinned((U_A[:, p] @ V_A[:, p].T).float().cpu())
    return out


def kl_batched(lab: Any, key: str, rows: Tensor) -> float:
    """Mean KL(W0 ‖ weight set `key`) per position, unmasked, `ST.chunk` rows per forward (Lab.kl_rows batched)."""
    lab.park_gamma()
    tot = 0.0
    for c in rows.split(ST.chunk):
        t = c.to(DEV)
        lab.use("W0")
        p = lab.model(t, use_cache=False).logits
        lab.use(key)
        tot += L.kl(p, lab.model(t, use_cache=False).logits)
        del p
    lab.use("W0")
    return tot / rows.numel()


def search(lab: Any, kind: str, dW: dict[str, Tensor], rows: Tensor) -> dict[str, Any]:
    """The scale b of each size (E57's search, `qr.scale_search`), on the 128 maths rows."""
    out = {}
    for size, target in ST.targets().items():

        def kl_at(b: float, size: str = size, target: float = target) -> float:
            lab.set_weights("state", dW, b)
            x = kl_batched(lab, "state", rows)
            lab.log(f"[{kind} {size} search] b {b:.6g}: KL {x:.9g}, target {target:.9g}")
            return x

        out[size] = qr.scale_search(kl_at, target, ST.match, ST.evals)
        lab.drop_set("state")
    return out


def energies(lab: Any, dW: dict[str, Tensor]) -> dict[str, float]:
    sig = lab.sigma()
    return {s: split.weighted_energy(m.to(DEV, F64), sig[s].to(DEV)) if m.any() else 0.0 for s, m in dW.items()}


def split_kind(lab: Any, kind: str, dW: dict[str, Tensor], mk: dict[str, Tensor], on0: Tensor) -> dict[str, Tensor]:
    """The least-squares split at b = 1 (Lab.split, as E132), its piece energies saved, its shares written; the
    often kind's planted share is the positive control."""
    sp = lab.split(dW, nulls=False, halves=False)
    pe = sp["pe"]["ls"]
    save_file({"pe": pe.contiguous()}, str(LARGE / f"pe_{kind}.safetensors"))
    rec = {"stored_error": sp["stored_error"], "split_s": sp["s"], "energy": energies(lab, dW),
           "planted": {k: qr.placement(pe, m) for k, m in mk.items()}, "on": qr.placement(pe, on0)}
    put(lab, f"split_{kind}", rec)
    if kind == "often":
        P = rec["planted"]["often"]
        lab.check("positive", qr.ok(P) and round(P, 9) >= RULES["positive_min"], {"planted_share": P})
    return sp["X"]


def read(lab: Any, name: str, dW: dict[str, Tensor], X: dict[str, Tensor], rows: Tensor, extra: dict[str, Any]) -> None:
    """E132's read of one state (cells, then the cells of the split), with the split given."""
    lab.set_weights("FT", dW)
    cells = lab.cells(rows, tag=name)
    cells |= lab.cells(rows, lab.change(dW, X), tag=f"{name}, split")
    lab.drop_set("FT")
    put(lab, f"S_{name}", {"decomp": NEW, "rows": cells, "means": {k: sum(v) / rows.numel() for k, v in cells.items()}}
            | qr.whole(cells, CFG.row_len) | extra)


def rescale(d: dict[str, Tensor], factor: float) -> None:
    for t in {id(t): t for t in d.values()}.values():  # shared zeros once
        t.mul_(factor)


def run_kind(lab: Any, kind: str, sets: dict[str, dict[str, Tensor]], mk: dict[str, Tensor], rows: Tensor,
             longest: list[float]) -> None:
    order = ("sft", "rl", "mid") if kind == "often" else PL.SIZES
    if all(lab.get(f"S_{kind}_{z}") for z in order):
        return
    dW = change_of(lab, kind, sets)
    if not lab.get(f"search_{kind}"):
        put(lab, f"search_{kind}", search(lab, kind, dW, rows))
    found = lab.get(f"search_{kind}")
    cut(lab, longest, 2)
    X = split_kind(lab, kind, dW, mk, load_file(str(LARGE / "on0.safetensors"))["on0"])
    b_now = 1.0
    for z in order:
        if lab.get(f"S_{kind}_{z}"):
            continue
        cut(lab, longest, 1)
        t0, b = time.time(), found[z]["b"]
        rescale(dW, b / b_now)
        rescale(X, b / b_now)
        b_now = b
        read(lab, f"{kind}_{z}", dW, X, rows, {"kind": kind, "size": z, "b": b, "search": found[z],
                                                "matched": found[z]["matched"]})
        longest.append(time.time() - t0)
        lab.log(f"[{kind}_{z}] read in {longest[-1]:.0f} s, {lab.peaks()}")


def cut(lab: Any, longest: list[float], states: int) -> None:
    """Exit 3 (resubmit to resume) when the time left cannot hold `states` more reads of the longest so far."""
    used, need = time.time() - lab.t0, states * max(longest, default=700.0) + ST.margin_s
    if CFG.limit_s - used < need:
        lab.log(f"cut short: {used:.0f} s used, {need:.0f} s needed; resubmit to resume")
        hard_exit(3)


def boots(rec: dict[str, Any], draws: list[list[int]]) -> list[float]:
    return [qr.shares(qr.sums_at(rec["rows"], i), len(i) * CFG.row_len)["regating"] for i in draws]


def pair(ra: dict[str, Any], rb: dict[str, Any], ba: list[float], bb: list[float], spread: float) -> dict[str, Any]:
    """E132's separation rule on the re-gating shares of two states, with their row-resampled shares (same draws)."""
    gap, g95 = ra["regating"] - rb["regating"], qr.interval([x - y for x, y in zip(ba, bb, strict=True)])
    statuses, matched = (ra["status"], rb["status"]), bool(ra["matched"] and rb["matched"])
    sep = PL.separated(gap, g95, spread, matched and statuses == ("scored", "scored"))
    return {"gap": gap, "gap95": g95, "separated": sep, "seen": PL.seen(sep, matched, statuses)}


def summarise(lab: Any, mk: dict[str, Tensor]) -> dict[str, Any]:
    """Chance from the random split rescaled per matrix to each planted change (E94's `null_pe`), the readings, the
    filed calls."""
    on0 = load_file(str(LARGE / "on0.safetensors"))["on0"]
    pe = {k: load_file(str(LARGE / f"pe_{k}.safetensors"))["pe"] for k in KINDS}
    sp = {k: lab.get(f"split_{k}") for k in KINDS}
    read_k = {}
    for k in ("often", "rarely"):
        nul = null_pe(pe["random"], sp["random"]["energy"], sp[k]["energy"], lab.lay.site_of)
        P, C = sp[k]["planted"][k], qr.placement(nul, mk[k])
        read_k[k] = {"planted": P, "chance": C, "set_share": int(mk[k].sum()) / int(lab.lay.kind_mask("down").sum()),
                     "recovery": PL.recovery(P, C, int(mk[k].sum())), "on": sp[k]["on"], "on_null": qr.placement(nul, on0)}
    names = [f"{k}_{z}" for k in KINDS for z in PL.SIZES]
    spread = json.loads((E132 / "summary.json").read_text())["new"]["seed_spread"]
    draws = qr.resample(CFG.n_rows, CFG.boot)
    recs = {n: lab.get(f"S_{n}") for n in names}
    bs = {n: boots(r, draws) for n, r in recs.items()}
    pairs = [{"a": a, "b": b} | pair(recs[a], recs[b], bs[a], bs[b], spread) for a, b in combinations(names, 2)]
    by = {(p["a"], p["b"]): p["seen"] for p in pairs}
    apart = {z: by[(f"often_{z}", f"rarely_{z}")] for z in PL.SIZES}
    calls = {"split": PL.split_call(read_k["often"]["recovery"], read_k["rarely"]["recovery"], read_k["often"]["on"],
                                    read_k["rarely"]["on"], RULES),
             "size": PL.size_call(apart), "regating": PL.blind_call([p["separated"] for p in pairs])}
    states = {n: {k: r[k] for k in ("regating", "status", "den_mean", "matched", "b")} for n, r in recs.items()}
    return {"decomp": NEW, "kinds": read_k, "random_on": sp["random"]["on"], "states": states, "seed_spread": spread,
            "pairs": pairs, "calls": calls}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--e40", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    a = ap.parse_args()
    require_filed(PRED)
    assert decomp_id(Path(a.decomp)) == NEW, a.decomp
    assert all(math.isclose(ST.targets()[k], v, rel_tol=1e-8) for k, v in FILED["settings"]["targets"].items())
    RESULTS.mkdir(parents=True, exist_ok=True)
    lab = L.Lab("E152 step 1", a.e40, a.decomp)
    with torch.no_grad():
        idx = {s: v.cpu() for s, v in lab.alive.idx.items()}
        sets = PL.planted_sets(load(L.GSTATS / "W0_maths.safetensors"), idx, list(SITES))
        n = {k: sum(len(v) for v in by.values()) for k, by in sets.items()}
        assert all(len(v) > 0 for k in sets for v in sets[k].values()) and n == FILED["settings"]["set_sizes"], n
        mk = masks(lab, sets)
        rows, r16 = lab.maths(), lab.maths(CFG.n16)
        if not lab.get("controls"):
            put(lab, "controls", controls(lab, rows, r16, mk))
        longest: list[float] = []
        for kind in KINDS:
            run_kind(lab, kind, sets, mk, rows, longest)
        put(lab, "summary", summarise(lab, mk))
    lab.log(f"[E152 step 1 done] {lab.peaks()}")
    hard_exit(0)


if __name__ == "__main__":
    main()

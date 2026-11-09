"""E132 (cell P3.1): E57's re-gating share read on the new decomposition (p-9707d6ff) for the four saved Qwen3-1.7B
updates E57 read, beside E57's reading on the old one. Plan: reports/plans/2026-11-09_plan_e132-regating-new.md; rules
in results/predictions.json (the run refuses to start without it, or while results/stopped.json exists).

One job on one L40, in order; each step writes its own results file and every number goes to results/run.log:
  controls   W0 as the fine-tuned model (the four cells equal K0), the unmasked KL of E40 step 10 on 16 rows against
             E57's (no decomposition enters it), and the do-nothing state through the split path (share exactly 1).
             A miss stops the job. Written to controls.json.
  gauss, inspan   the two random changes, each scaled to E40's KL on 16 rows. Written to S_<name>.json.
  e40, a2long_s0, a2long_s1, gold_s0   the updates, the seed pair ahead of gold. Written to S_<name>.json.
  summary    the shares, new beside old, and the filed separation rule. Written to summary.json.
A re-submitted job skips what is written. The share is E57's: (K00 - K01) / (K00 - K11), summed over 128 maths rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import qwen_reading as qr
from commute import split
from commute.jobs import load_experiment
from commute.qwen_d3 import ALIVE_TOTAL_OF, GSTATS_OF, alive_pieces, null_seed, pinned
from commute.qwen_vpd import SITES, decomp_id

NEW = "p-9707d6ff"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
E57 = HERE.parent / "qwen_reading/results"
RANDOM = ("gauss", "inspan")
UPDATES = ("e40", "a2long_s0", "a2long_s1", "gold_s0")  # the states E57 read; the seed pair is what the rule needs

# E57's Lab, pointed at the new decomposition's files. Its functions read these names when they are called.
L = load_experiment("qwen_reading", "lab")
L.RESULTS, L.PRED, L.LARGE, L.GSTATS = RESULTS, RESULTS / "predictions.json", RESULTS / "large", GSTATS_OF[NEW]
L.CFG = replace(L.CFG, alive_total=ALIVE_TOTAL_OF[NEW])
CFG = L.CFG


def total(rows: dict[str, list[float]], n: int | None = None) -> dict[str, float]:
    return {k: sum(v[:n]) for k, v in rows.items()}


def controls(lab: Any, rows: Tensor, r16: Tensor) -> dict[str, Any]:
    """W0 as the fine-tuned model, the decomposition-free KL of E40, the do-nothing state. Each a stop."""
    lab.set_weights("FT", None)
    w0 = total(lab.cells(rows, tag="W0 as FT"))
    k0_16 = total(lab.cells(rows[: CFG.n16], tag="W0 as FT, 16 rows"))["K0"]
    zero = total(lab.cells(rows[: CFG.n16], lab.change(None, "zero"), tag="W0, zero split, 16 rows"))
    lab.check("W0", all(qr.close(w0[k], w0["K0"], CFG.id_tol) for k in ("K00", "K01"))
              and all(qr.close(zero[k], k0_16, CFG.id_tol) for k in ("K10", "K11")),
              {"K0": w0["K0"], "K00": w0["K00"], "K01": w0["K01"], "K0_16": k0_16, "zero_split": zero})
    e40 = lab.load_dw("e40")
    lab.set_weights("FT", e40)
    kl16 = lab.kl_rows("FT", r16)
    lab.check("E40_KL", qr.close(kl16, L.REF["kl_e40_16"], CFG.rep_tol), {"kl16": kl16, "E57": L.REF["kl_e40_16"]})
    cells = total(lab.cells(r16, tag="E40, 16 rows"))
    held = total(lab.cells(r16, lab.change(e40, None), tag="do-nothing, 16 rows"))
    share = qr.shares(cells | held, r16.numel())["regating"]
    lab.check("do_nothing", qr.ok(share) and abs(share - 1) <= CFG.id_tol, {"share": share} | held)
    lab.drop_set("FT")
    return {"W0": w0, "means_W0": {k: v / rows.numel() for k, v in w0.items()}, "kl_e40_16": kl16, "do_nothing_share": share}


def scaled(lab: Any, name: str, dW: dict[str, Tensor], target: float, r16: Tensor) -> dict[str, Any]:
    """Scale the random change `dW` (in place) to E40's KL on 16 rows, by E57's search."""

    def kl_at(b: float) -> float:
        lab.set_weights("state", dW, b)
        x = lab.kl_rows("state", r16)
        lab.log(f"[{name} search] b {b:.6g}: KL {x:.9g}, target {target:.9g}")
        return x

    search = qr.scale_search(kl_at, target, CFG.kl_match, CFG.kl_evals)
    lab.drop_set("state")
    for m in dW.values():
        m.mul_(search["b"])
    return search


def random_changes(lab: Any) -> dict[str, dict[str, Tensor]]:
    """The Gaussian off-span change (E57's, its size set in `scaled`) and the in-span change on the live columns of this
    decomposition (E57's, at E44 Amendment 2's per-site scales; `scaled` sets its size here, which E57 did not need)."""
    e40, sigma = lab.load_dw("e40"), lab.sigma()
    gauss = {s: pinned(split.random_like(e40[s].to(L.DEV, L.F64), sigma[s].to(L.DEV), null_seed(0, s)).float().cpu())
             for s in SITES}
    per_site = json.loads((L.A2 / "S.json").read_text())["10"]["per_site"]
    return {"gauss": gauss, "inspan": alive_pieces(lab.alive, {s: per_site[s]["a"] * CFG.inspan_a for s in SITES}, L.DEV)}


def read(lab: Any, name: str, dW: dict[str, Tensor], rows: Tensor, extra: dict[str, Any] | None = None) -> None:
    """The four cells of one state on the maths rows, the least-squares split in between (no nulls: only K11 is read)."""
    lab.set_weights("FT", dW)
    cells = lab.cells(rows, tag=name)
    sp = lab.split(dW, nulls=False, halves=False)
    cells |= lab.cells(rows, lab.change(dW, sp["X"]), tag=f"{name}, split")
    lab.drop_set("FT")
    rec = {"decomp": NEW, "rows": cells, "stored_error": sp["stored_error"], "split_s": sp["s"], "means": {
        k: sum(v) / rows.numel() for k, v in cells.items()}} | qr.whole(cells, CFG.row_len) | (extra or {})
    lab.put(f"S_{name}", rec)


def table(recs: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    """Share and 95% row-resample interval per state; the seed spread; each pair against the filed rule."""
    draws = qr.resample(CFG.n_rows, CFG.boot)
    rows_n = CFG.row_len
    boots = {n: [qr.shares(qr.sums_at(r, i), len(i) * rows_n)["regating"] for i in draws] for n, r in recs.items()}
    states: dict[str, dict[str, Any]] = {n: qr.whole(r, rows_n) | {"boot95": qr.interval(boots[n])} for n, r in recs.items()}
    scored = {n: s["status"] == "scored" for n, s in states.items()}
    a, b = "a2long_s0", "a2long_s1"
    spread = abs(states[a]["regating"] - states[b]["regating"]) if scored[a] and scored[b] else None
    pairs = []
    for x, y in combinations(recs, 2):
        if {x, y} == {a, b}:
            continue
        kind = "update-update" if x in UPDATES and y in UPDATES else "update-random" if x in UPDATES or y in UPDATES else "random-random"
        if spread is None or not scored[x] or not scored[y]:
            pairs.append({"a": x, "b": y, "kind": kind, "reading": "unreadable"})
            continue
        gap = states[x]["regating"] - states[y]["regating"]
        low, high = qr.interval([p - q for p, q in zip(boots[x], boots[y], strict=True)])
        beyond = abs(gap) > spread and not low <= 0 <= high
        pairs.append({"a": x, "b": y, "kind": kind, "gap": gap, "gap95": [low, high], "separated": beyond})
    return {"states": states, "seed_spread": spread, "pairs": pairs}


def summarise(lab: Any) -> None:
    """New beside old, whatever states are written."""
    new = {n: lab.get(f"S_{n}")["rows"] for n in (*RANDOM, *UPDATES) if lab.get(f"S_{n}")}
    c0 = json.loads((E57 / "C0.json").read_text())
    old = {n: c0[n]["rows"]["maths"] for n in RANDOM} | {
        n: json.loads((E57 / f"C_{n}.json").read_text())["rows"]["maths"] for n in UPDATES}
    lab.put("summary", {"new": table(new) if len(new) == len(old) else {"read": sorted(new)}, "old": table(old)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--e40", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    a = ap.parse_args()
    assert decomp_id(Path(a.decomp)) == NEW, a.decomp
    lab = L.Lab("E132", a.e40, a.decomp)
    with torch.no_grad():
        assert lab.alive is not None  # the alive set's count is checked here, before any reading
        rows, r16 = lab.maths(), lab.maths(CFG.n16)
        if not lab.get("controls"):
            lab.put("controls", controls(lab, rows, r16))
        target = lab.get("controls")["kl_e40_16"]
        left = [n for n in RANDOM if not lab.get(f"S_{n}")]
        changes = random_changes(lab) if left else {}
        for name in left:
            search = scaled(lab, name, changes[name], target, r16)
            read(lab, name, changes.pop(name), rows, {"search": search, "matched": search["matched"]})
            lab.log(f"[{name}] read, {lab.peaks()}")
        for name in UPDATES:
            if not lab.get(f"S_{name}"):
                read(lab, name, lab.load_dw(name), rows)
        summarise(lab)
    lab.log(f"[E132 done] {lab.peaks()}")


if __name__ == "__main__":
    main()

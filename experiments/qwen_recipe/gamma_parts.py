"""E61 (qwen_recipe), side check: Qwen's g = clip(head + readout(x V), 0, 1). How much of g is the readout, and how much does
a moved V change g with Γ's parameters, x and the backbone held? Rules: results/predictions_gamma_parts.json; audit:
reports/reviews/2026-11-02_review_gamma-variants.md. One job per state: cluster/qwen_recipe_gamma_parts.sbatch.

Expected values, filed before coding:
  W0, moved V through the same path from X = 0: mean |g1 - g0| exactly 0, crossings 0 and 0 (a miss stops).
  W0, clip(head + readout) against `importances`' g: gap 0, at most max(8 fp32 epsilons, twice row 0's repeat gap) (stops).
  W0, carrying set: empty, its means NaN.
  Any other state: mean |g1 - g0| above 0 (stops). Every other number is descriptive, with no prediction.
  The KL effect is not measured here: checks_k10.json.

V stays on the host and moves a site at a time, as an L40 cannot hold fp32 Γ beside all of V. Exit 1: a stop or refusal.
Exit 3: the 25-min budget, resubmit (the fp32 split is saved, so the second job fits).
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from checks import RESULTS, Checks, split_x, state_dw
from lab import CFG, DEV, F32, F64, write_json
from torch import Tensor

from commute import qwen_reading as qr
from commute import split
from commute.qwen_sft import linears
from commute.qwen_vpd import (
    SITES,
    Components,
    backbone,
    clean_forward,
    importances,
    load_components,
    load_gamma,
    parse_site,
    readout,
    site_importance,
)

PRED, STOPPED = RESULTS / "predictions_gamma_parts.json", RESULTS / "gamma_parts_stopped.json"
STATES = ("gold_s0", "gauss", "e40", "a2long_s0")  # those whose split checks_k10 read
GAMMA = {"fp32": F32, "bf16": torch.bfloat16}
READ_S = 450.0  # two passes of 16 rows, one full pass at about 10 s a row: an estimate, the W0 job measures it
HOLD = 0.9  # share of the split's energy the carrying subcomponents hold
Sums = defaultdict[str, float]  # per site, or pooled: sums and counts, turned into means by part_1, part_2


class Parts(Checks):
    """checks.py's job, with its own log and stop file, and the components on the host."""

    gamma_name = "fp32"

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "gamma_parts.log").open("a") as f:
            f.write(line + "\n")

    def stop(self, name: str, why: str) -> None:
        write_json(STOPPED, {"name": name, "why": why, "stage": self.stage})
        self.log(f"STOP {name}: {why}")
        sys.exit(f"STOP {name}: {why}")

    @property
    def comps(self) -> Components:
        if self._comps is None:
            self._comps = load_components(self.decomp, "cpu", F32)
        return self._comps

    def load_gamma_in(self, dtype: torch.dtype) -> None:
        torch.cuda.empty_cache()
        self._gamma = load_gamma(self.decomp, DEV, dtype)
        self.log(f"[load] gamma {dtype}, {self.peaks()}")


class HostV(Components):
    """Hands `clean_forward` each site's V on the GPU; the stacks stay on the host. U is not read."""

    def vu(self, site: str) -> tuple[Tensor, Tensor]:
        V, U = super().vu(site)
        return V.to(DEV), U


def moves(lab: Parts, X: dict[str, Tensor] | str, sig: dict[str, Tensor] | None) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    """Per site dV, as `qwen_maps.applied` adds it, and the mask [C] of subcomponents carrying the update (none without Σ)."""
    change, dV, pe = lab.change(None, X), {}, {}
    for s in SITES:
        _, _, dU, dv = change(s)
        assert dU is not None and dv is not None
        dV[s] = dv.float().cpu()
        if sig is not None:
            U_A, V_A = lab.alive.uv(s, DEV, F64)
            pe[s] = split.piece_energies(dU, dv, U_A, V_A, sig[s].to(DEV)).cpu()
    energy = torch.cat(list(pe.values())) if pe else torch.zeros(lab.alive.count(), dtype=F64)
    top = torch.zeros(len(energy), dtype=torch.bool)
    if float(energy.sum()) > 0:
        order = energy.argsort(descending=True)
        top[order[: int((energy[order].cumsum(0) < HOLD * energy.sum()).sum()) + 1]] = True
    carry, o = {}, 0
    for s in SITES:
        V, idx = lab.comps.vu(s)[0], lab.alive.idx[s]
        carry[s] = torch.zeros(V.shape[1], dtype=torch.bool)
        carry[s][idx] = top[o : o + len(idx)]
        o += len(idx)
    return dV, carry, {"alive": len(top), "carrying": int(top.sum()), "holds": HOLD}


def forward(lab: Parts, t: Tensor, comps: HostV) -> tuple[Any, dict[str, Tensor]]:
    """`clean_forward` for one row, plus each site's input x."""
    xs: dict[str, Tensor] = {}
    hooks = [lin.register_forward_hook(lambda _m, a, _o, s=s: xs.__setitem__(s, a[0])) for s, lin in linears(lab.model).items()]
    try:
        return clean_forward(lab.model, t, lab.gamma, comps), xs
    finally:
        for h in hooks:
            h.remove()


def add_parts(c: Sums, gamma: Any, z: Tensor, s: str, r: Tensor) -> Tensor:
    """Part 1 for one site: the head and readout terms, and g without each. Returns clip(p + r)."""
    layer, kind = parse_site(s)
    p = z @ gamma.heads_w[kind][layer] + gamma.heads_b[kind][layer]  # as `site_importance`; the parts control checks it
    beta = gamma.readouts[s][6]
    g, no_h, no_head = (v.float().clamp(0, 1) for v in (p + r, p + beta, r))
    on = g > 0
    sizes = {"head": p.abs(), "read": r.abs(), "read_h": (r - beta).abs(), "dg_no_h": (g - no_h).abs(), "dg_no_head": (g - no_head).abs()}
    for k, v in sizes.items():
        c[k] += float(v.double().sum())
        c[f"{k}_on"] += float(v[on].double().sum())
    c["n"] += g.numel()
    c["on"] += int(on.sum())
    c["on_no_h"] += int((on & (no_h > 0)).sum())
    c["on_no_head"] += int((on & (no_head > 0)).sum())
    c["off_on_no_h"] += int((~on & (no_h > 0)).sum())
    return g


def add_moved(c: Sums, g0: Tensor, g1: Tensor, carry: Tensor) -> None:
    """Part 2 for one site, on every subcomponent and on those carrying the update."""
    for tag, a, b in (("all", g0, g1), ("carry", g0[..., carry], g1[..., carry])):
        c[f"{tag}_n"] += a.numel()
        c[f"{tag}_dg"] += float((b - a).abs().double().sum())
        c[f"{tag}_on"] += int((a > 0).sum())
        c[f"{tag}_on_to_off"] += int(((a > 0) & ~(b > 0)).sum())
        c[f"{tag}_off_to_on"] += int((~(a > 0) & (b > 0)).sum())


def row_pass(lab: Parts, rows: Tensor, key: str, comps: HostV, dV: dict[str, Tensor], carry: dict[str, Tensor], parts: dict[str, Sums] | None) -> tuple[dict[str, Sums], float]:
    """One pass at weights `key`. With `parts`, also Part 1, and the largest gap between clip(p + r) and `importances`' g."""
    lab.use(key)
    t0, out, gap = time.time(), {s: Sums(float) for s in SITES}, 0.0
    for i, row in enumerate(rows):
        t, gamma = row[None].to(DEV), lab.gamma
        ref = importances(lab.model, t, gamma, comps)[1] if parts is not None else {}
        clean, xs = forward(lab, t, comps)
        z = backbone(gamma, clean.taps)
        for s in SITES:
            r0, V = clean.readouts[s], comps.vu(s)[0]
            V[:, lab.alive.idx[s].to(DEV)] += dV[s].to(DEV)  # V here is the GPU copy `vu` made; the host's stays original
            g0, r1 = site_importance(gamma, z, s, r0), readout(gamma, s, xs[s], V)
            add_moved(out[s], g0, site_importance(gamma, z, s, r1), carry[s].to(DEV))
            if parts is not None:
                gap = max(gap, float((add_parts(parts[s], gamma, z, s, r0) - ref[s]).abs().max()))
            del V, r1, g0
        del clean, xs, z, ref
        if i == 0:
            lab.gpu_rule(f"row pass {key}")
            lab.log(f"[row 0 of {key}] {time.time() - t0:.0f} s, {lab.peaks()}")
    return out, gap


def pooled(sites: dict[str, Sums]) -> Sums:
    out = Sums(float)
    for c in sites.values():
        for k, v in c.items():
            out[k] += v
    return out


def part_1(c: Sums) -> dict[str, float]:
    on, off = c["on"], c["n"] - c["on"]
    means = {f"mean_{k}": qr.div(c[k], c["n"]) for k in ("head", "read", "read_h", "dg_no_h", "dg_no_head")}
    means |= {f"mean_{k}_on": qr.div(c[f"{k}_on"], on) for k in ("head", "read", "read_h", "dg_no_h", "dg_no_head")}
    return means | {"on_share": qr.div(on, c["n"]), "on_needing_h": qr.div(on - c["on_no_h"], on),
                    "on_needing_head": qr.div(on - c["on_no_head"], on), "off_that_h_holds_off": qr.div(c["off_on_no_h"], off)}


def part_2(c: Sums) -> dict[str, dict[str, float]]:
    return {t: {"subcomponent_positions": c[f"{t}_n"], "mean_abs_dg": qr.div(c[f"{t}_dg"], c[f"{t}_n"]), "on_in_g0": c[f"{t}_on"],
                "on_to_off": c[f"{t}_on_to_off"], "off_to_on": c[f"{t}_off_to_on"],
                "crossings_per_on": qr.div(c[f"{t}_on_to_off"] + c[f"{t}_off_to_on"], c[f"{t}_on"])} for t in ("all", "carry")}


def read_w0(lab: Parts, rows: Tensor, comps: HostV, R: dict[str, Any], path: Path) -> None:
    dV, carry, info = moves(lab, "zero", None)
    lab.load_gamma_in(GAMMA[lab.gamma_name])
    t = rows[:1].to(DEV)
    a, b = importances(lab.model, t, lab.gamma, comps)[1], importances(lab.model, t, lab.gamma, comps)[1]
    floor = max(float((a[s] - b[s]).abs().max()) for s in SITES)
    del a, b
    parts = {s: Sums(float) for s in SITES}
    sites, gap = row_pass(lab, rows, "W0", comps, dV, carry, parts)
    total, tol = pooled(sites), max(R["parts_tol"], 2 * floor)
    moved = max(c["all_dg"] + c["all_on_to_off"] + c["all_off_to_on"] for c in sites.values())
    lab.check("identity", moved == 0, {"largest_site_dg_plus_crossings": moved})
    lab.check("parts", gap <= tol, {"gap": gap, "row0_repeat_gap": floor, "tolerance": tol})
    out = {"state": "W0", "gamma": lab.gamma_name, "carrying": info, "parts_gap": gap, "row0_repeat_gap": floor, "tolerance": tol,
           "part_1": part_1(pooled(parts)), "part_2": part_2(total), "part_1_sites": {s: part_1(c) for s, c in parts.items()}}
    write_json(path, out)
    lab.log(f"[W0] {json.dumps({k: out[k] for k in ('parts_gap', 'row0_repeat_gap', 'part_1', 'part_2')}, default=float)}")


def read_state(lab: Parts, name: str, rows: Tensor, comps: HostV, path: Path) -> None:
    dW = state_dw(lab, name)
    X, rec = split_x(lab, f"parts_{name}", dW)
    sig = lab.sigma()
    lab.budget(READ_S)
    dV, carry, info = moves(lab, X, sig)
    del X
    lab.set_weights("FT", dW)
    del dW
    lab.load_gamma_in(GAMMA[lab.gamma_name])
    out: dict[str, Any] = {"state": name, "gamma": lab.gamma_name, "split": rec, "carrying": info, "x": {}}
    for x, key in (("W0", "W0"), ("state", "FT")):
        sites, _ = row_pass(lab, rows, key, comps, dV, carry, None)
        out["x"][x] = {"pooled": part_2(pooled(sites)), "sites": {s: part_2(c) for s, c in sites.items()}}
        lab.log(f"[{name} x from {x}] {json.dumps(out['x'][x]['pooled'], default=float)}")
        write_json(path, out)
    lab.check("moved", out["x"]["W0"]["pooled"]["all"]["mean_abs_dg"] > 0, {"mean_abs_dg": out["x"]["W0"]["pooled"]["all"]["mean_abs_dg"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, choices=("W0", *STATES))
    ap.add_argument("--gamma", default="fp32", choices=tuple(GAMMA))
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit(f"refusing to start: {PRED.name} is missing, or {STOPPED.name} exists (the lead decides)")
    path, w0 = (RESULTS / f"gamma_parts_{n}_{a.gamma}.json" for n in (a.state, "W0"))
    if a.state != "W0" and not w0.is_file():
        sys.exit("refusing: the W0 control runs first")
    lab = Parts(f"gamma_parts {a.state}", str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"), os.environ.get("COMMUTE_QWEN_DIR", ""))
    lab.gamma_name = a.gamma
    rows, comps = lab.maths(CFG.n16), HostV(lab.comps.V, lab.comps.U)
    with torch.no_grad():
        if a.state == "W0":
            read_w0(lab, rows, comps, json.loads(PRED.read_text())["rules"], path)
        else:
            read_state(lab, a.state, rows, comps, path)
    lab.log(f"[{a.state} done] {lab.peaks()}")


if __name__ == "__main__":
    main()

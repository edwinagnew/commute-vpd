"""E44 Amendment 2, the pass test under reading B ($0, one cluster L40, normal QoS). Plan:
reports/plans/2026-10-28_plan_qwen-q0q1.md, "Amendment record (2026-10-30, Amendment 2: masks from the original
model)"; every rule scored here is filed in results/predictions_amend2.json (the run refuses to start without it,
while it has a 'status' key, or while results/amend2/stopped.json exists).

Reading B: a state's CI masks g_B come from Γ run on W0's clean forward (W0's weights in the model, so W0's taps and
site inputs) with the state's own V′ in the readouts; the CI-masked forward (Δ off), the state's own logits and the
replays (m = g_B + (1 − g_B)s, d = s_Δ) use the state's weights and U′, V′.

Departure from the plan's code note ("one change to qwen_maps.read_rows"): read_rows holds the fine-tuned weights on
the GPU beside the state's, 5.6 GB each, and the L40 holds neither beside the fixed 43.5 GB (model, fp32 components,
Γ) under the 44 GiB rule. So, as in Amendment 1, every weight set is pinned on the host and copied into the model's own
tensors in place, per row (`Reader.use`); `qwen_maps.applied(host=...)` builds a state's weights on the host.

Stages, in the filed order; each writes amend2/<stage>.json after every reading (a rerun skips what is written) and
rescores amend2/score.json; every value and score goes to amend2/run.log.
  controls  floor (W0's K0 on row 0, twice); W0 on maths and web; K(D) at 10 and 1; do-nothing through the state path
            on the first 16 rows at 10 and 1. control_W0 or control_do_nothing missed: stop.
  S         Γ freed; run.py's stage S unchanged (E44's config: 128 maths rows × 512, Σ, Gram bases, pieces for steps
            1 and 10, on --out); Γ reloaded; control_S missed: stop.
  t10, m10  step 10: random's scale search, then lower, upper, random (K_B, maths); no teeth: stop. Then least squares,
            in-span, scaled adjoint (maths first; web only if maths is within every line).
  t1, m1    step 1 the same; no teeth at 1: its splits are read and labelled 'unscored'.
Usage (repo root, L40): python experiments/qwen_maps/amend2.py --decomp DIR [--e40 DIR] [--out /ephemeral/$USER/e44a2]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from amend1 import GIB, NAN, check_dw, div, ok, r9
from job import (
    CFG,
    DEV,
    E40_RESULTS,
    F32,
    FT_WEIGHTS,
    MATHS_SOURCE,
    RESULTS,
    SPLITS,
    WEB_SOURCES,
    Job,
    JobOpts,
    write_json,
)
from run import stage_S
from torch import Tensor

from commute import qwen_maps
from commute.qwen_sft import free_bytes, linears
from commute.qwen_vpd import SITES, importances, masked_logits
from commute.vpd import kl_per_position

PRED, OUT = RESULTS / "predictions_amend2.json", RESULTS / "amend2"
STAGES = ("controls", "S", "t10", "m10", "t1", "m1")


@dataclass(frozen=True)
class A2:
    """Sizes and the filed thresholds (predictions_amend2.json `rules`, `early`)."""

    row_len: int = 256
    n16: int = 16  # "first 16 rows": Amendments 1 and 1b's rows
    floor_max: float = 1e-3
    rep_tol: float = 1e-3
    k0_ref: float = 0.06558158050898716  # A1b KLself_0
    kd_ref: float = 1.4956748775221875  # A1 K_0
    id_tol: float = 1e-6
    r2_zero: float = 1e-5
    line: float = 1.5
    kept_min: float = 0.8
    r2_floor: float = 0.05  # max(0.05, 5 × 0.000208)
    share_tol: float = 1e-3
    store_max: float = 1e-3
    carried_min: float = 0.05
    margin: float = 0.2
    random_max: float = 0.2
    match: float = 1.5
    evals: int = 12
    boot: int = 2000
    gpu_max_gib: float = 44.0
    s_limit_min: float = 60.0
    disk_min: float = 80e9
    wall_s: float = 2 * 3600 + 45 * 60


L = A2()
SHARE_REF = {"10": {"lower": 0.7119745504490399, "upper": 0.5218200761203577, "ls": 0.36374349077979745,
                    "inspan": 0.001143571318707781, "adjoint": 0.00046571085136145186},
             "1": {"lower": 0.7139646894375388, "upper": 0.5267788745211925, "ls": 0.3715557940427841,
                   "inspan": 0.0002070508852748363, "adjoint": 7.318813066048722e-05}}

# ----------------------------------------------------------------------------- scoring (filed rules)


def mean(rd: dict | None, key: str, n: int | None = None) -> float:
    """Mean over every position of the first n rows (all: None) from a reading's per-row KL sums; NaN if absent."""
    xs = (rd or {}).get("sums", {}).get(key, [])[:n]
    return sum(xs) / (len(xs) * rd["T"]) if xs and rd else NAN


def kept(res: dict, t: str, kb: float) -> tuple[float, str]:
    """r2_B = [K(D) − K_B] / [K(D) − K0] on all maths rows: 'scored', 'unscored' (gap under the floor) or 'nan'."""
    c = res.get("controls", {})
    den = mean(c.get(f"KD_{t}"), "K") - mean(c.get("W0_maths"), "ci")
    if not ok(den):
        return NAN, "nan"
    if r9(den) < L.r2_floor:
        return NAN, "unscored"
    r2 = div(mean(c.get(f"KD_{t}"), "K") - kb, den)
    return r2, "scored" if ok(r2) else "nan"


def control_w0(c: dict) -> dict[str, Any]:
    f = abs(div(mean(c.get("floor_a"), "ci"), mean(c.get("floor_b"), "ci")) - 1)
    k0, kd = mean(c.get("W0_maths"), "ci", L.n16), mean(c.get("KD_10"), "K", L.n16)
    g0, gd = abs(k0 / L.k0_ref - 1), abs(kd / L.kd_ref - 1)
    return {"f": f, "K0_16": k0, "KD10_16": kd, "gap_K0": g0, "gap_KD": gd, "sha256": c.get("sha256") is True,
            "K0_all": mean(c.get("W0_maths"), "ci"), "KD10_all": mean(c.get("KD_10"), "K"),
            "KD1_all": mean(c.get("KD_1"), "K"), "e44_K0": 0.05888,
            "e44_KD10": 1.41555, "holds": (ok(f) and r9(f) <= L.floor_max and ok(g0) and r9(g0) <= L.rep_tol
                                           and ok(gd) and r9(gd) <= L.rep_tol and c.get("sha256") is True)}


def control_dn(c: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    k0 = mean(c.get("W0_maths"), "ci", L.n16)
    for t in ("10", "1"):
        kd, kb, own = mean(c.get(f"KD_{t}"), "K", L.n16), mean(c.get(f"dn_{t}"), "K"), mean(c.get(f"dn_{t}"), "ci")
        gap, r2, r1 = abs(div(kb, kd) - 1), div(kd - kb, kd - k0), div(own, k0)
        out[t] = {"KD_16": kd, "K_B": kb, "gap": gap, "r2": r2, "r1m": r1,
                  "holds": (ok(gap) and r9(gap) <= L.id_tol and ok(r2) and r9(abs(r2)) <= L.r2_zero
                            and ok(r1) and r9(r1) > L.line)}
    out["holds"] = out["10"]["holds"] and out["1"]["holds"]
    return out


def control_s(S: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"holds": True}
    for t, ref in SHARE_REF.items():
        got = S.get(t, {}).get("share", {})
        gaps = {n: abs(float(got.get(n, NAN)) - v) for n, v in ref.items()}
        err = float(S.get(t, {}).get("stored_error_max", NAN))
        holds = all(ok(g) and r9(g) <= L.share_tol for g in gaps.values()) and ok(err) and r9(err) <= L.store_max
        out[t] = {"share": got, "gap": gaps, "stored_error_max": err, "holds": holds}
        out["holds"] = out["holds"] and holds
    return out


def summary(res: dict, t: str, st: dict) -> dict[str, Any]:
    """A state's readings against the pass rule: maths first (any reading over its line: 'fail', web not read), then
    web; 'maths within' while web is unread; 'incomplete' when r2_B is unscored; NaN: 'unreadable'."""
    c, m, w = res["controls"], st.get("maths"), st.get("web")
    r1m = div(mean(m, "ci"), mean(c["W0_maths"], "ci"))
    r3m = div(mean(m, f"replay_{MATHS_SOURCE}"), mean(c["W0_maths"], f"replay_{MATHS_SOURCE}"))
    r2, how = kept(res, t, mean(m, "K"))
    r1w = div(mean(w, "ci"), mean(c["W0_web"], "ci"))
    r3w = sum(div(mean(w, f"replay_{n}"), mean(c["W0_web"], f"replay_{n}")) for n in WEB_SOURCES) / len(WEB_SOURCES)
    out = {"r1m": r1m, "r3m": r3m, "r2": r2, "r2_status": how, "r1w": r1w, "r3w": r3w, "K_B": mean(m, "K"),
           "share": res.get("S", {}).get(t, {}).get("share", {}).get(st.get("name", ""), NAN)}
    if how == "nan" or not (ok(r1m) and ok(r3m)):
        v = "unreadable"
    elif r9(r1m) > L.line or r9(r3m) > L.line or (how == "scored" and r9(r2) < L.kept_min):
        v = "fail"
    elif w is None:
        v = "maths within"
    elif not (ok(r1w) and ok(r3w)):
        v = "unreadable"
    elif r9(r1w) > L.line or r9(r3w) > L.line:
        v = "fail"
    else:
        v = "incomplete" if how == "unscored" else "pass"
    return out | {"verdict": v}


def boot_rows(c: dict, T: dict, t: str, cl: float, cu: float) -> dict[str, Any]:
    """Reported, not scored: rows where K_B(lower) > K_B(upper), and a bootstrap 95% interval (seed 0) of
    rho_upper − rho_lower over rows."""
    kd, k0 = c[f"KD_{t}"]["sums"]["K"], c["W0_maths"]["sums"]["ci"]
    lo, up = T["lower"]["sums"]["K"], T["upper"]["sums"]["K"]
    rng, n, diffs = random.Random(0), len(lo), []
    for _ in range(L.boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sd, s0, sl, su = (sum(x[i] for i in idx) for x in (kd, k0, lo, up))
        d = div(div(sd - su, sd - s0), cu) - div(div(sd - sl, sd - s0), cl)
        if ok(d):
            diffs.append(d)
    diffs.sort()
    ci = [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]] if diffs else [NAN, NAN]
    return {"rows_lower_above_upper": sum(a > b for a, b in zip(lo, up, strict=True)), "rows": n, "boot95": ci}


def teeth_i(res: dict, t: str) -> dict[str, Any]:
    T, share = res.get(f"t{t}", {}), res.get("S", {}).get(t, {}).get("share", {})
    if "lower" not in T or "upper" not in T:
        return {"result": "not read", "holds": False}
    (r2l, hl), (r2u, hu) = kept(res, t, mean(T["lower"], "K")), kept(res, t, mean(T["upper"], "K"))
    cl, cu = 1 - float(share.get("lower", NAN)), 1 - float(share.get("upper", NAN))
    rl, ru = div(r2l, cl), div(r2u, cu)
    out = {"r2_lower": r2l, "r2_upper": r2u, "carried_lower": cl, "carried_upper": cu, "rho_lower": rl, "rho_upper": ru}
    if "nan" in (hl, hu) or not (ok(cl) and ok(cu)):
        res_ = "unreadable"
    elif "unscored" in (hl, hu):
        res_ = "unscored"
    elif r9(cl) <= L.carried_min or r9(cu) <= L.carried_min or not (ok(rl) and ok(ru)):
        res_ = "unreadable"
    else:
        res_ = "holds" if r9(rl) <= r9(ru - L.margin) else "does not hold"
    out |= {"result": res_, "holds": res_ == "holds"}
    return out | boot_rows(res["controls"], T, t, cl, cu) if res_ in ("holds", "does not hold") else out


def teeth_ii(res: dict, t: str) -> dict[str, Any]:
    T = res.get(f"t{t}", {})
    if "random" not in T:
        return {"result": "not read", "holds": False}
    r2, how = kept(res, t, mean(T["random"], "K"))
    s = T["search"]
    out = {"r2": r2, "a": s["a"], "kl_random": s["kl"], "kl_ft": s["target"], "matched": s["matched"],
           "r1m": div(mean(T["random"], "ci"), mean(res["controls"]["W0_maths"], "ci"))}
    res_ = ("unreadable" if not s["matched"] or how != "scored" else "holds" if r9(r2) <= L.random_max
            else "does not hold")
    return out | {"result": res_, "holds": res_ == "holds"}


def score(res: dict, _pred: Any = None) -> dict[str, Any]:
    c, sc = res.get("controls", {}), {}
    sc["control_W0"], sc["control_do_nothing"] = control_w0(c), control_dn(c)
    if "S" in res:
        sc["control_S"] = control_s(res["S"])
    for t in ("10", "1"):
        if f"t{t}" in res:
            ti, tii = teeth_i(res, t), teeth_ii(res, t)
            teeth = sc["control_do_nothing"][t]["holds"] and ti["holds"] and tii["holds"]
            sc[f"teeth_{t}"] = {"teeth_i": ti, "teeth_ii": tii, "teeth": teeth, "scored": t == "10"}
        for n, st in res.get(f"m{t}", {}).items():
            if isinstance(st, dict):
                s = summary(res, t, st | {"name": n})
                scored = sc.get(f"teeth_{t}", {}).get("teeth", False)
                sc[f"pass_{n}_{t}"] = s | {"verdict": s["verdict"] if scored else f"unscored ({s['verdict']})"}
    return sc

# ----------------------------------------------------------------------------- reading B, row by row


def kl(p: Tensor, q: Tensor) -> float:
    return float(kl_per_position(p, q).double().sum())


def pinned(w: Tensor) -> Tensor:
    return torch.empty(w.shape, dtype=w.dtype, pin_memory=True).copy_(w)


class Reader:
    """Weight sets W0, FT (one step at a time) and a state's, pinned on the host and copied into the model in place
    (`use`; the model is at W0 between readings), and reading B's row loop."""

    def __init__(self, job: Job):
        self.job, self.lins = job, linears(job.model)
        self.sets = {"W0": {s: pinned(lin.weight.data) for s, lin in self.lins.items()}}
        self.now, self.ft_t = "W0", 0
        self.rows = {b: job.rows(b)[:, : L.row_len] for b in ("maths", "web")}
        self.w0_logits: list[Tensor] = []

    def use(self, key: str) -> None:
        if key != self.now:
            for s, lin in self.lins.items():
                lin.weight.data.copy_(self.sets[key][s])
            torch.cuda.synchronize()
            self.now = key

    def set_ft(self, t: int) -> None:
        """FT = W0 + δW_t, added on the GPU in fp32 as Job.ft and amend1.Weights do."""
        if self.ft_t != t:
            assert self.now == "W0"
            self.sets.pop("FT", None)
            dW = self.job.dw(t)
            self.sets["FT"] = {s: pinned(lin.weight.data + dW[s].to(DEV, F32)) for s, lin in self.lins.items()}
            self.ft_t = t

    def drop_ft(self) -> None:
        self.sets.pop("FT", None)
        self.ft_t = 0

    def read(self, name: str, t: int | None, block: str, want: set[str], n: int | None = None,
             scale: float = 1.0) -> dict[str, Any]:
        """Per-row KL sums of state `name` under reading B: "K" = KL(FT_t ‖ CI-masked), "ci" = KL(own ‖ CI-masked),
        "replay_<source>" = KL(own ‖ replayed). Stop: GPU peak after the first row over 44 GiB."""
        job, t0 = self.job, time.time()
        own = "W0" if name == "W0" else "FT" if name in FT_WEIGHTS else "state"
        if t is not None and ("K" in want or own == "FT"):
            self.set_ft(t)
        srcs = job.sources(block) if "replay" in want else {}
        host: dict[str, Tensor] | None = {} if own == "state" else None
        keep = None if host is not None else {s: lin.weight.data for s, lin in self.lins.items()}
        sums: dict[str, list[float]] = {}
        torch.cuda.reset_peak_memory_stats()
        with qwen_maps.applied(job.model, job.comps(), job.alive, job.change(name, t, scale=scale), keep, host):
            if host is not None:
                self.sets["state"] = host
            try:
                for i, tokens in enumerate(self.rows[block][:n]):
                    for k, v in self.row(tokens[None].to(DEV), own, want, srcs).items():
                        sums.setdefault(k, []).append(v)
                    if i == 0 and (peak := torch.cuda.max_memory_allocated() / GIB) > L.gpu_max_gib:
                        job.stop("memory", f"{name} {block}: GPU peak {peak:.2f} GiB after row 0 > {L.gpu_max_gib}")
            finally:
                self.use("W0")
                self.sets.pop("state", None)
        rd = {"T": L.row_len, "sums": sums}
        job.log(f"[{name} {t} {block}] " + " ".join(f"{k} {mean(rd, k):.9g}" for k in sums)
                + f"  rows {len(next(iter(sums.values())))}, {time.time() - t0:.0f} s, {job.peaks()}")
        return rd

    def row(self, tokens: Tensor, own_key: str, want: set[str], srcs: dict[str, dict[str, Tensor]]
            ) -> dict[str, float]:
        model, gamma, comps, out = self.job.model, self.job.gamma(), self.job.comps(), {}
        logits, g = importances(model, tokens, gamma, comps)  # g_B: W0's forward, the state's V′ in the readouts
        own = logits if own_key == "W0" else None
        del logits
        ci = masked_logits(model, tokens, comps, g)  # Δ off: the weights do not enter
        if "K" in want:
            self.use("FT")
            ref = model(tokens, use_cache=False).logits
            out["K"] = kl(ref, ci)
            own = ref if own_key == "FT" else own
            del ref
        if want & {"ci", "replay"}:
            self.use(own_key)  # the replays' Δ term reads the state's weights
            own = model(tokens, use_cache=False).logits if own is None else own
            if "ci" in want:
                out["ci"] = kl(own, ci)
            for name, src in srcs.items():  # as qwen_maps.read_rows
                masks = {s: g[s] + (1 - g[s]) * src[s].to(g[s]) for s in SITES}
                deltas = {s: src[f"{s}.delta"].to(g[s]).expand(tokens.shape) for s in SITES}
                out[f"replay_{name}"] = kl(own, masked_logits(model, tokens, comps, masks, deltas))
                del masks, deltas
        self.use("W0")
        return out

    def kl_w0(self, key: str) -> float:
        """KL(W0 ‖ the weight set `key`, unmasked) on the first 16 maths rows."""
        rows = self.rows["maths"][: L.n16]
        if not self.w0_logits:
            self.w0_logits = [self.job.model(r[None].to(DEV), use_cache=False).logits.cpu() for r in rows]
        self.use(key)
        s = [kl(p.to(DEV), self.job.model(r[None].to(DEV), use_cache=False).logits)
             for p, r in zip(self.w0_logits, rows, strict=True)]
        self.use("W0")
        return sum(s) / rows.numel()

    def search(self, t: int) -> dict[str, Any]:
        """Random's scale a on E44's Σ-matched pieces: bisection in log a (×10 steps until KL(W0 ‖ random) brackets
        KL(W0 ‖ FT), then geometric midpoints), at most 12 evaluations, stopping inside the factor 1.5; the
        closest evaluated a is kept (a non-monotone KL keeps the closest too)."""
        job = self.job
        self.set_ft(t)

        def kl_random(a: float) -> float:
            host: dict[str, Tensor] = {}
            with qwen_maps.applied(job.model, job.comps(), job.alive, job.change("random", t, scale=a), None, host):
                self.sets["state"] = host
                x = self.kl_w0("state")
                self.sets.pop("state")
            job.log(f"[search {t}] a {a:.6g}: KL(W0 ‖ random) {x:.9g}")
            return x

        return bisect(kl_random, self.kl_w0("FT"))


def within(x: float, target: float) -> bool:
    r = div(x, target)
    return ok(r) and r > 0 and r9(max(r, 1 / r)) <= L.match


def bisect(f: Any, target: float) -> dict[str, Any]:
    """f(a) = KL(W0 ‖ random at scale a). A non-finite KL counts as too large; no evaluation when the target is not a
    positive number ('unmatched' at a = 1)."""
    a, lo, hi, evals = 1.0, 0.0, math.inf, []
    for _ in range(L.evals if ok(target) and target > 0 else 0):
        x = f(a)
        evals.append([a, x])
        if within(x, target):
            break
        lo, hi = (a, hi) if ok(x) and x < target else (lo, a)
        a = a * 10 if hi == math.inf else a / 10 if lo == 0 else math.sqrt(lo * hi)
    gap = lambda e: abs(math.log(e[1] / target)) if ok(e[1]) and e[1] > 0 else math.inf
    best = min(evals, key=gap, default=[1.0, NAN])
    return {"target": target, "evals": evals, "a": best[0], "kl": best[1], "ratio": div(best[1], target),
            "matched": within(best[1], target)}

# ----------------------------------------------------------------------------- stages


def save(job: Job, stage: str) -> None:
    write_json(OUT / f"{stage}.json", job.res[stage])
    sc = score(job.res)
    write_json(OUT / "score.json", sc)
    job.log(f"[score] {json.dumps(sc, default=float)}")


def wall(job: Job) -> None:
    """Checked between states: past 2 h 45 min the job exits 0 with what it wrote (a rerun resumes)."""
    if time.time() - job.t0 > L.wall_s:
        job.log(f"cut short: wall clock {time.time() - job.t0:.0f} s > {L.wall_s:.0f}; rerun to resume")
        sys.exit(0)


def todo(job: Job, stage: str, items: list[tuple[str, Any]]) -> dict[str, Any]:
    d = job.res.setdefault(stage, {})
    job.stage = stage
    for key, fn in items:
        if key not in d:
            wall(job)
            d[key] = fn()
            save(job, stage)
    return d


def stage_controls(rd: Reader) -> None:
    items: list[tuple[str, Any]] = [
        ("floor_a", lambda: rd.read("W0", None, "maths", {"ci"}, n=1)),
        ("floor_b", lambda: rd.read("W0", None, "maths", {"ci"}, n=1)),
        ("W0_maths", lambda: rd.read("W0", None, "maths", {"ci", "replay"})),
        ("W0_web", lambda: rd.read("W0", None, "web", {"ci", "replay"}))]
    for t in (10, 1):
        items += [(f"KD_{t}", lambda t=t: rd.read("W0", t, "maths", {"K"})),
                  (f"dn_{t}", lambda t=t: rd.read("do_nothing", t, "maths", {"K", "ci"}, n=L.n16))]
    rd.job.res.setdefault("controls", {})["sha256"] = True  # check_dw stopped the job otherwise
    c = todo(rd.job, "controls", items)
    for name, rule in (("control_W0", control_w0), ("control_do_nothing", control_dn)):
        if not (r := rule(c))["holds"]:
            rd.job.stop(name, f"{name} missed: {json.dumps(r, default=float)}")


def stage_s(rd: Reader) -> None:
    job = rd.job
    job.stage = "S"
    files = all((job.site_dir(t) / f"{s}.safetensors").is_file() for t in job.cfg.steps for s in job.sites)
    if not (job.res.get("S", {}).get("done") and files):
        job.out.mkdir(parents=True, exist_ok=True)
        if (free := free_bytes(job.out)) < L.disk_min:
            job.stop("disk", f"{free / 1e9:.0f} GB free on {job.out} before S < {L.disk_min / 1e9:.0f}")
        rd.drop_ft()
        job.drop_gamma()  # fp32 components + S's working set would pass 44 GiB beside Γ
        stage_S(job)  # E44's stage S unchanged (writes S.json, rescored through job.put)
    job.gamma()
    if not (r := control_s(job.res["S"]))["holds"]:
        job.stop("control_S", f"control_S missed: {json.dumps(r, default=float)}")


def stage_teeth(rd: Reader, t: int) -> None:
    job, key = rd.job, f"t{t}"
    todo(job, key, [("search", lambda: rd.search(t))])
    a = job.res[key]["search"]["a"]
    todo(job, key, [("lower", lambda: rd.read("lower", t, "maths", {"K"})),
                    ("upper", lambda: rd.read("upper", t, "maths", {"K"})),
                    ("random", lambda: rd.read("random", t, "maths", {"K", "ci"}, scale=a))])
    if t == 10 and not score(job.res)["teeth_10"]["teeth"]:
        job.stop("teeth", "no teeth at step 10 (teeth_outcome): stop before the splits")


def stage_splits(rd: Reader, t: int) -> None:
    job, key = rd.job, f"m{t}"
    d = job.res.setdefault(key, {})
    job.stage = key
    for n in SPLITS:
        st = d.setdefault(n, {})
        if "maths" not in st:
            wall(job)
            st["maths"] = rd.read(n, t, "maths", {"K", "ci", "replay"})
            save(job, key)
        if "web" not in st and summary(job.res, str(t), st)["verdict"] == "maths within":
            st["web"] = rd.read(n, t, "web", {"ci", "replay"})
            save(job, key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--e40", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    ap.add_argument("--sums", default=str(RESULTS / "regen_SHA256SUMS"))
    ap.add_argument("--out", default=f"/ephemeral/{os.environ.get('USER', 'e.agnew')}/e44a2")  # S's files
    a = ap.parse_args()
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if (OUT / "stopped.json").is_file():
        sys.exit(f"refusing to start: {OUT / 'stopped.json'} exists (a stop rule fired; the user decides what next)")
    cfg = dataclasses.replace(CFG, s_limit_min=L.s_limit_min)
    job = Job(JobOpts(out=a.out, e40_large=str(E40_RESULTS / "large" / "nowarmup"), decomp=a.decomp,
                      results=str(OUT), e40=a.e40), cfg, score)
    job.res |= {s: json.loads(p.read_text()) for s in STAGES if (p := OUT / f"{s}.json").is_file()}
    job.log(f"[config] {L}; S: {cfg}; resumed {sorted(job.res)}; args {vars(a)}")
    for t in (10, 1):
        check_dw(job, Path(a.e40) / "dW" / f"step_{t}.safetensors", Path(a.sums))
    with torch.no_grad():
        job.gamma()
        rd = Reader(job)
        stage_controls(rd)
        stage_s(rd)
        for t in (10, 1):
            stage_teeth(rd, t)
            stage_splits(rd, t)
    job.log(f"[done] {job.peaks()}")


if __name__ == "__main__":
    main()

"""E77, the three open points of the independent check on the masked forward ($0, one cluster L40, normal QoS). Plan:
reports/plans/2026-11-03_plan_e77-open-points.md; rules: rules.py, results/predictions.json (the run refuses to start without it,
and while results/stopped.json exists). Every forward is E74's `forward` or E72's T1 row; two identities let E74's hook do all of it:
  m = 1, d = 1 gives ((x V) * 0) U + x Wᵀ = x Wᵀ, the library's routed-out site (decomposed_linear.py:142-175);
  m = 1 + ε g, d = 1 gives x Wᵀ + ε ((x V) * g) U, the own-output change in the clean model; m = (1 + ε) g, Δ off, in the masked one.
Order: the control on u1 alone, which also sets the no-call bands; read 1 (stop on a failure); read 2; read 3. Each read writes
results/<update>/<read>.json as it ends (a rerun skips it); every number goes to results/run.log. A stop: results/stopped.json,
exit 1. Past --wall-min, between reads: exit 0, resubmit to resume.

Usage (repo root; -P and PYTHONPATH as cluster/qwen_open_points.sbatch, so `run` means qwen_maps/run.py for amend2.py):
  PYTHONPATH=experiments/qwen_maps:experiments/qwen_pass_test:src .venv/bin/python -P experiments/qwen_open_points/run.py --decomp DIR --wall-min 110
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.jobs import hard_exit
from commute.qwen_vpd import (
    SITES,
    importances,
    masked_logits,
    parse_site,
    stochastic_masks,
)

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
E72, E74 = HERE.parent / "qwen_fixed_masks", HERE.parent / "qwen_transmission"
DEV, ROWS, EPS0, BLOCKS, MODES = "cuda", 32, 0.01, (0, 7), ("fixed", "stochastic")
STOCH_SEED, ROUTE_SEED = 74, 77_000  # 74 is E74 Amendment 2's, so the all-masked point has its draws
POINTS = {"k196": (196, 1), "k0": (0, 1), "k147": (147, 3), "k98": (98, 3), "k49": (49, 3), "k12": (12, 3), "library": (None, 3)}  # k, draws per row


def load_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rules = load_file("open_points_rules", HERE / "rules.py")
fr = load_file("fixed_run", E72 / "run.py")  # FixedReader; its pr (E68's run.py) and a1
e74 = load_file("e74_run", E74 / "run.py")
pr, a1 = fr.pr, fr.a1


class Lazy:
    """One site's masks built when E74's hook asks for them: a whole copy is 0.7 GB, with ~2 GB free on the L40."""

    def __init__(self, fn: Callable[[str], Tensor]):
        self.fn = fn

    def __getitem__(self, site: str) -> Tensor:
        return self.fn(site)


def uniform_k_routes(T: int, k: int | None, gen: torch.Generator) -> dict[str, Tensor]:
    """`UniformKRouting.__call__` (core/recon.py:224-230) at leading shape (1, T). k None is the library's own draw, uniform on
    1..196 per position; a given k fixes that one number. Per position exactly k sites route to the masked branch."""
    kk = torch.randint(1, len(SITES) + 1, (1, T), generator=gen, device=DEV) if k is None else torch.full((1, T), k, device=DEV)
    routed = torch.rand((len(SITES), 1, T), generator=gen, device=DEV).argsort(0) < kk
    return {s: routed[j] for j, s in enumerate(SITES)}


def routed(g: dict[str, Tensor], d: dict[str, Tensor] | None, route: dict[str, Tensor]) -> tuple[Lazy, Lazy]:
    """Masks and Δ shares for E74's `forward`: as given where routed, m = 1 and d = 1 where not."""
    return (Lazy(lambda s: torch.where(route[s][..., None], g[s], 1.0)),
            Lazy(lambda s: torch.where(route[s], 0.0 if d is None else d[s], 1.0)))


def own_clean(g: dict[str, Tensor], eps: float, block: int) -> tuple[Lazy, Lazy]:
    """The clean model with ε ((x V) * g) U added at the block's sites."""
    one = torch.ones((), device=DEV)
    return Lazy(lambda s: 1 + eps * g[s] if parse_site(s)[0] == block else one), Lazy(lambda s: one)


def own_masked(g: dict[str, Tensor], eps: float, block: int) -> Lazy:
    """The masked forward with the same term: the block's masks times 1 + ε."""
    return Lazy(lambda s: (1 + eps) * g[s] if parse_site(s)[0] == block else g[s])


def shift(ref: Tensor, q: Tensor) -> tuple[float, Tensor]:
    """KL(ref ‖ q), and q turned in place into the logit shift centred per position, as E74's `read_group`: the L40 is short of memory."""
    k = pr.kl(ref, q)
    q.sub_(ref)
    return k, q.sub_(q.mean(-1, keepdim=True))


def numbers(a: Tensor, b: Tensor) -> dict[str, float]:
    return {"dot": float((a * b).double().sum()), "bb": float((b * b).double().sum())}


def add(sums: dict[str, list[float]], one: dict[str, float]) -> None:
    for k, v in one.items():
        sums.setdefault(k, []).append(v)


class Reads:
    def __init__(self, job: Any, rd: Any, rows: dict[str, Tensor], wall_s: float):
        self.job, self.rd, self.all_rows, self.wall_s, self.u = job, rd, rows, wall_s, None
        self.model, self.gamma, self.comps = job.model, job.gamma(), job.comps()

    def select(self, u: Any) -> Tensor:
        if self.u is not u:
            self.job.select(u, RESULTS / "unused")  # stage S's folder, never written here
            self.rd.drop_ft()
            self.u = u
        return self.all_rows[u.rows][:ROWS]

    def cached(self, u: Any, key: str, fn: Callable[[Tensor], dict[str, Any]]) -> dict[str, Any]:
        path = RESULTS / u.name / f"{key}.json"
        if path.is_file():
            return json.loads(path.read_text())
        if time.time() - self.job.t0 > self.wall_s:
            self.job.log(f"cut short at {u.name}/{key}; resubmit to resume")
            sys.exit(0)
        t0, self.job.stage = time.time(), f"{u.name}/{key}"
        r = fn(self.select(u))
        path.parent.mkdir(parents=True, exist_ok=True)
        a1.write_json(path, r)
        shown = {k: r[k] for k in ("summary", "fit") if k in r} or {k: a1.mean_of(r, k) for k in r["sums"]}
        self.job.log(f"[{u.name}/{key}] {time.time() - t0:.0f} s, {self.job.peaks()}: {json.dumps(shown)}")
        return r

    def forward(self, t: Tensor, g: Any, dW: dict[str, Tensor] | None = None, d: Any = None) -> Tensor:
        return e74.forward(self.model, t, self.comps, g, dW or {}, deltas=d)

    def own(self, t: Tensor, g: dict[str, Tensor], eps: float, block: int) -> Tensor:
        G, D = own_clean(g, eps, block)
        return self.forward(t, G, d=D)

    def summed(self, rows: Tensor, arms: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        """The shape E74's `summary` reads (K0 is the same forward's, kept per arm here). Each arm: K_clean, K_masked, dot, aa, bb."""
        r = {"T": rows.shape[1], "base": {"K0": next(iter(arms.values()))["K0"]}, "arms": arms} | extra
        return r | {"summary": e74.summary(r)}

    def route(self, u: Any, mode: str, label: str) -> dict[str, Any]:
        """Read 2's one point: the whole update under routing. k196 must reproduce E74's read and k0 the identity, on a resume too."""
        k, draws = POINTS[label]

        def read(rows: Tensor) -> dict[str, Any]:
            dW = self.job.dw(u.step)
            stoch = torch.Generator(DEV).manual_seed(STOCH_SEED)
            gen = torch.Generator(DEV).manual_seed(ROUTE_SEED + (197 if k is None else k))
            arm: dict[str, list[float]] = {}
            for tokens in rows:
                t = tokens[None].to(DEV)
                clean, g = importances(self.model, t, self.gamma, self.comps)
                d = None
                if mode == "stochastic":  # one draw per row, as E74; the routes are drawn apart, so this stream stays E74's
                    g, d = stochastic_masks(g, stoch)
                kc, a = shift(clean, self.forward(t, None, dW))
                row: dict[str, float] = {}
                for _ in range(draws):
                    G, D = routed(g, d, uniform_k_routes(t.shape[1], k, gen))
                    base = self.forward(t, G, d=D)
                    km, b = shift(base, self.forward(t, G, dW, D))
                    one = {"K0": pr.kl(clean, base), "K_masked": km} | numbers(a, b)
                    row = {x: row.get(x, 0.0) + v / draws for x, v in one.items()}
                    del base, b
                add(arm, row | {"K_clean": kc, "aa": float((a * a).double().sum())})
                del clean, g, d, a
            return self.summed(rows, {"whole": arm}, {"k": k, "draws": draws})

        r = self.cached(u, f"route_{mode}_{label}", read)
        s = r["summary"]["whole"]
        if label == "k196":
            want = json.loads((E74 / "results" / u.name / ("all.json" if mode == "fixed" else "stochastic.json")).read_text())
            got, old = s | {"K0": sum(r["base"]["K0"])}, want["summary"]["all"] | {"K0": sum(want["base"]["K0"])}
            self.job.log(f"[control {u.name} {mode}] {json.dumps(got)} vs E74 {json.dumps(old)}")
            if not all(rules.close(got[x], old[x]) for x in ("K_clean", "K_masked", "tau", "amp", "K0")):
                self.job.stop("control_reproduce", f"{u.name} {mode}: {got} vs {old}")
        if label == "k0":
            self.job.log(f"[control {u.name} {mode} no site masked] tau {s['tau']:.9g}, amp {s['amp']:.9g}")
            if not (abs(s["tau"] - 1) <= rules.ID_TOL and abs(s["amp"] - 1) <= rules.ID_TOL):
                self.job.stop("control_identity", f"{u.name} {mode}: {s}")
        return r

    def displacement(self, u: Any) -> dict[str, Any]:
        """Read 3: E72's T1 row (`FixedReader.row`: masks from Γ on FT, W0's V), the update off and on, each read against the other."""

        def read(rows: Tensor) -> dict[str, Any]:
            rd, dW = self.rd, self.job.dw(u.step)
            rd.rows["maths"] = rows
            rd.set_ft(u.step)

            def row(t: Tensor) -> dict[str, float]:
                rd.use("FT")  # Δ is off, so the masked forward does not read the weights
                ft, g = importances(self.model, t, self.gamma, self.comps)
                off = masked_logits(self.model, t, self.comps, g)
                on = a1.forward_plus(self.model, t, self.comps, g, dW, 1.0)
                out = {"K": pr.kl(ft, off), "K_on": pr.kl(ft, on), "D": pr.kl(off, on), "g_sum": float(sum(v.double().sum() for v in g.values()))}
                del off, on, g
                rd.use("W0")
                return out | {"R": pr.kl(self.model(t, use_cache=False).logits, ft)}

            return rd.loop(f"displacement {u.name}", row)

        return self.cached(u, "displacement", read)

    def control_t1(self, u: Any) -> None:
        """E72's do-nothing and ceiling on the same rows, through this read's row."""
        s = self.displacement(u)["sums"]
        want = {f: json.loads((E72 / "results" / u.name / f"{f}.json").read_text())["sums"] for f in ("do_nothing", "ceiling")}
        got = {x: sum(s[x]) for x in ("K", "K_on", "g_sum")}
        old = {"K": sum(want["do_nothing"]["K"][:ROWS]), "K_on": sum(want["ceiling"]["K"][:ROWS]), "g_sum": sum(want["do_nothing"]["g_sum"][:ROWS])}
        self.job.log(f"[control T1 {u.name}] {got} vs E72 {old}")
        if not all(rules.close(got[x], old[x]) for x in got):
            self.job.stop("control_t1", f"{got} vs {old}")

    def fit(self, rows: Tensor, block: int, target: float) -> dict[str, float]:
        """ε with the own-output change's clean KL at δW's at this block: the quadratic rule, E74's rounds and tolerance."""
        eps, got = EPS0, 0.0
        for i in range(e74.C.fit_rounds):
            got = 0.0
            for tokens in rows:
                t = tokens[None].to(DEV)
                clean, g = importances(self.model, t, self.gamma, self.comps)
                got += pr.kl(clean, self.own(t, g, eps, block)) / rows.numel()
            self.job.log(f"[fit block {block}] eps {eps:.6g}: clean KL {got:.6g}, target {target:.6g}")
            if abs(got / target - 1) <= e74.C.fit_tol or i == e74.C.fit_rounds - 1 or got <= 0:
                break
            eps *= (target / got) ** 0.5
        return {"eps": eps, "K_clean_fit": got, "K_clean_target": target}

    def own_output(self, u: Any) -> dict[str, Any]:
        """Read 1: W0's fixed masks, Δ off; the on-subcomponents' own output, ε of it, added at blocks 0 and 7."""
        blocks = json.loads((E74 / "results" / u.name / "blocks.json").read_text())["summary"]

        def read(rows: Tensor) -> dict[str, Any]:
            fit = {str(b): self.fit(rows, b, blocks[f"dw_{b}"]["K_clean"]) for b in BLOCKS}
            arms: dict[str, dict[str, list[float]]] = {f"own_{b}": {} for b in BLOCKS}
            for tokens in rows:
                t = tokens[None].to(DEV)
                clean, g = importances(self.model, t, self.gamma, self.comps)
                base = self.forward(t, g)
                k0 = pr.kl(clean, base)
                for b in BLOCKS:
                    eps = fit[str(b)]["eps"]
                    kc, a = shift(clean, self.own(t, g, eps, b))
                    km, q = shift(base, self.forward(t, own_masked(g, eps, b)))
                    add(arms[f"own_{b}"], {"K0": k0, "K_clean": kc, "K_masked": km, "aa": float((a * a).double().sum())} | numbers(a, q))
                    del a, q
                del clean, g, base
            return self.summed(rows, arms, {"fit": fit})

        return self.cached(u, "own_output", read)


def read_file(name: str, key: str) -> dict[str, Any] | None:
    path = RESULTS / name / f"{key}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--wall-min", type=float, required=True)
    a = ap.parse_args()
    if not PRED.is_file() or (RESULTS / "stopped.json").is_file():
        sys.exit("refusing to start: predictions.json missing, or stopped.json exists")
    job = pr.PassJob(a1.JobOpts(out=str(RESULTS / "unused"), e40_large=str(a1.E40_RESULTS / "large" / "nowarmup"), decomp=a.decomp,
                                results=str(RESULTS)), a1.CFG, lambda *_: {})
    rows = {"maths": job.rows("maths")[:, : pr.P.row_len], "a1": pr.a1_rows()}  # built first: a failure shows in the first minute

    def score(b: dict[str, Any]) -> dict[str, Any]:
        out = rules.score(read_file, tuple(POINTS), b)
        a1.write_json(RESULTS / "score.json", out)
        job.log(f"[score] {json.dumps(out, default=float)}")
        return out

    with torch.no_grad():
        job.gamma()
        rd = fr.FixedReader(job)
        updates = {u.name: u for u in pr.P.updates[:3]}
        for u in updates.values():
            pr.check_file(job, rd, u)
        S = Reads(job, rd, rows, a.wall_min * 60)
        u1, u3 = updates["u1"], updates["u3"]
        all_masked = [S.route(u1, m, "k196")["arms"]["whole"] for m in MODES]  # the control, alone
        S.route(u1, "fixed", "k0")
        S.control_t1(u1)
        b = rules.bands_from(*all_masked, S.displacement(u1)["sums"])
        job.log(f"[bands] {json.dumps(b)}")
        if any(math.isnan(x) for x in b["w"].values()):
            job.stop("bands", json.dumps(b))
        for u in (u1, u3):  # read 1
            S.own_output(u)
        if score(b)["own_output"]["overall"] == "fails":
            job.stop("own_output", "a change the masked forward computes itself did not pass: report before reading on")
        for u in (u1, u3):  # read 2
            for mode in MODES:
                for label in POINTS:
                    S.route(u, mode, label)
            score(b)
        for u in updates.values():  # read 3
            S.displacement(u)
        score(b)
    job.log(f"[done] {job.peaks()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    hard_exit(code)

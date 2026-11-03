"""E77 Amendment 1 ($0, one cluster L40). Read 1 failed on all four cells (run 50608, results/stopped_run1.json): a change the masked
forward computes itself barely reaches the logits. Is that the hook, or does the masked forward under fixed masks not read its
input? Plan: reports/plans/2026-11-03_plan_e77-open-points.md, Amendment 1; rules: results/predictions_amend1.json and below.
Amendment 2: job 50613 stopped on the first hook control, which compared two streams; the control is now on the clean stream (hook_read).
Order: (a) the hook control, stop on a miss; (b) the swap read; (c) own-output change at later blocks; (d) reads 2 and 3 of run.py
as filed, which read 1's stop cut off. run.py stays as it ran. Each read writes its file as it ends; a rerun skips it.

Usage (repo root; -P and PYTHONPATH as cluster/qwen_open_points_amend1.sbatch):
  PYTHONPATH=experiments/qwen_maps:experiments/qwen_pass_test:src .venv/bin/python -P experiments/qwen_open_points/amend1.py --decomp DIR --wall-min 110
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.jobs import hard_exit
from commute.qwen_vpd import importances, parse_site

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


op = _load("open_points_run", HERE / "run.py")  # the filed reads, untouched
rules, pr, a1, e74 = op.rules, op.pr, op.a1, op.e74
RESULTS, DEV, ROWS = op.RESULTS, op.DEV, op.ROWS
LAST_SITE, LATER = "layers.27.mlp.down_proj", (14, 21, 27)


def swap_label(follow_a: float, follow_b: float, b_a: list[float], b_b: list[float]) -> str:
    """The swap read: masks from row A on row B's tokens. Does the output sit nearer A (the masks carry it) or B (the input does)?
    Near means under half of the distance between the two clean outputs. A NaN or a value inside a band: no verdict."""
    sa, sb = rules.side(follow_a, b_a), rules.side(follow_b, b_b)
    if "unread" in (sa, sb):
        return "unread"
    if sa == "below" and sb == "above":
        return "masks"
    if sb == "below" and sa == "above":
        return "input"
    return "between"


def swap_read(S: Any, rows: Tensor) -> dict[str, Any]:
    """Row i's masks on row i + 1's tokens. Control: each row's own masks on its own tokens reproduce E74's K0."""
    sums: dict[str, list[float]] = {}
    for i, tokens in enumerate(rows):
        ta, tb = tokens[None].to(DEV), rows[(i + 1) % len(rows)][None].to(DEV)
        clean_a, g = importances(S.model, ta, S.gamma, S.comps)
        k0 = pr.kl(clean_a, S.forward(ta, g))
        clean_b = S.model(tb, use_cache=False).logits
        out = S.forward(tb, g)
        op.add(sums, {"K0": k0, "K_AL": pr.kl(clean_a, out), "K_BL": pr.kl(clean_b, out),
                      "K_AB": pr.kl(clean_a, clean_b), "K_BA": pr.kl(clean_b, clean_a)})
        del clean_a, clean_b, out, g
    fa, fb = sum(sums["K_AL"]) / sum(sums["K_AB"]), sum(sums["K_BL"]) / sum(sums["K_BA"])
    wa, wb = rules.jack(sums["K_AL"], sums["K_AB"]), rules.jack(sums["K_BL"], sums["K_BA"])
    ba, bb = rules.band(0.5, wa), rules.band(0.5, wb)
    return {"T": rows.shape[1], "sums": sums, "summary": {"follow_A": fa, "follow_B": fb, "w_A": wa, "w_B": wb, "label": swap_label(fa, fb, ba, bb)}}


def scaled(g: dict[str, Tensor], eps: float, on: Callable[[str], bool], clean: bool) -> tuple[Any, Any]:
    """As run.py's own_clean, own_masked, for any set of sites. Clean: m = 1 + ε g, d = 1 at `on`, 1, 1 elsewhere. Masked: (1 + ε) g, Δ off."""
    one = torch.ones((), device=DEV)
    if clean:
        return op.Lazy(lambda s: 1 + eps * g[s] if on(s) else one), op.Lazy(lambda s: one)
    return op.Lazy(lambda s: (1 + eps) * g[s] if on(s) else g[s]), None


def own_read(S: Any, rows: Tensor, on: Callable[[str], bool], target: float) -> dict[str, Any]:
    """run.py's read 1 for the sites in `on`: ε fitted by the same quadratic rule, then amplitude, τ, cosine under W0's masks."""
    eps, got = op.EPS0, 0.0
    for i in range(e74.C.fit_rounds):
        got = 0.0
        for tokens in rows:
            t = tokens[None].to(DEV)
            clean, g = importances(S.model, t, S.gamma, S.comps)
            G, D = scaled(g, eps, on, True)
            got += pr.kl(clean, S.forward(t, G, d=D)) / rows.numel()
            del clean, g
        S.job.log(f"[own fit] eps {eps:.6g}: clean KL {got:.6g}, target {target:.6g}")
        if abs(got / target - 1) <= e74.C.fit_tol or i == e74.C.fit_rounds - 1 or got <= 0:
            break
        eps *= (target / got) ** 0.5
    arm: dict[str, list[float]] = {}
    for tokens in rows:
        t = tokens[None].to(DEV)
        clean, g = importances(S.model, t, S.gamma, S.comps)
        base = S.forward(t, g)
        k0 = pr.kl(clean, base)
        G, D = scaled(g, eps, on, True)
        kc, a = op.shift(clean, S.forward(t, G, d=D))
        M, _ = scaled(g, eps, on, False)
        km, q = op.shift(base, S.forward(t, M))
        op.add(arm, {"K0": k0, "K_clean": kc, "K_masked": km, "aa": float((a * a).double().sum())} | op.numbers(a, q))
        del clean, g, base, a, q
    return S.summed(rows, {"own": arm}, {"fit": {"eps": eps, "K_clean_fit": got, "K_clean_target": target}})


def hook_read(S: Any, rows: Tensor, eps: float) -> dict[str, Any]:
    """Amendment 2's hook control: only the last matrix masked, every other site x Wᵀ (m = 1, d = 1), so the input the term reads is
    the clean model's. The own-output change there must then read as in the clean model."""
    one, zero = torch.ones((), device=DEV), torch.zeros((), device=DEV)

    def alone(g: dict[str, Tensor], e: float) -> tuple[Any, Any]:
        return op.Lazy(lambda s: (1 + e) * g[s] if s == LAST_SITE else one), op.Lazy(lambda s: zero if s == LAST_SITE else one)

    arm: dict[str, list[float]] = {}
    for tokens in rows:
        t = tokens[None].to(DEV)
        clean, g = importances(S.model, t, S.gamma, S.comps)
        G, D = alone(g, 0.0)
        base = S.forward(t, G, d=D)
        k0 = pr.kl(clean, base)
        G, D = scaled(g, eps, lambda s: s == LAST_SITE, True)
        kc, a = op.shift(clean, S.forward(t, G, d=D))
        G, D = alone(g, eps)
        km, q = op.shift(base, S.forward(t, G, d=D))
        op.add(arm, {"K0": k0, "K_clean": kc, "K_masked": km, "aa": float((a * a).double().sum())} | op.numbers(a, q))
        del clean, g, base, a, q
    return S.summed(rows, {"own": arm}, {"eps": eps})


def bands_now() -> dict[str, Any]:
    """run.py's no-call bands, from the first stage's files that run 50608 wrote."""
    all_masked = [op.read_file("u1", f"route_{m}_k196")["arms"]["whole"] for m in op.MODES]  # type: ignore[index]
    return rules.bands_from(*all_masked, op.read_file("u1", "displacement")["sums"])  # type: ignore[index]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--wall-min", type=float, required=True)
    a = ap.parse_args()
    if not (RESULTS / "predictions_amend2.json").is_file() or (RESULTS / "stopped.json").is_file():
        raise SystemExit("refusing to start: predictions_amend2.json missing, or stopped.json exists")
    job = pr.PassJob(a1.JobOpts(out=str(RESULTS / "unused"), e40_large=str(a1.E40_RESULTS / "large" / "nowarmup"), decomp=a.decomp,
                                results=str(RESULTS)), a1.CFG, lambda *_: {})
    rows = {"maths": job.rows("maths")[:, : pr.P.row_len], "a1": pr.a1_rows()}
    with torch.no_grad():
        job.gamma()
        rd = op.fr.FixedReader(job)
        updates = {u.name: u for u in pr.P.updates[:3]}
        for u in updates.values():
            pr.check_file(job, rd, u)
        S = op.Reads(job, rd, rows, a.wall_min * 60)
        u1, u3 = updates["u1"], updates["u3"]
        blocks = json.loads((op.E74 / "results" / "u1" / "blocks.json").read_text())["summary"]
        b = bands_now()

        last = S.cached(u1, "amend1_own_last", lambda r: own_read(S, r, lambda s: s == LAST_SITE, blocks["dw_27"]["K_clean"]))
        s = last["summary"]["own"]
        job.log(f"[last matrix, masked stream] {json.dumps(s)}")
        hook = S.cached(u1, "amend2_hook", lambda r: hook_read(S, r, last["fit"]["eps"]))["summary"]["own"]
        job.log(f"[control last matrix, clean stream] {json.dumps(hook)}")
        if not hook["amp"] > b["own"][1] or not hook["cos"] > b["own"][1]:
            job.stop("hook", f"own-output change at the last matrix, the only site masked, clean stream: {hook}")

        verdict: dict[str, Any] = {"own_last": s, "hook": hook, "bands": b}
        for u in (u1, u3):
            r = S.cached(u, "amend1_swap", lambda rw: swap_read(S, rw))
            total = sum(r["sums"]["K0"])
            want = sum(json.loads((op.E74 / "results" / u.name / "all.json").read_text())["base"]["K0"])
            job.log(f"[control swap {u.name}] own masks K0 {total:.9g} vs E74 {want:.9g}")
            if not rules.close(total, want):
                job.stop("swap_control", f"{u.name}: {total} vs {want}")
            verdict[f"swap_{u.name}"] = r["summary"]
        for block in LATER:
            r = S.cached(u1, f"amend1_own_{block}", lambda rw, bl=block: own_read(S, rw, lambda st: parse_site(st)[0] == bl, blocks[f"dw_{bl}"]["K_clean"]))
            verdict[f"own_{block}"] = r["summary"]["own"]
        a1.write_json(RESULTS / "amend1.json", verdict)
        job.log(f"[amend1] {json.dumps(verdict, default=float)}")

        def score() -> None:  # score.json stays as run 50608 left it
            out = rules.score(op.read_file, tuple(op.POINTS), b)
            a1.write_json(RESULTS / "amend1_score.json", out)
            job.log(f"[score] {json.dumps(out, default=float)}")

        for u in (u1, u3):  # read 2, as filed
            for mode in op.MODES:
                for label in op.POINTS:
                    S.route(u, mode, label)
            score()
        for u in updates.values():  # read 3, as filed
            S.displacement(u)
        score()
    job.log(f"[done] {job.peaks()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    hard_exit(code)

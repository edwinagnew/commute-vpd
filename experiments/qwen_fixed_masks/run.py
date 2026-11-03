"""E72, a second test for D1 ($0, one cluster L40, normal QoS). Plan: reports/plans/2026-11-03_plan_e72-d1-second-test.md;
rules: results/predictions.json (the run refuses to start without it, and while results/stopped.json exists), scored in
rules.py. E68's reader, stage S and files (experiments/qwen_pass_test, read, never written).

Fixed masks are recomputed, not stored: per row, FT in the model, Γ with W0's V. A state's V′ sits in the components in
place (`qwen_maps.applied`), so `OriginalV` rebuilds one site's V at a time from the host's alive columns for the readouts.
Stored, g is 506 MB per row in fp32 on the alive columns (494598 x 256), 65 GB for 128 rows, 108 GB beside the reader's 43 GB
under a 110 GB limit; bf16 (32 GB) rounds g by 0.4% and alive-only drops the rest, which the control's 1e-3 may not survive.

Per update, in order: do_nothing (control_reproduce, stop); stage S; on u1, u2 T0a and T0b; ceiling (room: u1 stops);
upper, lower (room again, with w); random, ls, adjoint; the clean reads. Each read writes results/<update>/<read>.json as it
ends (a rerun skips it) and rescores results/score.json; every number goes to results/run.log. A stop: results/stopped.json,
exit 1. Past --wall-min: exit 0, resubmit to resume. An update's stage S files on --out go when its reads finish.

Usage (repo root; `-P` and importlib as qwen_pass_test/amend1.py: `run` must mean qwen_maps/run.py for amend2.py, and `rules`
qwen_pass_test/rules.py for its run.py, so neither of this folder's files is on the path):
  PYTHONPATH=experiments/qwen_maps:experiments/qwen_pass_test:src .venv/bin/python -P experiments/qwen_fixed_masks/run.py \
    --decomp DIR --out /ephemeral/$USER/e72 --wall-min 340
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import split
from commute.jobs import hard_exit
from commute.qwen_vpd import SITES, Components, importances, masked_logits

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
E68 = HERE.parent / "qwen_pass_test"
T0_UPDATES, F64 = ("u1", "u2"), torch.float64


def load_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rules = load_file("fixed_rules", HERE / "rules.py")
a1 = load_file("pass_amend1", E68 / "amend1.py")  # forward_plus; it loads E68's run.py as `pass_run`
pr = a1.pr
DEV = "cuda"
UPDATES = pr.P.updates[:3]


class OriginalV(Components):
    """What Γ's readouts read: W0's V of one site at a time, whatever the state has done to the components in place."""

    def __init__(self, comps: Components, alive: Any):
        super().__init__(comps.V, comps.U)
        self.alive = alive

    def vu(self, site: str) -> tuple[Tensor, Tensor]:
        V, U = super().vu(site)
        O = V.clone()
        O[:, self.alive.idx[site]] = self.alive.V[site].to(V.device)
        return O, U


def numbers(ref: Tensor, q: Tensor, tokens: Tensor) -> dict[str, float]:
    return {"K": pr.kl(ref, q), "nll": pr.nll(q, tokens), "ft": pr.nll(ref, tokens),
            "agree": float((ref.argmax(-1) == q.argmax(-1)).sum()), "count": float(tokens.numel())}


def entropy(logits: Tensor) -> float:
    lp = logits.float().log_softmax(-1)
    return float(-(lp.exp() * lp).sum(-1).double().sum())


class FixedReader(pr.PassReader):
    """`masked` is T1 through amend2's `read` (the state applied; this `row`). The rest loop over the rows themselves."""

    moved = False  # the state's components differ from W0's
    plus: dict[str, Tensor] | None = None  # the ceiling's δW

    def masked(self, name: str, t: int, scale: float = 1.0, plus: dict[str, Tensor] | None = None) -> dict[str, Any]:
        self.moved, self.plus = name != "do_nothing", plus
        return self.read(name, t, "maths", {"K"}, scale=scale)

    def row(self, tokens: Tensor, own_key: str, want: set[str], srcs: dict[str, dict[str, Tensor]]) -> dict[str, float]:
        job, comps = self.job, self.job.comps()
        self.use("FT")  # Δ is off, so the masked forward does not read the weights
        ref, g = importances(job.model, tokens, job.gamma(), OriginalV(comps, job.alive) if self.moved else comps)
        q = (masked_logits(job.model, tokens, comps, g) if self.plus is None
             else a1.forward_plus(job.model, tokens, comps, g, self.plus, 1.0))
        return numbers(ref, q, tokens) | {"g_sum": float(sum(v.double().sum() for v in g.values()))}

    def part(self, name: str, t: int, a: float, s: str) -> Tensor | None:
        """One site's weight change of a T2 part (float64, GPU): plan, T2; predictions.json, "T2 parts"."""
        job = self.job
        M = job.dw(t)[s].to(DEV, F64)
        if name == "W0":
            return None
        if name == "FT":
            return M
        if name == "remainder":
            return M - job.change("inspan", t)(s)[0]
        if name in ("upper", "lower"):
            _, pos, dU, dV = job.change(name, t)(s)
            U_A, V_A = job.alive.uv(s, DEV, F64)
            return split.dsigma(dU, dV, U_A[:, pos], V_A[:, pos])
        return job.change(name, t, scale=a)(s)[0]  # inspan; random's whole change at E68's scale

    def state_set(self, part: Callable[[str], Tensor | None]) -> None:
        """W0 + part, fp32 on the pinned host, as `qwen_maps.applied` builds a state's weights."""
        pinned = sys.modules["amend2"].pinned
        assert self.now == "W0"
        self.sets["state"] = {}
        for s, lin in self.lins.items():
            p = part(s)
            self.sets["state"][s] = pinned(lin.weight.data if p is None else lin.weight.data + p.to(lin.weight.data))

    def loop(self, label: str, fn: Callable[[Tensor], dict[str, float]]) -> dict[str, Any]:
        sums: dict[str, list[float]] = {}
        t0 = time.time()
        for tokens in self.rows["maths"]:
            for k, v in fn(tokens[None].to(DEV)).items():
                sums.setdefault(k, []).append(v)
        self.use("W0")
        self.sets.pop("state", None)
        rd = {"T": pr.P.row_len, "sums": sums}
        self.job.log(f"[{label}] " + " ".join(f"{k} {a1.mean_of(rd, k):.9g}" for k in sums)
                     + f"  rows {len(sums['K' if 'K' in sums else 'K_R0'])}, {time.time() - t0:.0f} s, {self.job.peaks()}")
        return rd

    def clean(self, name: str, t: int, a: float) -> dict[str, Any]:
        self.set_ft(t)
        self.state_set(lambda s: self.part(name, t, a, s))
        model = self.job.model

        def row(tokens: Tensor) -> dict[str, float]:
            self.use("FT")
            ref = model(tokens, use_cache=False).logits
            self.use("state")
            q = model(tokens, use_cache=False).logits
            return numbers(ref, q, tokens) | ({"entropy": entropy(q)} if name in ("W0", "FT") else {})

        return self.loop(f"clean {name} {t}", row)

    def split_check(self, t: int) -> None:
        """Rounding of the two weight sets: ‖(W0 + in-span) + (W0 + remainder) − W0 − FT‖ / ‖δW‖, fp32 weights."""
        err, norm = 0.0, 0.0
        for s, lin in self.lins.items():
            w0, M = lin.weight.data, self.job.dw(t)[s].to(DEV, F64)
            ins = self.job.change("inspan", t)(s)[0]
            off = (w0 + ins.to(w0)).double() + (w0 + (M - ins).to(w0)).double() - w0.double() - (w0 + M.to(w0)).double()
            err, norm = err + float(off.norm()) ** 2, norm + float(M.norm()) ** 2
        self.job.log(f"[T2 split {t}] in-span + remainder vs δW: {(err / norm) ** 0.5:.6g} of ‖δW‖")

    def t0a(self, t: int, a: float) -> dict[str, Any]:
        job, comps, model = self.job, self.job.comps(), self.job.model
        change = job.change("random", t, scale=a)
        R = {s: change(s)[0].float().cpu() for s in SITES}  # E68's random state's whole change; host, moved per site
        self.state_set(R.get)

        def row(tokens: Tensor) -> dict[str, float]:
            self.use("state")
            ref = model(tokens, use_cache=False).logits
            self.use("W0")
            logits, g = importances(model, tokens, job.gamma(), comps)
            q0 = masked_logits(model, tokens, comps, g)
            out = {"K_R0": pr.kl(ref, q0), "ci": pr.kl(logits, q0), "K_clean": pr.kl(ref, logits)}
            del q0, logits
            return out | {"K_R1": pr.kl(ref, a1.forward_plus(model, tokens, comps, g, R, 1.0))}

        return self.loop(f"T0a {t}", row)

    def t0b(self, t: int) -> dict[str, Any]:
        job, comps, model = self.job, self.job.comps(), self.job.model
        self.set_ft(t)

        def row(tokens: Tensor) -> dict[str, float]:
            self.use("W0")
            g = importances(model, tokens, job.gamma(), comps)[1]
            self.use("FT")  # d = 1 reads the weights: Δ₀ + δW
            ref = model(tokens, use_cache=False).logits
            d = torch.ones(tokens.shape, device=tokens.device)
            return numbers(ref, masked_logits(model, tokens, comps, g, dict.fromkeys(SITES, d)), tokens)

        return self.loop(f"T0b {t}", row)


class Runner(pr.Runner):
    """E68's Runner for `step`, `wall` and `stage_s`; these results, this order, these rules."""

    def __init__(self, job: Any, rd: FixedReader, res: dict[str, dict], root: Path, wall_s: float):
        super().__init__(job, rd, res, root, wall_s)
        self.w0 = {u.name: json.loads((pr.RESULTS / u.name / "W0.json").read_text()) for u in UPDATES}
        self.a = {u.name: json.loads((pr.RESULTS / u.name / "search.json").read_text())["a"] for u in UPDATES}
        self.e68 = json.loads((pr.RESULTS / "amend1" / "score.json").read_text())
        self.ref = rules.reference(self.w0["u1"], json.loads((pr.RESULTS / "u1" / "A_do_nothing.json").read_text()))

    def rescore(self) -> None:
        sc = rules.score(self.res, self.w0, self.ref, self.e68)
        a1.write_json(RESULTS / "score.json", sc)
        self.job.log(f"[score] {json.dumps(sc, default=float)}")
        if sc["control_reproduce"]["missed"]:
            self.job.stop("control_reproduce", json.dumps(sc["control_reproduce"], default=float))
        if sc["u1"]["T1"]["room"]["result"] in ("no room", "unreadable"):
            self.job.stop("room", json.dumps(sc["u1"]["T1"]["room"], default=float))
        for name in self.res:
            if sc[name]["T1"].get("fixed_g", {}).get("missed"):
                self.job.stop("fixed_g", f"{name}: {json.dumps(sc[name]['T1']['fixed_g'], default=float)}")
            if sc[name]["T2"].get("clean_FT", {}).get("missed"):
                self.job.stop("clean_FT", f"{name}: {json.dumps(sc[name]['T2']['clean_FT'], default=float)}")

    def put(self, u: Any, key: str, value: dict[str, Any]) -> None:
        self.res[u.name][key] = value
        a1.write_json(RESULTS / u.name / f"{key}.json", value)
        self.job.log(f"[read] {u.name}/{key} written")
        self.rescore()

    def update(self, u: Any, rows: Tensor) -> None:
        job, rd, t, a = self.job, self.rd, u.step, self.a[u.name]
        job.select(u, self.root)
        (RESULTS / u.name).mkdir(parents=True, exist_ok=True)
        rd.rows["maths"], rd.w0_logits = rows, []  # "maths" is the reader's one block: this update's rows
        rd.drop_ft()
        self.step(u, "do_nothing", lambda: rd.masked("do_nothing", t))
        later: list[tuple[str, Callable[[], dict[str, Any]]]] = []
        if u.name in T0_UPDATES:
            later += [("T0a", lambda: rd.t0a(t, a)), ("T0b", lambda: rd.t0b(t))]
        later += [("ceiling", lambda: rd.masked("do_nothing", t, plus=job.dw(t))),
                  ("upper", lambda: rd.masked("upper", t)), ("lower", lambda: rd.masked("lower", t)),
                  ("random", lambda: rd.masked("random", t, scale=a)),
                  ("ls", lambda: rd.masked("ls", t)), ("adjoint", lambda: rd.masked("adjoint", t))]
        later += [(f"clean_{n}", lambda n=n: rd.clean(n, t, a)) for n in rules.T2]
        if any(key not in self.res[u.name] for key, _ in later):
            self.stage_s(u)
        if u.name == "u1" and "clean_remainder" not in self.res[u.name]:
            rd.split_check(t)
        for key, fn in later:
            self.step(u, key, fn)
        shutil.rmtree(job.out, ignore_errors=True)
        job.log(f"[{u.name}] done; {job.out} removed, {job.peaks()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--out", required=True)  # stage S's files
    ap.add_argument("--wall-min", type=float, required=True)  # past this, between reads, exit 0
    a = ap.parse_args()
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if (RESULTS / "stopped.json").is_file():
        sys.exit(f"refusing to start: {RESULTS / 'stopped.json'} exists (a stop rule fired; the lead decides what next)")
    job = pr.PassJob(a1.JobOpts(out=a.out, e40_large=str(a1.E40_RESULTS / "large" / "nowarmup"), decomp=a.decomp,
                                results=str(RESULTS)), a1.CFG, lambda *_: {})
    res = {u.name: {p.stem: json.loads(p.read_text()) for p in (RESULTS / u.name).glob("*.json")} for u in UPDATES}
    job.log(f"[config] updates {[u.name for u in UPDATES]}; resumed { {k: sorted(v) for k, v in res.items()} }; args {vars(a)}")
    rows = {"maths": job.rows("maths")[:, : pr.P.row_len], "a1": pr.a1_rows()}  # built first: a failure shows in the first minute
    with torch.no_grad():
        job.gamma()
        assert job.alive.count() == a1.CFG.alive_total  # its host copy of W0's columns, taken while the components are W0's
        rd = FixedReader(job)
        runner = Runner(job, rd, res, Path(a.out), a.wall_min * 60)
        runner.rescore()  # a resume obeys the stops too
        for u in UPDATES:
            assert tuple(rows[u.rows].shape) == (u.n_rows, pr.P.row_len), f"{u.name}: rows {rows[u.rows].shape}"
            pr.check_file(job, rd, u)  # every file before any long stage
        for u in UPDATES:
            runner.update(u, rows[u.rows])
    job.log(f"[done] {job.peaks()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    hard_exit(code)

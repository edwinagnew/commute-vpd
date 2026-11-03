"""E75, do the D1 reads hold on an update with a real accuracy gain? ($0, one cluster L40, normal QoS.) Plan:
reports/plans/2026-11-03_plan_e75-rl-reads.md; rules: rules.py, results/predictions.json (the run refuses to start without
it, and while results/stopped.json exists). Nothing is measured here that E72 or E74 did not define: FixedReader.clean is E72's
T2, E74's read_group with one arm is its whole-update transmission, E68's Runner gives stage S and its search. The shared code
cannot move into src/commute while job 50441 imports from this clone; it should later.

First, alone, needing no stage S: the control on u1 (E72's clean_W0; E74's fixed and stochastic reads, same seed), stop on a
miss. Then per update of --updates, in order, cheapest first: clean_W0, clean_FT, fixed, stochastic; stage S; search; the rest
of T2 and the adjoint split's clean read. Each read writes results/<update>/<read>.json as it ends (a rerun skips it) and
rescores results/score.json; every number goes to results/run.log. A stop: results/stopped.json, exit 1. Past --wall-min, between
reads: exit 0, resubmit to resume. An update's stage S files on --out go when its reads finish.

Usage (repo root; -P and PYTHONPATH as cluster/qwen_rl_reads.sbatch, so `run` means qwen_maps/run.py for amend2.py):
  PYTHONPATH=experiments/qwen_maps:experiments/qwen_pass_test:src .venv/bin/python -P experiments/qwen_rl_reads/run.py \
    --decomp DIR --out /ephemeral/$USER/e75 --wall-min 340 --updates R100,R10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.jobs import hard_exit
from commute.qwen_e40_files import sums
from commute.qwen_sft import load, sha256
from commute.qwen_vpd import SITES

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
E72, E74 = HERE.parent / "qwen_fixed_masks", HERE.parent / "qwen_transmission"
RL_ROOT = Path.home() / "commute-data" / "results-large" / "qwen_rl" / "lr1e-6w10"
RL_SUMS = HERE.parent / "qwen_rl" / "results" / "lr1e-6w10" / "SHA256SUMS"  # R's sums; N's and F's arrive with a pull
T2_LATER = ("inspan", "remainder", "upper", "lower", "random", "adjoint")  # after stage S and the search


def load_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rules = load_file("rl_reads_rules", HERE / "rules.py")
fr = load_file("fixed_run", E72 / "run.py")  # FixedReader; its pr (E68's run.py) and a1
e74a2 = load_file("transmission_amend2", E74 / "amend2.py")  # never named "amend2": FixedReader finds qwen_maps's under that name
pr, a1, e74 = fr.pr, fr.a1, e74a2.run  # e74: read_group and summary only; its log, write and stop write to E74's results


def parse(spec: str) -> list[Any]:
    """"R100,R10": arm and step of RL's weight changes, E68's Update on the maths rows."""
    out = []
    for name in spec.split(","):
        m = re.fullmatch(r"([RNF])(\d+)", name)
        if m is None:
            sys.exit(f"--updates: {name!r} is not an arm R, N or F and a step")
        out.append(pr.Update(name, RL_ROOT / "dW" / m[1] / f"step_{m[2]}.safetensors", int(m[2]), "maths", 128))
    return out


def check_file(job: Any, rd: Any, u: Any) -> None:
    """E68's check_file for fp32: the file named, δW of exactly the 196 sites' shapes, small against W0."""
    key = str(u.path.relative_to(RL_ROOT))
    got, want = sha256(u.path) if u.path.is_file() else "absent", sums(RL_SUMS).get(key)
    job.log(f"[dW {u.name}] {u.path}: sha256 {got}, want {want}")
    if got == "absent" or (want is not None and got != want):
        job.stop("dW_file", f"{u.path}: sha256 {got} != {want}")
    dW, W0 = load(u.path), rd.sets["W0"]
    if set(dW) != set(SITES) or any(dW[s].dtype != torch.float32 or dW[s].shape != W0[s].shape for s in SITES):
        job.stop("dW_file", f"{u.path}: not fp32 tensors of the 196 sites' shapes")
    rel = math.sqrt(sum(float(dW[s].norm()) ** 2 for s in SITES) / sum(float(W0[s].norm()) ** 2 for s in SITES))
    job.log(f"[dW {u.name}] ‖δW‖/‖W0‖ {rel:.6g}")
    if rel > pr.P.dw_rel_max:
        job.stop("dW_file", f"{u.path}: ‖file‖/‖W0‖ = {rel:.3g} > {pr.P.dw_rel_max}: this is not a δW")


class Runner(pr.Runner):
    """E68's Runner for `step`, `wall` and `stage_s`; these results, this order, these rules."""

    def __init__(self, job: Any, rd: Any, res: dict[str, dict], root: Path, wall_s: float, rows: Tensor):
        super().__init__(job, rd, res, root, wall_s)
        self.rows = rows
        stored = {"clean_W0": E72 / "results" / "u1" / "clean_W0.json", "fixed": E74 / "results" / "u1" / "all.json",
                  "stochastic": E74 / "results" / "u1" / "stochastic.json"}
        self.ref = rules.readings({k: json.loads(p.read_text()) for k, p in stored.items()})

    def rescore(self) -> None:
        sc = rules.score(self.res, self.ref)
        a1.write_json(RESULTS / "score.json", sc)
        self.job.log(f"[score] {json.dumps(sc, default=float)}")
        if sc["control"]["missed"]:
            self.job.stop("control", json.dumps(sc["control"], default=float))
        for name, s in sc.items():
            if name != "control" and s["w0_same"]["missed"]:
                self.job.stop("w0_same", f"{name}: {json.dumps(s['w0_same'], default=float)}")
            if name != "control" and s["clean"].get("clean_FT", {}).get("missed"):
                self.job.stop("clean_FT", f"{name}: {json.dumps(s['clean']['clean_FT'], default=float)}")

    def put(self, u: Any, key: str, value: dict[str, Any]) -> None:
        self.res[u.name][key] = value
        a1.write_json(RESULTS / u.name / f"{key}.json", value)
        self.job.log(f"[read] {u.name}/{key} written")
        self.rescore()

    def whole(self, u: Any, stochastic: bool) -> dict[str, Any]:
        """E74's whole-update arm under W0's masks: `all` in run.py, and amend2's stochastic with its seed (a draw per row, made
        before the row's arms, so one arm gets the draws its fourteen had). The term stays on the host in fp32, as E74's."""
        job, t0 = self.job, time.time()
        dw = {s: v.float() for s, v in job.dw(u.step).items()}
        gen = torch.Generator(fr.DEV).manual_seed(e74a2.STOCH_SEED) if stochastic else None
        r = e74.read_group(pr, job.model, job.gamma(), job.comps(), self.rows, {"all": e74.Arm(dw)}, gen)
        r["summary"] = e74.summary(r)
        job.log(f"[{u.name} {'stochastic' if stochastic else 'fixed'}] K0 {sum(r['base']['K0']) / (len(r['base']['K0']) * r['T']):.9g} "
                f"{json.dumps(r['summary'])} {time.time() - t0:.0f} s, {job.peaks()}")
        return r

    def cheap(self, u: Any) -> dict[str, Any]:
        """What needs no stage S. On u1 this is the control, and only clean_FT is left out."""
        job, rd, t = self.job, self.rd, u.step
        job.select(u, self.root)
        (RESULTS / u.name).mkdir(parents=True, exist_ok=True)
        rd.w0_logits = []
        rd.drop_ft()
        d = self.res.setdefault(u.name, {})
        self.step(u, "clean_W0", lambda: rd.clean("W0", t, 1.0))
        if u.name != "u1":
            self.step(u, "clean_FT", lambda: rd.clean("FT", t, 1.0))
        self.step(u, "fixed", lambda: self.whole(u, False))
        self.step(u, "stochastic", lambda: self.whole(u, True))
        return d

    def update(self, u: Any) -> None:
        job, rd, t = self.job, self.rd, u.step
        d = self.cheap(u)
        later: list[tuple[str, Any]] = [("search", lambda: rd.search(t))]
        later += [(f"clean_{n}", lambda n=n: rd.clean(n, t, d["search"]["a"])) for n in T2_LATER]
        if any(key not in d for key, _ in later):
            self.stage_s(u)
            if "clean_remainder" not in d:
                rd.split_check(t)  # the rounding of W0 + part in fp32, on a smaller δW than E72's
        for key, fn in later:
            self.step(u, key, fn)
        shutil.rmtree(job.out, ignore_errors=True)
        job.log(f"[{u.name}] done; {job.out} removed, {job.peaks()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--out", required=True)  # stage S's files
    ap.add_argument("--wall-min", type=float, required=True)  # past this, between reads, exit 0
    ap.add_argument("--updates", required=True, type=parse)  # e.g. R100,R10
    a = ap.parse_args()
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if (RESULTS / "stopped.json").is_file():
        sys.exit(f"refusing to start: {RESULTS / 'stopped.json'} exists (a stop rule fired; the lead decides what next)")
    job = pr.PassJob(a1.JobOpts(out=a.out, e40_large=str(a1.E40_RESULTS / "large" / "nowarmup"), decomp=a.decomp,
                                results=str(RESULTS)), a1.CFG, lambda *_: {})
    u1 = pr.P.updates[0]
    res = {u.name: {p.stem: json.loads(p.read_text()) for p in (RESULTS / u.name).glob("*.json")} for u in (u1, *a.updates)}
    job.log(f"[config] updates {[u.name for u in a.updates]}; resumed { {k: sorted(v) for k, v in res.items()} }; args {vars(a)}")
    rows = job.rows("maths")[:, : pr.P.row_len]
    assert tuple(rows.shape) == (u1.n_rows, pr.P.row_len), f"rows {rows.shape}"
    with torch.no_grad():
        job.gamma()
        assert job.alive.count() == a1.CFG.alive_total  # its host copy of W0's columns, taken while the components are W0's
        rd = fr.FixedReader(job)
        runner = Runner(job, rd, res, Path(a.out), a.wall_min * 60, rows)
        runner.rescore()  # a resume obeys the stops too
        pr.check_file(job, rd, u1)
        for u in a.updates:
            check_file(job, rd, u)  # every file before any long stage
        runner.cheap(u1)
        for u in a.updates:
            runner.update(u)
    job.log(f"[done] {job.peaks()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    hard_exit(code)

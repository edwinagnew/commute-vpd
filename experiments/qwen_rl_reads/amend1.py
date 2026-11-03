"""E75 Amendment 1, three bad placements of the right size ($0, one cluster L40, debug QoS). Plan:
reports/plans/2026-11-03_plan_e75-rl-reads.md, "Amendment 1"; rules: rules_amend1.py, results/predictions_amend1.json (the run
refuses to start without it, and while results/stopped_amend1.json exists). Does the clean-model table fail a wrong direction at
δW's size? No fitting, no Σ, no stage S, no masks: only E72's clean read (FixedReader.clean), so neither Γ nor the components load.

Each state is W0 + a change, per matrix at δW's Frobenius norm: random (Gaussian), other (another update's matrix), signs (δW times
a sign per matrix). First, alone, on u1: W0 and W0 + δW by this path, stop on a miss. Then each state on each update. A state's K
is set against the update's remainder K, read from E75's and E72's committed clean_remainder.json.

Writes results/amend1/<update>/<read>.json as each read ends (a rerun skips it), results/amend1_score.json, results/amend1/run.log;
on a stop results/stopped_amend1.json, exit 1. Past --wall-min, between reads: exit 0, resubmit to resume. It writes nothing that
E75's run.py writes, so the two can run together.

Usage (repo root; -P and PYTHONPATH as cluster/qwen_rl_reads_amend1.sbatch):
  PYTHONPATH=experiments/qwen_maps:experiments/qwen_pass_test:src .venv/bin/python -P experiments/qwen_rl_reads/amend1.py \
    --wall-min 40
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

from commute.jobs import hard_exit
from commute.qwen_vpd import SITES

HERE = Path(__file__).resolve().parent
RESULTS, OUT = HERE / "results", HERE / "results" / "amend1"
PRED, SCORE, STOPPED = RESULTS / "predictions_amend1.json", RESULTS / "amend1_score.json", RESULTS / "stopped_amend1.json"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load("rl_reads_run", HERE / "run.py")  # its reader's parents, update parsing and file checks; never its writers
rules = _load("rl_reads_rules_amend1", HERE / "rules_amend1.py")
fr, pr, a1, DEV, F64 = run.fr, run.pr, run.a1, run.fr.DEV, run.F64

OTHER = {"u1": "u3", "u2": "u3", "u3": "u1", "R100": "u1", "R10": "u1"}  # plan's order
UPDATES, SEED = tuple(OTHER), 75_100
SIGNS = dict(zip(SITES, random.Random(SEED).choices((1.0, -1.0), k=len(SITES)), strict=True))


def at_norm(A: Tensor, M: Tensor) -> Tensor:
    """A at M's Frobenius norm. A zero A has no direction to scale: stop rather than read a smaller state."""
    n = A.norm()
    assert n > 0, "a bad placement's matrix is exactly zero"
    return A * (M.norm() / n)


def gaussian(M: Tensor, site_index: int) -> Tensor:
    gen = torch.Generator(M.device).manual_seed(SEED + site_index)
    return at_norm(torch.randn(M.shape, generator=gen, device=M.device, dtype=M.dtype), M)


def other_matrix(path: Path, s: str, M: Tensor) -> Tensor:
    """One matrix of another update's file, at M's norm; read per matrix, so the host never holds that whole file."""
    with safe_open(str(path), framework="pt") as f:
        return at_norm(f.get_tensor(s).to(M), M)


class Job(pr.PassJob):
    """PassJob with its stop file beside E75's; `dir` (the log) is results/amend1."""

    def stop(self, name: str, why: str) -> None:
        a1.write_json(STOPPED, {"name": name, "why": why, "stage": self.stage, "at_s": time.time() - self.t0})
        self.log(f"STOP {name}: {why}")
        sys.exit(f"STOP {name}: {why}")


class Reader(fr.FixedReader):
    """E72's clean read; `part` also gives the three states. `other` is the current update's other file."""

    other: Path

    def part(self, name: str, t: int, a: float, s: str) -> Tensor | None:
        if name not in rules.STATES:
            return super().part(name, t, a, s)  # W0, FT: the control
        M = self.job.dw(t)[s].to(DEV, F64)
        if name == "random":
            return gaussian(M, SITES.index(s))
        return other_matrix(self.other, s, M) if name == "other" else M * SIGNS[s]


class Runner(pr.Runner):
    """E68's Runner for `step` and `wall`; these files, this rule."""

    def __init__(self, job: Any, rd: Reader, res: dict[str, dict], wall_s: float, ref: dict[str, dict], rows: dict[str, Tensor],
                 updates: dict[str, Any]):
        super().__init__(job, rd, res, OUT, wall_s)
        self.ref, self.rows, self.updates = ref, rows, updates

    def rescore(self) -> None:
        sc = rules.score(self.res, self.ref, UPDATES)
        a1.write_json(SCORE, sc)
        self.job.log(f"[score] {json.dumps(sc, default=float)}")
        if sc["control"]["missed"]:
            self.job.stop("control", json.dumps(sc["control"], default=float))

    def put(self, u: Any, key: str, value: dict[str, Any]) -> None:
        self.res[u.name][key] = value
        a1.write_json(OUT / u.name / f"{key}.json", value)
        self.job.log(f"[read] {u.name}/{key} written")
        self.rescore()

    def select(self, u: Any) -> None:
        """One update's δW on the host; the last update's δW and FT freed first (u1 and R10 share a step, so drop_ft always)."""
        self.job.select(u, OUT)
        (OUT / u.name).mkdir(parents=True, exist_ok=True)
        self.rd.drop_ft()
        self.rd.rows["maths"], self.rd.w0_logits = self.rows[u.rows], []  # "maths" is the reader's one block: this update's rows
        self.rd.other = self.updates[OTHER[u.name]].path

    def control(self, u: Any) -> None:
        self.select(u)
        for name in ("W0", "FT"):
            self.step(u, f"clean_{name}", lambda name=name: self.rd.clean(name, u.step, 1.0))

    def update(self, u: Any) -> None:
        self.select(u)
        for name in rules.STATES:
            self.step(u, name, lambda name=name: self.rd.clean(name, u.step, 1.0))
        self.job.log(f"[{u.name}] done, {self.job.peaks()}")


def prior(name: str, read: str) -> dict[str, Any]:
    """A committed read: E75's file, or E72's for the saved updates, which E75 did not read again."""
    for root in (RESULTS, run.E72 / "results"):
        if (p := root / name / f"{read}.json").is_file():
            return json.loads(p.read_text())
    sys.exit(f"refusing to start: no {read}.json for {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-min", type=float, required=True)  # past this, between reads, exit 0
    a = ap.parse_args()
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if STOPPED.is_file():
        sys.exit(f"refusing to start: {STOPPED} exists (a stop rule fired; the lead decides what next)")
    updates = {u.name: u for u in run.parse(",".join(UPDATES))}
    ref = {n: {k: prior(n, k) for k in ("clean_W0", "clean_remainder")} for n in UPDATES}
    job = Job(a1.JobOpts(out=str(OUT), e40_large=str(a1.E40_RESULTS / "large" / "nowarmup"), decomp="", results=str(OUT)),
              a1.CFG, lambda *_: {})
    res = {n: {p.stem: json.loads(p.read_text()) for p in (OUT / n).glob("*.json")} for n in UPDATES}
    job.log(f"[config] holds { {k: sorted(v) for k, v in res.items()} }; signs flipped {sum(x < 0 for x in SIGNS.values())} of {len(SITES)}; args {vars(a)}")
    rows = {"maths": job.rows("maths")[:, : pr.P.row_len], "a1": pr.a1_rows()}  # built first: a failure shows in the first minute
    for u in updates.values():
        assert tuple(rows[u.rows].shape) == (u.n_rows, pr.P.row_len), f"{u.name}: rows {rows[u.rows].shape}"
        assert rules.n_rows(ref[u.name]["clean_remainder"]) == u.n_rows, f"{u.name}: the remainder's read has other rows"
    with torch.no_grad():
        rd = Reader(job)
        runner = Runner(job, rd, res, a.wall_min * 60, ref, rows, updates)
        runner.rescore()  # a resume obeys the stop too
        for u in updates.values():  # every file before any read
            (pr.check_file if u.name[0] == "u" else run.check_file)(job, rd, u)
        runner.control(updates["u1"])
        for u in updates.values():
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

"""E151's apparatus (run.py calls it): the states, the readers shared with E147 and E149, and the job plumbing.

The share is E136's `Job.trained` on a saved bf16 update, as E147 read it; strength-1 steering is E149's `hf_state` on
E143's `Job`; the prior is prior_vs_share's forward prior. One model serves all three; each state's reads are kept once made.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from commute import keydep as KD
from commute import steering as ST
from commute.jobs import load_experiment
from commute.open_pair import jsonable

HF = load_experiment("holes_fill", "run")  # Job and lo_at: E143's steering path
E136 = load_experiment("share_untrained", "run")  # Job.trained: the share as E147 read it
PS = load_experiment("prior_steering", "run")  # hf_state: E149's strength-1 steering
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
EXP = REPO / "experiments"
FILES = {"e147_inv": EXP / "refit_96key/results/inventory.json", "e147_0": EXP / "refit_96key/results/stage0.json",
         "e149": EXP / "prior_steering/results/stage1.json", "priors": EXP / "prior_vs_share/results/stage2.json",
         "e143": EXP / "holes_fill/results/predictions.json", "a3": EXP / "open_pair/results/amend3.json"}  # fmt: skip
SAVED, CONTROL, NEW = ("copy", "elicit", "rank1", "class_shuffled"), ("copy", "prior_cipher"), ("common_cipher", "one_word")
PAIRS = {"A_copy_common": ("copy", "common_cipher"), "B_elicit_common": ("elicit", "common_cipher"),
         "D_rank1_oneword": ("rank1", "one_word"), "C_rank1_classshuffled": ("rank1", "class_shuffled")}  # fmt: skip
SEEDS, STEP, S1 = (0, 1, 2), 30, 1.0
log = KD.log


class Stop(Exception):
    """A filed stop: the stage's file records why; the job exits 1."""


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"stage{n}.tmp"  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(RESULTS / f"stage{n}.json")


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


def check(ch: dict[str, Any], name: str, value: float, tol: float) -> None:
    """A filed control: |value| <= tol (nan misses), recorded and logged pass or fail; a miss raises Stop."""
    ok = abs(value) <= tol
    ch[name] = {"value": value, "tol": tol, "pass": ok}
    log(f"[check {name}] {value:.6g} (tol {tol:g}): {'pass' if ok else 'MISS'}")
    if not ok:
        raise Stop(f"{name}: {value:.6g} over {tol:g}")


def task(m: str) -> str:
    return f"{'new' if m in NEW else 'e58'}/{m}"


def state(m: str, s: int) -> str:
    return f"{task(m)}_s{s}"


def update_file(m: str, s: int) -> Path:
    return (LARGE if m in NEW else EXP / "refit_96key/results/large") / f"dW_{m}_s{s}.pt"


class Run:
    """One model shared by E136's share reader and E143's steering path; each state's reads kept once made."""

    def __init__(self, P: dict[str, Any]) -> None:
        self.th, self.F = P["thresholds"], {k: load(v) for k, v in FILES.items()}
        maps = self.F["e147_inv"]["maps"]
        self.inv: dict[str, Any] = {"tasks": {}, "states": {}}
        for m in (*SAVED, "prior_cipher"):
            self.add_map(m, maps[m]["windows"])
        self.e136 = E136.Job(self.inv)  # loads the model; the new maps' reads are added once built
        self.net: ST.Net = self.e136.net
        hf = self.F["e143"]
        self.hf = HF.Job(self.net, self.inv, hf["settings"], hf["thresholds"])  # the same inventory dict, tasks loaded lazily
        self.n_select = hf["settings"]["n_select"]
        self.ctx = KD.Ctx(self.net, {}, {}, {}, LARGE)  # keydep.finetune and save_map read only ctx.net
        self.steer: dict[str, dict[str, Any]] = {}
        self.share: dict[str, dict[str, Any]] = {}
        self.prior: dict[str, float] = {}

    def add_map(self, m: str, windows: str) -> None:
        self.inv["tasks"][task(m)] = {"format": "kv", "path": windows, "batch": 64}
        for s in SEEDS:
            self.inv["states"][state(m, s)] = {"task": task(m), "file": str(update_file(m, s).relative_to(REPO))}

    def add_new(self, m: str) -> None:
        """A built map: into the shared inventory, and its read rows into E136's Job (which loads reads only at its start)."""
        self.add_map(m, str((LARGE / f"windows_{m}_trained.pt").relative_to(REPO)))
        self.e136.reads[task(m)] = E136.load_read(task(m), self.inv["tasks"][task(m)])

    def steering(self, name: str) -> dict[str, Any]:
        """E149's strength-1 read (pooled scored, per-key reported)."""
        if name not in self.steer:
            self.steer[name] = PS.hf_state(self.hf, name, self.n_select)
        return self.steer[name]

    def read_share(self, name: str, dW: dict[str, torch.Tensor]) -> dict[str, Any]:
        rs = self.e136.trained(name, dW)
        r, gain = rs["shares"]["r"], rs["gain"]
        return {"r": r, "gain": gain, "share": r if gain >= self.th["gain_floor"] else math.nan}

    def shared(self, name: str) -> dict[str, Any]:
        """The share of the saved bf16 update, as E147 read it; nan under the gain floor."""
        if name not in self.share:
            self.share[name] = self.read_share(name, self.e136.delta(name))
        return self.share[name]

    def map_prior(self, m: str) -> float:
        """W0's geometric-mean P(trained answer) on the trained keys' held-out windows (prior_vs_share's forward_prior)."""
        path = REPO / self.inv["tasks"][task(m)]["path"]
        self.prior[m] = KD.geo_rate(ST.load_task(self.net, path, m, m).lp0)
        return self.prior[m]


def seeds_of(cache: dict[str, dict[str, Any]], m: str, read: Any) -> list[float]:
    """A map's value per seed; nan for a seed not read (a map stopped at the gain floor)."""
    return [float(read(cache[state(m, s)])) if state(m, s) in cache else math.nan for s in SEEDS]


def steer_seeds(run: Run, m: str) -> list[float]:
    return seeds_of(run.steer, m, lambda r: r["pooled"]["share"])


def share_seeds(run: Run, m: str) -> list[float]:
    return seeds_of(run.share, m, lambda r: r["share"])


def prior_gap(run: Run, reuse: str, new: str) -> float:
    return math.log10(run.prior[reuse]) - math.log10(run.prior[new])


class Clock:
    """After each state: the job's projected minutes; over the filed limit after four states of a stage, stop."""

    def __init__(self, t_job: float, n: int, limit: float) -> None:
        self.t_job, self.t0, self.n, self.i, self.limit = t_job, time.time(), n, 0, limit

    def tick(self, out: dict[str, Any]) -> None:
        self.i += 1
        now = time.time()
        out["projected_minutes"] = m = (now - self.t_job + (now - self.t0) / self.i * max(self.n - self.i, 0)) / 60
        log(f"[time] {self.i} of {self.n} states; projected {m:.1f} min for the job (limit {self.limit})")
        if self.i >= 4 and m > self.limit:
            raise Stop(f"projected {m:.1f} min over {self.limit}")

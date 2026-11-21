"""P2.49 run 2 of 2 (E-number in results/run2/predictions.json): the size ladder at site A (block 2's input). Run 1
(E235) found the planted control missed there: E206's early grafted head read a specific part of 0.21-0.37 against
the 0.5 bar, at a change of 211 per token (about 100 times elicit's). Run 2 scales that head's output weight by one
factor per rung, so its activation change per token at A equals the rung's size, and trains the same crosscoder
(run 1's settings, code and bars, unchanged) on each, cheapest rung first, until a rung passes on every seed. At the
first such rung X it reads E147's five 96-key tasks (3 seeds each) and a Gaussian random change (3 draws), each
scaled to X the same way. Plan reports/plans/2026-11-21_plan_p2-49-run2-size-ladder.md; rules results/run2/
predictions.json (refuses unless committed unchanged). Every step but the scaling is run.py's `Job`.

Item keys (run.py's what.site.cSEED; `-rI` is the rung, rung 0 = scale 0, the base against itself):
  self.A.c0                 the grafted head at scale 0: the base against itself, through the scaled path
  head-rI.A.cS              the grafted head scaled to rung I's size, crosscoder seed S
  e58__<task>_sN-rI.A.c0    E147's saved update (task, fine-tune seed N) scaled to rung I's size
  random_dD-rI.A.c0         draw D of a Gaussian at random_of's per-matrix norms, scaled to rung I's size
GPU jobs (each crosscoder writes results/<tag>/items/<key>.json and results/large/<tag>/<key>.pt as it ends):
  sizes          the head's scale at every rung, its native energy against run 1's, each state's and draw's own
                 energy at scale 1 (forwards only, minutes) -> results/<tag>/sizes.json
  item KEY...    crosscoders by key, in order; which keys come next is `score2.py ladder`
CPU jobs: score2.py.

    PRED_SHA=<sha> python experiments/crosscoder_proper/run2.py item head-r4.A.c0 2>&1 | tee -a <log>
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import crosscoder as CC
from commute import finetune as FT
from commute import graft as G
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

RUN: Any = load_experiment("crosscoder_proper", "run")  # run 1: Job, Side, Item, Out, Stop, OUT, exact
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "run2" / "predictions.json"
E147_INV = EXPERIMENTS / "refit_96key" / "results" / "inventory.json"
log = RUN.log


@dataclass(frozen=True)
class Config:  # results/run2/predictions.json "settings"
    site: str
    rungs: tuple[float, ...]  # activation change per token at the site; rung 0 is 0
    native: int  # the rung run 1 already read: the head at scale 1
    native_tags: dict[str, str]  # sizes -> run 1's tag whose head.A.c* items are rung `native`
    native_energy: float  # run 1's head.A energy, which the sizes job must reproduce
    ref_energy: float  # a fine-tune's size at the site (run 1: e54 elicit s0), for the stated range only
    seeds: tuple[int, ...]  # crosscoder seeds per rung, and E147's fine-tune seeds
    tasks: tuple[str, ...]  # E147's names (e58/<task>)
    random_of: str
    random_seed: int
    draws: int
    match_tol: float
    scale_max: float
    pairs: tuple[tuple[str, str], ...]  # (reuse, new) as built, E147's
    planted_part: float

    def __post_init__(self) -> None:
        r = self.rungs
        assert r[0] == 0 and all(a < b for a, b in itertools.pairwise(r)) and 0 < self.native < len(r)
        assert math.isclose(r[self.native], self.native_energy, rel_tol=1e-9) and len(self.seeds) == self.draws == 3
        assert self.random_of in self.states() and all(t in self.tasks for p in self.pairs for t in p)

    def states(self) -> list[str]:
        return [f"e58/{t}_s{s}" for t in self.tasks for s in self.seeds]

    def stage2(self, rung: int) -> list[str]:
        """Stage 2 at rung X: the random draws first (its control), then every task state."""
        return [f"random_d{d}-r{rung}.{self.site}.c0" for d in range(self.draws)] + [
            f"{s.replace('/', '__')}-r{rung}.{self.site}.c0" for s in self.states()
        ]


def split(what: str) -> tuple[str, int]:
    """'head-r4' -> ('head', 4); 'self' is the head at rung 0."""
    if what == "self":
        return "head", 0
    base, _, r = what.rpartition("-r")
    return base, int(r)


@dataclass
class Scaled(RUN.Side):
    """A run-1 side whose change is multiplied by one factor x: the head's output weight, or a saved or random dW."""

    rung: int = 0
    unit: dict[str, Tensor] | None = None  # the change at x = 1 (a state or a draw); None for the head
    wo: tuple[Tensor, Tensor] | None = None  # the head's output weight and its loaded value

    @torch.no_grad()
    def apply(self, x: float) -> None:
        if self.wo is not None:
            self.wo[0].copy_(self.wo[1] * x)
        else:
            assert self.unit is not None
            self.dW = {k: x * v for k, v in self.unit.items()}


class Job(RUN.Job):
    def __init__(self, cfg1: Any, cc: Any, th: dict[str, Any], cfg: Config) -> None:
        super().__init__(cfg1, cc, th)
        _, full = RUN.SM.R96.inventories(json.loads(E147_INV.read_text()))  # E147's tasks and states, elicit among them
        self.inv["tasks"].update({f"e58/{t}": full["tasks"][f"e58/{t}"] for t in cfg.tasks})
        self.inv["states"].update({s: full["states"][s] for s in cfg.states()})
        self.c2 = cfg

    def side(self, it: Any) -> Scaled:
        base, rung = split(it.what)
        if base == "head":
            s0 = super().side(RUN.Item("head", it.site, it.seed))
            layer = self.cfg.graft_layers[self.cfg.heads[it.site]]
            wo = G.out_param(s0.ft.net.h[layer].attn.graft)
            return Scaled(**vars(s0), rung=rung, wo=(wo, wo.detach().clone()))
        state = self.c2.random_of if base.startswith("random_d") else base.replace("__", "/")
        s0 = super().side(RUN.Item(state.replace("/", "__"), it.site, it.seed))
        unit = s0.dW
        if base.startswith("random_d"):  # the very construction run 1's Gaussian used (finetune.random_direction)
            draw = FT.random_direction({k: v.cpu() for k, v in s0.dW.items()}, self.c2.random_seed + int(base.removeprefix("random_d")))
            unit = {k: v.cuda() for k, v in draw.items()}
        return Scaled(**vars(s0), rung=rung, unit=unit)

    def size_at(self, s: Scaled, x: float) -> float:
        s.apply(x)
        return self.energy(s, s.dW)

    def match(self, s: Scaled) -> dict[str, Any]:
        """The factor x that gives the rung's activation change per token on the side's read rows: doubling, then
        bisection in log x until within a quarter of match_tol (run 1's Gaussian search, run.py `gauss`)."""
        target = self.c2.rungs[s.rung]
        if target == 0:
            return {"x": 0.0, "target": 0.0, "energy": self.size_at(s, 0.0), "rel": 0.0}
        lo, hi = 0.0, 1.0
        while self.size_at(s, hi) < target:
            if hi >= self.c2.scale_max:
                return {"x": hi, "target": target, "energy": self.size_at(s, hi), "rel": math.nan, "unreachable": True}
            lo, hi = hi, 2 * hi
        x = hi
        for _ in range(60):
            x = math.sqrt(lo * hi) if lo > 0 else hi / 2
            got = self.size_at(s, x)
            if abs(got / target - 1) <= self.c2.match_tol / 4:
                break
            lo, hi = (x, hi) if got < target else (lo, x)
        got = self.size_at(s, x)
        return {"x": x, "target": target, "energy": got, "rel": got / target - 1}

    def zero_diff(self, s: Scaled) -> float:
        """Max |activation difference| from W0 at the site with the side's change times 0 (8 read rows)."""
        s.apply(0.0)
        with RUN.exact():
            toks = s.read[:8]
            return float((CC.site_acts(self.net, toks, None, s.site) - CC.site_acts(s.ft, toks, s.dW, s.site)).abs().max())

    def before(self, s: Scaled, out: Any) -> None:  # type: ignore[override]
        """Run 2's checks before step 0, then run 1's (the change's energy; the self path for 'self')."""
        th, tol = self.th, self.c2.match_tol
        diff = self.zero_diff(s)
        out.check("the scaled path at 0 gives W0's activations", diff <= th["self_act_diff"], {"max_abs": diff, "bar": th["self_act_diff"]})
        m = self.match(s)
        out.d["match"] = m
        out.check("change matched to the rung's size", m["target"] == 0 or abs(m["rel"]) <= tol, m | {"tol": tol})
        s.apply(m["x"])
        if s.rung > 0:
            out.check("the change writes at the site", m["energy"] > th["graft_energy_min"], {"energy": m["energy"], "bar": th["graft_energy_min"]})
        out.d["energy_pile"] = self.energy(dataclasses.replace(s, read=self.graft_read()), s.dW)  # one row set for every side
        super().before(s, out)


def sizes(job: Job) -> None:
    """Forwards only: the head's scale per rung (and its native energy against run 1's), each state's and draw's own
    activation change at scale 1, on its read rows and on the common Pile rows."""
    c, site, t0 = job.c2, job.c2.site, time.time()
    h = job.side(RUN.Item(f"head-r{c.native}", site, 0))
    native = job.size_at(h, 1.0)
    out = RUN.Out(RUN.OUT["res"] / "sizes.json", {"experiment": RUN.EXPERIMENT, "head_native_energy": native, "rungs": list(c.rungs)})
    out.check("scale 1 reproduces run 1's head", math.isclose(native, c.native_energy, rel_tol=1e-4), {"energy": native, "run1": c.native_energy})
    head = {}
    for i in range(1, len(c.rungs)):
        h.rung = i
        head[i] = job.match(h)
        log(f"[sizes] head rung {i}: {json.dumps(head[i])}")
    out.check("rung native is scale 1", abs(head[c.native]["x"] - 1) <= c.match_tol, {"x": head[c.native]["x"], "tol": c.match_tol})
    own = {}
    for key in [*(s.replace("/", "__") for s in c.states()), *(f"random_d{d}" for d in range(c.draws))]:
        s = job.side(RUN.Item(f"{key}-r1", site, 0))
        own[key] = {"energy": job.size_at(s, 1.0), "energy_pile": job.energy(dataclasses.replace(s, read=job.graft_read()), s.dW)}
        log(f"[sizes] {key} at scale 1: {json.dumps(own[key])}")
    out.d |= {"head": head, "own_at_scale_1": own, "minutes": (time.time() - t0) / 60}
    out.write()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("sizes", "item"))
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stop-after", type=int, default=None)
    a = ap.parse_args()
    require_filed(PRED)
    P1, P2 = json.loads(RUN.PRED.read_text()), json.loads(PRED.read_text())
    assert P2["experiment"][1:].isdigit(), "claim the E-number (fleet/ecode) and file it in results/run2/predictions.json"
    size = "rehearsal" if a.rehearse else "full"
    S = P1["settings"]["cc"][size]
    cc = config_from_settings(CC.Cfg, S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])})
    cfg1 = config_from_settings(RUN.Config, {k: tuple(v) if isinstance(v, list) else v for k, v in P1["settings"]["run"].items()})
    s2 = {k: tuple(map(tuple, v)) if k == "pairs" else tuple(v) if isinstance(v, list) else v for k, v in P2["settings"].items()}
    cfg = config_from_settings(Config, s2)
    tag = a.tag or ("rehearse2" if a.rehearse else "run2")
    RUN.EXPERIMENT = P2["experiment"]  # run.py stamps each item with its module's E-code; run 2's here
    RUN.OUT.update(res=HERE / "results" / tag, large=HERE / "results" / "large" / tag)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    log(f"[{RUN.EXPERIMENT}] {a.job} {a.keys} tag {tag}, sizes {size}; {cfg}")
    job = Job(cfg1, cc, P1["thresholds"][size], cfg)  # run 1's filed bars, unchanged
    try:
        if a.job == "sizes":
            sizes(job)
        for key in a.keys:
            if RUN.done(key):
                log(f"[item] {key} done before, skipped")
                continue
            job.run_item(RUN.Item.parse(key), a.stop_after)
            torch.cuda.empty_cache()
    except RUN.Stop as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    log(f"FINISHED {a.job} {' '.join(a.keys)} in {(time.time() - job.t0) / 60:.1f} min; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())

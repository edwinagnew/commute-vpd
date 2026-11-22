"""E254 (cell P2.57) run A: key copy against the prior-matched cipher at 10 seeds. E237 read this pair at 3 seeds
(copy's log10 prior -4.23, the cipher's -4.43): every cheap reading split them while the prior did not. Here both are
fine-tuned again with E147's exact call (`keydep.finetune`, step 30, E147's windows files, seeds 0-9; seeds 0-2 must
repeat E147's saved files bit for bit), scaled to E237's common size, and read with P2.54's readers (world.py): the
copy-head switch-off at k = 1 ((2,4)) and k = 3, the re-gating share r, the effective rank, holes_fill's steering
(pooled, per-key) and neuron (k64, k matched) shares. Per reading, the seed-level rank test (exact Mann-Whitney U,
copy's 10 seeds against the cipher's 10) and the gap against the smallest gap that matters (the larger 3-seed range at
seeds 0-2, E237's measure). Beside seeds 0-2 of each task, one Gaussian random update at its per-matrix norms times
the same c. Rules and expected values in results/predictions_a.json (refuses unless committed unchanged).

  0  the controls, alone (world.controls); the do-nothing through every reader on copy's rows  -> results/a/control.json, do_nothing.json
  1  20 fine-tunes (exit 1 if a seed does not learn); seeds 0-2 against E147's files; c       -> results/a/train.json
  2  items: seeds 0-2 of both tasks, their random updates, then seeds 3-9                     -> results/a/items/<item>.json
  score  labels, the eight calls, the prior's gap, the random updates                        -> results/a/summary.json
Every scored number is also printed. A rerun skips what is written.

    sbatch --comment=E254 cluster/prior_matched_a.sbatch
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

W: Any = load_experiment("prior_matched", "world")  # the world, the fine-tune, the controls, the readings, the rules

PRED, OUT = W.RESULTS / "predictions_a.json", W.RESULTS / "a"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E147_LARGE = W.X / "refit_96key" / "results" / "large"
log = W.log


@dataclass(frozen=True)
class Config:  # filed in predictions_a.json "settings"
    tasks: tuple[str, ...]  # E147's maps, the reused one first
    seeds: tuple[int, ...]
    random_seeds: tuple[int, ...]  # the seeds whose state gets a matched random update
    random_seed: int

    def __post_init__(self) -> None:
        assert self.tasks == ("copy", "prior_cipher") and self.seeds == tuple(range(10)) and self.random_seeds == self.seeds[:3]

    def name(self, task: str) -> str:
        return f"pa_{task}"  # this run's task, on E147's windows file for <task>

    def states(self, task: str) -> list[str]:
        return [f"e58/{self.name(task)}_s{s}" for s in self.seeds]

    def items(self) -> list[str]:
        """Seeds 0-2 of both tasks (the band's seeds), their random updates, then seeds 3-9."""
        first = [st for t in self.tasks for st in self.states(t)[:3]]
        return first + [f"random/{st}" for st in first] + [st for t in self.tasks for st in self.states(t)[3:]]


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def world(cfg: Config) -> tuple[Any, str]:
    w = W.World(cfg.random_seed)
    zero = ""
    for t in cfg.tasks:
        z = w.add_task(cfg.name(t), W.E147_INV["maps"][t]["windows"])
        zero = zero or z  # the do-nothing reads copy's rows
        for s in cfg.seeds:
            w.add_state(cfg.name(t), s)
    return w, zero


def stage1(cfg: Config, th: dict[str, Any], w: Any) -> dict[str, Any]:
    path = OUT / "train.json"
    out = {"experiment": EXPERIMENT}
    for t in cfg.tasks:
        out |= W.train(w, cfg.name(t), W.E147_INV["maps"][t]["windows"], cfg.seeds, path, th["p_learn"])
        write_stage(path, jsonable(out))
    out["repeat_E147"] = {}
    for t in cfg.tasks:
        for s in cfg.seeds[:3]:
            mine = torch.load(W.LARGE / f"dW_{cfg.name(t)}_s{s}.pt", weights_only=True)
            ref = torch.load(E147_LARGE / f"dW_{t}_s{s}.pt", weights_only=True)
            out["repeat_E147"][f"{t}_s{s}"] = max(float((mine[k].float() - ref[k].float()).abs().max()) for k in ref)
    log(f"[1] seeds 0-2 against E147's saved files, max |diff| (bf16; 0 expected, reported): {out['repeat_E147']}")
    out["c"] = {cfg.name(t): W.common_c([out["states"][st]["size"] for st in cfg.states(t)]) for t in cfg.tasks}
    log(f"[1] c {out['c']}")
    write_stage(path, jsonable(out | {"done": True}))
    return out


def summary(cfg: Config, th: dict[str, Any]) -> None:
    item = lambda i: W.read_json(OUT / "items" / f"{W.R1.fname(i)}.json")
    reused, new = ([item(st) for st in cfg.states(t)] for t in cfg.tasks)
    va, vb = [W.values(r) for r in reused], [W.values(r) for r in new]
    bands = {p: W.band([x[p] for x in va], [x[p] for x in vb]) for p in W.READINGS}
    lp = W.read_json(W.E237 / "summary.json")["log10_prior"]
    rnd = [item(i) for i in cfg.items() if i.startswith("random/")]
    out = {"experiment": EXPERIMENT, "pair": list(cfg.tasks)} | W.score(reused, new, rnd, bands, {"reused": lp["copy"], "new": lp["prior_cipher"]}, th)
    write_stage(OUT / "summary.json", jsonable(out))
    W.say(out)


def main() -> int:
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 reads, as every saved reading
    log(f"{EXPERIMENT} run A start; settings {asdict(cfg)}; thresholds {th}")
    OUT.mkdir(parents=True, exist_ok=True)
    w, zero = world(cfg)
    job = W.k_job(w)
    W.controls(w, job, OUT)
    W.do_nothing(w, job, zero, OUT)
    w.rd.c |= stage1(cfg, th, w)["c"]
    for i, it in enumerate(cfg.items()):
        W.read_item(w, job, it, i, OUT)
        w.dws.clear()  # one update in memory at a time
    summary(cfg, th)
    log(f"{EXPERIMENT} run A DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

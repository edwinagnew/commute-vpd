"""E269 (cell P2.61; user approval 2026-11-22 ~18:15): the definition's K* search on the sequence-task RL and SFT states
without the 2,000-evaluation cap. E257 (experiments/rb_states) hit the cap on all 66 of its searches; on the sequence
task its E_b read RL and SFT alike while E260's re-gating share read RL above SFT. Settings and filed rules in
results/predictions.json, which the run refuses unless committed unchanged.

Per item: F, M and D exactly as E257 built them (rb_states World, its manifest results/manifest_p256.json: the P2.56
state and the task's held-out read rows), and K* by E226's read_item with E226's filed settings except `eval_cap`,
which is set here so high that the polish runs through every prefix member (commute.kstar_vpd.search). Writes
results/kstar/<item>.json (the search record) and results/large/kstar_<item>.npz (membership, attributions). The
search logs |K| every 500 evaluations: at 2,000 it should equal E257's capped |K*| (the search is the same up to the
cap). E_b is not computed here: score.py reads it on the dev node from E257's saved num, den, dd and changed arrays
with the new K*. A rerun skips an item whose JSON exists.

    PRED_SHA=<sha> python experiments/kstar_uncapped/run.py --shard 0 3      (every 3rd item from the 0th)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import torch

from commute import kstar_vpd as KV
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

RB: Any = load_experiment("rb_states", "run")  # E257: World (E242's world plus the P2.56 manifest), E226
E226 = RB.E226
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]


@dataclasses.dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    items: tuple[str, ...]  # manifest items of rb_states (E257), read in this order
    eval_cap: int  # replaces E226's 2,000: above any prefix length, so the polish is never cut

    def __post_init__(self) -> None:
        extra = {e["item"] for e in json.loads((RB.RESULTS / "manifest_p256.json").read_text())}
        assert set(self.items) <= extra, set(self.items) - extra
        assert self.eval_cap > 10 * 40_000  # every sequence prefix has under 40,000 members (E257 logs: at most 33,317)


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def world() -> Any:
    """E257's world, set up as rb_states.main does before its manifest stage."""
    w = RB.World(RB.load_config())
    w.bases = KV.bases(w.uv, E226.load_config()[0].split_tol)
    r = json.loads((RB.S242 / "items" / f"{RB.fname(RB.E242.load_config()[0].random_kl)}.json").read_text())
    w.kl_x = {k: r[k] for k in ("x", "kl_matched", "match", "match_evals")}
    return w


def main(shard: tuple[int, int]) -> int:
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226, E257
    c226, th226 = E226.load_config()
    c226 = dataclasses.replace(c226, eval_cap=cfg.eval_cap)
    E226.RESULTS, E226.EXPERIMENT = RESULTS, EXPERIMENT  # E226's read_item writes results/large/kstar_<item>.npz here
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    todo = list(cfg.items)[shard[0] :: shard[1]]
    RB.log(f"{EXPERIMENT} start, shard {shard}: {todo}; E226 settings {dataclasses.asdict(c226)}")
    w = world()
    for k, item in enumerate(todo):
        t0 = time.time()
        rec: dict[str, Any] = E226.read_item(w, item, c226, th226, RESULTS / "kstar")
        m = rec.get("searches", {}).get("main", {})
        RB.log(f"ITEM {k + 1} of {len(todo)} DONE ({item}): reading {rec.get('reading')}, |K*| {m.get('polished', {}).get('size')}, "
               f"{m.get('evals')} evaluations, capped {m.get('capped')}, {time.time() - t0:.0f} s")  # fmt: skip
    RB.log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main((args.shard[0], args.shard[1]))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

"""E321 (cell P1.65, plan reports/plans/2026-11-26_plan_p1-65-readthrough-fills.md, Run 1): main.tex:432's graded
cos^2 clause ("0.999 or more") read over the needed set K instead of E208's KL-closure set. K is E277's neuron swap test
(experiments/needed_neurons, `read_item`, its trained world: E229's 20 E196 pair fine-tunes, D = E208's 512 read windows,
last position, 256 pairs of seed 268) with one change, the band "width" (`topk_upper.feat_of`: hi = -inf), so every
activation is its neuron's value and each layer's remainder writes 0: E208's featurization of one variable per neuron.
No matched randoms are read. Settings and filed expected values in results/predictions.json, which the run refuses
unless committed unchanged; the rest of E277's settings are its filed ones, pinned by sha256.

Writes results/items/<item>.json and results/large/items_<item>.npz (E277's columns and arrays; an item already written
is skipped, so a killed shard resumes). score.py (CPU) then takes E208's saved per-part r_main over K.

Stages; each job runs stage 0 first (skipped once results/stage0_trained_<tag>.json says done), and a miss stops it:
    0     E277's trained-world stage 0 at the width band: do-nothing (F = a copy of M: 0 changed, dd max 0.0, E_b over K
          1.0, so K and changed is empty and the K value reads 1) and the identity swap (x' = x, copy24_s0's first 64
          windows, every variable: 0 argmax changes)
    read  the 20 fine-tunes, or those given by --items

    python experiments/p165_cos2_k/run.py --stage 0 --tag s0
    python experiments/p165_cos2_k/run.py --stage read --items copy24_s0 cipher24_s0 ... --tag a
    ~/commute-agents/fleet/cpu-run python3 experiments/p165_cos2_k/score.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

NNR: Any = load_experiment("needed_neurons", "run")  # E277: Config, PRED, Trained, stage0, read_item, load_e229, E229
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
NNR.RESULTS, NNR.EXPERIMENT = RESULTS, EXPERIMENT  # E277's stage0 and read_item write here, stamped with this code
log = NNR.log


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    band: str  # the one change to E277's settings
    e277_predictions_sha256: str  # E277's filed settings, used unchanged otherwise

    def __post_init__(self) -> None:
        assert self.band == NNR.E229.WIDTH, self.band


def load() -> tuple[Config, Any, Any]:
    """This run's config, E277's with the band replaced (its file checked against the pin), and E229's."""
    cfg = config_from_settings(Config, json.loads(PRED.read_text())["settings"])
    sha = hashlib.sha256(NNR.PRED.read_bytes()).hexdigest()
    assert sha == cfg.e277_predictions_sha256, f"{NNR.PRED} sha256 {sha}, pinned {cfg.e277_predictions_sha256}"
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(NNR.PRED.read_text())["settings"].items()}
    return cfg, config_from_settings(NNR.Config, s | {"band": cfg.band}), NNR.load_e229()


def main(stage: str, items: list[str] | None, tag: str) -> int:
    cfg, c277, c229 = load()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E277
    for d in ("items", "large"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    feat = NNR.E229.feat_of(c229, NNR.E229.load_pct(c229), c277.band)
    world = NNR.Trained(c277, c229)
    todo = items if items is not None else world.base
    assert set(todo) <= set(world.base), f"not E229's fine-tunes: {sorted(set(todo) - set(world.base))}"
    log(f"{EXPERIMENT} start, stage {stage}, {len(todo)} items {todo}; settings {asdict(cfg)}; E277's {asdict(c277)}")
    floor = NNR.stage0(world, c277, feat, tag)["identity"]["max_dev"]
    if stage == "0":
        return 0
    for k, item in enumerate(todo):
        NNR.read_item(world, item, c277, feat, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("0", "read"), required=True)
    ap.add_argument("--items", nargs="+", help="these fine-tunes only (a shard); default all 20")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_trained_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, args.items, args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

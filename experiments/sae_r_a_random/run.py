"""E251: E245's reading (R under wording A at E234's base-only SAE latents) on random updates of matched size (cell P1.40 run
3). Plan reports/plans/2026-11-21_plan_r-under-a-sae-random.md; card
reports/plans/2026-11-21_plan_r-under-a-sae-random-card.md; rules in results/predictions.json, which the run refuses
unless committed unchanged.

Items: E237's 15 random controls (random_<state>: a Gaussian per matrix at that matrix's norm in the key-task state's
update, times E237's common size c, on the state's own windows; E237's World builds them) and E242's two random
controls at VPD (random_size_241001: new_thin's update size on x2's windows; random_kl_241002: KL-matched on x10's),
so the SAE reading sits beside E242's VPD reading of the very same updates. Every reading step, control, threshold and
setting is E245's, imported and called unchanged; only the items and the results directory differ.

    PRED_SHA=... python experiments/sae_r_a_random/run.py --shard 0 2 --tag random0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from commute.jobs import hard_exit, load_experiment, require_filed

E245: Any = load_experiment("sae_r_a", "run")  # load_config, load_saes, stage0, read_item, E242
E242 = E245.E242
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E245.RESULTS, E245.EXPERIMENT = RESULTS, EXPERIMENT  # E245's read_item and stage0 write here, stamped with our code
log = E245.log


def items() -> list[str]:
    kg = load_experiment("key_grid", "run")
    E242.point_e226_here()
    e242 = E242.load_config()[0]
    return [e242.random, e242.random_kl, *(f"random_{s}" for s in kg.load_config()[0].states())]


class World(E242.World):
    """E242's World; E237's random controls built by E237's World, read against its base model."""

    def make(self, item: str) -> tuple[Any, Any, torch.Tensor, slice, dict[str, Any]]:
        if not item.startswith("random_e58/"):
            return super().make(item)
        if self.kg is None:
            kg = load_experiment("key_grid", "run")
            E242.point_e226_here()
            self.kg = kg.World(kg.load_config()[0])
            self.kg.c = json.loads((E242.SAVED["e237"] / "stage0.json").read_text())["c"]
        F, toks, at, rec = self.kg.make(item)
        return F, self.kg.M, toks, at, rec


def main(shard: tuple[int, int], tag: str) -> int:
    cfg, th = E245.load_config("scored")
    want = json.loads(PRED.read_text())["e245_predictions_sha256"]
    got = hashlib.sha256(E245.PRED.read_bytes()).hexdigest()
    assert got == want, f"E245 predictions.json sha256 {got} != filed {want}: E245's settings changed"
    todo = items()[shard[0] :: shard[1]]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E245
    log(f"{EXPERIMENT} start, tag {tag}; {len(todo)} items {todo}; E245 settings {asdict(cfg)}; thresholds {th}")
    w, saes = World(E242.load_config()[0]), E245.load_saes(cfg)
    floor = E245.stage0(w, saes, cfg, th, tag, todo)
    for k, item in enumerate(todo):
        E245.read_item(w, saes, item, cfg, th, RESULTS / cfg.out, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"FINISHED {EXPERIMENT} {tag}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/items/stage0_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main((args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

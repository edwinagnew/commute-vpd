"""E_TBD_RUN2 (cell P1.65, plan reports/plans/2026-11-26_plan_p1-65-readthrough-fills.md, Run 2): the rescale-only
control (E290's FFN-gain states: only down_proj's input gains trained, one per MLP neuron, saved 2026-11-23 as
ia3_control/results/large/gains_ffn_<task>_s<seed>.pt) read under E298's protocol at VPD (seeds 1, 2, and seed 0 again
as the repeat check) and at SAE latents (seeds 0-2). Nothing is trained. The reader is E298's (fig2_standard/run.py
read_item, feats.reader) under E298's filed settings unchanged (pinned by sha256); the states are E290's
World.make("ffn/<task>_s<seed>") at its read step, through a Worlds subclass that adds the block gain/<task>_s<seed>.
Settings and filed expected values in results/predictions.json, which the run refuses unless committed unchanged.

Per item results/items/<feat>/gain__<item>.json and results/large/<feat>/gain__<item>.npz (E298's columns and arrays);
an item already written is skipped, so a killed shard resumes. score.py (CPU) writes the rows.

Stages, per featurization (--feat); any miss stops the job:
    0   E298's stage 0 on this job's card (each part skipped once results/stage0/<feat>_<part>_<tag>.json says done):
        do_nothing (F = M: 0 changed, dd max 0.0, E_b 1.0), identity (x' = x on built/x2: 0 argmax changes), and at
        the SAE must_flip (every variable swapped from x' at once gives F's argmax on x')
    1   stage 0, then the filed items (VPD: seed 0 first, each checked as it lands against E290's
        vpd/ffn__<task>_s0.json E_b_K within E298's repro_rel_tol["vpd"], results/repeat/<item>.json; then seeds 1, 2.
        SAE: seeds 0-2), or those given by --items

    python experiments/p165_gain_reads/run.py --feat vpd --stage 0 --tag v0
    python experiments/p165_gain_reads/run.py --feat vpd --stage 1 --items gain/copy_s0 gain/copy_s1 --tag va
    python experiments/p165_gain_reads/run.py --feat sae --stage 1 --tag s0
    ~/commute-agents/fleet/cpu-run python3 experiments/p165_gain_reads/score.py
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

FS: Any = load_experiment("fig2_standard", "run")  # E298: load_config, PRED, read_item, stage0, rel; FT (feats), EB (eb)
FT, EB, IA = FS.FT, FS.EB, FS.FT.IA  # IA: E290 (its World, RESULTS, fname)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
FS.RESULTS, FS.EXPERIMENT = RESULTS, EXPERIMENT  # E298's read_item and stage-0 parts write here, stamped with this code
log, dump, stop, fname = FS.log, FS.dump, FS.stop, FS.fname
FEATS = ("vpd", "sae")


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    arm: str  # E290's arm read
    repeat_seed: int  # read again at VPD against E290's saved value
    vpd_seeds: tuple[int, ...]
    sae_seeds: tuple[int, ...]
    e298_predictions_sha256: str  # E298's filed settings, used unchanged

    def __post_init__(self) -> None:
        assert self.repeat_seed not in self.vpd_seeds and set(self.vpd_seeds) | {self.repeat_seed} == set(self.sae_seeds)

    def items(self, feat: str, tasks: tuple[str, ...]) -> list[str]:
        seeds = (self.repeat_seed, *self.vpd_seeds) if feat == "vpd" else self.sae_seeds
        return [f"gain/{t}_s{s}" for s in seeds for t in tasks]


def load() -> tuple[Config, Any]:
    cfg = config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()})
    sha = hashlib.sha256(FS.PRED.read_bytes()).hexdigest()
    assert sha == cfg.e298_predictions_sha256, f"{FS.PRED} sha256 {sha}, pinned {cfg.e298_predictions_sha256}"
    return cfg, FS.load_config()


class Worlds(FT.Worlds):
    """E298's worlds plus gain/<task>_s<seed>: E290's FFN-gain state at its read step, on E237's D at the last position
    (E290's World, the one E298's trained/ block reads, so `of` already gives it for gain/ items)."""

    def __init__(self, c298: Any, vpd: bool, arm: str) -> None:
        super().__init__(c298, vpd)
        self.arm = arm

    def make(self, item: str) -> tuple[Any, Any, torch.Tensor, slice, dict[str, Any]]:
        block, name = item.split("/")
        return self.trained.make(f"{self.arm}/{name}") if block == "gain" else super().make(item)


def repeat(item: str, got: dict[str, Any], cfg: Config, tol: float) -> None:
    """Seed 0 at VPD: E_b over K within tol (relative) of E290's saved read of the same state; a miss stops the job."""
    name = item.split("/")[1]
    src = IA.RESULTS / "vpd" / f"{fname(cfg.arm + '/' + name)}.json"
    saved = json.loads(src.read_text())["E_b_K"]
    out = {"experiment": EXPERIMENT, "item": item, "E_b_K": got["E_b_K"], "saved": saved, "source": str(src.relative_to(HERE.parents[1])),
           "rel_dev": FS.rel(got["E_b_K"], saved), "tol": tol}  # fmt: skip
    path = RESULTS / "repeat" / f"{fname(item)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"repeat {item}: E_b over K {out['E_b_K']:.6f}, E290's {saved:.6f}, relative deviation {out['rel_dev']:.3g} (tol {tol:.3g})")
    if out["rel_dev"] > tol:
        stop(path, out, f"repeat {item}: off E290's saved value by {out['rel_dev']:.3g} relative (tol {tol:.3g})")
    dump(path, out)


def main(feat: str, stage: str, items: list[str] | None, tag: str) -> int:
    cfg, c298 = load()
    src = EB.check_sources(c298)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E298
    worlds, rd = Worlds(c298, feat == "vpd", cfg.arm), FT.reader(feat, c298)
    filed = cfg.items(feat, c298.tasks)
    todo = items if items is not None else filed
    assert set(todo) <= set(filed), f"not this featurization's filed items: {sorted(set(todo) - set(filed))}"
    log(f"{EXPERIMENT} start, feat {feat}, stage {stage}, {len(todo)} items {todo}; sources {src}; settings {asdict(cfg)}")
    for d in ("items", "large"):
        (RESULTS / d / feat).mkdir(parents=True, exist_ok=True)
    (RESULTS / "stage0").mkdir(parents=True, exist_ok=True)
    if not FS.stage0(feat, rd, worlds, c298, False, tag):  # False: E298's own reproductions are not read again
        log(f"STOP: E298's repro_rel_tol[{feat}] is not filed")
        return 1
    if stage == "0":
        return 0
    for k, item in enumerate(todo):
        got = FS.read_item(feat, rd, worlds, item, c298)
        if feat == "vpd" and item.endswith(f"_s{cfg.repeat_seed}"):
            repeat(item, got, cfg, c298.repro_rel_tol["vpd"])
        log(f"ITEM {k + 1} of {len(todo)} DONE ({feat} {item})")
    log(f"STAGE 1 DONE ({feat})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=FEATS, required=True)
    ap.add_argument("--stage", choices=("0", "1"), required=True)
    ap.add_argument("--items", nargs="+", help="these gain/<task>_s<seed> items only (a shard); default the filed list")
    ap.add_argument("--tag", default="local", help="this job's name: its stage-0 files are results/stage0/<feat>_<part>_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.feat, args.stage, args.items, args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

"""Cell P1.67 run 2 (redteam of E329, reports/reviews/2026-11-27_review_e329-adjoint-rho.md; needs the user's yes): rho at
VPD under the adjoint split as GOALS.md Terms define it, with the per-matrix scale c = split.adjoint_scale(dW, U, V,
Sigma), on E329's seven built edits. E329 (run.py) read the unscaled split (c = 1). Settings and expected values in
results/predictions_run2.json, which the run refuses unless committed unchanged.

Per item: Sigma, each decomposed matrix's input second moment, from one pass of W0 (the decomposition's target model,
`split.input_second_moments`) over every position of the item's read rows D; then `kstar_vpd.featurize(...,
split="adjoint_scaled", sigma=Sigma)`: each matrix's adjoint pieces times its c, the remainder taking the rest so the
pieces sum to dW; then K and E_b over K exactly as E329's stage 2 (run.py `read`'s calls: `needed.needed_set`, E268's
settings, and `stand_in.write_change`). Recorded per item: c and adjoint_scale's reproduced-energy share R per changed
matrix, and their median, min and max.

One job, in this order; a miss in stage 0 stops it:
    0  controls: (a) the do-nothing (F = M) under the scaled split: no changed variable, every dd exactly 0, E_b exactly
       1; (b) the identity interchange (x' = x) on `control` under the scaled featurization: 0 argmax changes
    2  the scaled adjoint split on every item (this shard's)
results/items_run2_<tag>.jsonl, one line per reading as it lands, with seconds; a rerun skips lines already written.

    PRED_SHA=<sha> python experiments/adjoint_rho/run2.py --shard 0 4 --tag s0
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import kstar_vpd as KV
from commute import needed as IX
from commute import split as SP
from commute import stand_in as SI
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

R1: Any = load_experiment("adjoint_rho", "run")  # E329: E268's world, Lines, saved (E268's K and E_b_K), write_pass
RESULTS = Path(__file__).resolve().parent / "results"
PRED = RESULTS / "predictions_run2.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
SPLIT = "adjoint_scaled"
log, fname = R1.log, R1.fname


@dataclass(frozen=True)
class Config:  # filed in predictions_run2.json "settings"
    items: tuple[str, ...]
    control: str  # the identity interchange's item
    derangements: int  # E268's scan settings, as E329
    examples: int
    seed: int
    rows: int
    logit_rows: int
    cache_batch: int
    ident_vars: int
    gain_batch: int  # windows per batch of the write pass and of the Sigma pass

    def __post_init__(self) -> None:
        assert self.control in self.items and "do_nothing" not in self.items
        assert min(self.derangements, self.examples, self.rows, self.cache_batch, self.ident_vars, self.gain_batch) >= 1
        assert self.logit_rows >= self.rows


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def featurized(w: Any, item: str, cfg: Config) -> tuple[Any, KV.Feat, torch.Tensor, slice]:
    """F, the scaled-adjoint featurization with Sigma from W0's run on the item's D, D, the read positions."""
    F, M, toks, at, _ = w.make(item)
    sigma = SP.input_second_moments(w.kl.net.model, list(toks.split(cfg.gain_batch)), list(w.keys))
    return F, KV.featurize(M, F, w.uv, w.delta, w.bases, SPLIT, sigma), toks, at


def scales(feat: KV.Feat) -> dict[str, Any]:
    """c and R per changed matrix (adjoint_scale's; absent where dW = 0), with their median, min and max."""
    per = {k: {"c": ch["c"], "R": ch["R"]} for k, ch in feat.checks.items() if "c" in ch}
    out: dict[str, Any] = {"per_matrix": per, "split_fp64_max": max(ch["split_fp64"] for ch in feat.checks.values())}
    for q in ("c", "R"):
        v = [p[q] for p in per.values()]
        out[q] = {"median": statistics.median(v), "min": min(v), "max": max(v)} if v else None
    return out


def stage0(w: Any, cfg: Config, out: Any) -> None:
    if not out.has("0", SPLIT, "do_nothing"):
        t0 = time.time()
        F, feat, toks, _ = featurized(w, "do_nothing", cfg)
        num, den, dd, ch = R1.write_pass(F, feat, toks, cfg)
        r = {"experiment": EXPERIMENT, "stage": "0", "split": SPLIT, "item": "do_nothing", "n_changed": len(feat.changed),
             "dd_max": float(dd.abs().max()), "E_b": SI.explained_share(num, den, dd, ch), "seconds": time.time() - t0}  # fmt: skip
        out.add(r, None if (r["n_changed"], r["dd_max"], r["E_b"]) == (0, 0.0, 1.0) else "do-nothing is not exactly unchanged")
    if not out.has("0", SPLIT, f"identity:{cfg.control}"):
        t0 = time.time()
        F, feat, toks, at = featurized(w, cfg.control, cfg)
        idn = IX.identity_scan(F, feat, toks, at, cfg.examples, cfg.ident_vars, cfg.rows, cfg.logit_rows, cfg.cache_batch, log)
        out.add({"experiment": EXPERIMENT, "stage": "0", "split": SPLIT, "item": f"identity:{cfg.control}", **idn, "seconds": time.time() - t0},
                f"identity interchange: {idn['flips']} argmax changes" if idn["flips"] else None)  # fmt: skip


def read(w: Any, cfg: Config, item: str) -> dict[str, Any]:
    """K on the scaled featurization and E_b over it; E268's least-squares K and E_b beside it (E329's `read`)."""
    t0 = time.time()
    F, feat, toks, at = featurized(w, item, cfg)
    nd = IX.needed_set(F, feat, toks, at, cfg.derangements, cfg.examples, cfg.seed, cfg.rows, cfg.logit_rows, cfg.cache_batch, log)
    num, den, dd, ch = R1.write_pass(F, feat, toks, cfg)
    K0, eb0 = R1.saved(item)
    r = {"experiment": EXPERIMENT, "stage": "2", "split": SPLIT, "item": item, "E_b_K": SI.explained_share(num, den, dd, nd.K & ch),
         "E_b_K_E268": eb0, "E_b_over_E268_K": SI.explained_share(num, den, dd, K0 & ch), "size_K": int(nd.K.sum()),
         "size_K_E268": int(K0.sum()), "size_both": int((nd.K & K0).sum()), "n_changed": int(ch.sum()),
         "n_changed_K": int((nd.K & ch).sum()), "n_vars": feat.n, "n_windows": len(toks), "n_read_positions": nd.run.top.shape[1],
         "n_pairs": nd.n_pairs, "shares_of_dW2": KV.norm_shares(feat)["total"], "scale": scales(feat)}  # fmt: skip
    np.savez_compressed(RESULTS / "large" / f"{SPLIT}_{fname(item)}.npz", in_K=nd.K.numpy(), first_flip=nd.first_flip.numpy())
    del F, nd
    torch.cuda.empty_cache()
    return r | {"seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}


def main(shard: tuple[int, int], tag: str) -> int:
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E268 and E329
    log(f"{EXPERIMENT} start, shard {shard}, tag {tag}, card {torch.cuda.get_device_name()}; settings {asdict(cfg)}")
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    w = R1.E268.world()
    assert not w.w0_check, f"the decomposition's target model is not W0 at {w.w0_check[:3]}: Sigma would be read on another model"
    out = R1.Lines(f"run2_{tag}")
    stage0(w, cfg, out)
    log("STAGE 0 DONE")
    todo = list(cfg.items)[shard[0] :: shard[1]]
    log(f"stage 2: {todo}")
    for k, item in enumerate(todo):
        if not out.has("2", SPLIT, item):
            out.add(read(w, cfg, item))
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: its lines go to results/items_run2_<tag>.jsonl")
    args = ap.parse_args()
    require_filed(PRED)
    hard_exit(main((args.shard[0], args.shard[1]), args.tag))

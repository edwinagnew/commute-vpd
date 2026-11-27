"""Cell P1.67 (user, 2026-11-27: "yes, open it for regating-lead, built edits first"): rho at VPD, the write share E_b
over the needed set K (E268, experiments/kstar_needed), under the adjoint split beside the least-squares split, on
E268's built edits (tab:built). Settings and expected values in results/predictions.json, which the run refuses unless
committed unchanged.

The only change between the two arms is `commute.kstar_vpd.featurize(..., split=)`: each matrix's delta-W is split over
its subcomponents by the least-squares split (E268) or by `split.adjoint_split` unscaled (c = 1), and the remainder takes
the rest, so the pieces sum to delta-W in both. K is then found by E268's single-variable interchange
(`commute.needed.needed_set`, E268's settings) on the arm's own featurization, and E_b over K is
`commute.stand_in.explained_share` on a write pass (`stand_in.write_change`) over every position of D in F's own run.

One job, in this order; any miss stops it (a `.stopped` marker beside the line file):
    0  controls: (a) the do-nothing (F = M) under both splits: no changed variable, every dd exactly 0, E_b exactly 1;
       (b) the identity interchange (x' = x) on `control` under the adjoint featurization: 0 argmax changes (its
       largest logit deviation is recorded); (c) least squares on every item: a fresh write pass scored over E268's
       saved K gives E268's E_b_K within repro_rel_tol
    1  least squares, `rescan`: K scanned again, E_b over it against E268's (E268's scan, reproduced)
    2  adjoint, every item: K scanned, E_b over it; also E_b over E268's least-squares K (the split's effect on the
       write share with K held fixed) and the summed write, read and outside shares of |delta-W|^2
results/items_<tag>.jsonl: one line per reading as it lands, with its seconds; a rerun skips lines already written.
results/large/adjoint_<item>.npz: the adjoint K and the derangement of its first change.

    PRED_SHA=<sha> python experiments/adjoint_rho/run.py --shard 0 3 --tag p0
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import kstar_vpd as KV
from commute import needed as IX
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

E268: Any = load_experiment("kstar_needed", "run")  # world(), fname, log, stop; its items' saved K and E_b_K
SAVED = EXPERIMENTS / "kstar_needed" / "results"
RESULTS = Path(__file__).resolve().parent / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, fname = E268.log, E268.fname


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    items: tuple[str, ...]  # read under both splits, in this order
    control: str  # the identity interchange's item
    rescan: str  # the item whose least-squares K is scanned again
    derangements: int  # E268's scan settings, unchanged
    examples: int
    seed: int
    rows: int
    logit_rows: int
    cache_batch: int
    ident_vars: int
    gain_batch: int  # windows per batch of the write pass (E247's)
    repro_rel_tol: float

    def __post_init__(self) -> None:
        assert {self.control, self.rescan} <= set(self.items) and "do_nothing" not in self.items
        assert self.derangements >= 1 and self.examples >= 1 and self.rows >= 1 and self.logit_rows >= self.rows
        assert self.cache_batch >= 1 and self.ident_vars >= 1 and self.gain_batch >= 1 and 0 < self.repro_rel_tol < 1


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


class Lines:
    """results/items_<tag>.jsonl, one reading per line, keyed by (stage, split, item)."""

    def __init__(self, tag: str) -> None:
        self.path = RESULTS / f"items_{tag}.jsonl"
        rows = [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.is_file() else []
        self.done = {(r["stage"], r["split"], r["item"]) for r in rows}

    def has(self, stage: str, split: str, item: str) -> bool:
        return (stage, split, item) in self.done

    def add(self, rec: dict[str, Any], miss: str | None = None) -> None:
        rec = {"experiment": EXPERIMENT} | rec | ({"stopped": miss} if miss else {})
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"LINE {json.dumps(rec)}")
        if miss:
            self.path.with_suffix(".stopped").write_text(miss + "\n")
            log(f"STOPPED: {miss}")
            hard_exit(1)
        self.done.add((rec["stage"], rec["split"], rec["item"]))


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


def saved(item: str) -> tuple[torch.Tensor, float]:
    """E268's K (bool, (n,)) and E_b over it."""
    K = torch.from_numpy(np.load(SAVED / "large" / f"items_{fname(item)}.npz")["in_K"]).bool()
    return K, json.loads((SAVED / "items" / f"{fname(item)}.json").read_text())["E_b_K"]


def featurized(w: Any, item: str, split: str) -> tuple[Any, KV.Feat, torch.Tensor, slice]:
    F, M, toks, at, _ = w.make(item)
    return F, KV.featurize(M, F, w.uv, w.delta, w.bases, split), toks, at


def write_pass(F: Any, feat: KV.Feat, toks: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(cfg.gain_batch)))
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    return num, den, dd, ch


def stage0(w: Any, cfg: Config, out: Lines) -> None:
    for split in KV.SPLITS:
        if not out.has("0", split, "do_nothing"):
            t0 = time.time()
            F, feat, toks, _ = featurized(w, "do_nothing", split)
            num, den, dd, ch = write_pass(F, feat, toks, cfg)
            r = {"stage": "0", "split": split, "item": "do_nothing", "n_changed": len(feat.changed), "dd_max": float(dd.abs().max()),
                 "E_b": SI.explained_share(num, den, dd, ch), "seconds": time.time() - t0}  # fmt: skip
            out.add(r, None if (r["n_changed"], r["dd_max"], r["E_b"]) == (0, 0.0, 1.0) else "do-nothing is not exactly unchanged")
    if not out.has("0", "adjoint", f"identity:{cfg.control}"):
        t0 = time.time()
        F, feat, toks, at = featurized(w, cfg.control, "adjoint")
        idn = IX.identity_scan(F, feat, toks, at, cfg.examples, cfg.ident_vars, cfg.rows, cfg.logit_rows, cfg.cache_batch, log)
        out.add({"stage": "0", "split": "adjoint", "item": f"identity:{cfg.control}", **idn, "seconds": time.time() - t0},
                f"identity interchange: {idn['flips']} argmax changes" if idn["flips"] else None)  # fmt: skip
    for item in cfg.items:
        if out.has("0", "least_squares", item):
            continue
        t0 = time.time()
        K, eb_saved = saved(item)
        F, feat, toks, _ = featurized(w, item, "least_squares")
        num, den, dd, ch = write_pass(F, feat, toks, cfg)
        eb = SI.explained_share(num, den, dd, K & ch)
        r = {"stage": "0", "split": "least_squares", "item": item, "E_b_K_saved_K": eb, "E_b_K_E268": eb_saved,
             "rel_dev": rel(eb, eb_saved), "seconds": time.time() - t0}  # fmt: skip
        out.add(r, f"{item}: E_b over E268's K {eb!r}, E268 {eb_saved!r}" if r["rel_dev"] > cfg.repro_rel_tol else None)


def read(w: Any, cfg: Config, item: str, split: str) -> dict[str, Any]:
    """K on the split's featurization and E_b over it; E268's K and E_b beside it."""
    t0 = time.time()
    F, feat, toks, at = featurized(w, item, split)
    nd = IX.needed_set(F, feat, toks, at, cfg.derangements, cfg.examples, cfg.seed, cfg.rows, cfg.logit_rows, cfg.cache_batch, log)
    num, den, dd, ch = write_pass(F, feat, toks, cfg)
    K0, eb0 = saved(item)
    r = {"split": split, "item": item, "E_b_K": SI.explained_share(num, den, dd, nd.K & ch), "E_b_K_E268": eb0,
         "E_b_over_E268_K": SI.explained_share(num, den, dd, K0 & ch), "size_K": int(nd.K.sum()), "size_K_E268": int(K0.sum()),
         "size_both": int((nd.K & K0).sum()), "n_changed": int(ch.sum()), "n_changed_K": int((nd.K & ch).sum()),
         "n_vars": feat.n, "n_windows": len(toks), "n_read_positions": nd.run.top.shape[1], "n_pairs": nd.n_pairs,
         "shares_of_dW2": KV.norm_shares(feat)["total"]}  # fmt: skip
    np.savez_compressed(RESULTS / "large" / f"{split}_{fname(item)}.npz", in_K=nd.K.numpy(), first_flip=nd.first_flip.numpy())
    del F, nd
    torch.cuda.empty_cache()
    return r | {"seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}


def main(shard: tuple[int, int], tag: str) -> int:
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E268
    log(f"{EXPERIMENT} start, shard {shard}, tag {tag}, card {torch.cuda.get_device_name()}; settings {asdict(cfg)}")
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    w, out = E268.world(), Lines(tag)
    stage0(w, cfg, out)
    log("STAGE 0 DONE")
    if shard[0] == 0 and not out.has("1", "least_squares", cfg.rescan):
        r = read(w, cfg, cfg.rescan, "least_squares")
        out.add({"stage": "1"} | r, f"{cfg.rescan}: rescanned E_b {r['E_b_K']!r}, E268 {r['E_b_K_E268']!r}"
                if rel(r["E_b_K"], r["E_b_K_E268"]) > cfg.repro_rel_tol else None)  # fmt: skip
    todo = list(cfg.items)[shard[0] :: shard[1]]
    log(f"stage 2: {todo}")
    for k, item in enumerate(todo):
        if not out.has("2", "adjoint", item):
            out.add({"stage": "2"} | read(w, cfg, item, "adjoint"))
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: its lines go to results/items_<tag>.jsonl")
    args = ap.parse_args()
    require_filed(PRED)
    hard_exit(main((args.shard[0], args.shard[1]), args.tag))

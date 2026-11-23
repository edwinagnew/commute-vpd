"""E278 (cell P1.57; whip-lead, 2026-11-23, the drop audit the user asked for): E_b over the needed set K at the VPD
featurization on two item sets that were dropped before the frozen definition read them: E62's copy and lookup
fine-tunes (the user's worked example) and E237's P2.50 key grid with its matched randoms. Descriptive; card
experiments/eb_dropped/card.md; settings and the filed stage-0 expected values in results/predictions.json, which the
run refuses unless committed unchanged.

K and E_b over K are read as E275 read them (experiments/claim2_needed/run.py, whose functions are not callable here
without its own config, so the few calls are repeated): E268's filed scan settings (`kstar_needed` load_config),
`commute.needed.needed_set` at the last read position, one `stand_in.write_change` pass, E257's World (rb_states,
through E242's World, which builds E237's items through E237's own World and c).

An item already read with the same construction is not read again; its existing reading is written to
results/items/<item>.json with its source (no GPU):
    e62    E62's copy/lookup at own size, seeds 0-2 (E242's settings e62): all six read by E275 from E257's arrays
           of the same item in the same World (E62's generator at half 7, 8-token windows, the marker last)
    grid   E237's 15 '<state>_scaled' (E147's update times E237's c): read by E275 from E257's arrays, whose item JSON
           holds the same state and E237's c; E237's 15 'random_<state>' (Gaussian at the state's per-matrix norms,
           times the same c, on the state's D): read here, E_b over K and over E237's saved K*
A stage with something to read runs stage 0 first (skipped once results/stage0_<tag>.json says done); any miss stops
the job:
    (a) identity interchange (x' = x) on the control item: no argmax change; its largest logit deviation is the floor
    (b) the do-nothing (F = M): no changed variable, every dd exactly 0, E_b over K exactly 1
Each random item also stops the job unless its n_vars, n_changed and c are E237's saved ones (R0c).

    python experiments/eb_dropped/run.py --stage 0 --tag s0
    python experiments/eb_dropped/run.py --stage e62                     (no GPU: every E62 item is a prior reading)
    python experiments/eb_dropped/run.py --stage grid --shard 0 5 --tag g0
    python experiments/eb_dropped/run.py --stage summary                 (no GPU)
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
from torch import Tensor

from commute import kstar_vpd as KV
from commute import needed as IX
from commute import rule_retention as R
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)

KN: Any = load_experiment("kstar_needed", "run")  # E268: world, load_config (scan settings), featurized, rel
E242 = KN.RB.E242
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E257, E237 = (EXPERIMENTS / e / "results" for e in ("rb_states", "key_grid"))
E275 = EXPERIMENTS / "claim2_needed" / "results" / "items"
log, fname = KN.log, KN.fname
STAGES = ("e62", "grid")


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    control: str  # a grid random item: the identity control (a), on the read path of the items read here
    e62: tuple[str, ...]
    too_small: int  # |K| under this reads "too small to read"
    rival_gap: float  # the card's smallest gap (summary only)
    random_max: float  # the randoms' expected ceiling (summary only)

    def __post_init__(self) -> None:
        assert self.control.startswith("random_e58/") and self.e62 == E242.load_config()[0].e62 and self.too_small > 0


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    write_stage(path, out | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


# ---- items and prior readings ---------------------------------------------------------------------------------------


def items(cfg: Config, stage: str) -> list[str]:
    if stage == "e62":
        return list(cfg.e62)
    scaled = E242.key_items()  # E237's '<state>_scaled', in E237's state order
    return scaled + [f"random_{s.removesuffix('_scaled')}" for s in scaled]


def task(item: str) -> str:
    return item.removeprefix("random_").removeprefix("e58/").removeprefix("e62/").removesuffix("_scaled").rsplit("_s", 1)[0]


def same_construction(item: str, rec: dict[str, Any]) -> bool:
    """A prior reading is of this item's state when it read E257's arrays of the item, and E257's item JSON holds the
    state and c this run's World builds (E237's c for a scaled grid state, own size for E62)."""
    rb = E257 / "items" / f"{fname(item)}.json"
    if rec.get("arrays") != f"experiments/rb_states/results/large/items_{fname(item)}.npz" or not rb.is_file():
        return False
    j = json.loads(rb.read_text())
    if item.startswith("e62/"):
        return j["item"] == item and j.get("c") is None
    c237 = json.loads((E237 / "stage0.json").read_text())["c"][task(item)]
    return j["item"] == item and j["state"] == item.removesuffix("_scaled") and j["c"] == c237


def prior(item: str) -> dict[str, Any] | None:
    """The item's reading by E275 (E268's scan settings, through kstar_needed load_config) with this construction, or
    None. E268 (kstar_needed/results/summary.json) read none of these items: its are E247's built edits, RL and P2.56."""
    p = E275 / f"{fname(item)}.json"
    if p.is_file() and "stopped" not in (rec := json.loads(p.read_text())) and same_construction(item, rec):
        return rec | {"source": str(p.relative_to(EXPERIMENTS.parent))}
    return None


def record_prior(item: str, rec: dict[str, Any], cfg: Config) -> None:
    keep = ("E_b_K", "E_b_Kstar", "size_K", "n_changed_K", "n_vars", "n_changed", "min_margin", "by_derangement", "seconds", "cases_per_s", "empty_K")
    out = {"experiment": EXPERIMENT, "item": item, "reading": "prior", "source": rec["source"], "arrays": rec.get("arrays")}
    out |= {k: rec.get(k) for k in keep} | {"median_margin": None, "too_small": rec["size_K"] < cfg.too_small}
    write_stage(RESULTS / "items" / f"{fname(item)}.json", out)
    log(f"{item}: prior reading from {rec['source']}: E_b over K {rec['E_b_K']:.6f}, |K| {rec['size_K']}")


# ---- one item read here ---------------------------------------------------------------------------------------------


def make(w: Any, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
    """E242's World routes 'e58/' items to E237's World (w.kg, loaded on the first such item); a random goes there directly."""
    if not item.startswith("random_"):
        return w.make(item)
    if w.kg is None:
        w.make(item.removeprefix("random_") + "_scaled")
    kg: Any = w.kg
    F, toks, at, rec = kg.make(item)
    return F, kg.M, toks, at, rec


def changed_mask(feat: KV.Feat) -> Tensor:
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    return ch


def read_item(w: Any, item: str, cfg: Config, kc: Any, floor: float) -> None:
    path = RESULTS / "items" / f"{fname(item)}.json"
    t0 = time.time()
    F, M, toks, at, rec = make(w, item)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "reading": "read", "c": rec.get("c"), "state": rec.get("state"),
                           "read_positions": "last" if at == KN.KP.LAST else str(at), "n_vars": feat.n, "n_changed": len(feat.changed)}  # fmt: skip
    saved = json.loads((E237 / "vpd" / f"{fname(item)}.json").read_text())
    if (feat.n, len(feat.changed), rec.get("c")) != (saved["n_vars"], saved["n_changed"], saved["c"]):
        stop(path, out, f"{item}: n_vars, n_changed, c {(feat.n, len(feat.changed), rec.get('c'))} not E237's {(saved['n_vars'], saved['n_changed'], saved['c'])}")
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
    nd = IX.needed_set(F, feat, toks, at, kc.derangements, kc.examples, kc.seed, kc.rows, kc.logit_rows, kc.cache_batch, log)
    K, ch, mg = nd.K, changed_mask(feat), nd.run.margin
    out |= {"n_windows": len(toks), "window_len": toks.shape[1], "n_pairs": nd.n_pairs, "E_b_K": SI.explained_share(num, den, dd, K & ch),
            "size_K": int(K.sum()), "n_changed_K": int((K & ch).sum()), "empty_K": not bool((K & ch).any()),
            "too_small": int(K.sum()) < cfg.too_small, "median_margin": float(mg.median()), "min_margin": float(mg.min()),
            "n_fragile_positions": int((mg <= floor).sum()), "fragile_floor": floor,
            "by_derangement": [{"size_K": int((K & (nd.first_flip <= p)).sum()), "n_changed_K": int((K & (nd.first_flip <= p) & ch).sum()),
                                "E_b_K": SI.explained_share(num, den, dd, K & (nd.first_flip <= p) & ch)} for p in range(kc.derangements)]}  # fmt: skip
    ks = KV.saved_k_star(item)
    if ks is not None:
        Ks = torch.from_numpy(ks[0])
        out |= {"E_b_Kstar": SI.explained_share(num, den, dd, Ks & ch), "size_Kstar": int(Ks.sum()), "size_both": int((K & Ks).sum()),
                "Kstar_file": str(ks[1].relative_to(EXPERIMENTS.parent))}  # fmt: skip
    np.savez_compressed(RESULTS / "large" / f"items_{fname(item)}.npz", in_K=K.cpu().numpy(), first_flip=nd.first_flip.cpu().numpy(), flips=nd.scan.flips.cpu().numpy(),
                        num=num.cpu().numpy(), den=den.cpu().numpy(), dd=dd.cpu().numpy(), changed=ch.cpu().numpy(), margin=mg.cpu().numpy())  # fmt: skip
    out |= {"seconds": time.time() - t0, "cases_per_s": nd.scan.rows / nd.scan.seconds, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    write_stage(path, out)
    log(f"{item}: E_b over K {out['E_b_K']:.6f}, over K* {out.get('E_b_Kstar', float('nan')):.6f}; |K| {out['size_K']} "
        f"({out['n_changed_K']} changed); margin median {out['median_margin']:.4g}, min {out['min_margin']:.4g}; {out['seconds']:.0f} s")  # fmt: skip
    del F, nd
    torch.cuda.empty_cache()


# ---- stage 0, summary, main -----------------------------------------------------------------------------------------


def stage0(w: Any, cfg: Config, kc: Any, tag: str) -> float:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        log(f"{path.name}: done before")
        return float(done["floor"])
    t0 = time.time()
    out: dict[str, Any] = {"experiment": EXPERIMENT, "card": torch.cuda.get_device_name(), "control": cfg.control}
    F, feat, toks, _ = KN.featurized(w, "do_nothing")
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
    out["do_nothing"] = {"n_changed": len(feat.changed), "dd_max": float(dd.abs().max()),
                         "E_b_K": SI.explained_share(num, den, dd, torch.ones(feat.n, dtype=torch.bool) & changed_mask(feat))}  # fmt: skip
    del F
    F, M, toks, at, _ = make(w, cfg.control)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    out["identity"] = idn = IX.identity_scan(F, feat, toks, at, kc.examples, kc.ident_vars, kc.rows, kc.logit_rows, kc.cache_batch, log)
    del F
    out["est_seconds_per_item"] = idn["n_vars"] * kc.derangements * min(kc.examples, idn["n_windows"]) / idn["cases_per_s"]
    out |= {"floor": idn["max_dev"], "seconds": time.time() - t0}
    dn = out["do_nothing"]
    miss = [f"identity: {idn['flips']} argmax changes"] if idn["flips"] else []
    miss += [f"do-nothing: {k} {dn[k]}" for k, want in (("n_changed", 0), ("dd_max", 0.0), ("E_b_K", 1.0)) if dn[k] != want]
    log(f"stage 0: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    write_stage(path, out | {"done": True})
    log(f"STAGE 0 DONE: est. {out['est_seconds_per_item']:.0f} s per item")
    return float(out["floor"])


def summary(cfg: Config) -> None:
    """Every item JSON and stage-0 file into results/summary.json, with E62's copy-minus-lookup gap per seed and the
    grid per task (seeds scaled, beside their randoms). Works with any subset written (the grid may be absent)."""
    got = {r["item"]: r for r in (json.loads(p.read_text()) for p in sorted((RESULTS / "items").glob("*.json")))}
    eb = {i: r["E_b_K"] for i, r in got.items() if "stopped" not in r}
    out: dict[str, Any] = {"experiment": EXPERIMENT, "stage0": [json.loads(p.read_text()) for p in sorted(RESULTS.glob("stage0_*.json"))], "items": got}
    seeds = sorted({int(i.rsplit("_s", 1)[1]) for i in cfg.e62})
    gap = {s: eb[f"e62/copy_s{s}"] - eb[f"e62/lookup_s{s}"] for s in seeds if f"e62/copy_s{s}" in eb and f"e62/lookup_s{s}" in eb}
    out["e62"] = {"copy": [eb.get(f"e62/copy_s{s}") for s in seeds], "lookup": [eb.get(f"e62/lookup_s{s}") for s in seeds],
                  "gap_copy_minus_lookup": [gap.get(s) for s in seeds],
                  "mean_gap": sum(gap.values()) / len(gap) if len(gap) == len(seeds) else None}  # fmt: skip
    if out["e62"]["mean_gap"] is not None:
        out["e62"]["reads"] = "rival A (gap >= rival_gap)" if out["e62"]["mean_gap"] >= cfg.rival_gap else "rival B (gap < rival_gap)"
    grid: dict[str, dict[str, list[float | None]]] = {}
    for i in (i for i in eb if i.startswith(("e58/", "random_e58/"))):
        side = "random" if i.startswith("random_") else "scaled"
        grid.setdefault(task(i), {"scaled": [None] * 3, "random": [None] * 3})[side][int(i.rsplit("_s", 1)[1].removesuffix("_scaled"))] = eb[i]
    out["grid"] = grid
    rnd = [v for t in grid.values() for v in t["random"] if v is not None]
    out["randoms_at_or_under_random_max"] = all(v <= cfg.random_max for v in rnd) if rnd else None
    out["too_small"] = sorted(i for i, r in got.items() if r.get("too_small"))
    write_stage(RESULTS / "summary.json", out)
    log(f"summary: {len(got)} items; E62 {json.dumps(out['e62'])}; grid tasks {sorted(grid)}")


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    cfg, kc = load_config(), KN.load_config()
    if stage == "summary":
        summary(cfg)
        return 0
    for d in ("items", "large"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    every = [] if stage == "0" else items(cfg, stage)
    todo, read = every[shard[0] :: shard[1]], []
    for item in todo:
        p = RESULTS / "items" / f"{fname(item)}.json"
        if p.is_file() and "stopped" not in json.loads(p.read_text()):
            log(f"{item}: done before, skipped")
        elif (rec := prior(item)) is not None:
            record_prior(item, rec, cfg)
        else:
            read.append(item)
    log(f"{EXPERIMENT} stage {stage}, shard {shard}, tag {tag}: {len(read)} of {len(todo)} items to read {read}")
    if stage != "0" and not read:
        log(f"{EXPERIMENT} DONE (nothing to read)")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E257, E268, E275
    log(f"settings {asdict(cfg)}; E268's scan settings {asdict(kc)}")
    w = KN.world()
    floor = stage0(w, cfg, kc, tag)
    for k, item in enumerate(read):
        read_item(w, item, cfg, kc, floor)
        log(f"ITEM {k + 1} of {len(read)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("0", "config", *STAGES, "summary"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    if args.stage == "config":  # CPU dry run: settings, item lists and which items are prior readings; no model
        c = load_config()
        lists = {s: {i: (p["source"] if (p := prior(i)) else "read here") for i in items(c, s)} for s in STAGES}
        print(json.dumps({"settings": asdict(c), "E268_scan": asdict(KN.load_config()), "items": lists}, indent=1))
        hard_exit(0)
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

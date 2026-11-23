"""E275 (cell P2.62; whip-lead on the user's approval 2026-11-22 ~18:15): E_b over the needed set K for the claim-2
states that E268 (P1.54, experiments/kstar_needed) did not read, so that claim 2's comparisons can be recomputed over K
beside K* (compare.py, CPU). Descriptive; card experiments/claim2_needed/card.md; settings and the filed stage-0
expected values in results/predictions.json, which the run refuses unless committed unchanged.

K is found as E268 found it, with E268's filed scan settings (`kstar_needed` load_config: 4 derangements x 64 windows,
seed 268, its batch sizes) and the shared sequence `commute.needed.needed_set` (E268's calls in E268's order). F, M,
D and the featurization are E257's (`rb_states` World through `kstar_needed.world`, `kstar_needed.featurized`); every
item is read at the last position, as E257 read it. Each item also records K from derangements {0, 1} only and {2, 3}
only (`needed.scan`'s group_mask, saved in the npz) for compare.py's independent-halves sampling floor.

Items:
    pairs   the 76 states of E257's nine built reuse/new pairs (rb_states/results/summary_items.json), from E257's saved
            num, den, dd, changed and K* (`kstar_needed.saved`): E_b over K and over K* (the latter must repeat E257's
            saved E_b within transcription_rel_tol, as in E268)
    prior   P2.57's ten-seed prior-matched pair (E254: key copy, prior-matched cipher), all 20 seeds built fresh at the
            size every cheap reading of E254 used: E254's saved update (prior_matched/results/large/dW_pa_<task>_s<seed>.pt)
            times E254's c for that state (its item JSON), on E257's D of the same task (item e58/<task>_s0_scaled);
            one write pass (`stand_in.write_change`), E_b over K only (no K* exists at this size). E257's own
            e58/<task>_s<seed>_scaled items (seeds 0-2, c differs from E254's by up to 0.4%) are read in `rest`
    rest    the other 18 of E248's 94 items (E257's stage "items"), as pairs: the range of E_b over the 94 fine-tunes,
            and the K* against K comparison on seeds 0-2 of the prior-matched tasks
Stage 0 runs first in every job (skipped once results/stage0_<tag>.json says done); any miss stops the job:
    (a) identity interchange (x' = x) on the control pair item: no argmax change; its largest logit deviation is the floor
    (b) the do-nothing (F = M): no changed variable, every dd exactly 0, E_b over K exactly 1
    (c) the control pair item's E_b over K* from E257's saved arrays repeats E257's saved E_b within transcription_rel_tol
    (d) E254's seed 0 of copy built fresh as the prior stage builds seeds 3-9: its variables and changed set equal E257's
        e58/copy_s0_scaled arrays, and its E_b over that item's saved K* is within construction_rel_tol of E257's E_b

    python experiments/claim2_needed/run.py --stage 0 --tag c0
    python experiments/claim2_needed/run.py --stage pairs --shard 0 8 --tag c1
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
from commute import old_tasks as OT
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

KN: Any = load_experiment("kstar_needed", "run")  # E268: world, load_config, saved, featurized, readings, rel, RB
RB, KP = KN.RB, KN.KP
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E257 = EXPERIMENTS / "rb_states" / "results"
PM = EXPERIMENTS / "prior_matched" / "results"
log, fname = KN.log, KN.fname
STAGES = ("pairs", "prior", "rest")


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    control: str  # a pair item with saved arrays: identity control (a) and transcription check (c)
    transcription_rel_tol: float
    prior_tasks: tuple[str, ...]  # E254's two tasks; D is E257's item e58/<task>_s0_scaled's
    prior_seeds: tuple[int, ...]  # E254's seeds, each built fresh from its saved update
    construction_rel_tol: float

    def __post_init__(self) -> None:
        assert self.control in pairs() and self.prior_tasks == ("copy", "prior_cipher") and 0 in self.prior_seeds
        assert 0 < self.transcription_rel_tol < self.construction_rel_tol < 1


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    write_stage(path, out | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


# ---- items ----------------------------------------------------------------------------------------------------------


def pairs() -> list[str]:
    s = json.loads((E257 / "summary_items.json").read_text())
    return sorted({it["item"] for p in s["pairs"].values() for it in p["reuse"]["items"] + p["new"]["items"]})


def saved_item(task: str, seed: int) -> str:
    return f"e58/{task}_s{seed}_scaled"


def fresh_item(task: str, seed: int) -> str:
    return f"pa/{task}_s{seed}"


def items(w: Any, cfg: Config, stage: str) -> list[str]:
    if stage == "pairs":
        return pairs()
    if stage == "prior":
        return [fresh_item(t, s) for s in cfg.prior_seeds for t in cfg.prior_tasks]
    return [i for i in RB.stage_items(w, RB.load_config(), "items") if i not in set(pairs())]


def fresh(w: Any, item: str) -> tuple[R.Model, KV.Feat, Tensor, dict[str, Any]]:
    """F = M + E254's saved update for pa/<task>_s<seed> times its c (prior_matched World.dw), on E257's D of the task."""
    task, seed = item.removeprefix("pa/").rsplit("_s", 1)
    f, rec = PM / "large" / f"dW_pa_{task}_s{seed}.pt", PM / "a" / "items" / f"e58-pa_{task}_s{seed}.json"
    c = json.loads(rec.read_text())["c"]
    dw = torch.load(f, weights_only=True, map_location="cpu")
    assert sorted(dw) == sorted(w.kl.net.W0), f"{f.name}: not the 24 decomposed matrices"
    F = R.Model(OT.with_delta(w.kl.net, {f"{k}.weight": v.float() * c for k, v in dw.items()}).net)
    _, _, toks, at, _ = w.make(saved_item(task, 0))  # E257's D of the task (every seed of an item group reads one D)
    assert at == KP.LAST, item
    return F, KV.featurize(w.kl.M, F, w.uv, w.delta, w.bases), toks, {"dw": str(f.relative_to(EXPERIMENTS.parent)), "c": c}


def changed_mask(feat: KV.Feat) -> Tensor:
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    return ch


# ---- one item -------------------------------------------------------------------------------------------------------


def arrays(w: Any, item: str, kc: Any) -> tuple[R.Model, KV.Feat, Tensor, slice, dict[str, Any]]:
    """F, its featurization, D, the read positions and the per-variable arrays: E257's saved ones (with K* and E257's
    E_b), or a fresh write pass for a pa/ item (no K*)."""
    if item.startswith("pa/"):
        F, feat, toks, rec = fresh(w, item)
        num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
        return F, feat, toks, KP.LAST, {"num": num, "den": den, "dd": dd, "changed": changed_mask(feat), "Kstar": None} | rec
    a = KN.saved(item, "items")
    assert a is not None, f"{item}: no saved E257 arrays"
    F, feat, toks, at = KN.featurized(w, item)
    assert len(a["dd"]) == feat.n and torch.equal(changed_mask(feat), a["changed"]), f"{item}: saved arrays do not line up"
    return F, feat, toks, at, a


def read_item(w: Any, item: str, cfg: Config, kc: Any, floor: float) -> None:
    path = RESULTS / "items" / f"{fname(item)}.json"
    if path.is_file() and "stopped" not in json.loads(path.read_text()):
        log(f"{item}: done before, skipped")
        return
    t0 = time.time()
    F, feat, toks, at, a = arrays(w, item, kc)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "arrays": a.get("src") or a.get("dw"), "c": a.get("c")}
    nd = IX.needed_set(F, feat, toks, at, kc.derangements, kc.examples, kc.seed, kc.rows, kc.logit_rows, kc.cache_batch, log)
    K, ch, num, den, dd = nd.K, a["changed"], a["num"], a["den"], a["dd"]
    out |= {"n_windows": len(toks), "window_len": toks.shape[1], "n_vars": feat.n, "n_changed": int(ch.sum()), "n_pairs": nd.n_pairs,
            "min_margin": float(nd.run.margin.min()), "n_fragile_positions": int((nd.run.margin <= floor).sum()), "fragile_floor": floor,
            "E_b_K": SI.explained_share(num, den, dd, K & ch), "size_K": int(K.sum()), "n_changed_K": int((K & ch).sum()),
            "empty_K": not bool((K & ch).any()),
            "by_derangement": [{"size_K": int((K & (nd.first_flip <= p)).sum()), "n_changed_K": int((K & (nd.first_flip <= p) & ch).sum()),
                                "E_b_K": SI.explained_share(num, den, dd, K & (nd.first_flip <= p) & ch)}
                               for p in range(kc.derangements)]}  # fmt: skip
    # by_derangement in E268's form (K from derangements 0..p); halves: K from derangements {0, 1} only and {2, 3} only
    # (scan's group_mask), compare.py's independent-halves sampling floor
    gm = nd.scan.group_mask
    assert gm is not None and kc.derangements == 4 and torch.equal(gm > 0, K), f"{item}: group mask does not give K"
    out["halves"] = {name: {"size_K": int(h.sum()), "n_changed_K": int((h & ch).sum()), "E_b_K": SI.explained_share(num, den, dd, h & ch)}
                     for name, h in (("01", (gm & 0b0011) > 0), ("23", (gm & 0b1100) > 0))}  # fmt: skip
    if a["Kstar"] is not None:
        out |= KN.readings(a, K, kc.kept_floor) | {"E_b_Kstar_saved": a["E_b_saved"], "E_b_Kstar_rel_dev": KN.rel(out["E_b_Kstar"], a["E_b_saved"])}
    np.savez_compressed(RESULTS / "large" / f"items_{fname(item)}.npz", in_K=K.numpy(), first_flip=nd.first_flip.numpy(), flips=nd.scan.flips.numpy(),
                        group_mask=gm.numpy(), **({"num": num.numpy(), "den": den.numpy(), "dd": dd.numpy(), "changed": ch.numpy()} if a["Kstar"] is None else {}))  # fmt: skip
    out |= {"seconds": time.time() - t0, "cases_per_s": nd.scan.rows / nd.scan.seconds, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    if out.get("E_b_Kstar_rel_dev", 0.0) > cfg.transcription_rel_tol:
        stop(path, out, f"{item}: E_b over K* {out['E_b_Kstar']!r} off E257's saved {a['E_b_saved']!r}")
    write_stage(path, out)
    log(f"{item}: E_b over K {out['E_b_K']:.6f}, over K* {out.get('E_b_Kstar', float('nan')):.6f}; |K| {out['size_K']} "
        f"({out['n_changed_K']} changed); {out['seconds']:.0f} s")  # fmt: skip
    del F, nd
    torch.cuda.empty_cache()


# ---- stage 0, main --------------------------------------------------------------------------------------------------


def construction(w: Any, cfg: Config, kc: Any) -> dict[str, Any]:
    """(d): E254's copy seed 0 built as the prior stage builds seeds 3-9, against E257's arrays of the same state."""
    F, feat, toks, rec = fresh(w, fresh_item("copy", 0))
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
    a = KN.saved(saved_item("copy", 0), "items")
    del F
    same = len(dd) == len(a["dd"]) and torch.equal(changed_mask(feat), a["changed"])
    eb = SI.explained_share(num, den, dd, a["Kstar"] & a["changed"]) if same else float("nan")
    return {"c": rec["c"], "same_variables": same, "E_b_Kstar": eb, "E_b_Kstar_saved": a["E_b_saved"], "rel_dev": KN.rel(eb, a["E_b_saved"]),
            "dd_rel_dev": float((dd - a["dd"]).abs().sum() / a["dd"].abs().sum()) if same else None}  # fmt: skip


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
    F, feat, toks, at = KN.featurized(w, cfg.control)
    out["identity"] = IX.identity_scan(F, feat, toks, at, kc.examples, kc.ident_vars, kc.rows, kc.logit_rows, kc.cache_batch, log)
    del F
    a = KN.saved(cfg.control, "items")
    eb = SI.explained_share(a["num"], a["den"], a["dd"], a["Kstar"] & a["changed"])
    out["transcription"] = {"E_b_Kstar": eb, "E_b_Kstar_saved": a["E_b_saved"], "rel_dev": KN.rel(eb, a["E_b_saved"])}
    out["construction"] = construction(w, cfg, kc)
    idn = out["identity"]
    out["est_seconds_per_item"] = idn["n_vars"] * kc.derangements * min(kc.examples, idn["n_windows"]) / idn["cases_per_s"]
    out |= {"floor": idn["max_dev"], "seconds": time.time() - t0}
    dn, tr, co = out["do_nothing"], out["transcription"], out["construction"]
    miss = [f"identity: {idn['flips']} argmax changes"] if idn["flips"] else []
    miss += [f"do-nothing: {k} {dn[k]}" for k, want in (("n_changed", 0), ("dd_max", 0.0), ("E_b_K", 1.0)) if dn[k] != want]
    miss += [f"{cfg.control}: E_b over K* {tr['E_b_Kstar']!r} off E257's {tr['E_b_Kstar_saved']!r}"] if not tr["rel_dev"] <= cfg.transcription_rel_tol else []
    miss += ["construction: variables or changed set differ from E257's"] if not co["same_variables"] else []
    miss += [f"construction: E_b over K* rel. dev. {co['rel_dev']!r}"] if not co["rel_dev"] <= cfg.construction_rel_tol else []
    log(f"stage 0: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    write_stage(path, out | {"done": True})
    log(f"STAGE 0 DONE: est. {out['est_seconds_per_item']:.0f} s per item")
    return float(out["floor"])


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    cfg, kc = load_config(), KN.load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E257, E268
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}; settings {asdict(cfg)}; E268's scan settings {asdict(kc)}")
    for d in ("items", "large"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    w = KN.world()
    floor = stage0(w, cfg, kc, tag)
    if stage == "0":
        return 0
    every = items(w, cfg, stage)
    todo = every[shard[0] :: shard[1]]
    log(f"stage {stage}: {len(todo)} of {len(every)} items {todo}")
    for k, item in enumerate(todo):
        read_item(w, item, cfg, kc, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("0", "config", *STAGES), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    if args.stage == "config":  # CPU dry run: settings load, item lists that need no model
        c = load_config()
        print(json.dumps({"settings": asdict(c), "E268_scan": asdict(KN.load_config()), "n_pairs_items": len(pairs()), "control_in_pairs": c.control in pairs()}, indent=1))
        hard_exit(0)
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

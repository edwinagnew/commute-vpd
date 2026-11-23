"""E268 (cell P1.54; user, 2026-11-22): the needed set K of the frozen definition (theory/2026-11-22_definition_frozen.md,
section 2) by single-variable interchange on D, and the write share E_b re-read over K instead of over K*, the
mean-ablation circuit every saved E_b used (reports/analyses/2026-11-22_analysis_reading-admissibility.md). Settings and
filed expected values in results/predictions.json, which the run refuses unless committed unchanged.

Per item: F, M, D, the featurization and the read positions as E257 built them (experiments/rb_states/run.py World,
through E242's and E226's); K by `commute.needed.scan` over every subcomponent and remainder of the 24 decomposed
matrices, on `derangements` seeded derangements of D, each tested on `examples` windows; a variable is needed when any
tested pair changes F's argmax at any read position. E_b over K is `commute.stand_in.explained_share` on the saved
per-variable num, den, dd (built edits: experiments/thin_edit_diagnosis, E247; the rest: experiments/rb_states, E257)
with sel = K and changed, beside the same over the saved K*. results/items/<item>.json holds the readings;
results/large/items_<item>.npz first_pair, first_flip, flips, flip_frac, in_K, in_Kstar. A rerun skips written items.

Stages, each job running stage 0 first (skipped once results/stage0_<tag>.json says done); any miss stops the job:
    0         (a) identity interchange (x' = x) on the control: no argmax changes; its largest logit deviation is the
              floor, and its rate times the counts estimates each stage's time; (b) the do-nothing (F = M): no changed
              variable, every dd exactly 0, E_b over K exactly 1; (c) E_b over K* of the control from the saved arrays
              equals P1.49's within the filed tolerance
    built     the seven E247 built edits
    rl        E257's 21 saved RL/SFT repeat states (D = E222's read rows)
    manifest  compare-lead's 45 P2.56 states
    pairs     E248's 94 trained items (lowest priority)
    summary   CPU: every item JSON into results/summary.json

    python experiments/kstar_needed/run.py --stage built --tag b0
    python experiments/kstar_needed/run.py --stage rl --shard 0 4 --tag r0
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

RB: Any = load_experiment("rb_states", "run")  # E257: World, load_config, stage_items, e247 paths, log/dump/stop
E242, E226, KP = RB.E242, RB.E226, RB.KP
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
S247, S242, SRB = RB.S247, RB.S242, RB.RESULTS
P149 = EXPERIMENTS / "option_b_rescore" / "results" / "summary.json"
log, dump, stop, fname = RB.log, RB.dump, RB.stop, RB.fname
STAGES = {"built": None, "rl": "rl", "manifest": "manifest", "pairs": "items"}  # ours -> E257's stage name


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    control: str  # the identity control and the transcription check (a built edit)
    control_E_b_Kstar: float  # P1.49's E_b over K* of the control
    transcription_rel_tol: float
    built: tuple[str, ...]
    derangements: int  # P
    examples: int  # windows tested per derangement (all of D when larger)
    seed: int
    rows: int  # cases per forward with one read position
    logit_rows: int  # cases x read positions per forward, at most
    cache_batch: int  # windows per batch of F's cached run and of the do-nothing's write pass
    ident_vars: int  # subcomponents per matrix in the identity control, beside its remainder
    kept_floor: float  # a changed variable is kept when its explained share of dd is at least 1 - kept_floor

    def __post_init__(self) -> None:
        assert self.control in self.built and len(self.built) == 7 and self.derangements >= 1 and self.examples >= 1
        assert self.rows >= 1 and self.logit_rows >= self.rows and self.ident_vars >= 1 and 0 < self.kept_floor < 1


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def world() -> Any:
    """E257's World with the split's bases and E242's matched scale of the KL-matched random (as E257's main)."""
    w = RB.World(RB.load_config())
    w.bases = KV.bases(w.uv, E226.load_config()[0].split_tol)
    r = json.loads((S242 / "items" / f"{fname(E242.load_config()[0].random_kl)}.json").read_text())
    w.kl_x = {k: r[k] for k in ("x", "kl_matched", "match", "match_evals")}
    return w


# ---- saved arrays and readings ---------------------------------------------------------------------------------------


def saved(item: str, stage: str) -> dict[str, Any] | None:
    """num, den, dd, changed, K* (torch, (n,)) and the saved E_b over K*: E247's arrays with E242's changed set and the
    saved K* for the built edits (E_b from P1.49's summary), E257's arrays and item JSON otherwise. None when not saved."""
    f = fname(item)
    if stage == "built":
        e, ks = np.load(S247 / "large" / f"items_{f}.npz"), KV.saved_k_star(item)
        assert ks is not None, item
        a = {k: e[k] for k in ("num", "den", "dd")} | {"changed": np.load(S242 / "large" / f"items_{f}.npz")["changed"], "Kstar": ks[0]}
        src, eb = str(ks[1].relative_to(EXPERIMENTS.parent)), json.loads(P149.read_text())["items"][item]["E_b"]
    else:
        p = SRB / "large" / f"items_{f}.npz"
        if not p.is_file():
            return None
        z = np.load(p)
        a = {k: z[k] for k in ("num", "den", "dd", "changed")} | {"Kstar": z["in_K"]}
        src, eb = str(p.relative_to(EXPERIMENTS.parent)), json.loads((SRB / "items" / f"{f}.json").read_text())["E_b"]
    out: dict[str, Any] = {k: torch.from_numpy(v) for k, v in a.items()}
    out["changed"], out["Kstar"] = out["changed"].bool(), out["Kstar"].bool()
    return out | {"src": src, "E_b_saved": eb}


def readings(a: dict[str, Any], K: torch.Tensor, floor: float) -> dict[str, Any]:
    ch, Ks, dd = a["changed"], a["Kstar"], a["dd"]
    share = torch.where(dd > 0, SI.explained(a["num"], a["den"]) / torch.where(dd > 0, dd, 1.0), float("nan"))
    kept = share >= 1 - floor
    dK = float(dd[K].sum())
    return {"E_b_K": SI.explained_share(a["num"], a["den"], dd, K & ch), "E_b_Kstar": SI.explained_share(a["num"], a["den"], dd, Ks & ch),
            "size_K": int(K.sum()), "size_Kstar": int(Ks.sum()), "size_both": int((K & Ks).sum()),
            "n_changed_K": int((K & ch).sum()), "n_changed_Kstar": int((Ks & ch).sum()), "n_changed_both": int((K & Ks & ch).sum()),
            "n_kept_changed_K": int((K & ch & kept).sum()), "n_kept_changed_Kstar": int((Ks & ch & kept).sum()),
            "dd_share_K_outside_Kstar": float(dd[K & ~Ks].sum()) / dK if dK > 0 else None}  # fmt: skip


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


# ---- one item -------------------------------------------------------------------------------------------------------


def featurized(w: Any, item: str) -> tuple[Any, KV.Feat, torch.Tensor, slice]:
    F, M, toks, at, _ = w.make(item)
    assert toks.dim() == 2, f"{item}: the windows of D must have one length, got {tuple(toks.shape)}"
    return F, KV.featurize(M, F, w.uv, w.delta, w.bases), toks, at


def batch_rows(cfg: Config, n_at: int) -> int:
    return max(1, min(cfg.rows, cfg.logit_rows // n_at))


def read_item(w: Any, item: str, stage: str, cfg: Config, floor: float) -> None:
    path = RESULTS / "items" / f"{fname(item)}.json"
    if path.is_file() and "stopped" not in json.loads(path.read_text()):
        log(f"{item}: done before, skipped")
        return
    a, t0 = saved(item, stage), time.time()
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "stage": stage}
    if a is None:
        dump(path, out | {"reading": "no saved arrays"})
        log(f"{item}: no saved arrays, not read")
        return
    F, feat, toks, at = featurized(w, item)
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    if len(a["dd"]) != feat.n or not torch.equal(ch, a["changed"]):
        stop(path, out, f"{item}: saved arrays ({len(a['dd'])}) do not line up with the featurization ({feat.n} variables, changed set)")
    run = IX.cache(F, feat, toks, at, cfg.cache_batch)
    m = min(cfg.examples, len(toks))
    ex, pt = IX.derangements(len(toks), cfg.derangements, m, cfg.seed)
    n_at = run.top.shape[1]
    log(f"{item}: {len(toks)} windows of {toks.shape[1]} tokens, {n_at} read positions, {feat.n} variables, {len(feat.changed)} changed, "
        f"{len(ex)} pairs ({cfg.derangements} derangements x {m}), {batch_rows(cfg, n_at)} cases per forward")  # fmt: skip
    sc = IX.scan(F, feat, run, at, ex, pt, batch_rows(cfg, n_at), log)
    K, Q = sc.first_pair >= 0, len(ex)
    first_flip = torch.where(K, sc.first_pair // m, -1)
    eb_saved = SI.explained_share(a["num"], a["den"], a["dd"], a["Kstar"] & a["changed"])
    out |= {"reading": "read", "arrays": a["src"], "n_windows": len(toks), "window_len": toks.shape[1], "n_read_positions": n_at,
            "read_positions": "last" if at == KP.LAST else "all", "n_vars": feat.n, "n_tested": sc.rows // Q, "n_pairs": Q,
            "derangements": cfg.derangements, "examples": m, "min_margin": float(run.margin.min()),
            "n_fragile_positions": int((run.margin <= floor).sum()), "fragile_floor": floor} | readings(a, K, cfg.kept_floor)  # fmt: skip
    out |= {"E_b_Kstar_saved": a["E_b_saved"], "E_b_Kstar_rel_dev": rel(eb_saved, a["E_b_saved"]),
            "by_derangement": [{"size_K": int((K & (first_flip <= p)).sum()), "n_changed_K": int((K & (first_flip <= p) & a["changed"]).sum()),
                                "E_b_K": SI.explained_share(a["num"], a["den"], a["dd"], K & (first_flip <= p) & a["changed"])}
                               for p in range(cfg.derangements)]}  # fmt: skip
    rb = SRB / "items" / f"{fname(item)}.json"
    out["status"] = json.loads(rb.read_text()).get("status") if rb.is_file() else None
    np.savez_compressed(RESULTS / "large" / f"items_{fname(item)}.npz", first_pair=sc.first_pair.numpy(), first_flip=first_flip.numpy(),
                        flips=sc.flips.numpy(), flip_frac=(sc.flips.double() / Q).float().numpy(), in_K=K.numpy(), in_Kstar=a["Kstar"].numpy())  # fmt: skip
    out |= {"seconds": time.time() - t0, "cases_per_s": sc.rows / sc.seconds, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    if out["E_b_Kstar_rel_dev"] > cfg.transcription_rel_tol:
        stop(path, out, f"{item}: E_b over K* {eb_saved!r} off the saved {a['E_b_saved']!r}")
    dump(path, out)
    log(f"{item}: E_b over K {out['E_b_K']:.6f}, over K* {out['E_b_Kstar']:.6f}; |K| {out['size_K']} ({out['n_changed_K']} changed, "
        f"{out['n_kept_changed_K']} kept), |K*| {out['size_Kstar']} ({out['n_changed_Kstar']} changed), |K and K*| {out['size_both']}; "
        f"{out['seconds']:.0f} s")  # fmt: skip
    del F, run
    torch.cuda.empty_cache()


# ---- stage 0, main --------------------------------------------------------------------------------------------------


def identity(w: Any, cfg: Config, at: slice | None) -> dict[str, Any]:
    """x' = x on the control's first `examples` windows, `ident_vars` subcomponents and the remainder of every matrix,
    at the control's read positions (at None) or the last position only (KP.LAST: the other stages' reads, for timing)."""
    F, feat, toks, at0 = featurized(w, cfg.control)
    at = at0 if at is None else at
    run = IX.cache(F, feat, toks, at, cfg.cache_batch)
    ex = torch.arange(min(cfg.examples, len(toks)))
    only = {j: torch.cat([torch.arange(min(cfg.ident_vars, s.rem - s.off)), torch.tensor([s.rem - s.off])]) for j, s in enumerate(feat.sites)}
    sc = IX.scan(F, feat, run, at, ex, ex, batch_rows(cfg, run.top.shape[1]), log, only)
    return {"flips": int(sc.flips.sum()), "max_dev": sc.max_dev, "min_margin": float(run.margin.min()), "cases": sc.rows,
            "cases_per_s": sc.rows / sc.seconds, "n_read_positions": run.top.shape[1], "n_windows": len(toks), "window_len": toks.shape[1],
            "n_vars": feat.n}  # fmt: skip


def stage0(w: Any, cfg: Config, tag: str) -> dict[str, Any]:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        log(f"{path.name}: done before")
        return done
    t0 = time.time()
    a = saved(cfg.control, "built")
    assert a is not None
    eb = SI.explained_share(a["num"], a["den"], a["dd"], a["Kstar"] & a["changed"])
    out: dict[str, Any] = {"experiment": EXPERIMENT, "control": cfg.control, "E_b_Kstar": eb, "E_b_Kstar_filed": cfg.control_E_b_Kstar,
                           "E_b_Kstar_rel_dev": rel(eb, cfg.control_E_b_Kstar)}  # fmt: skip
    F, feat, toks, _ = featurized(w, "do_nothing")
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(cfg.cache_batch)))
    allK = torch.ones(feat.n, dtype=torch.bool)  # any K: with no changed variable, K and changed is empty
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    out |= {"dn_n_changed": len(feat.changed), "dn_dd_max": float(dd.abs().max()), "dn_E_b_K": SI.explained_share(num, den, dd, allK & ch)}
    del F
    out["identity"] = identity(w, cfg, None)
    out["identity_last"] = identity(w, cfg, KP.LAST)
    idn = out["identity"]
    out["est_seconds_per_item_last"] = idn["n_vars"] * cfg.derangements * cfg.examples / out["identity_last"]["cases_per_s"]  # D >= `examples` windows
    out["est_seconds_control"] = idn["n_vars"] * cfg.derangements * min(cfg.examples, idn["n_windows"]) / idn["cases_per_s"]
    miss = [f"identity {k}: {out[k]['flips']} argmax changes" for k in ("identity", "identity_last") if out[k]["flips"]]
    miss += [f"do-nothing: {k} {out[k]}" for k, want in (("dn_n_changed", 0), ("dn_dd_max", 0.0), ("dn_E_b_K", 1.0)) if out[k] != want]
    miss += [f"{cfg.control}: E_b over K* {eb!r} off P1.49's {cfg.control_E_b_Kstar!r}"] if out["E_b_Kstar_rel_dev"] > cfg.transcription_rel_tol else []
    out["seconds"] = time.time() - t0
    log(f"stage 0: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log(f"STAGE 0 DONE: est. {out['est_seconds_per_item_last']:.0f} s per item read at the last position, {out['est_seconds_control']:.0f} s for {cfg.control}")
    return out


def items(w: Any, cfg: Config, stage: str) -> list[str]:
    return list(cfg.built) if stage == "built" else RB.stage_items(w, RB.load_config(), STAGES[stage])


def summary() -> None:
    every = [json.loads(p.read_text()) for p in sorted((RESULTS / "items").glob("*.json"))]
    s0 = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("stage0_*.json"))]
    dump(RESULTS / "summary.json", {"experiment": EXPERIMENT, "stage0": s0, "items": {r["item"]: r for r in every}})
    log(f"summary: {len(every)} items, {len(s0)} stage-0 files")


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    if stage == "summary":
        summary()
        return 0
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226, E242, E257
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}; settings {asdict(cfg)}")
    for d in ("items", "large"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    w = world()
    floor = stage0(w, cfg, tag)["identity"]["max_dev"]
    if stage == "0":
        return 0
    every = items(w, cfg, stage)
    todo = every[shard[0] :: shard[1]]
    log(f"stage {stage}: {len(todo)} of {len(every)} items {todo}")
    for k, item in enumerate(todo):
        read_item(w, item, stage, cfg, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("0", *STAGES, "summary"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

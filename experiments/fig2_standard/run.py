"""E298 (cell P1.62; user 2026-11-24, "make sure we ran the exact same experiment for each of the 3 featurizations"):
Fig 2's write share E_b over the needed set K at VPD subcomponents, top neurons and SAE latents, read under one protocol
on the same 48 states (42 of the plan, and E280's three head-(2, 4) edits with their randoms, user 2026-11-24) (plan reports/plans/2026-11-24_plan_p1-62-fig2-standard.md; card card.md). Settings and filed
expected values in results/predictions.json, which the run refuses unless committed unchanged.

Per item (`read_item`, any featurization, feats.py): F, M, D and the read positions of the state; one write pass
(num, den, dd per variable over every position of D in F's own run); K by the featurization's swap test, 256 pairs =
`derangements` seeded derangements x `examples` windows (`commute.needed.derangements`, E268's seed), every variable
including the leftovers (VPD's per-matrix remainders, the pooled MLP remainders, the SAE error terms); the readings of
eb.py. Writes results/items/<feat>/<item>.json and results/large/<feat>/<item>.npz (num, den, dd, changed, in_K,
leftover, first_pair, flips); an item already written is skipped.

Stages, per featurization (--feat), cheapest first; each part of stage 0 writes results/stage0/<feat>_<part>_<tag>.json
and is skipped once done, so a killed job resumes; any miss stops the job:
    0   do_nothing (F = M on the built world: 0 changed, dd max 0.0, E_b 1.0); identity (x' = x on the control's first
        `examples` windows: 0 argmax changes, the largest logit deviation is the floor); repro_<item> (each filed saved
        item read again through this driver, E_b within repro_rel_tol[feat] of the saved value, relative; a null
        tolerance is the rehearsal: the deviation is measured and printed, and stage 1 refuses to start)
    1   do_nothing and identity on this job's card, then the filed items read[feat] (or --items), every N-th from the
        I-th with --shard. Its items count only once the same featurization's stage 0 passed with the tolerance filed

    python experiments/fig2_standard/run.py --feat sae --stage 0
    python experiments/fig2_standard/run.py --feat sae --stage 1 --shard 0 4
    python experiments/fig2_standard/run.py --feat vpd --stage 1 --items trained/copy_s1 trained/copy_s1_rand
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

from commute import needed as IX
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    manifest_only,
    note_inputs,
    require_filed,
)

HERE = Path(__file__).resolve().parent
EB: Any = load_experiment("fig2_standard", "eb")
FT: Any = load_experiment("fig2_standard", "feats")
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
FEATS = ("vpd", "neurons", "sae")
log, dump, stop, fname = FT.KN.log, FT.KN.dump, FT.KN.stop, EB.fname


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    control: str  # the identity control's state
    built: tuple[str, ...]  # E247's seven built edits (E268's)
    head_edits: tuple[str, ...]  # E280's head-(2, 4) edits, built/head_<e>
    built_random_of: tuple[str, ...]  # the built edits given a new matched random, built/rand_<edit>
    built_random_seed: int
    tasks: tuple[str, ...]  # E290's order
    seeds: tuple[int, ...]
    trained_random_seed: int  # E290's: its random of item k has seed trained_random_seed + k (E290's item order)
    band: str  # E229's neuron band
    derangements: int
    examples: int
    seed: int
    rows: int  # cases per forward with one read position
    logit_rows: int  # cases x read positions per forward, at most
    cache_batch: int
    ident_vars: int  # identity control: per matrix (VPD) or site (SAE) this many variables, beside its leftover
    sae_err_rows: int  # SAE error-term cases per forward (each encodes every latent of its rows)
    sae_chunk: int  # pairs per batch when finding the latents that move
    repro: dict[str, list[dict[str, str]]]  # feat -> saved items read again in stage 0
    repro_rel_tol: dict[str, float | None]  # feat -> tolerance on |E_b - saved| / saved; None until the rehearsal
    read: dict[str, list[str]]  # feat -> stage 1's items

    def states(self) -> list[str]:
        return EB.states(self.built, self.head_edits, self.built_random_of, self.tasks, self.seeds)

    def __post_init__(self) -> None:
        heads = {f"head_{h}" for h in self.head_edits}
        assert len(self.built) == 7 and set(self.built_random_of) <= set(self.built) | heads and len(self.states()) == 42 + 2 * len(heads)
        assert set(self.repro) == set(self.repro_rel_tol) == set(self.read) == set(FEATS) and self.control in self.states()
        bad = [i for f in FEATS for i in [*self.read[f], *(r["item"] for r in self.repro[f])] if i not in self.states()]
        assert not bad, f"not states: {bad}"
        assert all(r["mode"] in ("read", "writes") for f in FEATS for r in self.repro[f])


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


# ---- one item ---------------------------------------------------------------------------------------------------------


def read_item(feat: str, rd: Any, worlds: Any, item: str, cfg: Config) -> dict[str, Any]:
    path = RESULTS / "items" / feat / f"{fname(item)}.json"
    if path.is_file():
        log(f"{feat} {item}: done before, skipped")
        return json.loads(path.read_text())
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    F, M, toks, at, rec = worlds.make(item)
    assert toks.dim() == 2, f"{item}: the windows of D must have one length, got {tuple(toks.shape)}"
    st = rd.state(F, M, toks, worlds.of(item))
    num, den, dd, ch, left = rd.writes(st, F, toks, cfg.cache_batch)
    run = rd.cache(st, F, toks, at, cfg.cache_batch)
    m = min(cfg.examples, len(toks))
    ex, pt = IX.derangements(len(toks), cfg.derangements, m, cfg.seed)
    log(f"{feat} {item}: {len(toks)} windows of {toks.shape[1]} tokens, {run.top.shape[1]} read positions, {len(ch)} variables, "
        f"{int(ch.sum())} changed, {len(ex)} pairs")  # fmt: skip
    sc = rd.scan(st, F, run, at, ex, pt, cfg, log)
    K, mg = sc.first_pair >= 0, run.margin.flatten().cpu()
    out = {"experiment": EXPERIMENT, "feat": feat, "item": item, "n_windows": len(toks), "window_len": toks.shape[1],
           "n_read_positions": run.top.shape[1], "n_pairs": len(ex), "examples": m} | EB.readings(num, den, dd, ch, left, K)  # fmt: skip
    out |= {"margin_median": float(mg.median()), "margin_min": float(mg.min()), "max_dev": sc.max_dev, "cases_run": sc.rows} | rec
    (RESULTS / "large" / feat).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "large" / feat / f"{fname(item)}.npz", num=num.numpy(), den=den.numpy(), dd=dd.numpy(), changed=ch.numpy(),
                        in_K=K.numpy(), leftover=left.numpy(), first_pair=sc.first_pair.numpy(), flips=sc.flips.numpy())  # fmt: skip
    out |= {"seconds": time.time() - t0, "cases_per_s": sc.rows / sc.seconds, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
            "card": torch.cuda.get_device_name()}  # fmt: skip
    dump(path, out)
    log(f"{feat} {item}: E_b over K {out['E_b_K']:.6f} (leftovers dropped {out['E_b_K_no_leftover']:.6f}, their share of dd "
        f"{out['leftover_share']:.4f}); |K| {out['size_K']} ({out['n_changed_K']} changed) of {out['n_vars']}; {out['seconds']:.0f} s")  # fmt: skip
    del F, M, st, run
    torch.cuda.empty_cache()
    return out


# ---- stage 0 ----------------------------------------------------------------------------------------------------------


def part(feat: str, name: str, tag: str, fn: Any) -> dict[str, Any]:
    """Run fn() -> (record, misses) once: results/stage0/<feat>_<name>_<tag>.json, skipped once done; a miss stops."""
    path = RESULTS / "stage0" / f"{feat}_{name}_{tag}.json"
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        log(f"{path.name}: done before")
        return done
    t0 = time.time()
    out, miss = fn()
    out = {"experiment": EXPERIMENT, "feat": feat, "part": name} | out | {"seconds": time.time() - t0}
    log(f"stage 0 {feat} {name}: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, f"stage 0 {feat} {name}: " + "; ".join(miss))
    dump(path, out | {"done": True})
    return out


def do_nothing(rd: Any, worlds: Any, cfg: Config) -> tuple[dict[str, Any], list[str]]:
    F, M, toks, _, _ = worlds.make("built/do_nothing")
    num, den, dd, ch, _ = rd.writes(rd.state(F, M, toks, worlds.built), F, toks, cfg.cache_batch)
    out = {"n_changed": int(ch.sum()), "dd_max": float(dd.abs().max()), "E_b": SI.explained_share(num, den, dd, ch)}  # any K: K & changed is empty
    return out, [f"{k} {out[k]!r}, want {w!r}" for k, w in (("n_changed", 0), ("dd_max", 0.0), ("E_b", 1.0)) if out[k] != w]


def identity(rd: Any, worlds: Any, cfg: Config) -> tuple[dict[str, Any], list[str]]:
    F, M, toks, at, _ = worlds.make(cfg.control)
    st = rd.state(F, M, toks, worlds.of(cfg.control))
    run = rd.cache(st, F, toks, at, cfg.cache_batch)
    sc = rd.identity(st, F, run, at, torch.arange(min(cfg.examples, len(toks))), cfg, log)
    out = {"item": cfg.control, "flips": int(sc.flips.sum()), "max_dev": sc.max_dev, "min_margin": float(run.margin.min()), "cases": sc.rows,
           "cases_per_s": sc.rows / sc.seconds, "n_windows": len(toks), "n_read_positions": run.top.shape[1], "card": torch.cuda.get_device_name()}  # fmt: skip
    return out, [f"{out['flips']} argmax changes"] if out["flips"] else []


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


def repro(feat: str, rd: Any, worlds: Any, r: dict[str, str], cfg: Config) -> tuple[dict[str, Any], list[str]]:
    """A saved item read again: mode "read" reads it as any item (K by this run's scan) beside the saved E_b over the
    saved K, and reports how many variables' membership in K differ; mode "writes" (the SAE, whose saved reading is
    over E245's K*) reads only the write sums and takes E_b over the saved K* and changed."""
    repo, saved = EXPERIMENTS.parent, json.loads((EXPERIMENTS.parent / r["json"]).read_text())[r["key"]]
    mask = np.load(repo / r["npz"])[r["mask"]].astype(bool)
    if r["mode"] == "read":
        got = read_item(feat, rd, worlds, r["item"], cfg)
        K = np.load(RESULTS / "large" / feat / f"{fname(r['item'])}.npz")["in_K"]
        out = {"E_b": got["E_b_K"], "n_K_differ": int((K != mask).sum()), "size_K": int(K.sum()), "size_K_saved": int(mask.sum())}
    else:
        F, M, toks, _, _ = worlds.make(r["item"])
        num, den, dd, ch, _ = rd.writes(rd.state(F, M, toks, worlds.of(r["item"])), F, toks, cfg.cache_batch)
        out = {"E_b": SI.explained_share(num, den, dd, torch.from_numpy(mask) & ch)}
    tol = cfg.repro_rel_tol[feat]
    out |= {"item": r["item"], "saved": saved, "source": r["json"], "rel_dev": rel(out["E_b"], saved), "tol": tol}
    if tol is None:
        log(f"REHEARSAL {feat} {r['item']}: repro_rel_tol not filed; measured relative deviation {out['rel_dev']:.3g}")
        return out, []
    return out, [f"E_b {out['E_b']!r} off the saved {saved!r} by {out['rel_dev']:.3g} relative (tol {tol:.3g})"] if out["rel_dev"] > tol else []


def stage0(feat: str, rd: Any, worlds: Any, cfg: Config, with_repro: bool, tag: str) -> bool:
    """The parts in order (the reproductions only in stage 0 itself; each stage-1 job runs the two cheap controls on
    its own card); True when stage 1 may run (the tolerance filed)."""
    part(feat, "do_nothing", tag, lambda: do_nothing(rd, worlds, cfg))
    idn = part(feat, "identity", tag, lambda: identity(rd, worlds, cfg))
    for r in cfg.repro[feat] if with_repro else []:
        part(feat, f"repro_{fname(r['item'])}", tag, lambda r=r: repro(feat, rd, worlds, r, cfg))
    filed = cfg.repro_rel_tol[feat] is not None
    log(f"STAGE 0 DONE ({feat}{'' if with_repro else ', controls only'}): identity floor {idn['max_dev']:.3g} at {idn['cases_per_s']:.0f} "
        f"cases/s; {'tolerance filed' if filed else 'REHEARSAL: repro_rel_tol not filed, stage 1 will not start'}")  # fmt: skip
    return filed


# ---- main -------------------------------------------------------------------------------------------------------------


def manifest(feat: str, rd: Any, worlds: Any, cfg: Config, todo: list[str]) -> None:
    """INPUT_MANIFEST_ONLY dry run: every load the job makes, no compute and no writes."""
    note_inputs(*(EXPERIMENTS.parent / r[k] for r in cfg.repro[feat] for k in ("json", "npz")))
    for item in dict.fromkeys(["built/do_nothing", cfg.control, *(r["item"] for r in cfg.repro[feat]), *todo]):
        F, M, toks, _, _ = worlds.make(item)
        log(f"manifest {feat} {item}: {tuple(toks.shape)}")
        del F, M


def main(feat: str, stage: str, shard: tuple[int, int], items: list[str] | None, tag: str) -> int:
    cfg = load_config()
    src = EB.check_sources(cfg)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E268, E277, E263
    worlds, rd = FT.Worlds(cfg, feat == "vpd"), FT.reader(feat, cfg)
    todo = items if items is not None else cfg.read[feat][shard[0] :: shard[1]]
    assert all(i in cfg.states() for i in todo), todo
    log(f"{EXPERIMENT} start, feat {feat}, stage {stage}, {len(todo)} items {todo}; sources {src}; settings {asdict(cfg)}")
    if manifest_only():
        manifest(feat, rd, worlds, cfg, todo if stage == "1" else [])
        return 0
    for d in ("items", "large", "stage0"):
        (RESULTS / d / feat if d != "stage0" else RESULTS / d).mkdir(parents=True, exist_ok=True)
    filed = stage0(feat, rd, worlds, cfg, stage == "0", tag)
    if stage == "0":
        return 0
    if not filed:
        log(f"STOP: stage 1 at {feat} needs repro_rel_tol filed from the rehearsal")
        return 1
    for k, item in enumerate(todo):
        read_item(feat, rd, worlds, item, cfg)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({feat} {item})")
    log(f"STAGE 1 DONE ({feat})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=FEATS, required=True)
    ap.add_argument("--stage", choices=("0", "1"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th filed item from the I-th (0-based)")
    ap.add_argument("--items", nargs="+", help="these states instead of the filed list (stage 1)")
    ap.add_argument("--tag", default="local", help="this job's name: its stage-0 files are results/stage0/<feat>_<part>_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.feat, args.stage, (args.shard[0], args.shard[1]), args.items, args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

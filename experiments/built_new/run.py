"""E304 (cell P1.63 run 2; user 2026-11-24 via whip-lead): more of Fig 2's hand-built new edits. E214's new_few and
new_thin at slot layer 0 (commute.known_edits.build_swapping through E301's built_circuits/build.py new_edits) on other
E214 cue draws, each with one E290 matched random, read on the draw's cue windows at the last position by P1.62's
driver (E298's read_item and identity) at VPD, top neurons and SAE latents, as E301 reads its states. Card card.md;
settings, rules and expected values in results/predictions.json, which the run refuses unless committed unchanged.

Stages (any miss stops the job: the JSON gets "stopped", a .stopped marker beside it, exit 1):
    build   tries cue_seeds in order (a try = one cue_seed; it passes when both edits are built with no build miss; a
            detector that cannot reach the bar after a swap per cue fails the try, recorded, not a stop) until
            stop_edits edits pass or max_tries; every try to results/tries.json as it ends; per passing draw s<seed>
            the cues and the edits' changed parameters to results/large/s<seed>/, each random's norm check, and the
            record to results/build.json. Skipped once build.json is done.
    read    per featurization (needs build): the controls first (do-nothing write pass on the first passing draw's
            windows: results/stage0/<feat>_do_nothing_<tag>.json; identity swap on its new_few:
            <feat>_identity_<tag>.json), then the states s<seed>/{new_few,new_thin,rand_new_few,rand_new_thin} of every
            passing draw (or --items, or every N-th with --shard): results/items/<feat>/s<seed>__<state>.json and
            results/large/<feat>/s<seed>__<state>.npz, skipped once written

    PRED_SHA=<sha> python experiments/built_new/run.py build --tag b0
    PRED_SHA=<sha> python experiments/built_new/run.py read --feat vpd --shard 0 3 --tag v0
    PRED_SHA=<sha> python experiments/built_new/run.py read --feat neurons --tag n0
    PRED_SHA=<sha> python experiments/built_new/run.py read --feat sae --items s186001/new_few --tag s0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from commute import rule_retention as R
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.jobs import stop as jobs_stop

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E3: Any = load_experiment("built_circuits", "run")  # E301: part, do_nothing, vpd_world (and F2, FT through it)
B: Any = load_experiment("built_circuits", "build")  # E301: kl_world, new_edits, random_of, norm_check, changed, load
F2, FT = E3.F2, E3.FT  # E298's driver (read_item, identity) and readers
E3.RESULTS, E3.LARGE, E3.EXPERIMENT = RESULTS, LARGE, EXPERIMENT  # E301's part writes under its module's RESULTS
F2.RESULTS, F2.EXPERIMENT = RESULTS, EXPERIMENT  # read_item writes under its module's RESULTS (E301 pointed it at its own)
FIG2 = json.loads((EXPERIMENTS / "kstar_levels" / "results" / "edits.json").read_text())  # E214's draw (cue_seed 186000)
FEATS = ("vpd", "neurons", "sae")
NEW = ("new_few", "new_thin")
SWAPS: list[str] = []  # build_swapping's swap lines of the current try (it keeps a failed try's swaps to itself)


def log(msg: str) -> None:
    print(msg, flush=True)


def tee(msg: str) -> None:
    if msg.startswith("edits: swapping"):
        SWAPS.append(msg)
    log(msg)


B.RU.log = tee  # new_edits hands RU.log to build_swapping as its log


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    cue_seeds: tuple[int, ...]
    max_tries: int
    stop_edits: int
    slot_layer: int
    random_seeds: dict[str, list[int]]
    scan_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        assert self.cue_seeds == tuple(range(186001, 186011)) and len(self.cue_seeds) == self.max_tries
        assert self.random_seeds == {str(s): [304000 + 10 * k + j for j in range(len(NEW))] for k, s in enumerate(self.cue_seeds)}
        assert self.stop_edits % len(NEW) == 0 and self.slot_layer == 0

    def seed(self, draw: str, edit: str) -> int:
        return self.random_seeds[draw.removeprefix("s")][NEW.index(edit)]


def load() -> tuple[Config, dict[str, Any], SimpleNamespace]:
    pred = json.loads(PRED.read_text())
    cfg = config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()})
    e298 = json.loads((EXPERIMENTS / "fig2_standard" / "results" / "predictions.json").read_text())["settings"]
    return cfg, pred["thresholds"], SimpleNamespace(**{k: e298[k] for k in cfg.scan_keys})


# ---- build ------------------------------------------------------------------------------------------------------------

KNOWN_FAILS = ("no detector after a swap per cue", "cue ")  # build_swapping's swap cap; KE.cues' "cue i: only n of m windows hold"


def one_try(w: Any, cfg: Config, seed: int) -> tuple[dict[str, Any], Any]:
    """One cue draw: (its record, (edits, build record, cues) when it passes, else None)."""
    w.cfg, w.skip = dataclasses.replace(w.cfg, cue_seed=seed), ()
    w.__dict__.pop("cues", None)
    SWAPS.clear()
    t0 = time.time()
    try:
        edits, rec, cues = B.new_edits(w, cfg.slot_layer)
    except AssertionError as e:
        if not str(e).startswith(KNOWN_FAILS):
            raise
        return {"seed": seed, "pass": False, "why": [f"AssertionError: {e}"], "swap_log": list(SWAPS), "seconds": time.time() - t0}, None
    why = rec["misses"] + [f"{e} not built" for e in NEW if e not in edits]
    keys = ("slots", "A", "A_prime", "y", "swapped", "gamma_new_few", "gamma_new_thin", "top1_new_few", "top1_new_thin", "copy_M_D")
    out = {"seed": seed, "pass": not why, "why": why} | {k: rec.get(k) for k in keys}
    out |= {"same_as_fig2": {k: len(set(rec[k]) & set(FIG2[k])) for k in ("A", "A_prime", "y", "slots")}, "swap_log": list(SWAPS)}
    return out | {"seconds": time.time() - t0}, None if why else (edits, rec, cues)


def save_draw(w: Any, cfg: Config, th: dict[str, Any], draw: str, built: Any) -> tuple[dict[str, Any], list[str]]:
    edits, rec, cues = built
    (LARGE / draw).mkdir(parents=True, exist_ok=True)
    torch.save({"toks": cues.toks, "A_prime": cues.A2, "y": cues.y}, LARGE / draw / "cues.pt")
    out: dict[str, Any] = {"draw": draw} | {f"gamma_{e}": rec[f"gamma_{e}"] for e in NEW} | {f"top1_{e}": rec[f"top1_{e}"] for e in NEW}
    out["dW_norm"], miss = {e: rec["dW_norm"][e] for e in NEW}, []
    for e in NEW:
        torch.save(B.changed(edits[e], w.M), LARGE / draw / f"{e}.pt")
        Fr, out[f"rand_{e}"] = B.random_of(edits[e], w.M, w.net, cfg.seed(draw, e))
        miss += [f"{draw} rand_{e}: {m}" for m in B.norm_check(edits[e], Fr, w.M, th["norm_rel_tol"])]
    return out, miss


def build(cfg: Config, th: dict[str, Any], w: Any) -> None:
    tries: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    for k, seed in enumerate(cfg.cue_seeds[: cfg.max_tries]):
        t, built = one_try(w, cfg, seed)
        tries.append(t)
        write_stage(RESULTS / "tries.json", {"experiment": EXPERIMENT, "tries": tries})
        log(f"TRY {k + 1} (cue_seed {seed}): {'PASS' if t['pass'] else 'FAIL'}; {json.dumps(t)}")
        if built is not None:
            out, miss = save_draw(w, cfg, th, f"s{seed}", built)
            draws.append(out)
            log(f"draw s{seed}: {json.dumps(out)}")
            if miss:
                jobs_stop(RESULTS / "build.stopped.json", {"experiment": EXPERIMENT, "draws": draws}, "; ".join(miss), log)
        del built
        torch.cuda.empty_cache()
        if len(NEW) * len(draws) >= cfg.stop_edits:
            break
    rec = {"experiment": EXPERIMENT, "n_tries": len(tries), "n_pass_edits": len(NEW) * len(draws), "draws": draws}
    write_stage(RESULTS / "tries.json", {"experiment": EXPERIMENT, "tries": tries, "done": True})
    write_stage(RESULTS / "build.json", rec | {"done": True})
    log(f"BUILD DONE: {len(draws)} passing draws ({[d['draw'] for d in draws]}) in {len(tries)} tries")


# ---- read -------------------------------------------------------------------------------------------------------------


def states(draws: list[str]) -> list[str]:
    return [f"{d}/{s}" for d in draws for s in (*NEW, *(f"rand_{e}" for e in NEW))]


class Draws:
    """The passing draws' states for E298's read_item and identity: make(item) -> (F, M, D, read position, record).
    `c` and `vpd` are what E301's do_nothing reads: the first draw names the control's windows."""

    def __init__(self, cfg: Config, w: Any, draws: list[str], vpd: bool) -> None:
        self.cfg, self.w, self.c = cfg, w, draws[0]
        self.toks = {d: torch.load(LARGE / d / "cues.pt", weights_only=False)["toks"] for d in draws}
        self.vpd = E3.vpd_world(w) if vpd else None

    def of(self, item: str) -> Any:
        return self.vpd

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        d, name = item.split("/")
        base, toks = name.removeprefix("rand_"), self.toks[d]
        params = {} if base == "do_nothing" else torch.load(LARGE / d / f"{base}.pt", weights_only=True)
        F, rec = B.load(self.w.net, params), {"draw": d, "state": name}
        if name.startswith("rand_"):
            F, r = B.random_of(F, self.w.M, self.w.net, self.cfg.seed(d, base))
            rec |= r
        T = toks.shape[1]
        return F, self.w.M, toks.cuda(), slice(T - 1, T), rec


def read(cfg: Config, w: Any, feat: str, draws: list[str], todo: list[str], scan: SimpleNamespace, tag: str) -> None:
    scan = SimpleNamespace(**vars(scan), control=f"{draws[0]}/new_few")  # F2.identity and read_item's SAE empty-K stop read it
    dw, rd = Draws(cfg, w, draws, feat == "vpd"), FT.reader(feat, scan)
    for sub in ("items", "large"):
        (F2.RESULTS / sub / feat).mkdir(parents=True, exist_ok=True)
    E3.part(f"{feat}_do_nothing", tag, lambda: E3.do_nothing(rd, dw, scan))
    idn = E3.part(f"{feat}_identity", tag, lambda: F2.identity(rd, dw, scan))
    log(f"CONTROLS DONE ({feat}): identity floor {idn['max_dev']:.3g}, {idn['flips']} flips")
    for k, item in enumerate(todo):
        out = F2.read_item(feat, rd, dw, item, scan)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({feat} {item}): {json.dumps(out)}")
    log(f"READ DONE ({feat})")


# ---- main -------------------------------------------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    cfg, th, scan = load()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E298 and E301
    log(f"{EXPERIMENT} start: {vars(args)}; settings {asdict(cfg)}; thresholds {th}; E298 scan settings {vars(scan)}")
    w = B.kl_world()
    if args.stage == "build":
        if (p := RESULTS / "build.json").is_file() and json.loads(p.read_text()).get("done"):
            log("build.json: done before")
        else:
            build(cfg, th, w)
        return 0
    rec = json.loads((RESULTS / "build.json").read_text())
    assert rec.get("done") and rec["draws"], "build first: results/build.json done with at least one passing draw"
    draws = [d["draw"] for d in rec["draws"]]
    todo = args.items or states(draws)[args.shard[0] :: args.shard[1]]
    assert set(todo) <= set(states(draws)), todo
    if args.out:  # the reads and controls (not the build's inputs) under results/<out>/, e.g. the L40 cross-check
        F2.RESULTS = E3.RESULTS = RESULTS / args.out
    read(cfg, w, args.feat, draws, todo, scan, args.tag)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("build", "read"))
    ap.add_argument("--feat", choices=FEATS)
    ap.add_argument("--items", nargs="+", help="these states (s<seed>/<state>) instead of the read order")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"))
    ap.add_argument("--tag", default="local", help="this job's name: its control files end _<tag>")
    ap.add_argument("--out", help="write the read's items/, large/<feat>/ and stage0/ under results/<out>/ (cross-check)")
    a = ap.parse_args()
    if a.stage == "read" and not a.feat:
        ap.error("read needs --feat")
    require_filed(PRED)
    code = 1
    try:
        code = main(a)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

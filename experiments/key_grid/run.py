"""E237 (cell P2.50): the five 96-key tasks (copy, rank1 "top", elicit "mid", cipher, prior_cipher; E147's saved updates,
seeds 0-2, step 30, one recipe) read at ONE common update size, beside a random change of the same size. Every update
is multiplied by one scalar c_task so that each task's mean size ||dW|| / ||W0|| over the 24 matrices (E187's size)
equals the smallest task mean. The random control, one per state: a Gaussian direction at that state's per-matrix
Frobenius norms (E228's construction, `finetune.random_direction`) times the same c_task, so it has the state's scaled
size matrix by matrix, read on the state's own rows. Plan reports/plans/2026-11-20_plan_p2-50-key-grid.md; rules and
expected values in results/predictions.json (the run refuses unless it is committed unchanged). Nothing is trained.

Readings, each the function that wrote the saved reading, imported, on the same rows:
  r, kl_share   E228's `readings` (re-gating share, log-odds and KL forms) on the state's E147 read rows
  size, erank   E205's `sizes` (E187's formula); `update_spectrum.state` erank
  R coarse      E221's `read_item` at head_mlp, mlp_attn, block (E221's filed settings, its World with elicit added)
  R top neuron  E229's `read_pair` at p97, upper tail only (E229's filed settings, E217's percentiles)
  R VPD         E226's `read_item` (subcomponents plus site remainders; E226's filed settings)
  prior         the saved prior per task (experiments/prior_vs_share/results/stage1.json, not recomputed)
Items: '<state>_scaled' (the state times c_task), 'random_<state>' (its random control), '<state>' (as saved, shares
only). Stages, cheapest first, each writing as it goes; a rerun skips what is written:
  0  sizes and c; do-nothing (control state times 0) through the share and every coarse level; the control state at
     c = 1 repeats E228's r and kl_share and E221's R at every level (exit 1 on a miss)      -> results/stage0.json
  1  r, kl_share (with split halves), size, erank: 15 scaled, 15 random, 15 as saved          -> results/stage1.json
  2  R coarse: 15 scaled, 15 random                                                           -> results/items/
  3  R top neuron (sharded): E217's W0 check at p97 first, then 15 scaled, 15 random          -> results/topk/
  4  R VPD (sharded): the do-nothing first (R = 1, f = 0), then 15 scaled, 15 random          -> results/vpd/
  score (no GPU, score.py): per task mean, seed range and random band per reading; the rules  -> results/summary.json

    sbatch --comment=E237 cluster/key_grid.sbatch --stages 0 1 2
    sbatch --comment=E237 cluster/key_grid.sbatch --stages 3 --shard 0 2      (and --shard 1 2)
    sbatch --comment=E237 cluster/key_grid.sbatch --stages 4 --shard 0 3 --tag v0      (and 1 3 v1, 2 3 v2)
    ~/commute-agents/fleet/cpu-run python3 experiments/key_grid/score.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import finetune as FT
from commute import old_tasks as OT
from commute import rho
from commute import rule_retention as R
from commute import update_spectrum as US
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

E221: Any = load_experiment("kstar_pairs", "run")  # World, read_item, load_config, fname, LAST
E228: Any = load_experiment("random_kl_share", "run")  # state_read, Splitter, readings, ST
E229: Any = load_experiment("topk_upper", "run")  # Config, PRED, load_pct, feat_of, read_pair; E217 (w0)
E226: Any = load_experiment("kstar_vpd", "run")  # World, read_item, load_config
E205, R96 = E221.E205, E228.R96  # E205: sizes, SM (E187: times); R96: inventories, state_name
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, STAGE0, STAGE1 = RESULTS / "predictions.json", RESULTS / "stage0.json", RESULTS / "stage1.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E147_INV = EXPERIMENTS / "refit_96key" / "results" / "inventory.json"
E221_ITEMS = EXPERIMENTS / "kstar_pairs" / "results" / "items"
E228_STATES = EXPERIMENTS / "random_kl_share" / "results" / "states"
E221.EXPERIMENT = E226.EXPERIMENT = EXPERIMENT  # their read_item stamps each record with its module's E-code; ours here
E226.RESULTS = RESULTS  # E226's read_item writes its npz under RESULTS / "large"
log, dump, stop, fname = E221.log, E221.dump, E221.stop, E221.fname
DO_NOTHING = "do_nothing_c0"


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    tasks: tuple[str, ...]  # E147's names
    seeds: tuple[int, ...]
    control_state: str  # times 0 is the do-nothing; times 1 repeats the saved readings
    split_seed: int  # the share's two halves of rows (E228's)
    split_tol: float  # the least-squares split's bases (E228's)
    random_seed: int  # state i's random direction uses random_seed + i
    band: str  # E229's top-neuron band

    def __post_init__(self) -> None:
        assert len(set(self.tasks)) == len(self.tasks) == 5 and self.seeds == (0, 1, 2)
        assert self.control_state in self.states()

    def states(self) -> list[str]:
        return [R96.state_name(t, s) for t in self.tasks for s in self.seeds]

    def read_items(self) -> list[str]:
        """Stages 2-4: every state scaled, then every random control."""
        return [f"{s}_scaled" for s in self.states()] + [f"random_{s}" for s in self.states()]


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


class World(E221.World):
    """E221's world (M, its windows, its filed settings) with every E147 task in its inventory, E228's model for the
    share, and `make` for this run's items."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(E221.load_config()[0])
        _, self.full = R96.inventories(read_json(E147_INV))
        self.pairs.inv["tasks"].update({k: v for k, v in self.full["tasks"].items() if k.startswith("e58/")})  # adds elicit
        self.cfg_e237, self.c = cfg, dict[str, float]()
        self.share_net = E228.ST.Net("cuda")  # E228's model object, as E228 built it (E221's has requires_grad off)

    def task(self, state: str) -> str:
        return self.full["states"][state]["task"].removeprefix("e58/")

    def dw(self, state: str) -> dict[str, Tensor]:
        """The saved update, keyed by parameter name (`old_tasks.load_dw`)."""
        return OT.load_dw(self.net, EXPERIMENTS.parent / self.full["states"][state]["file"])

    def update(self, item: str) -> tuple[str, dict[str, Tensor], float]:
        """The item's state, its unscaled change (the saved update or its random direction) and the factor c."""
        if item == DO_NOTHING:
            return self.cfg_e237.control_state, self.dw(self.cfg_e237.control_state), 0.0
        if item.startswith("random_"):
            state = item.removeprefix("random_")
            seed = self.cfg_e237.random_seed + self.cfg_e237.states().index(state)
            return state, FT.random_direction(self.dw(state), seed), self.c[self.task(state)]
        state = item.removesuffix("_scaled")
        return state, self.dw(state), self.c[self.task(state)] if item != state else 1.0

    def make(self, item: str) -> tuple[R.Model, Tensor, slice, dict[str, Any]]:
        state, d, c = self.update(item)
        dw = E205.SM.times(d, c)
        rec = {"state": state, "c": c, "size": E205.sizes(dw, self.net.W0)}
        windows = self.read(self.read_half187(self.full["states"][state]["task"]))
        return R.Model(OT.with_delta(self.net, dw).net), windows, E221.LAST, rec


class VPDWorld(E226.World):
    """E226's reader on our World's items: the decomposition and the split's bases, no E214 world."""

    def __init__(self, w: World) -> None:
        from commute import vpd

        model, self.kp = w.net.model, w
        self.uv, self.delta, self.keys, self.bases = vpd.component_weights(model), vpd.weight_deltas(model), vpd.module_keys(model), {}

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        F, toks, at, rec = self.kp.make(item)
        return F, self.kp.M, toks, at, rec


# ---- stage 0: sizes, c, do-nothing, repeats --------------------------------------------------------------------------


def stage0(cfg: Config, th: dict[str, Any], w: World, sp: Any) -> None:
    if STAGE0.is_file() and read_json(STAGE0).get("done"):
        log("stage 0 done before")
        return
    sizes = {s: E205.sizes(w.dw(s), w.net.W0)["whole"] for s in cfg.states()}
    means = {t: fmean(sizes[R96.state_name(t, s)] for s in cfg.seeds) for t in cfg.tasks}
    target = min(means.values())
    out: dict[str, Any] = {"experiment": EXPERIMENT, "sizes": sizes, "task_mean_size": means, "target": target,
                           "target_task": min(means, key=lambda t: means[t]), "c": {t: target / m for t, m in means.items()}}  # fmt: skip
    log(f"stage 0 sizes: task means {means}; target {target:.6g} ({out['target_task']}); c {out['c']}")
    miss = share_checks(cfg, th, w, sp, out) + coarse_checks(th, w, out)
    log(f"stage 0: {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(STAGE0, out, "stage 0: " + "; ".join(miss))
    dump(STAGE0, out | {"done": True})
    log("STAGE 0 DONE")


def share_checks(cfg: Config, th: dict[str, Any], w: World, sp: Any, out: dict[str, Any]) -> list[str]:
    """Zero change: r's numerator and gain and E57's KL identities exactly 0. c = 1: E228's r and kl_share repeat."""
    r, dW = E228.state_read(cfg.control_state, w.full)
    z = E228.readings(w.share_net, sp, r, E205.SM.times(dW, 0.0))
    K = z["K"]
    exact = {"numerator": z["numerator"], "gain": z["gain"], "K00-K0": K["K00"] - K["K0"], "K01-K0": K["K01"] - K["K0"],
             "K10-K00": K["K10"] - K["K00"], "K11-K01": K["K11"] - K["K01"]}  # fmt: skip
    one = E228.readings(w.share_net, sp, r, dW)
    ref = read_json(E228_STATES / f"{E228.fname(cfg.control_state)}.json")["own"]
    out["share_do_nothing"] = exact | {"r": z["r"]}
    out["share_repeat"] = {k: {"got": one[k], "saved": ref[k]} for k in ("r", "kl_share")}
    miss = [f"zero change: {k} = {v}" for k, v in exact.items() if v != 0]
    return miss + [f"repeat {k}: {v}" for k, v in out["share_repeat"].items() if not abs(v["got"] - v["saved"]) <= th["share_repro_tol"]]


def coarse_checks(th: dict[str, Any], w: World, out: dict[str, Any]) -> list[str]:
    """E232's stage 0 on our control state: times 0 reads R = 1, f = 0, s = 1, nothing changed at every level; times 1
    repeats E221's item at every level (same K*, R per order within R_repro_tol)."""
    th221, levels, ctrl, miss = E221.load_config()[1], list(w.cfg.levels), w.cfg_e237.control_state, []
    for lv, it in E221.read_item(w, DO_NOTHING, levels, RESULTS / "stage0", th221, 1).items():
        rs = it["searches"]["main"]
        dev = max(abs(1 - v) for sv in it["s"].values() for v in sv)
        Rs = [r[f"R_{g}"] for r in rs for g in ("read", "all")]
        out[f"do_nothing_{lv}"] = {"max_dev_s": dev, "f": [r["f"] for r in rs], "R": Rs, "n_changed": it["n_changed"]}
        if not (dev <= th221["s_tol"] and it["n_changed"] == 0 and all(r["f"] == 0 for r in rs) and all(abs(x - 1) <= th221["s_tol"] for x in Rs)):
            miss.append(f"do-nothing at {lv}: {out[f'do_nothing_{lv}']}")
    for lv, it in E221.read_item(w, ctrl, levels, RESULTS / "stage0", th221, w.cfg.orders).items():
        mine, ref = it["searches"]["main"], read_json(E221_ITEMS / f"{fname(ctrl)}_{lv}.json")["searches"]["main"]
        diff = max(abs(a[f"R_{g}"] - b[f"R_{g}"]) for a, b in zip(mine, ref, strict=True) for g in ("read", "all"))
        same_K = all([u["name"] for u in a["K"]] == [u["name"] for u in b["K"]] for a, b in zip(mine, ref, strict=True))
        out[f"repeat_{lv}"] = {"max_abs_R_diff": diff, "same_K": same_K, "R_read": [r["R_read"] for r in mine], "R_read_E221": [r["R_read"] for r in ref]}
        if not (same_K and diff <= th["R_repro_tol"]):
            miss.append(f"repeat at {lv}: {out[f'repeat_{lv}']}")
    return miss


# ---- stage 1: shares, size, erank --------------------------------------------------------------------------------------


def share_record(w: World, sp: Any, r: Any, dW: dict[str, Tensor], halves: Tensor) -> dict[str, Any]:
    """Both share forms (E228's `readings`, module-keyed dW), the size (E187's) and the erank of one change."""
    size = E205.sizes({f"{k}.weight": v for k, v in dW.items()}, w.net.W0)["whole"]
    return E228.readings(w.share_net, sp, r, dW, halves) | {"size": size, "erank": US.state(dW)["erank"]}


def stage1(cfg: Config, th: dict[str, Any], w: World, sp: Any) -> None:
    out: dict[str, Any] = read_json(STAGE1) if STAGE1.is_file() else {"experiment": EXPERIMENT, "states": {}}
    for state in cfg.states():
        if state in out["states"]:
            continue
        t0, (r, dW), c = time.time(), E228.state_read(state, w.full), w.c[w.task(state)]
        halves = torch.randperm(len(r.toks), generator=torch.Generator().manual_seed(cfg.split_seed))
        _, rnd, _ = w.update(f"random_{state}")
        rnd = {k.removesuffix(".weight"): v for k, v in rnd.items()}
        rec: dict[str, Any] = {"task": w.task(state), "c": c, "saved": share_record(w, sp, r, dW, halves)}
        rec |= {"scaled": share_record(w, sp, r, E205.SM.times(dW, c), halves), "random": share_record(w, sp, r, E205.SM.times(rnd, c), halves)}
        if not abs(rec["scaled"]["erank"] / rec["saved"]["erank"] - 1) <= th["erank_rel"]:
            stop(STAGE1, out, f"{state}: erank moved under one scalar")
        out["states"][state] = rec
        dump(STAGE1, out)
        log(f"[1 {state}] c {c:.6f}; " + "; ".join(f"{k} r {rec[k]['r']:.4f} kl {rec[k]['kl_share']:.4f} size {rec[k]['size']:.6g} erank {rec[k]['erank']:.2f}"
                                                  for k in ("scaled", "random", "saved")) + f"; {time.time() - t0:.0f} s")  # fmt: skip
    target = read_json(STAGE0)["target"]
    got = {t: fmean(out["states"][R96.state_name(t, s)][f]["size"] for s in cfg.seeds) for t in cfg.tasks for f in ("scaled", "random")}
    if bad := {t: v for t, v in got.items() if not abs(v / target - 1) <= th["size_match_rel"]}:
        stop(STAGE1, out, f"task mean sizes off the target {target}: {bad}")
    dump(STAGE1, out | {"done": True})
    log("STAGE 1 DONE")


# ---- stages 2-4: R ------------------------------------------------------------------------------------------------------


def stage2(cfg: Config, w: World) -> None:
    th221, levels, todo = E221.load_config()[1], list(w.cfg.levels), cfg.read_items()
    for k, item in enumerate(todo):
        E221.read_item(w, item, levels, RESULTS / "items", th221, w.cfg.orders)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log("STAGE 2 DONE")


def topk_item(cfg229: Any, w: World, feat: Any, band: str, item: str) -> None:
    path = RESULTS / "topk" / f"{fname(item)}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return
    t0, (F, toks, _, rec) = time.time(), w.make(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {"experiment": EXPERIMENT, "item": item, "band": band}
    out, arrays = E229.read_pair(cfg229, rho.Pair(F, w.M), toks, w.M, feat, head, path)
    (RESULTS / "large" / "topk").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "large" / "topk" / f"{fname(item)}.npz", **arrays)
    dump(path, out | rec | {"item_seconds": time.time() - t0})
    log(f"{item}: R read {out['R_read']:.6f}, |K*| {out['size']}, f {out['f']:.4f}; {time.time() - t0:.0f} s")
    torch.cuda.empty_cache()


def stage3(cfg: Config, w: World, shard: tuple[int, int]) -> None:
    """E229's p97 reading on F = M + c dW, after E217's W0 check on the control state's windows (exit 1 on a miss)."""
    cfg229 = config_from_settings(E229.Config, read_json(E229.PRED)["settings"])
    feat = E229.feat_of(cfg229, E229.load_pct(cfg229), cfg.band)
    toks = w.read(w.read_half187(w.full["states"][cfg.control_state]["task"]))
    w0, miss = E229.E217.w0(cfg229, w.M, toks, feat)
    log(f"stage 3 W0 at {cfg.band}: {w0}")
    (RESULTS / "topk").mkdir(parents=True, exist_ok=True)
    if miss:
        stop(RESULTS / "topk" / f"w0_shard{shard[0]}.json", w0, f"W0 at {cfg.band} missed: {miss}")
    todo = cfg.read_items()[shard[0] :: shard[1]]
    for k, item in enumerate(todo):
        topk_item(cfg229, w, feat, cfg.band, item)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log("STAGE 3 DONE")


def stage4(cfg: Config, w: World, shard: tuple[int, int], tag: str) -> None:
    """E226's VPD reading, after the do-nothing reads R = 1, f = 0 (prefix and polished), s = 1 (exit 1 on a miss)."""
    vcfg, vth = E226.load_config()
    vw = VPDWorld(w)
    it = E226.read_item(vw, DO_NOTHING, vcfg, vth, RESULTS / f"stage4_{tag}")
    m = it["searches"]["main"]
    ok = it["n_changed"] == 0 and it["max_dev_s_unchanged"] <= vth["s_tol"]
    ok = ok and all(m[v]["f"] == 0 and abs(m[v][f"R_{g}"] - 1) <= vth["s_tol"] for v in ("prefix", "polished") for g in ("read", "all"))
    if not ok:
        stop(RESULTS / f"stage4_{tag}.json", {"do_nothing": m}, "VPD do-nothing does not read R = 1, f = 0")
    todo = cfg.read_items()[shard[0] :: shard[1]]
    for k, item in enumerate(todo):
        E226.read_item(vw, item, vcfg, vth, RESULTS / "vpd")
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log("STAGE 4 DONE")


def main(stages: list[str], shard: tuple[int, int], tag: str) -> int:
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as every saved reading
    log(f"{EXPERIMENT} start, stages {stages}, shard {shard}, tag {tag}; settings {asdict(cfg)}; thresholds {th}")
    w = World(cfg)
    sp = E228.Splitter(w.share_net, cfg.split_tol) if {"0", "1"} & set(stages) else None
    if "0" in stages:
        stage0(cfg, th, w, sp)
    if not (STAGE0.is_file() and read_json(STAGE0).get("done")):
        log("STOP: stage 0 has not passed")
        return 1
    w.c = read_json(STAGE0)["c"]
    if "1" in stages:
        stage1(cfg, th, w, sp)
    if "2" in stages:
        stage2(cfg, w)
    if "3" in stages:
        stage3(cfg, w, shard)
    if "4" in stages:
        stage4(cfg, w, shard, tag)
    log(f"{EXPERIMENT} DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", required=True, choices=["0", "1", "2", "3", "4"])
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="stages 3 and 4: every N-th item from the I-th")
    ap.add_argument("--tag", default="local", help="stage 4: its do-nothing writes results/stage4_<tag>/")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stages, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

"""E### (cell P2.59; user 2026-11-22: "run the activation difference lens too, on the cluster"): the activation
difference lens (Minder et al. 2025, arXiv 2510.13900) as a model-diffing baseline on the 67M, as one scorecard column.
Plan reports/plans/2026-11-22_plan_p2-59-adl.md; settings, thresholds and expected values in results/predictions.json,
which the run refuses unless committed unchanged. The lens is `commute.adl` (a port of the authors' toolkit, pinned).

Per item (a weight change dW on the 67M, and its task's read rows):
  delta_bar_j  mean of (W0 + dW) - W0 block-2 input over the first k = 5 positions of n_pile Pile val rows (unrelated text)
  D            the column: max over j of the share of the fine-tune's gain (answer log-odds at the read position, mean
               over the task's read rows, E143's `lo_at`) that the fine-tuned model loses when its block-2 input along
               delta_bar_j is set to the base model's at every position (paper eq. 1-2). For a random update the
               denominator is its paired fine-tune's gain (the random's own is ~0). Undefined (nan) under E143's gain
               floor, so the do-nothing reads nan with every numerator exactly 0.
  lens         per j, the share of the top-20 logit-lens tokens of delta_bar_j that are answer tokens of the task, and
               the best rank of any answer token (a side reading; the control's qualitative check)
Items: E237's five 96-key tasks x 3 seeds at E237's common size and their random updates (E244's `Items`, E237's own
construction); E187's nine built pairs, the larger side times E187's c (size_matched m67.json), and one random update per
pair side (a Gaussian at the side's first state's per-matrix norms, `finetune.random_direction`).

  stage0     alone; a miss writes results/stage0.json with passed false and exits 1: the do-nothing (delta_bar, every
             numerator and the gain exactly 0), the control's gain repeats E244's, then the positive control: D of the
             control above E143's inert line, above all 64 of the paper's random-difference directions and above its
             random update's D by 3 sd of those directions                       -> results/stage0.json, items/
  items I/N  refuses unless stage0.json says passed; every N-th item from the I-th; a rerun skips written items
  column     CPU, no model: results/column.json, the scorecard's shape (reuse, new, gap, seed-range call per pair)

    PRED_SHA=<sha> python experiments/adl/run.py stage0
    PRED_SHA=<sha> python experiments/adl/run.py items 0/1
    python experiments/adl/run.py column
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import adl as ADL
from commute import finetune as FT
from commute import steering as ST
from commute.data import pile_rows, target_tokens
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

KGB: Any = load_experiment("key_grid_baselines", "run")  # E244: Items (E237's items and sizes), log, dump, fname, HF, KG
SMR: Any = load_experiment("size_matched", "run")  # E187: pairs(), times, BUILT
HF, KG = KGB.HF, KGB.KG
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, ITEMS, STAGE0 = RESULTS / "predictions.json", RESULTS / "items", RESULTS / "stage0.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
M67 = EXPERIMENTS / "size_matched" / "results" / "m67.json"
E244_NEURON = EXPERIMENTS / "key_grid_baselines" / "results" / "run" / "neuron_0_1.json"
E143_PRED = EXPERIMENTS / "holes_fill" / "results" / "predictions.json"
log, fname = KGB.log, KGB.fname


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def dump(path: Path, x: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(x, indent=1, default=float))
    tmp.replace(path)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    k: int  # positions 0..k-1 of each text
    pile_split: str
    n_pile: int  # texts the mean difference is taken over
    batch: int  # texts per batch of the mean difference
    lens_top: int  # tokens kept per position from the logit lens
    n_random_diffs: int  # the paper's random-difference directions, stage 0
    random_diff_rows: int  # texts they are drawn from (first random_diff_len positions of each)
    random_diff_len: int
    random_seed: int  # the random-difference draw; pair-side random update i draws with random_seed + 1 + i
    control: str  # the positive control: an E237 item
    control_random: str  # its random update (E237's)

    def __post_init__(self) -> None:
        assert self.k >= 1 and self.n_pile >= self.batch >= 1 and self.lens_top >= 1 and self.n_random_diffs >= 2
        assert self.control_random == "random_" + self.control.removesuffix("_scaled")


@dataclass(frozen=True)
class Item:
    name: str
    group: str  # "keys" or "pairs"
    state: str
    c: float
    random: bool
    pair: str = ""
    side: str = ""
    seed: int = -1  # a pair-side random's draw
    size_want: float = math.nan  # E187's size of the item (pairs; E244's Items checks the keys)


def key_items(kg: Any) -> list[Item]:
    def one(n: str) -> Item:
        s = n.removeprefix("random_").removesuffix("_scaled")
        return Item(n, "keys", s, kg.c_of(s), n.startswith("random_"))

    return [one(n) for n in kg.cfg.read_items()]


def pair_items(cfg: Config) -> list[Item]:
    """E187's nine pairs: every state (the larger side times E187's c), then one random update per side."""
    names, m67, out = SMR.pairs()[0], read(M67)["pairs"], list[Item]()
    for p in SMR.BUILT:
        for side, states in zip(("reuse", "new"), names[p], strict=True):
            c = m67[p]["c"] if m67[p]["larger"] == side else 1.0
            sz = m67[p]["sizes"][side]
            out += [Item(f"{p}/{s}", "pairs", s, c, False, p, side, size_want=c * sz[i]) for i, s in enumerate(states)]
            seed = cfg.random_seed + 1 + sum(1 for it in out if it.random)
            out.append(Item(f"{p}/random_{side}", "pairs", states[0], c, True, p, side, seed, c * sz[0]))
    return out


def all_items(cfg: Config, kg: Any) -> dict[str, Item]:
    return {it.name: it for it in key_items(kg) + pair_items(cfg)}


class World:
    """The model, E143's task reader on E187's and E237's inventories, the Pile texts and every item."""

    def __init__(self, cfg: Config, th: dict[str, Any]) -> None:
        self.cfg, self.th, self.kg = cfg, th, KGB.Items()
        inv = SMR.pairs()[2]
        inv = {"tasks": inv["tasks"] | self.kg.inv["tasks"], "states": inv["states"] | self.kg.inv["states"], "pairs": []}
        self.items = all_items(cfg, self.kg)
        self.net = ST.Net("cuda")
        hf = read(E143_PRED)
        self.hf = HF.Job(self.net, inv, hf["settings"], hf["thresholds"])
        self.pile = pile_rows(cfg.pile_split, 0, cfg.n_pile)
        self.W0_norm = math.sqrt(sum(float(w.double().pow(2).sum()) for w in self.net.W0.values()))

    def task(self, state: str) -> Any:
        return self.hf.task(self.hf.inv["states"][state]["task"])

    def change(self, it: Item) -> tuple[ST.DW, ST.DW | None, float]:
        """The item's dW, the change whose gain is its denominator (None: its own), its size (checked against the
        item's source: E237's stage-1 size for the keys, E187's for the pairs)."""
        if it.group == "keys":
            dW = self.kg.change(it.name)[1]
            ref = self.kg.change(f"{it.state}_scaled")[1] if it.random else None
            return dW, ref, self.kg.size(it.name, it.state, dW, self.net.W0, self.th["size_rel"])
        d = SMR.times(self.hf.dW(it.state), it.c)
        dW = FT.random_direction(d, it.seed) if it.random else d
        size = math.sqrt(sum(float(v.double().pow(2).sum()) for v in dW.values())) / self.W0_norm
        if not abs(size / it.size_want - 1) <= self.th["size_rel"]:
            raise SystemExit(f"{it.name}: size {size} is not E187's {it.size_want}: not E187's item")
        return dW, d if it.random else None, size


def gain(w: World, t: Any, dW: ST.DW) -> float:
    return float((HF.lo_at(w.net, t, t.read.toks, dW) - t.lo0).mean())  # E143's Job.state gain


def removed_mean(w: World, t: Any, dW: ST.DW, V: Tensor) -> list[float]:
    """Per unit row of V, the mean over the task's read rows of the answer log-odds the fine-tune loses (adl.removed)."""
    tot, n = torch.zeros(len(V), dtype=torch.float64), 0
    for b in HF.batches(t, t.read.toks):
        tot += ADL.removed(w.net, t.x(b), dW, V, lambda lg, b=b: t.lo(lg[:, t.read.half], b)).sum(1)
        n += len(b)
    return (tot / n).tolist()


def shares(nums: list[float], G: float, floor: float) -> tuple[list[float], float]:
    s = [HF.share(x, G, floor) for x in nums]
    return s, (math.nan if any(math.isnan(x) for x in s) else max(s))


def read_item(w: World, it: Item) -> dict[str, Any]:
    t0, cfg = time.time(), w.cfg
    dW, ref, size = w.change(it)
    t = w.task(it.state)
    g = gain(w, t, dW)
    G = g if ref is None else gain(w, t, ref)
    dbar = ADL.mean_diff(w.net, w.pile[:, : cfg.k], dW, cfg.batch)
    answers = target_tokens(t.read.token, t.read.toks, t.read.half).unique()
    lens = ADL.lens_record(ADL.logit_lens(w.net, dbar), answers, cfg.lens_top)
    nums = removed_mean(w, t, dW, ADL.unit(dbar))
    per_j, D = shares(nums, G, w.th["gain_floor"])
    rec = asdict(it) | {"experiment": EXPERIMENT, "size": size, "gain": g, "G": G, "dbar_norm": dbar.norm(dim=-1).tolist(),
                        "removed": nums, "share": per_j, "D": D, "lens": lens, "n_answers": len(answers),
                        "seconds": time.time() - t0}  # fmt: skip
    rel = max(lens["relevance"])
    log(f"[{it.name}] c {it.c:.6f} size {size:.6g} gain {g:.4f} G {G:.4f} |dbar| {[round(x, 5) for x in rec['dbar_norm']]} "
        f"removed {[round(x, 5) for x in nums]} D {D:.5f} lens relevance max {rel:.3f} best rank {min(lens['best_rank'])}; "
        f"{rec['seconds']:.1f} s")  # fmt: skip
    return rec


def write_item(rec: dict[str, Any]) -> None:
    dump(ITEMS / f"{fname(rec['name'])}.json", rec)


# ---- stage 0 ----------------------------------------------------------------------------------------------------------


def stage0(w: World) -> bool:
    cfg, th, miss = w.cfg, w.th, list[str]()
    out: dict[str, Any] = {"experiment": EXPERIMENT}
    ctrl, rnd = w.items[cfg.control], w.items[cfg.control_random]
    t = w.task(ctrl.state)
    dn = w.kg.change(KG.DO_NOTHING)[1]  # the control state times 0 (E237's do-nothing)
    dbar = ADL.mean_diff(w.net, w.pile[:, : cfg.k], dn, cfg.batch)
    out["do_nothing"] = {"dbar_max": float(dbar.abs().max()), "removed": removed_mean(w, t, dn, ADL.unit(dbar)), "gain": gain(w, t, dn)}
    zero = [out["do_nothing"]["dbar_max"], out["do_nothing"]["gain"], *out["do_nothing"]["removed"]]
    miss += [f"do-nothing not exactly 0: {zero}"] if max(abs(x) for x in zero) > th["zero_tol"] else []
    log(f"[stage0 do-nothing] {json.dumps(out['do_nothing'])}")
    rc, rr = read_item(w, ctrl), read_item(w, rnd)
    saved = read(E244_NEURON)["items"][cfg.control]["gain"]
    out["repeat_gain"] = {"got": rc["gain"], "E244": saved, "bar": th["reader_tol"]}
    miss += [f"control gain {rc['gain']} is not E244's {saved}"] if abs(rc["gain"] - saved) > th["reader_tol"] else []
    gen = torch.Generator().manual_seed(cfg.random_seed)
    R = ADL.random_diffs(w.net, w.pile[: cfg.random_diff_rows, : cfg.random_diff_len], cfg.n_random_diffs, gen)
    dW = w.change(ctrl)[0]
    floor = shares(removed_mean(w, t, dW, ADL.unit(R)), rc["G"], th["gain_floor"])[0]
    sd, top = statistics.stdev(floor), max(floor)
    bars = {"inert": th["inert"], "random_diffs_max": top, "random_update_plus": rr["D"] + th["sd_mult"] * sd}
    passed = not miss and rc["D"] >= bars["inert"] and rc["D"] > top and rc["D"] >= bars["random_update_plus"]
    out |= {"control": {k: rc[k] for k in ("name", "gain", "G", "D", "share", "lens")}, "random": {k: rr[k] for k in ("name", "gain", "G", "D", "share", "lens")},
            "random_diffs": {"shares": floor, "sd": sd, "max": top, "mean": statistics.fmean(floor)}, "bars": bars, "misses": miss, "passed": passed}  # fmt: skip
    log(f"[stage0 control] D {rc['D']:.5f} vs random update D {rr['D']:.5f}; 64 random-difference directions mean "
        f"{out['random_diffs']['mean']:.5f} sd {sd:.5f} max {top:.5f}; bars {bars}; lens relevance control "
        f"{max(rc['lens']['relevance']):.3f} random {max(rr['lens']['relevance']):.3f}; misses {miss}; PASSED {passed}")  # fmt: skip
    for r in (rc, rr):
        write_item(r)
    dump(STAGE0, out)
    return passed


# ---- the column (CPU) -------------------------------------------------------------------------------------------------


def call(a: list[float], b: list[float]) -> str:
    """The scorecard's seed-range call (method_table plot_final.py): apart iff the ranges do not overlap; one value a
    side, or a nan, gets no call."""
    if len(a) < 2 or len(b) < 2 or any(math.isnan(x) for x in a + b):
        return "no call"
    return "apart" if max(a) < min(b) or max(b) < min(a) else "no call"


def column(cfg: Config) -> None:
    recs = {r["name"]: r for r in (read(p) for p in ITEMS.glob("*.json"))}
    kg = KGB.Items()
    items, cells, missing = all_items(cfg, kg), list[dict[str, Any]](), [n for n in all_items(cfg, kg) if n not in recs]

    def D(n: str) -> float:
        return recs[n]["D"] if n in recs else math.nan

    for p in SMR.BUILT:
        side = {s: [D(it.name) for it in items.values() if it.pair == p and it.side == s and not it.random] for s in ("reuse", "new")}
        rnd = {s: D(f"{p}/random_{s}") for s in ("reuse", "new")}
        gap = statistics.fmean(side["reuse"]) - statistics.fmean(side["new"])
        cells.append({"pair": p, "group": "built", "method": "adl", "reuse": side["reuse"], "new": side["new"], "n_reuse": len(side["reuse"]),
                      "n_new": len(side["new"]), "random": rnd, "gap": gap, "gap_rule": "reuse mean - new mean", "call": call(side["reuse"], side["new"]),
                      "call_rule": "seed ranges", "source": "experiments/adl/results/items/<pair>-<state>.json .D"})  # fmt: skip
    keys = {}
    for task in kg.cfg.tasks:
        mine = [it for it in items.values() if it.group == "keys" and it.state.startswith(f"e58/{task}_s")]
        keys[task] = {"D": [D(it.name) for it in mine if not it.random], "random": [D(it.name) for it in mine if it.random]}
    dump(RESULTS / "column.json", {"experiment": EXPERIMENT, "cells": cells, "keys": keys, "missing": missing})
    log(f"column: {len(cells)} pairs, {len(keys)} key tasks, {len(missing)} items missing; gaps {[round(c['gap'], 4) for c in cells]}")


def main(stage: str, shard: tuple[int, int]) -> int:
    P = read(PRED)
    cfg, th = config_from_settings(Config, P["settings"]), P["thresholds"]
    if stage == "column":
        column(cfg)
        return 0
    if not re.fullmatch(r"E\d+", EXPERIMENT):
        raise SystemExit(f"predictions.json experiment is {EXPERIMENT!r}: the lead claims the E-number first")
    if stage == "items" and not (STAGE0.is_file() and read(STAGE0).get("passed") is True):
        raise SystemExit(f"{STAGE0} does not say passed: stage 0 runs first and alone")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as E143 and E244 read
    log(f"{EXPERIMENT} {stage} {shard[0]}/{shard[1]}; settings {asdict(cfg)}; thresholds {th}")
    w = World(cfg, th)
    if stage == "stage0":
        code = 0 if stage0(w) else 1
    else:
        todo = [it for it in list(w.items.values())[shard[0] :: shard[1]] if not (ITEMS / f"{fname(it.name)}.json").is_file()]
        t0 = time.time()
        for k, it in enumerate(todo):
            write_item(read_item(w, it))
            log(f"ITEM {k + 1} of {len(todo)} DONE; projected minutes {(time.time() - t0) / 60 / (k + 1) * len(todo):.1f}")
        code = 0
    log(f"FINISHED {stage}; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage0", "items", "column"])
    ap.add_argument("part", nargs="?", default="0/1", help="items: I/N, every N-th item from the I-th")
    a = ap.parse_args()
    require_filed(PRED)
    i, n = (int(x) for x in a.part.split("/"))
    code = 1
    try:
        code = main(a.stage, (i, n))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int):
            print(e.code, flush=True)
    hard_exit(code)

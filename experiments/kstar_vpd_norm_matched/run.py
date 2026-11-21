"""P1.44 (E240): E239's size-normalised reuse score N (and N_w, N_in, R) on E226's six built edits, each update rescaled
to one common relative update size, new_thin's. Card and plan reports/plans/2026-11-20_plan_p1-44-card.md; filed rules
in results/predictions.json (the run refuses unless it is committed unchanged).

Per item: F_c = M + c (F - M) on every network parameter (fp64, cast to fp32; c = 0 is M, c = 1 is F exactly) and the
appended neurons' output weights times c (`scaled`); then E226's `read_item` unchanged (split, K* on all of D and its two
halves, grades; results/search/<item>.json, results/large/search_<item>.npz) and E239's `read_item` unchanged (N, N_w,
N_in, R on that K* and s; results/items/<item>.json, results/large/<item>.npz). Both modules' result paths are pointed
here (`redirect`); their code and filed settings are not changed. A rerun skips what is written.

Each job reads its shard (settings "shards") after stage 0: the do-nothing state (x2's windows at c = 0) through E226's
stage 0 (and, in job 0, E239's do-nothing guards). Per item, before the next (exit 1 on a miss): the update size at F_c equals the
target's (size_rel); new_thin, at c = 1 and first in its job, repeats E226's K* exactly and its R within repro_tol.
results/summary.json once all six items are read (by whichever job ends last).

    sbatch --comment='E240 ...' cluster/kstar_vpd_norm_matched.sbatch 0     (jobs 0, 1, 2 at once)
    python experiments/kstar_vpd_norm_matched/run.py --shard 0 --tag job0
    python experiments/kstar_vpd_norm_matched/run.py --summary              (no GPU: summary.json from the item files)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import kstar_vpd as KV
from commute import rule_retention as R
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

E226: Any = load_experiment("kstar_vpd", "run")  # World, read_item, stage0, load_config, KP (fname), log/dump/stop
E239: Any = load_experiment("kstar_vpd_norm", "run")  # read_item, load_config, order_rule; RR (npz_path)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, dump, stop, fname = E226.log, E226.dump, E226.stop, E226.KP.fname
REUSE, NEW = ("x2", "x10", "moved", "rebuilt"), ("new_few", "new_thin")
KNOWN = (*REUSE, *NEW)
SAVED_ITEMS = E226.RESULTS / "items"  # E226's own readings at the edits' own sizes (read before `redirect`)
SAVED_NPZ = E239.RR.npz_path  # E226's saved per-variable arrays
SAVED_E239 = json.loads((E239.RESULTS / "summary.json").read_text())["known"]
MEASURES = {"N": ("frob", "N"), "N_w": ("frob", "N_w"), "N_in": ("input", "N"), "R": ("frob", "R")}


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    target: str  # the item whose update size every item is rescaled to
    do_nothing_from: str  # the item whose windows the c = 0 state is read on
    shards: tuple[tuple[str, ...], ...]  # the items of each job, in order

    def __post_init__(self) -> None:
        every = [i for s in self.shards for i in s]
        assert sorted(every) == sorted(KNOWN) and self.target in KNOWN and self.do_nothing_from in KNOWN
        assert any(s[0] == self.target for s in self.shards), "the c = 1 reproduction must be first in its job"


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = pred["settings"] | {"shards": tuple(tuple(x) for x in pred["settings"]["shards"])}
    return config_from_settings(Config, s), pred["thresholds"]


def dw2(item_json: dict[str, Any]) -> float:
    """|dW|^2 summed over the 24 decomposed matrices, from the split checks (E238's `sizes` reads the same)."""
    return sum(c["dW2"] for c in item_json["checks"].values())


def saved(item: str) -> dict[str, Any]:
    return json.loads((SAVED_ITEMS / f"{fname(item)}.json").read_text())


def scale_factors(target: str) -> dict[str, float]:
    """c per item: its update times c has the target's size; exactly 1 for the target."""
    t = dw2(saved(target))
    return {i: 1.0 if i == target else math.sqrt(t / dw2(saved(i))) for i in KNOWN}


def redirect(tag: str) -> None:
    """Point E226's and E239's result paths and E-code at this run, and E239's saved-array lookup at E226's arrays
    as written here (E226 names them <out>_<item>.npz: out 'search' for items, 'stage0_<tag>' for the do-nothing)."""
    E226.RESULTS = E239.RESULTS = RESULTS
    E226.EXPERIMENT = E239.EXPERIMENT = EXPERIMENT
    E239.RR.npz_path = lambda item: RESULTS / "large" / (f"stage0_{tag}_do_nothing.npz" if item == "do_nothing" else f"search_{fname(item)}.npz")


@torch.no_grad()
def scaled(M: R.Model, F: R.Model, c: float) -> R.Model:
    """M + c (F - M) on every parameter; F's appended neurons with output weights times c (M lacks them: their update
    is their write from 0, and scaling it keeps moved's construction exact at every c)."""
    assert not M.extra
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(F.net.named_buffers(), M.net.named_buffers(), strict=True))
    G, m = R.Model(copy.deepcopy(M.net).requires_grad_(False)), dict(M.net.named_parameters())
    for n, p in G.net.named_parameters():
        f, b = F.net.get_parameter(n).double(), m[n].double()
        p.copy_((b + c * (f - b)).to(p.dtype))
    for layer, e in F.extra.items():
        e2 = copy.deepcopy(e)
        e2.w_out.mul_(c)
        G.append(layer, e2)
    return G


class World(E226.World):
    """E226's world; `make` returns each built item rescaled by its c, and the do-nothing as do_nothing_from at c = 0."""

    def __init__(self, c: dict[str, float], do_nothing_from: str) -> None:
        super().__init__()
        self.c, self.do_nothing_from = c, do_nothing_from

    def make(self, item: str) -> tuple[R.Model, R.Model, Any, slice, dict[str, Any]]:
        F, M, toks, at, rec = super().make(self.do_nothing_from if item == "do_nothing" else item)
        c = 0.0 if item == "do_nothing" else self.c[item]
        rec = rec | {"c": c}
        if c == 1:  # E221's load check: KL(F || M) on D against E226's reading of the same item
            rec |= {"kl_fm_saved": saved(item)["kl_fm"], "saved_from": "E226"}
        return scaled(M, F, c), M, toks, at, rec


def check_item(item: str, rec: dict[str, Any], c: dict[str, float], cfg: Config, th: dict[str, Any]) -> None:
    """The size at F_c against the target's; at c = 1, E226's K* and R repeated."""
    out: dict[str, Any] = {"item": item, "c": c[item], "size_ratio": dw2(rec) / dw2(saved(cfg.target))}
    miss = [f"size ratio {out['size_ratio']:.9g}"] if not abs(out["size_ratio"] - 1) <= th["size_rel"] else []
    if c[item] == 1:
        mine, ref = np.load(RESULTS / "large" / f"search_{fname(item)}.npz"), np.load(SAVED_NPZ(item))
        out["same_K"] = bool(np.array_equal(mine["main_in_K_polished"], ref["main_in_K_polished"]))
        out["R_diff"] = abs(rec["searches"]["main"]["polished"]["R_read"] - saved(item)["searches"]["main"]["polished"]["R_read"])
        miss += [f"c = 1: same K* {out['same_K']}, R off by {out['R_diff']:.3g}"] if not (out["same_K"] and out["R_diff"] <= th["repro_tol"]) else []
    log(f"{item}: checks {out}{'; MISS ' + '; '.join(miss) if miss else ''}")
    if miss:
        stop(RESULTS / "items" / f"{fname(item)}.stopped.json", out, f"{item}: " + "; ".join(miss))


def summary(c: dict[str, float]) -> dict[str, Any]:
    items = {i: json.loads((RESULTS / "items" / f"{fname(i)}.json").read_text()) for i in KNOWN}

    def val(i: str, k: str, m: str) -> float | None:
        r, f = MEASURES[m]
        return items[i]["searches"][k][r][f]

    get = {m: {i: val(i, "main", m) for i in KNOWN} for m in MEASURES}
    band = {}
    for m in MEASURES:
        d = [abs(a - b) for i in KNOWN if (a := val(i, "half1", m)) is not None and (b := val(i, "half2", m)) is not None]
        band[m] = max(d, default=0.0)
    rule = lambda m, top: E239.order_rule(get[m], list(top), NEW, band[m])
    K = {i: np.load(RESULTS / "large" / f"search_{fname(i)}.npz")["main_in_K_polished"] for i in ("x2", "x10")}
    d2 = {m: abs(get[m]["x2"] - get[m]["x10"]) if get[m]["x2"] is not None and get[m]["x10"] is not None else None for m in MEASURES}  # type: ignore[operator]
    return {"experiment": EXPERIMENT, "c": c, "band": band, "band_degenerate": {m: b == 0 for m, b in band.items()}, "matched": get,
            "own_size": SAVED_E239 | {"R": {i: saved(i)["searches"]["main"]["polished"]["R_read"] for i in KNOWN}},
            "P1": rule("N", REUSE), "S1": rule("N_w", REUSE), "S2": rule("N_in", REUSE), "S3": rule("N", ["x10"]), "D1": rule("R", REUSE),
            "D2": d2 | {"same_K": bool(np.array_equal(K["x2"], K["x10"]))}}  # fmt: skip


def write_summary(c: dict[str, float]) -> None:
    if all((RESULTS / "items" / f"{fname(i)}.json").is_file() for i in KNOWN):
        dump(RESULTS / "summary.json", sm := summary(c))
        log(f"summary: band {sm['band']}; P1 {sm['P1']}; S1 {sm['S1']}; S2 {sm['S2']}; S3 {sm['S3']}; D1 {sm['D1']}; D2 {sm['D2']}")


def main(shard: int, tag: str) -> int:
    cfg, th = load_config()
    c226, th226 = E226.load_config()
    c239, th239 = E239.load_config()
    assert set(c226.known) == set(KNOWN)
    c = scale_factors(cfg.target)
    todo = cfg.shards[shard]
    redirect(tag)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226
    log(f"{EXPERIMENT} start, tag {tag}; items {todo}; c {c}; settings {cfg}; thresholds {th}")
    (RESULTS / "items").mkdir(parents=True, exist_ok=True)
    w = World(c, cfg.do_nothing_from)
    w.bases = KV.bases(w.uv, c226.split_tol)
    E226.stage0(w, c226, th226, tag, 3 * len(todo))  # E226's do-nothing search and its stops, at c = 0
    if shard == 0:  # E239's do-nothing guards (no changed part, every ratio 0); one job only: one shared item file
        E239.read_item(w, "do_nothing", c239, th239)
    for k, item in enumerate(todo):
        rec = E226.read_item(w, item, c226, th226, RESULTS / "search")
        check_item(item, rec, c, cfg, th)
        E239.read_item(w, item, c239, th239)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item}, c {c[item]:.6g})")
    write_summary(c)
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=None, help="the job's index into settings 'shards'")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    ap.add_argument("--summary", action="store_true", help="only write results/summary.json from the item files (no GPU)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        if args.summary:
            write_summary(scale_factors(load_config()[0].target))
            code = 0
        else:
            code = main(args.shard, args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

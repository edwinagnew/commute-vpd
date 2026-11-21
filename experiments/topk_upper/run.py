"""Cell P2.42 run 2: E217's top-activating-neuron featurization of the 67M re-read with the upper tail only.

The 67M's MLP is a plain NewGELU, whose floor is about -0.17: E217's "below p3" side (Goodfire's two-sided rule, made for
SwiGLU neurons that fire both ways) sat on that floor, so near-zero firings counted as features. Here a neuron is a
feature on a token only when its activation is above its p97 (main; p99 and p90 beside), E217's saved base percentiles;
everything else at the site is the one exact remainder (`commute.kstar.Feat` with lo None). The readings are E217's own
functions, imported; each item is reported beside E217's at the matching band. Plan
reports/plans/2026-11-20_plan_p2-42-upper-tail.md; settings and expected values in results/predictions.json.

    python experiments/topk_upper/run.py --stage 0          Pile share, W0 at every band, the width control (stops)
    python experiments/topk_upper/run.py --stage 1 --task T  one band and seed (T = SLURM_ARRAY_TASK_ID, 0-5)
    python experiments/topk_upper/run.py --stage score       no GPU: results/summary.json from the item files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import kstar as K
from commute import rho
from commute import rule_retention as R
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

E217: Any = load_experiment("topk_neurons", "run")  # setup, windows, pair, read_vars, overlap, w0, control, log, dump, stop
HERE = Path(__file__).resolve().parent
REPO, RESULTS = HERE.parents[1], HERE / "results"
PRED, STAGE0, ITEMS, LARGE = RESULTS / "predictions.json", RESULTS / "stage0.json", RESULTS / "items", RESULTS / "large"
E217_RESULTS = REPO / "experiments" / "topk_neurons" / "results"
WIDTH, LAST, log, dump, stop, slug = E217.WIDTH, E217.LAST, E217.log, E217.dump, E217.stop, E217.slug
KEYS = ("R_read", "R_all", "f", "site_remainder_share", "size", "overlap", "s_rem")


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    pairs: list[list[str]]
    counted: int
    seeds: list[int]
    n_read: int
    read_seed: int
    n_max: int
    held_share: float
    bands: dict[str, float]  # name -> upper quantile
    main_band: str
    e217_bands: dict[str, str]  # name -> E217's band read beside it
    percentile_qs: list[float]  # the quantiles of E217's file, as its config's floats
    percentiles_sha256: str
    pile_rows: int
    pile_batch: int
    share_tol: float
    alpha: float
    chunk: int
    order_seed: int
    control_item: str
    e208_parts: str
    e213_summary: str
    gate_tol: float
    s_tol: float
    floor_factor: float
    floor_min: float

    def __post_init__(self) -> None:
        assert self.main_band in self.bands and WIDTH not in self.bands and self.e217_bands.keys() == self.bands.keys()
        assert all(0.5 < q < 1 and q in self.percentile_qs for q in self.bands.values())
        assert 0 < self.counted <= len(self.pairs) and self.control_item in self.items(self.seeds[0])

    def items(self, seed: int) -> list[str]:
        return [f"{m}_s{seed}" for pr in self.pairs for m in pr]

    def tasks(self) -> list[tuple[str, int]]:
        """Stage 1's array: the main band first, then the rest; each split by seed."""
        return [(b, s) for b in [self.main_band, *[b for b in self.bands if b != self.main_band]] for s in self.seeds]


def experiment() -> str:
    return json.loads(PRED.read_text())["experiment"]


def load_pct(cfg: Config) -> torch.Tensor:
    """E217's saved per-neuron quantiles (n_q, layers, n), checked against the filed sha and E217's stage 0."""
    path = E217_RESULTS / "large" / "percentiles.pt"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == cfg.percentiles_sha256 == json.loads((E217_RESULTS / "stage0.json").read_text())["percentiles_sha256"], f"percentile file sha {sha}"
    z = torch.load(path, weights_only=True)
    assert torch.equal(z["q"], torch.tensor(cfg.percentile_qs)), "the saved quantiles are not the filed ones"
    return z["values"]


def feat_of(cfg: Config, pct: torch.Tensor, band: str) -> K.Feat:
    """Upper tail only; the width makes every activation a feature (hi = -inf), so each site remainder writes zero."""
    hi = torch.full_like(pct[0], -torch.inf) if band == WIDTH else pct[cfg.percentile_qs.index(cfg.bands[band])]
    return K.Feat(None, list(hi.to("cuda")))


@torch.no_grad()
def pile_share(cfg: Config, M: R.Model, feats: dict[str, K.Feat]) -> dict[str, float]:
    """Share of the base's activations on E217's Pile rows that are features, per band."""
    m, hits, total = rho.Pair(M, M).m, dict.fromkeys(feats, 0), 0
    for t in cached_rows("val", 0, cfg.pile_rows).split(cfg.pile_batch):
        for l, a in enumerate(K.run(m, t.to("cuda"), None).a):
            for b, f in feats.items():
                hits[b] += int(f.strong(l, a).sum())
            total += a.numel()
    return {b: h / total for b, h in hits.items()}


def stage0(cfg: Config, net: Any, M: R.Model, data: Any) -> None:
    pct = load_pct(cfg)
    feats = {b: feat_of(cfg, pct, b) for b in [*cfg.bands, WIDTH]}
    out: dict[str, Any] = {"experiment": experiment(), "percentiles_sha256": cfg.percentiles_sha256}
    share = out["pile_strong_share"] = pile_share(cfg, M, {b: feats[b] for b in cfg.bands})
    log(f"feature share on {cfg.pile_rows} Pile rows {share} (expected {({b: round(1 - q, 4) for b, q in cfg.bands.items()})})")
    if bad := [b for b, q in cfg.bands.items() if abs(share[b] - (1 - q)) > cfg.share_tol * (1 - q)]:
        stop(STAGE0, out, f"feature share on the Pile far from its band: {bad}")
    toks = E217.windows(cfg, M, data, cfg.control_item)
    for band, feat in feats.items():
        out[f"w0_{slug(band)}"], miss = E217.w0(cfg, M, toks, feat)
        log(f"W0 {band}: {out[f'w0_{slug(band)}']}")
        if miss:
            stop(STAGE0, out, f"W0 at {band} missed: {miss}")
    e217_s0 = json.loads((E217_RESULTS / "stage0.json").read_text())
    out["floor_e217"] = e217_s0["floor"]
    bar = max(cfg.floor_factor * e217_s0["floor"], cfg.floor_min)
    log(f"control bar {bar:.3g} (E217's measured floor {e217_s0['floor']:.3g})")
    e213, miss = json.loads((REPO / cfg.e213_summary).read_text()), []
    out["control"] = {}
    for item in cfg.items(0) + cfg.items(1):
        lv, _, _, s = E217.read_vars(E217.pair(net, M, item), E217.windows(cfg, M, data, item), feats[WIDTH])
        c = out["control"][item] = E217.control(cfg, s, lv, item, e213) | {"R_parts_e217": e217_s0["control"][item]["R_parts"]}
        miss += [item] if max(abs(c["R_parts"] - c["R_parts_e213"]), abs(c["R_K"] - c["R_K_e213"])) > bar or float((s[:, lv.rem] - 1).abs().max()) > cfg.s_tol else []
        log(f"control {item}: R parts {c['R_parts']:.8f} (E213 {c['R_parts_e213']:.8f}, E217 {c['R_parts_e217']:.8f}), "
            f"R on E208's K {c['R_K']:.8f} (E213 {c['R_K_e213']:.8f})")  # fmt: skip
        torch.cuda.empty_cache()
    if miss:
        stop(STAGE0, out, f"the width control missed E213 on {miss}")
    dump(STAGE0, out | {"done": True})
    log("STAGE 0 DONE")


def read_item(cfg: Config, net: Any, M: R.Model, data: Any, feat: K.Feat, band: str, item: str) -> None:
    """E217's read_item, writing here: s of every variable, K* by mean ablation, R, f, the remainders and the overlap."""
    path = ITEMS / slug(band) / f"{item}.json"
    if path.is_file():
        log(f"{band} {item}: done before, skipped")
        return
    t0 = time.time()
    head = {"experiment": experiment(), "item": item, "band": band}
    out, arrays = read_pair(cfg, E217.pair(net, M, item), E217.windows(cfg, M, data, item), M, feat, head, path)
    out["item_seconds"] = time.time() - t0
    big = LARGE / "items" / slug(band)
    big.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(big / f"{item}.npz", **arrays)
    dump(path, out)
    log(f"{band} {item}: R read {out['R_read']:.6f}, all {out['R_all']:.6f}, |K*| {out['size']}, f {out['f']:.4f}, "
        f"remainders in K* {out['site_remainders_in_K']}, overlap {out['overlap']:.4f}; {out['item_seconds']:.0f} s")


def read_pair(cfg: Config, p: rho.Pair, toks: torch.Tensor, M: R.Model, feat: K.Feat, head: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """read_item's reading of one pair on its windows (split out for E237, which builds its own F): the record, with
    `head` first, and the arrays for the npz. Stops, writing path's .stopped.json, when F_K with every variable kept is not F."""
    lv, C, lacked, s = E217.read_vars(p, toks, feat)
    ctx = K.prepare(p, toks, LAST, None, M.logits, feat)
    g = K.gates(ctx, lv, p.F.logits)
    out: dict[str, Any] = head | {"n_vars": len(lv.vars), "n_changed": len(C), "gates": g,
                                  "kl_fm": float(ctx.kl_fm.mean()), "kl_none": float(ctx.kl_none.mean()), "overlap": E217.overlap(p, toks, feat),
                                  "s_rem": s[:, lv.rem].tolist() if lv.rem else None, "s_mean_all_vars": s.mean(1).tolist()}  # fmt: skip
    if g["all"] > cfg.gate_tol:
        stop(path.with_suffix(".stopped.json"), out, f"{head['band']} {head['item']}: F_K with every variable kept is not F (gate {g})")
    th = {"order_seed": cfg.order_seed, "chunk": cfg.chunk}
    res, arrays = K.search(ctx, lv, s, C, lacked, {"main": K.Variant(torch.arange(len(toks)), cfg.alpha, 1)}, th, log, lambda _s: None)
    rec = res["main"][0]
    rec.pop("K")  # the circuit goes to the npz
    return out | rec, {"s": s.numpy(), "names": np.array([K.vname(u) for u in lv.vars]), **arrays}


def stage1(cfg: Config, net: Any, M: R.Model, data: Any, task: int) -> None:
    if not (STAGE0.is_file() and json.loads(STAGE0.read_text()).get("done")):
        log("STOP: stage 0 has not passed")
        hard_exit(1)
    band, seed = cfg.tasks()[task]
    feat = feat_of(cfg, load_pct(cfg), band)
    log(f"stage 1 task {task}: band {band} (above q {cfg.bands[band]}), seed {seed}")
    for item in cfg.items(seed):
        read_item(cfg, net, M, data, feat, band, item)
        torch.cuda.empty_cache()
    log(f"STAGE 1 TASK {task} DONE")


# ---- the score (no GPU) ---------------------------------------------------------------------------------------------


def rows_of(root: Path) -> dict[str, dict[str, Any]]:
    return {f"{p.parent.name}/{p.stem}": json.loads(p.read_text()) for p in sorted(root.glob("*/*.json")) if not p.stem.endswith(".stopped")}


def gaps_by_seed(cfg: Config, rows: dict[str, dict[str, Any]], band: str, reuse: str, twin: str) -> dict[int, float]:
    """Reuse-minus-twin R_read per seed where both items were read."""
    a, b = (lambda s: f"{band}/{reuse}_s{s}"), (lambda s: f"{band}/{twin}_s{s}")
    return {s: rows[a(s)]["R_read"] - rows[b(s)]["R_read"] for s in cfg.seeds if a(s) in rows and b(s) in rows}


def score(cfg: Config) -> None:
    mine, old = rows_of(ITEMS), rows_of(E217_RESULTS / "items")
    trained = [i for s in cfg.seeds for i in cfg.items(s)]
    pick = lambda r: {k: r.get(k) for k in KEYS} if r else None
    table = {f"{b}/{i}": {"upper": pick(mine.get(f"{b}/{i}")), "e217": pick(old.get(f"{slug(cfg.e217_bands[b])}/{i}")),
                          "e217_width_R_read": old[f"width/{i}"]["R_read"]} for b in cfg.bands for i in trained}  # fmt: skip
    gaps = []
    for b in cfg.bands:
        for reuse, twin in cfg.pairs[: cfg.counted]:
            g = gaps_by_seed(cfg, mine, b, reuse, twin)
            gaps.append({"band": b, "pair": f"{reuse}/{twin}", "gap_by_seed": g, "seed_diff": g[0] - g[1] if len(g) == 2 else None,
                         "e217_gap_by_seed": gaps_by_seed(cfg, old, slug(cfg.e217_bands[b]), reuse, twin)})  # fmt: skip
    pred, main = json.loads(PRED.read_text())["predictions"], cfg.main_band
    done = [table[f"{main}/{i}"] for i in trained if table[f"{main}/{i}"]["upper"]]

    def check(name: str, value: Any) -> dict[str, Any]:
        lo, hi = pred[name]["range"]
        vals = [value(r) for r in done]
        return {"read": len(vals), "of": len(trained), "min": min(vals, default=None), "max": max(vals, default=None),
                "within": bool(vals) and all(lo <= v <= hi for v in vals)}  # fmt: skip

    leans: dict[str, Any] = {
        "P1": check("P1", lambda r: r["upper"]["R_read"]), "P2": check("P2", lambda r: r["upper"]["R_read"] - r["e217"]["R_read"]),
        "P3": check("P3", lambda r: r["upper"]["overlap"]), "P4": check("P4", lambda r: r["upper"]["overlap"] - r["e217"]["overlap"]),
        "P5": check("P5", lambda r: min(r["upper"]["s_rem"][1])), "P6": check("P6", lambda r: r["upper"]["site_remainder_share"]),
        "P7": check("P7", lambda r: r["upper"]["f"]), "P8": check("P8", lambda r: r["upper"]["size"]),
        "P1_not_below_width": [i for i in trained if table[f"{main}/{i}"]["upper"] and not table[f"{main}/{i}"]["upper"]["R_read"] < table[f"{main}/{i}"]["e217_width_R_read"]],
    }  # fmt: skip
    dump(RESULTS / "summary.json", {"experiment": experiment(), "table": table, "gaps": gaps, "leans": leans})
    for k, v in leans.items():
        log(f"{k}: {v}")
    for g in gaps:
        log(f"gap {g['band']:4s} {g['pair']:24s} {g['gap_by_seed']} seed diff {g['seed_diff']}; E217 {g['e217_gap_by_seed']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["0", "1", "score"], required=True)
    ap.add_argument("--task", type=int, help="stage 1: index into Config.tasks() (the array task id)")
    args = ap.parse_args()
    require_filed(PRED)
    cfg = config_from_settings(Config, json.loads(PRED.read_text())["settings"])
    if args.stage == "score":
        score(cfg)
        hard_exit(0)
    net, M, data = E217.setup(cfg)
    if args.stage == "0":
        stage0(cfg, net, M, data)
    else:
        assert args.task is not None and 0 <= args.task < len(cfg.tasks()), f"--task must be 0..{len(cfg.tasks()) - 1}"
        stage1(cfg, net, M, data, args.task)
    hard_exit(0)


if __name__ == "__main__":
    main()

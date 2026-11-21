"""P1.37 run 2 (E226 Amendment 1): the size-normalised reuse score N at the VPD featurization, on E226's saved K* and s.
Card and plan reports/plans/2026-11-20_plan_p1-37-run2-card.md; filed rules in results/predictions.json (the run
refuses unless it is committed unchanged). No search and no training: per item one rebuild of F and the split
(`commute.kstar_vpd.featurize`, as E226), one fp64 recompute of s (`grades`, checked against E226's saved fp32 s) and
the change ratio of every variable (`frob_ratios`, `input_ratios`).

Per variable u: t_u = (1 - s_u) / ratio_u, s_u at the read positions; N = 1 - mean t_u over K* members that changed and
have a ratio (|w_u| > 0, ratio_u > 0); N_w weights t_u by |attribution| (E226's ranking 0). Main ratio: the weight
norms |delta-w_u|^2 / |w_u|^2; side: the same ratio on the inputs (read positions). E226's three searches of a
known item (all of D, its two halves) each give a reading; the halves set the band.

Stage 0 (first, stops on a miss): the do-nothing item (no changed part, every ratio 0, s = 1) and two built
construction checks on a real site (a gain reads t = 0, a write-side turn t = 1 / (1 + r)). Then the six known-answer
items, then E187's 68 states; results/items/<item>.json per item (a rerun skips it), results/large/<item>.npz the
per-variable arrays (s fp64 read row, frob ratio, input ratio read row; fp32), results/summary.json once all are read.

    python experiments/kstar_vpd_norm/run.py 2>&1 | tee -a experiments/kstar_vpd_norm/results/run.log
    INPUT_MANIFEST=<file> INPUT_MANIFEST_CPU=1 python experiments/kstar_vpd_norm/run.py --items do_nothing   (dry run)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata
from torch import Tensor

from commute import kstar_vpd as KV
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

E226: Any = load_experiment("kstar_vpd", "run")  # World, load_config, e187_states, KP (fname, LAST), log/dump/stop
RR: Any = load_experiment("r_reanalysis", "run")  # npz_path (E226's saved arrays), sizes, PAIRS_JSON
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, dump, stop = E226.log, E226.dump, E226.stop
RATIOS = ("frob", "input")
GB = E226.load_config()[0].grade_batch  # windows per grade batch, E226's


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    row: int  # grade row: 1 = the read positions (E226's main grade)
    ranking: int  # the attribution ranking that weights N_w (0, as E238's Rw_K)
    control_size: float  # |du| / |u| of the built gain and turn
    control_site: int  # the site the construction checks use

    def __post_init__(self) -> None:
        assert self.row in (0, 1) and self.ranking >= 0 and 0 < self.control_size < 1


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    return config_from_settings(Config, pred["settings"]), pred["thresholds"]


# ---- readings -------------------------------------------------------------------------------------------------------


def reading(s: np.ndarray, ratio: np.ndarray, keep: np.ndarray, changed: np.ndarray, lacked: np.ndarray, w: np.ndarray,
            ratio_min: float) -> dict[str, Any]:  # fmt: skip
    """N and N_w over the changed K* members whose ratio is at least ratio_min (NaN: |w_u| = 0); the parts left out
    are counted (grafts; no ratio or under the floor). None when none is left."""
    kc = keep & changed
    used = kc & np.isfinite(ratio) & (ratio >= ratio_min)
    t = (1 - s[used]) / ratio[used]
    ww = np.abs(w[used].astype(np.float64))
    out: dict[str, Any] = {"R": float(s[keep].mean()), "n_K": int(keep.sum()), "n_KC": int(kc.sum()), "n_used": int(used.sum()),
                           "n_lacked": int((kc & lacked).sum()), "n_small": int((kc & ~lacked & ~used).sum()),
                           "n_t_over_1": int((t > 1).sum()), "t_max": float(t.max()) if t.size else None}  # fmt: skip
    out["N"] = float(1 - t.mean()) if t.size else None
    out["N_w"] = float(1 - (ww * t).sum() / ww.sum()) if t.size and ww.sum() > 0 else None
    return out


def readings(z: Any, s: np.ndarray, ratios: dict[str, np.ndarray], lacked: np.ndarray, cfg: Config, ratio_min: float) -> dict[str, Any]:
    """Per saved search of the item (main; half1, half2 for a known item) and per ratio."""
    out = {}
    for k in ("main", "half1", "half2"):
        if f"{k}_in_K_polished" in z.files:
            keep, w = z[f"{k}_in_K_polished"], z[f"{k}_attribution"][cfg.ranking]
            out[k] = {r: reading(s, ratios[r], keep, z["changed"], lacked, w, ratio_min) for r in RATIOS}
    return out


# ---- stage 0: construction checks -----------------------------------------------------------------------------------


def built(feat: KV.Feat, j: int, dU: Tensor) -> KV.Feat:
    """The do-nothing featurization with site j's write side changed by dU (dV = 0, remainder unchanged)."""
    s = replace(feat.sites[j], dU=dU.contiguous(), changed=True)
    return replace(feat, sites=[*feat.sites[:j], s, *feat.sites[j + 1 :]])


def construction(w: Any, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    """t on a built gain (du = g u: expected 0) and a built write-side turn (du orthogonal to u, |du| = g |u|: expected
    1 / (1 + g^2) for both ratios), through the same s and ratio code as the items, on M's inputs at the do-nothing windows."""
    F, M, toks, at, _ = w.make("do_nothing")
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    j, g = cfg.control_site, cfg.control_size
    U = feat.sites[j].U
    noise = torch.randn(U.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64).to(U.device)
    Ud = U.double()
    perp = noise - ((noise * Ud).sum(1, keepdim=True) / (Ud**2).sum(1, keepdim=True).clamp(min=1e-300)) * Ud
    turn = perp * (g * Ud.norm(dim=1, keepdim=True) / perp.norm(dim=1, keepdim=True).clamp(min=1e-300))
    out: dict[str, Any] = {"site": feat.sites[j].name}
    for name, dU, want in (("gain", g * U, 0.0), ("turn", turn.float(), 1 / (1 + g * g))):
        bf = built(feat, j, dU)
        s = KV.grades(M, F, bf, toks, at, GB)[cfg.row]  # F's run: its weights are M's (inputs only)
        site = bf.sites[j]
        cols = slice(site.off, site.rem)
        r = {"frob": KV.frob_ratios(bf)[cols], "input": KV.input_ratios(M, F, bf, toks, at, GB)[cfg.row, cols]}
        for k, ratio in r.items():
            ok = torch.isfinite(ratio) & (ratio > 0)
            t = (1 - s[cols][ok]) / ratio[ok]
            err = float(t.abs().max()) if want == 0 else float(((t - want) / want).abs().max())
            out[f"{name}_{k}"] = {"n": int(ok.sum()), "max_err": err, "t_mean": float(t.mean())}
    miss = [f"{k}: {v}" for k, v in out.items() if k != "site" and not (v["n"] > 0 and v["max_err"] <= th["control_tol"])]
    return out | {"miss": miss}


# ---- one item -------------------------------------------------------------------------------------------------------


def read_item(w: Any, item: str, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    path = RESULTS / "items" / f"{E226.KP.fname(item)}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return json.loads(path.read_text())
    t0 = time.time()
    F, M, toks, at, _ = w.make(item)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    z = np.load(RR.npz_path(item))
    changed = np.zeros(feat.n, dtype=bool)
    changed[sorted(feat.changed)] = True
    lacked = np.zeros(feat.n, dtype=bool)
    lacked[sorted(feat.lacked)] = True
    s64 = KV.grades(M, F, feat, toks, at, GB)
    s = s64[cfg.row].numpy()
    ratios = {"frob": KV.frob_ratios(feat).numpy(), "input": KV.input_ratios(M, F, feat, toks, at, GB)[cfg.row].numpy()}
    out = {"experiment": EXPERIMENT, "item": item, "n_vars": feat.n, "n_changed": int(changed.sum()), "n_windows": len(toks),
           "max_dev_s_vs_E226": float(np.abs(s64.numpy() - z["s"].astype(np.float64)).max()),
           "R_vs_E226": float(abs(s[z["main_in_K_polished"]].mean() - z["s"][cfg.row][z["main_in_K_polished"]].astype(np.float64).mean()))}  # fmt: skip
    miss = [f"changed set differs from E226's ({int((changed != z['changed']).sum())} variables)"] if not np.array_equal(changed, z["changed"]) else []
    miss += [f"{k} {k2} {v:.3g}" for k, c in feat.checks.items() for k2, v in c.items() if k2 in E226.CHECKS and not v <= th[f"{k2}_tol"]]
    miss += [f"R from saved K* moved {out['R_vs_E226']:.3g} from E226's"] if not out["R_vs_E226"] <= th["R_repro_tol"] else []
    if item == "do_nothing":
        miss += [f"do-nothing: {out['n_changed']} changed"] if out["n_changed"] else []
        nz = [r for r, v in ratios.items() if np.nanmax(np.abs(v)) != 0]
        miss += [f"do-nothing: non-zero {nz} ratio"] if nz else []
    if miss:
        stop(path.with_suffix(".stopped.json"), out, f"{item}: " + "; ".join(miss))
    out["searches"] = readings(z, s, ratios, lacked, cfg, th["ratio_min"])
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "large" / f"{E226.KP.fname(item)}.npz", s=s.astype(np.float32), ratio_frob=ratios["frob"].astype(np.float32),
                        ratio_input=ratios["input"].astype(np.float32))
    out["seconds"] = time.time() - t0
    dump(path, out)
    m = out["searches"]["main"]
    log(f"{item}: N {m['frob']['N']}, N_w {m['frob']['N_w']}, N_in {m['input']['N']}, R {m['frob']['R']:.6f}; {m['frob']['n_used']} of "
        f"{m['frob']['n_KC']} changed K* parts used ({m['frob']['n_lacked']} grafts, {m['frob']['n_small']} under the ratio floor, {m['frob']['n_t_over_1']} t > 1); "
        f"max |s - E226's| {out['max_dev_s_vs_E226']:.2g}; {out['seconds']:.0f} s")  # fmt: skip
    torch.cuda.empty_cache()
    return out


# ---- summary --------------------------------------------------------------------------------------------------------


def order_rule(v: dict[str, float | None], top: list[str], new: tuple[str, ...], band: float) -> dict[str, Any]:
    """min over `top` minus max over `new`: holds above the band, rival b at or below 0, no call between; not read if any is None."""
    if any(v[i] is None for i in (*top, *new)):
        return {"read": False}
    gap = min(v[i] for i in top) - max(v[i] for i in new)  # type: ignore[type-var]
    return {"read": True, "gap": gap, "band": band, "verdict": "holds" if gap > band else "rival b" if gap <= 0 else "no call"}


def spearman(a: list[float], b: list[float]) -> float | None:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if len(a) > 2 else None


def summary(items: dict[str, dict[str, Any]], known: tuple[str, ...], reuse: tuple[str, ...], new: tuple[str, ...]) -> dict[str, Any]:
    keys = {"N": ("frob", "N"), "N_w": ("frob", "N_w"), "N_in": ("input", "N")}
    get = {m: {i: items[i]["searches"]["main"][r][f] for i in items if i != "do_nothing"} for m, (r, f) in keys.items()}
    band = {}
    for m, (r, f) in keys.items():
        d = [abs(a - b) for i in known if (a := items[i]["searches"]["half1"][r][f]) is not None and (b := items[i]["searches"]["half2"][r][f]) is not None]
        band[m] = max(d) if d else 0.0  # degenerate: 0, the rule then a strict order
    out: dict[str, Any] = {"experiment": EXPERIMENT, "band": band, "known": {m: {i: get[m][i] for i in known} for m in keys},
                           "P1": order_rule(get["N"], ["x10"], new, band["N"]),
                           "side": {"P1_N_w": order_rule(get["N_w"], ["x10"], new, band["N_w"]),
                                    "P1_N_in": order_rule(get["N_in"], ["x10"], new, band["N_in"]),
                                    "all_reuse_N": order_rule(get["N"], list(reuse), new, band["N"])}}  # fmt: skip
    pairs = {p: v for p, v in RR.load(RR.PAIRS_JSON)["pairs"].items() if p.startswith("e187_")}
    trained = sorted({i for v in pairs.values() for i in (*v["reuse"], *v["new"])})
    size = RR.sizes(trained)
    for m in keys:
        gaps = {}
        for p, v in pairs.items():
            r, n = [get[m][i] for i in v["reuse"] if get[m][i] is not None], [get[m][i] for i in v["new"] if get[m][i] is not None]
            if r and n:
                gaps[p] = mean(r) - mean(n)
        st = [i for i in trained if get[m][i] is not None]
        out[f"pairs_{m}"] = {"gaps": gaps, "n": len(gaps), "reuse_higher": sum(g > 0 for g in gaps.values()),
                             "spearman_gap_vs_smaller_update": spearman(list(gaps.values()), [pairs[p]["smaller_update"] for p in gaps]),
                             "spearman_state_vs_size": spearman([get[m][i] for i in st], [math.log(size[i]) for i in st])}  # fmt: skip
    return out


def main(chosen: list[str] | None) -> int:
    cfg, th = load_config()
    c226 = E226.load_config()[0]
    every = ["do_nothing", *c226.known, *E226.e187_states()]
    todo = chosen or every
    assert set(todo) <= set(every), f"unknown items {sorted(set(todo) - set(every))}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226
    log(f"{EXPERIMENT} start; {len(todo)} items; settings {cfg}; thresholds {th}")
    (RESULTS / "items").mkdir(parents=True, exist_ok=True)
    w = E226.World()
    w.bases = KV.bases(w.uv, c226.split_tol)
    s0 = RESULTS / "stage0.json"
    if not (s0.is_file() and json.loads(s0.read_text()).get("done")):
        c = construction(w, cfg, th)
        log(f"stage 0 construction checks: {c}")
        if c["miss"]:
            stop(s0, c, "stage 0: " + "; ".join(c["miss"]))
        read_item(w, "do_nothing", cfg, th)
        dump(s0, c | {"done": True})
        log("STAGE 0 DONE")
    items = {i: read_item(w, i, cfg, th) for i in todo}
    if set(items) == set(every):
        dump(RESULTS / "summary.json", sm := summary(items, c226.known, c226.reuse, c226.new))
        log(f"summary: band {sm['band']}; P1 {sm['P1']}; side {sm['side']}")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=None, help="items by name (default: every item, do-nothing and the known items first)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.items)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

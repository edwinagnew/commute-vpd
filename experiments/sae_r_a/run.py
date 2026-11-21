"""E245 (cell P1.40 run 2): R under wording A at E234's base-only SAE latents. Plan
reports/plans/2026-11-21_plan_r-under-a-sae.md; card reports/plans/2026-11-21_plan_r-under-a-sae-card.md; settings,
thresholds and rules in results/predictions.json, which the run refuses unless committed unchanged. Variables, mechanisms
and F_K: commute.sae_feat; standing in: commute.stand_in; the search: commute.kstar_vpd.search; the R_A search over K*'s
changed members: E242's stand_in_search. Items are built by E242's World (E214's built edits, E237's keys, E232's pairs).

Per item: the gains (least squares over every position of D in F's own run); the controls (o_F recomputed bit for bit,
every unchanged-site gain exactly 1 and those variables stood in giving KL exactly 0, every variable stood in with
gain 1 giving M's logits within control_rel_tol: the positive control, its KL the item's floor); K* by E226's search
with bar (1 - alpha) KL(F || F_all_ablated) on all of D and, for the six built items, on each half; per K*, R_A = 1 -
(smallest set of K*'s standable changed members that must keep F's mechanism + K*'s changed members with gain <= 0) /
|K*| with KL(F || F_S) <= tol KL(F || M), tol 0.1 (= 1 - alpha) and on all of D also 0.05 and 0.2; a bar at or under
the floor is not read; the remainder share, 1 - KL(F || the 8 remainders stood in) / KL(F || M) (nan at KL(F || M) = 0).

Groups (each job reads the do-nothing first and alone, stage 0; any miss stops it): built (x2, x10, moved, rebuilt,
new_few, new_thin; after the six, the order check: unless every reuse item read has R_A above both new edits, STOP),
keys (E237's 15 scaled items), pairs (E187's nine pairs at matched size); keys and pairs refuse to start until the
order check has held. results/<out>/<item>.json per item (a rerun skips it), results/large/<out>_<item>.npz the gains
and, per search, K* and who keeps F's mechanism; every number is also printed; the last line is FINISHED.

    sbatch --comment='E245 built0' cluster/sae_r_a.sbatch built0 --group built --shard 0 3
    python experiments/sae_r_a/run.py --size rehearse --group built --tag rehearse
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import rule_retention as R
from commute import sae as S
from commute import sae_feat as SF
from commute import stand_in as SI
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

E242: Any = load_experiment("kstar_vpd_wording_a", "run")  # World, Read, stand_in_search, key_items, pair_sides
E234: Any = load_experiment("sae_train", "run")  # load_config, load_saes
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, dump, stop, fname = E242.log, E242.dump, E242.stop, E242.fname
GROUPS = ("built", "keys", "pairs")


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings" (+ sizes[size])
    alpha: float
    side_tols: tuple[float, ...]
    batch: int
    gain_batch: int
    reranks: int
    eval_cap: int
    chunk_div: int
    split_seed: int
    reuse: tuple[str, ...]
    new: tuple[str, ...]
    sae_size: str
    sae_seed: int
    only: tuple[str, ...]  # if set, only these items of the group (the rehearsal)
    out: str

    def __post_init__(self) -> None:
        assert 0 < self.alpha < 1 and self.reranks >= 0 and self.eval_cap > 0 and self.chunk_div >= 1
        assert set(self.reuse) == {"x2", "x10", "moved", "rebuilt"} and set(self.new) == {"new_few", "new_thin"}
        assert all(0 < t < 1 for t in self.side_tols) and set(self.only) <= {"do_nothing", *self.built}

    @property
    def built(self) -> tuple[str, ...]:
        return (*self.reuse, *self.new)

    @property
    def tol(self) -> float:
        return round(1 - self.alpha, 12)


def load_config(size: str) -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = dict(pred["settings"])
    sizes = s.pop("sizes")
    assert size == "scored" or size in sizes, size
    s |= sizes.get(size, {})
    return config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in s.items()}), pred["thresholds"]


def group_items(cfg: Config, g: str) -> list[str]:
    every = {"built": lambda: list(cfg.built), "keys": E242.key_items,
             "pairs": lambda: list(dict.fromkeys(i for r, n in E242.pair_sides().values() for i in (*r, *n)))}  # fmt: skip
    return [i for i in every[g]() if not cfg.only or i in cfg.only]


def load_saes(cfg: Config) -> dict[str, S.SAE]:
    c234, _ = E234.load_config(cfg.sae_size)
    got = E234.load_saes(c234, [f"{s}_s{cfg.sae_seed}" for s in SF.SITES], E234.RESULTS / "large" / cfg.sae_size)
    for ae in got.values():
        ae.requires_grad_(False)
    return {k.rsplit("_s", 1)[0]: ae for k, ae in got.items()}


def logits_at(F: R.Model, at: slice, x: Tensor) -> Tensor:
    return F.logits(x)[:, at]


def _rel(r: Tensor, m: Tensor) -> float:
    return float(((r - m).abs().amax((1, 2)) / m.abs().amax((1, 2))).max())


# ---- one item: its variables, gains and stand-in hooks ----------------------------------------------------------------


@dataclass
class Item:
    name: str
    F: R.Model
    M: R.Model
    toks: Tensor
    at: slice
    ss: list[SF.SAESite]
    feat: KV.Feat  # n and the changed variables, the layout E242's stand_in_search reads (no grafts: M's module lacks them)
    g: Tensor  # (n,) fp64 CPU gains
    si: SI.StandIn
    batches: list[Tensor]
    lF: list[Tensor]  # F's logits at the read positions per batch

    def full(self, x: Tensor) -> Tensor:
        return logits_at(self.F, self.at, x)

    def kl(self, t: Tensor) -> float:
        """KL(F || F's full forward with weights t stood in), mean over D."""
        return SI.kl_mean(self.si, self.full, self.batches, self.lF, t)

    def weights(self, idx: list[int] | Tensor) -> Tensor:
        t = torch.zeros(self.feat.n, device=self.toks.device)
        t[torch.as_tensor(idx, dtype=torch.long, device=self.toks.device)] = 1
        return t


def build(w: Any, saes: dict[str, S.SAE], item: str, cfg: Config) -> tuple[Item, dict[str, Any]]:
    F, M, toks, at, rec = w.make(item)
    for N in (F, M):
        N.net.requires_grad_(False)
        for e in N.extra.values():
            e.requires_grad_(False)
    ss = SF.sites(F, M, saes, toks.shape[1])
    feat = KV.Feat([], [], SF.n_vars(ss), SF.changed_vars(ss), set(), {})
    g = SI.gains(F.net, ss, feat.n, list(toks.split(cfg.gain_batch)))
    si, batches = SI.StandIn(F.net, ss, g, toks.device), list(toks.split(cfg.batch))
    with torch.no_grad():
        lF = [logits_at(F, at, x) for x in batches]
    return Item(item, F, M, toks, at, ss, feat, g, si, batches, lF), rec


# ---- controls --------------------------------------------------------------------------------------------------------


@torch.no_grad()
def pieces_floor(it: Item) -> dict[str, float]:
    """KL(F || F with every site's output rebuilt from its SAE pieces, sum_u f_u d_u + e) and the max relative logit
    error: mathematically 0, so the fp32 floor of writing a module output through the SAE's pieces (the do-nothing's)."""

    def rebuild(s: SF.SAESite, _m: Any, _a: Any, o: Tensor) -> Tensor:
        f, e = s.writes(o)
        return s.ae.decoder(f) + e

    hs = [it.F.net.get_submodule(s.module).register_forward_hook(partial(rebuild, s)) for s in it.ss]
    try:
        rs = [it.full(x) for x in it.batches]
    finally:
        for h in hs:
            h.remove()
    return {"pieces_kl": float(torch.cat([R.kl(lf, r) for lf, r in zip(it.lF, rs, strict=True)]).mean()),
            "pieces_err": max(_rel(r, lf) for lf, r in zip(it.lF, rs, strict=True))}  # fmt: skip


@torch.no_grad()
def revert(it: Item) -> dict[str, float]:
    """The positive control: every variable stood in with gain 1 writes M's module output at every site, so M's logits
    (KL(M || it) is mathematically 0: the item's own floor)."""
    g_fit, it.si.g = it.si.g, torch.ones_like(it.si.g)
    ones, errs, kls = torch.ones(it.feat.n, device=it.toks.device), [], []
    try:
        for x in it.batches:
            r, m = it.si.run(it.full, x, ones), it.M.logits(x)[:, it.at]
            errs.append(_rel(r, m))
            kls.append(R.kl(m, r))
    finally:
        it.si.g = g_fit
    return {"revert_err": max(errs), "revert_kl": float(torch.cat(kls).mean())}


def controls(it: Item, th: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    g, n = it.g, it.feat.n
    unch = [u for u in range(n) if u not in it.feat.changed]
    chg = torch.tensor(sorted(it.feat.changed), dtype=torch.long)
    out: dict[str, Any] = {"recompute_differs": SF.recompute_differs(it.F.net, it.ss, it.batches[0]), "changed_sites": [s.name for s in it.ss if s.changed],
                           "gain_dev_unchanged": float((g[unch] - 1).abs().max()) if unch else 0.0, "gain_dev_all": float((g - 1).abs().max()),
                           "kl_unchanged_stood_in": it.kl(it.weights(unch)) if unch else 0.0}  # fmt: skip
    t0 = time.time()
    out["kl_all_stood_in"] = it.kl(torch.ones(n, device=it.toks.device))
    out["seconds_per_eval"] = time.time() - t0
    out |= revert(it)
    out["gain_changed_quantiles"] = torch.quantile(g[chg], torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)).tolist() if len(chg) else None
    out["nonpositive_gains"] = int((g[chg] <= 0).sum()) if len(chg) else 0
    miss = [f"o_F recomputed differs at {out['recompute_differs']}"] if out["recompute_differs"] else []
    miss += [f"gain of an unchanged-site variable off 1 by {out['gain_dev_unchanged']:.3g}"] if out["gain_dev_unchanged"] != 0 else []
    miss += [f"KL with every unchanged-site variable stood in {out['kl_unchanged_stood_in']:.3g}"] if out["kl_unchanged_stood_in"] != 0 else []
    miss += [f"positive control: logits off M's by {out['revert_err']:.3g}"] if not out["revert_err"] <= th["control_rel_tol"] else []
    miss += ["a gain is not finite"] if not bool(torch.isfinite(g).all()) else []
    if it.name == "do_nothing":
        out |= pieces_floor(it)
        miss += [f"do-nothing: changed sites {out['changed_sites']}, gains off 1 by {out['gain_dev_all']:.3g}, KL all stood in {out['kl_all_stood_in']:.3g}"
                 ] if out["changed_sites"] or out["gain_dev_all"] != 0 or out["kl_all_stood_in"] != 0 else []  # fmt: skip
    return out, miss


def remainder_share(it: Item, kl_fm: float) -> dict[str, float]:
    """1 - KL(F || only the 8 remainders stood in, gains fitted) / KL(F || M); nan where KL(F || M) = 0."""
    kl = it.kl(it.weights([s.rem for s in it.ss]))
    return {"kl_remainders_stood_in": kl, "remainder_share": 1 - kl / kl_fm if kl_fm > 0 else float("nan")}


# ---- searches --------------------------------------------------------------------------------------------------------


def k_star(it: Item, sub: SF.Ctx, cfg: Config, label: str) -> tuple[dict[str, Any], list[int], dict[str, np.ndarray]]:
    """E226's search on these windows: bar (1 - alpha) KL(F || F_all_ablated), attribution d KL / d keep."""
    t0, n, bar = time.time(), it.feat.n, (1 - cfg.alpha) * float(sub.kl_none.mean())
    f = KV.search(lambda keep: SF.kl_mean(sub, keep), lambda c: SF.attribution(sub, c), n, bar, cfg.reranks, cfg.eval_cap, cfg.chunk_div, log)
    keep = torch.zeros(n, dtype=torch.bool)
    keep[f.polished] = True
    rems = {s.rem for s in it.ss}
    rec = {"bar_K": bar, "kl_none": float(sub.kl_none.mean()), "kl_fm": float(sub.kl_fm.mean()), "kl_at_K": SF.kl_mean(sub, keep), "size_K": len(f.polished),
           "size_prefix": len(f.prefix), "changed_in_K": sum(u in it.feat.changed for u in f.polished), "remainders_in_K": sum(u in rems for u in f.polished),
           "evals_K": f.evals, "capped_K": f.capped, "chunk_min": f.chunk_min, "prefix_sizes": f.prefix_sizes, "seconds_K": time.time() - t0}  # fmt: skip
    log(f"  {label}: |K*| {rec['size_K']} (prefix {rec['size_prefix']}), {rec['changed_in_K']} at changed sites, {rec['remainders_in_K']} remainders; "
        f"KL at K* {rec['kl_at_K']:.6g}, bar {bar:.6g}; {f.evals} evaluations{' (cap hit)' if f.capped else ''}, {rec['seconds_K']:.0f} s")  # fmt: skip
    return rec, f.polished, {"in_K_polished": keep.numpy(), "in_K_prefix": np.isin(np.arange(n), f.prefix)}


def one_variant(it: Item, sub: SF.Ctx, cfg: Config, tols: list[float], floor: float, label: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """K* on these windows, then R_A at each tolerance (a bar at or under the floor is not read; the do-nothing reads 1)."""
    rec, K, arrays = k_star(it, sub, cfg, label)
    rec["floor"], kl_fm = floor, rec["kl_fm"]
    rd = E242.Read(it.si, it.full, sub.toks, sub.lF)
    for tol in tols:
        if it.name != "do_nothing" and not tol * kl_fm > floor:
            log(f"  {label} tol {tol:g}: not read, {tol:g} KL(F || M) {tol * kl_fm:.3g} at or under the floor {floor:.3g}")
            rec[f"tol{tol:g}"] = {"R_A": None, "reading": "not read: tol KL(F || M) at or under the floor"}
            continue
        rec[f"tol{tol:g}"], arrays[f"keeps_F_tol{tol:g}"] = E242.stand_in_search(rd, it.feat, it.g, K, tol * kl_fm, cfg, f"{label} tol {tol:g}")
    rec["R_A"] = rec[f"tol{cfg.tol:g}"]["R_A"]
    return rec, arrays


def searches(it: Item, ctx: SF.Ctx, cfg: Config, floor: float, npz: Path) -> dict[str, Any]:
    """All of D at every tolerance; for the six built items also each half of D (E226's permutation) at tol."""
    variants = {"main": torch.arange(len(it.toks))}
    if it.name in cfg.built:
        perm = torch.randperm(len(it.toks), generator=torch.Generator().manual_seed(cfg.split_seed))
        variants |= {"half1": perm[: len(it.toks) // 2], "half2": perm[len(it.toks) // 2 :]}
    recs, arrays = {}, {"gain": it.g.numpy(), "changed": np.isin(np.arange(it.feat.n), sorted(it.feat.changed))}
    for k, idx in variants.items():
        tols = sorted({cfg.tol, *cfg.side_tols}) if k == "main" else [cfg.tol]
        recs[k], a = one_variant(it, ctx if k == "main" else ctx.subset(idx, cfg.batch), cfg, tols, floor, f"{it.name} {k}")
        arrays |= {f"{k}_{name}": v for name, v in a.items()}
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **arrays)
    return recs


def read_item(w: Any, saes: dict[str, S.SAE], item: str, cfg: Config, th: dict[str, Any], out: Path, floor: float) -> dict[str, Any]:
    path = out / f"{fname(item)}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return json.loads(path.read_text())
    t0 = time.time()
    it, rec = build(w, saes, item, cfg)
    c, miss = controls(it, th)
    ctx = SF.prepare(it.F, it.M, it.ss, it.toks, it.at, cfg.batch)
    base: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "n_windows": len(it.toks), "window": it.toks.shape[1], "read_all": it.at.start is None,
                            "n_vars": it.feat.n, "kl_none": float(ctx.kl_none.mean()), "kl_fm": float(ctx.kl_fm.mean()), "gate_all": SF.gate_all(ctx), "controls": c} | rec  # fmt: skip
    miss += [f"all kept differs from F by {base['gate_all']:.3g}"] if base["gate_all"] != 0 else []
    log(f"{item}: {len(it.toks)} windows x {it.toks.shape[1]}, {it.feat.n} variables; KL(F || F_all_ablated) {base['kl_none']:.6g}, "
        f"KL(F || M) {base['kl_fm']:.9g}; controls {json.dumps(c)}")  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    if miss:
        stop(out / f"{fname(item)}.stopped.json", base, f"{item}: " + "; ".join(miss))
    if not base["kl_none"] > th["kl_floor"]:
        dump(path, base | {"reading": "no behaviour"})
        return base | {"reading": "no behaviour"}
    base |= remainder_share(it, base["kl_fm"])
    log(f"{item}: remainder share {base['remainder_share']:.6g} (KL with the 8 remainders stood in {base['kl_remainders_stood_in']:.6g})")
    s = searches(it, ctx, cfg, max(floor, c["revert_kl"]), RESULTS / "large" / f"{out.name}_{fname(item)}.npz")
    res = base | {"reading": "read", "searches": s, "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, res)
    log(f"{item}: done in {res['seconds']:.0f} s; R_A " + ", ".join(f"{k} {v['R_A']}" for k, v in s["main"].items() if k.startswith("tol"))
        + f"; halves {[s[h]['R_A'] for h in s if h != 'main']}; remainder share {res['remainder_share']:.6g}; peak GPU {res['peak_gpu_gb']:.2f} GB")  # fmt: skip
    it.si.close()
    ctx.hy.close()
    del ctx, it
    torch.cuda.empty_cache()
    return res


# ---- stage 0, the order check, main ----------------------------------------------------------------------------------


def stage0(w: Any, saes: dict[str, S.SAE], cfg: Config, th: dict[str, Any], tag: str, todo: list[str]) -> float:
    """The do-nothing first and alone; returns the job's floor (the larger of its pieces KL and its revert KL)."""
    out = RESULTS / cfg.out
    path = out / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return json.loads(path.read_text())["kl_floor"]
    it = read_item(w, saes, "do_nothing", cfg, th, out / f"stage0_{tag}", 0.0)
    c, m = it["controls"], it["searches"]["main"]
    sec_k, sec_s = m["seconds_K"] / max(m["evals_K"], 1), c["seconds_per_eval"]
    n_k = sum(3 if i in cfg.built else 1 for i in todo)
    n_s = sum(3 + len(cfg.side_tols) if i in cfg.built else 1 + len(cfg.side_tols) for i in todo)
    hours = cfg.eval_cap * (n_k * sec_k + n_s * sec_s) / 3600
    rec = {"experiment": EXPERIMENT, "R_A": m["R_A"], "seconds_per_K_eval": sec_k, "seconds_per_stand_in_eval": sec_s, "projected_hours_max": hours,
           "pieces_kl": c["pieces_kl"], "pieces_err": c["pieces_err"], "revert_kl": c["revert_kl"], "revert_err": c["revert_err"],
           "kl_floor": max(c["pieces_kl"], c["revert_kl"]), "peak_gpu_gb": it["peak_gpu_gb"]}  # fmt: skip
    miss = [f"do-nothing R_A {m['R_A']}"] if m["R_A"] != 1 else []
    miss += [f"floor {rec['kl_floor']:.3g} over floor_max {th['floor_max']:.3g}"] if not rec["kl_floor"] <= th["floor_max"] else []
    miss += [f"projected up to {hours:.2f} h over {th['shard_hours_max']} h"] if hours > th["shard_hours_max"] else []
    log(f"stage 0: {json.dumps(rec)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, rec, "stage 0: " + "; ".join(miss))
    dump(path, rec | {"done": True})
    log("STAGE 0 DONE")
    return rec["kl_floor"]


def built_order(cfg: Config) -> None:
    """The whip-lead's stop: once all six built items are written, unless every reuse item read has R_A (tol) above both
    new edits, STOP (the keys and pairs are then not read). An item not read (bar under the floor) is left out, listed."""
    out = RESULTS / cfg.out
    files = {i: out / f"{fname(i)}.json" for i in cfg.built}
    if not all(p.is_file() for p in files.values()):
        log(f"order check waits for {[i for i, p in files.items() if not p.is_file()]}")
        return
    recs = {i: json.loads(p.read_text()) for i, p in files.items()}
    ra: dict[str, float | None] = {i: r["searches"]["main"]["R_A"] if r.get("reading") == "read" else None for i, r in recs.items()}
    reuse: list[float] = [v for i in cfg.reuse if (v := ra[i]) is not None]
    new: list[float] = [v for i in cfg.new if (v := ra[i]) is not None]
    rec = {"R_A": ra, "not_read": [i for i, v in ra.items() if v is None], "min_reuse": min(reuse, default=None), "max_new": max(new, default=None),
           "holds": bool(reuse and new and min(reuse) > max(new))}  # fmt: skip
    log(f"order check: {json.dumps(rec)}")
    if not rec["holds"]:
        stop(out / "built_order.json", rec, "built order: not every reuse item reads R_A above both new edits; keys and pairs not read; ask whip-lead")
    dump(out / "built_order.json", rec)
    log("ORDER CHECK HOLDS")


def main(group: str, shard: tuple[int, int], tag: str, size: str) -> int:
    cfg, th = load_config(size)
    todo = group_items(cfg, group)[shard[0] :: shard[1]]
    order = RESULTS / cfg.out / "built_order.json"
    if group != "built" and not (order.is_file() and json.loads(order.read_text()).get("holds") and "stopped" not in json.loads(order.read_text())):
        log(f"STOP: group {group} waits for the built group's order check ({order})")
        return 1
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226
    log(f"{EXPERIMENT} start, size {size}, group {group}, tag {tag}; {len(todo)} items {todo}; settings {asdict(cfg)}; thresholds {th}")
    w, saes = E242.World(E242.load_config()[0]), load_saes(cfg)
    floor = stage0(w, saes, cfg, th, tag, todo)
    for k, item in enumerate(todo):
        read_item(w, saes, item, cfg, th, RESULTS / cfg.out, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    if group == "built":
        built_order(cfg)
    log(f"FINISHED {EXPERIMENT} {group} {tag}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=GROUPS, required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item of the group from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/<out>/stage0_<tag>.json")
    ap.add_argument("--size", choices=("scored", "rehearse"), default="scored", help="predictions.json settings + sizes[size]")
    ap.add_argument("--pred", type=Path, default=PRED, help="the filed predictions file (default results/predictions.json; the pairs group runs with results/predictions_pairs_amend_2026-11-21.json)")
    args = ap.parse_args()
    PRED = args.pred.resolve()  # load_config reads this module global
    require_filed(PRED)
    code = 1
    try:
        code = main(args.group, (args.shard[0], args.shard[1]), args.tag, args.size)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

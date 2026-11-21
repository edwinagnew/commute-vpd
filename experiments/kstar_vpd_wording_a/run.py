"""E???: R under the definition's wording A (user, 2026-11-20 ~19:50, "A, reduced redo on built edits and nine pairs"),
at the VPD featurization with site remainders. Plan reports/plans/2026-11-20_plan_r-under-a.md; card
reports/plans/2026-11-20_plan_r-under-a-card.md; settings, thresholds and rules in results/predictions.json, which the
run refuses unless committed unchanged. The measurement is commute.stand_in on E226's variables (commute.kstar_vpd).

R_A per item, at E226's circuit K* (polished set of each search): the largest share of K* whose variables can take M's
mechanism, each times its least-squares output gain, all at once, in F's own forward, with KL(F || F_S) at the read
positions, mean over D, still within the circuit's allowance (1 - alpha) KL(F || F_all_ablated). Unchanged variables
stand in exactly; grafts never can. The changed members of K* are searched with E226's search code (`kstar_vpd.search`)
for the smallest set that must keep F's mechanism; R_A = 1 - (that set + K*'s grafts) / |K*|, a lower bound.

Items, each on E226's windows and read positions:
    stage 0  do_nothing (E226's stage 0 state and K*)          F = M: R_A 1, every gain exactly 1
    known    new_thin, new_few, x2, x10, moved, rebuilt        E226's saved K* on all of D and its two halves (the band)
    pairs    E187's nine pairs at matched size (E232's items)  smaller side: E226's saved K*; larger side times E187's c:
                                                               K* searched here by E226's read_item, unchanged
Per item results/items/<item>.json is written when it ends (a rerun skips it), with every number also in the log;
results/large/<item>.npz holds the gains and, per search, which K* members keep F's mechanism. Every control of
predictions.json R0 stops the job on a miss.

    sbatch --comment='E??? known0' cluster/kstar_vpd_wording_a.sbatch --items new_thin --tag known0
    python experiments/kstar_vpd_wording_a/run.py --shard 0 12 --pairs --tag pairs0
    python experiments/kstar_vpd_wording_a/run.py --summary          (no GPU: results/summary.json from the item files)
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
from commute import rule_retention as R
from commute import stand_in as SI
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

E226: Any = load_experiment("kstar_vpd", "run")  # World, read_item, guards, load_config, KP, KP_TH, log/dump/stop
SMR: Any = load_experiment("kstar_size_matched", "run")  # E232: World (scaled items), load_config, E221
SMS: Any = load_experiment("kstar_size_matched", "score")  # matched: the nine pairs with the larger side scaled
KP = E226.KP
SAVED = E226.RESULTS  # E226's own results, read before `main` points E226's writes here
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, dump, stop, fname = E226.log, E226.dump, E226.stop, KP.fname


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    eval_cap: int  # evaluations per stand-in search
    reranks: int
    chunk_div: int
    gain_batch: int  # windows per batch of the gain fit
    reuse: tuple[str, ...]
    new: tuple[str, ...]

    def __post_init__(self) -> None:
        assert self.eval_cap > 0 and self.reranks >= 0 and self.chunk_div >= 1 and self.gain_batch > 0
        assert set(self.reuse) == {"x2", "x10", "moved", "rebuilt"} and set(self.new) == {"new_few", "new_thin"}

    @property
    def known(self) -> tuple[str, ...]:
        return (*self.new, *self.reuse)


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def pair_sides() -> dict[str, tuple[list[str], list[str]]]:
    """E187's nine pairs, (reuse states, new states), the larger side's states as E232's scaled items."""
    return SMS.matched({p: v for p, v in KP.pairs(KP.load_config()[0]).items() if p.startswith("e187_")})


def pair_items() -> list[str]:
    return list(dict.fromkeys(i for r, n in pair_sides().values() for i in (*r, *n)))


class World(E226.World):
    """E226's world (the built items, the decomposition, the split's bases); E232's world for the pairs' states."""

    def __init__(self) -> None:
        super().__init__()
        self.sm: Any = None

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        if item in E226.E214_ITEMS:
            return super().make(item)
        if self.sm is None:
            self.sm = SMR.World(SMR.load_config()[0])
            assert all(torch.equal(self.sm.M.net.get_submodule(k).weight, self.kl.M.net.get_submodule(k).weight) for k in self.keys)
        F, toks, at, rec = self.sm.make(item) if item in self.sm.items else SMR.E221.World.make(self.sm, item)
        return F, self.sm.M, toks, at, rec


def k_source(w: World, item: str, c226: Any, th226: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    """E226's reading of the item and its per-variable arrays: saved, or (a scaled state) made here by E226's read_item."""
    if item == "do_nothing":
        return json.loads((SAVED / "stage0_pod0" / "do_nothing.json").read_text()), SAVED / "large" / "stage0_pod0_do_nothing.npz"
    if (SAVED / "items" / f"{fname(item)}.json").is_file():
        return json.loads((SAVED / "items" / f"{fname(item)}.json").read_text()), SAVED / "large" / f"items_{fname(item)}.npz"
    rec = E226.read_item(w, item, c226, th226, RESULTS / "kstar")  # writes results/kstar/<item>.json, results/large/kstar_<item>.npz
    return rec, (RESULTS / "large" / f"kstar_{fname(item)}.npz") if rec.get("reading") == "read" else None


# ---- the stand-in search ---------------------------------------------------------------------------------------------


@dataclass
class Read:
    si: SI.StandIn
    F: R.Model
    toks: list[Tensor]
    lF: list[Tensor]  # F's own logits at `at`
    at: slice

    def kl(self, t: Tensor) -> float:
        return SI.kl_mean(self.si, self.F, self.toks, self.lF, self.at, t)


def stand_in_search(rd: Read, feat: KV.Feat, K: list[int], bar: float, cfg: Config, label: str) -> tuple[dict[str, Any], np.ndarray]:
    """R_A at K: the smallest set of K's changed members (grafts aside) that must keep F's mechanism, every other member
    stood in. Returns the record and, over all variables, which ones keep F's mechanism (K's grafts included)."""
    t0, dev = time.time(), rd.lF[0].device
    lacked = [u for u in K if u in feat.lacked]
    cand = torch.tensor(sorted(u for u in K if u in feat.changed and u not in feat.lacked), dtype=torch.long)

    def weights(stood: Tensor) -> Tensor:
        t = torch.zeros(feat.n, device=dev)
        t[cand.to(dev)] = stood.to(dev, torch.float32)
        return t

    kl_all = rd.kl(weights(torch.ones(len(cand)))) if len(cand) else 0.0
    if kl_all <= bar:  # every member stood in at once preserves the behaviour: nothing to search
        kept, evals, capped, prefix = [], int(len(cand) > 0), False, []
    else:
        f = KV.search(lambda keep: rd.kl(weights((~keep).float())), lambda c: SI.attribution(rd.si, rd.F, rd.toks, rd.lF, rd.at, weights(1 - c))[cand],
                      len(cand), bar, cfg.reranks, cfg.eval_cap, cfg.chunk_div, log)  # fmt: skip
        kept, evals, capped, prefix = cand[f.polished].tolist(), f.evals + 1, f.capped, cand[f.prefix].tolist()
    keep = np.zeros(feat.n, dtype=bool)
    keep[kept + lacked] = True
    stood = torch.ones(len(cand), dtype=torch.bool)
    stood[torch.isin(cand, torch.tensor(kept, dtype=torch.long))] = False
    kl_found = rd.kl(weights(stood.float())) if len(cand) else 0.0
    rec = {"size_K": len(K), "changed_in_K": len(cand), "grafts_in_K": len(lacked), "kept_F": len(kept), "kept_F_prefix": len(prefix) if prefix else None,
           "R_A": 1 - (len(kept) + len(lacked)) / len(K), "lower_bound_1_minus_f": 1 - (len(cand) + len(lacked)) / len(K), "bar": bar,
           "kl_all_stood_in": kl_all, "kl_at_found": kl_found, "evals": evals, "capped": capped, "seconds": time.time() - t0}  # fmt: skip
    log(f"  {label}: R_A {rec['R_A']:.6f} (1 - f {rec['lower_bound_1_minus_f']:.6f}); |K*| {len(K)}, changed {len(cand)}, keep F's {len(kept)}, "
        f"grafts {len(lacked)}; KL all stood in {kl_all:.6g}, at the found set {kl_found:.6g}, bar {bar:.6g}; {evals} evaluations"
        f"{' (cap hit)' if capped else ''}, {rec['seconds']:.0f} s")  # fmt: skip
    return rec, keep


# ---- one item -------------------------------------------------------------------------------------------------------


def controls(rd: Read, M: R.Model, feat: KV.Feat, g: Tensor, th: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Every unchanged variable's gain exactly 1; all of them stood in at once leaves F's logits exactly (KL 0); every
    changed variable stood in with gain 1 gives M's logits (items without grafts), within revert_tol."""
    unchanged = torch.tensor([u for u in range(feat.n) if u not in feat.changed], dtype=torch.long)
    dev = rd.lF[0].device
    t = torch.zeros(feat.n, device=dev)
    t[unchanged.to(dev)] = 1
    t0 = time.time()
    out: dict[str, Any] = {"gain_dev_unchanged": float((g[unchanged] - 1).abs().max()), "kl_unchanged_stood_in": rd.kl(t)}
    out["seconds_per_eval"] = time.time() - t0
    out["revert_all_err"] = None
    if not feat.grafts:
        g_fit, rd.si.g = rd.si.g, torch.ones_like(rd.si.g)
        t = torch.zeros(feat.n, device=dev)
        t[torch.tensor(sorted(feat.changed), dtype=torch.long, device=dev)] = 1
        with torch.no_grad():
            errs = [float(((rd.si.logits(rd.F, x, rd.at, t) - (m := M.logits(x)[:, rd.at])).abs().amax((1, 2)) / m.abs().amax((1, 2))).max()) for x in rd.toks]
        rd.si.g, out["revert_all_err"] = g_fit, max(errs)
    changed = torch.tensor(sorted(feat.changed - feat.lacked), dtype=torch.long)
    out["gain_changed_quantiles"] = torch.quantile(g[changed], torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)).tolist() if len(changed) else None
    miss = [f"gain of an unchanged variable off 1 by {out['gain_dev_unchanged']:.3g}"] if out["gain_dev_unchanged"] != 0 else []
    miss += [f"KL with every unchanged variable stood in {out['kl_unchanged_stood_in']:.3g}"] if out["kl_unchanged_stood_in"] != 0 else []
    miss += [f"every change reverted: logits off M's by {out['revert_all_err']:.3g}"] if out["revert_all_err"] is not None and not out["revert_all_err"] <= th["revert_tol"] else []
    miss += ["a gain is not finite"] if not bool(torch.isfinite(g).all()) else []
    return out, miss


def read_item(w: World, item: str, cfg: Config, th: dict[str, Any], out: Path) -> dict[str, Any]:
    path = out / f"{fname(item)}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return json.loads(path.read_text())
    t0 = time.time()
    c226, th226 = E226.load_config()
    r226, npz = k_source(w, item, c226, th226)
    base: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "R_old": None}
    if npz is None:
        dump(path, base | {"reading": "no behaviour (E226's read_item)"})
        return base | {"reading": "no behaviour"}
    arrays = np.load(npz)
    F, M, toks, at, rec = w.make(item)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    ctx = KV.prepare(F, M, feat, toks, at, c226.batch)
    base |= {"n_windows": len(toks), "n_vars": feat.n, "n_changed": len(feat.changed), "n_grafts": len(feat.grafts), "kl_none": float(ctx.kl_none.mean()),
             "kl_none_E226": r226["kl_none"], "kl_fm": float(ctx.kl_fm.mean()), "gates": KV.gates(ctx, feat.n),
             "R_old": r226["searches"]["main"]["polished"]["R_read"]} | rec  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    miss = E226.guards(item, feat, KV.undecomposed_differences(M, F, w.keys), base["gates"], th226) + KP.load_checks(item, base, E226.KP_TH)
    miss += [f"KL(F || F_empty) {base['kl_none']:.9g}, E226's {r226['kl_none']:.9g}"] if not abs(base["kl_none"] / r226["kl_none"] - 1) <= th["kl_repro_rel"] else []
    g = SI.gains(F, feat, toks, cfg.gain_batch)
    batches = list(toks.split(c226.batch))
    with torch.no_grad():
        rd = Read(SI.StandIn(F, feat, g), F, batches, [F.logits(x)[:, at] for x in batches], at)
    ctrl, m2 = controls(rd, M, feat, g, th)
    base["controls"] = ctrl
    log(f"{item}: {len(toks)} windows, {feat.n} variables, {len(feat.changed)} changed, {len(feat.grafts)} grafts; KL(F || F_empty) {base['kl_none']:.6g} "
        f"(E226 {r226['kl_none']:.6g}), KL(F || M) {base['kl_fm']:.6g}; old R {base['R_old']:.6f}; controls {ctrl}")  # fmt: skip
    if miss + m2:
        stop(out / f"{fname(item)}.stopped.json", base, f"{item}: " + "; ".join(miss + m2))
    variants = {"main": torch.arange(len(toks))}
    if "half1_in_K_polished" in arrays.files:  # the six built items: E226's two halves of D, same permutation
        perm = torch.randperm(len(toks), generator=torch.Generator().manual_seed(c226.split_seed))
        variants |= {"half1": perm[: len(toks) // 2], "half2": perm[len(toks) // 2 :]}
    searches, keeps = {}, {}
    for k, idx in variants.items():
        sub = ctx if k == "main" else ctx.subset(idx, c226.batch)
        K = np.flatnonzero(arrays[f"{k}_in_K_polished"]).tolist()
        bar = (1 - c226.alpha) * float(sub.kl_none.mean())
        keepK = torch.zeros(feat.n, dtype=torch.bool)
        keepK[K] = True
        kl_K = KV.kl_mean(sub, keepK)
        if not kl_K <= bar * (1 + th["kl_repro_rel"]):
            stop(out / f"{fname(item)}.stopped.json", base, f"{item} {k}: E226's K* gives KL {kl_K:.9g} over its bar {bar:.9g}")
        rdk = rd if k == "main" else Read(rd.si, F, list(torch.cat(batches)[idx.to(toks.device)].split(c226.batch)), list(torch.cat(rd.lF)[idx.to(toks.device)].split(c226.batch)), at)
        searches[k], keeps[k] = stand_in_search(rdk, feat, K, bar, cfg, f"{item} {k}")
        searches[k]["kl_at_K"] = kl_K
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "large" / f"{out.name}_{fname(item)}.npz", gain=g.numpy(), changed=np.isin(np.arange(feat.n), sorted(feat.changed)),
                        **{f"{k}_keeps_F": v for k, v in keeps.items()})  # fmt: skip
    res = base | {"reading": "read", "searches": searches, "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, res)
    log(f"{item}: done in {res['seconds']:.0f} s; R_A main {searches['main']['R_A']:.6f}, old R {base['R_old']:.6f}; peak GPU {res['peak_gpu_gb']:.2f} GB")
    rd.si.close()
    ctx.hy.close()
    del ctx, rd, F
    torch.cuda.empty_cache()
    return res


# ---- stage 0, the summary, main --------------------------------------------------------------------------------------


def stage0(w: World, cfg: Config, th: dict[str, Any], tag: str, n_searches: int) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    it = read_item(w, "do_nothing", cfg, th, RESULTS / f"stage0_{tag}")
    m, c = it["searches"]["main"], it["controls"]
    hours = c["seconds_per_eval"] * cfg.eval_cap * n_searches / 3600
    out = {"experiment": EXPERIMENT, "R_A": m["R_A"], "n_changed": it["n_changed"], "evals": m["evals"], "kl_unchanged_stood_in": c["kl_unchanged_stood_in"],
           "gain_dev_unchanged": c["gain_dev_unchanged"], "seconds_per_eval": c["seconds_per_eval"], "projected_hours_max": hours}  # fmt: skip
    miss = [f"do-nothing: {out}"] if not (it["n_changed"] == 0 and m["R_A"] == 1 and m["evals"] == 0) else []
    miss += [f"projected up to {hours:.1f} h over {th['shard_hours_max']} h"] if hours > th["shard_hours_max"] else []
    log(f"stage 0: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def summary(cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    """Band, P1 (every reuse item above both new items on R_A), and the pairs' gaps under R_A and the old R (descriptive)."""
    its = {p.stem: json.loads(p.read_text()) for p in (RESULTS / "items").glob("*.json") if not p.stem.endswith(".stopped")}
    read = {k: v for k, v in its.items() if v.get("reading") == "read"}

    def ra(i: str, k: str = "main") -> float:
        return read[fname(i)]["searches"][k]["R_A"]

    out: dict[str, Any] = {"experiment": EXPERIMENT, "known": {i: {"R_A": ra(i), "R_old": read[fname(i)]["R_old"]} for i in cfg.known if fname(i) in read}}
    if all(fname(i) in read for i in cfg.known):
        d = [abs(ra(i, "half1") - ra(i, "half2")) for i in cfg.known]
        band = max(d) if max(d) > 0 else th["band_min"]
        gap = min(ra(i) for i in cfg.reuse) - max(ra(i) for i in cfg.new)
        out |= {"band": band, "band_degenerate": max(d) == 0, "P1": {"gap": gap, "verdict": "holds" if gap > band else "rival b" if gap <= 0 else "no call"}}
    pairs: dict[str, dict[str, float]] = {}
    for p, (r, n) in pair_sides().items():
        if all(fname(i) in read for i in (*r, *n)):
            side = {k: [float(np.mean([ra(i) if k == "R_A" else read[fname(i)]["R_old"] for i in xs])) for xs in (r, n)] for k in ("R_A", "R_old")}
            pairs[p] = {k: a - b for k, (a, b) in side.items()}
    out["pairs"] = {"gaps": pairs, "n_read": len(pairs), "reuse_higher_R_A": sum(v["R_A"] > 0 for v in pairs.values()),
                    "reuse_higher_R_old": sum(v["R_old"] > 0 for v in pairs.values())}  # fmt: skip
    return out


def main(chosen: list[str] | None, use_pairs: bool, shard: tuple[int, int], tag: str) -> int:
    cfg, th = load_config()
    every = pair_items() if use_pairs else list(cfg.known)
    todo = chosen or every[shard[0] :: shard[1]]
    assert set(todo) <= set(cfg.known) | set(pair_items()), f"unknown items {todo}"
    E226.RESULTS, E226.EXPERIMENT = RESULTS, EXPERIMENT  # E226's read_item (K* of the scaled states) writes here
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226
    log(f"{EXPERIMENT} start, tag {tag}; {len(todo)} items {todo}; settings {asdict(cfg)}; thresholds {th}")
    w = World()
    w.bases = KV.bases(w.uv, E226.load_config()[0].split_tol)
    stage0(w, cfg, th, tag, sum(3 if i in cfg.known else 1 for i in todo))
    for k, item in enumerate(todo):
        read_item(w, item, cfg, th, RESULTS / "items")
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=None, help="items by name (default: this shard of the known items, or of the pairs' items with --pairs)")
    ap.add_argument("--pairs", action="store_true", help="shard over E187's nine pairs' items at matched size instead of the six built items")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    ap.add_argument("--summary", action="store_true", help="only write results/summary.json from the item files (no GPU)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        if args.summary:
            dump(RESULTS / "summary.json", sm := summary(*load_config()))
            log(json.dumps(sm))
            code = 0
        else:
            code = main(args.items, args.pairs, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

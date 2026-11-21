"""E242: R under the definition's wording A (cell P1.46; user, 2026-11-20 ~19:50, "A, reduced redo on built edits and
nine pairs"), at the VPD featurization with site remainders. Plan reports/plans/2026-11-20_plan_r-under-a.md; card
reports/plans/2026-11-20_plan_r-under-a-card.md; settings, thresholds and rules in results/predictions.json, which the
run refuses unless committed unchanged. The measurement is commute.stand_in on E226's variables (commute.kstar_vpd).

R_A per item, at E226's circuit K* (polished set of each search): the largest share of K* whose variables can take M's
mechanism, each times its least-squares output gain (positive gains only), all at once, with KL(F || F_S) at the read
positions, mean over D, at or under (1 - alpha) KL(F || M): the stand-in keeps alpha of F's change from M (lead, phase
D). An item whose (1 - alpha) KL(F || M) is at or under the measured floor (stage 0's, or the item's own revert-all
KL) is not read; the do-nothing reads 1 by definition. R_A_abs, with E226's absolute allowance (1 - alpha)
KL(F || F_all_ablated) as the bar, is a side reading on the built items and the random controls, not filed. Where S is stood in is the
setting stand_in_forward: "full" (F's full forward, nothing ablated) or "circuit" (inside F_K*, the rest mean-ablated).
Unchanged variables stand in exactly; grafts and variables with a gain <= 0 never can. The changed members of K* are
searched with E226's search code (`kstar_vpd.search`) for the smallest set that must keep F's mechanism; R_A = 1 -
(that set + K*'s members that cannot stand in) / |K*|, a lower bound when the search is capped.

Groups, each its own submission and each refusing to start before the one above it has been read (stage 0, the
do-nothing, runs first and alone in every job and stops on a miss):
    control  two random updates (E241's `random_dw`, a Gaussian per matrix): at new_thin's update size on x2's windows;
             and E241's "kl" arm, scaled so KL(F || M) on new_few's windows is new_few's at E240's matched size
    built    new_thin, new_few, x2, x10, moved, rebuilt at their own size, E226's saved K* on all of D and its halves
    pairs    E187's nine pairs at matched size (smaller side: E226's saved K*; larger side times E187's c, E232's items)
    e62      E62's copy/lookup pair at its own size (E226's saved K*)
    keys     E237's five 96-key tasks x 3 seeds at E237's common size (E237's saved VPD K*)
An item without a saved K* (control, the scaled states) gets one here from E226's read_item, unchanged. Per item
results/items/<item>.json when it ends (a rerun skips it), every number also in the log; results/large/<item>.npz the
gains and, per search, which variables keep F's mechanism. Scoring: score.py (no GPU).

    sbatch --comment='E242 control' cluster/kstar_vpd_wording_a.sbatch control --group control
    python experiments/kstar_vpd_wording_a/run.py --group pairs --shard 0 12 --tag pairs0
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
from commute import old_tasks as OT
from commute import rule_retention as R
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

E226: Any = load_experiment("kstar_vpd", "run")  # World, read_item, guards, load_config, KP, KP_TH, log/dump/stop
SMR: Any = load_experiment("kstar_size_matched", "run")  # E232: World (scaled items), load_config, E221
SMS: Any = load_experiment("kstar_size_matched", "score")  # matched: the nine pairs with the larger side scaled
KP = E226.KP
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
SAVED = {"e226": EXPERIMENTS / "kstar_vpd" / "results", "e237": EXPERIMENTS / "key_grid" / "results", "e240": EXPERIMENTS / "kstar_vpd_norm_matched" / "results"}
log, dump, stop, fname = E226.log, E226.dump, E226.stop, KP.fname
GROUPS = ("control", "built", "pairs", "e62", "keys")


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    stand_in_forward: str  # "full" or "circuit"
    eval_cap: int  # evaluations per stand-in search
    reranks: int
    chunk_div: int
    gain_batch: int  # windows per batch of the gain fit
    reuse: tuple[str, ...]
    new: tuple[str, ...]
    random_seed: int
    random_windows: str  # the built item whose windows the random update is read on
    size_target: str  # the built item whose update size the random update has
    random_kl_seed: int  # the KL-matched random update (E241's "kl" arm): its direction, at size_target's size, then scaled
    kl_target_item: str  # the built item whose KL(F || M) at E240's matched size the scaled update matches, on its windows
    scale_cap: float  # E241's matcher (commute.perturb.match)
    kl_rel_tol: float
    e62: tuple[str, ...]
    side_tols: tuple[float, ...]  # descriptive tolerances of the relative bar beside 1 - alpha (no filed rule)

    def __post_init__(self) -> None:
        assert self.stand_in_forward in ("full", "circuit") and self.eval_cap > 0 and self.reranks >= 0 and self.chunk_div >= 1
        assert set(self.reuse) == {"x2", "x10", "moved", "rebuilt"} and set(self.new) == {"new_few", "new_thin"}
        assert {self.random_windows, self.size_target, self.kl_target_item} <= set(self.built) and self.random_kl_seed != self.random_seed

    @property
    def built(self) -> tuple[str, ...]:
        return (*self.new, *self.reuse)

    @property
    def random(self) -> str:
        return f"random_size_{self.random_seed}"

    @property
    def random_kl(self) -> str:
        return f"random_kl_{self.random_kl_seed}"


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def pair_sides() -> dict[str, tuple[list[str], list[str]]]:
    """E187's nine pairs, (reuse states, new states), the larger side's states as E232's scaled items."""
    return SMS.matched({p: v for p, v in KP.pairs(KP.load_config()[0]).items() if p.startswith("e187_")})


def key_items() -> list[str]:
    kg = load_experiment("key_grid", "run")
    point_e226_here()  # importing E237's run points E226's writes at E237's results
    return [f"{s}_scaled" for s in kg.load_config()[0].states()]


def group_items(cfg: Config, g: str) -> list[str]:
    every = {"control": lambda: [cfg.random, cfg.random_kl],"built": lambda: list(cfg.built), "e62": lambda: list(cfg.e62), "keys": key_items,
             "pairs": lambda: list(dict.fromkeys(i for r, n in pair_sides().values() for i in (*r, *n)))}  # fmt: skip
    return every[g]()


def point_e226_here() -> None:
    E226.RESULTS, E226.EXPERIMENT = RESULTS, EXPERIMENT  # E226's read_item (K* of items with none saved) writes here


class World(E226.World):
    """E226's world (the built items, the decomposition, the split's bases); E232's for the pairs, E237's for its keys."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg, self.sm, self.kg = cfg, None, None
        self.kl_x: dict[str, Any] | None = None  # the KL-matched random's scale, matched once per job
        self.floor: float | None = None  # the stood-in path's fp32 KL floor, measured in this job's stage 0

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        if item in E226.E214_ITEMS:
            return super().make(item)
        if item == self.cfg.random:
            return self.random()
        if item == self.cfg.random_kl:
            return self.random_kl()
        if item.startswith("e58/"):  # E237's key items, E237's World (its common size c per task)
            if self.kg is None:
                kg = load_experiment("key_grid", "run")
                point_e226_here()
                self.kg = kg.World(kg.load_config()[0])
                self.kg.c = json.loads((SAVED["e237"] / "stage0.json").read_text())["c"]
            F, toks, at, rec = self.kg.make(item)
            return F, self.kg.M, toks, at, rec
        if self.sm is None:
            self.sm = SMR.World(SMR.load_config()[0])
            assert all(torch.equal(self.sm.M.net.get_submodule(k).weight, self.kl.M.net.get_submodule(k).weight) for k in self.keys)
        F, toks, at, rec = self.sm.make(item) if item in self.sm.items else SMR.E221.World.make(self.sm, item)
        return F, self.sm.M, toks, at, rec

    def direction(self, seed: int) -> tuple[dict[str, Tensor], float]:
        """E241's construction: a Gaussian per decomposed matrix at one relative size, sum |dW|^2 = size_target's."""
        target2 = sum(c["dW2"] for c in json.loads((SAVED["e226"] / "items" / f"{self.cfg.size_target}.json").read_text())["checks"].values())
        return load_experiment("gating_vpd", "run").random_dw(self.kl.net.W0, target2, seed), target2

    def random(self) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        dw, target2 = self.direction(self.cfg.random_seed)
        _, M, toks, at, _ = super().make(self.cfg.random_windows)
        rec = {"target2": target2, "size_ratio": sum(float(v.double().pow(2).sum()) for v in dw.values()) / target2}
        return R.Model(OT.with_delta(self.kl.net, dw).net), M, toks, at, rec

    def random_kl(self) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        """E241's "kl" arm: the direction times x, x matched (commute.perturb.match) so KL(F || M) on kl_target_item's
        windows equals that item's KL(F || M) at E240's matched size (its saved search file)."""
        from commute import perturb as PT

        g241 = load_experiment("gating_vpd", "run")
        dw, _ = self.direction(self.cfg.random_kl_seed)
        _, M, toks, at, _ = super().make(self.cfg.kl_target_item)
        target = json.loads((SAVED["e240"] / "search" / f"{self.cfg.kl_target_item}.json").read_text())["kl_fm"]
        if self.kl_x is None:
            m = PT.match(lambda x: g241.kl_at(self.kl.net, M, PT.scaled(dw, x), toks, at, E226.load_config()[0].batch), target, self.cfg.scale_cap, self.cfg.kl_rel_tol)
            self.kl_x = {"x": m["x"], "kl_matched": m["f"], "match": m["status"], "match_evals": len(m["grid"]) + len(m["trace"])}
            log(f"{self.cfg.random_kl}: matched x {m['x']:.6g} to KL {m['f']:.6g} (target {target:.6g}), {m['status']}")
        rec = {"kl_target": target, "kl_ratio": self.kl_x["kl_matched"] / target} | self.kl_x
        return R.Model(OT.with_delta(self.kl.net, PT.scaled(dw, self.kl_x["x"])).net), M, toks, at, rec


def k_source(w: World, item: str) -> tuple[dict[str, Any], Path | None]:
    """The item's K* reading and per-variable arrays: E226's or E237's saved ones, else made here by E226's read_item."""
    e226, e237 = SAVED["e226"], SAVED["e237"]
    for js, npz in ((e226 / "stage0_pod0" / "do_nothing.json", e226 / "large" / "stage0_pod0_do_nothing.npz"),
                    (e226 / "items" / f"{fname(item)}.json", e226 / "large" / f"items_{fname(item)}.npz"),
                    (e237 / "vpd" / f"{fname(item)}.json", e237 / "large" / f"vpd_{fname(item)}.npz")):  # fmt: skip
        if (item == "do_nothing") == (js.parent.name == "stage0_pod0") and js.is_file():
            return json.loads(js.read_text()), npz
    point_e226_here()
    c226, th226 = E226.load_config()
    rec = E226.read_item(w, item, c226, th226, RESULTS / "kstar")  # results/kstar/<item>.json, results/large/kstar_<item>.npz
    return rec, (RESULTS / "large" / f"kstar_{fname(item)}.npz") if rec.get("reading") == "read" else None


# ---- the stand-in search ---------------------------------------------------------------------------------------------


@dataclass
class Read:
    si: SI.StandIn
    forward: Any  # x -> logits at the read positions, F's full forward or inside the circuit
    toks: list[Tensor]
    lF: list[Tensor]  # F's own logits at the read positions

    def kl(self, t: Tensor) -> float:
        return SI.kl_mean(self.si, self.forward, self.toks, self.lF, t)


def stand_in_search(rd: Read, feat: KV.Feat, g: Tensor, K: list[int], bar: float, cfg: Config, label: str) -> tuple[dict[str, Any], np.ndarray]:
    """R_A at K: the smallest set of K's standable changed members that must keep F's mechanism, every other standable
    member stood in; grafts and gains <= 0 keep F's. Returns the record and which variables keep F's mechanism."""
    t0, dev, ok = time.time(), rd.lF[0].device, SI.standable(g)
    fixed = [u for u in K if u in feat.lacked or (u in feat.changed and not bool(ok[u]))]
    cand = torch.tensor(sorted(u for u in K if u in feat.changed and u not in feat.lacked and bool(ok[u])), dtype=torch.long)

    def weights(stood: Tensor) -> Tensor:
        t = torch.zeros(feat.n, device=dev)
        t[cand.to(dev)] = stood.to(dev, torch.float32)
        return t

    kl_all = rd.kl(weights(torch.ones(len(cand)))) if len(cand) else 0.0
    kept, evals, capped, prefix = [], int(len(cand) > 0), False, None
    if kl_all > bar:
        grad = lambda c: SI.attribution(rd.si, rd.forward, rd.toks, rd.lF, weights(1 - c))[cand]
        f = KV.search(lambda keep: rd.kl(weights((~keep).float())), grad, len(cand), bar, cfg.reranks, cfg.eval_cap, cfg.chunk_div, log)
        kept, evals, capped, prefix = cand[f.polished].tolist(), f.evals + 1, f.capped, len(f.prefix)
    keep = np.zeros(feat.n, dtype=bool)
    keep[kept + fixed] = True
    stood = ~torch.isin(cand, torch.tensor(kept, dtype=torch.long))
    kl_found = rd.kl(weights(stood.float())) if len(cand) else 0.0
    rec = {"size_K": len(K), "standable_changed_in_K": len(cand), "grafts_in_K": sum(u in feat.lacked for u in K),
           "nonpositive_gain_in_K": len(fixed) - sum(u in feat.lacked for u in K), "kept_F": len(kept), "kept_F_prefix": prefix,
           "R_A": 1 - (len(kept) + len(fixed)) / len(K), "ceiling_all_standable": 1 - len(fixed) / len(K),
           "floor_1_minus_f": 1 - (len(cand) + len(fixed)) / len(K), "bar": bar, "kl_all_stood_in": kl_all, "kl_at_found": kl_found,
           "evals": evals, "capped": capped, "seconds": time.time() - t0}  # fmt: skip
    log(f"  {label}: R_A {rec['R_A']:.6f} (floor {rec['floor_1_minus_f']:.6f}, ceiling {rec['ceiling_all_standable']:.6f}); |K*| {len(K)}, "
        f"standable changed {len(cand)}, keep F's {len(kept)}, cannot stand in {len(fixed)} (gain <= 0 {rec['nonpositive_gain_in_K']}); "
        f"KL all stood in {kl_all:.6g}, at the found set {kl_found:.6g}, bar {bar:.6g}; {evals} evaluations{' (cap hit)' if capped else ''}, "
        f"{rec['seconds']:.0f} s")  # fmt: skip
    return rec, keep


# ---- one item -------------------------------------------------------------------------------------------------------


def logits_at(F: R.Model, at: slice, x: Tensor) -> Tensor:
    """F's full forward (nothing ablated) at the read positions."""
    return F.logits(x)[:, at]


@torch.no_grad()
def pieces_floor(F: R.Model, feat: KV.Feat, batches: list[Tensor], lF: list[Tensor], at: slice) -> float:
    """KL(F || F with every decomposed matrix's output recomputed from its pieces, (y V) U + y Delta), mean over the
    windows: for F = M mathematically 0, so the fp32 rounding floor of writing M's mechanisms through the stood-in path."""
    hs = [F.net.get_submodule(s.name).register_forward_hook(lambda _m, inp, _o, s=s: (inp[0] @ s.V) @ s.U + inp[0] @ s.rm) for s in feat.sites]
    try:
        return float(torch.cat([R.kl(lf, F.logits(x)[:, at]) for x, lf in zip(batches, lF, strict=True)]).mean())
    finally:
        for h in hs:
            h.remove()


def controls(si: SI.StandIn, F: R.Model, M: R.Model, feat: KV.Feat, g: Tensor, batches: list[Tensor], lF: list[Tensor], at: slice,
             th: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:  # fmt: skip
    """In F's full forward: every unchanged variable's gain exactly 1; all of them stood in at once leave F's logits
    exactly (KL 0); every changed variable stood in with gain 1 gives M's logits (items without grafts), within revert_tol."""
    full, dev = partial(logits_at, F, at), lF[0].device
    unchanged = torch.tensor([u for u in range(feat.n) if u not in feat.changed], dtype=torch.long)
    t = torch.zeros(feat.n, device=dev)
    t[unchanged.to(dev)] = 1
    t0 = time.time()
    out: dict[str, Any] = {"gain_dev_unchanged": float((g[unchanged] - 1).abs().max()), "kl_unchanged_stood_in": SI.kl_mean(si, full, batches, lF, t)}
    out |= {"seconds_per_eval": time.time() - t0, "revert_all_err": None, "revert_all_kl": None}
    changed = torch.tensor(sorted(feat.changed - feat.lacked), dtype=torch.long)
    out["gain_changed_quantiles"] = torch.quantile(g[changed], torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)).tolist() if len(changed) else None
    out["nonpositive_gains"] = int((g[changed] <= 0).sum()) if len(changed) else 0
    if not feat.grafts:
        g_fit, si.g = si.g, torch.ones_like(si.g)
        t = torch.zeros(feat.n, device=dev)
        t[changed.to(dev)] = 1
        errs, kls = [], []
        with torch.no_grad():
            for x in batches:
                r, m = si.run(full, x, t), M.logits(x)[:, at]
                errs.append(float(((r - m).abs().amax((1, 2)) / m.abs().amax((1, 2))).max()))
                kls.append(R.kl(m, r))
        # KL(M || every change stood in at gain 1): mathematically 0, so the fp32 floor of the stood-in path on this item
        si.g, out["revert_all_err"], out["revert_all_kl"] = g_fit, max(errs), float(torch.cat(kls).mean())
    if not feat.changed:  # the do-nothing (F = M): every matrix's output rebuilt from its pieces, the same floor on M
        out["pieces_floor"] = pieces_floor(F, feat, batches, lF, at)
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
    t0, (c226, th226) = time.time(), E226.load_config()
    r226, npz = k_source(w, item)
    base: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "R_old": None}
    if npz is None:
        dump(path, base | {"reading": "no behaviour (E226's read_item)"})
        return base | {"reading": "no behaviour"}
    arrays = np.load(npz)
    F, M, toks, at, rec = w.make(item)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    ctx = KV.prepare(F, M, feat, toks, at, c226.batch)
    base |= {"n_windows": len(toks), "n_vars": feat.n, "n_changed": len(feat.changed), "n_grafts": len(feat.grafts), "kl_none": float(ctx.kl_none.mean()),
             "kl_none_saved": r226["kl_none"], "kl_fm": float(ctx.kl_fm.mean()), "gates": KV.gates(ctx, feat.n),
             "R_old": r226["searches"]["main"]["polished"]["R_read"]} | rec  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    miss = E226.guards(item, feat, KV.undecomposed_differences(M, F, w.keys), base["gates"], th226) + KP.load_checks(item, base, E226.KP_TH)
    miss += [f"KL(F || F_empty) {base['kl_none']:.9g}, saved {r226['kl_none']:.9g}"] if not abs(base["kl_none"] / r226["kl_none"] - 1) <= th["kl_repro_rel"] else []
    miss += [f"size ratio {rec['size_ratio']:.9g}"] if "size_ratio" in rec and not abs(rec["size_ratio"] - 1) <= th["size_rel"] else []
    miss += [f"KL match {rec['match']}, ratio {rec['kl_ratio']:.6g}"] if "kl_ratio" in rec and not (rec["match"] == "ok" and abs(rec["kl_ratio"] - 1) <= cfg.kl_rel_tol) else []
    miss += [f"KL(F || M) {base['kl_fm']:.9g} off the matched {rec['kl_matched']:.9g}"] if "kl_matched" in rec and not abs(base["kl_fm"] / rec["kl_matched"] - 1) <= th["kl_repro_rel"] else []
    batches = list(toks.split(c226.batch))
    g = SI.gains(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(cfg.gain_batch)))
    si = SI.StandIn(F.net, SI.vpd_sites(feat), g, toks.device)
    with torch.no_grad():
        lF = [F.logits(x)[:, at] for x in batches]
    base["controls"], m2 = controls(si, F, M, feat, g, batches, lF, at, th)
    log(f"{item}: {len(toks)} windows, {feat.n} variables, {len(feat.changed)} changed, {len(feat.grafts)} grafts; KL(F || F_empty) {base['kl_none']:.6g} "
        f"(saved {r226['kl_none']:.6g}), KL(F || M) {base['kl_fm']:.6g}; old R {base['R_old']:.6f}; controls {base['controls']}")  # fmt: skip
    if miss + m2:
        stop(out / f"{fname(item)}.stopped.json", base, f"{item}: " + "; ".join(miss + m2))
    variants = {"main": torch.arange(len(toks))}
    if "half1_in_K_polished" in arrays.files:  # the six built items: E226's two halves of D, same permutation
        perm = torch.randperm(len(toks), generator=torch.Generator().manual_seed(c226.split_seed))
        variants |= {"half1": perm[: len(toks) // 2], "half2": perm[len(toks) // 2 :]}
    searches: dict[str, dict[str, Any]] = {}
    keeps: dict[str, Any] = {}  # per search, which variables keep F's mechanism (bool arrays)
    full = partial(logits_at, F, at)
    floor = max(w.floor or 0.0, base["controls"]["revert_all_kl"] or 0.0)  # the job's stage 0 floor and this item's own
    side = item in (*cfg.built, cfg.random, cfg.random_kl)  # R_A_abs, the absolute allowance, beside (not filed)
    for k, idx in variants.items():
        sub = ctx if k == "main" else ctx.subset(idx, c226.batch)
        K = np.flatnonzero(arrays[f"{k}_in_K_polished"]).tolist()
        bar_abs, cK = (1 - c226.alpha) * float(sub.kl_none.mean()), torch.zeros(feat.n, dtype=torch.bool)
        cK[K] = True
        kl_K, kl_fm = KV.kl_mean(sub, cK), float(sub.kl_fm.mean())
        if not kl_K <= bar_abs * (1 + th["kl_repro_rel"]):
            stop(out / f"{fname(item)}.stopped.json", base, f"{item} {k}: the saved K* gives KL {kl_K:.9g} over its bar {bar_abs:.9g}")
        bar = (1 - c226.alpha) * kl_fm  # the relative bar: the stand-in keeps (alpha) of F's change from M
        info = {"kl_at_K": kl_K, "kl_fm": kl_fm, "floor": floor, "bar_abs": bar_abs}
        if item != "do_nothing" and not bar > floor:
            log(f"  {item} {k}: not read, (1 - alpha) KL(F || M) {bar:.3g} at or under the measured floor {floor:.3g}")
            searches[k], keeps[k] = info | {"R_A": None, "reading": "not read: (1 - alpha) KL(F || M) at or under the floor", "bar": bar}, np.zeros(feat.n, dtype=bool)
            continue
        cKd = cK.to(toks.device, torch.float32)
        fwd = full if cfg.stand_in_forward == "full" else (lambda x, cKd=cKd, hy=ctx.hy: hy.logits(x, cKd))  # inside the circuit: the rest ablated
        rd = Read(si, fwd, sub.toks, list(torch.cat(lF)[idx.to(toks.device)].split(c226.batch)))
        searches[k], keeps[k] = stand_in_search(rd, feat, g, K, bar, cfg, f"{item} {k}")
        searches[k] |= info
        if k != "main" or item == "do_nothing":
            continue
        if side:
            a, keeps["main_abs"] = stand_in_search(rd, feat, g, K, bar_abs, cfg, f"{item} {k} (absolute allowance, side reading)")
            searches[k] |= {"R_A_abs": a["R_A"], "abs": a}
        for tol in cfg.side_tols:  # descriptive: the relative bar at other tolerances, the same floor rule
            if not tol * kl_fm > floor:
                searches[k] |= {f"R_A_tol{tol:g}": None}
                continue
            a, keeps[f"main_tol{tol:g}"] = stand_in_search(rd, feat, g, K, tol * kl_fm, cfg, f"{item} {k} (tolerance {tol:g}, side reading)")
            searches[k] |= {f"R_A_tol{tol:g}": a["R_A"], f"tol{tol:g}": a}
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "large" / f"{out.name}_{fname(item)}.npz", gain=g.numpy(), changed=np.isin(np.arange(feat.n), sorted(feat.changed)),
                        **{f"{k}_keeps_F": v for k, v in keeps.items()})  # fmt: skip
    res = base | {"reading": "read", "searches": searches, "seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, res)
    m = searches["main"]
    log(f"{item}: done in {res['seconds']:.0f} s; R_A main {m['R_A']}, R_A_abs {m.get('R_A_abs')}, old R {base['R_old']:.6f}; peak GPU {res['peak_gpu_gb']:.2f} GB")
    si.close()
    ctx.hy.close()
    del ctx, si, F
    torch.cuda.empty_cache()
    return res


# ---- stage 0, main --------------------------------------------------------------------------------------------------


def stage0(w: World, cfg: Config, th: dict[str, Any], tag: str, n_searches: int) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        w.floor = json.loads(path.read_text())["kl_floor"]
        log(f"{path.name}: done before; floor {w.floor:.3g}")
        return
    it = read_item(w, "do_nothing", cfg, th, RESULTS / f"stage0_{tag}")
    m, c = it["searches"]["main"], it["controls"]
    hours = c["seconds_per_eval"] * cfg.eval_cap * n_searches / 3600
    w.floor = max(c["pieces_floor"], c["revert_all_kl"] or 0.0)
    out = {"experiment": EXPERIMENT, "R_A": m["R_A"], "n_changed": it["n_changed"], "evals": m["evals"], "kl_unchanged_stood_in": c["kl_unchanged_stood_in"],
           "gain_dev_unchanged": c["gain_dev_unchanged"], "seconds_per_eval": c["seconds_per_eval"], "projected_hours_max": hours,
           "pieces_floor": c["pieces_floor"], "kl_floor": w.floor}  # fmt: skip
    miss = [f"do-nothing: {out}"] if not (it["n_changed"] == 0 and m["R_A"] == 1 and m["evals"] == 0) else []
    miss += [f"floor {w.floor:.3g} over floor_max {th['floor_max']:.3g}"] if not w.floor <= th["floor_max"] else []
    miss += [f"projected up to {hours:.1f} h over {th['shard_hours_max']} h"] if hours > th["shard_hours_max"] else []
    log(f"stage 0: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def unread(items: list[str]) -> list[str]:
    def ok(i: str) -> bool:
        p = RESULTS / "items" / f"{fname(i)}.json"
        return p.is_file() and json.loads(p.read_text()).get("reading") == "read"

    return [i for i in items if not ok(i)]


def main(group: str, chosen: list[str] | None, shard: tuple[int, int], tag: str) -> int:
    cfg, th = load_config()
    every = group_items(cfg, group)
    todo = chosen or every[shard[0] :: shard[1]]
    assert set(todo) <= set(every), f"not in group {group}: {sorted(set(todo) - set(every))}"
    before = GROUPS[: GROUPS.index(group)]
    missing = unread([i for g in before[:2] for i in group_items(cfg, g)])  # control before built; both before the rest
    if missing:
        log(f"STOP: group {group} waits for {missing}")
        return 1
    point_e226_here()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226
    log(f"{EXPERIMENT} start, group {group}, tag {tag}; {len(todo)} items {todo}; settings {asdict(cfg)}; thresholds {th}")
    w = World(cfg)
    w.bases = KV.bases(w.uv, E226.load_config()[0].split_tol)
    side = (*cfg.built, cfg.random, cfg.random_kl)  # + the absolute side reading; + the halves for the built items
    stage0(w, cfg, th, tag, sum(1 + len(cfg.side_tols) + (i in side) + 2 * (i in cfg.built) for i in todo))
    for k, item in enumerate(todo):
        read_item(w, item, cfg, th, RESULTS / "items")
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=GROUPS, required=True)
    ap.add_argument("--items", nargs="+", default=None, help="items of the group by name (default: this shard of the group)")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.group, args.items, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

"""E___ (cell P1.4, run 2 of 3; reopened 2026-11-23 by whip-lead under the user's standing authority): ordinary fine-tunes
of all weights of the 67M trained to imitate two built edits whose label is known by construction, E214's x2 (pure
strengthening: reuse) and new_few (new), 3 seeds each with E139 amendment 2's recipe, each beside a matched random, read
with E_b over the needed set K at E277's neuron featurization (scored) and E268's VPD featurization (seed 0,
descriptive). An imitator is read only when its leftover, TV(fine-tune, target) / TV(fine-tune, M) as E139 measured it,
is at most E139's measured 0.1924. Plan reports/plans/2026-11-24_plan_p1-4-run2-trained-reuse-control.md; card card.md;
settings and rules in results/predictions.json, which the run refuses unless committed unchanged.

Items: 'x2', 'new_few' (the built edits, read again by R0c), 'ft/<target>_s<seed>' (the imitator) and
'ft/<target>_s<seed>_rand' (needed_neurons.matched_random at the imitator's per-tensor norms). Every item is read on its
target's D at its read positions, as E277 and E268 read the built edits (x2: 75 copy24 held-out windows, every position;
new_few: 512 cue windows, the last position).

Stages; each job runs its checks first and stops on a miss; a rerun skips what is written:
    train    R0b, R0d, then each (target, seed) of its shard: 3000 steps with R1's early stops, leftover at each save
                                                              -> results/train/<t>_s<k>.json, results/large/dw_<t>_s<k>.pt
    check    E277's stage 0 (R0a), then x2 and new_few read again at neurons (R0c)  -> results/stage0_repeat_neurons.json
    neurons  needs check done: E277's stage 0, then its shard of the 12 items       -> results/items/, results/large/items_*
    vpd      E290's VPD stage 0 (R0a), R0c at VPD, then the 4 seed-0 items          -> results/vpd/, results/large/vpd_*
    summary  CPU: R2-R4                                                             -> results/summary.json

    python experiments/trained_reuse_control/run.py --stage train --shard 0 3 --tag t0
    python experiments/trained_reuse_control/run.py --stage check --tag c0
    python experiments/trained_reuse_control/run.py --stage neurons --shard 0 3 --tag n0
    python experiments/trained_reuse_control/run.py --stage vpd --tag v0
    python experiments/trained_reuse_control/run.py --stage summary
    python experiments/trained_reuse_control/run.py --stage train --dry OUT   (CPU dry run: the "dry" settings, under OUT)
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import old_tasks as OT
from commute import rule_retention as R
from commute.hidden_maps import kl, tv
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    manifest_only,
    require_filed,
)
from commute.qwen_sft import lr_at

IT: Any = load_experiment("eapig_overlap", "items")  # P2.69: Items (E242's World: E214's x2, new_few), edits_check, KL
NNR: Any = load_experiment("needed_neurons", "run")  # E277: stage0, read_item, matched_random, remainder_columns, E229
IA: Any = load_experiment("ia3_control", "run")  # E290: vpd_stage0, read_vpd (E268's VPD read with one write pass), KN
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
RESULTS = HERE / "results"  # --dry points it elsewhere (`point`)
log, dump, stop, fname = NNR.log, NNR.dump, NNR.stop, NNR.fname


def point(results: Path) -> None:
    """Every writer here and in the imported stages (E277's, E290's) writes under `results`."""
    global RESULTS
    RESULTS = NNR.RESULTS = IA.RESULTS = results
    NNR.EXPERIMENT = IA.EXPERIMENT = EXPERIMENT
    for d in ("items", "large", "train", "vpd"):
        (results / d).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings" (the "dry" block overrides some for --dry)
    targets: tuple[str, ...]
    seeds: tuple[int, ...]
    e195_C: tuple[int, ...]
    steps: int
    lr: float
    warmup: int
    schedule: str
    clip: float
    batch: dict[str, int]
    micro: int
    fit_per_cue: int
    saves: tuple[int, ...]
    early_steps: tuple[int, ...]
    overshoot_step: int
    kl_m_band: dict[str, tuple[float, float]]
    fall_step: int
    fall_ratio: float
    read_batch: int
    floor_gap: float
    leftover_max: float
    vpd_seeds: tuple[int, ...]
    random_seed: int
    prior_neurons: dict[str, float]
    prior_vpd: dict[str, float]
    repeat_tol_neurons: float
    repeat_tol_vpd: float
    too_small: int
    rival_gap: float
    random_max: float
    min_seeds: int
    wall_s: float

    def __post_init__(self) -> None:
        assert set(self.targets) <= {"x2", "new_few"} and set(self.batch) == {"x2", "new_few"} and set(self.vpd_seeds) <= set(self.seeds)
        assert self.warmup * 10 == self.steps and list(self.saves) == sorted(self.saves) and self.saves[-1] == self.steps
        need = {1, self.overshoot_step, self.fall_step, *map(int, self.kl_m_band), *self.saves}
        assert need <= set(self.early_steps) and self.early_steps[-1] == self.steps, "every checked step is an early step"
        assert set(self.prior_neurons) == set(self.prior_vpd) == {"x2", "new_few"} and 0 < self.leftover_max < 1


def load_config(dry: bool = False) -> Config:
    P = json.loads(PRED.read_text())
    s: dict[str, Any] = P["settings"] | (P["dry"] if dry else {})
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in s.items()}
    s["kl_m_band"] = {k: tuple(v) for k, v in s["kl_m_band"].items()}
    return config_from_settings(Config, s)


def pairs(cfg: Config) -> list[tuple[str, int]]:
    """new_few first: its trainings are the long ones, so a shard of 3 gets one of each target."""
    return [(t, s) for t in sorted(cfg.targets, key=lambda t: t != "new_few") for s in cfg.seeds]


def items(cfg: Config, seeds: tuple[int, ...] | None = None) -> list[str]:
    return [f"ft/{t}_s{s}{r}" for t in cfg.targets for s in seeds or cfg.seeds for r in ("", "_rand")]


def parse(item: str) -> tuple[str, int]:
    t, s = item.removeprefix("ft/").removesuffix("_rand").rsplit("_s", 1)
    return t, int(s)


def train_file(t: str, s: int) -> Path:
    return RESULTS / "train" / f"{t}_s{s}.json"


def dw_file(t: str, s: int) -> Path:
    return RESULTS / "large" / f"dw_{t}_s{s}.pt"


def done(p: Path) -> dict[str, Any] | None:
    return r if p.is_file() and (r := json.loads(p.read_text())).get("done") else None


# ---- the world --------------------------------------------------------------------------------------------------------


class World:
    """E242's World as P2.69's Items builds it (E214's edits rebuilt and checked against the committed record), and this
    run's imitators and randoms. `name`, `control`, `make`, `uv`, `delta`, `bases`, `cfg.too_small` are what E277's and
    E290's stages read."""

    def __init__(self, cfg: Config, name: str) -> None:
        self.cfg, self.name, self.control = cfg, name, "x2"
        self.it = IT.Items(SimpleNamespace(e195_C=cfg.e195_C), RESULTS)  # E214's edits record goes to RESULTS/edits.json
        self.w = self.it.w
        self.net, self.M = self.w.kl.net, self.w.kl.M
        self.uv, self.delta, self.bases = self.w.uv, self.w.delta, self.w.bases

    def target(self, t: str) -> tuple[R.Model, Tensor, slice]:
        F, _, toks, at, _ = self.w.make(t)
        if t == "new_few":  # R0d
            rec = self.it.edits_check()
            if not all(rec["discrete_equal"].values()):
                stop(RESULTS / "stage0_edits.json", rec, f"R0d: E214's rebuilt edits differ from the committed record: {rec['discrete_equal']}")
        return F, toks, at

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        if item == "do_nothing":
            return self.w.make(item)
        if item in ("x2", "new_few"):
            F, toks, at = self.target(item)
            return F, self.M, toks, at, {}
        t, s = parse(item)
        if done(train_file(t, s)) is None:
            stop(RESULTS / "items" / f"{fname(item)}.stopped.json", {"item": item}, f"{item}: not trained; run --stage train first")
        _, toks, at = self.target(t)
        F = R.Model(OT.with_delta(self.net, torch.load(dw_file(t, s), weights_only=True)).net)
        rec: dict[str, Any] = {"dw": str(dw_file(t, s).relative_to(RESULTS.parent))}
        if item.endswith("_rand"):
            dr, rec = NNR.matched_random(F.net, self.M.net, self.cfg.random_seed + items(self.cfg).index(item))
            F = R.Model(OT.with_delta(self.net, dr).net)
        return F, self.M, toks, at, rec


def fit_windows(w: World, t: str) -> Tensor:
    """x2: E195's 301 copy24 training windows. new_few: E214's cue builder with the edit's seed and skip list (so A, A', y
    are the edit's) at fit_per_cue windows per cue, any window equal to one of D's dropped."""
    if t == "x2":
        return torch.load(IT.KL.P.COPY24, weights_only=False)["train"].tokens
    kl_w, cfg = w.w.kl, w.cfg
    D = kl_w.cues  # built with the edit (its skip list final)
    me = SimpleNamespace(net=kl_w.net, skip=kl_w.skip, cfg=SimpleNamespace(**asdict(kl_w.cfg) | {"windows_per_cue": cfg.fit_per_cue}))
    c = IT.KL.World.cues.func(me)
    assert (c.A, c.A2, c.y) == (D.A, D.A2, D.y), "the fit windows' cues are not the edit's"
    seen = {tuple(r) for r in D.toks.tolist()}
    keep = torch.tensor([tuple(r) not in seen for r in c.toks.tolist()])
    log(f"new_few fit windows: {int(keep.sum())} of {len(keep)} kept (the rest equal one of D's)")
    return c.toks[keep]


# ---- train --------------------------------------------------------------------------------------------------------------


def distances(M: R.Model, F: R.Model, ft: R.Model, D: Tensor, at: slice, batch: int) -> dict[str, Any]:
    """E139's `distances` (reuse_variable_finetune/run.py, line 239) at the item's read positions: means over D's windows
    and read positions; leftover = TV(ft, F) / TV(ft, M), None when the gap is 0."""
    s: dict[str, float] = dict.fromkeys(("gap", "tv_to_F", "kl_to_F", "kl_from_M", "target_gap", "kl_F_M", "kl_M_F"), 0.0)
    n = 0
    with torch.no_grad():
        for t in D.split(batch):
            lm, lf, lft = (N.logits(t)[:, at] for N in (M, F, ft))
            for k, v in (("gap", tv(lft, lm)), ("tv_to_F", tv(lft, lf)), ("kl_to_F", kl(lf, lft)), ("kl_from_M", kl(lm, lft)),
                         ("target_gap", tv(lf, lm)), ("kl_F_M", kl(lf, lm)), ("kl_M_F", kl(lm, lf))):  # fmt: skip
                s[k] += float(v.double().sum())
            n += lm.shape[0] * lm.shape[1]
    out: dict[str, Any] = {k: v / n for k, v in s.items()}
    out["leftover"] = out["tv_to_F"] / out["gap"] if out["gap"] else None
    return out


def early_miss(cfg: Config, step: int, d: dict[str, Any], d0: dict[str, Any]) -> list[str]:
    """R1 at one recorded step (d0: the distances of the untrained copy, KL(F || M) and KL(M || F) among them)."""
    miss = [f"step 1 (rate 0): KL(M || ft) {d['kl_from_M']!r}, not 0.0"] if step == 1 and d["kl_from_M"] != 0.0 else []
    if (band := cfg.kl_m_band.get(str(step))) and not band[0] <= d["kl_from_M"] <= band[1]:
        miss.append(f"step {step}: KL(M || ft) {d['kl_from_M']:.3e} outside {band}")
    if step == cfg.overshoot_step and not (d["kl_to_F"] < d0["kl_F_M"] and d["kl_from_M"] <= d0["kl_M_F"]):
        miss.append(f"overshoot at step {step}: KL(F || ft) {d['kl_to_F']:.4g} vs KL(F || M) {d0['kl_F_M']:.4g}, KL(M || ft) {d['kl_from_M']:.4g} vs KL(M || F) {d0['kl_M_F']:.4g}")
    if step == cfg.fall_step and not d["kl_to_F"] <= cfg.fall_ratio * d0["kl_F_M"]:
        miss.append(f"step {step}: KL(F || ft) {d['kl_to_F']:.4g} over {cfg.fall_ratio} x KL(F || M) {d0['kl_F_M']:.4g}")
    return miss


def step_loss(F: R.Model, ft: R.Model, toks: Tensor, at: slice, micro: int) -> float:
    """KL(F || ft) at the read positions, the mean over the batch's windows and positions, its gradient accumulated over
    micro-batches of `micro` windows (memory only: the step is the whole batch's)."""
    total, n = 0.0, toks.shape[0] * len(range(toks.shape[1])[at])
    for t in toks.split(micro):
        with torch.no_grad():
            target = F.logits(t)[:, at]
        loss = kl(target, ft.logits(t)[:, at]).sum() / n
        loss.backward()
        total += float(loss.detach())
    return total


def delta_of(ft: Any, M: Any) -> dict[str, Tensor]:
    mp = dict(M.named_parameters())
    return {k: (p.detach() - mp[k].detach()).float().cpu() for k, p in ft.named_parameters()}


def untrained(w: World, t: str, F: R.Model, D: Tensor, at: slice) -> tuple[Any, dict[str, Any]]:
    """R0b: a copy of M before any step: delta 0, TV and KL to M exactly 0 on D, and the target's gap over the floor."""
    ft = OT.clone_net(w.net)
    d0 = distances(w.M, F, R.Model(ft.net), D, at, w.cfg.read_batch)
    nz = sum(int(v.count_nonzero()) for v in delta_of(ft.net, w.M.net).values())
    miss = [f"untrained copy: {nz} nonzero delta entries"] if nz else []
    miss += [f"untrained copy: {k} {d0[k]!r}, not 0.0" for k in ("gap", "kl_from_M") if d0[k] != 0.0]
    miss += [f"{t}: target gap TV(F, M) {d0['target_gap']:.4g} under the floor {w.cfg.floor_gap}"] if d0["target_gap"] < w.cfg.floor_gap else []
    log(f"R0b {t}: {json.dumps(d0)}; {'holds' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(RESULTS / "train" / f"stage0_{t}.json", d0, "R0b: " + "; ".join(miss))
    return ft, d0


def train_one(w: World, t: str, seed: int) -> None:
    path, cfg = train_file(t, seed), w.cfg
    if done(path):
        return log(f"{t} s{seed}: trained before, skipped")
    F, D, at = w.target(t)
    fit = fit_windows(w, t).to(D.device)
    ft, d0 = untrained(w, t, F, D, at)
    params = list(ft.net.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
    gen = torch.Generator().manual_seed(zlib.crc32(f"{t}_s{seed}".encode()))  # the seed draws the batches only
    out: dict[str, Any] = {"experiment": EXPERIMENT, "target": t, "seed": seed, "n_fit": len(fit), "batch": cfg.batch[t], "d0": d0, "early": []}
    curve, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        rate = lr_at(step, cfg.lr, cfg.warmup, cfg.steps, cfg.schedule)
        for g in opt.param_groups:
            g["lr"] = rate
        toks = fit[torch.randint(0, len(fit), (cfg.batch[t],), generator=gen).to(fit.device)]
        opt.zero_grad(set_to_none=True)
        curve.append(step_loss(F, R.Model(ft.net), toks, at, cfg.micro))
        torch.nn.utils.clip_grad_norm_(params, cfg.clip)
        opt.step()
        if step == min(30, cfg.steps):  # R1's time check, from the first steps' rate
            out["projected_s"] = (time.time() - t0) / step * cfg.steps
            log(f"  {t} s{seed}: {(time.time() - t0) / step:.3f} s per step, {out['projected_s']:.0f} s projected for {cfg.steps} steps")
            if out["projected_s"] > cfg.wall_s:
                stop(path, out | {"curve": curve}, f"R1: {out['projected_s']:.0f} s projected, over wall_s {cfg.wall_s}")
        if step in cfg.early_steps:
            d = distances(w.M, F, R.Model(ft.net), D, at, cfg.read_batch) | {"step": step, "lr": rate, "loss": curve[-1], "s": time.time() - t0}
            out["early"].append(d)
            log(f"  {t} s{seed} step {step}: loss {curve[-1]:.5f}, KL(F||ft) {d['kl_to_F']:.5f}, KL(M||ft) {d['kl_from_M']:.3e}, leftover {d['leftover']}, rate {rate:.2e}")
            if miss := early_miss(cfg, step, d, d0):
                stop(path, out | {"curve": curve}, "R1: " + "; ".join(miss))
    dw = delta_of(ft.net, w.M.net)
    torch.save(dw, dw_file(t, seed))
    keys = {f"{k}.weight" for k in w.w.keys}  # the 24 decomposed matrices
    size2 = {k: float(v.double().pow(2).sum()) for k, v in dw.items()}
    last = out["early"][-1]
    out |= {"curve": curve, "leftover": last["leftover"], "gap": last["gap"], "target_gap": last["target_gap"], "size2": sum(size2.values()),
            "size2_outside_decomposed_share": sum(v for k, v in size2.items() if k not in keys) / max(sum(size2.values()), 1e-300),
            "seconds": time.time() - t0, "done": True}  # fmt: skip
    dump(path, out)
    log(f"TRAINED {t} s{seed}: leftover {out['leftover']}, gap {out['gap']:.4g} (target {out['target_gap']:.4g}); {out['seconds']:.0f} s")
    del ft, opt
    torch.cuda.empty_cache()


def train_stage(cfg: Config, shard: tuple[int, int]) -> None:
    w = World(cfg, "train")
    todo = pairs(cfg)[shard[0] :: shard[1]]
    log(f"train: {todo}")
    for t, s in todo:
        train_one(w, t, s)


# ---- reads --------------------------------------------------------------------------------------------------------------


def repeat(where: str, prior: dict[str, float], tol: float, read: Any, path_of: Any) -> None:
    """R0c: x2 and new_few read again repeat their prior E_b over K within `tol`."""
    path = RESULTS / f"stage0_repeat_{where}.json"
    if done(path):
        return log(f"{path.name}: done before")
    out: dict[str, Any] = {"experiment": EXPERIMENT, "tol": tol}
    for item, want in prior.items():
        read(item)
        r = json.loads(path_of(item).read_text())
        out[item] = {"E_b_K": r["E_b_K"], "prior": want, "abs_diff": abs(r["E_b_K"] - want), "size_K": r["size_K"], "n_changed_K": r["n_changed_K"]}
    out["built_pair_d"] = out["x2"]["E_b_K"] - out["new_few"]["E_b_K"]  # R3's positive control
    miss = [f"{i}: E_b over K {out[i]['E_b_K']!r} vs prior {out[i]['prior']!r}" for i in prior if not out[i]["abs_diff"] <= tol]
    log(f"R0c ({where}): {json.dumps(out)}; {'holds' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "R0c: " + "; ".join(miss))
    dump(path, out | {"done": True})


def neurons_stage(cfg: Config, shard: tuple[int, int], tag: str, check: bool) -> None:
    ncfg, c229 = NNR.load_config(), NNR.load_e229()  # E277's filed scan settings, unchanged
    if not check and done(RESULTS / "stage0_repeat_neurons.json") is None:
        stop(RESULTS / f"stage0_neurons_{tag}.json", {}, "R0c at neurons has not passed: run --stage check first")
    w = World(cfg, "neurons")
    feat = NNR.E229.feat_of(c229, NNR.E229.load_pct(c229), ncfg.band)
    if manifest_only():
        return log("manifest only: world and featurization loaded, no reads")
    floor = NNR.stage0(w, ncfg, feat, tag)["identity"]["max_dev"]
    if check:
        return repeat("neurons", cfg.prior_neurons, cfg.repeat_tol_neurons, lambda i: NNR.read_item(w, i, ncfg, feat, floor),
                      lambda i: RESULTS / "items" / f"{fname(i)}.json")  # fmt: skip
    todo = items(cfg)[shard[0] :: shard[1]]
    log(f"neurons: {len(todo)} of {len(items(cfg))} items {todo}")
    for k, item in enumerate(todo):
        NNR.read_item(w, item, ncfg, feat, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")


def vpd_stage(cfg: Config, tag: str) -> None:
    kc = IA.KN.load_config()  # E268's filed scan settings, unchanged
    w = World(cfg, "vpd")
    w.bases = KV.bases(w.uv, IA.KN.E226.load_config()[0].split_tol)
    if manifest_only():
        return log("manifest only: world and bases loaded, no reads")
    floor = IA.vpd_stage0(w, kc, tag)
    repeat("vpd", cfg.prior_vpd, cfg.repeat_tol_vpd, lambda i: IA.read_vpd(w, i, kc, floor), lambda i: RESULTS / "vpd" / f"{fname(i)}.json")
    todo = items(cfg, cfg.vpd_seeds)
    for k, item in enumerate(todo):
        IA.read_vpd(w, item, kc, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")


# ---- summary (CPU) --------------------------------------------------------------------------------------------------------


def read_json(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text()) if p.is_file() else None


def closeness(cfg: Config, r: dict[str, Any] | None) -> dict[str, Any]:
    """R2 on one training record."""
    if r is None or not r.get("done"):
        return {"counts": False, "reading": "not trained"}
    row = {k: r[k] for k in ("leftover", "gap", "target_gap")}
    if min(r["gap"], r["target_gap"]) < cfg.floor_gap:
        return row | {"counts": False, "reading": "no reading: gap under the floor"}
    ok = r["leftover"] is not None and r["leftover"] <= cfg.leftover_max
    return row | {"counts": ok, "reading": "counts" if ok else "no reading: leftover over the bar"}


def neuron_row(item: str) -> dict[str, Any] | None:
    r = read_json(RESULTS / "items" / f"{fname(item)}.json")
    if r is None or "stopped" in r:
        return None
    return {k: r[k] for k in ("E_b_K", "size_K", "n_changed_K", "margin_median", "too_small")} | NNR.remainder_columns(item)


def rivals(cfg: Config, close: dict[str, Any], nr: dict[str, Any]) -> dict[str, Any]:
    """R3 at neurons, E_b over K as filed, on seeds where both imitators count and neither item is too small."""
    per: dict[str, Any] = {}
    for s in cfg.seeds:
        a, b = nr.get(f"ft/x2_s{s}"), nr.get(f"ft/new_few_s{s}")
        if not all(close.get(f"{t}_s{s}", {}).get("counts") for t in ("x2", "new_few")):
            per[str(s)] = {"call": "not read: an imitator misses the closeness bar"}
        elif a is None or b is None or a["too_small"] or b["too_small"]:
            per[str(s)] = {"call": "not read: an item missing or too small"}
        else:
            d = a["E_b_K"] - b["E_b_K"]
            per[str(s)] = {"d": d, "call": "A" if d >= cfg.rival_gap else "B"}
    calls = [v["call"] for v in per.values() if v["call"] in ("A", "B")]
    call = "no call" if len(calls) < cfg.min_seeds else calls[0] if len(set(calls)) == 1 else "split"
    return {"per_seed": per, "n_seeds": len(calls), "call": call}


def summary(cfg: Config) -> None:
    close = {f"{t}_s{s}": closeness(cfg, read_json(train_file(t, s))) for t, s in pairs(cfg)}
    nr = {i: neuron_row(i) for i in items(cfg)}
    vp = {i: read_json(RESULTS / "vpd" / f"{fname(i)}.json") for i in items(cfg, cfg.vpd_seeds)}
    s0 = {p.name: json.loads(p.read_text()) for p in sorted(RESULTS.glob("stage0_*.json"))}
    out = {"experiment": EXPERIMENT, "stage0": s0, "R2_closeness": close, "R3": rivals(cfg, close, nr),
           "R4_randoms_over": {i: r["E_b_K"] for i, r in nr.items() if r and i.endswith("_rand") and r["E_b_K"] > cfg.random_max},
           "neurons": nr, "vpd": vp, "not_read": [i for i, r in {**nr, **vp}.items() if r is None]}  # fmt: skip
    dump(RESULTS / "summary.json", out)
    log(f"summary: R3 {out['R3']['call']} on {out['R3']['n_seeds']} seeds; {len(out['not_read'])} items not read")


def main(stage: str, shard: tuple[int, int], tag: str, dry: Path | None) -> int:
    cfg = load_config(dry is not None)
    point(dry or HERE / "results")
    if stage == "summary":
        summary(cfg)
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E139, E277, E268
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}, results {RESULTS}; settings {asdict(cfg)}")
    if stage == "train":
        train_stage(cfg, shard)
    elif stage in ("check", "neurons"):
        neurons_stage(cfg, shard, tag, stage == "check")
    else:
        vpd_stage(cfg, tag)
    log(f"{EXPERIMENT} DONE ({stage})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "check", "neurons", "vpd", "summary"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: its stage-0 file is results/stage0_<world>_<tag>.json")
    ap.add_argument("--dry", type=Path, default=None, metavar="OUT", help="CPU dry run: the 'dry' settings, results under OUT")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag, args.dry)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

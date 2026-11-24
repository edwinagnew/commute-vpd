"""E290 (cell P1.61; user 2026-11-23, "yes open the IA3 gain control"): fine-tunes whose only trainable parameters are
IA3 gains (Liu et al. 2022), on E58's five key-grid tasks with E147's data, loss, batch, clip and sampler seed, read with
E_b over the needed set K at E277's neuron featurization (scored) and E268's VPD featurization (descriptive). Plan
reports/plans/2026-11-23_plan_p1-61-ia3-gain-control.md; card card.md; settings and rules in results/predictions.json,
which the run refuses unless committed unchanged.

Arms (settings.arms): ffn, gains on down_proj's input only (the control: every changed neuron's write is l times M's,
kept by construction on the neuron variables; NOT on the featurization's per-layer remainder, so E_b over K as filed is
not 1 by construction); ia3, IA3's published sites on this llama-style net (k_proj, v_proj outputs and down_proj's
input, `commute.gains.IA3`). Each gain state is folded into fp32 weights (`gains.delta_ia3`). The reference "full" is
E147's saved step-30 update of the same task and seed. Every read item has a matched random (`<item>_rand`, E277's
Gaussian per changed tensor at the item's Frobenius norm). Items: '<arm>/<task>_s<seed>[_rand]'.

Stages; each GPU stage runs its stage 0 first (skipped once its file says done), and a miss stops the job:
    train      R0c (gains of 1 fold to dW = 0 and leave held-out P(target) unchanged), then the 30 trainings (300 steps,
               gains at 10, 30, 100, 300; read step by E12's rule) and each full fine-tune's held-out P(target)
                                                                     -> results/train/*.json, results/large/gains_*.pt
    neurons    E277's stage 0 (identity on the built gain edit: 0 flips; do-nothing: E_b over K 1) and R0d (the built
               edit l = 1 + 0.003 z on every down_proj: no head changed, f0 = 1 - E_b over the needed changed neurons),
               then every N-th item from the I-th, each checked against R1 (and R1b for ffn) as it lands
                                                                     -> results/items/, results/large/items_*.npz
    vpd        stage 0 (do-nothing E_b 1, identity on the built edit 0 flips), then the seed-0 items  -> results/vpd/
    summary    CPU: R1-R4 and the descriptive columns                -> results/summary.json

    python experiments/ia3_control/run.py --stage train --tag t0
    python experiments/ia3_control/run.py --stage neurons --shard 0 8 --tag n0
    python experiments/ia3_control/run.py --stage vpd --shard 0 10 --tag v0
    python experiments/ia3_control/run.py --stage summary
"""

from __future__ import annotations

import argparse
import copy
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
from commute import gains as G
from commute import keydep as KD
from commute import kstar_vpd as KV
from commute import needed as IX
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

KG: Any = load_experiment("key_grid", "run")  # E237: World (M, each state's D, E147's saved updates), load_config
NNR: Any = load_experiment("needed_neurons", "run")  # E277: stage0, read_item, matched_random, remainder_columns, E229
KN: Any = load_experiment("kstar_needed", "run")  # E268: the VPD scan's filed settings (load_config), E226 (split_tol)
R96, NN = KG.R96, NNR.NN
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
NNR.RESULTS, NNR.EXPERIMENT = RESULTS, EXPERIMENT  # E277's stage0 and read_item write here, stamped E290
log, dump, stop, fname = NNR.log, NNR.dump, NNR.stop, NNR.fname
SITES = {"down_proj", "k_proj", "v_proj"}


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    arms: dict[str, tuple[str, ...]]  # arm -> IA3 sites trained
    control_arm: str
    init: float
    lr: float
    optimizer: str
    batch: int
    grad_clip: float
    kl_lambda: float
    checkpoints: tuple[int, ...]
    p_learn: float
    full_step: int
    band: str
    derangements: int
    examples: int
    seed: int
    rows: int
    logit_rows: int
    cache_batch: int
    vpd_seeds: tuple[int, ...]
    random_seed: int
    built_gain_sd: float
    built_gain_seed: int
    floor_factor: float
    floor_min: float
    floor_stop: float
    too_small: int
    rival_gap: float
    random_max: float

    def __post_init__(self) -> None:
        assert len(set(self.tasks)) == 5 and self.seeds == (0, 1, 2) and set(self.vpd_seeds) <= set(self.seeds)
        assert self.control_arm in self.arms and all("down_proj" in s and set(s) <= SITES for s in self.arms.values())
        assert all(set(s) in ({"down_proj"}, SITES) for s in self.arms.values()), "an arm is FFN-only or all three IA3 sites"
        assert self.checkpoints == KD.E12_CHECKS and self.p_learn == KD.P_LEARN and self.full_step in self.checkpoints
        assert self.init == 1.0 and self.optimizer == "adam", "commute.gains trains from ones with Adam"
        assert 0 < self.floor_min < self.floor_stop and self.floor_factor >= 1 and self.too_small >= 1


def load_config() -> Config:
    raw: dict[str, Any] = json.loads(PRED.read_text())["settings"]
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    s["arms"] = {a: tuple(v) for a, v in raw["arms"].items()}
    return config_from_settings(Config, s)


def states(cfg: Config, seeds: tuple[int, ...] | None = None) -> list[tuple[str, int]]:
    return [(t, s) for t in cfg.tasks for s in seeds or cfg.seeds]


def items(cfg: Config, seeds: tuple[int, ...] | None = None) -> list[str]:
    """The control arm first, then the other arms, then the full fine-tunes; each item followed by its matched random."""
    arms = [cfg.control_arm, *(a for a in cfg.arms if a != cfg.control_arm), "full"]
    return [f"{a}/{t}_s{s}{r}" for a in arms for t, s in states(cfg, seeds) for r in ("", "_rand")]


def parse(item: str) -> tuple[str, str, int]:
    arm, rest = item.split("/")
    task, seed = rest.removesuffix("_rand").rsplit("_s", 1)
    return arm, task, int(seed)


def gains_file(arm: str, task: str, seed: int) -> Path:
    return RESULTS / "large" / f"gains_{arm}_{task}_s{seed}.pt"


def train_file(arm: str, task: str, seed: int) -> Path:
    return RESULTS / "train" / f"{arm}_{task}_s{seed}.json"


# ---- the world --------------------------------------------------------------------------------------------------------


class World:
    """E237's World (its M, each key state's D at the last position, E147's saved updates) and this run's items. `name`
    and `control` are what E277's stage0 reads (its stage-0 file name and the identity item)."""

    def __init__(self, cfg: Config, name: str) -> None:
        self.cfg, self.name, self.control = cfg, name, "built_gain"
        self.kg = KG.World(KG.load_config()[0])
        self.net, self.M = self.kg.net, self.kg.M
        self.all = items(cfg)
        self.uv: dict[str, tuple[Tensor, Tensor]] = {}  # the VPD featurization's inputs, set by vpd_stage only
        self.delta: dict[str, Tensor] = {}
        self.bases: dict[str, Any] = {}

    def D(self, task: str, seed: int) -> Tensor:
        return self.kg.read(self.kg.read_half187(self.kg.full["states"][R96.state_name(task, seed)]["task"]))

    def full_dw(self, task: str, seed: int) -> dict[str, Tensor]:
        """E147's saved step-30 update, fp32, keyed by parameter name (`old_tasks.load_dw` through E237's World)."""
        return self.kg.dw(R96.state_name(task, seed))

    def gain_dw(self, g: G.IA3) -> dict[str, Tensor]:
        return {f"{k}.weight": v for k, v in G.delta_ia3(g, self.net.W0).items()}

    def built(self) -> G.IA3:
        ones = G.IA3.ones(self.net.net, kv=False)
        z = torch.randn(ones.ff.shape, generator=torch.Generator().manual_seed(self.cfg.built_gain_seed))
        return G.IA3(1 + self.cfg.built_gain_sd * z)

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        t0, s0 = self.cfg.tasks[0], self.cfg.seeds[0]
        if item in ("do_nothing", "built_gain"):
            dw = {} if item == "do_nothing" else self.gain_dw(self.built())
            return R.Model(OT.with_delta(self.net, dw).net), self.M, self.D(t0, s0), KG.E221.LAST, {"windows_of": f"{t0}_s{s0}"}
        arm, task, seed = parse(item)
        if arm == "full":
            dw, rec = self.full_dw(task, seed), {"dw": self.kg.full["states"][R96.state_name(task, seed)]["file"]}
        else:
            if not gains_file(arm, task, seed).is_file():
                stop(RESULTS / "items" / f"{fname(item)}.stopped.json", {"item": item}, f"{item}: no trained gains; run --stage train first")
            saved = torch.load(gains_file(arm, task, seed), weights_only=False)
            dw, rec = self.gain_dw(G.IA3.of(saved["steps"][saved["read_step"]])), {"read_step": saved["read_step"]}
        F = R.Model(OT.with_delta(self.net, dw).net)
        if item.endswith("_rand"):
            dr, rec = NNR.matched_random(F.net, self.M.net, self.cfg.random_seed + self.all.index(item))
            F = R.Model(OT.with_delta(self.net, dr).net)
        return F, self.M, self.D(task, seed), KG.E221.LAST, rec


# ---- train ------------------------------------------------------------------------------------------------------------


def windows(task: str) -> dict[str, Any]:
    spec = json.loads(KG.E147_INV.read_text())["maps"][task]
    return torch.load(EXPERIMENTS.parent / spec["windows"], weights_only=False)


def p_heldout(w: World, ref: Any, win: dict[str, Any], g: G.IA3 | None = None, dw: dict[str, Tensor] | None = None) -> dict[str, float]:
    """FT.eval_arm_a on the map's held-out windows (E147's p_heldout) at M, with gains g by hooks, or at M + dw."""
    net, dev = w.net, w.net.device
    ev = lambda: FT.eval_arm_a(net.net, ref, win["heldout"], win["table"], dev, ci_len=win.get("ci_len"))
    if g is not None:
        with G.applied_ia3(net.net, g):
            return ev()
    if dw is not None:
        with net.at({k.removesuffix(".weight"): v for k, v in dw.items()}):
            return ev()
    return ev()


def stage0_train(w: World, ref: Any) -> None:
    """R0c: gains of 1 (all three sites) fold to a dW of exactly 0 and leave the held-out evaluation bit-identical."""
    path = RESULTS / "stage0_train.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        return log(f"{path.name}: done before")
    task, win = w.cfg.tasks[0], windows(w.cfg.tasks[0])
    ones = G.IA3.ones(w.net.net, kv=True, device=w.net.device)
    out = {"experiment": EXPERIMENT, "task": task, "delta_nonzero": sum(int(v.count_nonzero()) for v in G.delta_ia3(ones, w.net.W0).values()),
           "at_M": p_heldout(w, ref, win), "at_ones": p_heldout(w, ref, win, g=ones)}  # fmt: skip
    miss = [f"gains of 1 fold to {out['delta_nonzero']} nonzero entries"] if out["delta_nonzero"] else []
    miss += [f"gains of 1 move {k}: {out['at_M'][k]!r} -> {out['at_ones'][k]!r}" for k in out["at_M"] if out["at_M"][k] != out["at_ones"][k]]
    log(f"stage 0 (train): {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "R0c: " + "; ".join(miss))
    dump(path, out | {"done": True})


def train_one(w: World, ref: Any, arm: str, task: str, seed: int, win: dict[str, Any], p0: float) -> None:
    path, cfg, net = train_file(arm, task, seed), w.cfg, w.net
    if path.is_file() and json.loads(path.read_text()).get("done"):
        return log(f"{arm} {task} s{seed}: trained before, skipped")
    t0, train, dev = time.time(), win["train"], net.device
    a = FT.ArmAConfig(o_token=win["table"], kl_lambda=cfg.kl_lambda, batch=cfg.batch, seed=seed, ci_len=win.get("ci_len"))
    gen = torch.Generator().manual_seed(seed)  # finetune_arm_a's sampler: one draw of `batch` windows per step
    sample = lambda _k: train.tokens[torch.randint(0, len(train), (cfg.batch,), generator=gen)].to(dev)
    loss_of = lambda t: FT.arm_a_loss(net.net, ref, t, a, train.half)
    evals: dict[int, dict[str, float]] = {}
    saved: dict[int, dict[str, Tensor]] = {}

    def on_check(step: int, g: G.IA3) -> bool:
        if step in cfg.checkpoints:
            evals[step], saved[step] = p_heldout(w, ref, win, g=g), g.state()
            log(f"  {arm} {task} s{seed} step {step}: held-out P(target) {evals[step]['p_o']:.4f}")
        return False

    g = G.IA3.ones(net.net, kv=set(cfg.arms[arm]) == SITES, device=dev)
    steps = G.train(net.net, g, sample, loss_of, cfg.lr, cfg.checkpoints[-1], cfg.grad_clip, 10, on_check, log, apply=G.applied_ia3)
    step, learned = KD.scored_step(evals)
    torch.save({"steps": saved, "read_step": step}, gains_file(arm, task, seed))
    last = torch.cat([p.flatten() for p in g.params()])
    out = {"experiment": EXPERIMENT, "arm": arm, "task": task, "seed": seed, "p0": p0, "evals": evals, "read_step": step, "learned": learned,
           "n_gains": len(last), "gain_min": float(last.min()), "gain_max": float(last.max()), "loss_first": steps[0]["loss"],
           "loss_last": steps[-1]["loss"], "seconds": time.time() - t0, "done": True}  # fmt: skip
    dump(path, out)
    log(f"TRAINED {arm} {task} s{seed}: read step {step} (learned {learned}), P(target) {evals[step]['p_o']:.4f} (M {p0:.4f}); "
        f"gains {out['gain_min']:.4f}-{out['gain_max']:.4f}; {out['seconds']:.0f} s")  # fmt: skip


def train_stage(w: World) -> None:
    ref = copy.deepcopy(w.net.net).eval().requires_grad_(False)
    stage0_train(w, ref)
    e147 = json.loads((KG.E147_INV.parent / "stage0.json").read_text())["states"]
    for task in w.cfg.tasks:
        win = windows(task)
        p0 = p_heldout(w, ref, win)["p_o"]
        for seed in w.cfg.seeds:
            path = train_file("full", task, seed)
            if not path.is_file():
                p = p_heldout(w, ref, win, dw=w.full_dw(task, seed))["p_o"]
                dump(path, {"experiment": EXPERIMENT, "arm": "full", "task": task, "seed": seed, "p0": p0, "p_full": p,
                            "p_full_e147": e147[R96.state_name(task, seed)]["p_heldout"], "step": w.cfg.full_step})  # fmt: skip
            for arm in w.cfg.arms:
                train_one(w, ref, arm, task, seed, win, p0)
    log("STAGE TRAIN DONE")


# ---- neurons ----------------------------------------------------------------------------------------------------------


def neuron_columns(item: str) -> dict[str, Any]:
    """From the item's saved arrays: changed heads, needed changed neurons, E_b over them; E277's remainder columns."""
    z = {k: torch.from_numpy(v) for k, v in np.load(RESULTS / "large" / f"items_{fname(item)}.npz").items()}
    j = torch.arange(len(z["dd"])) % NNR.NN.PER_LAYER
    head, neuron, sel = j < NN.H, (j >= NN.H) & (j < NN.H + NN.NEUR), z["in_K"].bool() & z["changed"].bool()
    # E277's columns first: its E_b_K_neurons is None on an empty set, ours (the convention's 1) is never None
    return NNR.remainder_columns(item) | {"n_changed_heads": int((z["changed"].bool() & head).sum()), "n_K_changed_neurons": int((sel & neuron).sum()),
            "E_b_K_neurons": SI.explained_share(z["num"], z["den"], z["dd"], sel & neuron)}  # fmt: skip


def built_floor(w: World, ncfg: Any, feat: Any, floor: float) -> float:
    """R0d: the built gain edit read like any item; f0 = 1 - E_b over its needed changed neurons; no head changed."""
    path = RESULTS / "stage0_built_gain.json"
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        return float(done["f0"])
    NNR.read_item(w, "built_gain", ncfg, feat, floor)
    out = neuron_columns("built_gain") | {"experiment": EXPERIMENT}
    out["f0"] = f0 = 1 - out["E_b_K_neurons"]
    miss = [f"{out['n_changed_heads']} heads changed"] if out["n_changed_heads"] else []
    miss += [f"f0 {f0!r} outside [-1e-15, {w.cfg.floor_stop}]"] if not -1e-15 <= f0 <= w.cfg.floor_stop else []
    miss += [f"{out['n_K_changed_neurons']} needed changed neurons"] if out["n_K_changed_neurons"] < w.cfg.too_small else []
    log(f"R0d: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "R0d: " + "; ".join(miss))
    dump(path, out | {"done": True})
    return f0


def check_r1(cfg: Config, item: str, f0: float) -> None:
    """R1 on a gain arm's item (not a random): 1 - E_b over the needed changed neurons within floor_factor x the floor;
    R1b: the control arm changes no head. A miss is an apparatus failure and stops the job."""
    arm = parse(item)[0]
    if arm == "full" or item.endswith("_rand"):
        return
    c, bar = neuron_columns(item), cfg.floor_factor * max(f0, cfg.floor_min)
    miss = [f"R1b: {c['n_changed_heads']} heads changed"] if arm == cfg.control_arm and c["n_changed_heads"] else []
    if c["n_K_changed_neurons"] >= cfg.too_small and not 1 - c["E_b_K_neurons"] <= bar:
        miss.append(f"R1: 1 - E_b over the needed changed neurons {1 - c['E_b_K_neurons']!r} > {bar!r}")
    log(f"{item}: R1 {1 - c['E_b_K_neurons']:.3e} (bar {bar:.3e}, {c['n_K_changed_neurons']} needed changed neurons); "
        f"{'holds' if not miss else 'MISSES ' + '; '.join(miss)}")  # fmt: skip
    if miss:
        stop(RESULTS / "items" / f"{fname(item)}.r1.json", c | {"item": item}, "; ".join(miss))


def neurons_stage(cfg: Config, shard: tuple[int, int], tag: str) -> None:
    ncfg, c229 = NNR.load_config(), NNR.load_e229()
    scan = ("band", "derangements", "examples", "seed", "rows", "logit_rows", "cache_batch")
    assert all(getattr(ncfg, k) == getattr(cfg, k) for k in scan), "the filed scan settings are not E277's"
    w = World(cfg, "neurons")
    feat = NNR.E229.feat_of(c229, NNR.E229.load_pct(c229), cfg.band)
    floor = NNR.stage0(w, ncfg, feat, tag)["identity"]["max_dev"]
    f0 = built_floor(w, ncfg, feat, floor)
    todo = items(cfg)[shard[0] :: shard[1]]
    log(f"neurons: f0 {f0:.3e}; {len(todo)} of {len(items(cfg))} items {todo}")
    for k, item in enumerate(todo):
        NNR.read_item(w, item, ncfg, feat, floor)
        check_r1(cfg, item, f0)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")


# ---- vpd --------------------------------------------------------------------------------------------------------------


def vpd_feat(w: World, item: str) -> tuple[R.Model, KV.Feat, Tensor, slice, dict[str, Any]]:
    F, M, toks, at, rec = w.make(item)
    return F, KV.featurize(M, F, w.uv, w.delta, w.bases), toks, at, rec


def changed_mask(feat: KV.Feat) -> Tensor:
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    return ch


def vpd_stage0(w: World, kc: Any, tag: str) -> float:
    path = RESULTS / f"stage0_vpd_{tag}.json"
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        return float(done["floor"])
    F, feat, toks, _, _ = vpd_feat(w, "do_nothing")
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
    out: dict[str, Any] = {"experiment": EXPERIMENT, "card": torch.cuda.get_device_name(), "dn_n_changed": len(feat.changed),
                           "dn_dd_max": float(dd.abs().max()), "dn_E_b_K": SI.explained_share(num, den, dd, changed_mask(feat))}  # fmt: skip
    F, feat, toks, at, _ = vpd_feat(w, w.control)
    out["identity"] = idn = IX.identity_scan(F, feat, toks, at, kc.examples, kc.ident_vars, kc.rows, kc.logit_rows, kc.cache_batch, log)
    out["floor"] = idn["max_dev"]
    out["est_seconds_per_item"] = idn["n_vars"] * kc.derangements * min(kc.examples, idn["n_windows"]) / idn["cases_per_s"]
    miss = [f"identity: {idn['flips']} argmax changes"] if idn["flips"] else []
    miss += [f"do-nothing: {k} {out[k]}" for k, want in (("dn_n_changed", 0), ("dn_dd_max", 0.0), ("dn_E_b_K", 1.0)) if out[k] != want]
    log(f"stage 0 (vpd): {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0 (vpd): " + "; ".join(miss))
    dump(path, out | {"done": True})
    log(f"STAGE 0 DONE (vpd): est. {out['est_seconds_per_item']:.0f} s per item")
    return float(out["floor"])


PRIOR_VPD = {"full/copy_s0": EXPERIMENTS / "claim2_needed" / "results" / "items" / "e58-copy_s0_scaled.json"}  # E275, c = 1


def read_vpd(w: World, item: str, kc: Any, floor: float) -> None:
    """E_b over K at VPD as E278 read its grid items: one write pass, K by `needed.needed_set` with E268's settings."""
    path = RESULTS / "vpd" / f"{fname(item)}.json"
    if path.is_file():
        return log(f"{item}: done before, skipped")
    if item in PRIOR_VPD:
        p = json.loads(PRIOR_VPD[item].read_text())
        return dump(path, {k: p.get(k) for k in ("E_b_K", "size_K", "n_changed_K", "min_margin", "n_vars")} | {
            "experiment": EXPERIMENT, "item": item, "reading": "prior", "source": str(PRIOR_VPD[item].relative_to(EXPERIMENTS.parent))})
    t0 = time.time()
    F, feat, toks, at, rec = vpd_feat(w, item)
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
    nd = IX.needed_set(F, feat, toks, at, kc.derangements, kc.examples, kc.seed, kc.rows, kc.logit_rows, kc.cache_batch, log)
    K, ch, mg = nd.K.cpu(), changed_mask(feat), nd.run.margin.flatten().cpu()
    out = {"experiment": EXPERIMENT, "item": item, "reading": "read", "n_vars": feat.n, "n_changed": len(feat.changed), "n_pairs": nd.n_pairs,
           "E_b_K": SI.explained_share(num, den, dd, K & ch), "E_b_changed": SI.explained_share(num, den, dd, ch), "size_K": int(K.sum()),
           "n_changed_K": int((K & ch).sum()), "too_small": int(K.sum()) < w.cfg.too_small, "margin_median": float(mg.median()),
           "min_margin": float(mg.min()), "n_fragile_positions": int((mg <= floor).sum()), "fragile_floor": floor} | rec  # fmt: skip
    np.savez_compressed(RESULTS / "large" / f"vpd_{fname(item)}.npz", in_K=K.numpy(), num=num.numpy(), den=den.numpy(), dd=dd.numpy(), changed=ch.numpy())
    out |= {"seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, out)
    log(f"{item}: E_b over K {out['E_b_K']:.6f}; |K| {out['size_K']} ({out['n_changed_K']} changed); margin median {out['margin_median']:.3g}; {out['seconds']:.0f} s")
    del F, nd
    torch.cuda.empty_cache()


def vpd_stage(cfg: Config, shard: tuple[int, int], tag: str) -> None:
    from commute import vpd

    kc = KN.load_config()
    w = World(cfg, "vpd")
    w.uv, w.delta = vpd.component_weights(w.net.model), vpd.weight_deltas(w.net.model)
    w.bases = KV.bases(w.uv, KN.E226.load_config()[0].split_tol)
    floor = vpd_stage0(w, kc, tag)
    todo = items(cfg, cfg.vpd_seeds)[shard[0] :: shard[1]]
    log(f"vpd: {len(todo)} of {len(items(cfg, cfg.vpd_seeds))} items {todo}")
    for k, item in enumerate(todo):
        read_vpd(w, item, kc, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")


# ---- summary (CPU) ----------------------------------------------------------------------------------------------------


def read_json(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text()) if p.is_file() else None


def gain_rows(cfg: Config) -> dict[str, Any]:
    """R2: per task, the bar (min over max of the full fine-tunes' gains in P(target)) and each arm's gain shares."""
    out: dict[str, Any] = {}
    for t in cfg.tasks:
        full = [read_json(train_file("full", t, s)) for s in cfg.seeds]
        if any(f is None for f in full):
            out[t] = {"reading": "not trained"}
            continue
        g_full = [f["p_full"] - f["p0"] for f in full if f]
        row: dict[str, Any] = {"p0": full[0]["p0"] if full[0] else None, "gain_full": g_full}
        if min(g_full) <= 0:
            out[t] = row | {"reading": "not read: a full fine-tune has no gain over M"}
            continue
        row["bar"] = min(g_full) / max(g_full)
        for arm in cfg.arms:
            r = [read_json(train_file(arm, t, s)) for s in cfg.seeds]
            if any(x is None for x in r):
                row[arm] = {"reading": "not trained"}
                continue
            share = [(x["evals"][str(x["read_step"])]["p_o"] - x["p0"]) / gf for x, gf in zip(r, g_full, strict=True) if x]
            at30 = [(x["evals"][str(cfg.full_step)]["p_o"] - x["p0"]) / gf for x, gf in zip(r, g_full, strict=True) if x]
            row[arm] = {"share": share, "share_step30": at30, "read_step": [x["read_step"] for x in r if x], "mean_share": fmean(share),
                        "reaches": fmean(share) >= row["bar"]}  # fmt: skip
        out[t] = row
    return out


def neuron_rows(cfg: Config) -> dict[str, Any]:
    out = {}
    for item in items(cfg):
        r = read_json(RESULTS / "items" / f"{fname(item)}.json")
        out[item] = None if r is None or "stopped" in r else {k: r[k] for k in ("E_b_K", "size_K", "n_changed_K", "margin_median", "too_small")} | neuron_columns(item)
    return out


def rivals(cfg: Config, gains: dict[str, Any], nr: dict[str, Any]) -> dict[str, Any]:
    """R3 on the control arm, E_b over K as filed, only on tasks where it reaches the full gain."""
    a, per = cfg.control_arm, {}
    for t in cfg.tasks:
        if not gains.get(t, {}).get(a, {}).get("reaches"):
            continue
        rows = [(nr.get(f"{a}/{t}_s{s}"), nr.get(f"full/{t}_s{s}")) for s in cfg.seeds]
        if any(x is None or y is None or x["too_small"] or y["too_small"] for x, y in rows):
            per[t] = {"call": "not read (an item missing or too small)"}
            continue
        d = [x["E_b_K"] - y["E_b_K"] for x, y in rows if x and y]
        per[t] = {"diff": d, "call": "A" if all(v >= cfg.rival_gap for v in d) else "B" if all(v < cfg.rival_gap for v in d) else "split"}
    calls = {v["call"] for v in per.values() if v["call"] in ("A", "B", "split")}
    overall = "no call" if not calls else calls.pop() if len(calls) == 1 else "split"
    return {"per_task": per, "call": overall}


def summary(cfg: Config) -> None:
    gains, nr = gain_rows(cfg), neuron_rows(cfg)
    vp = {i: read_json(RESULTS / "vpd" / f"{fname(i)}.json") for i in items(cfg, cfg.vpd_seeds)}
    rand = {i: r["E_b_K"] for i, r in nr.items() if r and i.endswith("_rand")}
    s0 = {p.name: json.loads(p.read_text()) for p in sorted(RESULTS.glob("stage0_*.json"))}
    out = {"experiment": EXPERIMENT, "stage0": s0, "R2_gain": gains, "R3": rivals(cfg, gains, nr),
           "R4_randoms_over": {i: v for i, v in rand.items() if v > cfg.random_max}, "neurons": nr, "vpd": vp,
           "not_read": [i for i, r in nr.items() if r is None] + [i for i, r in vp.items() if r is None]}  # fmt: skip
    dump(RESULTS / "summary.json", out)
    log(f"summary: R3 {out['R3']['call']}; {len(out['not_read'])} items not read")


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    cfg = load_config()
    if stage == "summary":
        summary(cfg)
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 reads, as E277 and E268
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}; settings {asdict(cfg)}")
    for d in ("items", "large", "train", "vpd"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    if stage == "train":
        train_stage(World(cfg, "train"))
    elif stage == "neurons":
        neurons_stage(cfg, shard, tag)
    else:
        vpd_stage(cfg, shard, tag)
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "neurons", "vpd", "summary"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: its stage-0 file is results/stage0_<stage>_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

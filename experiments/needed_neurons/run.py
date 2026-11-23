"""E277 (cell P1.56; user, 2026-11-23): the needed set K and the write share E_b over K at E229's top-activating-neuron
featurization of the 67M, beside E268's at VPD. Settings and filed expected values in results/predictions.json, which
the run refuses unless committed unchanged. Card: card.md.

Variables (`commute.needed_neurons`): every attention head, every MLP neuron (its value on a token is its GELU output
when above its base p97 on Pile text, E217's saved percentiles as E229 pins them, else 0) and one exact remainder per MLP
layer; 12,316 in all. The embedding and unembedding are not tested; norms belong to the heads and neurons of their
layer (E217's convention); the 67M has no biases. K: a variable is needed when, for some tested pair (x, x') of D
(`commute.needed.derangements`, E268's pairs and seed), setting its value on x to its value in F's own run on x' changes
F's argmax token at any read position. E_b over K: `commute.stand_in.explained_share(num, den, dd, K & changed)`, num,
den, dd from one `stand_in.write_change` pass with the module's sites (M's mechanism on F's own input to it).

Per item results/items/<item>.json (a rerun skips it) and results/large/items_<item>.npz (in_K, first_pair, flips, num,
den, dd, changed). Stages; each job runs its world's stage 0 first (skipped once results/stage0_<world>_<tag>.json
says done), and any miss stops the job:
    0        the built world's stage 0 alone: the identity interchange (x' = x) on x2, every variable, its first
             `examples` windows at its read positions: 0 argmax changes (largest logit deviation = the floor, and the
             rate); the do-nothing (F = M): no changed variable, dd max exactly 0, E_b over K exactly 1
    built    the seven E247 built edits, F, M, D and read positions as E268 builds them (E257's World)
    trained  E229's 20 fine-tunes (D = its 512 read windows, last position), each followed by a random update matched to
             it per parameter tensor (Frobenius norm of F - M); this world's stage 0 is the same on E229's control item
             and on M against a copy of itself
    e62      amendment 1 (results/predictions_a1.json): E257's E62 items, copy and lookup seeds 0-2 and lookup scaled to
             copy's size (E275's nine), in the built world; summary adds head (2, 4)'s columns to every item
    panel    amendment 1: E272's realistic panel (wide set, states picked at kl_hi: 12 tasks x 3 seeds and each task's
             seed-0 Gaussian), F, M, D and the last position as E272's eb.py builds them; score_panel.py scores it (CPU)
    summary  CPU: every item JSON into results/summary.json

    python experiments/needed_neurons/run.py --stage 0 --tag b0
    python experiments/needed_neurons/run.py --stage built --shard 0 2 --tag b0
    python experiments/needed_neurons/run.py --stage trained --shard 0 4 --tag t0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import kstar as K
from commute import needed as IX
from commute import needed_neurons as NN
from commute import rho
from commute import rule_retention as R
from commute import stand_in as SI
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

E229: Any = load_experiment("topk_upper", "run")  # Config, load_pct, feat_of; E217 (setup, windows, pair, LAST, log/dump/stop)
E217 = E229.E217
EB: Any = load_experiment("convergence_panel", "eb")  # E272: states, load_windows, random_file, RUN (TASKS, Config, PRED), KP
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, PRED_A1 = RESULTS / "predictions.json", RESULTS / "predictions_a1.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, dump, stop = E217.log, E217.dump, E217.stop


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    built: tuple[str, ...]
    control: str  # the built world's identity control
    band: str  # E229's band: the featurization's bounds
    derangements: int  # P
    examples: int  # windows tested per derangement (all of D when larger)
    seed: int  # the pairs' seed
    rows: int  # cases per forward with one read position
    logit_rows: int  # cases x read positions per forward, at most
    cache_batch: int  # windows per batch of F's cached run and of the write pass
    random_seed: int  # trained item k's matched random uses random_seed + k
    too_small: int  # |K| below this reads "too small to read"

    def __post_init__(self) -> None:
        assert self.control in self.built and len(self.built) == 7 and self.derangements >= 1 and self.examples >= 1
        assert self.rows >= 1 and self.logit_rows >= self.rows and self.too_small >= 1


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def load_e229() -> Any:
    return config_from_settings(E229.Config, json.loads(E229.PRED.read_text())["settings"])


def fname(item: str) -> str:
    return item.replace("/", "__")


@dataclass(frozen=True)
class A1:  # amendment 1 (2026-11-23), filed in predictions_a1.json "settings": the e62 and panel stages
    e62: tuple[str, ...]  # E257's E62 items (E275 read the same nine)
    panel_set: str  # E272's set ("wide") and the match its states were picked at ("kl_hi")
    panel_match: str


def load_a1() -> A1:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED_A1.read_text())["settings"].items()}
    return config_from_settings(A1, s)


def e272() -> tuple[Any, dict[str, Any]]:
    """E272's Config and thresholds from its predictions.json, as its run.load() reads them but without its
    require_filed (on a pod PRED_SHA is this experiment's, not E272's)."""
    P = json.loads(EB.RUN.PRED.read_text())
    return config_from_settings(EB.RUN.Config, P["settings"]), P["thresholds"]


def panel_items(a1: A1) -> dict[str, dict[str, Any]]:
    """panel/<task>/<state stem> (random_<stem> for its seed-0 Gaussian) -> E272's task, state and seed, in E272's order
    (eb.states: each task's Gaussian first, then its picked states)."""
    out = {}
    for t in EB.RUN.TASKS:
        for f in EB.states(t, a1.panel_set, a1.panel_match):
            stem, rnd = Path(f).stem, f.startswith("random/")
            out[f"panel/{t}/{'random_' if rnd else ''}{stem}"] = {"task": t, "state": f, "random": rnd, "seed": int(stem.rsplit("_step", 1)[0].rsplit("_s", 1)[1])}
    return out


# ---- the two worlds ---------------------------------------------------------------------------------------------------


class Built:
    """E268's items: E257's World with E242's matched scale of the KL-matched random (E268's world(), without the split's
    bases, which only the VPD featurization reads)."""

    name = "built"

    def __init__(self, cfg: Config, stage: str = "built") -> None:
        RB = load_experiment("rb_states", "run")
        self.w = RB.World(RB.load_config())
        r = json.loads((RB.S242 / "items" / f"{RB.fname(RB.E242.load_config()[0].random_kl)}.json").read_text())
        self.w.kl_x = {k: r[k] for k in ("x", "kl_matched", "match", "match_evals")}
        self.control, self.panel = cfg.control, {}
        if stage == "panel":
            self.panel = panel_items(load_a1())
        self.items = list(load_a1().e62) if stage == "e62" else list(self.panel) if stage == "panel" else list(cfg.built)

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        if item in self.panel:
            return self.panel_state(item)
        F, M, toks, at, _ = self.w.make(item)
        return F, M, toks, at, {}

    def panel_state(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        """E272's state as its eb.py reads it: its D from E272's windows manifest (sha checked), F = M + the saved dW (the
        seed-0 Gaussian from E272's random_file) through E257's World.from_dw, read at the last position."""
        e, (C, th) = self.panel[item], e272()
        toks = EB.load_windows(C, th, e["task"])
        f = EB.random_file(e["state"], C) if e["random"] else EB.RUN.REPO / e["state"]
        F, M, toks, at, _ = self.w.from_dw(f, toks.cuda(), EB.KP.LAST)
        return F, M, toks, at, e


class Trained:
    """E229's items as E217 builds them, each followed by its matched random (`<item>_rand`)."""

    name = "trained"

    def __init__(self, cfg: Config, c229: Any) -> None:
        self.c, self.seed0 = c229, cfg.random_seed
        self.net, self.M, self.data = E217.setup(c229)
        self.base = [i for s in c229.seeds for i in c229.items(s)]
        self.items, self.control = [x for i in self.base for x in (i, f"{i}_rand")], c229.control_item

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        from commute import old_tasks as OT

        base = self.control if item == "do_nothing" else item.removesuffix("_rand")
        toks = E217.windows(self.c, self.M, self.data, base)
        if item == "do_nothing":
            return R.Model(OT.clone_net(self.net).net), self.M, toks, E217.LAST, {}
        F = E217.pair(self.net, self.M, base).F
        if item == base:
            return F, self.M, toks, E217.LAST, {}
        dw, rec = matched_random(F.net, self.M.net, self.seed0 + self.base.index(base))
        del F
        return R.Model(OT.with_delta(self.net, dw).net), self.M, toks, E217.LAST, rec


def matched_random(Fnet: Any, Mnet: Any, seed: int) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """A Gaussian per parameter tensor F changed, each at the Frobenius norm of F - M on that tensor (one generator)."""
    gen, out, mp = torch.Generator().manual_seed(seed), {}, dict(Mnet.named_parameters())
    for k, p in Fnet.named_parameters():
        d = (p.detach().double() - mp[k].detach().double()).norm()
        if d > 0:
            r = torch.randn(p.shape, generator=gen, dtype=torch.float32).double()
            out[k] = (r * (d / r.norm())).float()
    size2 = sum(float((p.detach().double() - mp[k].detach().double()).pow(2).sum()) for k, p in Fnet.named_parameters())
    return out, {"random_seed": seed, "n_tensors": len(out), "size2_F": size2, "size2_rand": sum(float(v.double().pow(2).sum()) for v in out.values())}


# ---- one item ---------------------------------------------------------------------------------------------------------


def write_pass(F: R.Model, M: R.Model, toks: Tensor, feat: K.Feat, batch: int) -> tuple[Tensor, Tensor, Tensor, Tensor, list[str]]:
    p = rho.Pair(F, M)
    names, ch = NN.variables(p)
    num, den, dd = SI.write_change(F.net, NN.sites(p, feat, toks.shape[1]), len(names), list(toks.split(batch)))
    return num, den, dd, ch, names


def rows_for(cfg: Config, n_at: int) -> int:
    return IX.rows_per_forward(cfg.rows, cfg.logit_rows, n_at)


def read_item(world: Any, item: str, cfg: Config, feat: K.Feat, floor: float) -> None:
    path = RESULTS / "items" / f"{fname(item)}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    F, M, toks, at, rec = world.make(item)
    assert toks.dim() == 2, f"{item}: the windows of D must have one length, got {tuple(toks.shape)}"
    num, den, dd, ch, _ = write_pass(F, M, toks, feat, cfg.cache_batch)
    run = NN.cache(F.net, toks, at, cfg.cache_batch)
    m = min(cfg.examples, len(toks))
    ex, pt = IX.derangements(len(toks), cfg.derangements, m, cfg.seed)
    n_at = run.top.shape[1]
    log(f"{item}: {len(toks)} windows of {toks.shape[1]} tokens, {n_at} read positions, {len(ch)} variables, {int(ch.sum())} changed, "
        f"{len(ex)} pairs ({cfg.derangements} derangements x {m}), {rows_for(cfg, n_at)} cases per forward")  # fmt: skip
    sc = NN.scan(F.net, feat, run, at, ex, pt, rows_for(cfg, n_at), log)
    Kn = sc.first_pair >= 0
    first_flip = torch.where(Kn, sc.first_pair // m, -1)
    mg = run.margin.flatten().cpu()
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "world": world.name, "n_windows": len(toks), "window_len": toks.shape[1],
           "n_read_positions": n_at, "n_vars": len(ch), "n_changed": int(ch.sum()), "n_pairs": len(ex), "derangements": cfg.derangements,
           "examples": m, "size_K": int(Kn.sum()), "n_changed_K": int((Kn & ch).sum()), "E_b_K": SI.explained_share(num, den, dd, Kn & ch),
           "E_b_changed": SI.explained_share(num, den, dd, ch), "too_small": int(Kn.sum()) < cfg.too_small,
           "margin_median": float(mg.median()), "margin_min": float(mg.min()), "n_fragile_positions": int((mg <= floor).sum()),
           "fragile_floor": floor, "max_dev": sc.max_dev,
           "by_derangement": [{"size_K": int((Kn & (first_flip <= q)).sum()), "n_changed_K": int((Kn & (first_flip <= q) & ch).sum()),
                               "E_b_K": SI.explained_share(num, den, dd, Kn & (first_flip <= q) & ch)} for q in range(cfg.derangements)]} | rec  # fmt: skip
    np.savez_compressed(RESULTS / "large" / f"items_{fname(item)}.npz", in_K=Kn.numpy(), first_pair=sc.first_pair.numpy(), flips=sc.flips.numpy(),
                        num=num.numpy(), den=den.numpy(), dd=dd.numpy(), changed=ch.numpy())  # fmt: skip
    out |= {"seconds": time.time() - t0, "cases_per_s": sc.rows / sc.seconds, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, out)
    log(f"{item}: E_b over K {out['E_b_K']:.6f}, over all changed {out['E_b_changed']:.6f}; |K| {out['size_K']} ({out['n_changed_K']} changed) "
        f"of {out['n_vars']}; margin median {out['margin_median']:.3g}, min {out['margin_min']:.3g}; {out['seconds']:.0f} s")  # fmt: skip
    del F, M, run
    torch.cuda.empty_cache()


# ---- stage 0, main ----------------------------------------------------------------------------------------------------


def identity(world: Any, cfg: Config, feat: K.Feat) -> dict[str, Any]:
    """x' = x on the control's first `examples` windows, every variable, at its read positions: 0 argmax changes
    expected; the largest logit deviation is the scan path's fp32 floor and the rate times the counts each item's time."""
    F, _, toks, at, _ = world.make(world.control)
    run = NN.cache(F.net, toks, at, cfg.cache_batch)
    ex = torch.arange(min(cfg.examples, len(toks)))
    sc = NN.scan(F.net, feat, run, at, ex, ex, rows_for(cfg, run.top.shape[1]), log)
    return {"item": world.control, "flips": int(sc.flips.sum()), "max_dev": sc.max_dev, "min_margin": float(run.margin.min()), "cases": sc.rows,
            "cases_per_s": sc.rows / sc.seconds, "n_read_positions": run.top.shape[1], "n_windows": len(toks), "window_len": toks.shape[1]}  # fmt: skip


def stage0(world: Any, cfg: Config, feat: K.Feat, tag: str) -> dict[str, Any]:
    path = RESULTS / f"stage0_{world.name}_{tag}.json"
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        log(f"{path.name}: done before")
        return done
    t0 = time.time()
    F, M, toks, _, _ = world.make("do_nothing")
    num, den, dd, ch, _ = write_pass(F, M, toks, feat, cfg.cache_batch)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "world": world.name, "dn_n_changed": int(ch.sum()), "dn_dd_max": float(dd.abs().max()),
                           "dn_E_b_K": SI.explained_share(num, den, dd, ch), "n_vars": len(ch)}  # any K: K & changed is within changed  # fmt: skip
    del F
    idn = out["identity"] = identity(world, cfg, feat)
    out["est_seconds_per_item"] = len(ch) * cfg.derangements * min(cfg.examples, idn["n_windows"]) / idn["cases_per_s"]
    miss = [f"identity: {idn['flips']} argmax changes"] if idn["flips"] else []
    miss += [f"do-nothing: {k} {out[k]}" for k, want in (("dn_n_changed", 0), ("dn_dd_max", 0.0), ("dn_E_b_K", 1.0)) if out[k] != want]
    out["seconds"] = time.time() - t0
    log(f"stage 0 ({world.name}): {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log(f"STAGE 0 DONE ({world.name}): est. {out['est_seconds_per_item']:.0f} s per item read like {world.control}")
    return out


def remainder_columns(item: str) -> dict[str, Any]:
    """Descriptive, added before the scored items (redteam-lead 2026-11-23, note-2026-11-23-redteam-e277-prerun.md):
    the four whole-MLP remainders' share of dd over K & changed, and E_b over K without them (heads and neurons) and
    over the neurons only. From the saved arrays; layout per layer H heads, NEUR neurons, one remainder."""
    p = RESULTS / "large" / f"items_{fname(item)}.npz"
    if not p.is_file():
        return {}
    z = {k: torch.from_numpy(v) for k, v in np.load(p).items() if k in ("num", "den", "dd", "changed", "in_K")}
    sel = z["in_K"].bool() & z["changed"].bool()
    j = torch.arange(len(sel)) % (NN.H + NN.NEUR + 1)
    rem, neuron = j == NN.H + NN.NEUR, (j >= NN.H) & (j < NN.H + NN.NEUR)
    d = float(z["dd"][sel].sum())
    eb = lambda s: SI.explained_share(z["num"], z["den"], z["dd"], s) if bool(s.any()) else None
    return {"rem_dd_share_K_changed": float(z["dd"][sel & rem].sum()) / d if d > 0 else None, "E_b_K_no_rem": eb(sel & ~rem),
            "E_b_K_neurons": eb(sel & neuron), "n_changed_K_no_rem": int((sel & ~rem).sum())}


HEAD24 = 2 * (NN.H + NN.NEUR + 1) + 4  # variable L2.h4 (amendment 1: the head (2, 4) question at a second featurization)


def head24_columns(item: str) -> dict[str, Any]:
    """Head (2, 4), one variable here: in K, its flip count (of n_pairs), its kept share (explained / dd, None when dd
    is 0), its dd and its share of dd over K & changed. From the saved arrays; descriptive (amendment 1)."""
    p = RESULTS / "large" / f"items_{fname(item)}.npz"
    if not p.is_file():
        return {}
    z = {k: torch.from_numpy(v) for k, v in np.load(p).items() if k in ("num", "den", "dd", "changed", "in_K", "flips")}
    u, sel = HEAD24, z["in_K"].bool() & z["changed"].bool()
    dd, d = float(z["dd"][u]), float(z["dd"][sel].sum())
    return {"h24_in_K": bool(z["in_K"][u]), "h24_flips": int(z["flips"][u]), "h24_changed": bool(z["changed"][u]), "h24_dd": dd,
            "h24_kept_share": float(SI.explained(z["num"], z["den"])[u]) / dd if dd > 0 else None,
            "h24_dd_share_K_changed": dd / d if d > 0 and bool(sel[u]) else None}  # fmt: skip


def summary() -> None:
    every = [json.loads(p.read_text()) | remainder_columns(json.loads(p.read_text())["item"]) | head24_columns(json.loads(p.read_text())["item"])
             for p in sorted((RESULTS / "items").glob("*.json"))]  # fmt: skip
    s0 = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("stage0_*.json"))]
    dump(RESULTS / "summary.json", {"experiment": EXPERIMENT, "stage0": s0, "items": {r["item"]: r for r in every}})
    log(f"summary: {len(every)} items, {len(s0)} stage-0 files")


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    if stage == "summary":
        summary()
        return 0
    cfg, c229 = load_config(), load_e229()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E217, E229, E268
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}; settings {asdict(cfg)}")
    for d in ("items", "large"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    feat = E229.feat_of(c229, E229.load_pct(c229), cfg.band)
    world = Trained(cfg, c229) if stage == "trained" else Built(cfg, stage)
    floor = stage0(world, cfg, feat, tag)["identity"]["max_dev"]
    if stage == "0":
        return 0
    todo = world.items[shard[0] :: shard[1]]
    log(f"stage {stage}: {len(todo)} of {len(world.items)} items {todo}")
    for k, item in enumerate(todo):
        read_item(world, item, cfg, feat, floor)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("0", "built", "trained", "e62", "panel", "summary"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<world>_<tag>.json")
    args = ap.parse_args()
    if args.stage in ("e62", "panel"):  # amendment 1: its file is the filed one (PRED_SHA on a pod), and it pins the original's sha
        require_filed(PRED_A1)
        got, want = hashlib.sha256(PRED.read_bytes()).hexdigest(), json.loads(PRED_A1.read_text())["predictions_sha256"]
        assert got == want, f"{PRED.name} sha256 {got}, amendment 1 pins {want}"
    else:
        require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

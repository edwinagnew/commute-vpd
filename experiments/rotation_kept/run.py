"""E286 (cell P1.60; user 2026-11-23 via whip-lead, "yes run the rotation-tolerant diagnostic"): a rotation-tolerant
kept share beside the frozen definition's kept, a diagnostic and not the definition. Plan
reports/plans/2026-11-23_plan_p1-60-rotation-kept.md; card card.md; settings and filed values in
results/predictions.json, which the run refuses unless committed unchanged. Measurement: `commute.rotation_kept`.

Per item (F, M, D) at a basis (vpd: `kstar_vpd` subcomponents and remainders; nn: `needed_neurons` heads, neurons and
remainders), one pass of F over the first `windows` windows of D: for the read set N (the top 1% changed members of K
by flips of the saved scan) the strict, own-space, needed-space and matched-null parts of each write change, pooled
with remainders out (the read) and in (beside it). results/items/<basis>__<item>.json, per-variable rows in
results/large/items_<basis>__<item>.npz; a rerun skips written items.

Stages, each job running `control` first (skipped once results/stage0_<tag>.json says done); any miss stops the job:
    control    do-nothing at both bases (dd exactly 0, N empty, shares 1, own spaces hold own_energy); x2 at both
               bases (nn: every changed neuron's own share 1 - 1e-9, pooled own 0.99; both: num, den, dd repeat the
               saved E247/E277 arrays); the built rotation controls at nn (W_i turned by theta toward a needed W_j:
               own sin^2(theta/2), rotation share 1; toward a non-needed W_k: rotation share at most rot_other_max)
    built      x10, rebuilt, new_few, new_thin and the two randoms, at vpd and nn
    e62        E62's nine states at vpd (E275's K) and nn (E277's K)
    head24     head (2, 4) alone on the copy windows of E62's six states (nn): strict, own share, its null, flips
    trained    E229's 20 fine-tunes and their matched randoms (nn)
    panel_nn   E272's 48 panel states (nn)
    panel_vpd  E272's 24 seed-0 and Gaussian states (vpd), K found again on `panel_vpd_derangements` derangements (pod)
    gauge      (added 2026-11-24, gauge.py) M under a random per-head gauge, read at both bases; and the share of each
               trained update's head matrices along the gauge's tangent, beside a matched random
    gauge_pool, gauge_sub   (added 2026-11-24, gauge_follow.py) the gauged M read at VPD with each layer's attention
               pooled; E62's and E229's updates read at VPD with their gauge component removed
    summary    CPU: every item JSON into results/summary.json

    python experiments/rotation_kept/run.py --stage built --tag b0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import known_edits as KE
from commute import kstar_vpd as KV
from commute import needed as IX
from commute import needed_neurons as NN
from commute import rho
from commute import rotation_kept as RK
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

KN: Any = load_experiment("kstar_needed", "run")  # E268: world, featurized, saved, load_config (scan settings), log/dump/stop
NNR: Any = load_experiment("needed_neurons", "run")  # E277: Built, Trained, load_config, load_e229, panel_items, e272, EB
HE: Any = load_experiment("head_edit", "run")  # E280: targets (E62's copy windows)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E275 = EXPERIMENTS / "claim2_needed" / "results" / "large"
log, dump, stop = KN.log, KN.dump, KN.stop


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    windows: int
    top_frac: float
    own_energy: float
    rank_rtol: float
    too_small: int
    too_wide: float
    cache_batch: int
    control: str
    built: tuple[str, ...]
    head_layer: int
    head: int
    head_states: tuple[str, ...]
    panel_vpd_derangements: int
    theta_deg: float
    rot_other_max: float
    order_tol: float
    transcription_rel_tol: float
    seed: int

    def __post_init__(self) -> None:
        assert self.control == self.built[0] == "x2" and len(self.built) == 7 and 0 < self.top_frac < 1 and 0 < self.own_energy <= 1
        assert self.head_layer * NN.PER_LAYER + self.head == NNR.HEAD24 and len(self.head_states) == 6 and 0 < self.theta_deg < 90


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def fname(basis: str, item: str) -> str:
    return f"{basis}__{item.replace('/', '__')}"


def mask(idx: Any, n: int) -> Tensor:
    m = torch.zeros(n, dtype=torch.bool)
    m[sorted(idx)] = True
    return m


# ---- one item ------------------------------------------------------------------------------------------------------


@dataclass
class Item:
    net: Any  # F's network
    sites: list[RK.Site]
    toks: Tensor
    changed: Tensor
    flips: Tensor
    in_K: Tensor
    saved: dict[str, Tensor] | None = None  # num, den, dd of a saved write pass on the same rows (the transcription check)


def ranking(path: Path) -> tuple[Tensor, Tensor]:
    z = np.load(path)
    return torch.from_numpy(z["flips"]).long(), torch.from_numpy(z["in_K"]).bool()


def measure(it: Item, cfg: Config, N: list[int] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(it.changed)
    assert len(it.flips) == len(it.in_K) == n, f"saved scan ({len(it.flips)}) does not line up with the featurization ({n})"
    st = RK.sets(it.sites, it.flips, it.in_K, it.changed, cfg.top_frac, cfg.seed, N)
    V = RK.gather(it.sites, RK.accumulate(it.net, it.sites, RK.wanted(it.sites, st), list(it.toks[: cfg.windows].split(cfg.cache_batch))))
    rows = RK.read(it.sites, st, V, cfg.own_energy, cfg.rank_rtol)
    pin, pall = RK.pooled(rows, False), RK.pooled(rows, True)
    out = {"n_vars": n, "n_changed": int(it.changed.sum()), "size_K": int(it.in_K.sum()), "too_small": int(it.in_K.sum()) < cfg.too_small,
           "size_N": len(st.N), "size_R": len(st.R), "empty": not rows, "n_windows": min(cfg.windows, len(it.toks)), "window_len": it.toks.shape[1],
           "pooled": pin, "pooled_with_rem": pall, "nonrem_dd_share": pin["dd"] / pall["dd"] if pall["dd"] > 0 else None,
           "n_rank0": sum(r["dim_O"] == 0 for r in rows), "n_too_wide": sum(r["dim_S"] / r["d_out"] > cfg.too_wide for r in rows),
           "own_held_min": RK.own_held(V, cfg.own_energy), "dd_max_all": max((v.part(None) for v in V.values()), default=0.0)}  # fmt: skip
    if it.saved is not None:
        out["transcription"] = {k: rel(sum(r[k] for r in rows), float(it.saved[k][st.N].sum())) for k in ("num", "den", "dd")}
    return rows, out


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


def record(path: Path, rows: list[dict[str, Any]], out: dict[str, Any], cfg: Config, t0: float) -> None:
    keys = [k for k in (rows[0] if rows else {}) if k != "site"]
    arrays: dict[str, Any] = {k: np.array([r[k] for r in rows]) for k in keys}
    np.savez_compressed(RESULTS / "large" / f"items_{path.stem}.npz", **arrays)
    miss = RK.ordering_misses(rows, cfg.order_tol)
    out |= {"seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    if miss:
        stop(path, out, f"{path.stem}: ordering " + "; ".join(miss[:5]))
    dump(path, out)
    p, q = out["pooled"], out["pooled_with_rem"]
    log(f"{path.stem}: |K| {out['size_K']}, |N| {out['size_N']} ({p['n']} non-remainder); pooled strict {p['strict']:.6g}, own {p['own']:.6g}, "
        f"needed {p['needed']:.6g}, matched {p['matched']:.6g}, nu {p['nu']}; rotation {p['rotation']}, matched null {p['rotation_matched_null']}, "
        f"nu null {p['rotation_nu_null']}; with remainders strict {q['strict']:.6g}, own {q['own']:.6g}, needed {q['needed']:.6g}, "
        f"rotation {q['rotation']}; non-remainder dd share {out['nonrem_dd_share']}; {out['seconds']:.0f} s")  # fmt: skip


# ---- the two bases ---------------------------------------------------------------------------------------------------


class Vpd:
    name = "vpd"

    def __init__(self) -> None:
        self.w = KN.world()

    def item(self, item: str, kind: str) -> Item:
        if kind == "panel":
            F, M, toks, at, _ = panel_state(self.w, item)
            feat = KV.featurize(M, F, self.w.uv, self.w.delta, self.w.bases)
        else:
            F, feat, toks, at = KN.featurized(self.w, item)
        ch, saved = mask(feat.changed, feat.n), None
        if kind == "panel":
            kc = KN.load_config()
            nd = IX.needed_set(F, feat, toks, at, CFG.panel_vpd_derangements, kc.examples, kc.seed, kc.rows, kc.logit_rows, kc.cache_batch, log)
            flips, K = nd.scan.flips, nd.K
            del nd
        else:
            src = "x2" if item == "do_nothing" else item
            flips, K = ranking((E275 if kind == "e62" else KN.RESULTS / "large") / f"items_{KN.fname(src)}.npz")
            if item == CFG.control:
                saved = KN.saved(item, "built")
        return Item(F.net, RK.vpd_sites(feat), toks, ch, flips, K, saved)


class Nn:
    name = "nn"

    def __init__(self, stage: str) -> None:
        self.cfg, c229 = NNR.load_config(), NNR.load_e229()
        self.feat = NNR.E229.feat_of(c229, NNR.E229.load_pct(c229), self.cfg.band)
        self.world = NNR.Trained(self.cfg, c229) if stage == "trained" else NNR.Built(self.cfg, {"e62": "e62", "head24": "e62", "panel_nn": "panel"}.get(stage, "built"))

    def pair(self, F: Any, M: Any, toks: Tensor) -> tuple[list[RK.Site], Tensor]:
        p = rho.Pair(F, M)
        _, ch = NN.variables(p)
        return RK.neuron_sites(NN.sites(p, self.feat, toks.shape[1])), ch

    def item(self, item: str, kind: str) -> Item:
        F, M, toks, _, _ = self.world.make(item)
        sites, ch = self.pair(F, M, toks)
        z = np.load(NNR.RESULTS / "large" / f"items_{NNR.fname('x2' if item == 'do_nothing' else item)}.npz")
        saved = {k: torch.from_numpy(z[k]) for k in ("num", "den", "dd")} if item == CFG.control else None
        return Item(F.net, sites, toks, ch, torch.from_numpy(z["flips"]).long(), torch.from_numpy(z["in_K"]).bool(), saved)


def panel_state(w: Any, item: str) -> tuple[Any, ...]:
    """E272's state as `needed_neurons.Built.panel_state` reads it, through the VPD world's from_dw."""
    e, (C, th) = NNR.panel_items(NNR.load_a1())[item], NNR.e272()
    toks = NNR.EB.load_windows(C, th, e["task"])
    f = NNR.EB.random_file(e["state"], C) if e["random"] else NNR.EB.RUN.REPO / e["state"]
    return w.from_dw(f, toks.cuda(), NNR.EB.KP.LAST)


def read_item(basis: Any, item: str, kind: str, cfg: Config) -> dict[str, Any]:
    path = RESULTS / "items" / f"{fname(basis.name, item)}.json"
    if path.is_file() and "stopped" not in (done := json.loads(path.read_text())):
        log(f"{path.stem}: done before, skipped")
        return done
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    it = basis.item(item, kind)
    rows, out = measure(it, cfg)
    out = {"experiment": EXPERIMENT, "item": item, "basis": basis.name, "kind": kind} | out
    record(path, rows, out, cfg, t0)
    del it
    torch.cuda.empty_cache()
    return out


# ---- control stage ---------------------------------------------------------------------------------------------------


def rotated(M: Any, l: int, i: int, j: int, theta: float) -> Any:
    """M with neuron i's output column at layer l turned by theta toward column j, its norm kept."""
    F = KE.clone(M)
    W = F.net.h[l].mlp.down_proj.weight
    wi, wj = W[:, i].double(), W[:, j].double()
    e = wi / wi.norm()
    t = wj - (wj @ e) * e
    with torch.no_grad():
        W[:, i] = (wi.norm() * (math.cos(theta) * e + math.sin(theta) * t / t.norm())).to(W.dtype)
    return F


def pick(it: Item, cfg: Config) -> tuple[int, int, int, int]:
    """Layer l, neuron i (the most-flipping neuron of R), j (the next neuron of R at layer l) and k (the fewest-flipping
    neuron of layer l outside K, else outside R), from x2's ranking at the neuron basis."""
    st = RK.sets(it.sites, it.flips, it.in_K, torch.ones_like(it.changed), cfg.top_frac, cfg.seed)
    neuron = lambda g: NN.H <= g % NN.PER_LAYER < NN.H + NN.NEUR
    R = sorted((g for g in st.R if neuron(g)), key=lambda g: (-int(it.flips[g]), g))
    for g in R:
        l = g // NN.PER_LAYER
        same = [h for h in R if h != g and h // NN.PER_LAYER == l]
        if same:
            pool = [l * NN.PER_LAYER + NN.H + c for c in range(NN.NEUR) if l * NN.PER_LAYER + NN.H + c not in st.R]
            k = min(pool, key=lambda h: (bool(it.in_K[h]), int(it.flips[h]), h))
            return l, g % NN.PER_LAYER - NN.H, same[0] % NN.PER_LAYER - NN.H, k % NN.PER_LAYER - NN.H
    raise SystemExit("STOP: no two neurons of R share a layer")


def rotation_controls(nn: Nn, cfg: Config, x2: Item, out: dict[str, Any]) -> list[str]:
    l, i, j, k = pick(x2, cfg)
    _, M, toks, _, _ = nn.world.make(cfg.control)
    th, miss, g = math.radians(cfg.theta_deg), [], l * NN.PER_LAYER + NN.H + i
    want_own = (1 - math.cos(th)) / 2
    for name, to in (("rot_needed", j), ("rot_other", k)):
        F = rotated(M, l, i, to, th)
        sites, ch = nn.pair(F, M, toks)
        rows, rec = measure(Item(F.net, sites, toks, ch, x2.flips, x2.in_K), cfg, N=[g])
        r = rows[0]
        rec |= {"layer": l, "i": i, "toward": to, "theta_deg": cfg.theta_deg, "own_share": r["own"] / r["dd"], "own_expected": want_own,
                "rotation": rec["pooled"]["rotation"]}  # fmt: skip
        out[name] = rec
        log(f"{name}: L{l} n{i} toward n{to}: own {rec['own_share']!r} (expected {want_own!r}), rotation {rec['rotation']!r}, "
            f"matched null {rec['pooled']['rotation_matched_null']!r}, nu null {rec['pooled']['rotation_nu_null']!r}")  # fmt: skip
        miss += [f"{name}: own {rec['own_share']!r} off {want_own!r}"] if abs(rec["own_share"] - want_own) > 1e-6 else []
        rot = rec["rotation"]
        if name == "rot_needed" and not (rot is not None and rot >= 1 - 1e-6):
            miss.append(f"rot_needed: rotation {rot!r} below 1 - 1e-6")
        if name == "rot_other" and not (rot is not None and rot <= cfg.rot_other_max):
            miss.append(f"rot_other: rotation {rot!r} above {cfg.rot_other_max}")
        del F
    return miss


def control(cfg: Config, tag: str) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    t0, out, miss = time.time(), {"experiment": EXPERIMENT, "card": torch.cuda.get_device_name()}, []
    for make in (Vpd, lambda: Nn("built")):
        basis = make()
        for item in ("do_nothing", cfg.control):
            it = basis.item(item, "built")
            rows, rec = measure(it, cfg)
            key = f"{basis.name}/{item}"
            out[key] = rec
            log(f"{key}: {json.dumps(rec)}")
            miss += [f"{key}: {m}" for m in RK.ordering_misses(rows, cfg.order_tol)]
            if item == "do_nothing":
                p = rec["pooled"]
                miss += [f"{key}: {k} {v!r}" for k, v in (("n_changed", rec["n_changed"]), ("size_N", rec["size_N"]), ("dd_max_all", rec["dd_max_all"])) if v != 0]
                miss += [f"{key}: pooled {k} {p[k]!r}" for k in ("strict", "own", "needed", "matched") if p[k] != 1.0]
                miss += [f"{key}: own spaces hold {rec['own_held_min']!r}"] if rec["own_held_min"] < cfg.own_energy - 1e-12 else []
            else:
                miss += [f"{key}: {k} {v!r} off the saved arrays" for k, v in rec["transcription"].items() if v > cfg.transcription_rel_tol]
                if isinstance(basis, Nn):
                    neu = [r for r in rows if not r["rem"] and r["dd"] > 0]
                    worst = min((r["own"] / r["dd"] for r in neu), default=None)
                    rec["own_share_min_neuron"] = worst
                    miss += [f"{key}: a changed neuron's own share {worst!r} below 1 - 1e-9"] if worst is None or worst < 1 - 1e-9 else []
                    miss += [f"{key}: pooled own {rec['pooled']['own']!r} below 0.99"] if rec["pooled"]["own"] < 0.99 else []
                    if not miss:
                        miss += rotation_controls(basis, cfg, it, out)
            del it
            torch.cuda.empty_cache()
        del basis
    out["seconds"] = time.time() - t0
    if miss:
        stop(path, out, "control: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log(f"CONTROL DONE in {out['seconds']:.0f} s")


# ---- main ------------------------------------------------------------------------------------------------------------


def items(stage: str, basis: Any) -> list[tuple[str, str]]:
    if stage == "built":
        return [(i, "built") for i in CFG.built[1:]]
    if stage == "e62":
        return [(i, "e62") for i in NNR.load_a1().e62]
    if stage in ("trained", "panel_nn"):
        return [(i, "built") for i in basis.world.items]
    return [(k, "panel") for k, e in NNR.panel_items(NNR.load_a1()).items() if e["seed"] == 0]  # panel_vpd


def head24(nn: Nn, item: str, cfg: Config) -> None:
    path = RESULTS / "items" / f"{fname('head24', item)}.json"
    if path.is_file() and "stopped" not in json.loads(path.read_text()):
        log(f"{path.stem}: done before, skipped")
        return
    t0, kc = time.time(), KN.load_config()
    F, M, toks, at, _ = nn.world.make(item)
    copy, _ = HE.targets(toks)
    tc = toks[copy.to(toks.device)]
    run = NN.cache(F.net, tc, at, kc.cache_batch)
    ex, pt = IX.derangements(len(tc), kc.derangements, min(kc.examples, len(tc)), kc.seed)
    sc = NN.scan(F.net, nn.feat, run, at, ex, pt, IX.rows_per_forward(kc.rows, kc.logit_rows, run.top.shape[1]), log)
    sites, ch = nn.pair(F, M, tc)
    u = NNR.HEAD24
    st = RK.Sets([u], [], {si: [] for si in range(len(sites))})
    V = RK.gather(sites, RK.accumulate(F.net, sites, RK.wanted(sites, st), list(tc.split(cfg.cache_batch))))
    r = RK.read(sites, st, V, cfg.own_energy, cfg.rank_rtol)[0]
    out = {"experiment": EXPERIMENT, "item": item, "head": "L2.h4", "n_copy_windows": len(tc), "changed": bool(ch[u]), "dd": r["dd"],
           "strict": r["strict"] / r["dd"] if r["dd"] > 0 else None, "own": r["own"] / r["dd"] if r["dd"] > 0 else None, "dim_O": r["dim_O"],
           "null": r["dim_O"] / r["d_out"], "flips": int(sc.flips[u]), "n_pairs": len(ex), "seconds": time.time() - t0}  # fmt: skip
    dump(path, out)
    log(f"head24 {item}: strict {out['strict']}, own {out['own']} (null {out['null']:.4f}, dim O {out['dim_O']}), flips {out['flips']} of {len(ex)}; {out['seconds']:.0f} s")
    del F, run
    torch.cuda.empty_cache()


def summary() -> None:
    every = [json.loads(p.read_text()) for p in sorted((RESULTS / "items").glob("*.json"))]
    s0 = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("stage0_*.json"))]
    dump(RESULTS / "summary.json", {"experiment": EXPERIMENT, "stage0": s0, "items": {f"{r.get('basis', 'head24')}/{r['item']}": r for r in every}})
    log(f"summary: {len(every)} items, {len(s0)} stage-0 files")


CFG = load_config()


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    if stage == "summary":
        summary()
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E268 and E277
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}; settings {asdict(CFG)}")
    for d in ("items", "large"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    if stage == "gauge":  # its own checks, independent of the control stage (gauge.py)
        return load_experiment("rotation_kept", "gauge").main(CFG, tag, shard)
    if stage in ("gauge_pool", "gauge_sub"):  # added 2026-11-24 (gauge_follow.py)
        return load_experiment("rotation_kept", "gauge_follow").main(stage, CFG, tag, shard)
    control(CFG, tag)
    if stage == "control":
        return 0
    makers = {"built": (Vpd, Nn), "e62": (Vpd, Nn), "panel_vpd": (Vpd,)}.get(stage, (Nn,))
    for make in makers:
        basis = Vpd() if make is Vpd else Nn(stage)
        todo = (list(CFG.head_states) if stage == "head24" else items(stage, basis))[shard[0] :: shard[1]]
        log(f"stage {stage} at {basis.name}: {len(todo)} items")
        for n, x in enumerate(todo):
            if stage == "head24":
                assert isinstance(basis, Nn)
                head24(basis, str(x), CFG)
            else:
                read_item(basis, x[0], x[1], CFG)
            log(f"ITEM {n + 1} of {len(todo)} DONE ({x})")
        del basis
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("control", "built", "e62", "head24", "trained", "panel_nn", "panel_vpd", "gauge", "gauge_pool", "gauge_sub", "summary"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: the control stage writes results/stage0_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

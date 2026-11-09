"""steering_copy_lookup (E135, cell P2.5): does steering separate copy from lookup as widely as the share, once it has
per-key vectors and a tuned strength? Plan: reports/plans/2026-11-09_plan_e135-steering-copy-lookup.md. The filed rule,
thresholds and expected values: results/predictions.json (the run refuses unless it is committed unchanged).

On the six saved updates of E62's new-marker pair (induction_cue/{copy,lookup}_s{0,1,2}.pt, delta-W at step 30) and its
200 read windows (even indices = fit half, odd = score half). Steering adds a vector to block l's input at the answer
position of the original model (W0): a pooled vector (mean over the fit half of tuned minus W0 there) and a per-key
vector (the same mean over the fit windows of the window's key, the token at half - 1). Every number is written in
log-odds (the share's unit) and log P. The share is E62's `r`, by the same path as E62 (`keydep.route_windows`).

Stage 0, the control, alone: a zero update reads rise exactly 0 through the whole steering path, strength 0 on copy_s0
reads exactly 0, and `r` on all 200 windows reproduces E62's run.json. Stage 1: selection on the fit half, the steering
reads, the share on the score windows, random vectors, cross-application, cosines, the filed verdict.

Usage (repo root, a GPU node): cluster/steering_copy_lookup.sbatch. Writes results/stage0.json, results/stage1.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import copying as CP
from commute import elicit as EL
from commute import keydep as KD
from commute import steering as ST
from commute.data import LAST
from commute.jobs import hard_exit, require_filed
from commute.local_faith import MIN_GAIN, route_cells
from commute.open_pair import jsonable
from commute.revival_kv import kl_rows

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, STAGE0, STAGE1 = RESULTS / "predictions.json", RESULTS / "stage0.json", RESULTS / "stage1.json"
E62_DIR = HERE.parent / "induction_cue" / "results"
SIDES, SEEDS, KINDS = ("copy", "lookup"), (0, 1, 2), ("pooled", "perkey")
NAMES = tuple(f"{m}_s{s}" for m in SIDES for s in SEEDS)
LAYERS = tuple(range(ST.N_LAYERS))
FIT, SCORE = slice(0, None, 2), slice(1, None, 2)  # as `steering.Task.part`
log = KD.log


class Stopped(Exception):
    """A filed stop: the stage's file records why; the job exits 1."""


@dataclass(frozen=True)
class Cfg:
    half: int
    n: int
    strengths: tuple[float, ...]
    n_random: int
    random_seed: int

    def __post_init__(self) -> None:
        assert self.n == 2 * self.half + 1 and self.strengths[0] == 0 and list(self.strengths) == sorted(self.strengths), "predictions.json disagrees with the code"


@dataclass
class Pair:
    toks: Tensor  # (200, 41)
    keys: Tensor  # (200,) index into the pool of each window's key
    n_keys: int
    tgt: dict[str, Tensor]  # side -> (200,)
    half: int


def write(path: Path, out: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))


def halt(path: Path, out: dict[str, Any], why: str) -> Stopped:
    out["stopped"] = why
    write(path, out)
    return Stopped(why)


# ----------------------------------------------------------------------------- the pure rules (dry-run on stand-ins)


def pick(rises: dict[tuple[int, float], float]) -> tuple[int, float] | None:
    """The filed selection: the largest fit-half rise, ties to the lower layer, then the lower strength. A nan among
    the rises leaves the setting undefined (None), so it cannot be silently ordered."""
    if any(math.isnan(v) for v in rises.values()):
        return None
    return max(rises, key=lambda ls: (rises[ls], -ls[0], -ls[1]))


def key_index(toks: Tensor, keys: Tensor, half: int) -> Tensor | None:
    """(N,) the pool index of each window's key (the token at half - 1); None if some window's key is not in the pool."""
    hits = toks[:, half - 1, None] == keys[None]
    return hits.long().argmax(1) if bool((hits.sum(1) == 1).all()) else None


def key_gap_problem(idx: Tensor, n_keys: int) -> str | None:
    """Why the per-key vectors cannot be built from the fit half, or None: every key needs a fit window."""
    fit = torch.bincount(idx[FIT], minlength=n_keys)
    missing = (fit == 0).nonzero().flatten().tolist()
    return f"keys with no fit-half window: {missing}" if missing else None


def per_key_mean(diff: Tensor, idx: Tensor, n_keys: int) -> Tensor:
    """(n_keys, d): the mean of diff (N, d) over each key's windows. `key_gap_problem` has ruled out an empty key."""
    out = torch.zeros(n_keys, diff.shape[1], device=diff.device, dtype=diff.dtype).index_add_(0, idx, diff)
    return out / torch.bincount(idx, minlength=n_keys).to(diff.dtype)[:, None]


def per_window(v: Tensor, keys: Tensor) -> Tensor:
    """A pooled vector (d,) as it is (it broadcasts over windows); a per-key vector (K, d) indexed by each window's key."""
    return v if v.ndim == 1 else v[keys]


def random_like(v: Tensor, gen: torch.Generator) -> Tensor:
    """A Gaussian direction with v's shape, scaled to v's norm (row by row for a per-key vector)."""
    g = torch.randn(v.shape, generator=gen).to(v.device, v.dtype)
    return g / g.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)


def verdict(states: dict[str, dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    """The filed rule. Order: unreadable (a gain below MIN_GAIN or any nan among the six), then inert (no call), then the
    edge. The gaps are signed copy minus lookup; steering's gap is the larger of the two kinds."""
    col = {"share": lambda s: s["r_score"], "pooled": lambda s: s["steering"]["pooled"]["share_lo"],
           "perkey": lambda s: s["steering"]["perkey"]["share_lo"]}  # fmt: skip
    mean = lambda side, f: statistics.fmean(f(states[f"{side}_s{s}"]) for s in SEEDS)
    gaps = {k: mean("copy", f) - mean("lookup", f) for k, f in col.items()}
    steering = max(gaps["pooled"], gaps["perkey"])
    edge = gaps["share"] - steering
    steer_vals = [col[k](states[n]) for n in NAMES for k in KINDS]
    out = {"gap_share": gaps["share"], "gap_pooled": gaps["pooled"], "gap_perkey": gaps["perkey"], "gap_steering": steering, "edge": edge}
    low = [n for n in NAMES if not states[n]["G"]["lo"] >= th["min_gain"]]
    nans = [n for n in NAMES if any(math.isnan(f(states[n])) for f in col.values())]
    if low or nans:
        return {**out, "verdict": "unreadable", "low_gain": low, "nan_states": nans}
    if all(v < th["inert"] for v in steer_vals):
        return {**out, "verdict": "no call: steering inert"}
    return {**out, "verdict": "rival A: the share keeps an edge" if edge >= th["edge"] else "rival B: no edge"}


def marks(states: dict[str, dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    """Reported, not stops: each side's mean random-vector share (per seed, then over seeds), per kind, within the band."""
    means = {k: {side: statistics.fmean(states[f"{side}_s{s}"]["steering"][k]["random"]["mean_lo"] for s in SEEDS) for side in SIDES} for k in KINDS}
    return {"random_mean_share": means, "random_within_band": {k: {side: bool(abs(v) <= th["random_band"]) for side, v in by.items()} for k, by in means.items()}}


# ----------------------------------------------------------------------------- reads


def reads(net: ST.Net, toks: Tensor, tgt: Tensor, half: int, dW: ST.DW | None = None, edit: ST.Edit | None = None) -> tuple[Tensor, Tensor]:
    """(log-odds, log P) of the target at the answer position, per window, on CPU in float64."""
    with net.at(dW):
        lg = net.run(toks, edit)[0]
    return ST.logodds_at(lg, half, tgt), ST.logp_at(lg, half, tgt)


def steer(v: Tensor, layer: int, s: float, keys: Tensor, half: int) -> ST.Edit:
    return {f"in{layer}": ST.add_at(half, s * per_window(v, keys.to(v.device)))}


def vectors(net: ST.Net, p: Pair, dW: ST.DW) -> dict[str, dict[int, Tensor]]:
    """Per layer, tuned minus W0 at block l's input at the answer position, on the fit half: pooled (d,) and per key (K, d)."""
    f, idx = p.toks[FIT], p.keys[FIT].to(net.device)
    with net.at(dW):
        c1 = net.run(f, capture=True)[1]
    c0 = net.run(f, capture=True)[1]
    diff = {l: c1[f"in{l}"][:, p.half] - c0[f"in{l}"][:, p.half] for l in LAYERS}
    return {"pooled": {l: d.mean(0) for l, d in diff.items()}, "perkey": {l: per_key_mean(d, idx, p.n_keys) for l, d in diff.items()}}


def grid_rises(net: ST.Net, p: Pair, side: str, v: dict[int, Tensor], lo0: Tensor, part: slice, strengths: tuple[float, ...]) -> dict[tuple[int, float], float]:
    """(layer, strength) -> mean log-odds rise over W0 on the windows of `part`."""
    keys = p.keys[part]
    return {(l, s): float((reads(net, p.toks[part], p.tgt[side][part], p.half, edit=steer(v[l], l, s, keys, p.half))[0] - lo0).mean()) for l in LAYERS for s in strengths}


@torch.no_grad()
def off_kl(net: ST.Net, toks: Tensor, half: int, dW: ST.DW | None = None, edit: ST.Edit | None = None) -> float:
    """Mean KL(W0 || edited or state) over the positions other than the answer (`revival_kv.off_answer_kl`'s form)."""
    l0 = torch.log_softmax(net.run(toks)[0].double(), -1)
    with net.at(dW):
        l1 = torch.log_softmax(net.run(toks, edit)[0].double(), -1)
    kl = kl_rows(l0, l1)
    keep = torch.ones(kl.shape[1], dtype=torch.bool, device=kl.device)
    keep[half] = False
    return float(kl[:, keep].mean())


def read_rec(net: ST.Net, p: Pair, side: str, w0: dict[str, dict[str, tuple[Tensor, Tensor]]], edit: ST.Edit, g: dict[str, float]) -> dict[str, float]:
    """Rise over W0 and rise / gain, on the score windows, for `side`'s targets, in both units."""
    lo, lp = reads(net, p.toks[SCORE], p.tgt[side][SCORE], p.half, edit=edit)
    lo0, lp0 = w0[side]["score"]
    return {"rise_lo": float((lo - lo0).mean()), "rise_lp": float((lp - lp0).mean()), "share_lo": ST.share(lo, lo0, g["lo"]), "share_lp": ST.share(lp, lp0, g["lp"])}


def kind_record(net: ST.Net, p: Pair, c: Cfg, side: str, other: str, v: dict[int, Tensor], w0: dict[str, Any], g: dict[str, float], g_other: dict[str, float], gen: torch.Generator) -> dict[str, Any]:
    """One vector kind on one state: the filed selection on the fit half; at that setting the score reads, the
    off-target KL, the random controls and the cross-application to the other side's targets."""
    rises = grid_rises(net, p, side, v, w0[side]["fit"][0], FIT, c.strengths)
    setting = pick(rises)
    rec: dict[str, Any] = {"grid_fit": {f"{l},{s}": r for (l, s), r in rises.items()}, "layer": None, "strength": None}
    if setting is None:
        return {**rec, "share_lo": math.nan, "share_lp": math.nan, "random": {"share_lo": [], "share_lp": [], "mean_lo": math.nan, "mean_lp": math.nan}}
    l, s = setting
    keys = p.keys[SCORE]
    rec.update(layer=l, strength=s, fit_rise=rises[setting], off_kl=off_kl(net, p.toks[SCORE], p.half, edit=steer(v[l], l, s, keys, p.half)))
    rec.update(read_rec(net, p, side, w0, steer(v[l], l, s, keys, p.half), g))
    rec["cross"] = read_rec(net, p, other, w0, steer(v[l], l, s, keys, p.half), g_other)
    draws = [read_rec(net, p, side, w0, steer(random_like(v[l], gen), l, s, keys, p.half), g) for _ in range(c.n_random)]
    rec["random"] = {"share_lo": [d["share_lo"] for d in draws], "share_lp": [d["share_lp"] for d in draws], "mean_lo": statistics.fmean(d["share_lo"] for d in draws), "mean_lp": statistics.fmean(d["share_lp"] for d in draws)}
    return rec


def cosines(pooled: dict[str, dict[int, Tensor]]) -> dict[int, dict[str, Any]]:
    """Per layer: the 6 x 6 cosine matrix of the pooled vectors (NAMES order), and the means for copy against lookup and
    across seeds within a side."""
    out = {}
    for l in LAYERS:
        m = torch.nn.functional.normalize(torch.stack([pooled[n][l].double() for n in NAMES]), dim=1)
        C = (m @ m.T).cpu()
        cross, cp, lk = C[:3, 3:], C[:3, :3], C[3:, 3:]
        off = lambda M: float(M[~torch.eye(3, dtype=torch.bool)].mean())
        out[l] = {"names": list(NAMES), "matrix": C.tolist(), "copy_vs_lookup": float(cross.mean()), "within_copy": off(cp), "within_lookup": off(lk)}
    return out


# ----------------------------------------------------------------------------- stages


def stage0(ctx: KD.Ctx, c: Cfg, p: Pair, states: dict[str, ST.DW], route_toks: dict[str, Tensor], run62: dict[str, Any], th: dict[str, float], head: dict[str, Any]) -> None:
    net, out = ctx.net, dict(head)
    w0 = baselines(net, p)
    zero = {k: torch.zeros_like(x) for k, x in net.W0.items()}
    vs0 = vectors(net, p, zero)
    out["a_zero_update_max_rise"] = max(abs(r) for k in KINDS for part, nm in ((FIT, "fit"), (SCORE, "score")) for r in grid_rises(net, p, "copy", vs0[k], w0["copy"][nm][0], part, c.strengths).values())
    log(f"[0a] zero update, largest |rise| over layers, strengths, kinds and halves: {out['a_zero_update_max_rise']}")
    if not out["a_zero_update_max_rise"] <= th["zero_tol"]:
        raise halt(STAGE0, out, f"a zero update reads a rise of {out['a_zero_update_max_rise']}")
    vs = vectors(net, p, states["copy_s0"])
    out["b_strength0_max_rise"] = max(abs(r) for k in KINDS for r in grid_rises(net, p, "copy", vs[k], w0["copy"]["score"][0], SCORE, (0.0,)).values())
    out["b_off_kl_strength0"] = {k: off_kl(net, p.toks[SCORE], p.half, edit=steer(vs[k][0], 0, 0.0, p.keys[SCORE], p.half)) for k in KINDS}
    log(f"[0b] strength 0 on copy_s0, largest |rise|: {out['b_strength0_max_rise']}; off-target KL at strength 0 {out['b_off_kl_strength0']}")
    if not out["b_strength0_max_rise"] <= th["zero_tol"]:
        raise halt(STAGE0, out, f"strength 0 reads a rise of {out['b_strength0_max_rise']}")
    base = {m: route_cells(net.model, route_toks[m], p.half, LAST, KD.BATCH, c.n) for m in SIDES}
    out["c_r200"] = {n: reproduce(ctx, c, p, route_toks, base, states[n], n, run62) for n in NAMES}
    out["c_gap_share_200"] = statistics.fmean(out["c_r200"][f"copy_s{s}"]["r"] for s in SEEDS) - statistics.fmean(out["c_r200"][f"lookup_s{s}"]["r"] for s in SEEDS)
    misses = [n for n, r in out["c_r200"].items() if not abs(r["r"] - r["r_e62"]) <= th["r_tol"]]
    if misses:
        raise halt(STAGE0, out, f"r does not reproduce E62's on {misses}")
    write(STAGE0, out)
    print("STAGE 0 DONE", flush=True)


def reproduce(ctx: KD.Ctx, c: Cfg, p: Pair, route_toks: dict[str, Tensor], base: dict[str, Any], dW: ST.DW, name: str, run62: dict[str, Any]) -> dict[str, float]:
    side = name.split("_")[0]
    rd = KD.route_windows(ctx, route_toks[side], p.half, LAST, dW, base[side], c.n)
    out = {"r": rd["r"], "r_e62": run62["states"][name]["r"], "G_route": rd["G_logodds"]}
    log(f"[0c {name}] r on 200 windows {out['r']:.4f} (E62's run.json {out['r_e62']:.4f})")
    return out


def baselines(net: ST.Net, p: Pair) -> dict[str, dict[str, tuple[Tensor, Tensor]]]:
    """W0's (log-odds, log P) per side and half."""
    return {m: {nm: reads(net, p.toks[part], p.tgt[m][part], p.half) for nm, part in (("fit", FIT), ("score", SCORE))} for m in SIDES}


def gains(net: ST.Net, p: Pair, states: dict[str, ST.DW], w0: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Each state's own gain over W0 on the score windows, on its own side's targets, in both units."""
    out = {}
    for n, dW in states.items():
        m = n.split("_")[0]
        lo, lp = reads(net, p.toks[SCORE], p.tgt[m][SCORE], p.half, dW)
        out[n] = {"lo": float((lo - w0[m]["score"][0]).mean()), "lp": float((lp - w0[m]["score"][1]).mean())}
    return out


def stage1(ctx: KD.Ctx, c: Cfg, p: Pair, states: dict[str, ST.DW], route_toks: dict[str, Tensor], th: dict[str, float], head: dict[str, Any]) -> None:
    net, out = ctx.net, {**head, "states": {}}
    w0 = baselines(net, p)
    G = gains(net, p, states, w0)
    base = {m: route_cells(net.model, route_toks[m][SCORE], p.half, LAST, KD.BATCH, c.n) for m in SIDES}
    pooled: dict[str, dict[int, Tensor]] = {}
    for i, (name, dW) in enumerate(states.items()):
        side, seed = name.split("_")
        other = "lookup" if side == "copy" else "copy"
        vs = vectors(net, p, dW)
        pooled[name] = vs["pooled"]
        gen = torch.Generator().manual_seed(c.random_seed + i)
        rd = KD.route_windows(ctx, route_toks[side][SCORE], p.half, LAST, dW, base[side], c.n)
        rec: dict[str, Any] = {"G": G[name], "r_score": rd["r"], "G_route": rd["G_logodds"], "cells": rd["cells"], "off_kl_state": off_kl(net, p.toks[SCORE], p.half, dW=dW),
                               "steering": {k: kind_record(net, p, c, side, other, vs[k], w0, G[name], G[f"{other}_{seed}"], gen) for k in KINDS}}
        out["states"][name] = rec
        log(f"[1 {name}] gain {G[name]} r_score {rd['r']:.4f} off_kl_state {rec['off_kl_state']:.4e}")
        for k in KINDS:
            s = rec["steering"][k]
            log(f"[1 {name} {k}] layer {s['layer']} strength {s['strength']} share_lo {s['share_lo']} share_lp {s['share_lp']} off_kl {s.get('off_kl')} cross {s.get('cross')} random mean {s.get('random', {}).get('mean_lo')}")
        write(STAGE1, out)
    out["cosines"] = cosines(pooled)
    out["score"] = {**verdict(out["states"], th), **marks(out["states"], th)}
    write(STAGE1, out)
    log(f"[score] {json.dumps(jsonable(out['score']))}")
    print("STAGE 1 DONE", flush=True)


def load_pair(c: Cfg, large: Path, run62: dict[str, Any]) -> tuple[Pair, dict[str, Tensor]]:
    """The saved read windows; the per-window target as a last column for the route split (`Cue.read_windows`)."""
    assert KD.sha256(large / "windows.pt") == run62["windows_sha256"], "windows.pt is not the one E62 recorded"
    w = torch.load(large / "windows.pt", weights_only=False)
    keys = torch.tensor(w["pools"]["keys"])
    idx = key_index(w["read"].tokens, keys, c.half)
    assert idx is not None, "a window's key is not in the pool"
    problem = key_gap_problem(idx, len(keys))
    assert problem is None, problem
    assert w["read"].half == c.half and w["read"].tokens.shape == (200, c.n)
    tgt = {m: w["target"][m]["read"] for m in SIDES}
    return Pair(w["read"].tokens, idx, len(keys), tgt, c.half), {m: CP.with_target(w["read"], tgt[m]).tokens for m in SIDES}


def load_states(large: Path, run62: dict[str, Any]) -> dict[str, ST.DW]:
    out = {}
    for n in NAMES:
        rec, f = run62["states"][n], large / f"{n}.pt"
        assert KD.sha256(f) == rec["sha256"] and rec["step"] == 30, f"{f} is not the one E62 recorded at step 30"
        out[n] = torch.load(f, weights_only=True, map_location="cpu")
        assert abs(EL.summed_norm(out[n]) - rec["norm"]) <= 1e-6, f"{n}: the summed norm differs from run.json's delta-W norm"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--large", type=Path, default=E62_DIR / "large")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    a = ap.parse_args()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    s = P["settings"]
    c, th = Cfg(s["half"], s["n"], tuple(s["strengths"]), s["n_random"], s["random_seed"]), P["thresholds"]
    assert th["min_gain"] == MIN_GAIN, "predictions.json disagrees with local_faith.MIN_GAIN"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    large = a.large.resolve()
    run62 = json.loads((E62_DIR / "run.json").read_text())
    p, route_toks = load_pair(c, large, run62)
    states = load_states(large, run62)
    ctx = KD.make_ctx(a.device, a.alive, P, large)
    head = {"predictions_sha256": KD.sha256(PRED), "host": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}
    log(f"[host] {head['host']}")
    try:
        stage0(ctx, c, p, states, route_toks, run62, th, head)
        stage1(ctx, c, p, states, route_toks, th, head)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())

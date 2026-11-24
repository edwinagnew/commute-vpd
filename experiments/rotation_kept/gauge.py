"""E286 stage `gauge` (cell P1.60; whip-lead 2026-11-24, from redteam-lead's note
~/commute-data/agents/whip-lead/note-2026-11-24-redteam-p271-lit-challenges.md item 4). Run through
`run.py --stage gauge`; settings and filed values in predictions.json's "gauge" block (added before this stage ran).

(i) M with a random orthogonal gauge per head (`commute.head_gauge.rotate`: QK turned per rotary pair, VO by a random
    orthogonal S) computes M's function. First its logits on x2's D against M's, within gauge_floor_mult times the
    floor (the logit deviation of the round trip, the gauge then its transpose). Then the three shares (`measure`, x2's
    ranking) at the neuron-and-head basis and at VPD, with each basis's change at the scale of M's writes:
    max dd / den over the read set, and 1 - sum (dd - part) / sum den for the strict and needed parts.
(ii) Per trained state (E62's nine, E229's twenty, the panel's 36 trained states), the share of the update's squared
    norm on the head matrices that lies along the gauge's tangent at M (`head_gauge.tangent_share`), beside the same for
    a random update matched per tensor (`needed_neurons.matched_random`).
results/gauge_<tag>.json for (i); results/items/gauge__<class>__<item>.json for (ii), skipped on a rerun.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import head_gauge as HG
from commute import kstar_vpd as KV
from commute import rotation_kept as RK
from commute.jobs import load_experiment

RUN: Any = load_experiment("rotation_kept", "run")
NNR, KN = RUN.NNR, RUN.KN
G = json.loads(RUN.PRED.read_text())["gauge"]["settings"]
log, dump, stop = RUN.log, RUN.dump, RUN.stop


@torch.no_grad()
def logits(F: Any, toks: Tensor, batch: int) -> Tensor:
    return torch.cat([F.net(t)[0].double() for t in toks.split(batch)])


def at_scale(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r = [x for x in rows if not x["rem"]]
    den = sum(x["den"] for x in r)
    if not r or den <= 0:
        return {"max_rel_change": None, "kept_strict_scale": None, "kept_needed_scale": None}
    return {"max_rel_change": max(x["dd"] / x["den"] if x["den"] > 0 else float("inf") for x in r),
            "kept_strict_scale": 1 - sum(x["dd"] - x["strict"] for x in r) / den,
            "kept_needed_scale": 1 - sum(x["dd"] - x["needed"] for x in r) / den}  # fmt: skip


def basis_read(F: Any, sites: list[Any], toks: Tensor, ch: Tensor, cfg: Any) -> dict[str, Any]:
    flips, K = RUN.ranking(NNR.RESULTS / "large" / "items_x2.npz") if isinstance(sites[0], (RK.Heads, RK.Neurons)) else RUN.ranking(
        KN.RESULTS / "large" / f"items_{KN.fname('x2')}.npz")  # fmt: skip
    rows, rec = RUN.measure(RUN.Item(F.net, sites, toks, ch, flips, K), cfg)
    rec |= at_scale(rows)
    rec["changed_kinds"] = sorted({r["site"].rsplit(".", 1)[-1] for r in rows})
    return rec


def invariance(cfg: Any, tag: str) -> None:
    path = RUN.RESULTS / f"gauge_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    t0, out, miss = time.time(), {"experiment": RUN.EXPERIMENT, "card": torch.cuda.get_device_name()}, []
    nn = RUN.Nn("built")
    _, M, toks, _, _ = nn.world.make(cfg.control)
    Fg, used = HG.rotate(M, G["seed"])
    back, _ = HG.rotate(Fg, G["seed"], inverse_of=used)
    lM = logits(M, toks, cfg.cache_batch)
    dev, floor = float((logits(Fg, toks, cfg.cache_batch) - lM).abs().max()), float((logits(back, toks, cfg.cache_batch) - lM).abs().max())
    out["invariance"] = {"max_logit_dev": dev, "round_trip_floor": floor, "bar": G["floor_mult"] * floor, "logit_scale": float(lM.abs().max())}
    log(f"gauge invariance: max logit deviation {dev!r}, round-trip floor {floor!r}, bar {G['floor_mult'] * floor!r}")
    if not dev <= G["floor_mult"] * floor:
        stop(path, out, f"gauge: F's logits {dev!r} off M's, over {G['floor_mult']} x the round-trip floor {floor!r}")
    del back
    sites, ch = nn.pair(Fg, M, toks)
    out["nn"] = basis_read(Fg, sites, toks, ch, cfg)
    del nn, Fg
    v = RUN.Vpd()
    _, Mv, toksv, _, _ = v.w.make(cfg.control)
    Fv, _ = HG.rotate(Mv, G["seed"])
    feat = KV.featurize(Mv, Fv, v.w.uv, v.w.delta, v.w.bases)
    out["vpd"] = basis_read(Fv, RK.vpd_sites(feat), toksv, RUN.mask(feat.changed, feat.n), cfg)
    for b in ("nn", "vpd"):
        r = out[b]
        log(f"gauge {b}: pooled {json.dumps(r['pooled'])}; max dd/den {r['max_rel_change']!r}, kept at write scale strict "
            f"{r['kept_strict_scale']!r}, needed {r['kept_needed_scale']!r}; changed kinds {r['changed_kinds']}")  # fmt: skip
    nnr = out["nn"]
    out["nn_reads_1"] = nnr["max_rel_change"] is not None and nnr["max_rel_change"] <= G["scale_tol"] and nnr["kept_strict_scale"] >= 1 - G["scale_tol"]
    log(f"gauge: heads basis reads 1: {out['nn_reads_1']} {'' if out['nn_reads_1'] else '(MISS: filed 1)'}; VPD reported as found")
    out["seconds"] = time.time() - t0
    dump(path, out | {"done": True, "misses": miss})


def classes(cfg: Any) -> list[tuple[str, str]]:
    a1 = NNR.load_a1()
    out = [("e62", i) for i in a1.e62]
    c229 = NNR.load_e229()
    out += [("e229", i) for s in c229.seeds for i in c229.items(s)]
    out += [("panel", k) for k, e in NNR.panel_items(a1).items() if not e["random"]]
    return out


def tangent(world: Any, cls: str, item: str, seed: int) -> None:
    path = RUN.RESULTS / "items" / f"gauge__{cls}__{item.replace('/', '__')}.json"
    if path.is_file():
        log(f"{path.stem}: done before, skipped")
        return
    t0 = time.time()
    F, M, _, _, _ = world.make(item)
    mp = dict(M.net.named_parameters())
    dW = {k: (p.detach() - mp[k].detach()) for k, p in F.net.named_parameters() if not torch.equal(p, mp[k])}
    rnd, rec = NNR.matched_random(F.net, M.net, seed)
    out = {"experiment": RUN.EXPERIMENT, "class": cls, "item": item, "update": HG.tangent_share(M, dW), "random": HG.tangent_share(M, rnd),
           "random_seed": rec["random_seed"], "seconds": time.time() - t0}  # fmt: skip
    dump(path, out)
    u, r = out["update"], out["random"]
    log(f"gauge tangent {cls}/{item}: share {u['share']!r} (QK {u['qk']!r}, VO {u['vo']!r}); matched random {r['share']!r}; {out['seconds']:.0f} s")
    del F


def main(cfg: Any, tag: str, shard: tuple[int, int]) -> int:
    (RUN.RESULTS / "items").mkdir(parents=True, exist_ok=True)
    invariance(cfg, tag)
    todo = classes(cfg)[shard[0] :: shard[1]]
    log(f"gauge tangent: {len(todo)} states")
    worlds: dict[str, Any] = {}
    for n, (cls, item) in enumerate(todo):
        stage = {"e62": "e62", "e229": "trained", "panel": "panel_nn"}[cls]
        if stage not in worlds:
            worlds.clear()
            torch.cuda.empty_cache()
            worlds[stage] = RUN.Nn(stage).world
        tangent(worlds[stage], cls, item, G["seed"] + n + shard[0] * 1000)
    shares = {c: [json.loads(p.read_text())["update"]["share"] for p in sorted((RUN.RESULTS / "items").glob(f"gauge__{c}__*.json"))] for c in ("e62", "e229", "panel")}
    log(f"gauge tangent by class (median share): {json.dumps({c: float(np.median(v)) if v else None for c, v in shares.items()})}")
    log(f"{RUN.EXPERIMENT} gauge DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit("run through run.py --stage gauge")

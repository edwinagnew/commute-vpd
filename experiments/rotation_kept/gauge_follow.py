"""E286 stages `gauge_pool` and `gauge_sub` (cell P1.60; whip-lead 2026-11-24, from redteam-lead's verdict
~/commute-data/agents/definition-lead/notes-read/note-2026-11-23-redteam-p160-gauge-verdict.md). Run through
`run.py --stage gauge_pool|gauge_sub`; filed values in predictions.json's "gauge_follow" block (added before they ran).

gauge_pool: M under g0's gauge (the same seed), featurized at VPD; the attention matrices rebuilt from the VPD pieces
    (M: U^T V^T + Delta; F: (U + dU)^T V^T + U^T dV^T + (Delta + dW_out)), and each layer's q, k, v and o pooled into one
    variable whose write is the layer's attention output (`rotation_kept.Attention`). Subcomponents of q, k, v and o
    each span all six heads of a layer, so a per-head pool is not defined from them; the layer is the finest pool.
    The strict, own and needed shares and the change at write scale, as gauge.py reads them.
gauge_sub: per trained state of E62 (9) and E229 (20), its update with the gauge component of its head matrices
    removed (`head_gauge.tangent_component`), read at VPD beside the update itself: pooled strict and needed shares over
    the same read set N (E275's saved K for E62; for E229, which has no VPD scan, K found on the update itself with one
    derangement of E268's pairs, as panel_vpd). Per class (E62: copy, lookup, lookup at copy's size; E229: each task)
    the mean |change| against the seed sd of the unremoved reading.
results/gauge_pool_<tag>.json; results/items/gauge_sub__<item>.json and results/gauge_sub_summary.json.
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

import torch

from commute import head_gauge as HG
from commute import known_edits as KE
from commute import kstar_vpd as KV
from commute import needed as IX
from commute import needed_neurons as NN
from commute import rho
from commute import rotation_kept as RK
from commute.jobs import load_experiment

GA: Any = load_experiment("rotation_kept", "gauge")  # RUN, G (gauge settings), at_scale, log/dump/stop
RUN, NNR, KN = GA.RUN, GA.NNR, GA.KN
GF = json.loads(RUN.PRED.read_text())["gauge_follow"]["settings"]
log, dump, stop = GA.log, GA.dump, GA.stop
ATTN = (".attn.q_proj", ".attn.k_proj", ".attn.v_proj", ".attn.o_proj")


# ---- gauge_pool -------------------------------------------------------------------------------------------------------


@torch.no_grad()
def rebuilt(base: Any, feat: KV.Feat, side: str) -> Any:
    """A copy of `base` whose attention matrices are rebuilt from the VPD pieces of `side` ("M" or "F"), fp64 then cast."""
    out = KE.clone(base)
    for s in feat.sites:
        if not s.name.endswith(ATTN):
            continue
        U, V, dU, dV = (getattr(s, k).double() for k in ("U", "V", "dU", "dV"))
        W = U.T @ V.T + s.rm.double().T if side == "M" else (U + dU).T @ V.T + U.T @ dV.T + s.rf.double().T
        w = out.net.get_submodule(s.name).weight
        w.copy_(W.to(w.dtype))
    return out


def pool(cfg: Any, tag: str) -> None:
    path = RUN.RESULTS / f"gauge_pool_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    t0, v = time.time(), RUN.Vpd()
    _, M, toks, _, _ = v.w.make(cfg.control)
    Fg, _ = HG.rotate(M, GA.G["seed"])
    feat = KV.featurize(M, Fg, v.w.uv, v.w.delta, v.w.bases)
    out: dict[str, Any] = {"experiment": RUN.EXPERIMENT, "card": torch.cuda.get_device_name(), "seed": GA.G["seed"],
                           "pieces": {k: max(c[k] for c in feat.checks.values()) for k in ("F_from_pieces", "M_from_pieces")}}  # fmt: skip
    for name, (F, Mm) in {"vpd_pooled": (rebuilt(Fg, feat, "F"), rebuilt(M, feat, "M")), "weights_pooled": (Fg, M)}.items():
        p, T = rho.Pair(F, Mm), toks.shape[1]
        sites: list[RK.Site] = [RK.Attention(NN.HeadSite(l, l, p.f.layers[l], p.m.layers[l], T)) for l in range(len(p.f.layers))]
        n = len(sites)
        it = RUN.Item(F.net, sites, toks, torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.bool))
        rows, rec = RUN.measure(it, cfg, N=list(range(n)))
        out[name] = rec | GA.at_scale(rows) | {"per_layer_rel_change": [r["dd"] / r["den"] for r in rows]}
        log(f"gauge_pool {name}: max dd/den {out[name]['max_rel_change']!r}, kept at write scale strict {out[name]['kept_strict_scale']!r}, "
            f"needed {out[name]['kept_needed_scale']!r}; pooled {json.dumps(rec['pooled'])}")  # fmt: skip
    r, bar = out["vpd_pooled"], GF["pool_bar"]
    out["reads_1"] = r["max_rel_change"] <= bar and r["kept_strict_scale"] >= 1 - bar
    log(f"gauge_pool: VPD pooled per layer reads 1 within {bar!r}: {out['reads_1']}{'' if out['reads_1'] else ' (MISS: filed about 1; reported as found)'}")
    dump(path, out | {"seconds": time.time() - t0, "done": True})


# ---- gauge_sub -------------------------------------------------------------------------------------------------------


def todo() -> list[tuple[str, str]]:
    c229 = NNR.load_e229()
    return [("e62", i) for i in NNR.load_a1().e62] + [("e229", i) for s in c229.seeds for i in c229.items(s)]


def klass(group: str, item: str) -> str:
    if group == "e62":
        return "lookup_scaled" if "scaled" in item else item.split("/")[1].rsplit("_s", 1)[0]
    return item.rsplit("_s", 1)[0]


@torch.no_grad()
def removed(F: Any, comp: dict[str, torch.Tensor]) -> Any:
    out = KE.clone(F)
    for k, c in comp.items():
        w = out.net.get_parameter(k)
        w.copy_((w.double() - c.to(w.device)).to(w.dtype))
    return out


def sub_item(vpd: Any, worlds: dict[str, Any], cfg: Any, group: str, item: str) -> None:
    path = RUN.RESULTS / "items" / f"gauge_sub__{item.replace('/', '__')}.json"
    if path.is_file():
        log(f"{path.stem}: done before, skipped")
        return
    t0 = time.time()
    F, M, toks, at, _ = (vpd.w if group == "e62" else worlds["e229"]).make(item)
    mp = dict(M.net.named_parameters())
    dW = {k: p.detach() - mp[k].detach() for k, p in F.net.named_parameters() if not torch.equal(p, mp[k])}
    comp, share = HG.tangent_component(M, dW)
    Fs = removed(F, comp)
    feats = {k: KV.featurize(M, X, vpd.w.uv, vpd.w.delta, vpd.w.bases) for k, X in (("update", F), ("removed", Fs))}
    worst = max(c["M_from_pieces"] for c in feats["update"].checks.values())
    if worst > GF["pieces_tol"]:
        stop(path, {"item": item}, f"{item}: M is not the decomposed model (M from pieces {worst!r})")
    fu = feats["update"]
    if group == "e62":
        flips, K = RUN.ranking(RUN.E275 / f"items_{KN.fname(item)}.npz")
    else:
        kc = KN.load_config()
        nd = IX.needed_set(F, fu, toks, at, cfg.panel_vpd_derangements, kc.examples, kc.seed, kc.rows, kc.logit_rows, kc.cache_batch, log)
        flips, K = nd.scan.flips, nd.K
        del nd
    N = RK.sets(RK.vpd_sites(fu), flips, K, RUN.mask(fu.changed, fu.n), cfg.top_frac, cfg.seed).N
    out: dict[str, Any] = {"experiment": RUN.EXPERIMENT, "group": group, "class": klass(group, item), "item": item, "gauge_share": share, "size_N": len(N)}
    for k, X in (("update", F), ("removed", Fs)):
        it = RUN.Item(X.net, RK.vpd_sites(feats[k]), toks, RUN.mask(feats[k].changed, feats[k].n), flips, K)
        _, rec = RUN.measure(it, cfg, N=N)
        out[k] = {"pooled": rec["pooled"], "pooled_with_rem": rec["pooled_with_rem"]}
    main = RUN.RESULTS / "items" / f"vpd__{item.replace('/', '__')}.json"
    out["main_stage_pooled"] = json.loads(main.read_text())["pooled"] if main.is_file() else None
    out["change"] = {s: out["removed"]["pooled"][s] - out["update"]["pooled"][s] for s in ("strict", "needed")}
    out["seconds"] = time.time() - t0
    dump(path, out)
    log(f"gauge_sub {item}: gauge share {share['share']!r}; strict {out['update']['pooled']['strict']:.6g} -> {out['removed']['pooled']['strict']:.6g}, "
        f"needed {out['update']['pooled']['needed']:.6g} -> {out['removed']['pooled']['needed']:.6g}; {out['seconds']:.0f} s")  # fmt: skip
    del F, Fs, feats
    torch.cuda.empty_cache()


def sub_summary() -> None:
    got = [json.loads(p.read_text()) for p in sorted((RUN.RESULTS / "items").glob("gauge_sub__*.json"))]
    by: dict[str, list[dict[str, Any]]] = {}
    for r in got:
        by.setdefault(f"{r['group']}/{r['class']}", []).append(r)
    out: dict[str, Any] = {"experiment": RUN.EXPERIMENT, "n_items": len(got), "classes": {}}
    for c, rs in sorted(by.items()):
        rec: dict[str, Any] = {"n_seeds": len(rs)}
        for s in ("strict", "needed"):
            base = [r["update"]["pooled"][s] for r in rs]
            sd = statistics.stdev(base) if len(base) > 1 else None
            moved = statistics.fmean(abs(r["change"][s]) for r in rs)
            rec[s] = {"seed_sd": sd, "mean_abs_change": moved, "moves_less_than_sd": None if sd is None else moved < sd}
        out["classes"][c] = rec
    dump(RUN.RESULTS / "gauge_sub_summary.json", out)
    log(f"gauge_sub summary: {json.dumps(out['classes'])}")


def main(stage: str, cfg: Any, tag: str, shard: tuple[int, int]) -> int:
    if stage == "gauge_pool":
        pool(cfg, tag)
        return 0
    vpd, worlds = RUN.Vpd(), {}
    items = todo()[shard[0] :: shard[1]]
    log(f"gauge_sub: {len(items)} states")
    for n, (group, item) in enumerate(items):
        if group == "e229" and "e229" not in worlds:
            worlds["e229"] = RUN.Nn("trained").world
        sub_item(vpd, worlds, cfg, group, item)
        log(f"ITEM {n + 1} of {len(items)} DONE ({item})")
    sub_summary()
    log(f"{RUN.EXPERIMENT} gauge_sub DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit("run through run.py --stage gauge_pool|gauge_sub")

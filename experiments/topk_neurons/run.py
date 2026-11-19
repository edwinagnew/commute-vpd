"""E217 (cell P2.42, run 1): the working definition read at a top-activating-neuron featurization of the 67M.

The featurization is fixed from the base model M: at each MLP site a neuron is a feature on a token when its activation
is above that neuron's upper percentile or below its lower one (Goodfire's p97/3; p99/1 and p90/10 beside), percentiles
computed once on M over Pile text; the rest of the site's activations is one remainder variable. The same rule reads
the fine-tune F. Per item: s_u of every variable, K* by mean ablation, R, f and the remainder's share (`commute.kstar`
with its Feat argument), and the overlap of the strong-neuron sets of M and F. Plan
reports/plans/2026-11-19_plan_topk-neurons.md; settings and expected values in results/predictions.json.

    python experiments/topk_neurons/run.py --stage 0          percentiles, W0, the floor, the width control (stops)
    python experiments/topk_neurons/run.py --stage 1 --task T  one band and seed (T = SLURM_ARRAY_TASK_ID, 0-7)
    python experiments/topk_neurons/run.py --stage score       no GPU: results/summary.json from the item files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import kstar as K
from commute import rho
from commute import rule_retention as R
from commute.data import cached_rows
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

E196: Any = load_experiment("rule_retention_pairs", "run")  # Data, fine_tune, subset, NUMBERS: E208's item loaders
HERE = Path(__file__).resolve().parent
REPO, RESULTS = HERE.parents[1], HERE / "results"
PRED, STAGE0, ITEMS, LARGE = RESULTS / "predictions.json", RESULTS / "stage0.json", RESULTS / "items", RESULTS / "large"
PCT = LARGE / "percentiles.pt"
LAST = slice(-1, None)
WIDTH = "width"


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    pairs: list[list[str]]
    counted: int
    seeds: list[int]
    n_read: int
    read_seed: int
    n_max: int
    held_share: float
    bands: dict[str, list[float]]  # name -> [lower quantile, upper quantile]
    main_band: str
    pile_rows: int
    pile_batch: int
    alpha: float
    chunk: int
    order_seed: int
    control_item: str
    profile_batch: int
    e208_parts: str
    e213_summary: str
    gate_tol: float
    s_tol: float
    floor_factor: float
    floor_min: float

    def __post_init__(self) -> None:
        assert self.main_band in self.bands and WIDTH not in self.bands and all(0 < lo < 0.5 < hi < 1 for lo, hi in self.bands.values())
        assert 0 < self.counted <= len(self.pairs) and self.control_item in self.items(self.seeds[0])

    def items(self, seed: int) -> list[str]:
        return [f"{m}_s{seed}" for pr in self.pairs for m in pr]

    def tasks(self) -> list[tuple[str, int]]:
        """Stage 1's array: the main band first, then the width, then the rest; each split by seed."""
        return [(b, s) for b in [self.main_band, WIDTH, *[b for b in self.bands if b != self.main_band]] for s in self.seeds]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    dump(path, out | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


def slug(band: str) -> str:
    return band.replace("/", "_")


# ---- the model, the items and the featurization ---------------------------------------------------------------------


def setup(cfg: Config) -> tuple[Any, R.Model, Any]:
    from commute import old_tasks as OT
    from commute import variable_read as VR
    from commute import vpd
    from commute.steering import Net

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E208
    net = Net("cuda")
    net.net.requires_grad_(False)
    M, tok, dcfg = R.Model(net.net), vpd.tokenizer(), OT.DSizes()
    x = VR.make_numbers(tok, int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    assert x.keys.tolist() == json.loads(E196.NUMBERS.read_text())["number_tokens"], "the number tokens are not E154's"
    return net, M, E196.Data(VR.make_sets(dcfg, tok, log), x, dcfg, {})


def windows(cfg: Config, M: R.Model, data: Any, item: str) -> Tensor:
    """E208's 512 read windows of the item (E196.subset of its read half)."""
    return E196.subset(data.read_half(item.rsplit("_s", 1)[0]), cfg, M)


def pair(net: Any, M: R.Model, item: str) -> rho.Pair:
    return rho.Pair(R.Model(E196.fine_tune(net, item).net), M)


@torch.no_grad()
def percentiles(cfg: Config, M: R.Model) -> tuple[list[float], Tensor, dict[str, float]]:
    """M's per-neuron activation quantiles over every position of the Pile rows, (n_q, layers, n), and each band's share
    of those activations that are features."""
    m = rho.Pair(M, M).m
    acts: list[list[Tensor]] = [[] for _ in m.layers]
    for t in cached_rows("val", 0, cfg.pile_rows).split(cfg.pile_batch):
        for l, a in enumerate(K.run(m, t.to("cuda"), None).a):
            acts[l].append(a.reshape(-1, a.shape[-1]))
    qs, pct, hits, total = sorted({q for b in cfg.bands.values() for q in b}), [], dict.fromkeys(cfg.bands, 0), 0
    for l in range(len(acts)):
        a, acts[l] = torch.cat(acts[l]), []
        pct.append(torch.stack([a.kthvalue(round(q * (a.shape[0] - 1)) + 1, 0).values for q in qs]))  # (n_q, n), lower rank
        for b, (lo, hi) in cfg.bands.items():
            hits[b] += int(((a > pct[-1][qs.index(hi)]) | (a < pct[-1][qs.index(lo)])).sum())
        total += a.numel()
    return qs, torch.stack(pct, 1).cpu(), {b: h / total for b, h in hits.items()}


def feat_of(cfg: Config, qs: list[float], pct: Tensor, band: str) -> K.Feat:
    """The band's rule; the width makes every activation a feature (hi = -inf), so each site remainder writes zero."""
    if band == WIDTH:
        n = pct.shape[1:]
        return K.Feat([torch.full(n[1:], -torch.inf, device="cuda")] * n[0], [torch.full(n[1:], -torch.inf, device="cuda")] * n[0])
    lo, hi = cfg.bands[band]
    return K.Feat(list(pct[qs.index(lo)].to("cuda")), list(pct[qs.index(hi)].to("cuda")))


def load_pct(cfg: Config) -> tuple[list[float], Tensor]:
    """The quantiles are the config's own floats: the file stores them in float32, where 0.03 != 0.03 (E217 pods)."""
    z, qs = torch.load(PCT, weights_only=True), sorted({q for b in cfg.bands.values() for q in b})
    assert hashlib.sha256(PCT.read_bytes()).hexdigest() == json.loads(STAGE0.read_text())["percentiles_sha256"], "percentiles changed since stage 0"
    assert torch.equal(z["q"], torch.tensor(qs)), "the saved quantiles are not the config's"
    return qs, z["values"]


# ---- the readings ---------------------------------------------------------------------------------------------------


@torch.no_grad()
def overlap(p: rho.Pair, toks: Tensor, feat: K.Feat) -> float:
    """Mean over windows of the Jaccard of M's and F's strong-neuron sets at the read position, sites pooled (empty union: 1)."""
    out = []
    for t in toks.split(K.GRADE_BATCH):
        am, af = K.run(p.m, t, None).a, K.run(p.f, t, None).a
        sm = torch.cat([feat.strong(l, a[:, LAST]) for l, a in enumerate(am)], -1)
        sf = torch.cat([feat.strong(l, a[:, LAST, : am[l].shape[-1]]) for l, a in enumerate(af)], -1)
        inter, union = (sm & sf).sum((1, 2)).double(), (sm | sf).sum((1, 2)).double()
        out.append(torch.where(union > 0, inter / union.clamp(min=1), 1.0))
    return float(torch.cat(out).mean())


def read_vars(p: rho.Pair, toks: Tensor, feat: K.Feat) -> tuple[K.Level, set[int], set[int], Tensor]:
    lv = K.level(p, "neuron_head", False, rem=True)
    C, lacked = K.changed_vars(p, lv)
    return lv, C, lacked, K.level_grades(K.grades(p, toks, LAST, None, feat), lv, lacked)


def w0(cfg: Config, M: R.Model, toks: Tensor, feat: K.Feat) -> tuple[dict[str, Any], list[str]]:
    p = rho.Pair(M, M)
    lv, C, _, s = read_vars(p, toks, feat)
    g = K.gates(K.prepare(p, toks, LAST, None, M.logits, feat), lv, M.logits)
    strong = float(torch.cat([feat.strong(l, a[:, LAST]).double().mean().reshape(1) for l, a in enumerate(K.run(p.m, toks, None).a)]).mean())
    out = {"changed": len(C), "s_min": float(s.min()), "s_max": float(s.max()), "gates": g, "overlap": overlap(p, toks, feat), "read_strong_share": strong}
    miss = [k for k, bad in (("changed", len(C) > 0), ("s", float((s - 1).abs().max()) > cfg.s_tol), ("gates", max(g.values()) > cfg.gate_tol),
                             ("overlap", out["overlap"] != 1.0)) if bad]  # fmt: skip
    return out, miss


def control(cfg: Config, s: Tensor, lv: K.Level, item: str, e213: dict[str, Any]) -> dict[str, Any]:
    """At the width: R ('all' grade) over E208's parts and over its five circuits, against E213; per-part gap to E208."""
    z = np.load(REPO / cfg.e208_parts / f"{item}.npz")
    index = {K.vname(u): j for j, u in enumerate(lv.vars)}
    mine = s[0, [index[str(n)] for n in z["parts"]]].numpy()
    orders = [mine[z[f"K_replace_{j}"]].mean() for j in range(5)]
    ref = e213["items"][item]["main"]
    return {"R_parts": float(mine.mean()), "R_parts_e213": ref["worst_k"]["all"], "R_K": float(np.mean(orders)), "R_K_e213": ref["e208_K_mean"],
            "max_part_diff": float(np.abs(mine - z["r_main"]).max()), "rem_s": s[:, lv.rem].tolist() if lv.rem else None}  # fmt: skip


def stage0(cfg: Config, net: Any, M: R.Model, data: Any) -> None:
    qs, pct, share = percentiles(cfg, M)
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({"q": torch.tensor(qs), "values": pct}, PCT)
    out: dict[str, Any] = {"experiment": "E217", "percentiles_sha256": hashlib.sha256(PCT.read_bytes()).hexdigest(), "pile_strong_share": share}
    log(f"percentiles at {qs} over {cfg.pile_rows} Pile rows; feature share {share}")
    if bad := [b for b, (lo, hi) in cfg.bands.items() if abs(share[b] - (lo + 1 - hi)) > 0.02]:
        stop(STAGE0, out, f"feature share on the Pile far from its band: {bad}")
    toks = windows(cfg, M, data, cfg.control_item)
    for band in [*cfg.bands, WIDTH]:
        out[f"w0_{slug(band)}"], miss = w0(cfg, M, toks, feat_of(cfg, qs, pct, band))
        log(f"W0 {band}: {out[f'w0_{slug(band)}']}")
        if miss:
            stop(STAGE0, out, f"W0 at {band} missed: {miss}")
    p = pair(net, M, cfg.control_item)
    prof = rho.profile(p, toks.split(cfg.profile_batch))["r"]["main"]  # E208's own code, rerun on this card
    z = np.load(REPO / cfg.e208_parts / f"{cfg.control_item}.npz")
    names = {rho.name(u): v for u, v in prof.items()}
    out["floor"] = float(np.abs(np.array([names[str(n)] for n in z["parts"]]) - z["r_main"]).max())
    bar = max(cfg.floor_factor * out["floor"], cfg.floor_min)
    log(f"floor (E208's code rerun vs its saved scores, largest per-part difference) {out['floor']:.3g}; control bar {bar:.3g}")
    e213, width, miss = json.loads((REPO / cfg.e213_summary).read_text()), feat_of(cfg, qs, pct, WIDTH), []
    out["control"] = {}
    for item in cfg.items(0) + cfg.items(1):
        p = pair(net, M, item)
        lv, _, _, s = read_vars(p, windows(cfg, M, data, item), width)
        c = out["control"][item] = control(cfg, s, lv, item, e213)
        miss += [item] if max(abs(c["R_parts"] - c["R_parts_e213"]), abs(c["R_K"] - c["R_K_e213"])) > bar or float((s[:, lv.rem] - 1).abs().max()) > cfg.s_tol else []
        log(f"control {item}: R parts {c['R_parts']:.8f} (E213 {c['R_parts_e213']:.8f}), R on E208's K {c['R_K']:.8f} (E213 {c['R_K_e213']:.8f}), "
            f"largest part gap {c['max_part_diff']:.3g}")  # fmt: skip
        torch.cuda.empty_cache()
    if miss:
        stop(STAGE0, out, f"the width control missed E213 on {miss}")
    dump(STAGE0, out | {"done": True})
    log("STAGE 0 DONE")


def read_item(cfg: Config, net: Any, M: R.Model, data: Any, feat: K.Feat, band: str, item: str) -> None:
    path = ITEMS / slug(band) / f"{item}.json"
    if path.is_file():
        log(f"{band} {item}: done before, skipped")
        return
    t0, p, toks = time.time(), pair(net, M, item), windows(cfg, M, data, item)
    lv, C, lacked, s = read_vars(p, toks, feat)
    ctx = K.prepare(p, toks, LAST, None, M.logits, feat)
    g = K.gates(ctx, lv, p.F.logits)
    out: dict[str, Any] = {"experiment": "E217", "item": item, "band": band, "n_vars": len(lv.vars), "n_changed": len(C), "gates": g,
                           "kl_fm": float(ctx.kl_fm.mean()), "kl_none": float(ctx.kl_none.mean()), "overlap": overlap(p, toks, feat),
                           "s_rem": s[:, lv.rem].tolist() if lv.rem else None, "s_mean_all_vars": s.mean(1).tolist()}  # fmt: skip
    if g["all"] > cfg.gate_tol:
        stop(path.with_suffix(".stopped.json"), out, f"{band} {item}: F_K with every variable kept is not F (gate {g})")
    th = {"order_seed": cfg.order_seed, "chunk": cfg.chunk}
    res, arrays = K.search(ctx, lv, s, C, lacked, {"main": K.Variant(torch.arange(len(toks)), cfg.alpha, 1)}, th, log, lambda _s: None)
    rec = res["main"][0]
    rec.pop("K")  # the circuit goes to the npz: thousands of entries
    out |= rec | {"item_seconds": time.time() - t0}
    big = LARGE / "items" / slug(band)
    big.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(big / f"{item}.npz", s=s.numpy(), names=np.array([K.vname(u) for u in lv.vars]), **arrays)
    dump(path, out)
    log(f"{band} {item}: R read {out['R_read']:.6f}, all {out['R_all']:.6f}, |K*| {out['size']}, f {out['f']:.4f}, "
        f"remainders in K* {out['site_remainders_in_K']}, overlap {out['overlap']:.4f}; {out['item_seconds']:.0f} s")


def stage1(cfg: Config, net: Any, M: R.Model, data: Any, task: int) -> None:
    if not (STAGE0.is_file() and json.loads(STAGE0.read_text()).get("done")):
        log("STOP: stage 0 has not passed")
        hard_exit(1)
    band, seed = cfg.tasks()[task]
    qs, pct = load_pct(cfg)
    feat = feat_of(cfg, qs, pct, band)
    log(f"stage 1 task {task}: band {band}, seed {seed}")
    for item in cfg.items(seed):
        read_item(cfg, net, M, data, feat, band, item)
        torch.cuda.empty_cache()
    log(f"STAGE 1 TASK {task} DONE")


# ---- the score (no GPU) ---------------------------------------------------------------------------------------------


def score(cfg: Config) -> None:
    rows = {f"{p.parent.name}/{p.stem}": json.loads(p.read_text()) for p in sorted(ITEMS.glob("*/*.json")) if not p.stem.endswith(".stopped")}
    keys = ("R_read", "R_all", "f", "site_remainder_share", "size", "overlap", "s_rem")
    table = {k: {x: r.get(x) for x in keys} for k, r in rows.items()}
    gaps = []
    for band in [WIDTH, *cfg.bands]:
        for reuse, twin in cfg.pairs[: cfg.counted]:
            got = {s: (rows.get(f"{slug(band)}/{reuse}_s{s}"), rows.get(f"{slug(band)}/{twin}_s{s}")) for s in cfg.seeds}
            g = {s: a["R_read"] - b["R_read"] for s, (a, b) in got.items() if a and b}
            gaps.append({"band": band, "pair": f"{reuse}/{twin}", "gap_by_seed": g, "seed_diff": g[0] - g[1] if len(g) == 2 else None})
    pred = json.loads(PRED.read_text())["predictions"]
    trained = [i for s in cfg.seeds for i in cfg.items(s)]

    def check(name: str, band: str, value: Any) -> dict[str, Any]:
        lo, hi = pred[name]["range"]
        vals = [value(rows[f"{slug(band)}/{i}"]) for i in trained if f"{slug(band)}/{i}" in rows]
        return {"read": len(vals), "of": len(trained), "min": min(vals, default=None), "max": max(vals, default=None),
                "within": bool(vals) and all(lo <= v <= hi for v in vals)}  # fmt: skip

    leans: dict[str, Any] = {"P1": check("P1", WIDTH, lambda r: r["R_read"]), "P2": check("P2", cfg.main_band, lambda r: r["R_read"]),
             "P3": check("P3", cfg.main_band, lambda r: r["site_remainder_share"]), "P4": check("P4", cfg.main_band, lambda r: min(r["s_rem"][1])),
             "P5": check("P5", cfg.main_band, lambda r: r["overlap"]), "P6": check("P6", cfg.main_band, lambda r: r["f"])}  # fmt: skip
    leans["P2_below_width"] = [i for i in trained if f"{slug(cfg.main_band)}/{i}" in rows and f"width/{i}" in rows
                               and not rows[f"{slug(cfg.main_band)}/{i}"]["R_read"] < rows[f"width/{i}"]["R_read"]]  # fmt: skip
    dump(RESULTS / "summary.json", {"experiment": "E217", "table": table, "gaps": gaps, "leans": leans})
    for k, v in leans.items():
        log(f"{k}: {v}")
    for g in gaps:
        log(f"gap {g['band']:7s} {g['pair']:24s} {g['gap_by_seed']} seed diff {g['seed_diff']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["0", "1", "score"], required=True)
    ap.add_argument("--task", type=int, help="stage 1: index into Config.tasks() (the array task id)")
    args = ap.parse_args()
    require_filed(PRED)
    cfg = config_from_settings(Config, json.loads(PRED.read_text())["settings"])
    if args.stage == "score":
        score(cfg)
        hard_exit(0)
    net, M, data = setup(cfg)
    if args.stage == "0":
        stage0(cfg, net, M, data)
    else:
        assert args.task is not None and 0 <= args.task < len(cfg.tasks()), f"--task must be 0..{len(cfg.tasks()) - 1}"
        stage1(cfg, net, M, data, args.task)
    hard_exit(0)


if __name__ == "__main__":
    main()

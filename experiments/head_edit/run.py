"""E280 (cell P1.57; whip-lead, 2026-11-23, after the E62 E_b diagnosis): three built edits of M on E62's copy head
(2, 4), each read by E_b over the needed set K at VPD and at the neuron/head featurization. Descriptive; card
experiments/head_edit/card.md; settings and the filed expected values in results/predictions.json, which the run
refuses unless committed unchanged.

Edits (everything else exactly M; head 4 = q/k/v rows and o input columns 512:640):
    v2   v_proj rows times 2: head 4's z, so its write, exactly doubled
    o2   o_proj input columns times 2: the same function as v2
    q2   q_proj rows times 2: head 4's attention logits doubled
D and the read position: E257's World's item e62/copy_s0, the D E275 read: 512 windows of 8 tokens from E187's read
half of the e62 group (copy windows 'A B .. A M', windows without the cue, far text, 3:3:2), read at the last position
(the marker in the copy windows). Its F is not used.

Featurizations, the scan settings E268's (`kstar_needed` load_config) at both:
    vpd      E278's path (experiments/eb_dropped/run.py read_item): `kstar_vpd.featurize(M, F, ...)`, one
             `stand_in.write_change` pass, `needed.needed_set`; plus per layer-2 attention matrix the dd weight and kept
             share over K and changed, and for q/k/v the kept share redteam-lead derived (sum s_i dd_i / sum dd_i)
    neurons  E277's path (experiments/needed_neurons/run.py read_item): its `write_pass` and `needed_neurons.cache` and
             `scan` on E229's bounds at E277's band; plus head (2, 4)'s own columns (one variable there)
Stage 0 runs first in every job, per featurization read (skipped once results/stage0_<feat>_<tag>.json says done);
any miss stops the job: the do-nothing (F a copy of M) has no changed variable, dd max exactly 0 and E_b over K
exactly 1; the identity interchange (x' = x) on v2 changes no argmax (its largest logit deviation is the floor). It
also writes the sanity line: each edit's copy-task loss at the marker minus M's, and v2 against o2's largest logit gap.

    python experiments/head_edit/run.py --stage read --feat vpd --tag v0
    python experiments/head_edit/run.py --stage read --feat neurons --tag n0
    python experiments/head_edit/run.py --stage summary          (no GPU)
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
from commute import needed as IX
from commute import needed_neurons as NN
from commute import old_tasks as OT
from commute import rule_retention as R
from commute import stand_in as SI
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)

KN: Any = load_experiment("kstar_needed", "run")  # E268: world (E257's with the split's bases), load_config, KP, log
NR: Any = load_experiment("needed_neurons", "run")  # E277: write_pass, load_config (band), load_e229, E229, HEAD24
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log = KN.log
EDITS = {"v2": ("v_proj", 0), "o2": ("o_proj", 1), "q2": ("q_proj", 0)}  # matrix, and the head's axis (0 rows, 1 columns)
FEATS = ("vpd", "neurons")


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    item: str  # E257's item whose D and read position are read
    layer: int
    head: int
    factor: float
    too_small: int
    q_band: float  # summary only: q2 within this of v2 at VPD

    def __post_init__(self) -> None:
        assert self.item == "e62/copy_s0" and 0 <= self.head < R.N_HEAD and self.factor != 1 and self.too_small > 0

    @property
    def cols(self) -> slice:
        return slice(self.head * R.D_HEAD, (self.head + 1) * R.D_HEAD)


def load_config() -> Config:
    return config_from_settings(Config, json.loads(PRED.read_text())["settings"])


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    write_stage(path, out | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


# ---- F, D -----------------------------------------------------------------------------------------------------------


def edited(w: Any, cfg: Config, name: str) -> R.Model:
    """M with the head's slice of one matrix times factor (`do_nothing`: a copy of M)."""
    if name == "do_nothing":
        return R.Model(OT.clone_net(w.kl.net).net)
    mat, axis = EDITS[name]
    p = f"h.{cfg.layer}.attn.{mat}.weight"
    W = dict(w.kl.net.net.named_parameters())[p].detach()
    d = torch.zeros_like(W)
    if axis == 0:
        d[cfg.cols] = W[cfg.cols] * (cfg.factor - 1)
    else:
        d[:, cfg.cols] = W[:, cfg.cols] * (cfg.factor - 1)
    return R.Model(OT.with_delta(w.kl.net, {p: d}).net)


def data(w: Any, cfg: Config) -> tuple[Tensor, slice]:
    _, _, toks, at, _ = w.make(cfg.item)
    assert toks.dim() == 2 and at == KN.KP.LAST, f"{cfg.item}: D {tuple(toks.shape)}, read at {at}"
    return toks, at


def targets(toks: Tensor) -> tuple[Tensor, Tensor]:
    """D is E187's read half (old_tasks.task_windows): copy windows 'A B .. A M' mixed with windows without the cue and
    far text. The copy windows (marker last, A once before A M) and each one's B, the token after that first A."""
    marker = int(torch.load(OT.TASKS["e62_copy"].sources[0], weights_only=False)["pools"]["marker"][0])
    hit = toks[:, :-2] == toks[:, -2:-1]
    copy = ((toks[:, -1] == marker) & (hit.sum(1) == 1)).nonzero().squeeze(1)
    return copy, toks[copy, hit[copy].long().argmax(1) + 1]


@torch.no_grad()
def marker_logits(F: R.Model, toks: Tensor, batch: int) -> Tensor:
    return torch.cat([F.logits(t)[:, -1].double() for t in toks.split(batch)])


def sanity(w: Any, cfg: Config, kc: Any, M: R.Model, toks: Tensor) -> dict[str, Any]:
    """The copy task's loss at the marker per edit minus M's; v2 against o2 (the same function) at the marker."""
    (copy, B), lg = targets(toks), {"M": marker_logits(M, toks, kc.cache_batch)}
    for name in EDITS:
        F = edited(w, cfg, name)
        lg[name] = marker_logits(F, toks, kc.cache_batch)
        del F
    loss = {k: float(torch.nn.functional.cross_entropy(v[copy.to(v.device)], B.to(v.device))) for k, v in lg.items()}
    return {"n_windows": len(toks), "n_copy_windows": len(copy), "loss_M": loss["M"], "loss_change": {k: loss[k] - loss["M"] for k in EDITS}, "logit_scale": float(lg["M"].abs().max()),
            "v2_o2_max_logit_diff": float((lg["v2"] - lg["o2"]).abs().max()), "v2_o2_loss_diff": loss["v2"] - loss["o2"]}  # fmt: skip


# ---- the two reads --------------------------------------------------------------------------------------------------


@dataclass
class Read:
    num: Tensor
    den: Tensor
    dd: Tensor
    ch: Tensor
    K: Tensor
    first_flip: Tensor
    flips: Tensor
    margin: Tensor
    n_pairs: int
    cases_per_s: float
    extra: dict[str, Any]


def vpd(w: Any, cfg: Config, kc: Any, M: R.Model, F: R.Model, toks: Tensor, at: slice) -> Read:
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(kc.cache_batch)))
    nd = IX.needed_set(F, feat, toks, at, kc.derangements, kc.examples, kc.seed, kc.rows, kc.logit_rows, kc.cache_batch, log)
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    r = Read(num, den, dd, ch, nd.K, nd.first_flip, nd.scan.flips, nd.run.margin.flatten().cpu(), nd.n_pairs, nd.scan.rows / nd.scan.seconds, {})
    r.extra = {"L2_attn": l2_attn(feat, r, cfg)}
    return r


def neurons(kc: Any, nfeat: Any, M: R.Model, F: R.Model, toks: Tensor, at: slice) -> Read:
    num, den, dd, ch, _ = NR.write_pass(F, M, toks, nfeat, kc.cache_batch)
    run = NN.cache(F.net, toks, at, kc.cache_batch)
    m = min(kc.examples, len(toks))
    ex, pt = IX.derangements(len(toks), kc.derangements, m, kc.seed)
    sc = NN.scan(F.net, nfeat, run, at, ex, pt, IX.rows_per_forward(kc.rows, kc.logit_rows, run.top.shape[1]), log)
    K = sc.first_pair >= 0
    r = Read(num, den, dd, ch, K, torch.where(K, sc.first_pair // m, -1), sc.flips, run.margin.flatten().cpu(), len(ex), sc.rows / sc.seconds, {})
    r.extra = {"head_2_4": head24(r)}
    return r


def l2_attn(feat: KV.Feat, r: Read, cfg: Config) -> dict[str, Any]:
    """Per layer-2 attention matrix: its subcomponents' and remainder's dd weight over K and changed and their pooled kept
    share; for q/k/v the derived kept share sum s_i dd_i / sum dd_i, s_i = subcomponent i's share of |u_i v_i^T|^2 in
    head 4's rows (e62_diag/diag.py head4_share). o's head columns are its input; its kept share is not s_i."""
    sel, expl = r.K & r.ch, SI.explained(r.num, r.den)
    tot = float(r.dd[sel].sum())
    out = {}
    for s in (s for s in feat.sites if s.name.startswith(f"h.{cfg.layer}.attn.")):
        sub = torch.zeros_like(sel)
        sub[s.off : s.rem] = True
        a, rem = sel & sub, s.rem
        x = s.V.T.double().cpu() if s.name.endswith("o_proj") else s.U.double().cpu()  # (C, head axis)
        assert x.shape == (s.rem - s.off, R.N_HEAD * R.D_HEAD), (s.name, tuple(x.shape))
        share = (x[:, cfg.cols].pow(2).sum(1) / x.pow(2).sum(1).clamp(min=1e-300)).to(r.dd.dtype)
        dsub = float(r.dd[a].sum())
        out[s.name.rsplit(".", 1)[1]] = {
            "n_sub": s.rem - s.off, "n_changed": int((r.ch & sub).sum()), "n_changed_K": int(a.sum()),
            "dd_weight_sub": dsub / tot if tot > 0 else None, "kept_sub": float(expl[a].sum()) / dsub if dsub > 0 else None,
            "kept_derived_s": float((share * r.dd[s.off : s.rem])[a[s.off : s.rem]].sum()) / dsub if dsub > 0 and not s.name.endswith("o_proj") else None,
            "head4_share_median": float(share.median()), "rem_in_K_changed": bool(sel[rem]),
            "dd_weight_rem": float(r.dd[rem]) / tot if tot > 0 and bool(sel[rem]) else None,
            "kept_rem": float(expl[rem] / r.dd[rem]) if float(r.dd[rem]) > 0 else None}  # fmt: skip
    return out


def head24(r: Read) -> dict[str, Any]:
    u, sel = NR.HEAD24, r.K & r.ch
    dd, tot = float(r.dd[u]), float(r.dd[sel].sum())
    return {"in_K": bool(r.K[u]), "changed": bool(r.ch[u]), "flips": int(r.flips[u]), "kept": float(SI.explained(r.num, r.den)[u]) / dd if dd > 0 else None,
            "dd_weight_K_changed": dd / tot if tot > 0 and bool(sel[u]) else None}  # fmt: skip


def record(r: Read, cfg: Config, kc: Any, floor: float) -> dict[str, Any]:
    K, ch = r.K, r.ch
    eb = lambda s: SI.explained_share(r.num, r.den, r.dd, s)
    return {"n_vars": len(ch), "n_changed": int(ch.sum()), "size_K": int(K.sum()), "n_changed_K": int((K & ch).sum()), "E_b_K": eb(K & ch),
            "E_b_changed": eb(ch), "empty_K": not bool((K & ch).any()), "too_small": int(K.sum()) < cfg.too_small, "n_pairs": r.n_pairs,
            "median_margin": float(r.margin.median()), "min_margin": float(r.margin.min()), "n_fragile_positions": int((r.margin <= floor).sum()),
            "fragile_floor": floor, "cases_per_s": r.cases_per_s,
            "by_derangement": [{"size_K": int((K & (r.first_flip <= p)).sum()), "E_b_K": eb(K & (r.first_flip <= p) & ch)} for p in range(kc.derangements)]} | r.extra  # fmt: skip


# ---- stage 0, items, summary ----------------------------------------------------------------------------------------


def reader(feat: str, w: Any, cfg: Config, kc: Any, nfeat: Any) -> Any:
    return (lambda M, F, toks, at: vpd(w, cfg, kc, M, F, toks, at)) if feat == "vpd" else (lambda M, F, toks, at: neurons(kc, nfeat, M, F, toks, at))


def identity(feat: str, w: Any, kc: Any, nfeat: Any, F: R.Model, toks: Tensor, at: slice) -> dict[str, Any]:
    if feat == "vpd":
        return IX.identity_scan(F, KV.featurize(w.kl.M, F, w.uv, w.delta, w.bases), toks, at, kc.examples, kc.ident_vars, kc.rows, kc.logit_rows, kc.cache_batch, log)
    run, ex = NN.cache(F.net, toks, at, kc.cache_batch), torch.arange(min(kc.examples, len(toks)))
    sc = NN.scan(F.net, nfeat, run, at, ex, ex, IX.rows_per_forward(kc.rows, kc.logit_rows, run.top.shape[1]), log)
    return {"flips": int(sc.flips.sum()), "max_dev": sc.max_dev, "cases": sc.rows, "cases_per_s": sc.rows / sc.seconds, "n_windows": len(toks)}


def stage0(feat: str, w: Any, cfg: Config, kc: Any, nfeat: Any, toks: Tensor, at: slice, tag: str) -> float:
    path = RESULTS / f"stage0_{feat}_{tag}.json"
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        log(f"{path.name}: done before")
        return float(done["floor"])
    t0, M = time.time(), w.kl.M
    out: dict[str, Any] = {"experiment": EXPERIMENT, "feat": feat, "card": torch.cuda.get_device_name()}
    F = edited(w, cfg, "do_nothing")
    if feat == "vpd":
        fv = KV.featurize(M, F, w.uv, w.delta, w.bases)
        num, den, dd = SI.write_change(F.net, SI.vpd_sites(fv), fv.n, list(toks.split(kc.cache_batch)))
        ch = torch.zeros(fv.n, dtype=torch.bool)
        ch[sorted(fv.changed)] = True
    else:
        num, den, dd, ch, _ = NR.write_pass(F, M, toks, nfeat, kc.cache_batch)
    out["do_nothing"] = dn = {"n_changed": int(ch.sum()), "dd_max": float(dd.abs().max()), "E_b_K": SI.explained_share(num, den, dd, ch)}  # any K
    del F
    F = edited(w, cfg, "v2")
    out["identity"] = idn = identity(feat, w, kc, nfeat, F, toks, at)
    del F
    out["sanity"] = sanity(w, cfg, kc, M, toks)
    out |= {"floor": idn["max_dev"], "est_seconds_per_item": len(ch) * kc.derangements * min(kc.examples, len(toks)) / idn["cases_per_s"], "seconds": time.time() - t0}
    miss = [f"identity: {idn['flips']} argmax changes"] if idn["flips"] else []
    miss += [f"do-nothing: {k} {dn[k]}" for k, want in (("n_changed", 0), ("dd_max", 0.0), ("E_b_K", 1.0)) if dn[k] != want]
    log(f"stage 0 ({feat}): {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, f"stage 0 ({feat}): " + "; ".join(miss))
    write_stage(path, out | {"done": True})
    log(f"STAGE 0 DONE ({feat}): est. {out['est_seconds_per_item']:.0f} s per item")
    return float(out["floor"])


def read_item(feat: str, name: str, w: Any, cfg: Config, kc: Any, nfeat: Any, toks: Tensor, at: slice, floor: float) -> None:
    path = RESULTS / "items" / f"{name}_{feat}.json"
    if path.is_file() and "stopped" not in json.loads(path.read_text()):
        log(f"{name} at {feat}: done before, skipped")
        return
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    F = edited(w, cfg, name)
    r = reader(feat, w, cfg, kc, nfeat)(w.kl.M, F, toks, at)
    out = {"experiment": EXPERIMENT, "edit": name, "feat": feat} | record(r, cfg, kc, floor)
    np.savez_compressed(RESULTS / "large" / f"items_{name}_{feat}.npz", in_K=r.K.numpy(), first_flip=r.first_flip.numpy(), flips=r.flips.numpy(),
                        num=r.num.numpy(), den=r.den.numpy(), dd=r.dd.numpy(), changed=r.ch.numpy(), margin=r.margin.numpy())  # fmt: skip
    out |= {"seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    write_stage(path, out)
    log(f"{name} at {feat}: E_b over K {out['E_b_K']:.6f}, over all changed {out['E_b_changed']:.6f}; |K| {out['size_K']} "
        f"({out['n_changed_K']} changed of {out['n_changed']}); margin median {out['median_margin']:.4g}, min {out['min_margin']:.4g}; {out['seconds']:.0f} s")  # fmt: skip
    del F, r
    torch.cuda.empty_cache()


def summary(cfg: Config) -> None:
    got = {f"{r['edit']}_{r['feat']}": r for r in (json.loads(p.read_text()) for p in sorted((RESULTS / "items").glob("*.json")))}
    eb = {k: r["E_b_K"] for k, r in got.items() if "stopped" not in r}
    v, q = eb.get("v2_vpd"), eb.get("q2_vpd")
    out: dict[str, Any] = {"experiment": EXPERIMENT, "stage0": [json.loads(p.read_text()) for p in sorted(RESULTS.glob("stage0_*.json"))],
                           "table": {e: {f: eb.get(f"{e}_{f}") for f in FEATS} for e in EDITS}, "items": got}  # fmt: skip
    out["checks"] = {
        "v2_o2_neurons_at_least_0.9": [eb[k] >= 0.9 for k in ("v2_neurons", "o2_neurons") if k in eb],
        "q2_within_q_band_of_v2_vpd": abs(q - v) <= cfg.q_band if v is not None and q is not None else None,
        "v2_vpd_reads": None if v is None else "rival A (>= 0.5)" if v >= 0.5 else "rival B (< 0.3)" if v < 0.3 else "neither (0.3 to 0.5)",
    }  # fmt: skip
    write_stage(RESULTS / "summary.json", out)
    log(f"summary: {json.dumps(out['table'])}; {json.dumps(out['checks'])}")


def main(stage: str, feats: tuple[str, ...], tag: str) -> int:
    cfg, kc = load_config(), KN.load_config()
    if stage == "summary":
        summary(cfg)
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E257, E268, E277
    log(f"{EXPERIMENT} start, stage {stage}, feats {feats}, tag {tag}; settings {asdict(cfg)}; E268's scan settings {asdict(kc)}")
    for d in ("items", "large"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    w = KN.world()
    toks, at = data(w, cfg)
    nfeat = None
    if "neurons" in feats:
        c229 = NR.load_e229()
        nfeat = NR.E229.feat_of(c229, NR.E229.load_pct(c229), NR.load_config().band)
    floor = {f: stage0(f, w, cfg, kc, nfeat, toks, at, tag) for f in feats}
    if stage == "0":
        return 0
    for f in feats:
        for k, name in enumerate(EDITS):
            read_item(f, name, w, cfg, kc, nfeat, toks, at, floor[f])
            log(f"ITEM {k + 1} of {len(EDITS)} DONE ({name} at {f})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("0", "read", "summary", "config"), required=True)
    ap.add_argument("--feat", choices=(*FEATS, "both"), default="both")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<feat>_<tag>.json")
    args = ap.parse_args()
    if args.stage == "config":  # CPU dry run: the settings load
        print(json.dumps({"settings": asdict(load_config()), "E268_scan": asdict(KN.load_config()), "E277_band": NR.load_config().band}, indent=1))
        hard_exit(0)
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, FEATS if args.feat == "both" else (args.feat,), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

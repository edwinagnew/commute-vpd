"""E247 (cell P1.47; user, 2026-11-21 ~13:20, answer to ask 57: "i would do 2 first"): why the thin new edit (new_thin)
reads kept under wording A at the VPD featurization in E242. Plan reports/plans/2026-11-21_plan_thin-edit-diagnosis.md;
card reports/plans/2026-11-21_plan_thin-edit-diagnosis-card.md; settings, thresholds and rules in
results/predictions.json, which the run refuses unless committed unchanged.

Nothing is searched. Each item is rebuilt by E242's World, featurized by commute.kstar_vpd as in E242, and E242's saved
stood-in set S (K*'s changed members that do not keep F's mechanism, results/large/items_<item>.npz of E242, with E226's
or E242's saved K*) is put in again with commute.stand_in, in F's full forward, at E242's saved gains:
    q_fit  KL(F || F_S) / KL(F || M), gains fitted: must repeat E242's kl_at_found (the reproduction check)
    q_one  the same S with every gain set to 1 (M's mechanism unscaled): reading 1
    q_out  S and every changed variable outside K* stood in, gains fitted (descriptive: the change K* never asks about)
and, from one pass over every position of D in F's own run (commute.stand_in.write_change; the rows the gains are
fitted on), per variable the write change dd = |write_F - write_M|^2 and the part its gain explains (num - den)^2 / den:
    e_stood  share of S's write change that the gains explain: reading 2
    c_kept   share of the write change of K*'s changed members carried by the members that keep F's mechanism
             (beside their share by count); c_out, c_rem: shares of all changed variables' write change outside K*
             and in the site remainders (descriptive)
Items in `halves` are also read on E226's two halves of D, each with its own K* and E242's S for that half (q_fit, q_one).
E242's item controls (`controls`: unchanged gains exactly 1, unchanged stood in exactly F, every change at gain 1 gives
M's logits within revert_tol) run on every item. Stage 0, the do-nothing, runs first and alone: every gain exactly 1,
every write change exactly 0, every variable stood in gives KL exactly 0; any miss stops the job.

    sbatch --comment='E247' cluster/thin_edit_diagnosis.sbatch
"""

from __future__ import annotations

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
from commute import rule_retention as R
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

E242: Any = load_experiment("kstar_vpd_wording_a", "run")  # World, logits_at, controls, log/dump/stop, E226
E226 = E242.E226
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
S242, S226 = EXPERIMENTS / "kstar_vpd_wording_a" / "results", EXPERIMENTS / "kstar_vpd" / "results"
log, dump, stop, fname = E242.log, E242.dump, E242.stop, E242.fname


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    items: tuple[str, ...]  # read in this order after the do-nothing
    halves: tuple[str, ...]  # also read on E226's two halves of D
    target: str  # the item the filed rule reads

    def __post_init__(self) -> None:
        assert "do_nothing" not in self.items and set(self.halves) <= set(self.items) and self.target in self.halves


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    return config_from_settings(Config, s), pred["thresholds"]


def saved_sets(item: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """E242's arrays (gain, changed, <variant>_keeps_F) with K* per variant (<variant>_in_K), and E242's item record."""
    a = dict(np.load(S242 / "large" / f"items_{fname(item)}.npz"))
    kp = S226 / "large" / f"items_{fname(item)}.npz"
    k = np.load(kp if kp.is_file() else S242 / "large" / f"kstar_{fname(item)}.npz")
    a |= {f"{v}_in_K": k[f"{v}_in_K_polished"] for v in ("main", "half1", "half2") if f"{v}_in_K_polished" in k.files}
    return a, json.loads((S242 / "items" / f"{fname(item)}.json").read_text())


@dataclass
class Item:
    si: SI.StandIn
    fwd: Any  # x -> F's logits at the read positions
    batches: list[Tensor]
    lF: Tensor  # (N, ...) F's own logits at the read positions
    kl_fm: Tensor  # (N,) KL(F || M) per window
    batch: int
    dev: torch.device

    def ratio(self, idx: Tensor, stood: np.ndarray, g: Tensor) -> float:
        """KL(F || F with `stood` in at gains g) / KL(F || M), means over the windows idx."""
        self.si.g = g.to(self.dev, torch.float32)
        t = torch.from_numpy(stood).to(self.dev, torch.float32)
        toks = list(torch.cat(self.batches)[idx.to(self.dev)].split(self.batch))
        lF = list(self.lF[idx.to(self.dev)].split(self.batch))
        return SI.kl_mean(self.si, self.fwd, toks, lF, t) / float(self.kl_fm[idx].mean())


def shares(num: Tensor, den: Tensor, dd: Tensor, sets: dict[str, np.ndarray], rem: np.ndarray) -> dict[str, Any]:
    """Reading 2 and the descriptive shares, from the write-change sums (all fp64, every position of D)."""
    expl = torch.where(den > 0, (num - den) ** 2 / den.clamp(min=1e-300), 0.0).numpy()
    d = dd.numpy()
    st, kp, out, ch = sets["stood"], sets["kept"], sets["out"], sets["changed"]
    chK = st | kp
    return {"e_stood": float(expl[st].sum() / d[st].sum()), "e_kept": float(expl[kp].sum() / d[kp].sum()) if kp.any() else None,
            "c_kept": float(d[kp].sum() / d[chK].sum()), "c_kept_count": float(kp.sum() / chK.sum()),
            "c_out": float(d[out].sum() / d[ch].sum()), "c_out_count": float(out.sum() / ch.sum()),
            "c_rem": float(d[ch & rem].sum() / d[ch].sum()), "n_stood": int(st.sum()), "n_kept": int(kp.sum()), "n_out": int(out.sum())}  # fmt: skip


def read_item(w: Any, item: str, th: dict[str, Any], halves: bool) -> dict[str, Any]:
    path = RESULTS / "items" / f"{fname(item)}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return json.loads(path.read_text())
    t0, c226 = time.time(), E226.load_config()[0]
    F, M, toks, at, _ = w.make(item)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    sites = SI.vpd_sites(feat)
    num, den, dd = SI.write_change(F.net, sites, feat.n, list(toks.split(E242.load_config()[0].gain_batch)))
    g = torch.where(den > 0, num / den.clamp(min=1e-300), 1.0)
    batches = list(toks.split(c226.batch))
    with torch.no_grad():
        lF = [F.logits(x)[:, at] for x in batches]
        kl_fm = torch.cat([R.kl(lf, M.logits(x)[:, at]) for x, lf in zip(batches, lF, strict=True)]).cpu()
    si = SI.StandIn(F.net, sites, g, toks.device)
    rd = Item(si, partial(E242.logits_at, F, at), batches, torch.cat(lF), kl_fm, c226.batch, toks.device)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "n_vars": feat.n, "n_changed": len(feat.changed), "kl_fm": float(kl_fm.mean())}
    if item == "do_nothing":
        return stage0_checks(rd, feat, g, dd, out, t0)
    a, r242 = saved_sets(item)
    controls, miss = E242.controls(si, F, M, feat, g, batches, lF, at, E242.load_config()[1])
    gs = torch.from_numpy(a["gain"])
    out |= {"controls": controls, "gain_dev_saved": float((g - gs).abs().max()), "kl_fm_saved": r242["kl_fm"]}
    miss += [f"KL(F || M) {out['kl_fm']:.9g}, E242 {r242['kl_fm']:.9g}"] if not abs(out["kl_fm"] / r242["kl_fm"] - 1) <= th["kl_repro_rel"] else []
    ch, rem = a["changed"], np.isin(np.arange(feat.n), feat.rem)
    miss += ["E242's changed set differs"] if not np.array_equal(ch, np.isin(np.arange(feat.n), sorted(feat.changed))) else []
    perm = torch.randperm(len(toks), generator=torch.Generator().manual_seed(c226.split_seed))
    idx = {"main": torch.arange(len(toks)), "half1": perm[: len(toks) // 2], "half2": perm[len(toks) // 2 :]}
    for v in ("main", "half1", "half2") if halves else ("main",):
        K, keep = a[f"{v}_in_K"].astype(bool), a[f"{v}_keeps_F"].astype(bool)
        sets = {"stood": ch & K & ~keep, "kept": ch & K & keep, "out": ch & ~K, "changed": ch}
        r: dict[str, Any] = {"q_fit": rd.ratio(idx[v], sets["stood"], gs), "q_one": rd.ratio(idx[v], sets["stood"], torch.ones_like(gs))}
        saved = r242["searches"][v]["kl_at_found"] / r242["searches"][v]["kl_fm"]
        r["q_fit_e242"] = saved
        miss += [f"{v}: q_fit {r['q_fit']:.9g}, E242 {saved:.9g}"] if not abs(r["q_fit"] / saved - 1) <= th["kl_repro_rel"] else []
        if v == "main":
            r["q_out"] = rd.ratio(idx[v], sets["stood"] | sets["out"], gs)
            r |= shares(num, den, dd, sets, rem)
        out[v] = r
        log(f"  {item} {v}: " + ", ".join(f"{k} {x:.6g}" if isinstance(x, float) else f"{k} {x}" for k, x in r.items()))
    if miss:
        stop(RESULTS / "items" / f"{fname(item)}.stopped.json", out, f"{item}: " + "; ".join(miss))
    np.savez_compressed(RESULTS / "large" / f"items_{fname(item)}.npz", num=num.numpy(), den=den.numpy(), dd=dd.numpy())
    out |= {"seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, out)
    log(f"{item}: done in {out['seconds']:.0f} s; peak GPU {out['peak_gpu_gb']:.2f} GB")
    si.close()
    return out


def stage0_checks(rd: Item, feat: KV.Feat, g: Tensor, dd: Tensor, out: dict[str, Any], t0: float) -> dict[str, Any]:
    """The do-nothing (F = M): every gain exactly 1, every write change exactly 0, every variable stood in exactly F."""
    t1 = time.time()
    rd.si.g = g.to(rd.dev, torch.float32)
    kl_all = SI.kl_mean(rd.si, rd.fwd, rd.batches, list(rd.lF.split(rd.batch)), torch.ones(feat.n, device=rd.dev))
    out |= {"gain_dev": float((g - 1).abs().max()), "dd_max": float(dd.abs().max()), "kl_all_stood_in": kl_all,
            "seconds_per_eval": time.time() - t1, "seconds": time.time() - t0}  # fmt: skip
    rd.si.close()
    return out


def stage0(w: Any, th: dict[str, Any], n_evals: int) -> None:
    path = RESULTS / "stage0.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log("stage 0 done before")
        return
    out = read_item(w, "do_nothing", th, False)
    hours = out["seconds_per_eval"] * n_evals / 3600
    out["projected_eval_hours"] = hours
    miss = [k for k in ("gain_dev", "dd_max", "kl_all_stood_in") if out[k] != 0] + (["changed variables"] if out["n_changed"] else [])
    miss += [f"projected {hours:.2f} h of evaluations over {th['hours_max']} h"] if hours > th["hours_max"] else []
    log(f"stage 0: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def score(cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    """The filed rule on the target item (predictions.json "rules"), and every item's readings beside it."""
    items = {i: json.loads((RESULTS / "items" / f"{fname(i)}.json").read_text()) for i in cfg.items}
    m = items[cfg.target]["main"]
    rise = m["q_one"] - m["q_fit"]
    halves = [items[cfg.target][h]["q_one"] - items[cfg.target][h]["q_fit"] for h in ("half1", "half2")]
    if rise > th["rise_min"] and m["e_stood"] >= th["share_min"]:
        verdict = "rival a: the gains carry the new computation"
    elif rise <= th["rise_min"] and m["c_kept"] >= th["share_min"]:
        verdict = "rival b: the change sits in the parts that keep F's mechanism; R_A counts parts"
    else:
        verdict = "neither rival's pair of readings holds: reported as read"
    keys = ("q_fit", "q_one", "q_out", "e_stood", "e_kept", "c_kept", "c_kept_count", "c_out", "c_out_count", "c_rem")
    return {"experiment": EXPERIMENT, "target": cfg.target, "rise": rise, "rise_halves": halves, "half_spread": abs(halves[0] - halves[1]) / 2**0.5,
            "verdict": verdict, "items": {i: {k: d["main"][k] for k in keys} for i, d in items.items()}}  # fmt: skip


def main() -> int:
    cfg, th = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226 and E242
    log(f"{EXPERIMENT} start; settings {asdict(cfg)}; thresholds {th}")
    (RESULTS / "items").mkdir(parents=True, exist_ok=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    c242 = E242.load_config()[0]
    w = E242.World(c242)
    w.bases = KV.bases(w.uv, E226.load_config()[0].split_tol)
    r = json.loads((S242 / "items" / f"{fname(c242.random_kl)}.json").read_text())  # E242's matched scale, not matched again
    w.kl_x = {k: r[k] for k in ("x", "kl_matched", "match", "match_evals")}
    stage0(w, th, sum(4 + 4 * (i in cfg.halves) for i in cfg.items))
    for k, item in enumerate(cfg.items):
        read_item(w, item, th, item in cfg.halves)
        log(f"ITEM {k + 1} of {len(cfg.items)} DONE ({item})")
    s = score(cfg, th)
    dump(RESULTS / "summary.json", s)
    log(f"summary: {json.dumps(s)}")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

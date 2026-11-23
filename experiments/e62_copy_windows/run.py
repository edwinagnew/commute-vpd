"""E281 (cell P1.57; whip-lead, 2026-11-23, after redteam-lead's mixed-D note): E_b over the needed set K for E62's six
states on the copy windows of their D alone. Descriptive; card experiments/e62_copy_windows/card.md; settings, the
filed rule and expected values in results/predictions.json, which the run refuses unless committed unchanged.

Per state (e62/copy_s0-2, e62/lookup_s0-2 as E257's World builds them, through `kstar_needed.featurized`; D is E187's
read half, copy windows mixed with cue-free and far windows): the copy windows (E280's `targets`: marker last, A once
before 'A M'); two `stand_in.write_change` passes, on all of D and on the copy windows; K_copy by
`commute.needed.needed_set` with E268's settings, x and x' both from the copy windows (the reading the rule reads);
E_b over K_copy on the copy windows, beside E_b over E275's saved K (unchanged, found on all of D) there and on all of
D, and the copy windows' share of K's dd.

Stage 0 first in every job (skipped once results/stage0_<tag>.json says done); a miss stops the job: the do-nothing
(E257's) has no changed variable, dd max 0 and E_b over K 1; the control's all-D pass repeats E275's E_b over K and
E257's E_b over K* within repeat_tol (10 times the measured cross-device repeat of that pass; the same-card repeat at half
the batch is recorded beside it).

    python experiments/e62_copy_windows/run.py --stage read --shard 0 2 --tag c0      (and --shard 1 2 --tag c1)
    python experiments/e62_copy_windows/run.py --stage summary          (no GPU)
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

from commute import needed as IX
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)

KN: Any = load_experiment("kstar_needed", "run")  # E268: world, load_config (scan settings), featurized, saved, fname, log
HE: Any = load_experiment("head_edit", "run")  # E280: targets (the copy windows of E62's D)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E275 = EXPERIMENTS / "claim2_needed" / "results"
log, fname = KN.log, KN.fname
MIXED = {"copy": [0.03761000378175188, 0.039910438669951795, 0.03617183164425989],  # E275's E_b over K on all of D
         "lookup": [0.03994768024927968, 0.04308385745860625, 0.042870824136771526]}  # fmt: skip


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    states: tuple[str, ...]
    control: str
    too_small: int  # on K_copy
    ratio: float  # the rule's bar
    repeat_tol: float  # absolute, on E_b: 10 times the measured cross-device repeat (predictions.json)

    def __post_init__(self) -> None:
        assert self.control in self.states and len(self.states) == 6 and all(s.startswith("e62/") for s in self.states)
        assert rule(MIXED["copy"], MIXED["lookup"], self.ratio) == "rival B", "the rule must read no separation on the mixed-D values"


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def rule(copy: list[float], lookup: list[float], ratio: float) -> str:
    """Rival A when mean(copy) / mean(lookup) >= ratio and copy > lookup on every seed (seeds paired by index)."""
    up = sum(copy) / sum(lookup) >= ratio and all(c > lk for c, lk in zip(copy, lookup, strict=True))
    return "rival A" if up else "rival B"


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    write_stage(path, out | {"stopped": why})
    log(f"STOP: {why}")
    hard_exit(1)


# ---- one state ------------------------------------------------------------------------------------------------------


def write_pass(F: Any, feat: Any, toks: Tensor, batch: int) -> tuple[Tensor, Tensor, Tensor]:
    return SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(batch)))


def changed_mask(feat: Any) -> Tensor:
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    return ch


def saved_K(item: str) -> tuple[Tensor, dict[str, Any]]:
    z = np.load(E275 / "large" / f"items_{fname(item)}.npz")
    return torch.from_numpy(z["in_K"]).bool(), json.loads((E275 / "items" / f"{fname(item)}.json").read_text())


def read_item(w: Any, item: str, cfg: Config, kc: Any) -> None:
    path = RESULTS / "items" / f"{fname(item)}.json"
    if path.is_file() and "stopped" not in json.loads(path.read_text()):
        log(f"{item}: done before, skipped")
        return
    t0 = time.time()
    F, feat, toks, at = KN.featurized(w, item)
    ch, (K, rec) = changed_mask(feat), saved_K(item)
    assert len(K) == feat.n, f"{item}: E275's K ({len(K)}) does not line up with the featurization ({feat.n})"
    copy, _ = HE.targets(toks)
    tc = toks[copy.to(toks.device)]
    allD, cw = write_pass(F, feat, toks, kc.cache_batch), write_pass(F, feat, tc, kc.cache_batch)
    nd = IX.needed_set(F, feat, tc, at, kc.derangements, kc.examples, kc.seed, kc.rows, kc.logit_rows, kc.cache_batch, log)
    Kc, sel = nd.K, K & ch
    out = {"experiment": EXPERIMENT, "item": item, "n_vars": feat.n, "n_changed": int(ch.sum()), "n_windows": len(toks), "n_copy_windows": len(tc),
           "size_Kcopy": int(Kc.sum()), "n_changed_Kcopy": int((Kc & ch).sum()), "too_small": int(Kc.sum()) < cfg.too_small, "n_pairs": nd.n_pairs,
           "E_b_Kcopy": SI.explained_share(*cw, Kc & ch), "empty_Kcopy": not bool((Kc & ch).any()),
           "size_K": int(K.sum()), "size_both": int((K & Kc).sum()), "E_b_K_copy": SI.explained_share(*cw, sel), "E_b_K_all": SI.explained_share(*allD, sel),
           "E_b_K_all_E275": rec["E_b_K"], "E_b_changed_copy": SI.explained_share(*cw, ch), "E_b_changed_all": SI.explained_share(*allD, ch),
           "copy_dd_share_K": float(cw[2][sel].sum() / allD[2][sel].sum()),
           "median_margin": float(nd.run.margin.median()), "min_margin": float(nd.run.margin.min()),
           "by_derangement": [{"size_Kcopy": int((Kc & (nd.first_flip <= p)).sum()), "E_b_Kcopy": SI.explained_share(*cw, Kc & (nd.first_flip <= p) & ch)}
                              for p in range(kc.derangements)],
           "K_source": str((E275 / "large" / f"items_{fname(item)}.npz").relative_to(EXPERIMENTS.parent)),
           "cases_per_s": nd.scan.rows / nd.scan.seconds, "seconds": time.time() - t0}  # fmt: skip
    np.savez_compressed(RESULTS / "large" / f"items_{fname(item)}.npz", in_Kcopy=Kc.numpy(), first_flip=nd.first_flip.numpy(), flips=nd.scan.flips.numpy(),
                        copy_windows=copy.cpu().numpy(), num_copy=cw[0].numpy(), den_copy=cw[1].numpy(), dd_copy=cw[2].numpy())  # fmt: skip
    write_stage(path, out)
    log(f"{item}: E_b over K_copy {out['E_b_Kcopy']:.6f} (|K_copy| {out['size_Kcopy']}), over E275's K on copy windows {out['E_b_K_copy']:.6f}, "
        f"on all of D {out['E_b_K_all']:.6f} (E275 {rec['E_b_K']:.6f}); {out['seconds']:.0f} s")  # fmt: skip
    del F, nd
    torch.cuda.empty_cache()


# ---- stage 0, summary, main -----------------------------------------------------------------------------------------


def stage0(w: Any, cfg: Config, kc: Any, tag: str) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    t0 = time.time()
    out: dict[str, Any] = {"experiment": EXPERIMENT, "card": torch.cuda.get_device_name(), "control": cfg.control}
    F, feat, toks, _ = KN.featurized(w, "do_nothing")
    num, den, dd = write_pass(F, feat, toks, kc.cache_batch)
    del F
    out["do_nothing"] = dn = {"n_changed": len(feat.changed), "dd_max": float(dd.abs().max()), "E_b_K": SI.explained_share(num, den, dd, changed_mask(feat))}  # any K
    F, feat, toks, _ = KN.featurized(w, cfg.control)
    ch, (K, rec), a = changed_mask(feat), saved_K(cfg.control), KN.saved(cfg.control, "items")
    p, q = write_pass(F, feat, toks, kc.cache_batch), write_pass(F, feat, toks, kc.cache_batch // 2)
    del F
    same = len(a["dd"]) == feat.n and torch.equal(a["changed"], ch)
    eK, eKs = SI.explained_share(*p, K & ch), SI.explained_share(*p, a["Kstar"] & ch)
    floor = abs(SI.explained_share(*q, K & ch) - eK)
    bar = cfg.repeat_tol
    out["repeat"] = r = {"same_variables": same, "E_b_K": eK, "E_b_K_E275": rec["E_b_K"], "E_b_Kstar": eKs, "E_b_Kstar_E257": a["E_b_saved"],
                         "dev_K": abs(eK - rec["E_b_K"]), "dev_Kstar": abs(eKs - a["E_b_saved"]), "floor": floor, "bar": bar,
                         "dd_rel_l1": float((p[2] - a["dd"]).abs().sum() / a["dd"].abs().sum()) if same else None}  # fmt: skip
    out["seconds"] = time.time() - t0
    miss = [f"do-nothing: {k} {dn[k]}" for k, want in (("n_changed", 0), ("dd_max", 0.0), ("E_b_K", 1.0)) if dn[k] != want]
    miss += [f"{cfg.control}: saved arrays do not line up with the featurization"] if not same else []
    miss += [f"{cfg.control}: {k} {r[k]!r} over the bar {bar!r}" for k in ("dev_K", "dev_Kstar") if not r[k] <= bar]
    log(f"stage 0: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    write_stage(path, out | {"done": True})
    log("STAGE 0 DONE")


def summary(cfg: Config) -> None:
    got = {r["item"]: r for r in (json.loads(p.read_text()) for p in sorted((RESULTS / "items").glob("*.json"))) if "stopped" not in r}
    side = lambda arm, key: [got[s][key] if s in got else None for s in cfg.states if s.startswith(f"e62/{arm}_")]
    out: dict[str, Any] = {"experiment": EXPERIMENT, "stage0": [json.loads(p.read_text()) for p in sorted(RESULTS.glob("stage0_*.json"))], "items": got}
    for name, key in (("Kcopy_copy_windows", "E_b_Kcopy"), ("K_copy_windows", "E_b_K_copy"), ("K_all_of_D_E275", "E_b_K_all_E275")):
        c, lk = side("copy", key), side("lookup", key)
        cf, lf = [v for v in c if v is not None], [v for v in lk if v is not None]
        out[name] = {"copy": c, "lookup": lk, "ratio": sum(cf) / sum(lf) if len(cf) == len(lf) == 3 else None}
    c, lk = out["Kcopy_copy_windows"]["copy"], out["Kcopy_copy_windows"]["lookup"]
    out["reads"] = rule(c, lk, cfg.ratio) if None not in c + lk else None  # the filed rule: E_b over K_copy only
    out["too_small"] = sorted(i for i, r in got.items() if r["too_small"])
    write_stage(RESULTS / "summary.json", out)
    log(f"summary: {json.dumps({k: v for k, v in out.items() if k not in ('stage0', 'items')})}")


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    cfg, kc = load_config(), KN.load_config()
    if stage == "summary":
        summary(cfg)
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E257, E268, E275
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}; settings {asdict(cfg)}; E268's scan settings {asdict(kc)}")
    for d in ("items", "large"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    w = KN.world()
    stage0(w, cfg, kc, tag)
    if stage == "read":
        todo = list(cfg.states)[shard[0] :: shard[1]]
        for k, item in enumerate(todo):
            read_item(w, item, cfg, kc)
            log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("0", "read", "summary", "config"), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th state from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    if args.stage == "config":  # CPU dry run: settings (Config checks the rule on the mixed-D values) and E275's K files
        c = load_config()
        print(json.dumps({"settings": asdict(c), "rule_on_mixed_D": rule(MIXED["copy"], MIXED["lookup"], c.ratio),
                          "K_sizes": {s: int(saved_K(s)[0].sum()) for s in c.states}}, indent=1))  # fmt: skip
        hard_exit(0)
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

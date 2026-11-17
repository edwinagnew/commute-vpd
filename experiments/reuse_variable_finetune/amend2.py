"""E139 Amendment 2 (cell P1.4): the Always on imitators trained again, to a much smaller leftover, and read at three points of
training. Plan: reports/predictions/2026-11-10_prediction_e139-amendments-1-2.md; filed rules: results/predictions_amend2.json
(the run refuses unless it is committed unchanged).

What differs from E139's `train`, and nothing else: 3,000 steps instead of 300, peak rate 1e-5 instead of 3e-5 (warm-up ratio 0.1
kept, so 300 warm-up steps, then cosine to 0, `qwen_sft.lr_at`), the weights saved at steps 300, 1000 and 3000, and the step-10
overshoot check compares like with like: KL(F || ft) against KL(F || M), and KL(M || ft) against KL(M || F) (E139 compared
KL(M || ft) with KL(F || M)). Same seeds (batch draws), loss, batch, clip and windows as E139.

One job per seed: train (about 14 min), then read step 3000 in full as E139 (G at k = 4, four sites, three starts; H at five
sites), then steps 1000 and 300 with G at in2 only (E139's best site on every seed) and H at five sites. At each checkpoint
also the gap, the distance to the pair, and E137's witness read against it (`amend1.read_witness`). The witness is
regenerated first and must repeat its G against the pair (control, exit 1). results/amend2_always_on_s<seed>.json is
written after each search; results/large/amend2_always_on_s<seed>_step<N>.pt holds the weights. A rerun resumes (training
restarts from scratch if any state is missing). Exit 0 done, 1 a stop, 3 the wall limit (resubmit the same line).

    sbatch --comment='E139 amend2' cluster/reuse_variable_finetune_amend.sbatch amend2 0     (and 1, 2: one job per seed)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.hidden_maps import kl, run
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_sft import lr_at
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139's Config, distances, clone_net, last, log, dump, stop, budget
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup, witness, read_witness
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions_amend2.json"


@dataclass(frozen=True)
class Config(R.Config):
    steps: int = 3000  # 10x E139; sized so one seed's training and reads fit one 1:55 debug job (prediction page, cost)
    lr: float = 1e-5  # LLaMA-Factory's documented full-SFT peak (`qwen_sft.lr_at`); E139's loss rose at 3e-5 on every seed
    warmup: int = 300  # warm-up ratio 0.1, as E139
    early_steps: tuple[int, ...] = (1, 2, 3, 5, 10, 30, 100, 300, 1000, 2000, 3000)
    saves: tuple[int, ...] = (3000, 1000, 300)  # read in this order: the scored state first

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.warmup * 10 == self.steps and self.saves[0] == self.steps and set(self.saves) <= set(self.early_steps)


def state_path(cfg: Config, item: str, step: int) -> Path:
    return cfg.results / "large" / f"amend2_{item}_step{step}.pt"


def save(ft: Net, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(ft.net.state_dict(), tmp)
    os.replace(tmp, path)


def train(net: Net, F: VR.Target, fit: Tensor, check: Tensor, cfg: Config, item: str, path: Path, out: dict[str, Any]) -> None:
    """E139's `train` with Config's recipe, the states saved at `cfg.saves`, and the step-10 check in matching directions."""
    ft = R.clone_net(net)
    params = list(ft.net.parameters())
    for x in params:
        x.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
    gen = torch.Generator().manual_seed(300_000 + zlib.crc32(item.encode()) % 65_536)  # E139's seed for this item
    n, curve, early = fit.shape[0], [], []
    lm, lf = R.last(VR.net_target(net), check, cfg.read_batch), R.last(F, check, cfg.read_batch)
    kl_fm, kl_mf = float(kl(lf, lm).mean()), float(kl(lm, lf).mean())
    for t in range(1, cfg.steps + 1):
        rate = lr_at(t, cfg.lr, cfg.warmup, cfg.steps, "hf_cosine")
        for g in opt.param_groups:
            g["lr"] = rate
        toks = fit[torch.randint(0, n, (cfg.batch,), generator=gen)]
        with torch.no_grad():
            target = F(toks)[0][:, -1]
        loss = kl(target, run(ft, toks, grad=True)[0][:, -1]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.clip)
        opt.step()
        curve.append(float(loss.detach()))
        if t in cfg.early_steps:
            lft = R.last(VR.net_target(ft), check, cfg.read_batch)
            early.append({"step": t, "lr": rate, "kl_to_F": float(kl(lf, lft).mean()), "kl_from_M": float(kl(lm, lft).mean())})
            R.log(f"  step {t}: loss {curve[-1]:.5f}, KL(F||ft) {early[-1]['kl_to_F']:.5f}, KL(M||ft) {early[-1]['kl_from_M']:.3e}, rate {rate:.2e}")
        if t == cfg.early_bar_step and not (early[-1]["kl_to_F"] < kl_fm and early[-1]["kl_from_M"] <= kl_mf):
            R.stop(path, {**out, "train": {"curve": curve, "early": early, "kl_F_M": kl_fm, "kl_M_F": kl_mf}},
                   f"overshoot at step {t}: KL(F||ft) not below KL(F||M), or KL(M||ft) above KL(M||F)")
        if t in cfg.saves:
            save(ft, state_path(cfg, item, t))
    out["train"] = {"curve": curve, "early": early, "kl_F_M": kl_fm, "kl_M_F": kl_mf, "steps": cfg.steps, "lr": cfg.lr}
    R.dump(path, out)


def search_g(net: Net, cfg: Config, tag: str, sites: tuple[str, ...], fit: Tensor, read: Tensor, F: VR.Target, gap: float, rec: dict[str, Any], save_out: Any) -> None:
    """E139's G search (`run.search_g`) over `sites`, writing into one checkpoint's record."""
    for site in sites:
        for start in range(cfg.starts):
            key = f"k{cfg.k}/{site}/s{start}"
            if key in rec["G"]:
                continue
            R.budget(cfg)
            t0 = time.monotonic()
            found = VR.g_search(net, fit, site, cfg.k, start, F, cfg, 200_000 + zlib.crc32(f"{tag}/{key}".encode()) % 65_536)
            A, c = found["A"], found["c"]
            r = VR.read_pair(net, read, site, found["Q"], lambda v, _s, A=A, c=c: v @ A + c, F, cfg)
            rec["G"][key] = {"G": 1 - r["d"] / gap, "reversion": 1 - r["d_rev"] / gap, "mismatch": r["mismatch"], "fit_kl": found["last_kl"], "s": time.monotonic() - t0}
            save_out()
            R.log(f"G {tag}/{key}: G {rec['G'][key]['G']:.4f}, reversion {rec['G'][key]['reversion']:.4f}, mismatch {r['mismatch']:.4f}")


def search_h(net: Net, cfg: Config, tag: str, fit: Tensor, read: Tensor, F: VR.Target, gap: float, rec: dict[str, Any], save_out: Any) -> None:
    """E139's H search (`run.search_h`), writing into one checkpoint's record."""
    missing = [s for s in VR.H_SITES if s not in rec["H"]]
    if not missing:
        return
    R.budget(cfg)
    ridge, X_val, Y_val, ps = VR.h_fit(net, fit, F, cfg, R.log)
    n_val = fit.shape[0] * 4 // 5
    for site in missing:
        R.budget(cfg)
        beta, lam, _mse = VR.ridge_select(ridge, site, ps[site], X_val, Y_val[site], cfg.ridge_grid)
        b, y, info = VR.h_refine(net, fit[:n_val], fit[n_val:], site, ridge, beta, F, cfg, 137_800 + VR.H_SITES.index(site))
        r = VR.read_pair(net, read, site, None, VR.ridge_s(ridge, site, b, y), F, cfg)
        rec["H"][site] = {"H": 1 - r["d"] / gap, "reversion": 1 - r["d_rev"] / gap, "mismatch": r["mismatch"], "lam": lam, **info}
        save_out()
        R.log(f"H {tag}/{site}: H {rec['H'][site]['H']:.4f}, reversion {rec['H'][site]['reversion']:.4f}, mismatch {r['mismatch']:.4f}")
    del ridge, X_val, Y_val


def read_step(net: Net, cfg: Config, th: dict[str, Any], item: str, step: int, w: dict[str, Any], F: VR.Target, fit: Tensor, read: Tensor, path: Path, out: dict[str, Any]) -> None:
    rec = out["steps"].setdefault(str(step), {"G": {}, "H": {}})
    if rec.get("complete"):
        return
    ft = R.clone_net(net)
    ft.net.load_state_dict(torch.load(state_path(cfg, item, step), map_location=net.device))
    target = VR.net_target(ft)
    if "dist" not in rec:
        rec["dist"] = R.distances(net, F, ft, read, cfg)
        R.log(f"{item}/step{step}: {rec['dist']}, leftover share {rec['dist']['tv_to_F'] / max(rec['dist']['gap'], 1e-12):.4f}")
        R.dump(path, out)
    gap = rec["dist"]["gap"]
    if gap < th["floor_gap"]:
        rec.update(no_reading="gap under the floor", complete=True)
        R.dump(path, out)
        return
    if "witness" not in rec:
        rec["witness"] = A1.read_witness(net, read, w, target, gap, cfg)
        R.log(f"{item}/step{step}: witness {rec['witness']}")
        R.dump(path, out)
    tag = f"amend2/{item}/step{step}"
    search_g(net, cfg, tag, VR.SITES if step == cfg.steps else ("in2",), fit, read, target, gap, rec, lambda: R.dump(path, out))
    search_h(net, cfg, tag, fit, read, target, gap, rec, lambda: R.dump(path, out))
    rec["complete"] = True
    R.dump(path, out)
    del ft, target
    torch.cuda.empty_cache()


def main(seed: int, net: Net) -> int:
    cfg, th = Config(), json.loads(PRED.read_text())["thresholds"]
    if not json.loads((cfg.results / "control.json").read_text()).get("passed"):
        R.log("E139's control has not passed")
        return 1
    item = f"always_on_s{seed}"
    path = cfg.results / f"amend2_{item}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"experiment": f"{R.EXPERIMENT} amend2", "item": item, "seed": seed, "steps": {}}
    sets, task, p = A1.setup(net, cfg)
    fit, read = VR.pair_windows(cfg, "Always on", sets, task)
    F = VR.piece_target(net, p, "Always on")
    w = A1.witness(net, fit, F, cfg)
    ctrl = A1.read_witness(net, read, w, F, th["pair_gap"], cfg)
    out["control"] = ctrl
    R.log(f"control: witness against the pair {ctrl}")
    if abs(ctrl["G"] - th["witness_G"]) > th["replay_tol"]:
        R.stop(path, out, "the regenerated witness does not repeat E137's G against the pair")
    (cfg.results / "large").mkdir(parents=True, exist_ok=True)
    if "train" not in out or not all(state_path(cfg, item, s).is_file() for s in cfg.saves):
        R.log(f"training {item}: {cfg.steps} steps, peak {cfg.lr}")
        train(net, F, fit, read[: cfg.check_n], cfg, item, path, out)
        torch.cuda.empty_cache()
    try:
        for step in cfg.saves:
            read_step(net, cfg, th, item, step, w, F, fit, read, path, out)
    except R.MoreToDo:
        R.log("more to do: wall limit reached, saved; resubmit the same line to resume")
        return 3
    out["complete"] = True
    R.dump(path, out)
    R.log(f"AMEND2 {item} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int, choices=(0, 1, 2))
    seed_ = ap.parse_args().seed
    require_filed(PRED)
    code = 1
    try:
        code = main(seed_, Net("cuda"))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

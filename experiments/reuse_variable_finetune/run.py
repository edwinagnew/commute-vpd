"""R2 (cell P1.4): ordinary fine-tunes of all weights of the 67M, each trained to imitate one of the four pairs E137 built (R1),
read with E137's searches at k = 4. Plan: reports/plans/2026-11-09_plan_r2-finetunes-imitating-pairs.md. Rules and cut-offs:
results/predictions.json (the run refuses unless it is committed unchanged). The reading code is `commute.variable_read`, moved
from E137's folder; F is here M + its piece for the target of training and the fine-tuned copy of M for the readings.

A job runs the control or a list of items, so several jobs run side by side, each writing its own files:
    run.py control                      stage 0, alone, first (results/control.json; exit 1 if it fails)
    run.py xor:0 xor:1 cue:2 ...        items pair:seed (always_on, cue, xor, table; seeds 0, 1, 2); refuses without a passed control
    run.py all                          every item not finished

Stage 0 (the W0 control, alone): a fine-tune at zero steps is M, so its gap is at the float floor and under the 0.02 floor (no
reading); the rebuilt pairs read R1's gaps; R1's G and H of one key (Always on, in2) replay through the moved code.
An item trains (or loads) the fine-tune, checks its first steps and, for the XOR, whether it learned the XOR at all; then the
G search at k = 4, 4 sites, 3 starts; then H at 5 sites; then done. results/<pair>_s<seed>.json is written after each search,
results/large/<pair>_s<seed>.pt holds the fine-tuned weights. A rerun resumes.
Exit 0 done, 1 a stop or refusal, 3 more to do (wall limit; resubmit).

    sbatch cluster/reuse_variable_finetune.sbatch control
    sbatch cluster/reuse_variable_finetune.sbatch always_on:0 cue:0 table:0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.hidden_maps import add_last, chunks, kl, run, tv
from commute.jobs import hard_exit, require_filed
from commute.qwen_sft import lr_at
from commute.steering import Net
from commute.variable_read import PAIRS, Sizes, Target

EXPERIMENT = "E???"  # The lead fills this in when the cell is claimed (`fleet/ecode`); only the log line reads it.
HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "reuse_variable_test" / "results"  # R1's pieces and stage files, read only
PRED = HERE / "results" / "predictions.json"
SLUG = {"always_on": "Always on", "cue": "Cue", "xor": "XOR", "table": "Table"}
SEEDS = (0, 1, 2)
_t0 = time.monotonic()


class MoreToDo(Exception):
    pass


@dataclass(frozen=True)
class Config(Sizes):
    results: Path = HERE / "results"
    steps: int = 300  # `finetune.ArmAConfig`'s 300 steps
    lr: float = 3e-5  # `finetune.ArmAConfig`'s Adam rate; the peak, reached after the warm-up
    warmup: int = 30  # warm-up ratio 0.1 of `qwen_sft.lr_at` (LLaMA-Factory's documented setting), then cosine to 0
    clip: float = 1.0  # `finetune.ArmAConfig`
    batch: int = 1024  # windows per step, one position each: as R1's g_batch (arm A's 32 windows of 41 tokens read ~1,300 positions)
    check_n: int = 4096  # windows of the read half used for the training reads
    early_steps: tuple[int, ...] = (1, 2, 3, 5, 10, 30, 100, 300)
    early_bar_step: int = 10
    wall_limit_s: int = 6300  # stop cleanly with exit 3 before the 1:55 limit, leaving one H site's time (~140 s on an L40)

    def __post_init__(self) -> None:
        super().__post_init__()
        assert 0 < self.warmup < self.steps and self.steps == self.early_steps[-1] and self.early_bar_step in self.early_steps
        assert self.batch <= self.D // 2 and self.check_n <= self.D // 2


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _t0:7.0f}s] {msg}", flush=True)


def dump(path: Path, out: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    os.replace(tmp, path)


def budget(cfg: Config) -> None:
    if time.monotonic() - _t0 > cfg.wall_limit_s:
        raise MoreToDo


def last(F: Target, windows: Tensor, batch: int) -> Tensor:
    return torch.cat([F(t)[0][:, -1] for t in chunks(windows, batch)])


def m_of(net: Net) -> Target:
    return VR.net_target(net)


def clone_net(net: Net) -> Net:
    """A second copy of M's weights with the same `run`: a `Net` sharing M's ComponentModel holder, its `target_model` deep-copied."""
    ft = copy.copy(net)
    ft.net = copy.deepcopy(net.net).requires_grad_(False)
    return ft


def stop(path: Path, out: dict[str, Any], msg: str) -> None:
    out["stopped"] = msg
    dump(path, out)
    log(f"STOPPED: {msg}")
    raise SystemExit(1)


# ---- stage 0: the W0 control, alone ------------------------------------------------------------------------------


def _floor(net: Net, windows: Tensor, cfg: Config) -> float:
    """The float floor between two identical runs of M at different batch sizes (largest per-window distance)."""
    return float(tv(last(m_of(net), windows, cfg.read_batch), last(m_of(net), windows, cfg.read_batch // 2)).max())


def _replay_g(net: Net, fit: Tensor, read: Tensor, F: Target, gap: float, cfg: Config) -> float:
    """R1's `Always on/k4/in2/s2` through the moved code: same key, same seed, read G (R1 read 0.9225, `stage2.json`)."""
    found = VR.g_search(net, fit, "in2", 4, 2, F, cfg, 137_000 + zlib.crc32(b"Always on/k4/in2/s2") % 65_536)
    A, c = found["A"], found["c"]
    r = VR.read_pair(net, read, "in2", found["Q"], lambda v, _s, A=A, c=c: v @ A + c, F, cfg)
    return 1 - r["d"] / gap


def _replay_h(net: Net, fit: Tensor, read: Tensor, F: Target, gap: float, cfg: Config) -> float:
    """R1's `Always on/in2` H through the moved code, same seeds. Deterministic ridge, whose refinement R1's log shows it rejected."""
    ridge, X_val, Y_val, ps = VR.h_fit(net, fit, F, cfg, log)
    beta, _lam, _mse = VR.ridge_select(ridge, "in2", ps["in2"], X_val, Y_val["in2"], cfg.ridge_grid)
    n_val = fit.shape[0] * 4 // 5
    b, y, _ = VR.h_refine(net, fit[:n_val], fit[n_val:], "in2", ridge, beta, F, cfg, 137_800 + VR.H_SITES.index("in2"))
    return 1 - VR.read_pair(net, read, "in2", None, VR.ridge_s(ridge, "in2", b, y), F, cfg)["d"] / gap


def control(net: Net, cfg: Config, th: dict[str, Any], sets: VR.Sets, task: VR.Task, p: VR.Pieces) -> None:
    path = cfg.results / "control.json"
    if path.is_file() and json.loads(path.read_text()).get("passed"):
        log("control already passed")
        return
    r1 = json.loads((R1 / "stage0.json").read_text())
    out: dict[str, Any] = {"pairs": {}, "passed": False}
    ft = clone_net(net)  # a fine-tune at zero steps
    for pair in PAIRS:
        _, read = VR.pair_windows(cfg, pair, sets, task)
        if pair == PAIRS[0]:
            out["floor_runs_tv"] = _floor(net, read[:8192], cfg)
        F = VR.piece_target(net, p, pair)
        lm = last(m_of(net), read, cfg.read_batch)
        lft = last(VR.net_target(ft), read, cfg.read_batch)
        lf = last(F, read, cfg.read_batch)
        w0_gap, w0_max = float(tv(lm, lft).mean()), float(tv(lm, lft).max())
        target_gap = float(tv(lm, lf).mean())
        rec = {"w0_gap": w0_gap, "w0_max_tv": w0_max, "target_gap": target_gap, "r1_gap": r1["pairs"][pair]["gap"], "w0_reading": w0_gap >= th["floor_gap"]}
        out["pairs"][pair] = rec
        log(f"control {pair}: W0 gap {w0_gap:.3e} (max TV {w0_max:.3e}), target gap {target_gap:.5f} (R1 {rec['r1_gap']:.5f}), W0 reading {rec['w0_reading']}")
        if w0_max > out["floor_runs_tv"] or rec["w0_reading"] or abs(target_gap - rec["r1_gap"]) > th["gap_replay_tol"]:
            stop(path, out, f"{pair}: the W0 fine-tune is off the float floor, reads at or above the floor, or the rebuilt pair's gap misses R1's")
    fit, read = VR.pair_windows(cfg, "Always on", sets, task)
    F, gap = VR.piece_target(net, p, "Always on"), out["pairs"]["Always on"]["target_gap"]
    r1_st2 = json.loads((R1 / "stage2.json").read_text())
    out["replay"] = {"G": _replay_g(net, fit, read, F, gap, cfg), "G_r1": r1_st2["G"]["Always on/k4/in2/s2"]["G"],
                     "H": _replay_h(net, fit, read, F, gap, cfg), "H_r1": r1_st2["H"]["Always on/in2"]["H"]}
    log(f"control replay: {out['replay']}")
    if abs(out["replay"]["G"] - out["replay"]["G_r1"]) > th["replay_tol"] or abs(out["replay"]["H"] - out["replay"]["H_r1"]) > th["replay_tol"]:
        stop(path, out, "the moved code does not repeat R1's reading")
    out["passed"] = True
    dump(path, out)
    print("STAGE 0 DONE", flush=True)


# ---- one item: train, check, read ---------------------------------------------------------------------------------


def _record(step: int, lft: Tensor, lm: Tensor, lf: Tensor, lr: float) -> dict[str, float]:
    return {"step": step, "lr": lr, "kl_from_M": float(kl(lm, lft).mean()), "kl_to_F": float(kl(lf, lft).mean())}


def train(net: Net, F: Target, fit: Tensor, check: Tensor, cfg: Config, th: dict[str, Any], seed: int, path: Path, out: dict[str, Any]) -> Net:
    """All weights of a copy of M (`clone_net`): Adam, peak `cfg.lr`, linear warm-up then cosine (`qwen_sft.lr_at`), clip 1, loss
    KL(F || fine-tune) at the last position, batches drawn from the fit half. The first steps' KL from M is read on `check`; a
    miss of the bar stops the job (exit 1). Returns the fine-tuned Net with its gradients off."""
    ft = clone_net(net)
    params = list(ft.net.parameters())
    for x in params:
        x.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    n, curve, early = fit.shape[0], [], []
    lm, lf = last(m_of(net), check, cfg.read_batch), last(F, check, cfg.read_batch)
    kl_start = float(kl(lf, lm).mean())
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
            early.append(_record(t, last(VR.net_target(ft), check, cfg.read_batch), lm, lf, rate))
            log(f"  step {t}: loss {curve[-1]:.5f}, KL from M {early[-1]['kl_from_M']:.3e}, KL to F {early[-1]['kl_to_F']:.5f}, rate {rate:.2e}")
        if t == cfg.early_bar_step and early[-1]["kl_from_M"] > th["first_steps_share"] * kl_start:
            stop(path, {**out, "train": {"curve": curve, "early": early, "kl_F_M_start": kl_start}}, f"KL from M after step {t} is above {th['first_steps_share']} of KL(F || M): the first steps are too large")
    for x in params:
        x.requires_grad_(False)
    out["train"] = {"curve": curve, "early": early, "kl_F_M_start": kl_start, "steps": cfg.steps}
    return ft


def xor_learned(net: Net, F: Target, ft: Net, check: Tensor, p: VR.Pieces, cfg: Config, th: dict[str, Any]) -> dict[str, float | bool]:
    """Whether the fine-tune beats the best fixed push: KL(F || fine-tune) against KL(F || M with half the XOR's push), on `check`.
    A model that reads nothing of the XOR is at best at the half push; a do-nothing M reads about twice that. Learned only if the
    ratio is at most the filed bar (th["xor_kl_ratio"], under 1)."""
    dev = next(net.net.parameters()).device
    half = (p.Q @ (0.5 * p.b)).to(dev)
    lf = last(F, check, cfg.read_batch)
    lft = last(VR.net_target(ft), check, cfg.read_batch)
    lhalf = torch.cat([run(net, t, edit={"in2": add_last(half)})[0][:, -1] for t in chunks(check, cfg.read_batch)])
    lm = last(m_of(net), check, cfg.read_batch)
    k_ft, k_half, k_m = float(kl(lf, lft).mean()), float(kl(lf, lhalf).mean()), float(kl(lf, lm).mean())
    ratio = k_ft / k_half
    return {"kl_ft": k_ft, "kl_half_push": k_half, "kl_M": k_m, "ratio": ratio, "learned": ratio <= th["xor_kl_ratio"]}


def distances(net: Net, F: Target, ft: Net, read: Tensor, cfg: Config) -> dict[str, float]:
    """Means on the read half: the gap TV(F_ft, M); TV(F_ft, F); KL(F || F_ft); and the target's own gap TV(F, M)."""
    gap, to_f, kl_f, tgap, n = 0.0, 0.0, 0.0, 0.0, 0
    for t in chunks(read, cfg.read_batch):
        lm, lf, lft = run(net, t)[0][:, -1], F(t)[0][:, -1], run(ft, t)[0][:, -1]
        gap += float(tv(lft, lm).sum())
        to_f += float(tv(lft, lf).sum())
        kl_f += float(kl(lf, lft).sum())
        tgap += float(tv(lf, lm).sum())
        n += t.shape[0]
    return {"gap": gap / n, "tv_to_F": to_f / n, "kl_to_F": kl_f / n, "target_gap": tgap / n}


def search_g(net: Net, cfg: Config, item: str, fit: Tensor, read: Tensor, F: Target, gap: float, path: Path, out: dict[str, Any]) -> None:
    for site in VR.SITES:
        for start in range(cfg.starts):
            key = f"k{cfg.k}/{site}/s{start}"
            if key in out["G"]:
                continue
            budget(cfg)
            t0 = time.monotonic()
            seed = 200_000 + zlib.crc32(f"{item}/{key}".encode()) % 65_536  # not hash(): a resume must draw the same seeds
            found = VR.g_search(net, fit, site, cfg.k, start, F, cfg, seed)
            A, c = found["A"], found["c"]
            r = VR.read_pair(net, read, site, found["Q"], lambda v, _s, A=A, c=c: v @ A + c, F, cfg)
            out["G"][key] = {"G": 1 - r["d"] / gap, "reversion": 1 - r["d_rev"] / gap, "mismatch": r["mismatch"], "fit_kl": found["last_kl"], "s": time.monotonic() - t0}
            dump(path, out)
            log(f"G {item}/{key}: read G {out['G'][key]['G']:.4f}, reversion {out['G'][key]['reversion']:.4f}, mismatch {r['mismatch']:.4f}, fit KL {found['last_kl']:.5f}")


def search_h(net: Net, cfg: Config, item: str, fit: Tensor, read: Tensor, F: Target, gap: float, path: Path, out: dict[str, Any]) -> None:
    """H as R1's `_h_pair`: one ridge accumulation, then per site the penalty, 200 KL steps and the read."""
    missing = [s for s in VR.H_SITES if s not in out["H"]]
    if not missing:
        return
    budget(cfg)
    ridge, X_val, Y_val, ps = VR.h_fit(net, fit, F, cfg, log)
    n_val = fit.shape[0] * 4 // 5
    for site in missing:
        budget(cfg)
        t0 = time.monotonic()
        beta, lam, mse = VR.ridge_select(ridge, site, ps[site], X_val, Y_val[site], cfg.ridge_grid)
        b, y, info = VR.h_refine(net, fit[:n_val], fit[n_val:], site, ridge, beta, F, cfg, 137_800 + VR.H_SITES.index(site))
        r = VR.read_pair(net, read, site, None, VR.ridge_s(ridge, site, b, y), F, cfg)
        out["H"][site] = {"H": 1 - r["d"] / gap, "reversion": 1 - r["d_rev"] / gap, "mismatch": r["mismatch"], "lam": lam, "val_mse": mse, "s": time.monotonic() - t0, **info}
        dump(path, out)
        log(f"H {item}/{site}: read H {out['H'][site]['H']:.4f}, reversion {out['H'][site]['reversion']:.4f}, mismatch {r['mismatch']:.4f}, val KL ridge {info['kl_val_ridge']:.5f} refined {info['kl_val_refined']:.5f} ({info['used']})")


def finetuned(net: Net, F: Target, fit: Tensor, check: Tensor, cfg: Config, th: dict[str, Any], seed: int, state: Path, path: Path, out: dict[str, Any]) -> Net:
    """The item's fine-tuned Net: from its saved state, or trained and saved (`train`; its record is written before the state is read back)."""
    if state.is_file() and "train" in out:
        ft = clone_net(net)
        ft.net.load_state_dict(torch.load(state, map_location=net.device))
        log(f"state read from {state.name}")
        return ft
    ft = train(net, F, fit, check, cfg, th, seed, path, out)
    tmp = state.with_suffix(".tmp")
    torch.save(ft.net.state_dict(), tmp)
    os.replace(tmp, state)
    dump(path, out)
    return ft


def do_item(net: Net, cfg: Config, th: dict[str, Any], sets: VR.Sets, task: VR.Task, p: VR.Pieces, slug: str, seed: int) -> None:
    item, pair = f"{slug}_s{seed}", SLUG[slug]
    path = cfg.results / f"{item}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"item": item, "pair": pair, "seed": seed, "G": {}, "H": {}, "complete": False}
    if out.get("complete"):
        log(f"{item} already complete")
        return
    log(f"item {item}")
    (cfg.results / "large").mkdir(parents=True, exist_ok=True)
    fit, read = VR.pair_windows(cfg, pair, sets, task)
    F = VR.piece_target(net, p, pair)
    check = read[: cfg.check_n]
    ft = finetuned(net, F, fit, check, cfg, th, 300_000 + zlib.crc32(item.encode()) % 65_536, cfg.results / "large" / f"{item}.pt", path, out)
    if "dist" not in out:
        out["dist"] = distances(net, F, ft, read, cfg)
        log(f"{item}: {out['dist']}")
        if pair == "XOR":
            out["xor_check"] = xor_learned(net, F, ft, check, p, cfg, th)
            log(f"{item}: XOR check {out['xor_check']}")
        dump(path, out)
    no = "XOR not learned" if pair == "XOR" and not out["xor_check"]["learned"] else None
    if no is None and out["dist"]["gap"] < th["floor_gap"]:
        no = "gap under the floor"
    if no:
        out.update(no_reading=no, complete=True)
        dump(path, out)
        log(f"{item}: no reading ({no})")
        return
    gap, ft_target = out["dist"]["gap"], VR.net_target(ft)
    search_g(net, cfg, item, fit, read, ft_target, gap, path, out)
    search_h(net, cfg, item, fit, read, ft_target, gap, path, out)
    out["complete"] = True
    dump(path, out)
    log(f"ITEM {item} DONE")


def parse(args: list[str]) -> list[tuple[str, int]] | None:
    """`control` -> None. Otherwise the (slug, seed) items of `all` or the listed pair:seed."""
    if args == ["control"]:
        return None
    if args == ["all"]:
        return [(s, i) for s in SLUG for i in SEEDS]
    out = []
    for a in args:
        s, _, i = a.partition(":")
        if s not in SLUG or not i.isdigit() or int(i) not in SEEDS:
            sys.exit(f"bad item {a!r}: pair:seed, with pair in {sorted(SLUG)} and seed in {SEEDS}, or `control`, or `all`")
        out.append((s, int(i)))
    return out


def main(items: list[tuple[str, int]] | None, cfg: Config, net: Net) -> int:
    th = json.loads(PRED.read_text())["thresholds"]
    net.net.requires_grad_(False)
    from commute import vpd

    tok = vpd.tokenizer()
    sets = VR.make_sets(cfg, tok, log)
    vocab = int(net.net.lm_head.weight.shape[0])  # type: ignore[union-attr]
    task = VR.make_task(sets, tok, vocab, cfg)
    r1 = json.loads((R1 / "stage0.json").read_text())
    if task.cue != r1["pieces"]["cue_token"]:
        log(f"the cue {task.cue} is not R1's {r1['pieces']['cue_token']}: the rows or the vocabulary differ")
        return 1
    p = VR.load_pieces(R1 / "large" / "pieces.pt", net.device, task)
    cfg.results.mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT}: rows ready, cue {task.cue}")
    if items is None:
        control(net, cfg, th, sets, task, p)
        return 0
    path = cfg.results / "control.json"
    if not (path.is_file() and json.loads(path.read_text()).get("passed")):
        log("no passed control: run `control` first and alone")
        return 1
    try:
        for slug, seed in items:
            do_item(net, cfg, th, sets, task, p, slug, seed)
            torch.cuda.empty_cache()
    except MoreToDo:
        log("more to do: wall limit reached, saved; rerun to resume")
        return 3
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+")
    chosen = parse(ap.parse_args().what)
    require_filed(PRED)
    code = 1
    try:
        code = main(chosen, Config(), Net("cuda"))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

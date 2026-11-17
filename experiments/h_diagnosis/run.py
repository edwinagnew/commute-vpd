"""E156 (cell P1.13): why E154's no-teaching control reads H under 0.5. Card: card.txt (rivals and the stop); plan:
reports/plans/2026-11-11_plan_h-diagnosis.md; filed rules: results/predictions.json (the run refuses unless it is committed
unchanged).

E154 fine-tuned the plain 67M toward F = 67M + alpha Q b at block 2's input (in2), last position, on windows whose last
number is odd (alpha 4). Both seeds learned F, yet the best found H was 0.466 and 0.459 (at in3). H = 1 - d / gap, d the mean
TV between the fine-tune and the 67M with an affine map of its own activations written at one site, last position. On E154's
own windows, number keys, alpha, push, saved fine-tunes and search code, cheapest first:

    control  H with F itself standing in for the fine-tune: E154's search (`amend2.search_h`), five sites. F is exactly an
             affine edit at in2, last position, so H at in2 should be about 1. Under 0.5 is the card's stop: it is written,
             and nothing else runs (exit 0).
    dist     per seed, the fine-tune's distances again (they must repeat E154's) and the ceiling 1 - TV(ft, F) / TV(ft, M):
             H if the map reproduced F exactly.
    c        at the seed's best E154 site, the same last-position ridge at penalties 1e-3 (E154's smallest grid value, which
             E154 chose there: a repeat of E154's H that must agree), 1e-5, 1e-6 and 1e-8 (effectively unregularised).
    a        the map writes the site at every position, not only the last (in1, in2, in3; at out3 and final a write reaches the
             last output only through the last position, so it is E154's own read there).
    b        two sites written in order, every pair of the five: the first map as E154's (from the 67M's activations), the
             second fitted on and read from the run already edited at the first. Two maps both from the unedited 67M would
             be degenerate: a whole-activation write at the later site, last position, overwrites everything the earlier
             write could reach.
The variants use the ridge map only, without E154's 200-step KL refinement, which E154 rejected at all ten of its readings.
Writes results/control.json and results/ft_s{0,1}.json after each reading; a rerun resumes. Exit 0 done (also at the card's
stop), 1 a stop, 3 the wall limit (resubmit the same line).

    sbatch --comment=E156 cluster/h_diagnosis.sbatch
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.hidden_maps import Edit, chunks, flat, run, tv, up_to, write_last
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139: Config, clone_net, distances, budget, log, dump, stop, MoreToDo
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup: rows, E137's task and pieces
A2: Any = load_experiment("reuse_variable_finetune", "amend2")  # search_h, E154's H search
B: Any = load_experiment("built_control_sum", "run")  # E150/E154's Config: windows, numbers, alpha grid
EXPERIMENT = "E156"
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
E154 = HERE.parent / "built_control_sum" / "results"  # amend1_{stage0,stage1,control_s0,control_s1}.json, large/amend1_control_s*.pt
log, dump, stop = R.log, R.dump, R.stop
EditOf = Callable[[Tensor], Edit]


@dataclass(frozen=True)
class Config(B.Config):
    results: Path = HERE / "results"
    lams_c: tuple[float, ...] = (1e-3, 1e-5, 1e-6, 1e-8)  # per sample, as Ridge.solve; the first is E154's smallest grid value
    all_sites: tuple[str, ...] = ("in1", "in2", "in3")
    pairs: tuple[tuple[str, str], ...] = (("in2", "in3"), ("in1", "in3"), ("in1", "in2"), ("in3", "out3"), ("in2", "out3"),
                                          ("in1", "out3"), ("in3", "final"), ("out3", "final"), ("in2", "final"), ("in1", "final"))

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.lams_c[0] == self.ridge_grid[0] and list(self.lams_c) == sorted(self.lams_c, reverse=True)
        assert sorted(self.pairs) == sorted(itertools.combinations(VR.H_SITES, 2)), "every pair of the five sites once, in order"
        assert list(self.all_sites) == [s for s in VR.SITES if s in self.all_sites] and "out3" not in self.all_sites


class CardStop(Exception):
    pass


@dataclass
class Ctx:
    net: Net
    cfg: Config
    th: dict[str, Any]
    fit: Tensor
    read: Tensor
    F: VR.Target  # E154's target for the control arm

    @property
    def dev(self) -> str:
        return str(next(self.net.net.parameters()).device)

    def z(self, toks: Tensor, site: str, edit: Edit | None = None) -> Tensor:
        """z for `site`: the 67M's activations up to it, every position, with `edit` applied."""
        return flat(run(self.net, toks, sites=up_to(site), edit=edit)[1], site)

    def h(self, T: VR.Target, edit_of: EditOf, gap: float) -> float:
        """1 - d / gap, d the mean TV at the last position between the 67M edited by edit_of(toks) and T, on the read half."""
        d, n = 0.0, 0
        for toks in chunks(self.read, self.cfg.read_batch):
            ln = run(self.net, toks, edit=edit_of(toks))[0][:, -1]
            d += float(tv(ln, T(toks)[0][:, -1]).sum())
            n += toks.shape[0]
        return 1 - d / n / gap

    def fit_last(self, T: VR.Target, site: str, z_of: Callable[[Tensor], Tensor]) -> VR.Affine:
        """The ridge map from z_of(toks) to T's activation at `site`, last position; penalty from E154's grid on validation."""
        Rg, Xv, Yv = VR.ridge_on(self.fit, self.cfg.ridge_batch, z_of, lambda t: {site: T(t, (site,))[1][site][:, -1]}, self.dev)
        m, _ = VR.affine_of(Rg, site, Xv.shape[1], Xv, Yv[site], self.cfg.ridge_grid)
        del Rg, Xv, Yv
        torch.cuda.empty_cache()
        return m


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def last_edit(c: Ctx, site: str, m: VR.Affine) -> EditOf:
    return lambda t: {site: write_last(None, m(c.z(t, site)))}


def all_edit(c: Ctx, site: str, m: VR.Affine) -> EditOf:
    def e(t: Tensor) -> Edit:
        y = m(c.z(t, site)).view(t.shape[0], t.shape[1], -1)
        return {site: lambda x: y.to(x.dtype)}

    return e


def pair_edit(c: Ctx, s1: str, m1: VR.Affine, s2: str, m2: VR.Affine | None) -> EditOf:
    """The first map written at s1; then (m2 given) the second, read from the run already edited at s1, written at s2."""

    def e(t: Tensor) -> Edit:
        e1: Edit = {s1: write_last(None, m1(c.z(t, s1)))}
        return e1 if m2 is None else {**e1, s2: write_last(None, m2(c.z(t, s2, e1)))}

    return e


# ---- 1: the positive control, alone, first ----------------------------------------------------------------------------


def control(c: Ctx) -> bool:
    """H with F as the fine-tune, five sites. False at the card's stop (H at in2 under stop_bar)."""
    path = c.cfg.results / "control.json"
    out = load(path) or {"experiment": EXPERIMENT, "H": {}}
    if out.get("card_stop"):
        return False
    if out.get("done"):
        return True
    if "gap" not in out:
        out["gap"] = VR.mean_tv(c.F, VR.net_target(c.net), c.read, c.cfg.read_batch)["mean"]
        out["gap_e154"] = load(E154 / "amend1_stage1.json")["gaps"]["control"]
        log(f"target F: gap {out['gap']:.6f}, E154 {out['gap_e154']:.6f}")
        if abs(out["gap"] - out["gap_e154"]) > c.th["reload_tol"]:
            stop(path, out, "F's gap does not repeat E154's: the windows, alpha or push differ")
        dump(path, out)

    def save_out() -> None:
        h = out["H"].get("in2", {}).get("H")
        if h is not None and h < c.th["stop_bar"]:
            out["card_stop"] = f"H on the exact target F at in2 is {h:.4f}, under {c.th['stop_bar']}: the search cannot read an affine edit it should read perfectly"
        dump(path, out)
        if out.get("card_stop"):
            raise CardStop

    try:
        save_out()  # a resume that already holds in2
        A2.search_h(c.net, c.cfg, f"{EXPERIMENT}/target_F", c.fit, c.read, c.F, out["gap"], out, save_out)
    except CardStop:
        log(f"CARD STOP: {out['card_stop']}; nothing further runs")
        return False
    out["done"] = True
    dump(path, out)
    print("STAGE control DONE", flush=True)
    return True


# ---- 2, c, a, b: per seed on E154's saved fine-tunes ----------------------------------------------------------------------


def distances(c: Ctx, seed: int, ft: Net, out: dict[str, Any], path: Path) -> None:
    if "dist" in out:
        return
    old = load(E154 / f"amend1_control_s{seed}.json")
    out["dist"], out["dist_e154"] = R.distances(c.net, c.F, ft, c.read, c.cfg), old["dist"]
    hs: dict[str, float] = {s: v["H"] for s, v in old["H"].items()}
    out["e154_H"], out["e154_best_site"] = hs, max(hs, key=lambda s: hs[s])
    out["e154_best"] = out["e154_H"][out["e154_best_site"]]
    out["e154_lam"] = old["H"][out["e154_best_site"]]["lam"]  # 1e-3 on both seeds: the first of lams_c, so c's first read repeats E154
    out["ceiling"] = 1 - out["dist"]["tv_to_F"] / out["dist"]["gap"]
    log(f"s{seed}: {out['dist']} (E154 {old['dist']}), ceiling {out['ceiling']:.4f}, E154 best H {out['e154_best']:.4f} at {out['e154_best_site']}")
    if max(abs(out["dist"][k] - old["dist"][k]) for k in ("gap", "tv_to_F")) > c.th["reload_tol"]:
        stop(path, out, f"seed {seed}: the saved fine-tune's distances do not repeat E154's")
    dump(path, out)


def penalty(c: Ctx, seed: int, ft: Net, out: dict[str, Any], path: Path) -> None:
    rec = out.setdefault("c", {})
    lams = [lam for lam in c.cfg.lams_c if str(lam) not in rec]
    if not lams:
        return
    site, T = out["e154_best_site"], VR.net_target(ft)
    Rg, Xv, Yv = VR.ridge_on(c.fit, c.cfg.ridge_batch, lambda t: c.z(t, site), lambda t: {site: T(t, (site,))[1][site][:, -1]}, c.dev)
    maps: dict[float, VR.Affine] = {}
    for lam in lams:
        try:
            maps[lam] = VR.affine_of(Rg, site, Xv.shape[1], Xv, Yv[site], (lam,))[0]
        except RuntimeError as e:  # Ridge.solve's factorisation failed at this penalty
            rec[str(lam)] = {"site": site, "H": None, "failed": str(e)}
    del Rg, Xv, Yv
    torch.cuda.empty_cache()
    for lam, m in maps.items():
        R.budget(c.cfg)
        h = c.h(T, last_edit(c, site, m), out["dist"]["gap"])
        rec[str(lam)] = {"site": site, "H": h}
        dump(path, out)
        log(f"c s{seed}/{site}/lam {lam}: H {h:.4f}")
        if lam == out["e154_lam"] and abs(h - out["e154_best"]) > c.th["repeat_tol"]:
            stop(path, out, f"seed {seed}: the new fit at E154's penalty does not repeat E154's H at {site}")


def all_positions(c: Ctx, seed: int, ft: Net, out: dict[str, Any], path: Path) -> None:
    rec = out.setdefault("a", {})
    sites = [s for s in c.cfg.all_sites if s not in rec]
    if not sites:
        return
    T, top = VR.net_target(ft), sites[-1]
    Rg, Xv, Yv = VR.ridge_on(c.fit, c.cfg.ridge_batch, lambda t: c.z(t, top),
                             lambda t: {s: v.flatten(1) for s, v in T(t, sites)[1].items()}, c.dev)
    width = Xv.shape[1] // len(up_to(top))  # one stream, every position
    maps = {s: VR.affine_of(Rg, s, width * len(up_to(s)), Xv, Yv[s], c.cfg.ridge_grid)[0] for s in sites}
    del Rg, Xv, Yv
    torch.cuda.empty_cache()
    for s, m in maps.items():
        R.budget(c.cfg)
        h = c.h(T, all_edit(c, s, m), out["dist"]["gap"])
        rec[s] = {"H": h, "lam": m.lam}
        dump(path, out)
        log(f"a s{seed}/{s} every position: H {h:.4f}, lam {m.lam}")


def two_sites(c: Ctx, seed: int, ft: Net, out: dict[str, Any], path: Path) -> None:
    rec = out.setdefault("b", {})
    T = VR.net_target(ft)
    firsts: dict[str, VR.Affine] = {}
    for s1, s2 in c.cfg.pairs:
        key = f"{s1}+{s2}"
        if key in rec:
            continue
        R.budget(c.cfg)
        if s1 not in firsts:
            firsts[s1] = c.fit_last(T, s1, lambda t, s1=s1: c.z(t, s1))
        first = pair_edit(c, s1, firsts[s1], s2, None)
        m2 = c.fit_last(T, s2, lambda t, s2=s2, first=first: c.z(t, s2, first(t)))
        h = c.h(T, pair_edit(c, s1, firsts[s1], s2, m2), out["dist"]["gap"])
        rec[key] = {"H": h, "lam1": firsts[s1].lam, "lam2": m2.lam}
        dump(path, out)
        log(f"b s{seed}/{key}: H {h:.4f}")


STEPS = (("dist", distances), ("c", penalty), ("a", all_positions), ("b", two_sites))  # cheapest first


def main(net: Net) -> int:
    cfg, th = Config(), json.loads(PRED.read_text())["thresholds"]
    from commute import vpd

    sets, _task, p = A1.setup(net, cfg)
    x = VR.make_numbers(vpd.tokenizer(), int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    if x.keys.tolist() != load(E154 / "amend1_stage0.json")["number_tokens"]:
        log("the number tokens are not E154's")
        return 1
    fit, read = VR.xor2_windows(cfg, sets, x)
    alpha = load(E154 / "amend1_stage1.json")["calibration"]["alpha"]
    F = VR.label_target(net, lambda t: VR.odd_at(x, t, -1), alpha * (p.Q @ p.b).to(net.device))
    c = Ctx(net, cfg, th, fit, read, F)
    log(f"{EXPERIMENT}: alpha {alpha}, {fit.shape[0]} fit and {read.shape[0]} read windows")
    if not control(c):
        return 0
    torch.cuda.empty_cache()
    for name, step in STEPS:
        for seed in cfg.seeds:
            path = cfg.results / f"ft_s{seed}.json"
            out = load(path) or {"experiment": EXPERIMENT, "seed": seed}
            ft = VR.load_state(net, E154 / "large" / f"amend1_control_s{seed}.pt", R.clone_net)
            step(c, seed, ft, out, path)
            dump(path, out)
            del ft
            torch.cuda.empty_cache()
        print(f"STAGE {name} DONE", flush=True)
    return 0


def cli() -> None:
    require_filed(PRED)
    code = 1
    try:
        code = main(Net("cuda"))
        log(f"{EXPERIMENT} DONE" if code == 0 else f"{EXPERIMENT} exit {code}")
    except R.MoreToDo:
        log("more to do: wall limit reached; what was read is saved (resubmit the same line)")
        code = 3
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()

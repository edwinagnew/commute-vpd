"""E157 (cell P1.14): the sum-parity fine-tune read beside the last-number-odd control, between a planted ceiling and floor.
Plan: reports/plans/2026-11-11_plan_sum-parity-beside-control.md; card: card.txt; filed rules: results/predictions.json (the
run refuses unless it is committed unchanged). Exploratory: no label.

On E154's windows, number tokens, alpha (4) and push direction Q b, with E154's one-site search (`amend2.search_h`, five
sites) and E156's two-site reading (`commute.two_sites`, all ten pairs in order), cheapest first:
    ceiling  planted, read first and alone: at in2, last position, h -> h + s l(h) Q b, l(h) = (h - mu0).(mu1 - mu0) /
             |mu1 - mu0|^2 from the 67M's own in2 (last position) averaged over the fit half's keyed windows whose last
             number is odd (mu1) or even (mu0). Computed from the value arriving at in2, so exactly affine in it. s by
             bisection so the gap TV(F, 67M) on the first calib_n read windows matches the sum-parity target's gap; the
             gap on the whole read half must then be within gap_tol. One site; H at in2 under ceiling_stop is the card's
             stop (written as card_stop, exit 0, nothing further runs).
    repeat   E156's two-site reading in2+out3 on E154's saved control fine-tune, seed 0, through the moved code: must repeat
             E156's H within repeat_tol (after the reloaded gap repeats E156's within reload_tol), else exit 1.
    ceiling  two sites.
    floor    planted: h -> h + s r(a, b) Q b at in2, last position, r a seeded 0/1 bit per ordered pair of number keys (half
             ones), 0 off keyed windows. The read half holds only pairs the fit half never sees. Same calibration; one site,
             two sites.
    plain    E150's arm "plain" (bit a + b odd, push alpha Q b), seeds 0 and 1, through E150's `item_h` unchanged (its ladder,
             learned check, distances, one-site H); then two sites per seed. The control is not re-read (E154's and
             E156's saved values; plot.py reads them).
Writes results/{ceiling,repeat,floor,plain_s0,plain_s1}.json after each reading (a rerun resumes) and
results/large/plain_s{0,1}.pt. Exit 0 done (also at the card's stop), 1 a stop or refusal, 3 the wall limit (resubmit).

    sbatch --comment=E157 cluster/sum_parity_beside.sbatch
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute import two_sites as TS
from commute.hidden_maps import Edit, chunks, merged, run
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139: clone_net, distances, budget, log, dump, stop, MoreToDo
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup: rows, E137's task and pieces
A2: Any = load_experiment("reuse_variable_finetune", "amend2")  # search_h, E154's one-site search
B: Any = load_experiment("built_control_sum", "run")  # E150: Ctx, item_h (the arm "plain")
D: Any = load_experiment("h_diagnosis", "run")  # E156: Config with the ten pairs
EXPERIMENT = "E157"
B.EXPERIMENT = EXPERIMENT  # item_h's log tags
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
E154 = HERE.parent / "built_control_sum" / "results"  # amend1_stage{0,1}.json, large/amend1_control_s0.pt
E156 = HERE.parent / "h_diagnosis" / "results"  # ft_s0.json: the two-site reading the repeat must match
log, dump, stop = R.log, R.dump, R.stop


@dataclass(frozen=True)
class Config(D.Config):
    results: Path = HERE / "results"
    floor_seed: int = 157_001  # the planted floor's pair bits
    bisect_steps: int = 20  # halvings of the bracket on s: a bracket width under 1e-6 of its start
    s_max: float = 1024.0  # in units of Q b: the doubling bracket gives up here (alpha is 4)


class CardStop(Exception):
    pass


@dataclass
class Ctx:
    net: Net
    cfg: Config
    th: dict[str, Any]
    x: VR.Xor2
    fit: Tensor
    read: Tensor
    unit: Tensor  # Q b
    target_gap: float  # the sum-parity target's gap on the read half (amend1_stage1.json, gaps.plain)

    def tick(self) -> None:
        R.budget(self.cfg)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


# ---- the planted edits -------------------------------------------------------------------------------------------------


def class_means(c: Ctx) -> tuple[Tensor, Tensor]:
    """The 67M's in2, last position, averaged (float32) over the fit half's keyed windows whose last number is even, odd."""
    w = c.fit[VR.xor2_keyed(c.x, c.fit)]
    sums: list[list[Tensor]] = [[], []]
    n = [0, 0]
    for t in chunks(w, c.cfg.read_batch):
        h = run(c.net, t, sites=("in2",))[1]["in2"][:, -1].double()
        odd = VR.odd_at(c.x, t, -1).bool().to(h.device)
        for k, m in enumerate((~odd, odd)):
            sums[k].append(h[m].sum(0))
            n[k] += int(m.sum())
    assert min(n) > 0
    mu0, mu1 = (torch.stack(sums[k]).sum(0) / n[k] for k in (0, 1))
    log(f"class means: {n[0]} even and {n[1]} odd fit windows, |mu1 - mu0| {float((mu1 - mu0).norm()):.4f}")
    return mu0.float(), mu1.float()


def with_in2(net: Net, fn: Callable[[Tensor], Tensor]) -> VR.Target:
    """F = `net` with `fn` applied at in2 before any caller's edit there."""

    def f(toks: Tensor, sites: Iterable[str] = (), edit: Edit | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        return run(net, toks, sites=sites, edit=merged({"in2": fn}, edit or {}))

    return f


def ceiling_of(c: Ctx, mu0: Tensor, mu1: Tensor) -> Callable[[float], VR.Target]:
    """s -> F with h -> h + s l(h) Q b at in2, last position, l read from the value arriving there."""
    dmu = (mu1 - mu0) / (mu1 - mu0).pow(2).sum()

    def make(s: float) -> VR.Target:
        def fn(x: Tensor) -> Tensor:
            x = x.clone()
            h = x[:, -1].clone()
            x[:, -1] = h + ((h - mu0) @ dmu)[:, None] * (s * c.unit)
            return x

        return with_in2(c.net, fn)

    return make


def floor_of(c: Ctx) -> Callable[[float], VR.Target]:
    """s -> F = 67M + s r(a, b) Q b at in2, last position; r a seeded bit per ordered key pair, half ones, 0 off keys."""
    k = c.x.keys.numel()
    idx = torch.full_like(c.x.bit, -1)
    idx[c.x.keys] = torch.arange(k)
    r = (torch.randperm(k * k, generator=torch.Generator().manual_seed(c.cfg.floor_seed)) < k * k // 2).view(k, k)

    def label(t: Tensor) -> Tensor:
        a, b = idx[t[:, -2].cpu()], idx[t[:, -1].cpu()]
        return ((a >= 0) & (b >= 0) & r[a.clamp_min(0), b.clamp_min(0)]).float()

    return lambda s: VR.label_target(c.net, label, s * c.unit)


def calibrate(c: Ctx, make: Callable[[float], VR.Target], path: Path, out: dict[str, Any]) -> float:
    """s with gap(s) = TV(make(s), 67M) on the first calib_n read windows at the target gap: doubling, then bisection."""
    w, M = c.read[: c.cfg.calib_n], VR.net_target(c.net)
    trace: list[dict[str, float]] = out.setdefault("calibration", [])

    def gap(s: float) -> float:
        g = VR.mean_tv(make(s), M, w, c.cfg.read_batch)["mean"]
        trace.append({"s": s, "gap": g})
        log(f"  s {s:.6f}: gap {g:.6f}")
        return g

    lo, hi = 0.0, 1.0
    while gap(hi) < c.target_gap:
        lo, hi = hi, 2 * hi
        if hi > c.cfg.s_max:
            stop(path, out, f"no s up to {c.cfg.s_max} reaches the target gap {c.target_gap}")
    for _ in range(c.cfg.bisect_steps):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < c.target_gap else (lo, mid)
    return (lo + hi) / 2


def planted(c: Ctx, name: str, make: Callable[[float], VR.Target]) -> tuple[VR.Target, Path, dict[str, Any]]:
    """The calibrated planted target, its record and path; stops if its gap on the whole read half misses by gap_tol."""
    path = c.cfg.results / f"{name}.json"
    out = load(path) or {"experiment": EXPERIMENT, "item": name, "H": {}, "b": {}}
    if "gap" not in out:
        log(f"{name}: calibrating s to the target gap {c.target_gap:.6f}")
        out.pop("calibration", None)
        s = calibrate(c, make, path, out)
        out["s"], out["target_gap"] = s, c.target_gap
        out["gap"] = VR.mean_tv(make(s), VR.net_target(c.net), c.read, c.cfg.read_batch)["mean"]
        log(f"{name}: s {s:.6f} (units of Q b), gap on the read half {out['gap']:.6f}")
        if abs(out["gap"] - c.target_gap) > c.th["gap_tol"]:
            stop(path, out, f"{name}: gap {out['gap']:.6f} misses the target {c.target_gap:.6f} by more than gap_tol")
        dump(path, out)
    return make(out["s"]), path, out


def one_site(c: Ctx, name: str, T: VR.Target, path: Path, out: dict[str, Any], card_stop: bool) -> bool:
    """E154's search, five sites. False at the card's stop (card_stop set and H at in2 under ceiling_stop)."""
    if out.get("card_stop"):
        return False
    if out.get("h_done"):
        return True

    def save() -> None:
        h = out["H"].get("in2", {}).get("H")
        if card_stop and h is not None and h < c.th["ceiling_stop"]:
            out["card_stop"] = f"H on the planted ceiling at in2 is {h:.4f}, under {c.th['ceiling_stop']}: the search cannot read an edit it can reach exactly"
        dump(path, out)
        if out.get("card_stop"):
            raise CardStop

    try:
        save()  # a resume that already holds in2
        A2.search_h(c.net, c.cfg, f"{EXPERIMENT}/{name}", c.fit, c.read, T, out["gap"], out, save)
    except CardStop:
        log(f"CARD STOP: {out['card_stop']}; nothing further runs")
        return False
    out["h_done"] = True
    dump(path, out)
    print(f"STAGE {name}_one_site DONE", flush=True)
    return True


def two_sites(c: Ctx, name: str, T: VR.Target, gap: float, path: Path, out: dict[str, Any]) -> None:
    if out.get("b_done"):
        return
    TS.two_site_h(c.net, c.cfg, c.cfg.pairs, c.fit, c.read, T, gap, out.setdefault("b", {}), lambda: dump(path, out), c.tick, log)
    out["b_done"] = True
    dump(path, out)
    print(f"STAGE {name}_two_sites DONE", flush=True)


# ---- the repeat of E156 through the moved code ---------------------------------------------------------------------------


def repeat(c: Ctx) -> None:
    path = c.cfg.results / "repeat.json"
    out = load(path) or {"experiment": EXPERIMENT, "b": {}}
    if out.get("passed"):
        return
    old = load(E156 / "ft_s0.json")
    ft = VR.load_state(c.net, E154 / "large" / "amend1_control_s0.pt", R.clone_net)
    T = VR.net_target(ft)
    if "gap" not in out:
        out["gap"], out["gap_e156"] = VR.mean_tv(T, VR.net_target(c.net), c.read, c.cfg.read_batch)["mean"], old["dist"]["gap"]
        log(f"repeat: reloaded gap {out['gap']:.6f}, E156 {out['gap_e156']:.6f}")
        if abs(out["gap"] - out["gap_e156"]) > c.th["reload_tol"]:
            stop(path, out, "E154's saved control fine-tune (seed 0) does not repeat E156's gap")
        dump(path, out)
    TS.two_site_h(c.net, c.cfg, (("in2", "out3"),), c.fit, c.read, T, old["dist"]["gap"], out["b"], lambda: dump(path, out), c.tick, log)
    out["H"], out["H_e156"] = out["b"]["in2+out3"]["H"], old["b"]["in2+out3"]["H"]
    if abs(out["H"] - out["H_e156"]) > c.th["repeat_tol"]:
        stop(path, out, f"the moved two-site code reads {out['H']:.4f} where E156 read {out['H_e156']:.4f}")
    out["passed"] = True
    dump(path, out)
    del ft, T
    torch.cuda.empty_cache()
    print("STAGE repeat DONE", flush=True)


# ---- the new arm: E150's "plain" -----------------------------------------------------------------------------------------


def arm_one_site(c: Ctx, bc: Any, seed: int, push: Tensor) -> None:
    """E150's item_h on the arm "plain" (train, learned check, distances, one-site H). Seed 1 runs on seed 0's rung, so
    not at all when seed 0 was not learned."""
    item, rec = f"plain_s{seed}", bc.load(f"plain_s{seed}")
    if seed == 1 and "rung" not in bc.load("plain_s0"):
        log("plain_s1 not run: seed 0 was not learned, so there is no rung to repeat")
        return
    if not (rec.get("h_done") or rec.get("no_reading") or "rung" in rec) and time.monotonic() - R._t0 > c.cfg.wall_limit_s - 1200:
        raise R.MoreToDo  # training's worst case, 300 then 3000 steps with their checks, is ~16 min on an L40
    ft = B.item_h(bc, "plain", seed, c.net, VR.label_target(c.net, bc.bit("plain"), push), push)
    del ft
    torch.cuda.empty_cache()
    print(f"STAGE {item}_one_site DONE", flush=True)


def arm_two_sites(c: Ctx, bc: Any, seed: int) -> None:
    item = f"plain_s{seed}"
    out = bc.load(item)
    if not out.get("h_done"):  # not learned, gap under the floor, or not run: no reading
        return
    ft = VR.load_state(c.net, bc.pt(item), R.clone_net)
    two_sites(c, item, VR.net_target(ft), out["dist"]["gap"], bc.path(item), out)
    del ft
    torch.cuda.empty_cache()


def main(net: Net) -> int:
    cfg, th = Config(), json.loads(PRED.read_text())["thresholds"]
    from commute import vpd

    sets, _task, p = A1.setup(net, cfg)
    x = VR.make_numbers(vpd.tokenizer(), int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    if x.keys.tolist() != load(E154 / "amend1_stage0.json")["number_tokens"]:
        log("the number tokens are not E154's")
        return 1
    fit, read = VR.xor2_windows(cfg, sets, x)
    s1 = load(E154 / "amend1_stage1.json")
    alpha, unit = s1["calibration"]["alpha"], (p.Q @ p.b).to(net.device)
    c = Ctx(net, cfg, th, x, fit, read, unit, s1["gaps"]["plain"])
    bc = B.Ctx(net, cfg, th, x, fit, read, unit, unit)  # E150's context; its second push (the taught arm's) is unused here
    (cfg.results / "large").mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT}: alpha {alpha}, target gap {c.target_gap:.6f}, {fit.shape[0]} fit and {read.shape[0]} read windows")
    ceiling, cpath, cout = planted(c, "ceiling", ceiling_of(c, *class_means(c)))
    if not one_site(c, "ceiling", ceiling, cpath, cout, card_stop=True):
        return 0
    repeat(c)
    two_sites(c, "ceiling", ceiling, cout["gap"], cpath, cout)
    floor, fpath, fout = planted(c, "floor", floor_of(c))
    one_site(c, "floor", floor, fpath, fout, card_stop=False)
    two_sites(c, "floor", floor, fout["gap"], fpath, fout)
    for seed in cfg.seeds:
        arm_one_site(c, bc, seed, alpha * unit)
    for seed in cfg.seeds:
        arm_two_sites(c, bc, seed)
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

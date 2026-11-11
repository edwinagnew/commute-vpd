"""E159 (cell P1.15): the difficulty ladder. The rewiring share read as graded on three fine-tunes of the plain 67M (the last
number is odd, the sum of two is odd, the sum of three is odd), beside a linear and a small MLP probe of the plain 67M for
each target. Plan: reports/plans/2026-11-11_plan_difficulty-ladder.md; card: card.txt; filed rules: results/predictions.json
(the run refuses unless it is committed unchanged). Graded reading only: no label, no cut-off (user, 2026-11-11, ask 34).

The first two rungs are E154's saved control and E157's sum-of-two arm, already read on E150's windows (the last two
positions hold a number pair) between E157's planted ceiling and floor; they are not re-read. E150's windows hold two
numbers, so the third rung needs windows ending in three (`VR.triple_windows`: a number c before E150's pairs, the read half
on held-out pairs), and its planted ceiling and floor are read again on them. Same number tokens, alpha (4), push direction
Q b, one-site search (`amend2.search_h`, five sites) and two-site reading (`commute.two_sites`, ten pairs in order) as E157.
Cheapest first, each stage writing its JSON as it finishes:
    repeat   E157's planted ceiling rebuilt through `commute.planted` at E157's s on E150's windows: its gap must repeat
             E157's within reload_tol and its two-site H at in1+in2 within repeat_tol, else exit 1.
    probes   per layout (pairs: odd, sum2; triples: odd, sum2, sum3), `commute.probes` on the plain 67M at in1, in2, in3 and
             out3, last position: linear and MLP, real and shuffled labels, on keyed windows, scored on held-out pairs. The
             last-number-odd probes (the probes' positive control) must read at least probe_ctrl on each layout, else exit 1.
    target3  the sum-of-three target's gap (67M + alpha Q b where the sum of the last three numbers is odd) on the read half.
    one site the planted ceiling on the triples (stop, exit 1, if its H at in2 is under ceiling_stop), the planted floor,
             then the sum-of-three fine-tune, seeds 0 and 1, through E150's `item_h` unchanged (ladder, learned check,
             distances, one-site H).
    two sites ceiling, floor, then each seed.
Writes results/{repeat,probes,target3,ceiling,floor,sum3_s0,sum3_s1}.json and results/large/sum3_s{0,1}.pt. Exit 0 done,
1 a stop or refusal, 3 the wall limit (resubmit the same line; it resumes).

    sbatch --comment=E159 cluster/difficulty_ladder.sbatch
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute import planted as PL
from commute import two_sites as TS
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.probes import ProbeSizes, last_states, probe_target
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139: clone_net, distances, budget, log, dump, stop, MoreToDo
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup: rows, E137's task and pieces
A2: Any = load_experiment("reuse_variable_finetune", "amend2")  # search_h, E154's one-site search
B: Any = load_experiment("built_control_sum", "run")  # E150: Ctx, item_h
D: Any = load_experiment("h_diagnosis", "run")  # E156: Config with the ten pairs
EXPERIMENT = "E159"
B.EXPERIMENT = EXPERIMENT  # item_h's log tags
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
E154 = HERE.parent / "built_control_sum" / "results"  # amend1_stage{0,1}.json: number tokens, alpha
E157 = HERE.parent / "sum_parity_beside" / "results"  # ceiling.json: the reading the repeat must match
log, dump, stop = R.log, R.dump, R.stop
TARGETS = {"odd": lambda x, t: VR.odd_at(x, t, -1), "sum2": lambda x, t: VR.sum_parity(x, t, 2),
           "sum3": lambda x, t: VR.sum_parity(x, t, 3)}
LAYOUTS = {"pairs": (2, ("odd", "sum2")), "triples": (3, ("odd", "sum2", "sum3"))}  # numbers at the end; targets probed


@dataclass(frozen=True)
class Config(D.Config):
    results: Path = HERE / "results"
    floor_seed: int = 157_001  # E157's: the planted floor's pair bits
    bisect_steps: int = 20  # E157's
    s_max: float = 1024.0  # E157's, in units of Q b
    repeat_pair: tuple[str, str] = ("in1", "in2")  # E157's ceiling read 0.9843 there (ceiling.json, b)
    probe: ProbeSizes = field(default_factory=ProbeSizes)
    probe_sites: tuple[str, ...] = VR.SITES
    probe_seed: int = 159_200

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.repeat_pair in self.pairs and self.probe.n_fit <= self.D // 2 * 3 // 4


@dataclass
class Ctx:
    net: Net
    cfg: Config
    th: dict[str, Any]
    x: VR.Xor2
    unit: Tensor  # Q b
    alpha: float
    windows: dict[str, tuple[Tensor, Tensor]]  # layout -> (fit, read)

    def tick(self) -> None:
        R.budget(self.cfg)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


# ---- the repeat of E157 through the moved code ---------------------------------------------------------------------------


def repeat(c: Ctx) -> None:
    path = c.cfg.results / "repeat.json"
    out = load(path) or {"experiment": EXPERIMENT, "b": {}}
    if out.get("passed"):
        return
    old, (fit, read) = load(E157 / "ceiling.json"), c.windows["pairs"]
    T = PL.ceiling_of(c.net, c.unit, *PL.class_means(c.net, c.x, fit, c.cfg.read_batch, log))(old["s"])
    if "gap" not in out:
        out["gap"], out["gap_e157"] = VR.mean_tv(T, VR.net_target(c.net), read, c.cfg.read_batch)["mean"], old["gap"]
        log(f"repeat: rebuilt ceiling's gap {out['gap']:.6f}, E157 {old['gap']:.6f}")
        if abs(out["gap"] - old["gap"]) > c.th["reload_tol"]:
            stop(path, out, "E157's planted ceiling, rebuilt through commute.planted, does not repeat its gap")
        dump(path, out)
    TS.two_site_h(c.net, c.cfg, (c.cfg.repeat_pair,), fit, read, T, old["gap"], out["b"], lambda: dump(path, out), c.tick, log)
    key = "+".join(c.cfg.repeat_pair)
    out["H"], out["H_e157"] = out["b"][key]["H"], old["b"][key]["H"]
    if abs(out["H"] - out["H_e157"]) > c.th["repeat_tol"]:
        stop(path, out, f"the rebuilt ceiling reads {out['H']:.4f} at {key} where E157 read {out['H_e157']:.4f}")
    out["passed"] = True
    dump(path, out)
    print("STAGE repeat DONE", flush=True)


# ---- the probes of the plain 67M -----------------------------------------------------------------------------------------


def probes(c: Ctx) -> None:
    path = c.cfg.results / "probes.json"
    out = load(path) or {"experiment": EXPERIMENT, "sites": list(c.cfg.probe_sites), "sizes": asdict(c.cfg.probe)}
    for layout, (n, names) in LAYOUTS.items():
        rec = out.setdefault(layout, {})
        if all(t in rec for t in names):
            continue
        c.tick()
        fit, read = c.windows[layout]
        kf, kr = fit[VR.keyed_last(c.x, fit, n)][: c.cfg.probe.n_fit], read[VR.keyed_last(c.x, read, n)]
        Xf, Xr = (last_states(c.net, w, c.cfg.probe_sites, c.cfg.read_batch) for w in (kf, kr))
        for name in names:
            if name not in rec:
                log(f"probes: {layout}, {name}")
                rec[name] = probe_target(Xf, TARGETS[name](c.x, kf), Xr, TARGETS[name](c.x, kr), c.cfg.probe, c.cfg.probe_seed, log)
                dump(path, out)
            if name == "odd" and min(rec["odd"]["max"].values()) < c.th["probe_ctrl"]:
                stop(path, out, f"the last-number-odd probes read {rec['odd']['max']} on the {layout}, under probe_ctrl")
        del Xf, Xr
        torch.cuda.empty_cache()
    print("STAGE probes DONE", flush=True)


# ---- the third rung: planted edits and the sum-of-three fine-tune on the triples ----------------------------------------


def target3(c: Ctx) -> float:
    path = c.cfg.results / "target3.json"
    if "gap" in load(path):
        return load(path)["gap"]
    F = VR.label_target(c.net, lambda t: TARGETS["sum3"](c.x, t), c.alpha * c.unit)
    out = {"experiment": EXPERIMENT, "alpha": c.alpha, "gap": VR.mean_tv(F, VR.net_target(c.net), c.windows["triples"][1], c.cfg.read_batch)["mean"]}
    log(f"sum-of-three target gap {out['gap']:.6f}")
    if out["gap"] <= c.th["target_gap_min"]:
        stop(path, out, "the sum-of-three target's gap is not above target_gap_min")
    dump(path, out)
    return out["gap"]


def planted(c: Ctx, name: str, make: PL.Maker, target_gap: float) -> tuple[VR.Target, Path, dict[str, Any]]:
    """The calibrated planted target on the triples, its record and path; stops if its gap on the read half misses by gap_tol."""
    path, (_, read) = c.cfg.results / f"{name}.json", c.windows["triples"]
    out = load(path) or {"experiment": EXPERIMENT, "item": name, "H": {}, "b": {}}
    if "gap" not in out:
        log(f"{name}: calibrating s to the target gap {target_gap:.6f}")
        out["calibration"] = []
        s = PL.calibrate(c.net, read[: c.cfg.calib_n], make, target_gap, c.cfg.read_batch, c.cfg.s_max, c.cfg.bisect_steps, out["calibration"], log)
        if s is None:
            return stop(path, out, f"{name}: no s up to {c.cfg.s_max} reaches the target gap {target_gap}")  # exits
        out["s"], out["target_gap"] = s, target_gap
        out["gap"] = VR.mean_tv(make(s), VR.net_target(c.net), read, c.cfg.read_batch)["mean"]
        log(f"{name}: s {out['s']:.6f} (units of Q b), gap on the read half {out['gap']:.6f}")
        if abs(out["gap"] - target_gap) > c.th["gap_tol"]:
            stop(path, out, f"{name}: gap {out['gap']:.6f} misses the target {target_gap:.6f} by more than gap_tol")
        dump(path, out)
    return make(out["s"]), path, out


def one_site(c: Ctx, name: str, T: VR.Target, path: Path, out: dict[str, Any], is_ceiling: bool) -> None:
    """E154's search, five sites; on the ceiling, a stop as soon as its H at in2 is read under ceiling_stop."""
    if out.get("h_done"):
        return

    def save() -> None:
        h = out["H"].get("in2", {}).get("H")
        if is_ceiling and h is not None and h < c.th["ceiling_stop"]:
            stop(path, out, f"H on the planted ceiling at in2 is {h:.4f}, under ceiling_stop: the search cannot read an edit it can reach exactly")
        dump(path, out)

    save()  # a resume that already holds in2
    fit, read = c.windows["triples"]
    A2.search_h(c.net, c.cfg, f"{EXPERIMENT}/{name}", fit, read, T, out["gap"], out, save)
    out["h_done"] = True
    dump(path, out)
    print(f"STAGE {name}_one_site DONE", flush=True)


def two_sites(c: Ctx, name: str, T: VR.Target, gap: float, path: Path, out: dict[str, Any]) -> None:
    if out.get("b_done"):
        return
    fit, read = c.windows["triples"]
    TS.two_site_h(c.net, c.cfg, c.cfg.pairs, fit, read, T, gap, out.setdefault("b", {}), lambda: dump(path, out), c.tick, log)
    out["b_done"] = True
    dump(path, out)
    print(f"STAGE {name}_two_sites DONE", flush=True)


def arm_one_site(c: Ctx, bc: Any, seed: int) -> None:
    """E150's item_h on the sum of three (train, learned check, distances, one-site H). Seed 1 runs on seed 0's rung, so not
    at all when seed 0 was not learned."""
    rec = bc.load(f"sum3_s{seed}")
    if seed == 1 and "rung" not in bc.load("sum3_s0"):
        log("sum3_s1 not run: seed 0 was not learned, so there is no rung to repeat")
        return
    if not (rec.get("h_done") or rec.get("no_reading") or "rung" in rec) and time.monotonic() - R._t0 > c.cfg.wall_limit_s - 1200:
        raise R.MoreToDo  # training's worst case, 300 then 3000 steps with their checks, is ~16 min on an L40
    push = c.alpha * c.unit
    ft = B.item_h(bc, "sum3", seed, c.net, VR.label_target(c.net, lambda t: TARGETS["sum3"](c.x, t), push), push)
    del ft
    torch.cuda.empty_cache()
    print(f"STAGE sum3_s{seed}_one_site DONE", flush=True)


def arm_two_sites(c: Ctx, bc: Any, seed: int) -> None:
    item = f"sum3_s{seed}"
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
    alpha = load(E154 / "amend1_stage1.json")["calibration"]["alpha"]
    windows = {"pairs": VR.xor2_windows(cfg, sets, x), "triples": VR.triple_windows(cfg, sets, x)}
    c = Ctx(net, cfg, th, x, (p.Q @ p.b).to(net.device), alpha, windows)
    (cfg.results / "large").mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT}: alpha {alpha}, {windows['triples'][0].shape[0]} fit and {windows['triples'][1].shape[0]} read windows per layout")
    repeat(c)
    probes(c)
    gap3 = target3(c)
    fit3, read3 = windows["triples"]
    ceiling, cpath, cout = planted(c, "ceiling", PL.ceiling_of(net, c.unit, *PL.class_means(net, x, fit3, cfg.read_batch, log)), gap3)
    one_site(c, "ceiling", ceiling, cpath, cout, is_ceiling=True)
    floor, fpath, fout = planted(c, "floor", PL.floor_of(net, x, c.unit, cfg.floor_seed), gap3)
    one_site(c, "floor", floor, fpath, fout, is_ceiling=False)
    bc = B.Ctx(net, cfg, th, x, fit3, read3, c.unit, c.unit)  # E150's context; its second push is unused here
    for seed in cfg.seeds:
        arm_one_site(c, bc, seed)
    if only := os.environ.get("E159_ONLY_SEED"):  # a parallel job: one seed's two-site reading, nothing else written
        arm_two_sites(c, bc, int(only))
        return 0
    two_sites(c, "ceiling", ceiling, cout["gap"], cpath, cout)
    two_sites(c, "floor", floor, fout["gap"], fpath, fout)
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

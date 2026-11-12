"""E168 (cell P1.16): a memorised fact against its reuse twin. Plan: reports/plans/2026-11-12_plan_memorised-fact.md; card:
card.txt; filed rules: results/predictions.json (the run refuses unless it is committed unchanged). Exploratory: read under
the adopted wording and under the tentative "new = not re-weighted" reading; no change to decisions.md.

The fact: a seeded random bit per number token (as many 1s as there are odd numbers among the 100 tokens, so the prior
matches), pushed on when the last number's bit is 1. Its twin: the push when the last number is odd (E154's saved control).
E150's two-number windows, E154's number tokens and alpha, the push alpha Q b at block 2's input, last position. G is E139's
search (k = 4, sites in1, in2, in3, out3, three starts, `amend2.search_g`, as E154 read its control); H is E154's one-site
search (`amend2.search_h`, five sites). Items, each writing results/<item>.json as it finishes and resuming from it:
    planted_odd   the control, first: E157's planted ceiling rebuilt through `commute.planted` at E157's s; its gap must
                  repeat E157's within reload_tol; G only (H is E157's, 0.985 at in2). Passed as soon as one G reads at
                  least g_stop; a finished search with best G under g_stop is the card's stop (exit 1)
    planted_fact  the same class-means construction with the fact's classes, calibrated (as E159's `planted`) to the gap
                  E157 calibrated its planted edits to (ceiling.json, target_gap); G, then H
    floor         E157's planted floor rebuilt (E157's floor seed, s from floor.json), its gap repeated; G only
    twin_s1       E154's saved control fine-tune, seed 1, its gap repeated; G only (seed 0's G is E154's)
    fact_s0/_s1   E150's `item_h` unchanged (train 300 steps, else 3000, learned check, distances, H) toward the fact
                  target; then G on the saved state (results/large/fact_s{seed}.pt). Seed 1 trains on seed 0's rung and
                  waits for it when seed 0 runs in a parallel job
Every item but planted_odd refuses (exit 1) until planted_odd.json is "passed". `E168_ONLY=<item>` runs that item alone after
the shared set-up; `E168_ONLY=readings` writes results/readings.json from the JSONs (no GPU); unset runs every item, then
the readings. Exit 0 done, 1 a stop or refusal, 3 the wall limit (resubmit the same line; it resumes).

    sbatch --comment=E168 cluster/memorised_fact.sbatch <item>
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute import planted as PL
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139: clone_net, log, dump, stop, budget, MoreToDo
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup: rows, E137's task and pieces
A2: Any = load_experiment("reuse_variable_finetune", "amend2")  # search_g, search_h
B: Any = load_experiment("built_control_sum", "run")  # E150: Config, Ctx, item_h
EXPERIMENT = "E168"
B.EXPERIMENT = EXPERIMENT  # item_h's log tags
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
RESULTS = HERE / "results"
E154 = HERE.parent / "built_control_sum" / "results"  # amend1_stage{0,1}.json, amend1_control_s{0,1}.json and .pt
E157 = HERE.parent / "sum_parity_beside" / "results"  # ceiling.json, floor.json
ITEMS = ("planted_odd", "planted_fact", "floor", "twin_s1", "fact_s0", "fact_s1")
EPS = 1e-9  # float rounding at a threshold
log, dump, stop = R.log, R.dump, R.stop


@dataclass(frozen=True)
class Config(B.Config):
    results: Path = RESULTS
    floor_seed: int = 157_001  # E157's: the planted floor's pair bits
    fact_seed: int = 168_001  # the fact's bits
    bisect_steps: int = 20  # E157's
    s_max: float = 1024.0  # E157's, in units of Q b
    seed0_wait_s: int = 2400  # fact_s1 waits this long for a parallel fact_s0 job's rung, then exits 3

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.seed0_wait_s < self.wall_limit_s - 1200 and self.results == RESULTS


@dataclass
class Ctx:
    net: Net
    cfg: Config
    th: dict[str, Any]
    x: VR.Xor2  # the numbers, bit = parity
    fact: VR.Xor2  # the same numbers and held-out pairs, bit = the fact
    fit: Tensor
    read: Tensor
    unit: Tensor  # Q b
    alpha: float

    def path(self, item: str) -> Path:
        return self.cfg.results / f"{item}.json"

    def gap(self, T: VR.Target) -> float:
        return VR.mean_tv(T, VR.net_target(self.net), self.read, self.cfg.read_batch)["mean"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def best(rec: dict[str, Any], key: str) -> float | None:
    """The best share in a finished search ("G" or "H", over sites and starts); None if the search did not finish or is empty."""
    vals = [v[key] for v in rec.get(key, {}).values()]
    return max(vals) if vals and rec.get(f"{key.lower()}_done") else None


def fact_numbers(x: VR.Xor2, seed: int) -> VR.Xor2:
    """`x` with each number token's bit a seeded random bit, exactly as many 1s as odd numbers among the tokens."""
    k, n_odd = x.keys.numel(), int((x.bit[x.keys] == 1).sum())
    bit = torch.full_like(x.bit, -1)
    bit[x.keys] = (torch.randperm(k, generator=torch.Generator().manual_seed(seed)) < n_odd).long()
    return VR.Xor2(x.keys, bit, x.held)


# ---- the searches ----------------------------------------------------------------------------------------------------------


def search_g(c: Ctx, item: str, F: VR.Target, gap: float, out: dict[str, Any], tag: str = "", each: Callable[[], None] = lambda: None) -> None:
    """E139's G search as E154 read its control: k = 4, four sites, three starts; `each` runs after every start."""
    path = c.path(item)
    if not out.get("g_done"):

        def save() -> None:
            each()
            dump(path, out)

        A2.search_g(c.net, c.cfg, tag or f"{EXPERIMENT}/{item}", VR.SITES, c.fit, c.read, F, gap, out, save)
        out["g_done"] = True
        dump(path, out)
    log(f"{item}: best G {best(out, 'G'):.4f}")
    print(f"STAGE {item} G DONE", flush=True)


def search_h(c: Ctx, item: str, F: VR.Target, gap: float, out: dict[str, Any]) -> None:
    path = c.path(item)
    if not out.get("h_done"):
        A2.search_h(c.net, c.cfg, f"{EXPERIMENT}/{item}", c.fit, c.read, F, gap, out, lambda: dump(path, out))
        out["h_done"] = True
        dump(path, out)
    log(f"{item}: best H {best(out, 'H'):.4f}")
    print(f"STAGE {item} H DONE", flush=True)


def repeat_gap(c: Ctx, item: str, T: VR.Target, old_gap: float, out: dict[str, Any]) -> None:
    """A rebuilt or reloaded target's gap on the read half must repeat the saved one within reload_tol, else exit 1."""
    if "gap" in out:
        return
    out["gap"], out["gap_saved"] = c.gap(T), old_gap
    log(f"{item}: gap {out['gap']:.6f}, saved {old_gap:.6f}")
    if abs(out["gap"] - old_gap) > c.th["reload_tol"]:
        stop(c.path(item), out, f"{item}: the gap does not repeat the saved one within reload_tol")
    dump(c.path(item), out)


# ---- the items -------------------------------------------------------------------------------------------------------------


def planted_odd(c: Ctx) -> None:
    out = load(c.path("planted_odd")) or {"experiment": EXPERIMENT, "item": "planted_odd", "G": {}}
    old = load(E157 / "ceiling.json")
    T = PL.ceiling_of(c.net, c.unit, *PL.class_means(c.net, c.x, c.fit, c.cfg.read_batch, log))(old["s"])
    out["s"] = old["s"]
    repeat_gap(c, "planted_odd", T, old["gap"], out)

    def passes() -> None:
        if not out.get("passed") and max(v["G"] for v in out["G"].values()) >= c.th["g_stop"]:
            out["passed"] = True
            log("planted_odd: a G at least g_stop is read: the control passes, the other items may start")
            print("STAGE planted_odd PASSED", flush=True)

    search_g(c, "planted_odd", T, out["gap"], out, each=passes)
    if not out.get("passed"):
        stop(c.path("planted_odd"), out, f"the planted odd edit's best G {best(out, 'G'):.4f} is under g_stop: the card's stop")


def planted_fact(c: Ctx) -> None:
    path = c.path("planted_fact")
    out = load(path) or {"experiment": EXPERIMENT, "item": "planted_fact", "G": {}, "H": {}}
    make = PL.ceiling_of(c.net, c.unit, *PL.class_means(c.net, c.fact, c.fit, c.cfg.read_batch, log))
    if "gap" not in out:
        target = load(E157 / "ceiling.json")["target_gap"]
        out["fact_agrees_with_parity"] = float((c.fact.bit[c.x.keys] == c.x.bit[c.x.keys]).float().mean())
        log(f"planted_fact: the fact's bit equals the parity on {out['fact_agrees_with_parity']:.2f} of the numbers; target gap {target:.6f}")
        out["calibration"] = []
        s = PL.calibrate(c.net, c.read[: c.cfg.calib_n], make, target, c.cfg.read_batch, c.cfg.s_max, c.cfg.bisect_steps, out["calibration"], log)
        if s is None:
            return stop(path, out, f"planted_fact: no s up to {c.cfg.s_max} reaches the target gap")  # exits
        out["s"], out["target_gap"], out["gap"] = s, target, c.gap(make(s))
        log(f"planted_fact: s {s:.6f} (units of Q b), gap on the read half {out['gap']:.6f}")
        if abs(out["gap"] - target) > c.th["gap_tol"]:
            stop(path, out, "planted_fact: the gap misses the target by more than gap_tol")
        dump(path, out)
    T = make(out["s"])
    search_g(c, "planted_fact", T, out["gap"], out)
    search_h(c, "planted_fact", T, out["gap"], out)


def floor(c: Ctx) -> None:
    out = load(c.path("floor")) or {"experiment": EXPERIMENT, "item": "floor", "G": {}}
    old = load(E157 / "floor.json")
    T = PL.floor_of(c.net, c.x, c.unit, c.cfg.floor_seed)(old["s"])
    out["s"] = old["s"]
    repeat_gap(c, "floor", T, old["gap"], out)
    search_g(c, "floor", T, out["gap"], out)


def twin_s1(c: Ctx) -> None:
    out = load(c.path("twin_s1")) or {"experiment": EXPERIMENT, "item": "twin_s1", "G": {}}
    ft = VR.load_state(c.net, E154 / "large" / "amend1_control_s1.pt", R.clone_net)
    repeat_gap(c, "twin_s1", VR.net_target(ft), load(E154 / "amend1_control_s1.json")["dist"]["gap"], out)
    search_g(c, "twin_s1", VR.net_target(ft), out["gap"], out, tag="E154/control_s1")  # E154's tag rule for seed 1
    del ft
    torch.cuda.empty_cache()


def seed0_ready(c: Ctx, bc: Any) -> bool:
    """fact_s1 trains on seed 0's rung: wait for a parallel fact_s0 job to write it. False if seed 0 has no reading."""
    while True:
        rec = bc.load("fact_s0")
        if rec.get("no_reading") or rec.get("stopped"):
            return False
        if "rung" in rec:
            return True
        if time.monotonic() - R._t0 > c.cfg.seed0_wait_s:
            raise R.MoreToDo
        log("fact_s1: waiting for fact_s0's rung")
        time.sleep(60)


def fact(c: Ctx, seed: int) -> None:
    item, bc = f"fact_s{seed}", B.Ctx(c.net, c.cfg, c.th, c.x, c.fit, c.read, c.unit, c.unit)  # the second push is unused
    if seed == 1 and not seed0_ready(c, bc):
        log("fact_s1 not run: seed 0 has no reading")
        return
    rec = bc.load(item)
    if not (rec.get("h_done") or rec.get("no_reading") or "rung" in rec) and time.monotonic() - R._t0 > c.cfg.wall_limit_s - 1200:
        raise R.MoreToDo  # training's worst case, 300 then 3000 steps with their checks, is ~16 min on an L40
    push = c.alpha * c.unit
    ft = B.item_h(bc, "fact", seed, c.net, VR.label_target(c.net, lambda t: VR.odd_at(c.fact, t, -1), push), push)
    if ft is not None:
        out = bc.load(item)
        log(f"{item}: best H {best(out, 'H'):.4f}")
        search_g(c, item, VR.net_target(ft), out["dist"]["gap"], out)
    del ft
    torch.cuda.empty_cache()


STEPS: dict[str, Callable[[Ctx], None]] = {"planted_odd": planted_odd, "planted_fact": planted_fact, "floor": floor,
                                          "twin_s1": twin_s1, "fact_s0": lambda c: fact(c, 0), "fact_s1": lambda c: fact(c, 1)}


# ---- the readings (JSONs only) ---------------------------------------------------------------------------------------------


def labels(g: float | None, h: float | None, th: dict[str, Any]) -> dict[str, str]:
    """Adopted: G >= cut_G and H >= cut_H re-weighted, H >= cut_H rewired, H < cut_H no map found. Tentative: re-weighted as
    adopted, anything else new. No H, or no G where H >= cut_H, is no reading."""
    if h is None or (g is None and h >= th["cut_H"]):
        return {"adopted": "no reading", "tentative": "no reading"}
    adopted = "no map found" if h < th["cut_H"] else "re-weighted" if g is not None and g >= th["cut_G"] else "rewired"
    return {"adopted": adopted, "tentative": "re-weighted" if adopted == "re-weighted" else "new"}


def side(ref: float, g: float, gap: float) -> str:
    """'a' if g is at least `gap` under ref (a tie counts as a), 'b' if within `gap` of it, else 'none'."""
    return "a" if ref - g >= gap - EPS else "b" if abs(ref - g) <= gap + EPS else "none"


def rule_r1(g_odd: float | None, g_fact: float | None, gap: float) -> str:
    if g_odd is None or g_fact is None:
        return "no call: no reading"
    return {"a": "rival a", "b": "rival b", "none": "no call"}[side(g_odd, g_fact, gap)]


def rule_r2(g_twin: list[float | None], seeds: list[str | float], gap: float) -> str:
    """`seeds`: per fact seed its best G, or "no reading" (not learned, gap under the floor, seed 0 without a reading) or
    "incomplete". Rival a if every read seed is at least `gap` under the twin mean, b if every one is within it."""
    twin = [g for g in g_twin if g is not None]
    if len(twin) < len(g_twin) or "incomplete" in seeds:
        return "no call: incomplete"
    read = [g for g in seeds if not isinstance(g, str)]
    if not read:
        return "no call: no fact seed has a reading"
    sides = {side(statistics.mean(twin), g, gap) for g in read}
    return "rival a" if sides == {"a"} else "rival b" if sides == {"b"} else "no call"


def fact_reading(results: Path, seed: int) -> str | float:
    rec, rec0 = load(results / f"fact_s{seed}.json"), load(results / "fact_s0.json")
    if rec.get("no_reading") or (seed == 1 and not rec and rec0.get("no_reading")):
        return "no reading"
    g = best(rec, "G")
    return "incomplete" if g is None else g


def readings(results: Path, th: dict[str, Any]) -> dict[str, Any]:
    recs = {i: load(results / f"{i}.json") for i in ITEMS}
    saved = {"planted_odd": load(E157 / "ceiling.json"), "floor": load(E157 / "floor.json"),
             "twin_s0": load(E154 / "amend1_control_s0.json"), "twin_s1": load(E154 / "amend1_control_s1.json")}
    G = {i: best(r, "G") for i, r in recs.items()} | {"twin_s0": best(saved["twin_s0"], "G")}
    H = {i: best(recs[i], "H") for i in ("planted_fact", "fact_s0", "fact_s1")} | {i: best(saved[i], "H") for i in saved}
    twin = [G["twin_s0"], G["twin_s1"]]
    seeds = [fact_reading(results, s) for s in (0, 1)]
    out: dict[str, Any] = {"experiment": EXPERIMENT, "G": G, "H": H, "H_saved_from": {"planted_odd": "E157", "floor": "E157", "twin_s0": "E154", "twin_s1": "E154"},
                           "labels": {i: labels(G[i], H[i], th) for i in H}, "fact_seeds": seeds,
                           "twin_mean": statistics.mean(g for g in twin if g is not None) if None not in twin else None,
                           "stopped": {i: r["stopped"] for i, r in recs.items() if r.get("stopped")}}
    if "planted_odd" in out["stopped"]:
        out["R1"] = out["R2"] = "no call: stopped (the planted odd edit's best G under g_stop)"
    else:
        out["R1"] = rule_r1(G["planted_odd"], G["planted_fact"], th["smallest_gap"])
        out["R2"] = rule_r2(twin, seeds, th["smallest_gap"])
    for i in H:
        log(f"reading {i}: best G {G[i]}, best H {H[i]} (from {out['H_saved_from'].get(i, 'this run')}), labels {out['labels'][i]}")
    log(f"floor: best G {G['floor']}; fact seeds {seeds}; twin mean G {out['twin_mean']}; R1 {out['R1']}; R2 {out['R2']}")
    dump(results / "readings.json", out)
    return out


def main(net: Net, only: str) -> int:
    cfg, th = Config(), json.loads(PRED.read_text())["thresholds"]
    from commute import vpd

    sets, _task, p = A1.setup(net, cfg)
    x = VR.make_numbers(vpd.tokenizer(), int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    if x.keys.tolist() != load(E154 / "amend1_stage0.json")["number_tokens"]:
        log("the number tokens are not E154's")
        return 1
    alpha = load(E154 / "amend1_stage1.json")["calibration"]["alpha"]
    c = Ctx(net, cfg, th, x, fact_numbers(x, cfg.fact_seed), *VR.xor2_windows(cfg, sets, x), (p.Q @ p.b).to(net.device), alpha)
    (cfg.results / "large").mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT}: alpha {alpha}, {c.fit.shape[0]} fit and {c.read.shape[0]} read windows, {int((c.fact.bit == 1).sum())} fact ones")
    if only and only != "planted_odd" and not load(c.path("planted_odd")).get("passed"):
        log(f"{only} refused: the planted odd control has not passed")
        return 1
    for item in [only] if only else ITEMS:
        STEPS[item](c)
    if not only:
        readings(cfg.results, th)
    return 0


def cli() -> None:
    require_filed(PRED)
    only = os.environ.get("E168_ONLY", "")
    if only not in ("", "readings", *ITEMS):
        log(f"E168_ONLY={only} is not one of {ITEMS} or readings")
        hard_exit(1)
    code = 1
    try:
        if only == "readings":
            readings(RESULTS, json.loads(PRED.read_text())["thresholds"])
            code = 0
        else:
            code = main(Net("cuda"), only)
        log(f"{EXPERIMENT} {only or 'all'} DONE" if code == 0 else f"{EXPERIMENT} {only or 'all'} exit {code}")
    except R.MoreToDo:
        log("more to do: wall limit reached; what was read is saved (resubmit the same line)")
        code = 3
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()

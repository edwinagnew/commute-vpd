"""E180 (cell P1.20): the fourth reuse wording (testing only, not adopted) read by a ranking of interventions on M's list
of variables. Plan: reports/plans/2026-11-13_plan_simplest-intervention-ranking.md; card: card.md; filed rules:
results/predictions.json (the run refuses unless it is committed unchanged). Every reading is "under the fourth wording,
ranked, provisional"; no change to decisions.md. The machinery is commute.rank_read.

Per item: KL(F || M) on a read subset; for each list (neurons, directions, all; tokens and logits in each) the greedy
rung-1 and rung-2 curves (closure after each n, the names) and the rung-3 fits; the reading (rung, n) at each bar. Items
are E173's (experiments/soft_intervention/run.py builds F and its windows) and run 1's controls, built here on M's list.
    1   do_nothing, l1_neuron, l2_triggered, l4_bit (the controls, alone; the stop rule after each)
    2a  planted_constant, planted_trigger, floor        2b  always_on, cue, xor, table
    2c  planted_fact, fact_s0, fact_s1                  2d  twin_s0, twin_s1
    3a  copy24, cipher24      3b  ident32, rand32      3c  case_real, case_twin      3d  repeat_real, repeat_twin (seeds 0, 1)
Every item writes results/<item>.json after each greedy step or rung-3 fit and resumes from it; "done" items are skipped.
A job past --wall-min (default 26.5, a debug slot) exits 3 (fleet/resubmit resubmits it); on a pod pass a large --wall-min.
Runs 2 and 3 refuse until results/run1.json says passed. Paths are the clone's only; on a pod, sync results/pcs.pt and the inputs
E173's builders read (E154, E157, E168 states and the old tasks' delta-W, several under results/large symlinks: stage by hard link).
Exit 0 done, 1 a stop or refusal, 3 more to do.

    sbatch --comment=E182 cluster/simplest_intervention.sbatch 1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.rank_read as RK
import commute.variable_read as VR
from commute import planted as PL
from commute import soft_read as SR
from commute.hidden_maps import Edit, stash_last
from commute.jobs import (
    MORE_TO_DO,
    hard_exit,
    load_experiment,
    require_filed,
    run_items,
)
from commute.steering import Net

SI: Any = load_experiment("soft_intervention", "run")  # E173: its items, setup and log/dump/stop
EXPERIMENT = "E182"  # run 1 (controls); E180 filed the rules, its run 1 stopped at stage 0. Runs 2, 3 take their own codes
WORDING = "under the fourth wording, ranked, provisional (testing only, not adopted)"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
PCS = RESULTS / "pcs.pt"  # under 1 MB, so not in results/large (a symlink that arrives dangling on a pod)
CONTROLS = ("l1_neuron", "l2_triggered", "l4_bit")
OLD_PAIRS = (("copy24", "cipher24"), ("ident32", "rand32"), ("case_real", "case_twin"), ("repeat_real", "repeat_twin"))
RUNS = {"1": ("do_nothing", *CONTROLS), "2a": ("planted_constant", "planted_trigger", "floor"),
        "2b": ("always_on", "cue", "xor", "table"), "2c": ("planted_fact", "fact_s0", "fact_s1"), "2d": ("twin_s0", "twin_s1"),
        **{f"3{k}": tuple(f"{t}_s{s}" for t in pr for s in (0, 1)) for k, pr in zip("abcd", OLD_PAIRS, strict=True)}}  # fmt: skip
E173_NAME = {"table": "table_u"}  # E173 read Table twice (two alignments) on one F
log, dump, stop = SI.log, SI.dump, SI.stop
_t0 = time.monotonic()


class MoreToDo(Exception):
    pass


@dataclass(frozen=True)
class Config(RK.Sizes):
    results: Path = RESULTS
    k_pc: int = 64
    n_pca: int = 32_768  # ordinary windows (E137's validation rows) the principal directions are fixed on
    pca_seed: int = 180_000
    subset_seed: int = 180_010
    search_seed: int = 180_100
    l1_seed: int = 180_001  # the neuron at mlp2 of L1
    l2_seed: int = 180_002  # the neuron at mlp2 of L2
    l4_seed: int = 180_004  # the hash of L4's bit
    wall_s: float = 26.5 * 60  # a 30-min debug slot (--wall-min): one step (~15 s) and a dump fit in what is left

    def __post_init__(self) -> None:
        assert self.n_score % self.batch == 0 and self.line <= self.n_opt and self.cap == 16 and self.results == RESULTS


@dataclass
class Ctx:
    net: Net
    cfg: Config
    th: dict[str, Any]
    e: Any  # E173's Ctx: the E168 layout's windows, Q b, the number tokens
    sp: RK.Space
    pcs: dict[str, Tensor]
    cache: dict[str, Any]


def budget(cfg: Config) -> None:
    if time.monotonic() - _t0 > cfg.wall_s:
        raise MoreToDo


# ---- the items ------------------------------------------------------------------------------------------------------


def control_maker(c: Ctx, name: str) -> tuple[PL.Maker, dict[str, Any]]:
    """s -> F for run 1's controls, built on M's list: L1 neuron j of mlp2 shifted by s; L2 neuron j' of mlp2 shifted by s
    times principal direction 0 at in1 (standardised by its mean and sd on the PCA windows); L4 F = M + s bit Q b at in2,
    the bit a seeded hash of the whole window (half ones), which no read of held-out windows can know."""
    net, cfg = c.net, c.cfg
    if name == "l1_neuron":
        j = int(torch.randint(RK.D_MLP, (1,), generator=torch.Generator().manual_seed(cfg.l1_seed)))
        return (lambda s: RK.edit_target(net, lambda: {"mlp2": RK.add_unit(j, lambda: s)})), {"names": [f"mlp2#{j}"]}
    if name == "l2_triggered":
        j = int(torch.randint(RK.D_MLP, (1,), generator=torch.Generator().manual_seed(cfg.l2_seed)))
        p = c.sp.pcs["in1"][:, 0]
        mu, sd = float(c.pcs["in1_mean"] @ c.pcs["in1"][:, 0]), float(c.pcs["in1_var"][0]) ** 0.5

        def make2(s: float) -> VR.Target:
            def edit() -> Edit:
                st: dict[str, Tensor] = {}
                return {"in1": stash_last(st, "h"), "mlp2": RK.add_unit(j, lambda: s * (st["h"] @ p - mu) / sd)}

            return RK.edit_target(net, edit)

        return make2, {"names": [f"mlp2#{j}", f"in1.pc0->mlp2#{j}"]}
    w = torch.randint(1, 2**31 - 1, (8,), generator=torch.Generator().manual_seed(cfg.l4_seed))

    def bit(t: Tensor) -> Tensor:
        return ((((t.cpu().long() * w).sum(1) % (2**31 - 1)) >> 15) & 1).float()

    return (lambda s: VR.label_target(net, bit, s * c.e.unit)), {"bit_mean_read": float(bit(c.e.read).mean())}


def control(c: Ctx, name: str, out: dict[str, Any]) -> VR.Target:
    """Calibrated as E173 calibrated its planted edits (planted.calibrate to E157's target gap in TV on the first calib_n read
    windows), so their KL(F || M) is comparable; the s found is kept in the item's JSON and reused on a resume."""
    make, info = control_maker(c, name)
    out |= info
    if "s" not in out:
        e, trace = c.e, []
        target = json.loads((SI.E157 / "ceiling.json").read_text())["target_gap"]
        s = PL.calibrate(c.net, e.read[: e.cfg.calib_n], make, target, e.cfg.read_batch, e.cfg.s_max, e.cfg.bisect_steps, trace, log)
        out |= {"target_gap": target, "calibration": trace}
        if s is None:
            stop(c.cfg.results / f"{name}.json", out, f"{name}: no s up to s_max reaches the target gap")
        assert s is not None
        gap = VR.mean_tv(make(s), VR.net_target(c.net), e.read, e.cfg.read_batch)["mean"]
        out |= {"s": s, "gap": gap}
        log(f"{name}: s {s:.6f}, gap {gap:.6f} (target {target:.6f}); {info}")
    if abs(out["gap"] - out["target_gap"]) > c.th["gap_tol"]:  # read again on a resume
        stop(c.cfg.results / f"{name}.json", out, f"{name}: the calibrated gap misses the target by more than gap_tol")
    return make(out["s"])


def e173_item(c: Ctx, name: str) -> tuple[VR.Target, Tensor, Tensor, dict[str, Any]]:
    """E173's F, fit and read windows for `name` (its builders; the planted edits rebuilt at E173's calibrated s)."""
    e, saved = c.e, json.loads((SI.RESULTS / f"{E173_NAME.get(name, name)}.json").read_text())
    if name == "planted_constant":
        it = SI.e168_item(e, SR.planted_constant(e.net, e.unit)(saved["s"]))
    elif name == "planted_trigger":
        it = SI.e168_item(e, SR.planted_trigger(e.net, e.unit, *e.parity_dir())(saved["s"]))
    elif name in ("always_on", "cue", "xor", "table"):
        it = SI.e137_item(e, E173_NAME.get(name, name))
    elif name.rsplit("_s", 1)[0] in SI.OT.TASKS:
        it = SI.old_item(e, name, c.cache)
    else:
        it = SI.e168_items(e, name)
    return it.F, it.fit, it.read, {**it.info, "gap_e173": saved["gap"]}


# ---- reading one item -----------------------------------------------------------------------------------------------


def search_list(c: Ctx, m: RK.Model, d: RK.Data, out: dict[str, Any], lst: str, path: Path) -> None:
    """Rung 1, rung 2, rung 3 for one list, each skipped once a lower rung reached the top bar (no reading can change)."""
    rec, top, seed = out.setdefault("lists", {}).setdefault(lst, {}), max(c.th["bars"]), c.cfg.search_seed
    mk = m.sp.masks(lst, m.net.device)
    for r in ("1", "2"):
        if RK.reached(rec, top):
            return
        st = rec.setdefault(r, {"curve": [], "state": [], "done": False})
        if st["done"]:
            continue
        I = RK.Interv.from_json(st["state"], m.net.device)

        def tick(st: dict[str, Any] = st, I: RK.Interv = I, r: str = r) -> None:
            st["state"] = I.to_json()
            e = st["curve"][-1]
            log(f"{path.stem} [{lst}] rung {r} n {e['n']}: closure {e['closure']:.4f}, check KL {e['check_kl']:.5g} (+ {e['names'][-1]}, score {e['score']:.4g}, x{e['mult']})")
            dump(path, out)
            budget(c.cfg)

        RK.greedy(m, d, I, st["curve"], mk, r == "2", c.cfg, top, tick, seed)
        st["done"] = True
        dump(path, out)
    st3 = rec.setdefault("3", {"fits": [], "done": False})
    if RK.reached(rec, top) or st3["done"]:
        return

    def tick3() -> None:
        f = st3["fits"][-1]
        log(f"{path.stem} [{lst}] rung 3 {f['stage']}, {f['edits']} edit(s) at {f['pair']}: closure {f['closure']:.4f}, check KL {f['check_kl']:.5g} (n {f['n']} kept)")
        dump(path, out)
        budget(c.cfg)

    base = RK.best_of(m, d, c.cfg, [RK.Interv.from_json(rec[r]["state"], m.net.device) for r in ("1", "2")])
    RK.rung3(m, d, base, st3["fits"], c.cfg, top, tick3, seed)
    st3["done"] = True
    dump(path, out)


def read_item(c: Ctx, name: str, floor: float) -> dict[str, Any]:
    path = c.cfg.results / f"{name}.json"
    out = json.loads(path.read_text()) if path.is_file() else {}
    if out.get("done"):
        return out
    out |= {"experiment": EXPERIMENT, "item": name, "wording": WORDING}
    if name in CONTROLS:
        F, fit, read = control(c, name, out), c.e.fit, c.e.read
    else:
        F, fit, read, info = e173_item(c, name)
        out |= info
        if "gap" not in out:
            out["gap"] = VR.mean_tv(F, VR.net_target(c.net), read, c.cfg.eval_batch)["mean"]
            log(f"{name}: gap (mean TV) {out['gap']:.6f}, E173's {info['gap_e173']:.6f}")
        if abs(out["gap"] - info["gap_e173"]) > c.th["reload_tol"]:  # read again on a resume
            stop(path, out, f"{name}: the rebuilt F's gap does not repeat E173's within reload_tol")
    m, d = RK.prepare(c.net, c.sp, F, fit, read, c.cfg, c.cfg.subset_seed)
    out |= {"kl": d.kl0, "kl_floor": floor}
    log(f"{name}: KL(F || M) {d.kl0:.6g} nats on {d.read.shape[0]} read windows (floor {floor:.3g})")
    if d.kl0 >= floor:
        for lst in RK.LISTS:
            search_list(c, m, d, out, lst, path)
    recs = out.get("lists", {})
    out["readings"] = {lst: {str(b): RK.reading(d.kl0, floor, recs.get(lst, {}), b) for b in c.th["bars"]} for lst in RK.LISTS}
    for lst, rd in out["readings"].items():
        log(f"{name} [{lst}] readings: " + "; ".join(f"{b}: rung {v['rung']}, n {v['n']}" for b, v in rd.items()))
    out["done"] = True
    dump(path, out)
    del m, d, F
    torch.cuda.empty_cache()
    return out


# ---- the filed rules ------------------------------------------------------------------------------------------------


def misses(name: str, out: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Each (list, bar) of `spec` whose reading is off it: rung in spec["rung"], n <= spec["n_max"], and with "own_names" the
    names are the edit's own (out["names"]). An item not read (no readings) misses everything."""
    miss = []
    for lst in spec["lists"]:
        for b in spec["bars"]:
            rd: dict[str, Any] = out.get("readings", {}).get(lst, {}).get(str(b)) or {"rung": None, "n": None, "names": []}
            ok = rd["rung"] in spec["rung"] and ("n_max" not in spec or (rd["n"] is not None and rd["n"] <= spec["n_max"]))
            ok = ok and (not spec.get("own_names") or sorted(rd.get("names", [])) == sorted(out.get("names", [None])))
            miss += [] if ok else [f"{name} [{lst}] at {b}: {rd} against {spec}"]
    return miss


def rule_r1(results: Path, th: dict[str, Any]) -> str:
    """R1: rival a if the twin's rung (list all, bar r1_bar) is at most 2 and the fact's at least 3 on both seeds; rival b
    if they are equal on both seeds; else no call; "not read" until all four items are done."""
    rung = {}
    for n in ("twin_s0", "twin_s1", "fact_s0", "fact_s1"):
        p = results / f"{n}.json"
        rec = json.loads(p.read_text()) if p.is_file() else {}
        if not rec.get("done"):
            return "not read"
        rung[n] = rec["readings"]["all"][str(th["r1_bar"])]["rung"]
    if all(rung[f"twin_s{s}"] <= 2 and rung[f"fact_s{s}"] >= 3 for s in (0, 1)):
        return "rival a"
    return "rival b" if all(rung[f"twin_s{s}"] == rung[f"fact_s{s}"] for s in (0, 1)) else "no call"


def rule_pairs(results: Path, th: dict[str, Any]) -> dict[str, Any]:
    """R2: each old pair and seed, the reuse side's rung (list all, bar r1_bar) at or below the new side's."""
    out: dict[str, Any] = {}
    for reuse, new in OLD_PAIRS:
        for s in (0, 1):
            recs = [json.loads(p.read_text()) if (p := results / f"{n}_s{s}.json").is_file() else {} for n in (reuse, new)]
            if not all(r.get("done") for r in recs):
                out[f"{reuse}/{new} s{s}"] = "not read"
                continue
            a, b = (r["readings"]["all"][str(th["r1_bar"])]["rung"] for r in recs)
            out[f"{reuse}/{new} s{s}"] = {"reuse": a, "new": b, "holds": a <= b}
    done = [v for v in out.values() if isinstance(v, dict)]
    out["all_hold"] = all(v["holds"] for v in done) if len(done) == len(out) else "not read"
    return out


def run_summary(c: Ctx, run: str, pred: dict[str, Any]) -> dict[str, Any]:
    """run<k>.json from every item of run k done so far, with the filed expected readings and run k's rule."""
    k, res = run[0], c.cfg.results
    names = [n for r, ns in RUNS.items() if r[0] == k for n in ns]
    recs = {n: json.loads(p.read_text()) for n in names if (p := res / f"{n}.json").is_file()}
    recs = {n: r for n, r in recs.items() if r.get("done")}
    out: dict[str, Any] = {"experiment": EXPERIMENT, "run": k, "wording": WORDING, "readings": {n: r["readings"] for n, r in recs.items()},
                           "kl": {n: r["kl"] for n, r in recs.items()}}  # fmt: skip
    out["expected"] = {n: {"spec": s, "misses": misses(n, recs[n], s)} for n, s in pred["expected"].items() if n in recs}
    if k == "2":
        out["R1"] = rule_r1(res, c.th)
    if k == "3":
        out["R2"] = rule_pairs(res, c.th)
    return out


# ---- the job --------------------------------------------------------------------------------------------------------


def setup(cfg: Config, th: dict[str, Any], run: str) -> Ctx:
    from commute import vpd

    net = Net("cuda")
    e173 = SI.Config()
    sets, task, p = SI.A1.setup(net, e173)
    x = VR.make_numbers(vpd.tokenizer(), int(net.net.lm_head.weight.shape[0]), e173.n_max, e173.held_share)  # type: ignore[union-attr]
    if x.keys.tolist() != json.loads((SI.E154 / "amend1_stage0.json").read_text())["number_tokens"]:
        raise SystemExit("S0: the number tokens are not E154's")
    e = SI.Ctx(net, e173, json.loads(SI.PRED.read_text())["thresholds"], sets, task, p, x, *VR.xor2_windows(e173, sets, x), (p.Q @ p.b).to(net.device))
    if not PCS.is_file():
        assert run == "1", "the principal directions are fixed in run 1"
        pca = VR.cut(sets.e_rows, cfg.n_pca, e173.window, torch.Generator().manual_seed(cfg.pca_seed))
        torch.save(RK.principal_directions(net, pca, cfg.k_pc, cfg.eval_batch), PCS)
        log(f"principal directions fixed on M over {cfg.n_pca} ordinary windows: {PCS}")
    pcs = torch.load(PCS)
    sp = RK.Space({s: pcs[s].to(net.device) for s in RK.RES}, int(net.net.lm_head.weight.shape[0]))  # type: ignore[union-attr]
    log("variance share of the top directions: " + ", ".join(f"{s} top-1 {float(pcs[f'{s}_var'][0]):.3g}" for s in RK.RES))
    return Ctx(net, cfg, th, e, sp, pcs, {})


def kl_floor(c: Ctx) -> float:
    """The rung-0 floor: kl_floor, or ten times the do-nothing's KL if that is larger (read from run 1)."""
    p = c.cfg.results / "do_nothing.json"
    dn = json.loads(p.read_text()).get("kl", 0.0) if p.is_file() else 0.0
    return max(c.th["kl_floor"], 10 * dn)


def main(run: str, wall_min: float) -> int:
    cfg, pred = Config(wall_s=60 * wall_min), json.loads(PRED.read_text())
    th, res = pred["thresholds"], cfg.results
    run1 = res / "run1.json"
    if run != "1" and not (run1.is_file() and json.loads(run1.read_text()).get("passed")):
        log(f"run {run} refused: run 1 has not passed (results/run1.json)")
        return 1
    c = setup(cfg, th, run)
    try:
        if run == "1":
            for name in RUNS["1"]:
                out = read_item(c, name, th["kl_floor"])
                miss = [x for s in pred["run1_stop"][name] for x in misses(name, out, s)]
                if miss:
                    dump(res / "run1.json", {**run_summary(c, run, pred), "passed": False, "misses": miss})
                    log(f"RUN 1 STOP: {miss}; nothing further runs and the cell closes 'no call'")
                    return 1
            dump(res / "run1.json", {**run_summary(c, run, pred), "passed": True})
            log(f"{EXPERIMENT} run 1 passed")
            return 0
        floor = kl_floor(c)
        code = run_items(RUNS[run], lambda n: read_item(c, n, floor), log)
    except MoreToDo:
        log(f"wall limit near after {time.monotonic() - _t0:.0f} s: exit {MORE_TO_DO}, resubmit run {run}")
        return MORE_TO_DO
    summary = run_summary(c, run, pred)
    dump(res / f"run{run[0]}.json", summary)
    log(f"run {run[0]} so far: R1 {summary.get('R1')}, R2 {summary.get('R2')}; expected misses "
        + str({n: v["misses"] for n, v in summary["expected"].items() if v["misses"]}))
    log(f"{EXPERIMENT} run {run} DONE")
    return code


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=tuple(RUNS))
    ap.add_argument("--wall-min", type=float, default=26.5, help="exit 3 after this many minutes (a debug slot; large on a pod)")
    args = ap.parse_args()
    run = args.run
    require_filed(PRED)
    code = 1
    try:
        code = main(run, args.wall_min)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()

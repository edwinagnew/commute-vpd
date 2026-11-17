"""E___ (cell P1.25, run 2): causal-rule retention under the fixed choices, read on the five trained matched pairs of
E184 (experiments/simplest_intervention), seeds 0 and 1, on the 67M (plan
reports/plans/2026-11-17_plan_rule-retention-fixed-choices.md, "Run 2"; card card.md; filed settings, rule and stops in
results/predictions.json; the run refuses unless that file is committed unchanged). The measurement is
commute.rule_retention_fixed, as run 1 (experiments/rule_retention_fixed) ran it, with the contrast's KL read at the
last position of each window, where every task here is scored. Each fine-tune F and its windows come from the builders
E173 and E184 call: commute.old_tasks (old_f, task_windows) for the four old pairs, variable_read (load_state,
xor2_windows) for the parity twin and the memorised fact.

Per item: F; E184's gap repeated on the item's whole read half (stop unless within reload_tol); a seeded subset of n_read
read windows; on it the profile, the hybrid gate (stop over gate_tol) and the retention for the contrast F against M at
level A (neurons, filed) and level B (heads and MLP blocks, reported), 5 orders each.
Stages, cheapest first; stage 0 runs first and alone, and is skipped on a rerun once results/stage0.json says done:
    0  the control, on copy24's read windows: do-nothing must read "no change" at both levels, and run 1's circuit turned
       up x2 must read 1 at level A in every order (changed units exactly C, gain 2). Exit 1 on any miss.
    1-5  one pair per stage (PAIRS), reuse member then twin, for each seed asked: each item writes results/items/<item>.json
       and results/large/units/<item>.npz (every unit's partner, r, gain, same; the level A sets) when it finishes and is
       skipped once its JSON is there; `STAGE n sS DONE` after both members of a seed. results/summary.json (the filed
       rule over whatever items are done) is rewritten after each item.
Inputs (sync by hard link; results/large is a symlink that arrives dangling): artifacts/vpd, artifacts/sets
train_0_32768.npy, train_32768_8192.npy, val_0_2000.npy, the tokenizer, every state in old_tasks.TASKS for the 8 old
tasks and STATES below, the old tasks' sources, built_control_sum/results/amend1_stage0.json and E184's item JSONs.
Exit 0 done, 1 a stop or refusal. Several pods split the work by stage and seed; the log is standard output:

    python experiments/rule_retention_pairs/run.py [STAGE ...] [--seeds 0 1] 2>&1 | tee -a experiments/rule_retention_pairs/results/run.log
    python experiments/rule_retention_pairs/run.py --summary      (no GPU: the summary from the item files present)
    (on a pod: E120_PRED_SHA=<sha of the predictions file> python ...)
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

import commute.variable_read as VR
from commute import old_tasks as OT
from commute import rule_retention as R
from commute import rule_retention_fixed as RF
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed

P: Any = load_experiment("rule_retention", "run")  # P1.24: clone, rescaled, chunks, log, dump, stop
CELL = "P1.25"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, ITEMS, UNITS = RESULTS / "predictions.json", RESULTS / "items", RESULTS / "large" / "units"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
PAIRS = (("copy24", "cipher24"), ("ident32", "rand32"), ("case_real", "case_twin"), ("repeat_real", "repeat_twin"), ("twin", "fact"))
COUNTED = 4  # the first four pairs are scored; the parity twin / memorised fact pair is reported, not counted
STATES = {"fact": EXPERIMENTS / "memorised_fact" / "results" / "large" / "fact_s{}.pt",  # E173's e168_items paths
          "twin": EXPERIMENTS / "built_control_sum" / "results" / "large" / "amend1_control_s{}.pt"}  # fmt: skip
E184 = EXPERIMENTS / "simplest_intervention" / "results"
NUMBERS = EXPERIMENTS / "built_control_sum" / "results" / "amend1_stage0.json"
LAST = slice(-1, None)
log, dump, stop = P.log, P.dump, P.stop


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    layer: int  # stage 0's circuit, run 1's C
    C: tuple[int, ...]
    factor: float  # stage 0's turned-up item
    control_task: str  # whose read windows stage 0 uses
    seeds: tuple[int, ...]
    n_read: int  # read windows per item
    read_seed: int
    n_max: int  # E173's numbers (the parity twin and the fact)
    held_share: float
    eval_batch: int  # the gap check
    profile_batch: int

    def __post_init__(self) -> None:
        assert len(set(self.C)) == len(self.C) and max(self.C) < R.D_MLP and self.factor != 1 and self.control_task in OT.TASKS
        assert set(self.seeds) <= {0, 1} and 0 < self.n_read <= OT.DSizes().D // 2 and self.profile_batch > 0


@dataclass
class Data:
    sets: VR.Sets
    x: VR.Xor2
    dcfg: OT.DSizes
    cache: dict[str, Tensor]

    def read_half(self, name: str) -> Tensor:
        """The item's read half, as E173 and E184 built it (twins share it)."""
        key = "xor2" if name in STATES else OT.TASKS[name].group
        if key not in self.cache:
            self.cache.clear()
            self.cache[key] = VR.xor2_windows(self.dcfg, self.sets, self.x)[1] if key == "xor2" else OT.task_windows(OT.TASKS[name], self.sets, self.dcfg)[1]
        return self.cache[key]


def subset(read: Tensor, cfg: Config, M: R.Model) -> Tensor:
    pick = torch.randperm(read.shape[0], generator=torch.Generator().manual_seed(cfg.read_seed))[: cfg.n_read]
    return read[pick].to(M.net.wte.weight.device)


def gap(Fnet: Any, net: Any, half: Tensor, cfg: Config) -> float:
    """Mean TV between F and M at the last position over the read half, as E184 measured its gap."""
    return VR.mean_tv(VR.net_target(Fnet), VR.net_target(net), half, cfg.eval_batch)["mean"]


def fine_tune(net: Any, item: str) -> Any:
    name, seed = item.rsplit("_s", 1)
    if name in STATES:
        return VR.load_state(net, Path(str(STATES[name]).format(seed)), OT.clone_net)
    return OT.old_f(net, OT.TASKS[name].states[int(seed)])


# ---- reading one item ---------------------------------------------------------------------------------------------


def read(F: R.Model, M: R.Model, toks: Tensor, cfg: Config, th: dict[str, Any], item: str) -> tuple[RF.Profile, dict[str, Any], dict[str, Any]]:
    """Profile, hybrid gate and the contrast at both levels; level A's sets come back as masks (4, 3072) per order, not names."""
    t0 = time.time()
    prof = RF.profile(F, M, P.chunks(toks, cfg.profile_batch))
    out: dict[str, Any] = {"gate": RF.gate(F, M, toks, prof), "profile_seconds": time.time() - t0}
    log(f"{item}: hybrid gate {out['gate']:.3g}, profile {out['profile_seconds']:.1f} s")
    masks: dict[str, Any] = {}
    for lv in "AB":
        us, ch = RF.units(prof, lv), RF.changed(prof, lv, th["r_tol"])
        r = [RF.get(prof, u, "r") for u in us]
        c = RF.contrast(F, M, toks, prof, lv, th, log, LAST)
        orders = c.get("orders", [])
        if lv == "A":
            for k, o in enumerate(orders):
                m = np.zeros((len(M.net.h), R.D_MLP), dtype=bool)
                for x in o.pop("units"):
                    l, i = x["unit"][1:].split(".n")
                    m[int(l), int(i)] = True
                masks[f"setA_order{k}"] = m
        ev, sec = sum(o["evals"] for o in orders), sum(o["seconds"] for o in orders)
        out[lv] = {"n_units": len(us), "n_changed": len(ch), "min_r": min(r), "mean_r": sum(r) / len(r), "contrast": c, "evals": ev, "seconds": sec}
        log(f"{item} level {lv}: {len(us)} units, {len(ch)} changed, mean r {out[lv]['mean_r']:.6f}; contrast {c['reading']}, KL(F || M) {c['kl']:.6g} "
            f"(last position), retention {c.get('retention')}; {ev} evaluations, {sec:.1f} s" + (f", {1000 * sec / ev:.1f} ms each" if ev else ""))
    return prof, out, masks


def save_units(item: str, prof: RF.Profile, masks: dict[str, Any]) -> None:
    UNITS.mkdir(parents=True, exist_ok=True)
    fields = ("partner", "zero", "r", "gain", "same")
    np.savez_compressed(UNITS / f"{item}.npz", **{f"{k}{l}_{f}": getattr(p, f).numpy() for (k, l), p in prof.items() for f in fields}, **masks)


def read_item(net: Any, M: R.Model, data: Data, cfg: Config, th: dict[str, Any], item: str) -> dict[str, Any]:
    path = ITEMS / f"{item}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return json.loads(path.read_text())
    t0 = time.time()
    Fnet = fine_tune(net, item)
    half = data.read_half(item.rsplit("_s", 1)[0])
    g, e184 = gap(Fnet, net, half, cfg), json.loads((E184 / f"{item}.json").read_text())["gap"]
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "gap": g, "gap_e184": e184, "n_read_half": half.shape[0]}
    log(f"{item}: gap (mean TV, last position, {half.shape[0]} read windows) {g:.6f}, E184's {e184:.6f}")
    ITEMS.mkdir(parents=True, exist_ok=True)
    if not abs(g - e184) <= th["reload_tol"]:
        stop(path.with_suffix(".stopped.json"), out, f"{item}: the gap does not repeat E184's within reload_tol")
    prof, rd, masks = read(R.Model(Fnet.net), M, subset(half, cfg, M), cfg, th, item)
    out |= rd | {"seconds": time.time() - t0}
    if not rd["gate"] <= th["gate_tol"]:
        stop(path.with_suffix(".stopped.json"), out, f"{item}: hybrid gate {rd['gate']:.3g}")
    save_units(item, prof, masks)
    dump(path, out)
    log(f"{item}: done in {out['seconds']:.0f} s")
    del Fnet
    torch.cuda.empty_cache()
    return out


# ---- stage 0: the control -----------------------------------------------------------------------------------------


def misses0(rd: dict[str, dict[str, Any]], C: list[R.Unit], cfg: Config, th: dict[str, Any]) -> list[str]:
    """Every stage-0 check missed (predictions.json "stops")."""
    out = [f"{k} gate {v['gate']:.3g}" for k, v in rd.items() if not v["gate"] <= th["gate_tol"]]
    out += [f"do-nothing level {lv} {rd['do_nothing'][lv]['contrast']['reading']}" for lv in "AB" if rd["do_nothing"][lv]["contrast"]["reading"] != "no change"]
    a = rd["x2"]["A"]
    out += [f"x2 changed {a['changed']}, not C"] if a["changed"] != [R.name(u) for u in C] else []
    c = a["contrast"]
    if c["reading"] != "read":
        return out + [f"x2 level A {c['reading']}"]
    out += [f"x2 retention {c['retention']}"] if not all(abs(1 - v) <= th["r_tol"] for v in c["retention"]) else []
    g = [x for o in c["orders"] for x in (o["gain_min"], o["gain_max"])]
    return out + ([f"x2 gains {min(g)}..{max(g)}"] if not all(abs(v - cfg.factor) <= th["gain_tol"] for v in g) else [])


def stage0(M: R.Model, data: Data, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    toks, C = subset(data.read_half(cfg.control_task), cfg, M), [("neuron", cfg.layer, i) for i in cfg.C]
    out: dict[str, Any] = {"experiment": EXPERIMENT, "items": {}}
    for item, F in (("do_nothing", P.clone(M)), ("x2", P.rescaled(M, cfg.layer, list(cfg.C), cfg.factor))):
        prof, rd, _ = read(F, M, toks, cfg, th, item)
        rd["A"]["changed"] = [R.name(u) for u in RF.changed(prof, "A", th["r_tol"])]
        out["items"][item] = rd
    miss = misses0(out["items"], C, cfg, th)
    log(f"stage 0: {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(RESULTS / "stage0.json", out, "stage 0: " + "; ".join(miss))
    return out | {"done": True}


# ---- the filed rule -----------------------------------------------------------------------------------------------


def compare(reuse: dict[str, Any] | None, twin: dict[str, Any] | None, lv: str) -> dict[str, Any]:
    """One pair and seed at one level: 'holds' when the reuse member's mean retention over its orders exceeds the twin's
    (strictly; a tie fails), 'fails' otherwise; 'not read' when either item is missing or its reading is not 'read'."""
    if reuse is None or twin is None:
        return {"outcome": "not read", "why": "item not done"}
    a, b = reuse[lv]["contrast"], twin[lv]["contrast"]
    if a["reading"] != "read" or b["reading"] != "read":
        return {"outcome": "not read", "reuse": a["reading"], "twin": b["reading"]}
    ma, mb = sum(a["retention"]) / len(a["retention"]), sum(b["retention"]) / len(b["retention"])
    return {"outcome": "holds" if ma > mb else "fails", "reuse_mean": ma, "twin_mean": mb, "difference": ma - mb,
            "every_order": min(a["retention"]) > max(b["retention"]), "reuse": a["retention"], "twin": b["retention"]}  # fmt: skip


def summary(cfg: Config) -> dict[str, Any]:
    items = {p.stem: json.loads(p.read_text()) for p in ITEMS.glob("*.json") if not p.stem.endswith(".stopped")}
    out: dict[str, Any] = {"experiment": EXPERIMENT, "cell": CELL, "counted": {}, "reported": {}}
    for k, (reuse, twin) in enumerate(PAIRS):
        for s in cfg.seeds:
            rows = {lv: compare(items.get(f"{reuse}_s{s}"), items.get(f"{twin}_s{s}"), lv) for lv in "AB"}
            out["counted" if k < COUNTED else "reported"][f"{reuse}/{twin} s{s}"] = {"level A (filed)": rows["A"], "level B (reported)": rows["B"]}
    got = [v["level A (filed)"]["outcome"] for v in out["counted"].values()]
    out["rule"] = {"holds": got.count("holds"), "fails": got.count("fails"), "not read": got.count("not read"), "of": len(got),
                   "every comparison holds": got.count("holds") == len(got)}  # fmt: skip
    return out


# ---- the job ------------------------------------------------------------------------------------------------------


def main(stages: list[int], seeds: list[int], summary_only: bool) -> int:
    pred = json.loads(PRED.read_text())
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    cfg = Config(**settings)
    th = pred["thresholds"]
    if summary_only:
        dump(RESULTS / "summary.json", summary(cfg))
        log(f"summary: {summary(cfg)['rule']}")
        return 0
    from commute import vpd
    from commute.steering import Net

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout
    log(f"{EXPERIMENT} ({CELL}, run 2) start; stages {stages}, seeds {seeds}; settings {asdict(cfg)}; thresholds {th}")
    net = Net("cuda")
    net.net.requires_grad_(False)
    M, tok, dcfg = R.Model(net.net), vpd.tokenizer(), OT.DSizes()
    x = VR.make_numbers(tok, int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    if x.keys.tolist() != json.loads(NUMBERS.read_text())["number_tokens"]:
        log("STOP: the number tokens are not E154's")
        return 1
    data = Data(VR.make_sets(dcfg, tok, log), x, dcfg, {})
    if not (RESULTS / "stage0.json").is_file() or not json.loads((RESULTS / "stage0.json").read_text()).get("done"):
        dump(RESULTS / "stage0.json", stage0(M, data, cfg, th))
    log("STAGE 0 DONE")
    for n in stages:
        for s in seeds:
            for member in PAIRS[n - 1]:
                read_item(net, M, data, cfg, th, f"{member}_s{s}")
                dump(RESULTS / "summary.json", summary(cfg))
            log(f"STAGE {n} s{s} DONE")
    log(f"summary so far: {summary(cfg)['rule']}")
    log(f"{EXPERIMENT} DONE")
    return 0


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", type=int, choices=range(1, len(PAIRS) + 1), help="pairs to read (default all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1], choices=(0, 1))
    ap.add_argument("--summary", action="store_true", help="only rewrite results/summary.json from the item files present")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stages or list(range(1, len(PAIRS) + 1)), args.seeds, args.summary)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()

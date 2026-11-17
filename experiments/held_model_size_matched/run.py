"""P1.27: the held-model reading (reading 3, the per-site linear map, main; reading 1, one gain per unit, beside it) on
E187's size-matched pairs, on the 67M (plan reports/plans/2026-11-18_plan_held-model-size-matched.md; card card.md;
settings, tolerances, code checks and rules in results/predictions.json, which the run refuses to start without,
committed unchanged). The measurement is commute.held_model through E197's `read` and `windows`
(experiments/held_model_retention/run.py), unchanged: S = 1 - KL(F || H) / KL(F || M) at the last position of 512 read
windows, the maps fitted on 16,384 fit windows of the item's own task. No bars and no labels.

E187 (experiments/size_matched) matched each of its nine built pairs in update size: every saved update of the side with
the larger mean size ||dW|| / ||W0|| (24 matrices) is multiplied by one number c (E187's results/m67.json), so the two
sides' mean sizes agree. An item here is one saved state at its own size (c = 1) or, on the larger side, times c. Each
item's update size (whole update and MLP) is reported; its unscaled size must equal E187's within size_rel.

Stages, cheapest first:
    0  alone, the code checks: do-nothing on the scaling path (copy24 seed 0 times 0), E197's gains-per-block check
       with its update times gains_c (still a gain per block, so every reading reproduces F), and four unscaled items
       that E197 read, which must repeat E197's S within repro_tol. Any miss: exit 1, results/stage0.json "stopped".
       Skipped on a rerun once stage0.json says done.
    1-9  one E187 pair per stage (SM.BUILT order): its states at c = 1 not read in an earlier stage, then its larger
       side times c. Each item writes results/items/<item>.json and results/large/units/<item>.npz when it finishes and
       is skipped once its JSON is there; results/summary.json is rewritten after each item.
Exit 0 done, 1 a stop. The log is standard output:

    python experiments/held_model_size_matched/run.py [STAGE ...] 2>&1 | tee -a experiments/held_model_size_matched/results/run.log
    python experiments/held_model_size_matched/run.py --summary     (no GPU: the summary from the item files present)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

import commute.variable_read as VR
from commute import held_model as HM
from commute import old_tasks as OT
from commute import rule_retention as R
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed

HMR: Any = load_experiment("held_model_retention", "run")  # E197: read, windows, gains_block, ITEMS
SM: Any = load_experiment("size_matched", "run")  # E187: BUILT, pairs, times, OUT (m67.json)
CELL = "P1.27"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, ITEMS, LARGE = RESULTS / "predictions.json", RESULTS / "items", RESULTS / "large"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
REPO = EXPERIMENTS.parent
RULED = ("3", "1")  # the readings the rules read: the per-site map (main) and one gain per unit beside it
# E187's task names that are an old task of commute.old_tasks (E197's windows); every other task is a key table
OT_NAMES = {"c3/copy": "copy24", "c3/cipher": "cipher24", "c3/rank1": "rank1", "e46/ident32": "ident32", "e46/rand32": "rand32",
            "e62/copy": "e62_copy", "e62/lookup": "e62_lookup", "e53/lowercase": "lowercase", "e53/cipher": "cipher50",
            "e53/broad_cipher": "cipher500"}  # fmt: skip
log, dump, stop = HMR.log, HMR.dump, HMR.stop


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    n_fit: int  # E197's
    fit_seed: int
    n_read: int
    read_seed: int
    sum_batch: int
    check_seed: int  # E197's gains-per-block check
    gain_lo: float
    gain_hi: float
    gains_c: float  # the scaled gains check: each block's gain 1 + gains_c (g - 1) stays away from 0
    control_state: str  # whose task the stage 0 checks read, and the do-nothing's update

    def __post_init__(self) -> None:
        assert 0 < self.n_fit <= OT.DSizes().D // 2 and 0 < self.n_read and 0 < self.gain_lo < self.gain_hi
        assert 0 < self.gains_c and self.gains_c * (1 + self.gain_hi) < 1, "a scaled block gain could reach 0"


@dataclass(frozen=True)
class Item:
    state: str  # E187's state name, e.g. "c3/cipher_s0"
    task: str
    file: Path
    c: float
    pair: str | None  # the pair whose c scales it; None at c = 1

    @property
    def name(self) -> str:
        return self.state.replace("/", "-") + (f"_scaled_{self.pair}" if self.pair else "")


# ---- E187's pairs ----------------------------------------------------------------------------------------------------


@dataclass
class Pairs:
    names: dict[str, tuple[list[str], list[str]]]  # pair -> (reuse states, new states)
    inv: dict[str, Any]  # E187's inventory: states -> task and file, tasks -> windows
    m67: dict[str, Any]  # E187's per pair: larger, c, sizes

    def item(self, state: str, pair: str | None) -> Item:
        s = self.inv["states"][state]
        return Item(state, s["task"], REPO / s["file"], self.m67[pair]["c"] if pair else 1.0, pair)

    def larger(self, pair: str) -> list[str]:
        return self.names[pair][0 if self.m67[pair]["larger"] == "reuse" else 1]

    def e187_size(self, state: str) -> float:
        for pair, (reuse, new) in self.names.items():
            for side, states in (("reuse", reuse), ("new", new)):
                if state in states:
                    return self.m67[pair]["sizes"][side][states.index(state)]
        raise KeyError(state)

    def stages(self) -> list[list[Item]]:
        """Per pair (SM.BUILT order) its unscaled states not in an earlier stage, then its larger side times c."""
        seen: set[str] = set()
        out = []
        for pair in SM.BUILT:
            todo = [self.item(s, None) for s in [*self.names[pair][0], *self.names[pair][1]] if s not in seen]
            seen |= {i.state for i in todo}
            out.append(todo + [self.item(s, pair) for s in self.larger(pair)])
        return out


def load_pairs() -> Pairs:
    names, _, inv = SM.pairs()
    return Pairs(names, inv, json.loads(SM.OUT.read_text())["pairs"])


# ---- windows and fine-tunes ------------------------------------------------------------------------------------------


def old_task(task: str, spec: dict[str, Any]) -> OT.OldTask:
    """E197's old task where E187's task is one; else a key table from E187's windows file (and its held-out keys' file)."""
    if task in OT_NAMES:
        return OT.TASKS[OT_NAMES[task]]
    assert spec["format"] == "kv", (task, spec)
    p = REPO / spec["path"]
    held = p.with_name(p.name.replace("_trained", "_new"))
    name = task.replace("/", "_")
    return OT.OldTask(name, "kv", name, {}, (p, held) if "_trained" in p.name and held.is_file() else (p,))


@dataclass
class Data:
    sets: VR.Sets
    dcfg: OT.DSizes
    tasks: dict[str, Any]
    cache: dict[str, tuple[Tensor, Tensor]]

    def halves(self, task: str) -> tuple[Tensor, Tensor]:
        """(fit, read) as commute.old_tasks builds them (twins share them), for E197's `windows`."""
        t = old_task(task, self.tasks.get(task, {}))
        if t.group not in self.cache:
            self.cache[t.group] = OT.task_windows(t, self.sets, self.dcfg)[:2]
        return self.cache[t.group]


def rel_norm(dw: dict[str, Tensor], W0: dict[str, Tensor], keys: list[str]) -> float:
    num = sum(float(dw[f"{k}.weight"].double().pow(2).sum()) for k in keys)
    return math.sqrt(num / sum(float(W0[k].double().pow(2).sum()) for k in keys))


def sizes(dw: dict[str, Tensor], W0: dict[str, Tensor]) -> dict[str, float]:
    """||dW|| / ||W0|| over the 24 matrices (E187's size) and over the 8 MLP matrices."""
    return {"whole": rel_norm(dw, W0, list(W0)), "mlp": rel_norm(dw, W0, [k for k in W0 if ".mlp." in k])}


# ---- reading -------------------------------------------------------------------------------------------------------


def save_units(name: str, fit: HM.Fit) -> None:
    (LARGE / "units").mkdir(parents=True, exist_ok=True)
    arrays = {f"r{r}_{k}{l}_{f}": getattr(p, f).numpy().astype(dt) for r, ps in fit.pairs.items() for (k, l), p in ps.items()
              for f, dt in (("partner", np.int16), ("gain", np.float32), ("err", np.float16))}  # fmt: skip
    np.savez_compressed(LARGE / "units" / f"{name}.npz", **arrays)


def read_delta(net: Any, M: R.Model, data: Data, cfg: Config, th: dict[str, Any], name: str, task: str, dw: dict[str, Tensor]) -> dict[str, Any]:
    """F = M + dw, read by E197's `read` on the task's windows; the item's record with its update sizes."""
    F = OT.with_delta(net, dw)
    fit, rd = HMR.read(R.Model(F.net), M, *HMR.windows(data, task, cfg, M), cfg, th, name)
    save_units(name, fit)
    del F, fit
    torch.cuda.empty_cache()
    return {"experiment": EXPERIMENT, "item": name, "task": task, "size": sizes(dw, net.W0)} | rd


def read_item(net: Any, M: R.Model, data: Data, cfg: Config, th: dict[str, Any], it: Item, pairs: Pairs) -> dict[str, Any]:
    path = ITEMS / f"{it.name}.json"
    if path.is_file():
        log(f"{it.name}: done before, skipped")
        return json.loads(path.read_text())
    ITEMS.mkdir(parents=True, exist_ok=True)
    dw = OT.load_dw(net, it.file)
    own, e187 = sizes(dw, net.W0)["whole"], pairs.e187_size(it.state)
    head = {"state": it.state, "pair": it.pair, "c": it.c, "size_unscaled": own, "size_e187": e187}
    if not abs(own / e187 - 1) <= th["size_rel"]:
        stop(path.with_suffix(".stopped.json"), head, f"{it.name}: size {own:.9g} is not E187's {e187:.9g} within size_rel")
    out = head | read_delta(net, M, data, cfg, th, it.name, it.task, SM.times(dw, it.c))
    log(f"{it.name}: c {it.c:.6g}, size whole {out['size']['whole']:.6g}, MLP {out['size']['mlp']:.6g}; done in {out['seconds']:.0f} s")
    dump(path, out)
    return out


# ---- stage 0: the code checks --------------------------------------------------------------------------------------


def check_misses(rd: dict[str, Any], above_floor: bool, th: dict[str, Any]) -> list[str]:
    out = [] if (rd["kl_FM"] > th["kl_floor"]) == above_floor else [f"KL(F || M) {rd['kl_FM']:.3g}"]
    return out + [f"reading {r} logit error {v['logit_error']:.3g}" for r, v in rd["readings"].items() if not v["logit_error"] <= th["logit_tol"]]


def stage0(net: Any, M: R.Model, data: Data, cfg: Config, th: dict[str, Any], want: dict[str, Any], pairs: Pairs) -> dict[str, Any]:
    ctrl = pairs.item(cfg.control_state, None)
    dw = OT.load_dw(net, ctrl.file)
    Fg, _ = HMR.gains_block(M, cfg)
    pF, pM = dict(Fg.net.named_parameters()), dict(M.net.named_parameters())
    gains = {f"{k}.weight": pF[f"{k}.weight"] - pM[f"{k}.weight"] for k in net.W0}
    out: dict[str, Any] = {"experiment": EXPERIMENT, "items": {}}
    bad = []
    for name, d, above in (("do_nothing", SM.times(dw, 0.0), False), ("gains_block_scaled", SM.times(gains, cfg.gains_c), True)):
        rd = read_delta(net, M, data, cfg, th, name, ctrl.task, d)
        rd["misses"] = check_misses(rd, above, th)
        out["items"][name] = rd
        bad += [f"{name}: {m}" for m in rd["misses"]]
        log(f"{name}: {'checks hold' if not rd['misses'] else 'MISSES ' + '; '.join(rd['misses'])}")
    del Fg
    for state, e197 in want["reproduce"].items():
        rd = read_item(net, M, data, cfg, th, pairs.item(state, None), pairs)
        ref = json.loads((HMR.ITEMS / f"{e197}.json").read_text())["readings"]
        diff = {r: abs(rd["readings"][r]["S"] - ref[r]["S"]) for r in HM.READINGS}
        out["items"][f"reproduce {state}"] = {"e197_item": e197, "abs_S_diff": diff}
        miss = [f"reading {r} |S - E197's| {v:.3g}" for r, v in diff.items() if not v <= th["repro_tol"]]
        log(f"reproduce {state} against E197's {e197}: |S diff| {diff}; {'holds' if not miss else 'MISSES'}")
        bad += [f"{state}: {m}" for m in miss]
    if bad:
        stop(RESULTS / "stage0.json", out, "stage 0: " + "; ".join(bad))
    return out | {"done": True}


# ---- summary -------------------------------------------------------------------------------------------------------


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def sides_S(S: dict[str, dict[str, float | None]], items: dict[str, list[str]], r: str) -> dict[str, list[float]] | None:
    """Per side the items' S in order; None if any item is missing or null (no gap then)."""
    vals = {side: [S.get(i, {}).get(r) for i in names] for side, names in items.items()}
    if any(v is None for vs in vals.values() for v in vs):
        return None
    return {side: [float(v) for v in vs if v is not None] for side, vs in vals.items()}


def pair_reading(S: dict[str, Any], pairs: Pairs, pair: str, r: str) -> dict[str, Any]:
    """The filed rules for one pair and reading (predictions.json "rules"); null where an item is missing or null."""
    big = pairs.m67[pair]["larger"]
    unscaled = {side: [pairs.item(s, None).name for s in states] for side, states in zip(("reuse", "new"), pairs.names[pair], strict=True)}
    scaled = {side: [pairs.item(s, pair if side == big else None).name for s in states] for side, states in zip(("reuse", "new"), pairs.names[pair], strict=True)}
    o, m = sides_S(S, unscaled, r), sides_S(S, scaled, r)
    if o is None or m is None:
        return {"missing": True}
    orig, matched = statistics.fmean(o["reuse"]) - statistics.fmean(o["new"]), statistics.fmean(m["reuse"]) - statistics.fmean(m["new"])
    se = math.sqrt(sum(statistics.variance(v) / len(v) for v in o.values())) if all(len(v) > 1 for v in o.values()) else None
    shifts = [b - a for a, b in zip(o[big], m[big], strict=True)]
    shift = statistics.fmean(shifts)
    return {"S_original": o, "S_matched": m, "gap_original": orig, "gap_matched": matched, "se_original": se,
            "shift_by_state": shifts, "shift_mean": shift,
            "R1_sign": "zero" if sign(orig) * sign(matched) == 0 else "same" if sign(orig) == sign(matched) else "flipped",
            "R2_kept": matched / orig if orig != 0 else None,
            "R3_shift_within_se": abs(shift) <= se if se is not None else None}  # fmt: skip


def summary() -> dict[str, Any]:
    pairs = load_pairs()
    items = {p.stem: json.loads(p.read_text()) for p in ITEMS.glob("*.json") if not p.stem.endswith(".stopped")}
    S = {k: {r: v["readings"][r]["S"] for r in HM.READINGS} for k, v in items.items()}
    out: dict[str, Any] = {"experiment": EXPERIMENT, "cell": CELL, "S": S, "size": {k: v["size"] for k, v in items.items()}, "pairs": {}}
    for pair in SM.BUILT:
        p = pairs.m67[pair]
        kl = {}
        for s in pairs.larger(pair):
            a, b = items.get(pairs.item(s, None).name), items.get(pairs.item(s, pair).name)
            kl[s] = b["kl_FM"] / a["kl_FM"] / p["c"] ** 2 if a and b else None  # leading order: 1
        out["pairs"][pair] = {"larger": p["larger"], "c": p["c"], "kl_ratio_over_c2": kl, **{f"reading {r}": pair_reading(S, pairs, pair, r) for r in RULED}}
    return out


def main(stages: list[int], summary_only: bool) -> int:
    pred = json.loads(PRED.read_text())
    if pred["experiment"] == "E???":
        log("STOP: predictions.json has no experiment number")
        return 1
    if summary_only:
        dump(RESULTS / "summary.json", summary())
        return 0
    from commute import vpd
    from commute.steering import Net

    cfg, th = Config(**pred["settings"]), pred["thresholds"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E197
    log(f"{EXPERIMENT} ({CELL}) start; stages {stages}; settings {asdict(cfg)}; thresholds {th}")
    net = Net("cuda")
    net.net.requires_grad_(False)
    M, dcfg, pairs = R.Model(net.net), OT.DSizes(), load_pairs()
    data = Data(VR.make_sets(dcfg, vpd.tokenizer(), log), dcfg, pairs.inv["tasks"], {})
    s0 = RESULTS / "stage0.json"
    if not (s0.is_file() and json.loads(s0.read_text()).get("done")):
        dump(s0, stage0(net, M, data, cfg, th, pred["code_checks"], pairs))
    log("STAGE 0 DONE")
    stages = [k for k in stages if k]  # 0 alone: stage 0 only
    plan, t0, n = pairs.stages(), time.time(), 0
    left = sum(len(plan[k - 1]) for k in stages)
    for k in stages:
        for it in plan[k - 1]:
            read_item(net, M, data, cfg, th, it, pairs)
            n, left = n + 1, left - 1
            log(f"projected minutes left in this job: {(time.time() - t0) / n * left / 60:.1f}")
            dump(RESULTS / "summary.json", summary())
        log(f"STAGE {k} DONE ({SM.BUILT[k - 1]})")
    for pair, v in summary()["pairs"].items():
        for r in RULED:
            g = v[f"reading {r}"]
            log(f"{pair} reading {r}: " + ("items missing" if g.get("missing") else
                f"gap original {g['gap_original']:.6g} (se {g['se_original']}), matched {g['gap_matched']:.6g}, "
                f"mean shift {g['shift_mean']:.3g}; R1 {g['R1_sign']}, R2 {g['R2_kept']}, R3 {g['R3_shift_within_se']}"))  # fmt: skip
    log(f"{EXPERIMENT} DONE")
    return 0


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", type=int, choices=range(len(SM.BUILT) + 1), help="1-9 the pairs in SM.BUILT order (default all); 0 alone: stage 0 only")
    ap.add_argument("--summary", action="store_true", help="only rewrite results/summary.json from the item files present")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stages or list(range(1, len(SM.BUILT) + 1)), args.summary)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()

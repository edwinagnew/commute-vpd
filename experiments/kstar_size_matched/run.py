"""P1.38: E221's reuse score R at its three coarse levels (head_mlp, mlp_attn, block) on E187's nine built pairs with
each larger-side state rescaled by E187's c to its partner's mean update size: the 40 size-matched states E205 read
(E205's Item names, scaled by E187's `times`). Plan reports/plans/2026-11-20_plan_p1-38-size-matched-r.md; filed rules
in results/predictions.json, which the run refuses unless committed unchanged. The measurement is E221's `read_item`,
called unchanged with E221's filed settings (asserted at start): its K* search, alpha, polish_cap, grades, 512 read
windows, last position. The smaller side of each pair is E221's saved reading (same states, same code); stage 0 checks
that this code repeats it. Nothing is trained.

Stage 0, first and alone on every shard (results/stage0_<tag>.json, skipped once done; exit 1 on a miss): the control
state (a larger-side state) times 0 is the do-nothing on the scaling path and must read R = 1, f = 0, every s within
s_tol of 1 at every level; the same state times 1 must repeat E221's item file (R per order within repro_tol, same K*).
Then each scaled item writes results/items/<item>_<level>.json as it ends; a rerun skips what is written. Load checks per
item (exit 1 on a miss): KL(F || M) on D and the update size equal E205's saved values for the same scaled state.

    python experiments/kstar_size_matched/run.py --tag pod0 --shard 0 4 2>&1 | tee -a experiments/kstar_size_matched/results/run_pod0.log
    python experiments/kstar_size_matched/run.py --tag rehearse --out rehearse --items e54-cipher_own_s0_scaled_e54_own
    (on a pod: PRED_SHA=<sha> python ...)
    python experiments/kstar_size_matched/score.py     (no GPU)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import old_tasks as OT
from commute import rule_retention as R
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed

E221: Any = load_experiment("kstar_pairs", "run")  # World, read_item, Config, tuples, fname, LAST, RESULTS, PRED
E205: Any = E221.E205  # load_pairs, Item, sizes, ITEMS; E205.SM is E187's run (BUILT, times)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
E221.EXPERIMENT = EXPERIMENT  # E221's read_item stamps each record with its module's E-code; these records are ours
log, dump, stop = E221.log, E221.dump, E221.stop


@dataclass(frozen=True)
class Config(E221.Config):  # E221's filed settings, unchanged, and the stage 0 state
    control_state: str  # a larger-side E187 state: times 0 (do-nothing) and times 1 (repeats E221's item)


def load_config() -> tuple[Config, dict[str, Any]]:
    pred, e221 = json.loads(PRED.read_text()), json.loads(E221.PRED.read_text())["settings"]
    same = all(pred["settings"].get(k) == v for k, v in e221.items() if not k.endswith("_source"))
    assert same, "a measurement setting differs from E221's filed one"
    return config_from_settings(Config, {k: E221.tuples(v) for k, v in pred["settings"].items()}), pred["thresholds"]


def scaled_items() -> dict[str, Any]:
    """The larger side of every E187 pair times its c, in SM.BUILT order, by E205's item name (40 states)."""
    pairs = E205.load_pairs()
    return {it.name: it for p in E205.SM.BUILT for it in (pairs.item(s, p) for s in pairs.larger(p))}


class World(E221.World):
    """E221's world (M, windows, E187's inventory); `make` builds a state times its c."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        ctrl = self.pairs.item(cfg.control_state, None)
        assert cfg.control_state in {s for p in E205.SM.BUILT for s in self.pairs.larger(p)}, "the control state is not on a larger side"
        self.items = scaled_items() | {"do_nothing_c0": dataclasses.replace(ctrl, c=0.0), "repeat_c1": ctrl}

    def make(self, item: str) -> tuple[R.Model, Tensor, slice, dict[str, Any]]:
        """F = M + c dW (E187's `times`), D = the state's 512 read windows as E221 cut them, and the load record."""
        it = self.items[item]
        spec = self.pairs.inv["states"][it.state]
        dw = E205.SM.times(OT.load_dw(self.net, E221.REPO / spec["file"]), it.c)
        rec: dict[str, Any] = {"state": it.state, "pair": it.pair, "c": it.c, "size": E205.sizes(dw, self.net.W0)}
        if it.c != 0:  # E205's reading of the same state at the same c (c = 1: its unscaled item)
            saved = json.loads((E205.ITEMS / f"{it.name}.json").read_text())
            rec |= {"size_saved": saved["size"]["whole"], "kl_fm_saved": saved["kl_FM"], "saved_from": "E205"}
        return R.Model(OT.with_delta(self.net, dw).net), self.read(self.read_half187(spec["task"])), E221.LAST, rec


def stage0(w: World, th: dict[str, Any], tag: str) -> None:
    path, levels, out_dir = RESULTS / f"stage0_{tag}.json", list(w.cfg.levels), RESULTS / f"stage0_{tag}"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    out: dict[str, Any] = {"experiment": EXPERIMENT, "control_state": w.cfg.control_state}
    miss: list[str] = []
    for lv, it in E221.read_item(w, "do_nothing_c0", levels, out_dir, th, 1).items():  # one order, as E221's do-nothing
        rs = it["searches"]["main"]
        dev = max(abs(1 - v) for sv in it["s"].values() for v in sv)
        R_ = [r[f"R_{g}"] for r in rs for g in ("read", "all")]
        out[f"do_nothing_{lv}"] = {"max_dev_s": dev, "f": [r["f"] for r in rs], "R": R_, "n_changed": it["n_changed"]}
        if not (dev <= th["s_tol"] and it["n_changed"] == 0 and all(r["f"] == 0 for r in rs) and all(abs(x - 1) <= th["s_tol"] for x in R_)):
            miss.append(f"do-nothing at {lv}: {out[f'do_nothing_{lv}']}")
    for lv, it in E221.read_item(w, "repeat_c1", levels, out_dir, th, w.cfg.orders).items():
        ref = json.loads((E221.RESULTS / "items" / f"{E221.fname(w.cfg.control_state)}_{lv}.json").read_text())["searches"]["main"]
        mine = it["searches"]["main"]
        diff = max(abs(a[f"R_{g}"] - b[f"R_{g}"]) for a, b in zip(mine, ref, strict=True) for g in ("read", "all"))
        same_K = all([u["name"] for u in a["K"]] == [u["name"] for u in b["K"]] for a, b in zip(mine, ref, strict=True))
        out[f"repeat_{lv}"] = {"max_abs_R_diff": diff, "same_K": same_K, "R_read": [r["R_read"] for r in mine], "R_read_E221": [r["R_read"] for r in ref]}
        if not (same_K and diff <= th["repro_tol"]):
            miss.append(f"repeat at {lv}: {out[f'repeat_{lv}']}")
    log(f"stage 0: {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}; {json.dumps(out)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def main(chosen: list[str] | None, shard: tuple[int, int], tag: str, out: str) -> int:
    cfg, th = load_config()
    every = list(scaled_items())
    todo = chosen or every[shard[0] :: shard[1]]
    assert set(todo) <= set(every), f"unknown items {sorted(set(todo) - set(every))}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E221
    log(f"{EXPERIMENT} start, tag {tag}; {len(todo)} of {len(every)} items {todo}; settings {dataclasses.asdict(cfg)}; thresholds {th}")
    w = World(cfg)
    stage0(w, th, tag)
    for k, item in enumerate(todo):
        E221.read_item(w, item, list(cfg.levels), RESULTS / out, th, cfg.orders)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=None, help="scaled items by E205 name (default: this shard of all 40, in pair order)")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this machine's name: stage 0 writes results/stage0_<tag>.json")
    ap.add_argument("--out", default="items", help="results subdirectory for the item files (the rehearsal writes elsewhere)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.items, (args.shard[0], args.shard[1]), args.tag, args.out)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

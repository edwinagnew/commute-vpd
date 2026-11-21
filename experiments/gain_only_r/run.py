"""P1.41 (E236): R and the held model on compare-lead's E231 gain-only states, the KL-matched state per (reference,
seed). Card reports/plans/2026-11-20_plan_p1-41-card.md, plan reports/plans/2026-11-20_plan_p1-41-gain-only-r.md;
filed settings, thresholds and rules in results/predictions.json, which the run refuses unless committed unchanged.
Nothing is trained, and no measurement is new: each reader is a finished experiment's, with its own filed settings.

    VPD     R, f, |K*| at the VPD featurization with site remainders: commute.kstar_vpd through E226's guards and search
            (experiments/kstar_vpd/run.py `guards`, `one_search`; E226's settings and thresholds)
    coarse  R, f, |K*| at head_mlp, mlp_attn and block: E221's `read_item` (experiments/kstar_pairs/run.py), unchanged
    held    the held model, readings 1 (one gain per neuron and head, the ruled one), 2 and 3: E197's `read` and
            `windows` (experiments/held_model_retention/run.py) with E205's settings and thresholds

An item is E231's state file (a bf16 weight change W0 diag(g - 1) on down_proj and o_proj, zero elsewhere) added to M,
read on the windows E221 and E226 read for the same task: E187's tasks as E205 built them; the twins case_label and fresh
as E196's case_real/case_twin and repeat_real/repeat_twin (asserted the same task file); E58's twin key tables from
E231's spec. 512 read windows (E196's subset), last position. Per item and reader, results/<out>/<reader>/<item>.json is
written when it ends and skipped on a rerun; results/large/vpd_<item>.npz holds the VPD grades and K* membership.

Stage 0 runs first and alone on every shard (results/stage0_<tag>.json, skipped once done; exit 1 on a miss):
do-nothing (the control state times 0) at VPD (no search: every s is 1, so R is 1 for any K*), at the three coarse levels
(R = 1, f = 0) and in the held model (KL 0, logit error at most logit_tol); the exact fp32 build of the control's gains in
the held model (readings 1 and 3 reproduce F within logit_tol); the control's bf16 state in the held model (its 1 - S is
logged as the export's floor). With --reproduce (the rehearsal) also the full fine-tune of the control's task and seed
through every reader, against E226's, E221's and E205's saved items.

    python experiments/gain_only_r/run.py --tag pod0 --shard 0 9 2>&1 | tee -a experiments/gain_only_r/results/run_pod0.log
    python experiments/gain_only_r/run.py --tag rehearse --out rehearse --reproduce --items e58/cipher_s0 case_label/twin_s0
    (on a pod: PRED_SHA=<sha> python ...)
    python experiments/gain_only_r/score.py     (no GPU: the rules and the per-item table beside the full fine-tunes)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import gains as G
from commute import kstar_vpd as KV
from commute import old_tasks as OT
from commute import rule_retention as R
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

KP: Any = load_experiment("kstar_pairs", "run")  # E221: World, read_item, load_config, e187_task, load_checks, fname
VPD: Any = load_experiment("kstar_vpd", "run")  # E226: load_config, guards, one_search, membership, COLUMNS
E205: Any = load_experiment("held_model_size_matched", "run")  # Config, sizes, PRED, ITEMS; E205.HMR is E197's read, windows
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
REPO = EXPERIMENTS.parent
KP.EXPERIMENT = EXPERIMENT  # E221's read_item stamps its records with its module's EXPERIMENT
log, dump, stop, fname = KP.log, KP.dump, KP.stop, KP.fname
VCFG, VTH = VPD.load_config()  # E226's filed settings and thresholds, unchanged
KCFG, KTH = KP.load_config()  # E221's
_h = json.loads(E205.PRED.read_text())
HCFG, HTH = E205.Config(**_h["settings"]), _h["thresholds"]  # E205's
EXACT = ("1", "3")  # the held readings that hold a per-neuron and per-head gain exactly (reading 2 makes the MLP one unit)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    handoff: str  # E231's manifest (repo-relative)
    gains_dir: str  # E231's fp32 gains per (task, seed)
    match: str  # which of E231's kept states: "kl" (the KL-matched one)
    seeds: tuple[int, ...]
    n_items: int
    control: str  # the gain item whose state times 0 is do-nothing and whose gains build the exact control
    twins: tuple[tuple[str, str], ...]  # E231 task -> E196's old task of the same file (E221 read those names)

    def __post_init__(self) -> None:
        assert self.match == "kl", "the cell reads the KL-matched state (47 of 81 gain-matched ones hit the step cap)"


@dataclass(frozen=True)
class Item:
    name: str  # "<reference>_s<seed>", "do_nothing", "gain_exact", or "full/<E187 state>"
    task: str  # whose windows
    seed: int
    ref: str  # E231's reference the state was KL-matched to
    file: str | None  # repo-relative state file; None: the exact fp32 build of the gains
    sha256: str | None
    scale: float = 1.0


def load_config() -> Config:
    s = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def gain_items(cfg: Config, hand: dict[str, Any]) -> dict[str, Item]:
    """Every KL-matched state, grouped by task (windows are built once per task group)."""
    out: dict[str, Item] = {}
    for st in hand["states"].values():
        for ref in (f.split(":", 1)[1] for f in st["for"] if f.startswith(f"{cfg.match}:")):
            seed = int(re.search(r"_s(\d+)_step\d+\.pt$", st["file"]).group(1))  # type: ignore[union-attr]
            out[f"{ref}_s{seed}"] = Item(f"{ref}_s{seed}", st["task"], seed, ref, st["file"], st["sha256"])
    assert len(out) == cfg.n_items and {i.seed for i in out.values()} == set(cfg.seeds), (len(out), cfg.n_items)
    return dict(sorted(out.items(), key=lambda kv: (kv[1].task, kv[1].seed, kv[0])))


class World:
    """M and E221's window builders (E221's World), the VPD split's pieces, and every item's F, built once per item."""

    def __init__(self, cfg: Config) -> None:
        from commute import vpd

        self.kp = KP.World(KCFG)
        self.cfg, self.M, self.net = KCFG, self.kp.M, self.kp.net  # E221's read_item reads w.cfg and w.M
        assert (HCFG.n_read, HCFG.read_seed) == (KCFG.n_read, KCFG.read_seed), "the held model would read other windows than R"
        model = self.net.model
        self.uv, self.delta, self.keys = vpd.component_weights(model), vpd.weight_deltas(model), vpd.module_keys(model)
        w0 = vpd.checkpoint_weights()[0]
        self.w0_differs = [k for k in self.keys if not torch.equal(self.M.net.get_submodule(k).weight.cpu(), w0[k])]
        self.bases: dict[str, Any] = {}
        hand = json.loads((REPO / cfg.handoff).read_text())
        self.tasks = hand["tasks"] | self.kp.pairs.inv["tasks"]  # E187's spec where both name a task (E221 read it)
        self.twins, self.gains_dir = dict(cfg.twins), REPO / cfg.gains_dir
        for t, old in self.twins.items():
            assert (REPO / hand["tasks"][t]["path"]).resolve() == OT.TASKS[old].sources[0].resolve(), (t, old)
        self.match, self.gain = cfg.match, gain_items(cfg, hand)
        c = self.gain[cfg.control]
        state = c.name  # the full fine-tune of the control's task and seed, an E187 state
        self.items = self.gain | {"do_nothing": dataclasses.replace(c, name="do_nothing", scale=0.0),
                       "gain_exact": dataclasses.replace(c, name="gain_exact", file=None, sha256=None),
                       f"full/{state}": Item(f"full/{state}", self.kp.pairs.inv["states"][state]["task"], c.seed, c.ref, None, None)}  # fmt: skip
        self.cache: dict[str, tuple[Tensor, Tensor]] = {}
        self.last: tuple[str, Any] = ("", None)

    def halves(self, task: str) -> tuple[Tensor, Tensor]:
        """(fit, read) halves of the task's D as commute.old_tasks builds them; E197's `windows` calls this."""
        t = OT.TASKS[self.twins[task]] if task in self.twins else KP.e187_task(task, self.tasks)
        if t.group not in self.cache:
            self.cache = {t.group: OT.task_windows(t, self.kp.sets, self.kp.dcfg)[:2]}
        return self.cache[t.group]

    def change(self, it: Item) -> dict[str, Tensor]:
        if it.file is None:  # the same gains, exact, fp32
            g = torch.load(self.gains_dir / f"{it.task.replace('/', '_')}_s{it.seed}.pt", weights_only=True)[f"{self.match}:{it.ref}"]["gains"]
            return {f"{k}.weight": v for k, v in G.delta(G.Gains.of(g), self.net.W0).items()}
        path = REPO / it.file
        assert hashlib.sha256(path.read_bytes()).hexdigest() == it.sha256, f"{path} is not the handed-over state"
        return {k: v * it.scale for k, v in OT.load_dw(self.net, path).items()}

    def make(self, name: str) -> tuple[R.Model, Tensor, slice, dict[str, Any]]:
        """F, D, the read positions and the load record (E221's World.make interface); the last item's kept."""
        if self.last[0] != name:
            self.last = ("", None)  # free the previous F first
            torch.cuda.empty_cache()
            it = self.items[name]
            if name.startswith("full/"):
                got = self.kp.make(name.removeprefix("full/"))
            else:
                dw = self.change(it)
                got = (R.Model(OT.with_delta(self.net, dw).net), self.kp.read(self.halves(it.task)[1]), KP.LAST, {"size": E205.sizes(dw, self.net.W0)})
            self.last = (name, got)
        return self.last[1]


# ---- the three readers ----------------------------------------------------------------------------------------------


def read_vpd(w: World, name: str, out: Path, search: bool = True) -> dict[str, Any]:
    """E226's reading of one item (its main search only, as E226 read E187's states); no search: grades only."""
    path = out / "vpd" / f"{fname(name)}.json"
    if path.is_file():
        return json.loads(path.read_text())
    t0 = time.time()
    F, toks, at, rec = w.make(name)
    w.bases = w.bases or KV.bases(w.uv, VCFG.split_tol)
    feat = KV.featurize(w.M, F, w.uv, w.delta, w.bases)
    ctx = KV.prepare(F, w.M, feat, toks, at, VCFG.batch)
    base = {"experiment": EXPERIMENT, "item": name, "n_windows": len(toks), "n_vars": feat.n, "n_changed": len(feat.changed),
            "kl_none": float(ctx.kl_none.mean()), "kl_fm": float(ctx.kl_fm.mean()), "checks": feat.checks,
            "split_shares": KV.norm_shares(feat), "gates": KV.gates(ctx, feat.n)} | rec  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    miss = VPD.guards(name, feat, KV.undecomposed_differences(w.M, F, w.keys), base["gates"], VTH) + KP.load_checks(name, base, KTH)
    if miss:
        stop(path.with_suffix(".stopped.json"), base, f"{name} VPD: " + "; ".join(miss))
    s = KV.grades(w.M, F, feat, toks, at, VCFG.grade_batch)
    base["max_dev_s"] = float((1 - s).abs().max())
    R_read = None
    if not search or not base["kl_none"] > VTH["kl_floor"]:
        res = base | {"reading": "grades only" if not search else "no behaviour"}
    else:
        found, arrays = VPD.one_search(ctx, s, feat, VCFG, f"{name} VPD")
        LARGE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(LARGE / f"vpd_{out.name}_{fname(name)}.npz", s=s.float().numpy(), changed=VPD.membership(sorted(feat.changed), feat.n),
                            site_names=np.array([x.name for x in feat.sites]), site_off=np.array([x.off for x in feat.sites]),
                            columns=np.array(VPD.COLUMNS), **{f"main_{k}": v for k, v in arrays.items()})  # fmt: skip
        res, R_read = base | {"reading": "read", "searches": {"main": found}}, found["polished"]["R_read"]
    res |= {"seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, res)
    log(f"{name} VPD: {feat.n} variables, {len(feat.changed)} changed, KL(F || M) {base['kl_fm']:.6g}, max |1 - s| {base['max_dev_s']:.3g}; "
        f"{res['reading']}, R read {R_read}; {res['seconds']:.0f} s")  # fmt: skip
    ctx.hy.close()
    del ctx
    return res


def read_coarse(w: World, name: str, out: Path, orders: int | None = None) -> dict[str, dict[str, Any]]:
    return KP.read_item(w, name, list(KCFG.levels), out / "coarse", KTH, orders or KCFG.orders)


def read_held(w: World, name: str, out: Path) -> dict[str, Any]:
    path = out / "held" / f"{fname(name)}.json"
    if path.is_file():
        return json.loads(path.read_text())
    F, _, _, rec = w.make(name)
    _, rd = E205.HMR.read(F, w.M, *E205.HMR.windows(w, w.items[name].task, HCFG, w.M), HCFG, HTH, name)
    res = {"experiment": EXPERIMENT, "item": name, "task": w.items[name].task} | rec | rd
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, res)
    return res


# ---- stage 0 --------------------------------------------------------------------------------------------------------


def stage0(w: World, cfg: Config, tag: str, out: Path, reproduce: bool) -> None:
    path, here = RESULTS / f"stage0_{tag}.json", RESULTS / f"stage0_{tag}"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    rec: dict[str, Any] = {"experiment": EXPERIMENT, "w0_differs": w.w0_differs}
    miss = [f"VPD target weights differ from M at {w.w0_differs}"] if w.w0_differs else []
    v = read_vpd(w, "do_nothing", here, search=False)
    rec["do_nothing_vpd"] = {"n_changed": v["n_changed"], "max_dev_s": v["max_dev_s"]}
    miss += [f"do-nothing VPD {rec['do_nothing_vpd']}"] if v["n_changed"] or not v["max_dev_s"] <= VTH["s_tol"] else []
    for lv, it in read_coarse(w, "do_nothing", here, 1).items():
        rs = it["searches"]["main"]
        rec[f"do_nothing_{lv}"] = {"max_dev_s": max(abs(1 - x) for sv in it["s"].values() for x in sv), "f": [r["f"] for r in rs], "R": [r[f"R_{g}"] for r in rs for g in ("read", "all")]}
        d = rec[f"do_nothing_{lv}"]
        miss += [] if d["max_dev_s"] <= KTH["s_tol"] and not any(d["f"]) and all(abs(x - 1) <= KTH["s_tol"] for x in d["R"]) else [f"do-nothing at {lv}: {d}"]
    for name, above in (("do_nothing", False), ("gain_exact", True)):
        h = read_held(w, name, here)
        rec[f"{name}_held"] = {"kl_FM": h["kl_FM"], **{r: h["readings"][r]["logit_error"] for r in EXACT}}
        bad = (h["kl_FM"] > HTH["kl_floor"]) != above or any(not h["readings"][r]["logit_error"] <= HTH["logit_tol"] for r in EXACT)
        miss += [f"{name} held model: {rec[f'{name}_held']}"] if bad else []
    S = read_held(w, cfg.control, out)["readings"]["1"]["S"]  # a scored item, read here first: the bf16 export's floor
    rec["export_floor_1_minus_S"] = None if S is None else 1 - S
    if reproduce:
        miss += reproduction(w, f"full/{cfg.control}", here, rec)
    log(f"stage 0: {rec}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, rec, "stage 0: " + "; ".join(miss))
    dump(path, rec | {"done": True})
    log("STAGE 0 DONE")


def reproduction(w: World, name: str, here: Path, rec: dict[str, Any]) -> list[str]:
    """The full fine-tune through this run's path against E226's, E221's and E205's saved items of the same state."""
    state, X = fname(name.removeprefix("full/")), EXPERIMENTS
    mine = read_vpd(w, name, here)["searches"]["main"]["polished"]["R_read"]
    ref = json.loads((X / "kstar_vpd/results/items" / f"{state}.json").read_text())["searches"]["main"]["polished"]["R_read"]
    rec["repro_vpd"] = {"R_read": mine, "E226": ref}
    miss = [] if abs(mine - ref) <= KTH["repro_tol"] else [f"VPD R {mine:.9f} against E226's {ref:.9f}"]
    for lv, it in read_coarse(w, name, here).items():
        a = [r["R_read"] for r in it["searches"]["main"]]
        b = [r["R_read"] for r in json.loads((X / "kstar_pairs/results/items" / f"{state}_{lv}.json").read_text())["searches"]["main"]]
        rec[f"repro_{lv}"] = {"R_read": a, "E221": b}
        miss += [] if len(a) == len(b) and all(abs(x - y) <= KTH["repro_tol"] for x, y in zip(a, b, strict=True)) else [f"{lv} R {a} against E221's {b}"]
    h, ref_h = read_held(w, name, here)["readings"], json.loads((E205.ITEMS / f"{state}.json").read_text())["readings"]
    rec["repro_held"] = {r: [h[r]["S"], ref_h[r]["S"]] for r in EXACT}
    return miss + [f"held reading {r} S {h[r]['S']} against E205's {ref_h[r]['S']}" for r in EXACT if not abs(h[r]["S"] - ref_h[r]["S"]) <= HTH["repro_tol"]]


def main(chosen: list[str] | None, shard: tuple[int, int], tag: str, out_name: str, reproduce: bool) -> int:
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E221 and E226
    w = World(cfg)
    every = list(w.gain)
    todo = chosen or every[shard[0] :: shard[1]]
    assert set(todo) <= set(every), f"unknown items {sorted(set(todo) - set(every))}"
    log(f"{EXPERIMENT} start, tag {tag}; {len(todo)} of {len(every)} items {todo}; settings {asdict(cfg)}")
    out = RESULTS / out_name
    stage0(w, cfg, tag, out, reproduce)
    for k, name in enumerate(todo):
        t0 = time.time()
        read_vpd(w, name, out)
        read_coarse(w, name, out)
        h = read_held(w, name, out)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({name}): held reading 1 S {h['readings']['1']['S']}; {time.time() - t0:.0f} s")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=None, help="items by name, '<reference>_s<seed>' (default: this shard of every item)")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this pod's name: stage 0 writes results/stage0_<tag>.json")
    ap.add_argument("--out", default="items", help="results subdirectory for the item files (the rehearsal writes elsewhere)")
    ap.add_argument("--reproduce", action="store_true", help="stage 0 also reads the control's full fine-tune against E226, E221, E205")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.items, (args.shard[0], args.shard[1]), args.tag, args.out, args.reproduce)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

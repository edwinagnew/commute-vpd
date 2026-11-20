"""P1.36 (tests T3 and T4 of reports/analyses/2026-11-18_analysis_definition-vs-experiments.md): R, f and the remainder
share of K* on the trained matched pairs, at every coarse level, because E214 (P1.35) passed at none
(reports/analyses/2026-11-19_analysis_e214-kstar-levels.md). Plan reports/plans/2026-11-19_plan_p1-36-pairs.md and its
appendix; filed settings, thresholds and rules in results/predictions.json, which the run refuses unless committed
unchanged. The measurement is commute.kstar, called as E214 (experiments/kstar_levels) calls it. Nothing is trained.

Items: every fine-tune of E196's five pairs at seeds 0 and 1 and of E187's nine size-matched pairs, each as saved (no
size matching). Where an E196 member is the same file as an E187 state (copy24, cipher24, ident32, rand32) it is read
once, under its E187 name. F is built and its windows D are cut exactly as E196 and E205 built them: 512 read windows
(E196's subset, read_seed), KL at the last position. Per item and level, results/<out>/<item>_<level>.json is written
when it ends; a rerun skips what is written. Both grades are saved: "read" (the last position, main) and "all".
Load checks per item (stop on a miss): KL(F || M) on D equals E208's or E205's saved value, and an E187 state's update
size equals E187's.
Stage 0 runs first and alone on every pod (results/stage0_<tag>.json, skipped once done; exit 1 on a miss): do-nothing
at every level reads R = 1, f = 0, every s within s_tol of 1; E195's x10 at block and mlp_attn repeats E214's K* and R.

    python experiments/kstar_pairs/run.py --tag pod3 --shard 3 8 2>&1 | tee -a experiments/kstar_pairs/results/run_pod3.log
    python experiments/kstar_pairs/run.py --tag rehearse --out rehearse --items twin_s0 e58/cipher_s0     (the $0 rehearsal)
    (on a pod: PRED_SHA=<sha> python ...)
    python experiments/kstar_pairs/score.py     (no GPU: the per-pair table and the filed rules from the item files present)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import kstar as K
from commute import old_tasks as OT
from commute import rho
from commute import rule_retention as R
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

P: Any = load_experiment("rule_retention", "run")  # clone, rescaled, COPY24, log, dump, stop
E196: Any = load_experiment("rule_retention_pairs", "run")  # Data, subset, fine_tune, STATES, NUMBERS
E205: Any = load_experiment("held_model_size_matched", "run")  # load_pairs, Data, sizes, SM (E187: BUILT, pairs)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
REPO = EXPERIMENTS.parent
E208_ITEMS = EXPERIMENTS / "rho_saved_items" / "results" / "items"
E214_ITEMS = EXPERIMENTS / "kstar_levels" / "results" / "items"
LAST, ALL = slice(-1, None), slice(None)
log, dump, stop = P.log, P.dump, P.stop


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    levels: tuple[str, ...]
    alpha: float
    alphas: tuple[float, ...]
    restricted_share: float
    orders: int  # head_mlp search orders per item (the exhaustive levels need one)
    order_seed: int
    polish_cap: int
    split_seed: int
    e196_pairs: tuple[tuple[str, str], ...]  # (reuse, new) as E196 named them
    e196_seeds: tuple[int, ...]
    e196_as_e187: tuple[tuple[str, str], ...]  # E196 member -> the E187 state stem saved in the same file
    n_read: int  # E196's read windows per item, and its seed
    read_seed: int
    n_max: int  # E173's numbers (E196's parity twin and fact windows)
    held_share: float
    control_task: str  # whose windows the do-nothing reads
    e195_layer: int  # the positive control x10: E195's circuit times e195_factor, read as E214 read it
    e195_C: tuple[int, ...]
    e195_factor: float

    def __post_init__(self) -> None:
        assert set(self.levels) == {"head_mlp", "mlp_attn", "block"}, "coarse levels only (cell P1.36: no neuron level)"
        assert self.alpha in self.alphas and list(self.alphas) == sorted(self.alphas) and 0 < self.restricted_share < 1 and self.orders >= 1
        assert set(self.e196_seeds) <= {0, 1} and {a for a, _ in self.e196_as_e187} <= {m for pr in self.e196_pairs for m in pr}


def tuples(v: Any) -> Any:
    return tuple(tuples(x) for x in v) if isinstance(v, list) else v


def load_config() -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    return config_from_settings(Config, {k: tuples(v) for k, v in pred["settings"].items()}), pred["thresholds"]


def fname(item: str) -> str:
    return item.replace("/", "-")


def pairs(cfg: Config) -> dict[str, tuple[list[str], list[str]]]:
    """The fourteen pairs, each (reuse items, new items): E196's five as 'e196_<reuse>', then E187's nine as 'e187_<id>'."""
    alias = dict(cfg.e196_as_e187)

    def name(m: str, s: int) -> str:
        return f"{alias.get(m, m)}_s{s}"

    out = {f"e196_{a}": ([name(a, s) for s in cfg.e196_seeds], [name(b, s) for s in cfg.e196_seeds]) for a, b in cfg.e196_pairs}
    names, _, _ = E205.SM.pairs()
    return out | {f"e187_{p}": names[p] for p in E205.SM.BUILT}


def items(cfg: Config) -> list[str]:
    return list(dict.fromkeys(i for r, n in pairs(cfg).values() for i in (*r, *n)))


def e187_task(task: str, tasks: dict[str, Any]) -> OT.OldTask:
    """E187's task as E205 read it. A key table E205 built from E187's windows file (E58's 96-key tables) used that file
    alone: E205's `old_task` adds the held-out keys' file only when it exists, and on E205's pod it did not. Built with
    both files, e58/cipher_s0 reads KL(F || M) 4.6463 against E205's 5.2393; with the trained file alone 5.2393 and
    e58/copy_s0 4.3984, as E205 (CPU check 2026-11-19, rehearsal job 61952's stop)."""
    t = E205.old_task(task, tasks.get(task, {}))
    return t if task in E205.OT_NAMES else dataclasses.replace(t, sources=t.sources[:1])


# ---- the items --------------------------------------------------------------------------------------------------------


class World:
    """M, E196's and E205's window builders, and every item's F; each input loaded when first needed."""

    def __init__(self, cfg: Config) -> None:
        from commute import variable_read as VR
        from commute import vpd
        from commute.steering import Net

        self.cfg, self.net = cfg, Net("cuda")
        self.net.net.requires_grad_(False)
        self.M = R.Model(self.net.net)
        tok, dcfg = vpd.tokenizer(), OT.DSizes()
        x = VR.make_numbers(tok, int(self.net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
        assert x.keys.tolist() == json.loads(E196.NUMBERS.read_text())["number_tokens"], "the number tokens are not E154's"
        sets = VR.make_sets(dcfg, tok, log)
        self.pairs = E205.load_pairs()
        self.sets, self.dcfg, self.d196 = sets, dcfg, E196.Data(sets, x, dcfg, {})
        self.cache: dict[str, Tensor] = {}  # an E187 task's read half, by windows group
        for m, st in cfg.e196_as_e187:  # an alias is the same file and the same windows' group
            for s in cfg.e196_seeds:
                spec = self.pairs.inv["states"][f"{st}_s{s}"]
                same = Path(OT.TASKS[m].states[s]).resolve() == (REPO / spec["file"]).resolve()
                assert same and E205.old_task(spec["task"], self.pairs.inv["tasks"].get(spec["task"], {})).group == OT.TASKS[m].group, (m, st, s)

    def read(self, toks: Tensor) -> Tensor:
        return E196.subset(toks, self.cfg, self.M)

    def read_half187(self, task: str) -> Tensor:
        t = e187_task(task, self.pairs.inv["tasks"])
        if t.group not in self.cache:
            self.cache = {t.group: OT.task_windows(t, self.sets, self.dcfg)[1]}
        return self.cache[t.group]

    def make(self, item: str) -> tuple[R.Model, Tensor, slice, dict[str, Any]]:
        """F, D, the read positions, and the load record: update sizes (E205's ||dW|| / ||W0||) and the saved values to repeat."""
        if item == "do_nothing":
            return P.clone(self.M), self.read(self.d196.read_half(self.cfg.control_task)), LAST, {}
        if item == "x10":
            F = P.rescaled(self.M, self.cfg.e195_layer, list(self.cfg.e195_C), self.cfg.e195_factor)
            return F, torch.load(P.COPY24, weights_only=False)["heldout"].tokens.cuda(), ALL, {}
        if item in self.pairs.inv["states"]:  # an E187 state, loaded as E205 loads it
            spec = self.pairs.inv["states"][item]
            dw = OT.load_dw(self.net, REPO / spec["file"])
            saved = json.loads((E205.ITEMS / f"{fname(item)}.json").read_text())
            rec = {"size": E205.sizes(dw, self.net.W0), "size_saved": self.pairs.e187_size(item), "kl_fm_saved": saved["kl_FM"], "saved_from": "E205"}
            return R.Model(OT.with_delta(self.net, dw).net), self.read(self.read_half187(spec["task"])), LAST, rec
        Fnet = E196.fine_tune(self.net, item)  # an E196 member no E187 state shares
        pF, pM = dict(Fnet.net.named_parameters()), dict(self.net.net.named_parameters())
        dw = {f"{k}.weight": pF[f"{k}.weight"] - pM[f"{k}.weight"] for k in self.net.W0}
        rec = {"size": E205.sizes(dw, self.net.W0), "kl_fm_saved": json.loads((E208_ITEMS / f"{item}.json").read_text())["replace"]["kl"], "saved_from": "E208"}
        return R.Model(Fnet.net), self.read(self.d196.read_half(item.rsplit("_s", 1)[0])), LAST, rec


def variants(ctx: K.Ctx, cfg: Config, th: dict[str, Any], exhaustive: bool, orders: int) -> dict[str, K.Variant]:
    """main (alpha, all of D); at the exhaustive levels also the other alphas, the restricted D and two halves (free: one table)."""
    N, every = len(ctx.kl_none), torch.arange(len(ctx.kl_none))
    out = {"main": K.Variant(every, cfg.alpha, 1 if exhaustive else orders)}
    if not exhaustive:
        return out
    out |= {f"alpha{a:g}": K.Variant(every, a, 1) for a in cfg.alphas if a != cfg.alpha}
    if float(ctx.kl_fm.max()) > th["kl_floor"]:  # not defined for do-nothing
        out["restricted"] = K.Variant(ctx.kl_fm.argsort(descending=True)[: int(N * cfg.restricted_share)], cfg.alpha, 1)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(cfg.split_seed))
    return out | {"half1": K.Variant(perm[: N // 2], cfg.alpha, 1), "half2": K.Variant(perm[N // 2 :], cfg.alpha, 1)}


def load_checks(item: str, base: dict[str, Any], th: dict[str, Any]) -> list[str]:
    """KL(F || M) on D against the saved reading (relative), and an E187 state's size against E187's."""
    miss = []
    if "kl_fm_saved" in base and not abs(base["kl_fm"] / base["kl_fm_saved"] - 1) <= th["kl_repro_rel"]:
        miss.append(f"KL(F || M) {base['kl_fm']:.9g}, {base['saved_from']}'s {base['kl_fm_saved']:.9g}")
    if "size_saved" in base and not abs(base["size"]["whole"] / base["size_saved"] - 1) <= th["size_rel"]:
        miss.append(f"size {base['size']['whole']:.9g}, E187's {base['size_saved']:.9g}")
    return miss


def read_item(w: World, item: str, levels: list[str], out: Path, th: dict[str, Any], orders: int) -> dict[str, dict[str, Any]]:
    """Every level of `levels` not written under `out`; returns each level's record (read back when written before)."""
    done = {lv: json.loads((out / f"{fname(item)}_{lv}.json").read_text()) for lv in levels if (out / f"{fname(item)}_{lv}.json").is_file()}
    todo = [lv for lv in levels if lv not in done]
    if not todo:
        log(f"{item}: {levels} done before, skipped")
        return done
    F, toks, at, rec = w.make(item)
    p = rho.Pair(F, w.M)
    ctx = K.prepare(p, toks, at, None, w.M.logits)
    base = {"experiment": EXPERIMENT, "item": item, "n_windows": len(toks), "read_positions": "last" if at == LAST else "all",
            "kl_none": float(ctx.kl_none.mean()), "kl_fm": float(ctx.kl_fm.mean())} | rec  # fmt: skip
    log(f"{item}: {len(toks)} windows, KL(F || F_empty) {base['kl_none']:.6g}, KL(F || M) {base['kl_fm']:.9g}, saved {rec.get('kl_fm_saved')}, size {rec.get('size')}")
    out.mkdir(parents=True, exist_ok=True)
    if miss := load_checks(item, base, th):
        stop(out / f"{fname(item)}.stopped.json", base, f"{item}: load check missed: " + "; ".join(miss))
    acc = K.grades(p, toks, at, None)
    for name in todo:
        lv = K.level(p, name, False)
        C, lacked = K.changed_vars(p, lv)
        s = K.level_grades(acc, lv, lacked)
        rd = base | {"level": name, "n_vars": len(lv.vars), "n_changed": len(C), "gates": K.gates(ctx, lv, F.logits),
                     "s": {K.vname(u): s[:, j].tolist() for j, u in enumerate(lv.vars)}, "changed": [K.vname(lv.vars[j]) for j in sorted(C)]}  # fmt: skip
        log(f"{item} {name}: {len(lv.vars)} variables, {len(C)} changed; gates {rd['gates']} (tol {th['gate_tol']})")
        if not max(rd["gates"].values()) <= th["gate_tol"]:
            stop(out / f"{fname(item)}_{name}.stopped.json", rd, f"{item} {name}: gate over {th['gate_tol']}")
        if not base["kl_none"] > th["kl_floor"]:
            done[name] = rd | {"reading": "no behaviour"}
            dump(out / f"{fname(item)}_{name}.json", done[name])
            continue
        res, _ = K.search(ctx, lv, s, C, lacked, variants(ctx, w.cfg, th, name in K.EXHAUSTIVE, orders), asdict(w.cfg), log, lambda _s: None)
        done[name] = rd | {"reading": "read", "searches": res}
        dump(out / f"{fname(item)}_{name}.json", done[name])
        m = res["main"]
        log(f"{item} {name}: R read {[r['R_read'] for r in m]}, all {[r['R_all'] for r in m]}, f {[r['f'] for r in m]}, |K*| {[r['size'] for r in m]}; "
            f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    del ctx, F, p
    torch.cuda.empty_cache()
    return done


# ---- stage 0 ----------------------------------------------------------------------------------------------------------


def stage0(w: World, th: dict[str, Any], tag: str) -> None:
    path, levels = RESULTS / f"stage0_{tag}.json", list(w.cfg.levels)
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    out: dict[str, Any] = {"experiment": EXPERIMENT}
    miss: list[str] = []
    for lv, it in read_item(w, "do_nothing", levels, RESULTS / f"stage0_{tag}", th, 1).items():
        rs = it["searches"]["main"]
        dev = max(abs(1 - v) for sv in it["s"].values() for v in sv)
        out[f"do_nothing_{lv}"] = {"max_dev_s": dev, "f": [r["f"] for r in rs], "R": [r[f"R_{g}"] for r in rs for g in ("read", "all")]}
        if not (dev <= th["s_tol"] and all(r["f"] == 0 for r in rs) and all(abs(x - 1) <= th["s_tol"] for x in out[f"do_nothing_{lv}"]["R"])):
            miss.append(f"do-nothing at {lv}: {out[f'do_nothing_{lv}']}")
    for lv, it in read_item(w, "x10", list(K.EXHAUSTIVE), RESULTS / f"stage0_{tag}", th, 1).items():
        mine, ref = it["searches"]["main"][0], json.loads((E214_ITEMS / f"x10_{lv}.json").read_text())["searches"]["main"][0]
        same_K = [u["name"] for u in mine["K"]] == [u["name"] for u in ref["K"]]
        out[f"x10_{lv}"] = {"same_K": same_K, "R_read": mine["R_read"], "R_read_E214": ref["R_read"]}
        if not (same_K and abs(mine["R_read"] - ref["R_read"]) <= th["repro_tol"]):
            miss.append(f"x10 at {lv}: {out[f'x10_{lv}']}")
    log(f"stage 0: {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def main(chosen: list[str] | None, shard: tuple[int, int], tag: str, out: str) -> int:
    cfg, th = load_config()
    every = items(cfg)
    todo = chosen or every[shard[0] :: shard[1]]
    assert set(todo) <= set(every), f"unknown items {sorted(set(todo) - set(every))}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E214
    log(f"{EXPERIMENT} start, tag {tag}; {len(todo)} of {len(every)} items {todo}; settings {asdict(cfg)}; thresholds {th}")
    w = World(cfg)
    stage0(w, th, tag)
    for k, item in enumerate(todo):
        read_item(w, item, list(cfg.levels), RESULTS / out, th, cfg.orders)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=None, help="items by name (default: this shard of every item, in pair order)")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this pod's name: stage 0 writes results/stage0_<tag>.json")
    ap.add_argument("--out", default="items", help="results subdirectory for the item files (the rehearsal writes elsewhere)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.items, (args.shard[0], args.shard[1]), args.tag, args.out)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

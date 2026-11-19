"""E214 (cell P1.35, test T2): R, the mean squared-cosine grade s_u over the minimal circuit K* under mean ablation, read
at four levels on items whose label is known by construction (plan reports/plans/2026-11-18_plan_p1-35-t2.md, appendix
..._appendix.md; filed settings, thresholds and rules in results/predictions.json, which the run refuses unless
committed unchanged). The measurement is commute.kstar; P1.31's edits are commute.known_edits. Nothing is trained.

Items (each read on its own windows D; KL at the read positions):
    do_nothing, reuse_small, reuse_large, new_few, new_thin   P1.31's cue windows, last position
    x2, x10, moved, rebuilt                                   E195's copy24 held-out windows, every position
    e201, last_lowest, last_rank5, last_rank9, last_rank13, neurons   E206's read windows, every position
    always_on, cue, xor, table                                E137's read half (first piece_windows), last position
Per item and level, results/items/<item>_<level>.json (and, at neuron_head, results/large/<item>_<level>.npz with s and
each K*) is written when it ends; a rerun skips what is written. Both grade readings are always computed and saved:
"all" (every position of D, both runs) and "read" (the read positions only). The four edits and e201 get 5 orders plus
alpha 0.8, 0.95, the restricted D and two halves at the greedy levels; every item gets them at the exhaustive levels.
Stage 0 runs first and alone (results/stage0.json, skipped once done; exit 1 on a miss): E62's induction top-1,
do-nothing at every level run, x10's C graded at neuron_head (no search), e201 at head_mlp, the four pieces' one-line forms.

    python experiments/kstar_levels/run.py --levels block mlp_attn --items reuse_small reuse_large new_few new_thin
    python experiments/kstar_levels/run.py --levels block mlp_attn head_mlp          (every item)
    (neuron_head was dropped by the user on 2026-11-19: the run refuses it while predictions.json "stages_run" says so)
    (each 2>&1 | tee -a experiments/kstar_levels/results/run.log; on a pod PRED_SHA=<sha> python ...)
    python experiments/kstar_levels/score.py   (no GPU: bands and the filed rules from the item files present)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import graft as G
from commute import known_edits as KE
from commute import kstar as K
from commute import rho
from commute import rule_retention as R
from commute.data import pile_rows
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)
from commute.variable_read import cut

P: Any = load_experiment("rule_retention", "run")  # E193: clone, rescaled, COPY24, log, dump, stop
E195: Any = load_experiment("rule_retention_fixed", "run")  # moved
E208: Any = load_experiment("rho_saved_items", "run")  # rebuilt, graft_windows
PC: Any = load_experiment("reuse_variable_test", "pieces")  # E137: Config, make_sets, make_task, pair_windows, Pieces, piece_edit, xor_f
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, ITEMS, LARGE = RESULTS / "predictions.json", RESULTS / "items", RESULTS / "large"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
DROPPED = ("neuron_head",) if "neuron_head dropped" in json.loads(PRED.read_text()).get("stages_run", "") else ()  # by the user, 2026-11-19
RUN_LEVELS = [lv for lv in K.LEVELS if lv not in DROPPED]
E62 = EXPERIMENTS / "induction_cue" / "results"
log, dump, stop = P.log, P.dump, P.stop
EDITS = ("reuse_small", "reuse_large", "new_few", "new_thin")
HEADS = ("e201", "last_lowest", "last_rank5", "last_rank9", "last_rank13")
PIECES = {"always_on": "Always on", "cue": "Cue", "xor": "XOR", "table": "Table"}
FULL = (*EDITS, "e201")  # every variant at every level
LAST, ALL = slice(-1, None), slice(None)


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    levels: tuple[str, ...]
    alpha: float
    alphas: tuple[float, ...]
    restricted_share: float
    orders: int
    order_seed: int
    chunk: int
    polish_cap: int
    split_seed: int
    eval_batch: int  # Pile held-out windows for the slots and detectors
    slot_layer: int
    n_cues: int
    windows_per_cue: int
    window: int
    cue_seed: int
    data_seed: int
    circuit_heads: tuple[tuple[int, int], ...]  # documentation: E62's circuit the reuse edits re-trigger
    gain_head: tuple[int, int]
    detector_ratio: float
    detector_quantile: float
    gamma_t_max: float
    copy_bar_frac: float
    lookup_bar: float
    g_fallback_ratio: float
    e195_layer: int
    e195_C: tuple[int, ...]
    e195_factors: tuple[float, ...]
    rebuilt: str
    e201_graft: str
    e206_grafts: tuple[str, ...]
    graft_layer: int
    graft_window: int
    graft_val_rows: int
    graft_n_read: int
    graft_read_seed: int
    pieces: str
    piece_names: tuple[str, ...]
    piece_windows: int

    def __post_init__(self) -> None:
        assert self.levels == K.LEVELS and list(self.alphas) == sorted(self.alphas) and self.alpha in self.alphas and self.n_cues == 8
        assert set(self.e206_grafts) == {*HEADS[1:], "neurons"} and tuple(PIECES.values()) == self.piece_names == PC.PAIRS
        assert self.windows_per_cue * self.n_cues == 512 and 1 not in self.e195_factors and 0 < self.restricted_share < 1

    @property
    def items(self) -> tuple[str, ...]:
        return ("do_nothing", *EDITS, *(f"x{f:g}" for f in self.e195_factors), "moved", "rebuilt", *HEADS, "neurons", *PIECES)


Made = tuple[R.Model, Tensor, slice, K.Piece | None, Callable[[Tensor], Tensor]]  # F, D, read positions, piece, native logits


class World:
    """M and every item's builder, each input loaded once when first needed."""

    def __init__(self, cfg: Config, th: dict[str, Any]) -> None:
        from commute.steering import Net

        self.cfg, self.th, self.net = cfg, th, Net("cuda")
        self.net.net.requires_grad_(False)
        self.M = R.Model(self.net.net)

    @cached_property
    def cues(self) -> KE.Cues:
        from commute import keydep as KD

        ctx: Any = SimpleNamespace(net=self.net)  # KD.vocab reads ctx.net only
        rows, _, _, ranked, tok = KD.vocab(ctx)
        lo, hi = json.loads((E62 / "predictions.json").read_text())["settings"]["windows"]["pool_ranks"]
        pool = [t for t in ranked[lo:hi] if tok.encode(tok.decode([t])) == [t]]
        keys = json.loads((E62 / "run.json").read_text())["pools"]["keys"]
        c = self.cfg
        return KE.cues(rows[torch.arange(rows.shape[0]) % 5 == 0], keys, pool, c.n_cues, c.windows_per_cue, c.window, c.cue_seed)

    @cached_property
    def edits(self) -> dict[str, R.Model]:
        pile = cut(pile_rows("val", 0, 256, log=log), self.cfg.eval_batch, self.cfg.window, torch.Generator().manual_seed(self.cfg.data_seed))
        edits, rec = KE.build(self.M, self.cues, pile, SimpleNamespace(**asdict(self.cfg), a_prime_copy_max=self.th["a_prime_copy_max"]))
        c = self.cues
        rec |= {"A": c.A, "A_prime": c.A2, "y": c.y}
        log(f"edits: {json.dumps(rec)}")
        if rec["misses"]:
            stop(RESULTS / "edits.json", rec, "build checks: " + "; ".join(rec["misses"]))
        dump(RESULTS / "edits.json", rec | {"experiment": EXPERIMENT})
        return edits

    @cached_property
    def e137(self) -> tuple[Any, Any, Any]:
        from commute import vpd

        pc, tok = PC.Config(), vpd.tokenizer()
        sets = PC.make_sets(pc, tok, log)
        task = PC.make_task(sets, tok, int(self.net.net.lm_head.weight.shape[0]), pc)
        st = torch.load(EXPERIMENTS.parent / self.cfg.pieces, weights_only=False, map_location="cuda")
        return pc, sets, PC.Pieces(st["Q"], st["b"], st["w"], st["a"], st["C"], st["c"], task)

    def piece(self, pair: str) -> K.Piece:
        """E137's piece as one line at block 2's input, last position (pieces.piece_edit's four forms)."""
        pc = self.e137[2]
        Qb = pc.Q @ pc.b
        forms: dict[str, Callable[[list[Tensor], Tensor], Tensor]] = {
            "Always on": lambda xs, t: Qb.expand(len(t), -1),
            "Cue": lambda xs, t: (xs[1][:, -1] @ pc.w + pc.a)[:, None] * Qb,
            "XOR": lambda xs, t: PC.xor_f(pc, t).to(Qb.device)[:, None] * Qb,
            "Table": lambda xs, t: xs[0][:, -1] @ pc.C.T + pc.c}  # fmt: skip
        f = forms[pair]
        return lambda xs, t: torch.cat([torch.zeros(len(t), t.shape[1] - 1, Qb.shape[0], device=t.device), f(xs, t)[:, None]], 1)

    def grafted(self, item: str) -> R.Model:
        path = EXPERIMENTS.parent / self.cfg.e201_graft if item == "e201" else EXPERIMENTS / "graft_sites" / "results" / "large" / f"{item}.pt"
        raw = json.loads((EXPERIMENTS / "graft_sites" / "results" / "predictions.json").read_text())["settings"]["recipe"]
        fields: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
        rec = G.Recipe(**fields)
        F, n = P.clone(self.M), torch.load(path, weights_only=True).get("w_in", torch.empty(0)).shape[0]
        g = G.attach_neurons(F, self.cfg.graft_layer, n, 0, rec) if item == "neurons" else G.attach_head(F, self.cfg.graft_layer, 0, rec)
        G.load(g, path)
        return F

    def make(self, item: str) -> Made:
        c, M = self.cfg, self.M
        if item == "do_nothing" or item in EDITS:
            F = P.clone(M) if item == "do_nothing" else self.edits[item]
            return F, self.cues.toks.cuda(), LAST, None, F.logits
        if item in PIECES:
            pc, sets, pieces = self.e137
            toks = PC.pair_windows(pc, PIECES[item], sets, pieces.task)[1][: c.piece_windows].cuda()
            return M, toks, LAST, self.piece(PIECES[item]), lambda t: self.net.run(t, PC.piece_edit(self.net, pieces, PIECES[item], t))[0]
        if item in (*HEADS, "neurons"):
            F = self.grafted(item)
            return F, E208.graft_windows(c), ALL, None, F.logits
        old = SimpleNamespace(layer=c.e195_layer, C=c.e195_C)
        F = E208.rebuilt(M, old) if item == "rebuilt" else E195.moved(M, c.e195_layer, list(c.e195_C)) if item == "moved" else P.rescaled(M, c.e195_layer, list(c.e195_C), float(item[1:]))
        return F, torch.load(P.COPY24, weights_only=False)["heldout"].tokens.cuda(), ALL, None, F.logits


def variants(item: str, ctx: K.Ctx, cfg: Config, th: dict[str, Any], exhaustive: bool) -> dict[str, K.Variant]:
    N, every = len(ctx.kl_none), torch.arange(len(ctx.kl_none))
    out = {"main": K.Variant(every, cfg.alpha, cfg.orders if item in FULL and not exhaustive else 1)}
    if not (exhaustive or item in FULL):
        return out
    out |= {f"alpha{a:g}": K.Variant(every, a, 1) for a in cfg.alphas if a != cfg.alpha}
    if float(ctx.kl_fm.max()) > th["kl_floor"]:  # not defined for do-nothing
        out["restricted"] = K.Variant(ctx.kl_fm.argsort(descending=True)[: int(N * cfg.restricted_share)], cfg.alpha, 1)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(cfg.split_seed))
    return out | {"half1": K.Variant(perm[: N // 2], cfg.alpha, 1), "half2": K.Variant(perm[N // 2 :], cfg.alpha, 1)}


class Clock:
    """The neuron_head time stop: the first search's seconds times the searches planned, against l1_hours_cap."""

    def __init__(self, planned: int, cap_h: float) -> None:
        self.planned, self.cap_h, self.first = planned, cap_h, True

    def __call__(self, seconds: float) -> None:
        if self.first:
            self.first, hours = False, seconds * self.planned / 3600
            log(f"neuron_head: first search {seconds:.0f} s; {self.planned} searches planned, about {hours:.2f} h (cap {self.cap_h} h)")
            if hours > self.cap_h:
                stop(RESULTS / "l1_stop.json", {"seconds_first": seconds, "planned": self.planned, "hours": hours}, "neuron_head projected over the cap")


def read_item(w: World, item: str, levels: list[str], clock: Clock) -> None:
    cfg, th = w.cfg, w.th
    todo = [lv for lv in levels if not (ITEMS / f"{item}_{lv}.json").is_file()]
    if not todo:
        log(f"{item}: {levels} done before, skipped")
        return
    F, toks, at, piece, native = w.make(item)
    p = rho.Pair(F, w.M)
    ctx = K.prepare(p, toks, at, piece, w.M.logits)
    base = {"experiment": EXPERIMENT, "item": item, "n_windows": len(toks), "kl_none": float(ctx.kl_none.mean()), "kl_fm": float(ctx.kl_fm.mean())}
    log(f"{item}: {len(toks)} windows, KL(F || F_empty) {base['kl_none']:.6g}, KL(F || M) {base['kl_fm']:.6g}")
    if item in EDITS and not min(base["kl_none"], base["kl_fm"]) > th["kl_floor"]:
        stop(ITEMS / f"{item}.stopped.json", base, f"{item}: KL at or under kl_floor")
    acc = K.grades(p, toks, at, piece)
    ITEMS.mkdir(parents=True, exist_ok=True)
    for name in todo:
        lv = K.level(p, name, piece is not None)
        C, lacked = K.changed_vars(p, lv)
        s = K.level_grades(acc, lv, lacked)
        kept = [j for j in range(len(lv.vars)) if j not in lacked]
        out = base | {"level": name, "n_vars": len(lv.vars), "n_changed": len(C), "changed": [K.vname(lv.vars[j]) for j in sorted(C)][:64],
                      "max_dev_s": {g: float((1 - s[r, kept]).abs().max()) for r, g in enumerate(("all", "read"))}, "gates": K.gates(ctx, lv, native)}  # fmt: skip
        out["s_changed"] = {K.vname(lv.vars[j]): s[:, j].tolist() for j in sorted(C)[:64]}
        tol = th["piece_gate_tol"] if piece is not None else th["gate_tol"]
        log(f"{item} {name}: {len(lv.vars)} variables, {len(C)} changed; gates {out['gates']} (tol {tol}); max |1 - s| {out['max_dev_s']}")
        if not max(out["gates"].values()) <= tol:
            stop(ITEMS / f"{item}_{name}.stopped.json", out, f"{item} {name}: gate over {tol}")
        if not base["kl_none"] > th["kl_floor"]:
            dump(ITEMS / f"{item}_{name}.json", out | {"reading": "no behaviour"})
            continue
        vs = variants(item, ctx, cfg, th, name in K.EXHAUSTIVE)
        res, arrays = K.search(ctx, lv, s, C, lacked, vs, asdict(cfg), log, clock if name == "neuron_head" else lambda _s: None)
        if name == "neuron_head":
            LARGE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(LARGE / f"{item}_{name}.npz", s=s.numpy(), changed=np.array(sorted(C)), names=np.array([K.vname(u) for u in lv.vars]), **arrays)
        else:
            out["s"] = {K.vname(u): s[:, j].tolist() for j, u in enumerate(lv.vars)}
        dump(ITEMS / f"{item}_{name}.json", out | {"reading": "read", "searches": res})
        m = res["main"]
        log(f"{item} {name}: R all {[r['R_all'] for r in m]}, read {[r['R_read'] for r in m]}, f {[r['f'] for r in m]}, |K*| {[r['size'] for r in m]}")
    del ctx
    torch.cuda.empty_cache()


# ---- stage 0 ----------------------------------------------------------------------------------------------------------


def stage0(w: World) -> None:
    path, cfg, th = RESULTS / "stage0.json", w.cfg, w.th
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log("stage 0: done before")
        return
    from commute import copying as CP

    out: dict[str, Any] = {"experiment": EXPERIMENT, "induction": CP.induction(w.net, json.loads((E62 / "predictions.json").read_text())["settings"]["induction"])}
    miss = [f"induction top-1 {out['induction']['top1_second']:.4f}"] if not abs(out["induction"]["top1_second"] - th["e62_base"]) <= th["e62_tol"] else []
    read_item(w, "do_nothing", RUN_LEVELS, Clock(1, th["l1_hours_cap"]))
    for lv in RUN_LEVELS:
        it = json.loads((ITEMS / f"do_nothing_{lv}.json").read_text())
        rs = [r for v in it["searches"].values() for r in v]
        out[f"do_nothing_{lv}"] = {"max_dev_s": it["max_dev_s"], "f": max(r["f"] for r in rs), "R": [r["R_all"] for r in rs] + [r["R_read"] for r in rs]}
        if not (max(it["max_dev_s"].values()) <= th["s_tol"] and max(r["f"] for r in rs) == 0 and all(abs(x - 1) <= th["s_tol"] for x in out[f"do_nothing_{lv}"]["R"])):
            miss.append(f"do-nothing at {lv}: {out[f'do_nothing_{lv}']}")
    F, toks, at, _, _ = w.make("x10")
    a = K.grades(rho.Pair(F, w.M), toks, at, None)[("neuron", cfg.e195_layer)][:, list(cfg.e195_C)].cpu()  # sums of 1 - cos^2, counts
    dev = torch.cat([a[0] / a[1], a[2] / a[3]])  # 1 - s, both readings
    out["x10_C_1_minus_s"] = dev.tolist()
    miss += [f"x10 C 1 - s {dev.tolist()}"] if not float(dev.abs().max()) <= th["s_tol"] else []
    read_item(w, "e201", ["head_mlp"], Clock(1, th["l1_hours_cap"]))
    for r in json.loads((ITEMS / "e201_head_mlp.json").read_text())["searches"]["main"]:
        if not (r["lacked_in_K"] >= 1 and r["rho_all"] == 0 and abs(r["R_all"] - (1 - r["f"])) <= th["identity_tol"]):
            miss.append(f"e201 at head_mlp: graft in K* {r['lacked_in_K']}, rho' {r['rho_all']}, R {r['R_all']}, f {r['f']}")
    for item, pair in PIECES.items():
        _, toks, _, piece, native = w.make(item)
        t = toks[: cfg.eval_batch]
        with torch.no_grad():
            f = rho.Pair(w.M, w.M).f
            mine, ref = f.logits(K.run(f, t, piece).out[:, -1]).float(), native(t)[:, -1].float()
        out[f"piece_gate_{item}"] = float(((mine - ref).abs().amax(1) / ref.abs().amax(1)).max())
        miss += [f"{pair} one-line form off by {out[f'piece_gate_{item}']:.3g}"] if not out[f"piece_gate_{item}"] <= th["piece_gate_tol"] else []
    log(f"stage 0: {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def main(levels: list[str], chosen: list[str] | None) -> int:
    pred = json.loads(PRED.read_text())
    settings = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    cfg, th = config_from_settings(Config, settings), pred["thresholds"]
    items = chosen or list(cfg.items)
    assert set(levels) <= set(RUN_LEVELS) and set(items) <= set(cfg.items), f"unknown levels or items: {levels} {items}"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout
    log(f"{EXPERIMENT} start; levels {levels}, items {items}; settings {asdict(cfg)}; thresholds {th}")
    w = World(cfg, th)
    stage0(w)
    planned = sum(cfg.orders + len(cfg.alphas) + 2 if i in FULL else 1 for i in items if not (ITEMS / f"{i}_neuron_head.json").is_file()) if "neuron_head" in levels else 0
    clock = Clock(planned, th["l1_hours_cap"])
    for item in items:
        read_item(w, item, levels, clock)
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=RUN_LEVELS)
    ap.add_argument("--items", nargs="+", default=None, help="default: every item, in the order of Config.items")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.levels, args.items)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)

"""E309 (cell P2.79): writer 2b's readings through the NEW functions (moves 5-7: steering, E_b and the needed set K, ADL
column D), keyed exactly as old.py keys them, to results/new_b.json, for compare.py's reproduction table. Every value
below is a call of `fair_steering.steering_share`, `eb.write_sums` / `eb.eb`, `needed.scan_sites` or `adl.column_D` on
rows from `windows.read_rows` and updates from `items.load_dw`; no old experiment's reading function is called.

The fresh write passes and the neuron K scan read F itself: the built item x2 (E214's rescale) and E229's trained item
copy24_s0 are built by `rescaled` / `e229_f` (below) on `World`, with their D (`copy24_held`, `e229_read`)
and E217's neuron bounds (`neuron_bounds`); no `load_experiment` (P2.79 2c).

  --part data  CPU (`fleet/cpu-run`): E_b from the saved sums (VPD, neurons, SAE)
  --part gpu   one L40 (new_b.sbatch): steering E135 and E143, the panel's steering and ADL D, the fresh E_b and the K scan
  --dry        with --part gpu: build every input that needs no model (rows, updates, settings) and stop before the model

    ~/commute-agents/fleet/cpu-run --mem 16G python3 experiments/reading_refactor/new_b.py --part data
    sbatch --comment=E309 experiments/reading_refactor/new_b.sbatch
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from old import Out as OldOut
from torch import Tensor

from commute import adl as ADL
from commute import eb as EB
from commute import fair_steering as FS
from commute import items as IT
from commute import masked_reads as MR
from commute import old_tasks as OT
from commute import rule_retention as R
from commute import share as SH
from commute import steering as ST
from commute import variable_read as VR
from commute import vpd
from commute.data import cached_rows
from commute.jobs import EXPERIMENTS, hard_exit
from commute.local_faith import MIN_GAIN
from commute.windows import Rows, file_sha256, read_rows

RESULTS, REPO = EXPERIMENTS / "reading_refactor" / "results", EXPERIMENTS.parent
LOWER_128 = "experiments/convergence_panel/results/large/states/lower_s0_step128.pt"
LO = ST.logodds_at  # the share's unit, every reading here is scored in it
E229_AT = slice(-1, None)  # E217's and E229's read position: the last (topk_neurons LAST)


# ----------------------------------------------------------------------------- items built with their F, M and D
# (moved from items.py: this file is their only caller).
# The fresh write passes and the neuron K scan need F itself, not a saved delta-W: E214's built x<f> (a
# rescale of M) and E229's trained items (M plus an old task's delta-W), with their D and E217's neuron bounds. Copied
# from the finished run.py files that built them (kstar_levels `World.make`, rule_retention `rescaled` / COPY24,
# topk_neurons `setup` / `windows` / `pair`, rule_retention_pairs `Data.read_half` / `subset` / `fine_tune`, topk_upper
# `load_pct` / `feat_of`), which stay as they ran. Every setting is an argument; callers take them from those files'
# predictions.json "settings".

COPY24 = EXPERIMENTS / "open_pair" / "results" / "large" / "amend1" / "windows_copy_trained.pt"  # E195's copy24 windows
E217_PCT = EXPERIMENTS / "topk_neurons" / "results" / "large" / "percentiles.pt"


def rescaled(M: R.Model, layer: int, C: Sequence[int], factor: float) -> R.Model:
    """E214's x<factor>: a copy of M with block `layer`'s down_proj columns C times `factor`."""
    F = R.Model(copy.deepcopy(M.net).requires_grad_(False))
    with torch.no_grad():
        F.net.h[layer].mlp.down_proj.weight[:, list(C)] *= factor
    return F


def copy24_held() -> Tensor:
    """E214's D of x<f>: E195's copy24 held-out windows (CPU), read at every position."""
    return torch.load(COPY24, weights_only=False)["heldout"].tokens


def e229_read(task: str, read_seed: int, n_read: int) -> Tensor:
    """E229's D of an old task's items: n_read windows of the task's read half (`old_tasks.task_windows`), drawn by
    read_seed (CPU), read at the last position."""
    sizes = OT.DSizes()
    read = OT.task_windows(OT.TASKS[task], VR.make_sets(sizes, vpd.tokenizer(), print), sizes)[1]
    return read[torch.randperm(read.shape[0], generator=torch.Generator().manual_seed(read_seed))[:n_read]]


def e229_f(net: Any, item: str) -> R.Model:
    """E229's F of `<task>_s<seed>`: M (net, a `steering.Net`) plus the old task's saved delta-W (`old_tasks.old_f`)."""
    task, seed = item.rsplit("_s", 1)
    return R.Model(OT.old_f(net, OT.TASKS[task].states[int(seed)]).net)


def neuron_bounds(q: float, qs: Sequence[float], sha256: str) -> Tensor:
    """E217's per-neuron q-quantile of M's MLP activations over Pile rows, (layers, n) CPU: E229's upper bounds (an
    activation above it is a feature's value). The saved file is checked against `sha256` and its quantiles against qs."""
    if (got := file_sha256(E217_PCT)) != sha256:
        raise SystemExit(f"STOP: {E217_PCT}: sha256 {got}, the filed {sha256}")
    z = torch.load(E217_PCT, weights_only=True)
    assert torch.equal(z["q"], torch.tensor(list(qs))), "the saved quantiles are not the filed ones"
    return z["values"][list(qs).index(q)]


class Out(OldOut):
    """old.py's writer (none of old.py's readings is called), to results/new_b.json."""

    def __init__(self) -> None:
        self.path = RESULTS / "new_b.json"
        self.d: dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {}
        self.errors: list[str] = []


def val(o: OldOut, key: str, value: Any, **settings: Any) -> None:
    o.put(key, value, None, **settings)  # no saved value here: compare.py takes it from old_A / old_B


def settings(exp: str) -> dict[str, Any]:
    P = json.loads((EXPERIMENTS / exp / "results" / "predictions.json").read_text())
    return P["settings"] | {"thresholds": P.get("thresholds", {})}  # kstar_vpd, topk_upper and needed_neurons file none


def npz(path: str) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in np.load(REPO / path).items()}


def take(r: Rows, idx: slice) -> Rows:
    assert r.tgt is not None
    return replace(r, toks=r.toks[idx], tgt=r.tgt[idx])


# ----------------------------------------------------------------------------- data part (CPU)


def saved(sums: dict[str, torch.Tensor], changed: torch.Tensor) -> EB.Sums:
    return EB.Sums(sums["num"], sums["den"], sums["dd"], changed.bool(), None)


def eb_saved_sums(o: OldOut) -> None:
    for item in ("x2", "new_few"):  # E268: E247's sums, E242's changed set, E268's K
        s = saved(npz(f"experiments/thin_edit_diagnosis/results/large/items_{item}.npz"), npz(f"experiments/kstar_vpd_wording_a/results/large/items_{item}.npz")["changed"])
        val(o,f"E_b_vpd/{item}", EB.eb(s, npz(f"experiments/kstar_needed/results/large/items_{item}.npz")["in_K"], leftovers=True), sums="E247 npz", K="E268 in_K")
    z = npz("experiments/rb_states/results/large/items_e58-copy_s0.npz")  # E275: E257's sums and changed, E275's K
    K = npz("experiments/claim2_needed/results/large/items_e58-copy_s0.npz")["in_K"]
    val(o,"E_b_vpd/e58/copy_s0", EB.eb(saved(z, z["changed"]), K, leftovers=True), sums="E257 npz", K="E275 in_K")
    for item in ("x2", "copy24_s0"):  # E277: its own sums, changed and K
        z = npz(f"experiments/needed_neurons/results/large/items_{item}.npz")
        val(o,f"E_b_neurons/{item}", EB.eb(saved(z, z["changed"]), z["in_K"], leftovers=True), sums="E277 npz", K="E277 in_K")
    z, k = npz("experiments/sae_eb/results/large/items_e58-copy_s0_scaled.npz"), npz("experiments/sae_r_a/results/large/items_e58-copy_s0_scaled.npz")
    val(o,"E_b_sae/e58/copy_s0_scaled", EB.eb(saved(z, k["changed"]), k["main_in_K_polished"], leftovers=True), sums="E263 npz", K="E245 K* (main_in_K_polished)")


# ----------------------------------------------------------------------------- gpu part: inputs (CPU, no model)


def steering_inputs() -> dict[str, Any]:
    """Rows, updates and rules of E135's copy_s0, E143's e46/ident32_s0 and E272's lower_s0_step128."""
    s135, s143, cp = settings("steering_copy_lookup"), settings("holes_fill"), settings("convergence_panel")
    e62 = read_rows("e62/copy", "held")
    held46, train46 = read_rows("e46/ident32", "held"), read_rows("e46/ident32", "train")
    lower, lower_fit = read_rows("e53/lowercase", "held"), read_rows("e53/lowercase", "train")
    fit62 = take(e62, slice(0, None, 2))
    return {
        "e135": {"fit": fit62, "select": fit62, "read": take(e62, slice(1, None, 2)), "dW": IT.load_dw(IT.resolve("e62/copy_s0")),
                 "rule": FS.Rule("pooled", None, tuple(s135["strengths"]), None, len(fit62.toks), torch.float32)},
        "e143": {"fit": train46, "select": take(train46, slice(0, s143["n_select"])), "read": held46, "dW": IT.load_dw(IT.resolve("e46/ident32_s0")),
                 "rule": FS.Rule("pooled", None, tuple(s143["strengths"]), s143["thresholds"]["gain_floor"], FS.BATCH, torch.float64)},
        "panel": {"fit": lower_fit, "select": take(lower_fit, slice(0, cp["n_select"])), "read": lower, "dW": IT.load_dw(IT.Item("panel/lower_s0_step128", LOWER_128, None)),
                  "rule": FS.Rule("pooled", None, tuple(cp["strengths"]), MIN_GAIN, FS.BATCH, torch.float64),
                  "pile": cached_rows("val", 0, cp["adl_rows"])[:, : cp["adl_k"]], "adl_batch": cp["adl_batch"]},
    }  # fmt: skip


def fresh_inputs() -> dict[str, Any]:
    """The fresh items' D, E217's neuron bounds and the builders' settings (E214's x2, E229's copy24_s0, E277's scan),
    from the finished experiments' filed settings; CPU, no model."""
    lv, c229, nr = settings("kstar_levels"), settings("topk_upper"), settings("needed_neurons")
    return {"x2_D": copy24_held(), "copy_D": e229_read("copy24", c229["read_seed"], c229["n_read"]),
            "hi": neuron_bounds(c229["bands"][nr["band"]], c229["percentile_qs"], c229["percentiles_sha256"]),
            "layer": lv["e195_layer"], "C": lv["e195_C"], "split_tol": settings("kstar_vpd")["split_tol"],
            "gain_batch": settings("kstar_vpd_wording_a")["gain_batch"], "nr": nr}  # fmt: skip


# ----------------------------------------------------------------------------- gpu part: readings


def steering(net: ST.Net, o: OldOut, x: dict[str, Any]) -> None:
    for key, name, gain_of in (("e135", "e62/copy_s0", "target"), ("e143", "e46/ident32_s0", "target"), ("panel", "lower_s0_step128", "share")):
        a = x[key]
        gain = FS.target_gain(net, a["read"], a["dW"], LO, a["rule"].batch) if gain_of == "target" else share_gain(net, a["read"], a["dW"])
        rec = FS.steering_share(net, a["fit"], a["select"], a["read"], a["dW"], a["rule"], LO, gain, None)
        val(o,f"steering_pooled/{name}", rec["share"], rule=repr(a["rule"]), gain=gain, gain_from=gain_of, setting=rec["setting"])
        if key == "panel":
            adl = ADL.column_D(net, a["read"], a["dW"], a["pile"], LO, gain, MIN_GAIN, a["adl_batch"], FS.BATCH)
            val(o,"adl_D/lower_s0_step128", adl["D"], pile_rows=tuple(a["pile"].shape), gain=gain, floor=MIN_GAIN)


def share_gain(net: ST.Net, r: Rows, dW: ST.DW) -> float:
    """E272's denominator: the share's T cells (T1 - T0 of `share.share_cells`, batch 64), as `realistic_reads.share_of`."""
    return SH.cells_gain(SH.share_cells(net.model, net.W0, dW, r, {"lo": LO}, MR.BATCH), "lo")


def eb_fresh(o: OldOut, w: Any, x: dict[str, Any]) -> None:
    """Fresh write passes over each item's saved K: x2 at VPD and at neurons, copy24_s0 at neurons; copy24_s0's neuron K."""
    from commute import kstar as K
    from commute import kstar_vpd as KV
    from commute import needed as IX
    from commute import needed_neurons as NN
    from commute import rho
    from commute import stand_in as SI

    nr, feat_n, M = x["nr"], K.Feat(None, list(x["hi"].to("cuda"))), R.Model(w.net.net)
    F, toks = rescaled(M, x["layer"], x["C"], 2.0), x["x2_D"].cuda()
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases(x["split_tol"]))
    changed = torch.isin(torch.arange(feat.n), torch.tensor(sorted(feat.changed), dtype=torch.long))
    s = EB.write_sums(F.net, SI.vpd_sites(feat), toks, x["gain_batch"], changed)
    val(o,"E_b_vpd/x2@fresh", EB.eb(s, npz("experiments/kstar_needed/results/large/items_x2.npz")["in_K"], leftovers=True), batch=x["gain_batch"])
    p = rho.Pair(F, M)
    s = EB.write_sums(F.net, NN.sites(p, feat_n, toks.shape[1]), toks, nr["cache_batch"], NN.variables(p)[1])
    val(o,"E_b_neurons/x2@fresh", EB.eb(s, npz("experiments/needed_neurons/results/large/items_x2.npz")["in_K"], leftovers=True), batch=nr["cache_batch"])
    del F, p
    torch.cuda.empty_cache()
    F, toks, at = e229_f(w.net, "copy24_s0"), x["copy_D"].cuda(), E229_AT
    p = rho.Pair(F, M)
    z = npz("experiments/needed_neurons/results/large/items_copy24_s0.npz")
    s = EB.write_sums(F.net, NN.sites(p, feat_n, toks.shape[1]), toks, nr["cache_batch"], NN.variables(p)[1])
    val(o,"E_b_neurons/copy24_s0@fresh", EB.eb(s, z["in_K"], leftovers=True), batch=nr["cache_batch"])
    run = NN.cache(F.net, toks, at, nr["cache_batch"])
    m = min(nr["examples"], len(toks))
    ex, pt = IX.derangements(len(toks), nr["derangements"], m, nr["seed"])
    sites = [NN.Swap(run, l, feat_n.hi[l]) for l in range(len(F.net.h))]
    sc = IX.scan_sites(F.net, sites, run, at, ex, pt, IX.rows_per_forward(nr["rows"], nr["logit_rows"], run.top.shape[1]), len(F.net.h) * NN.PER_LAYER, print)
    Kn = sc.first_pair >= 0
    val(o,"needed_set_size/copy24_s0@neurons", int(Kn.sum()), derangements=nr["derangements"], examples=m, seed=nr["seed"], mask_equals_saved=bool(torch.equal(Kn, z["in_K"].bool())))


def main(part: str, dry: bool) -> int:
    o = Out()
    if part == "data":
        o.run("eb_saved_sums", eb_saved_sums)
    else:
        x = {"steering": steering_inputs(), "fresh": fresh_inputs()}
        f = x["fresh"]
        print({n: tuple(a["read"].toks.shape) for n, a in x["steering"].items()}, {k: tuple(f[k].shape) for k in ("x2_D", "copy_D", "hi")}, flush=True)
        if dry:
            print("NEW_B gpu inputs built (dry: no model)", flush=True)
            return 0
        from commute.world import World

        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        net = World.load("cuda").net
        o.run("steering_and_adl", lambda o: steering(net, o, x["steering"]))
        del net
        torch.cuda.empty_cache()
        w = World.load("cuda")  # a fresh model for the fresh items, as the old worlds loaded their own
        o.run("eb_fresh", lambda o: eb_fresh(o, w, x["fresh"]))
        print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"NEW_B {part} DONE; errors: {o.errors or 'none'}", flush=True)
    return 1 if o.errors else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("data", "gpu"), required=True)
    ap.add_argument("--dry", action="store_true", help="gpu part: build the inputs, stop before the model")
    a = ap.parse_args()
    hard_exit(main(a.part, a.dry))

"""neuron_vpd_count (E117): VPD's switched-on set and re-switching, repeated on the 67M's MLP neurons at VPD's count, on
the states whose class is known by construction (E12/E46 copy against random maps, E53 lowercase against two ciphers).
Plan: reports/plans/2026-11-06_plan_e117-neurons-at-vpd-count.md. Every definition, setting, control and rule:
results/predictions.json (filed before the run, committed; the run refuses otherwise). $0, saved states only.

Per state, at the read position (the trigger; E53's position 63), with K_VPD = |S_on|:
  S_on   alive subcomponents whose mean gate change, state minus W0, is above 0.3 (recruit.trigger_dg, recruit.switched)
  N_on   the K_VPD neurons with the largest mean firing change, state minus W0
  B      share of the gain removed when the set is put back at the read position in the fine-tuned model
         (VPD: recruitment.restore_hooks; neurons: W0's c_fc and GELU on the state's MLP input, neuron_r.half_acts)
  C      share of the gain carried when only the set takes the state's values (VPD: recruit.hybrid gates in C0's masked
         forward; neurons: the state's firing at every position in W0's model, whose all-neuron limit is E98's r_n)
  A      (per task, after both seeds) Jaccard of the set between seeds
B and C also on 20 random same-count sets and on the whole set (the limits; C's limits are controls).

  W0 control on every task (delta-W = 0): exact zeros, log-odds differences <= 1e-5      -> results/cells.json "w0"
  the states, random maps first; after each: S_on count, r at every alive subcomponent and r_N at every neuron
  against their references (a miss writes, logs FAIL and exits 1)                         -> results/cells.json "rows"
  Jaccards per task                                                                        -> results/cells.json "jaccard"
Every number is printed. score.py applies the separation rule; plot.py draws figures/panels.png.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import neuron_r as NR
from commute import recruit as RC
from commute import recruitment as RT
from commute import saved_states, vpd
from commute import steering as ST
from commute.data import NEXT, target_tokens
from commute.finetune import logits_of, random_direction
from commute.identity import log_odds
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.keydep import log, sha256
from commute.local_faith import masked_lo_windows
from commute.recruit import Sel

HERE = Path(__file__).resolve().parent
RESULTS, EXP, REPO = HERE / "results", HERE.parent, HERE.parents[1]
PRED, OUT = RESULTS / "predictions.json", RESULTS / "cells.json"
ALIVE = EXP / "transport_test" / "results" / "large"
TABLE, E98_CELLS = EXP / "regating_truth_table" / "results" / "table.json", EXP / "neuron_r" / "results" / "cells.json"
E12_A2 = EXP / "new_vs_existing" / "results" / "new_vs_existing_amend2.json"
E98 = load_experiment("neuron_r", "run")  # E98's files (with recorded sha256), state table, state loader; E63's task loaders
TT = E98.E63.TT  # the truth table's run.py: E46_TASKS, NextTask
DEVICE, N_LAYERS, D_MLP = "cuda", ST.N_LAYERS, ST.D_MLP
Acts = NR.Acts


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout.strip()


class Out:
    """cells.json, written whole after every record; a miss writes, logs FAIL and exits 1."""

    def __init__(self, P: dict[str, Any]) -> None:
        self.out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "git_dirty": git("status", "--porcelain", "--untracked-files=no"),
                                    "predictions_sha256": sha256(PRED), "settings": P["settings"], "w0": {}, "rows": {}, "jaccard": {}}

    def put(self, part: str, key: str, rec: dict[str, Any], failed: list[str] | None = None) -> None:
        self.out[part][key] = {**rec, **({} if failed is None else {"failed": failed})}
        OUT.write_text(json.dumps(E98.E63.clean(self.out), indent=1, sort_keys=True))
        log(f"[{part} {key}] {json.dumps(E98.E63.clean(rec), sort_keys=True)}")
        if failed:
            self.out["stopped"] = f"{part}/{key}: {failed}"
            OUT.write_text(json.dumps(E98.E63.clean(self.out), indent=1, sort_keys=True))
            log(f"FAIL: {part}/{key}: {failed} stop and tell the lead")
            sys.exit(1)


class Acc:
    """Per named per-window difference: its sum (for the mean) and its largest absolute value (for the W0 control)."""

    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.max: dict[str, float] = {}

    def add(self, key: str, d: Tensor) -> None:
        d = d.double().cpu()
        self.sum[key] = self.sum.get(key, 0.0) + float(d.sum())
        self.max[key] = max(self.max.get(key, 0.0), float(d.abs().max()))


class Ctx:
    def __init__(self) -> None:
        self.net = ST.Net(DEVICE)
        self.m = self.net.model
        self.comps0 = vpd.component_weights(self.m)
        self.keys = sorted(self.comps0)
        self.alive = saved_states.load_alive(ALIVE)
        self.all_alive: Sel = {k: v.nonzero().flatten() for k, v in self.alive.items()}
        self.neuron_pool = {str(l): torch.ones(D_MLP, dtype=torch.bool) for l in range(N_LAYERS)}
        self.all_neurons: Sel = {str(l): torch.arange(D_MLP) for l in range(N_LAYERS)}


# ----------------------------------------------------------------------------- the sets


@torch.no_grad()
def change_at_read(c: Ctx, t: ST.Task, dW: ST.DW, B: int) -> tuple[dict[str, Tensor], Tensor, float, float]:
    """Mean over windows, at the read position, of each alive subcomponent's gate change (`recruit.trigger_dg`) and of each
    neuron's firing change (12288,) float64, state minus W0; and max |g1 - g0|, max |a1 - a0| anywhere (the W0 control)."""
    h, g0s, g1s, fire, max_g, max_a = t.half, [], [], torch.zeros(N_LAYERS * D_MLP, dtype=torch.float64), 0.0, 0.0
    for toks in t.toks.split(B):
        g0, a0 = vpd.compute_ci(c.m, toks, chunk=len(toks)).g, NR.forward(c.net, toks)[2]
        with c.net.at(dW):
            g1, a1 = vpd.compute_ci(c.m, toks, chunk=len(toks)).g, NR.forward(c.net, toks)[2]
        max_g = max(max_g, *(float((g1[k].float() - g0[k].float()).abs().max()) for k in g0))
        max_a = max(max_a, *(float((a1[l] - a0[l]).abs().max()) for l in a0))
        g0s.append({k: v[:, h : h + 1].cpu() for k, v in g0.items()})
        g1s.append({k: v[:, h : h + 1].cpu() for k, v in g1.items()})
        fire += torch.cat([(a1[l][:, h] - a0[l][:, h]).double().sum(0).cpu() for l in range(N_LAYERS)])
    return RC.trigger_dg(g1s, g0s, 0, c.alive), fire / len(t.toks), max_g, max_a


def top_neurons(fire: Tensor, k: int) -> Sel:
    """The k neurons of largest mean firing change (stable sort: ties to the lower flat index), as {block: indices}."""
    order = torch.sort(fire, descending=True, stable=True).indices[:k]
    return {str(l): (order[order // D_MLP == l] % D_MLP).sort().values for l in range(N_LAYERS)}


def put(sel: Sel, src: Acts, at: int | None = None) -> ST.Edit:
    """`Net.run` edits: in block l, the firing of sel's neurons replaced by src's, at position `at` (every position if
    None); every other neuron and position as the forward computes it."""

    def edit(l: int, idx: Tensor) -> Any:
        def f(x: Tensor) -> Tensor:
            x, i = x.clone(), idx.to(x.device)
            if at is None:
                x[..., i] = src[l][..., i]
            else:
                x[:, at, i] = src[l][:, at, i]
            return x

        return f

    return {f"mlp{l}": edit(int(l), idx) for l, idx in sel.items() if idx.numel()}


# ----------------------------------------------------------------------------- the cells, one batch at a time


def vpd_cells(c: Ctx, acc: Acc, toks: Tensor, h: int, token: int | Tensor, dW: ST.DW, sets: dict[str, Sel]) -> None:
    """C (v_G, v_r_*): masked log-odds under W0's gates with the set's gates from the state, minus L00, over Gamma's T1 - T0.
    B (v_Gb, v_rho_*): the fine-tuned model's log-odds minus the same with the set restored at the read position."""
    x = toks.to(c.net.device)
    tgt = target_tokens(token, x, h)
    lo = lambda lg: log_odds(lg[:, h], tgt).double().cpu()
    full = lambda: lo(logits_of(c.m.target_model, x))
    ci0, io0, P0 = vpd.compute_ci(c.m, x, chunk=len(x)), RT.trigger_io(c.m, c.keys, x, h, len(x)), full()
    with c.net.at(dW):
        ci1, P1 = vpd.compute_ci(c.m, x, chunk=len(x)), full()
        for n, sel in sets.items():
            with RT.restore_hooks(c.m, c.comps0, sel, io0, h):
                acc.add(f"v_rho_{n}", P1 - full())
    L = lambda g: masked_lo_windows(c.m, x, h, token, [g], len(x))
    L00 = L(ci0.g)
    acc.add("v_G", lo(ci1.logits) - lo(ci0.logits))
    acc.add("v_Gb", P1 - P0)
    acc.add("v_r_every", L(ci1.g) - L00)  # reported beside the all-alive control
    for n, sel in sets.items():
        acc.add(f"v_r_{n}", L(RC.hybrid([ci0.g], [ci1.g], sel)[0]) - L00)


def neuron_cells(c: Ctx, acc: Acc, toks: Tensor, h: int, tgt: Tensor, dW: ST.DW, sets: dict[str, Sel]) -> None:
    """C (n_G, n_r_*): W0's model with the set's firing from the state at every position, minus T0, over T1 - T0.
    B (n_rho_*): T1 minus the fine-tuned model with the set's firing at the read position set to Ah (W0's c_fc and GELU
    on the state's MLP input); n_rho0_*: the same set to W0's own firing (reported)."""
    net, lo = c.net, lambda lg: ST.logodds_at(lg, h, tgt)
    lg0, a0, ah = NR.half_acts(net, toks, dW)
    with net.at(dW):
        lg1, _, a1 = NR.forward(net, toks)
    T0, T1 = lo(lg0), lo(lg1)
    acc.add("n_G", T1 - T0)
    for n, sel in sets.items():
        acc.add(f"n_r_{n}", lo(net.run(toks, put(sel, a1))[0]) - T0)
    with net.at(dW):
        for n, sel in sets.items():
            acc.add(f"n_rho_{n}", T1 - lo(net.run(toks, put(sel, ah, h))[0]))
        for n in ("all", "set"):
            acc.add(f"n_rho0_{n}", T1 - lo(net.run(toks, put(sets[n], a0, h))[0]))


def share_block(mean: dict[str, float], G: float, prefix: str, n_random: int, K: int, floor: float) -> dict[str, Any]:
    """One side of one panel: the set's share, the whole set's, the random sets' mean / min / max; nan when |G| < floor."""
    ok = abs(G) >= floor
    sh = lambda n: mean[f"{prefix}_{n}"] / G if ok else math.nan
    rand = [sh(f"rand{j}") for j in range(n_random)]
    return {"G": G, "readable": bool(ok and K > 0), "set": sh("set"), "all": sh("all"), "num_set": mean[f"{prefix}_set"],
            "random": {"mean": sum(rand) / len(rand), "min": min(rand), "max": max(rand)} if rand else None}


@torch.no_grad()
def read_state(c: Ctx, S: dict[str, Any], t: ST.Task, token: int | Tensor, B: int, dW: ST.DW) -> tuple[dict[str, Any], Sel, Tensor]:
    dg, fire, max_g, max_a = change_at_read(c, t, dW, B)
    S_on = RC.switched(dg, S["s_on_thr"])
    K = RC.size(S_on)
    N_on = top_neurons(fire, K)
    nr = S["n_random"] if K else 0
    v_sets = {"set": S_on, "all": c.all_alive, **{f"rand{j}": RC.random_like(S_on, c.alive, j) for j in range(nr)}}
    n_sets = {"set": N_on, "all": c.all_neurons, **{f"rand{j}": RC.random_like(N_on, c.neuron_pool, j) for j in range(nr)}}
    acc = Acc()
    for toks in t.toks.split(B):
        vpd_cells(c, acc, toks, t.half, token, dW, v_sets)
        neuron_cells(c, acc, toks, t.half, t.tgt(toks), dW, n_sets)
    m, f = {k: v / len(t.toks) for k, v in acc.sum.items()}, S["gain_floor"]
    flat = sorted(int(l) * D_MLP + int(i) for l, v in N_on.items() for i in v)
    rec = {"K_VPD": K, "n_windows": len(t.toks), "S_on": sorted([k, i] for k, i in RC.members(S_on)), "S_on_by_layer": RC.by_layer(S_on),
           "N_on": flat, "N_on_by_block": {l: int(v.numel()) for l, v in N_on.items()},
           "N_on_min_change": float(fire[flat].min()) if K else None,
           "max_abs_dg": max_g, "max_abs_dfire": max_a, "max_abs": acc.max, "means": m,
           "B": {"vpd": share_block(m, m["v_Gb"], "v_rho", nr, K, f), "neurons": share_block(m, m["n_G"], "n_rho", nr, K, f),
                 "neurons_a0": share_block(m, m["n_G"], "n_rho0", 0, K, f)},
           "C": {"vpd": share_block(m, m["v_G"], "v_r", nr, K, f), "neurons": share_block(m, m["n_G"], "n_r", nr, K, f)}}
    rec["C"]["vpd"]["every"] = m["v_r_every"] / m["v_G"] if abs(m["v_G"]) >= f else math.nan
    return rec, S_on, fire


# ----------------------------------------------------------------------------- controls


def w0_failed(rec: dict[str, Any], tol: float) -> list[str]:
    bad = [f"max |dg| {rec['max_abs_dg']}"] if rec["max_abs_dg"] != 0 else []
    bad += [f"max |dfire| {rec['max_abs_dfire']}"] if rec["max_abs_dfire"] != 0 else []
    return bad + [f"{k} max |d| {v:.3g}" for k, v in rec["max_abs"].items() if not v <= tol]


def references(P: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The controls' targets from their source files, checked against the values filed (rounded) in predictions.json."""
    C, table, e98 = P["apparatus_controls"], load(TABLE)["runs"], load(E98_CELLS)["rows"]
    key = lambda n: ("e46/" if n.rsplit("_", 1)[0] in TT.E46_TASKS else "e53/") + n.replace("e53_cipher", "cipher")
    out = {"r": {n: table[key(n)]["r"]["value"] for n in C["r_all_alive"]["expected"]},
           "r_n": {n: e98[n]["r_n"] for n in C["r_n_all_neurons"]["expected"]},
           "n_on": {n: load(E12_A2)["fine_tunes"][n]["n_on"] for n in C["s_on_counts"]["expected"]}}
    for name, filed in (("r", "r_all_alive"), ("r_n", "r_n_all_neurons")):
        assert all(abs(out[name][n] - v) <= 5e-5 for n, v in C[filed]["expected"].items()), name
    assert out["n_on"] == C["s_on_counts"]["expected"]
    return out


def control_failed(name: str, rec: dict[str, Any], ref: dict[str, dict[str, float]]) -> list[str]:
    bad = []
    if name in ref["n_on"] and rec["K_VPD"] != ref["n_on"][name]:
        bad.append(f"K_VPD {rec['K_VPD']} != E12's {ref['n_on'][name]}")
    if name in ref["r"] and not abs(rec["C"]["vpd"]["all"] - ref["r"][name]) <= 0.005:
        bad.append(f"r at every alive {rec['C']['vpd']['all']:.5f} vs {ref['r'][name]:.5f} (r_every {rec['C']['vpd']['every']:.5f})")
    if name in ref["r_n"] and not abs(rec["C"]["neurons"]["all"] - ref["r_n"][name]) <= 0.01:
        bad.append(f"r_N at every neuron {rec['C']['neurons']['all']:.5f} vs E98's r_n {ref['r_n'][name]:.5f}")
    return bad


def jaccard_rec(a: tuple[int, Sel, Tensor], b: tuple[int, Sel, Tensor]) -> dict[str, Any]:
    """Panel A for one task: S_on's Jaccard between the seeds, and N_on's at K_A = the seeds' mean K_VPD, rounded half up."""
    K_A = math.floor((a[0] + b[0]) / 2 + 0.5)
    return {"K_VPD": [a[0], b[0]], "K_A": K_A, "readable": a[0] > 0 and b[0] > 0, "vpd": RC.jaccard(a[1], b[1]),
            "neurons": RC.jaccard(top_neurons(a[2], K_A), top_neurons(b[2], K_A))}


# ----------------------------------------------------------------------------- main


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    require_filed(PRED)
    P = load(PRED)
    S, order = P["settings"], P["states"]["order"]
    ref = references(P)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    table = E98.state_table()
    names = [n for n in order if n != "random_update"]
    F_all = E98.files()
    keep = {table[n][0] for n in names} | {f"e46/windows_{t}" for t in TT.E46_TASKS} | {"e54/windows_elicit", "e54/windows_cipher"}
    F = {k: F_all[k] for k in sorted(keep)}
    bad = [k for k, (p, h) in F.items() if not p.is_file() or sha256(p) != h]
    o = Out(P)
    o.out["sha256"] = {"n_files": len(F), "mismatch": bad}
    if bad:
        sys.exit(f"sha256 mismatch or missing: {bad}")
    log(f"[sha256] {len(F)} files match")
    c = Ctx()
    tasks, _ = E98.E63.load_tasks(c.net, TT.performed_anchor(), F)
    tok = lambda t: NEXT if isinstance(t, TT.NextTask) else t.table
    bat = lambda task: S["batch"]["kv" if task in TT.E46_TASKS else "pile"]
    used = sorted({table[n][1] for n in names})
    zero = {k: torch.zeros_like(w) for k, w in c.net.W0.items()}
    for task in used:  # the W0 control, every task, before any state
        rec = read_state(c, S, tasks[task], tok(tasks[task]), bat(task), zero)[0]
        o.put("w0", task, rec, w0_failed(rec, P["w0_control"]["tol"]))
    kept: dict[str, tuple[int, Sel, Tensor]] = {}
    for n in order:
        if n == "random_update":
            task, dW = S["random_update"]["windows"], random_direction(E98.state(F, table[S["random_update"]["like"]][0]), S["random_update"]["seed"])
        else:
            task, dW = table[n][1], E98.state(F, table[n][0])
        rec, S_on, fire = read_state(c, S, tasks[task], tok(tasks[task]), bat(task), dW)
        rec["task"] = task
        o.put("rows", n, rec, control_failed(n, rec, ref))
        kept[n] = (rec["K_VPD"], S_on, fire)
        stem = n.rsplit("_", 1)[0]
        if n.endswith("_s1") and f"{stem}_s0" in kept:
            o.put("jaccard", stem, jaccard_rec(kept[f"{stem}_s0"], kept[n]))
    o.out["done"], o.out["peak_gpu_gb"] = True, torch.cuda.max_memory_allocated() / 2**30
    OUT.write_text(json.dumps(E98.E63.clean(o.out), indent=1, sort_keys=True))
    log(f"[done] peak GPU {o.out['peak_gpu_gb']:.1f} GB")
    hard_exit(0)


if __name__ == "__main__":
    main()

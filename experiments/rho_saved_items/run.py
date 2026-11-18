"""E208 (cell P1.30): rho as the working definition's note writes it (theory/2026-11-18_definition_causal_abstraction.md),
read on saved items, with each testable open choice read beside it (plan plan.md, appendix appendix.md, card card.md;
filed settings, expected values and rules in results/predictions.json, which the run refuses unless committed
unchanged). The measurement is commute.rho. Items are rebuilt exactly as E196 (rule_retention_pairs), E195
(rule_retention_fixed) and E206 (graft_sites) built them; nothing is trained.

Per item: the changed parts, the per-part r under four readings (main, i, iii, b; one profile pass), the hybrid gate and
the M gate (stops: code errors), then K searched by mechanism replacement ("replace") and by activation patching
("patch"), 5 orders each; on E195 rebuilt also with its zeroed parts restored ("restore"). Each item writes
results/items/<item>.json and results/large/parts/<item>.npz when it ends and is skipped on a rerun.
Stage 0 runs first and alone on every pod, once per tag: do-nothing, C x2, C x10 on copy24's 512 read windows; exit 1 on
a miss. Items, in the order given on the command line:
    <reuse or twin>_s<seed>   an E196 fine-tune (gap check against E184 first), KL at the last position
    slotted                   E206's 256 lookup neurons written into M's 256 quietest layer-3 slots (control, stops)
    neurons_appended, head_appended   E206's grafts as built (rule checks, stop)
    rebuilt                   E195's rebuilt item, KL at every position of its 75 windows

    python experiments/rho_saved_items/run.py --tag pod3 ident32_s0 rand32_s0 2>&1 | tee -a experiments/rho_saved_items/results/run_pod3.log
    python experiments/rho_saved_items/run.py --summary     (no GPU: R1 and the R3 table from the item files present)
    (on a pod: PRED_SHA=<sha of the predictions file> python ...)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import graft as G
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

P: Any = load_experiment("rule_retention", "run")  # P1.24: clone, rescaled, chunks, log, dump, stop, COPY24
E196: Any = load_experiment("rule_retention_pairs", "run")  # Data, fine_tune, gap, subset, E184, NUMBERS
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, ITEMS, PARTS = RESULTS / "predictions.json", RESULTS / "items", RESULTS / "large" / "parts"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
REBUILT = EXPERIMENTS / "rule_retention_fixed" / "results" / "large" / "rebuilt_mlp.pt"
GRAFTS = EXPERIMENTS / "graft_sites" / "results" / "large"
CONTROLS = ("slotted", "neurons_appended", "head_appended")
LAST, ALL = slice(-1, None), slice(None)
log, dump, stop = P.log, P.dump, P.stop


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    layer: int  # C's layer (P1.24's circuit)
    C: tuple[int, ...]
    factors: tuple[float, ...]  # stage 0's turned-up items
    control_task: str
    pairs: tuple[tuple[str, str], ...]  # (reuse, twin)
    counted: int  # the first `counted` pairs are scored by R1
    seeds: tuple[int, ...]
    n_read: int  # E196's read windows per item
    read_seed: int
    n_max: int  # E173's numbers (twin and fact windows)
    held_share: float
    eval_batch: int  # E196's gap check
    profile_batch: int  # windows per batch of the profile pass
    hybrid_batch: int  # windows per batch of a hybrid forward
    grafts: tuple[str, ...]  # E206's graft files: neurons, then the head
    graft_layer: int
    graft_window: int
    graft_val_rows: int
    graft_n_read: int
    graft_read_seed: int
    slot_n: int

    def __post_init__(self) -> None:
        assert len(set(self.C)) == len(self.C) and max(self.C) < R.D_MLP and 1 not in self.factors and 0 < self.counted <= len(self.pairs)
        assert self.grafts == ("neurons", "last_lowest") and 0 < self.slot_n <= R.D_MLP and set(self.seeds) <= {0, 1}


def pair_items(cfg: Config) -> list[str]:
    return [f"{m}_s{s}" for pr in cfg.pairs for s in cfg.seeds for m in pr]


# ---- the items --------------------------------------------------------------------------------------------------------


def graft_windows(cfg: Config) -> Tensor:
    """E206's read windows (graft_negative.data: the second half of val rows 0..val_rows, cut by read_seed)."""
    va = pile_rows("val", 0, cfg.graft_val_rows, log=log)
    return cut(va[cfg.graft_val_rows // 2 :], cfg.graft_n_read, cfg.graft_window, torch.Generator().manual_seed(cfg.graft_read_seed)).to("cuda")


def appended(M: R.Model, cfg: Config, kind: str) -> R.Model:
    """E206's graft as built: its saved weights on a fresh attachment (the init seed is overwritten by the load)."""
    F = P.clone(M)
    raw: dict[str, Any] = json.loads((EXPERIMENTS / "graft_sites" / "results" / "predictions.json").read_text())["settings"]["recipe"]
    fields: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    rec = G.Recipe(**fields)
    g = G.attach_neurons(F, cfg.graft_layer, cfg.slot_n, 0, rec) if kind == "neurons" else G.attach_head(F, cfg.graft_layer, 0, rec)
    G.load(g, GRAFTS / f"{cfg.grafts[0] if kind == 'neurons' else cfg.grafts[1]}.pt")
    return F


@torch.no_grad()
def slotted(M: R.Model, cfg: Config, toks: Tensor) -> tuple[R.Model, list[int]]:
    """E206's lookup neurons written into the slot_n layer-3 slots of M whose writes on the read windows are smallest."""
    l, e = cfg.graft_layer, appended(M, cfg, "neurons").extra[cfg.graft_layer]
    energy = torch.zeros(R.D_MLP, device=toks.device, dtype=torch.float64)
    for t in P.chunks(toks, cfg.profile_batch):
        energy += R.acts(M, l, R.m_inputs(M, t)[l][1]).double().pow(2).sum((0, 1))
    slots = (energy * R.cols(M, l).double().pow(2).sum(0)).argsort()[: cfg.slot_n]
    F = P.clone(M)
    F.net.h[l].mlp.c_fc.weight[slots] = e.w_in
    F.net.h[l].mlp.down_proj.weight[:, slots] = e.w_out
    return F, sorted(slots.tolist())


@torch.no_grad()
def rebuilt(M: R.Model, cfg: Config) -> R.Model:
    """E195's rebuilt item: C's output columns zeroed, layers 2-3 MLP weights from rebuilt_mlp.pt (rule_retention_fixed)."""
    F = P.clone(M)
    F.net.h[cfg.layer].mlp.down_proj.weight[:, list(cfg.C)] = 0
    for k, v in torch.load(REBUILT, weights_only=True).items():
        F.net.get_parameter(k).copy_(v)
    return F


# ---- reading one item -------------------------------------------------------------------------------------------------


def read(F: R.Model, M: R.Model, toks: Tensor, at: slice, cfg: Config, th: dict[str, Any], item: str) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.time()
    p = rho.Pair(F, M)
    cand, lacks = rho.changed(p)
    prof = rho.profile(p, P.chunks(toks, cfg.profile_batch))
    b = rho.batches(p, toks, cfg.hybrid_batch, at)
    out: dict[str, Any] = {"n_candidates": len(cand), "n_lacks": len(lacks), "candidates_heads": [rho.name(u) for u in cand if u[0] != "neuron"],
                           "candidates": [rho.name(u) for u in cand] if len(cand) <= 512 else None,
                           "gate": rho.gate(p, b, [], F.logits), "m_gate": rho.gate(p, b, cand + lacks, M.logits), "profile_seconds": time.time() - t0}  # fmt: skip
    common = [u for u in cand + lacks if u[0] == "neuron" and not rho.lacked_by_m(p, u)]
    ok = [u for u in common if prof["one_zero"][u] == 0]
    zc = {u: prof["both_zero"][u] + prof["both_active"][u] * prof["col_cos2"][u] for u in common}
    out["identity"] = {"neurons": len(common), "with_one_zero_write": len(common) - len(ok),
                       "max_dev_filed": max((abs(prof["r"]["main"][u] - prof["col_cos2"][u]) for u in ok), default=0.0),  # reported only
                       "max_dev": max((abs(prof["r"]["main"][u] - zc[u]) for u in common), default=0.0),  # amendment 1: r = z + a c
                       "mean_both_zero": sum(prof["both_zero"][u] for u in common) / len(common) if common else None}  # fmt: skip
    out["r_changed"] = {x: {"mean": sum(rho.rvalue(p, prof["r"][x], u) for u in cand) / len(cand), "min": min(rho.rvalue(p, prof["r"][x], u) for u in cand)} if cand else None for x in rho.READS}
    log(f"{item}: {len(cand)} changed parts ({len(out['candidates_heads'])} not neurons: {out['candidates_heads'][:30]}), {len(lacks)} parts F lacks; "
        f"hybrid gate {out['gate']:.3g}, M gate {out['m_gate']:.3g}; identity check {out['identity']}; r over changed parts {out['r_changed']}")
    if not (out["gate"] <= th["gate_tol"] and out["m_gate"] <= th["m_gate_tol"] and out["identity"]["max_dev"] <= th["identity_tol"]):
        stop(ITEMS / f"{item}.stopped.json", out, f"{item}: a code check missed (gate, M gate or identity)")
    hows = {"replace": (False, cand), "patch": (True, cand)} | ({"restore": (False, cand + lacks)} if lacks else {})
    arrays: dict[str, Any] = {"parts": np.array([rho.name(u) for u in cand + lacks])}
    arrays |= {f"r_{x}": np.array([rho.rvalue(p, prof["r"][x], u) for u in cand + lacks]) for x in rho.READS}
    for how, (patch, cs) in hows.items():
        c = rho.contrast(p, b, cs, prof["r"], patch, th, log, f"{item} {how}")
        for k, o in enumerate(c.get("orders", [])):
            K = [cs[i] for i in o["K"]]  # active share: 1 - both-zero share for a neuron M has, 1 otherwise (amendment 1)
            o["active_share"] = sum(1 - prof["both_zero"].get(u, 0.0) for u in K) / len(K)
            arrays[f"K_{how}_{k}"] = np.array(o.pop("K"))
        out[how] = c
        log(f"{item} {how}: {c['reading']}, KL(F || M) {c['kl']:.6g}, closure with nothing kept {c.get('closure_none_kept')}; "
            + ", ".join(f"rho_{x} {c.get(f'rho_{x}')}" for x in rho.READS))
    out["seconds"] = time.time() - t0
    return out, arrays


def save(item: str, out: dict[str, Any], arrays: dict[str, Any]) -> dict[str, Any]:
    PARTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PARTS / f"{item}.npz", **arrays)
    dump(ITEMS / f"{item}.json", out | {"experiment": EXPERIMENT, "item": item})
    log(f"{item}: done in {out['seconds']:.0f} s")
    return out


# ---- stage 0 and the controls -----------------------------------------------------------------------------------------


def misses(item: str, rd: dict[str, Any], want_parts: set[rho.Part] | None, want: dict[str, float], th: dict[str, Any]) -> list[str]:
    """The filed checks: no change for do-nothing; else the changed parts as built, every order reading, and each rho in
    `want` met in every order of both constructions: main, i, iii, b the value; 'b_amend1' the per-order value
    1 - d * (mean over K of the active share), d the per-setting d_b of the turned-up write (amendment 1)."""
    if want_parts is None:
        return [] if rd["n_candidates"] == 0 and rd["replace"]["reading"] == rd["patch"]["reading"] == "no change" else [f"{item} not 'no change'"]
    built = sorted(rho.name(u) for u in want_parts)
    out = [f"{item}: changed parts {rd['n_candidates']} ({(rd['candidates'] or [])[:10]}...), not the {len(built)} built"] if sorted(rd["candidates"] or []) != built else []
    for how in ("replace", "patch"):
        c = rd[how]
        if c["reading"] != "read":
            out.append(f"{item} {how} {c['reading']}")
            continue
        for k, v in want.items():
            exp = [1 - v * o["active_share"] for o in c["orders"]] if k == "b_amend1" else [v] * len(c["orders"])
            got = c["rho_b" if k == "b_amend1" else f"rho_{k}"]
            bad = [(g, e) for g, e in zip(got, exp, strict=True) if abs(g - e) > th["r_tol"]]
            out += [f"{item} {how} rho_{k} (got, expected) {bad}"] if bad else []
    return out


def stage0(M: R.Model, data: Any, cfg: Config, th: dict[str, Any], tag: str) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    toks, out, miss = E196.subset(data.read_half(cfg.control_task), cfg, M), {"experiment": EXPERIMENT}, []
    C = {("neuron", cfg.layer, i) for i in cfg.C}
    for item, F, want in [("do_nothing", P.clone(M), {})] + [(f"x{a:g}", P.rescaled(M, cfg.layer, list(cfg.C), a), {"main": 1.0, "i": 1.0, "iii": 1.0, "b_amend1": (a - 1) ** 2 / (a * a + 1)}) for a in cfg.factors]:
        rd, _ = read(F, M, toks, LAST, cfg, th, item)
        miss += misses(item, rd, None if item == "do_nothing" else C, want, th)
        out[item] = rd
    log(f"stage 0: {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def control(M: R.Model, cfg: Config, th: dict[str, Any], item: str) -> dict[str, Any]:
    toks = graft_windows(cfg)
    if item == "slotted":
        F, slots = slotted(M, cfg, toks)  # amendment 1: rho_main against slot_bar is reported, not a stop (shared silence reads as agreement)
        want_parts, want = {("neuron", cfg.graft_layer, i) for i in slots}, {}
    else:
        F = appended(M, cfg, item.split("_")[0])
        want_parts = {("neuron", cfg.graft_layer, R.D_MLP + i) for i in range(cfg.slot_n)} if item == "neurons_appended" else {("head", cfg.graft_layer, R.N_HEAD)}
        want = {"main": 0.0, "i": 0.0, "iii": 0.0, "b": 0.0}
    rd, arrays = read(F, M, toks, ALL, cfg, th, item)
    if item == "slotted":
        rd["slot_bar_filed"] = {how: [v <= th["slot_bar"] for v in rd[how].get("rho_main", [])] for how in ("replace", "patch")}
        log(f"slotted: rho_main <= slot_bar (filed, reported only since amendment 1) {rd['slot_bar_filed']}; mean both-zero share {rd['identity']['mean_both_zero']}")
    if miss := misses(item, rd, want_parts, want, th):
        stop(ITEMS / f"{item}.stopped.json", rd, "; ".join(miss))
    return save(item, rd, arrays)


def pair_item(net: Any, M: R.Model, data: Any, cfg: Config, th: dict[str, Any], item: str) -> dict[str, Any]:
    Fnet, half = E196.fine_tune(net, item), data.read_half(item.rsplit("_s", 1)[0])
    g, e184 = E196.gap(Fnet, net, half, cfg), json.loads((E196.E184 / f"{item}.json").read_text())["gap"]
    log(f"{item}: gap (mean TV, last position) {g:.6f}, E184's {e184:.6f}")
    if not abs(g - e184) <= th["reload_tol"]:
        stop(ITEMS / f"{item}.stopped.json", {"gap": g, "gap_e184": e184}, f"{item}: the gap does not repeat E184's")
    rd, arrays = read(R.Model(Fnet.net), M, E196.subset(half, cfg, M), LAST, cfg, th, item)
    return save(item, rd | {"gap": g, "gap_e184": e184}, arrays)


# ---- R1 and the R3 table ----------------------------------------------------------------------------------------------


def mean(v: list[float]) -> float:
    return sum(v) / len(v)


def sd(v: list[float]) -> float:
    return math.sqrt(sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1))


def r1(items: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """The filed rule (predictions.json R1): per seed 'A', 'B', 'neither' or 'not read'; the outcome over both seeds."""
    names = [pr[0].split("_")[0] for pr in cfg.pairs[: cfg.counted]]  # copy24, ident32, case, repeat
    out: dict[str, Any] = {}
    for s in cfg.seeds:
        got: list[tuple[Any, Any]] = [(items.get(f"{a}_s{s}"), items.get(f"{b}_s{s}")) for a, b in cfg.pairs[: cfg.counted]]
        if any(x is None or x["replace"]["reading"] != "read" for pr in got for x in pr):
            out[f"s{s}"] = {"call": "not read"}
            continue
        g = {n: mean(a["replace"]["rho_main"]) - mean(b["replace"]["rho_main"]) for n, (a, b) in zip(names, got, strict=True)}
        cp, idn, case, rep = (g[n] for n in names)
        call = "A" if idn > max(rep, 0.0) else "B" if case > rep > max(cp, idn) else "neither"
        out[f"s{s}"] = {"call": call, "gaps": g, "ident_minus_repeat": idn - rep,
                        "largest_order_sd": max(sd(x["replace"]["rho_main"]) for pr in got for x in pr)}  # fmt: skip
    calls = [v["call"] for v in out.values()]
    outcome = "not read" if "not read" in calls else calls[0] if len(set(calls)) == 1 and calls[0] in ("A", "B") else "mixed"
    return out | {"outcome": outcome}


def summary(cfg: Config) -> dict[str, Any]:
    items = {p.stem: json.loads(p.read_text()) for p in ITEMS.glob("*.json") if not p.stem.endswith(".stopped")}
    table = {}
    for name, it in sorted(items.items()):
        rep = it["replace"]
        row = {f"{x}_minus_main": mean(rep[f"rho_{x}"]) - mean(rep["rho_main"]) for x in ("i", "iii", "b")} if rep["reading"] == "read" else {}
        for how in ("replace", "patch", "restore"):
            c = it.get(how)
            if c and c["reading"] == "read":
                row |= {f"rho_main_{how}": mean(c["rho_main"]), f"order_sd_{how}": sd(c["rho_main"]), f"size_{how}": mean([o["size"] for o in c["orders"]])}
        table[name] = row
    return {"experiment": EXPERIMENT, "R1": r1(items, cfg), "R3": table}


# ---- the job ------------------------------------------------------------------------------------------------------------


def main(items: list[str], tag: str, summary_only: bool) -> int:
    pred = json.loads(PRED.read_text())
    cfg, th = config_from_settings(Config, {k: tuple(map(tuple, v)) if k == "pairs" else tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}), pred["thresholds"]
    if summary_only:
        dump(RESULTS / "summary.json", s := summary(cfg))
        log(f"R1: {s['R1']}")
        return 0
    unknown = [i for i in items if i not in pair_items(cfg) + [*CONTROLS, "rebuilt"]]
    assert not unknown, f"unknown items {unknown}"
    from commute import old_tasks as OT
    from commute import variable_read as VR
    from commute import vpd
    from commute.steering import Net

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout
    log(f"{EXPERIMENT} start, tag {tag}, items {items}; settings {asdict(cfg)}; thresholds {th}")
    net = Net("cuda")
    net.net.requires_grad_(False)
    M, tok, dcfg = R.Model(net.net), vpd.tokenizer(), OT.DSizes()
    x = VR.make_numbers(tok, int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    if x.keys.tolist() != json.loads(E196.NUMBERS.read_text())["number_tokens"]:
        log("STOP: the number tokens are not E154's")
        return 1
    data = E196.Data(VR.make_sets(dcfg, tok, log), x, dcfg, {})
    stage0(M, data, cfg, th, tag)
    ITEMS.mkdir(parents=True, exist_ok=True)
    for item in items:
        if (ITEMS / f"{item}.json").is_file():
            log(f"{item}: done before, skipped")
            continue
        if item in CONTROLS:
            control(M, cfg, th, item)
        elif item == "rebuilt":
            read_rebuilt = read(rebuilt(M, cfg), M, torch.load(P.COPY24, weights_only=False)["heldout"].tokens.to("cuda"), ALL, cfg, th, item)
            save(item, *read_rebuilt)
        else:
            pair_item(net, M, data, cfg, th, item)
        torch.cuda.empty_cache()
    log(f"R1 so far: {summary(cfg)['R1']}")
    log(f"{EXPERIMENT} DONE")
    return 0


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*", help="items to read, in order (pods.md)")
    ap.add_argument("--tag", default="local", help="this pod's name: stage 0 writes results/stage0_<tag>.json")
    ap.add_argument("--summary", action="store_true", help="only rewrite results/summary.json from the item files present")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.items, args.tag, args.summary)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()

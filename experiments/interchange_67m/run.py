"""interchange_67m (E66), the feasibility probe: can a context patch and a neuron ablation read anything on the 67M's fine-tunes?
Plan: reports/plans/2026-11-02_plan_e66-interchange-67m.md, "Amendment 1: feasibility probe", and Fable's review. Rules,
settings, expected values: results/predictions.json (controls only; committed first; the run refuses otherwise; never edited).
Definitions and rules: `commute.ctxpatch`. Reads saved states only; the one training is the sharpening control.

  0 W0 alone: x_hi's input written into x_lo's run reproduces x_hi's read; R(W0) per block and l*; K. Exit 1 on a miss.
  1 the built controls per map: write, amplify, sharpen. T, T_off, Q, U. Exit 1 if amplify's T misses on half the readable maps.
  2 E62's six states, 3 the eight copy and random maps: reproduce G (exit 1 on a miss), steering, T, T_off, Q, U. Exploratory.
  4 E62 Amendment 2's size-swapped copies: T, Q beside their unscaled values. Exploratory. T against size: the analysis, from run.json.
results/run.json after every stage; every number printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

from commute import anchors as AN
from commute import copying as CP
from commute import ctxpatch as X
from commute import elicit as EL
from commute import keydep as KD
from commute import kv
from commute import steering as ST
from commute.ctxpatch import Map
from commute.data import Windows, pile_rows
from commute.open_pair import jsonable
from commute.steering import DW

HERE = Path(__file__).resolve().parent
RESULTS, REPO, EXP = HERE / "results", HERE.parents[1], HERE.parent
PRED, OUT = RESULTS / "predictions.json", RESULTS / "run.json"
log = KD.log
Rec = dict[str, Any]
Maps = dict[str, Map]
SEEDS = {"copy": (0, 1, 2), "lookup": (0, 1, 2), "ident32": (0, 1), "ident32b": (0, 1), "rand32": (0, 1), "rand32b": (0, 1)}
E62, E46 = "induction_cue", "regating_table"


def large(folder: str) -> Path:
    return EXP / folder / "results" / "large"


def state(name: str) -> DW:
    """A state this probe reads: E62's or E46's."""
    return torch.load(large(E62 if name.split("_")[0] in ("copy", "lookup") else E46) / f"{name}.pt", weights_only=True)


def filed_json(folder: str, file: str) -> Rec:
    return json.loads((EXP / folder / "results" / file).read_text())


def dump(out: Rec) -> None:
    OUT.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))


def note(out: Rec, key: str, rec: Rec) -> None:
    out[key] = rec
    dump(out)
    log(f"[{key}] {json.dumps(jsonable(rec))}")


def stop(out: Rec, why: str) -> NoReturn:
    out["stopped"] = why
    dump(out)
    sys.exit(f"STOP: {why}: tell the lead")


def e62_read() -> Rec:
    return torch.load(large(E62) / "windows.pt", weights_only=False)


def cue_maps(net: ST.Net, rows: Tensor, run: Rec, S: Rec, th: Rec) -> list[Map]:
    """The copy map and the lookup on one pool from `copying.cue_windows` (finding 6). B's pool: the Bs of E62's read windows."""
    half, p = S["half"], run["pools"]
    ks, bs = torch.tensor(p["keys"]), e62_read()["target"]["copy"]["read"].unique().tolist()
    w, b, at = CP.cue_windows(rows, p["marker"][0], p["keys"], bs, half, S["per_key"] * len(ks), S["seed"])
    word = torch.full((CP.VOCAB,), -1)
    word[ks] = torch.tensor([p["words"][str(k)] for k in p["keys"]])
    next_word = word.clone()
    next_word[ks] = word[ks.roll(-1)]
    toks, keys, n = w.tokens, w.tokens[:, half - 1], S["copy_pairs"]
    gone = CP.absent_b(toks[:n], at[:n], b[:n], bs, S["seed"])
    return [X.make_map(net, "copy", toks, keys, b, next_word[keys], X.Pairs(toks[:n], gone, b[:n], None), S, th),
            X.make_map(net, "lookup", toks, keys, word[keys], next_word[keys], None, S, th)]


def key_map(net: ST.Net, rows: Tensor, name: str, S: Rec, th: Rec) -> Map:
    w = torch.load(large(E46) / f"windows_{name}.pt", weights_only=False)
    task, table, half = w["task"], w["table"], S["half"]
    toks = kv.windows(rows, task.marker, task.keys, half, S["per_key"] * len(task.keys), S["seed"]).tokens
    ks, keys = torch.tensor(task.keys), toks[:, half]
    next_value = table.clone()
    next_value[ks] = table[ks.roll(-1)]
    return X.make_map(net, name, toks, keys, table[keys], next_value[keys], None, S, th)


def filed_steering() -> dict[str, dict[int, float]]:
    """The eight E46 states' per-block steering share on the scoring half (E52, in log P). E62's six have none filed."""
    b3 = filed_json("baseline_neurons", "baseline_neurons.json")["b3"]
    return {n: {l: v["layers"][str(l)]["score"] for l in X.BLOCKS} for n, v in b3.items()}


def fix_K(net: ST.Net, rows: Tensor, S: Rec, th: Rec, rec: Rec) -> tuple[int | None, Tensor]:
    """K on the half-known map, which W0 performs (finding 8), and W0's neuron means on plain Pile spans."""
    half = S["half"]
    means = X.neuron_stats(net, CP.natural_spans(rows, half, S["mean_spans"], S["seed"]))[0]
    w = torch.load(large("keydependent_elicit") / "windows_elicit.pt", weights_only=False)
    toks = w["heldout"].tokens
    tgt = w["table"][toks[:, half]]
    order, base = X.top_abs(X.neuron_scores(net, toks, tgt, half), S["k_grid"][-1]), X.read(net, toks, tgt, half)
    rec["K_falls"] = {k: float((base - X.read(net, toks, tgt, half, None, X.ablate(order[:k], means))).mean()) for k in S["k_grid"]}
    rec["K"] = X.first_k(rec["K_falls"], th["k_fall"])
    return rec["K"], means


def stage0(net: ST.Net, maps: Maps, rows: Tensor, S: Rec, th: Rec, out: Rec) -> tuple[int | None, Tensor]:
    half = S["half"]
    sd = lambda r: float(r[X.boot_index(len(r), th["n_boot"], S["seed"])].mean(1).std())
    rec: Rec = {"maps": {}}
    for m in maps.values():
        rec["maps"][m.name] = {"lstar": m.lstar, "pairs": len(m.on.pairs.tgt), "pairs_off": len(m.off.pairs.tgt), "keys": len(m.keys.unique()),
                               "R0": {l: [float(m.on.R0[l].mean()), sd(m.on.R0[l])] for l in X.BLOCKS},
                               "R0_off": {l: [float(m.off.R0[l].mean()), sd(m.off.R0[l])] for l in X.BLOCKS},
                               "target_in_prefix_hi_lo": X.prefix_share(m.on, half),
                               "transplant_max": X.nan_max([X.transplant_max(net, m.on, half), X.transplant_max(net, m.off, half)])}
    r = rec["maps"]
    rec["status"] = X.stage0_status(X.nan_max([v["transplant_max"] for v in r.values()]), {n: v["lstar"] for n, v in r.items()}, th)
    note(out, "stage0", rec)  # before K, so a stop reads the patch's own numbers
    if rec["status"] != "pass":
        stop(out, f"stage 0: {rec['status']}")
    K, means = fix_K(net, rows, S, th, rec)
    note(out, "stage0", rec)
    return K, means


def controls(net: ST.Net, m: Map, K: int | None, sharp: DW, means: Tensor, S: Rec, th: Rec) -> Rec:
    """The map's gain target is its saved states' mean; only that gain is read from them here."""
    Gt = sum(X.gain_under(net, m, state(f"{m.name}_s{s}")) for s in SEEDS[m.name]) / len(SEEDS[m.name])
    built: dict[str, tuple[DW, Rec]] = {"sharpen": (sharp, {"norm": EL.summed_norm(sharp), "G": X.gain_under(net, m, sharp)})}
    if K is not None:
        built["amplify"] = X.amplify_state(net, m, K, Gt, S["amp_cap"], S["match_tol"])
    if m.name != "copy":
        built["write"] = X.write_state(net, m, Gt, S["match_tol"])
    return {n: {"info": i, "row": X.patch_row(net, m, d, th, S["seed"]), "units": X.units(net, m, K, means, d, th) if n != "sharpen" else {"void": True}}
            for n, (d, i) in built.items()}


def stage1(net: ST.Net, maps: Maps, rows: Tensor, K: int | None, means: Tensor, S: Rec, th: Rec, out: Rec) -> None:
    sh, half = S["sharpen"], S["half"]
    spans = CP.natural_spans(rows, half, sh["n_spans"], sh["seed"])
    no = torch.zeros(len(spans))
    sharp = X.sharpened(net, Windows(spans, no, no.long(), no.long(), half), sh, log)
    rec: Rec = {"sharpen_norm": EL.summed_norm(sharp), "maps": {}}
    for m in maps.values():
        if K is not None:
            X.neuron_sets(net, m, K, S)
        rec["maps"][m.name] = {"lstar": m.lstar, "controls": controls(net, m, K, sharp, means, S, th)} if m.lstar else {"lstar": None, "controls": {}}
        log(f"[stage1 {m.name}] {json.dumps(jsonable(rec['maps'][m.name]))}")
        dump(out | {"stage1": rec})
    for n, v in rec["maps"].items():
        write = rec["maps"]["lookup" if n == "copy" else n]["controls"].get("write")
        if write and "amplify" in v["controls"]:
            v["verdicts"] = X.control_verdicts(v["controls"], write, v["lstar"], th)
    rec["status"] = X.stage1_status({n: v["verdicts"]["amplify_T"] for n, v in rec["maps"].items() if "verdicts" in v})
    note(out, "stage1", rec)
    if rec["status"] != "pass":
        stop(out, "stage 1: amplify's T does not clear the write's on half the readable maps or more, or no map is readable; see verdicts")


def reproduce(net: ST.Net, name: str, m: Map, run: Rec, e52: Rec, half: int) -> tuple[float, float]:
    """(G re-read, |difference| from its filed value): E62's in log-odds on its read windows, E46's in log P on theirs."""
    if m.name in ("copy", "lookup"):
        v = e62_read()
        toks, tgt = v["read"].tokens, v["target"][m.name]["read"]
        G = float((X.read(net, toks, tgt, half, state(name)) - X.read(net, toks, tgt, half)).mean())
        return G, abs(G - run["states"][name]["G"])
    t = ST.load_task(net, large(E46) / f"windows_{m.name}.pt", m.name, "kv")
    G = ST.gains(net, t, state(name))["G"]
    return G, abs(G - e52["gains"][name]["G"])


def read_map(net: ST.Net, m: Map, K: int | None, means: Tensor, run: Rec, S: Rec, th: Rec, out: Rec, key: str) -> None:
    half, filed = S["half"], filed_steering()
    e52 = filed_json("baseline_neurons", "baseline_neurons.json")
    for s in SEEDS[m.name]:
        name = f"{m.name}_s{s}"
        G, diff = reproduce(net, name, m, run, e52, half)
        if diff > th["g_tol"]:
            stop(out, f"{key}: {name} does not reproduce its filed gain: {G} ({diff} off)")
        dW, shares = state(name), filed.get(name)
        if shares is None:
            v = e62_read()
            shares = X.steer_shares(net, v["read"].tokens, v["target"][m.name]["read"], half, dW)
        row = X.patch_row(net, m, dW, th, S["seed"])
        at = row.get(m.lstar or 0, {})
        rec = {"exploratory": True, "G": G, "G_diff": diff, "norm": EL.summed_norm(dW), "lstar": m.lstar, "steering": shares, "steering_measured": m.name in ("copy", "lookup"),
               "steering_at_lstar": shares.get(m.lstar or 0), "mark": X.blind_mark(shares, m.lstar, th), "T": at.get("T"), "Q": at.get("Q"), "row": row, "units": X.units(net, m, K, means, dW, th)}
        out.setdefault(key, {})[name] = rec
        dump(out)
        log(f"[{key} {name}] {json.dumps(jsonable(rec))}")


def stage4(net: ST.Net, maps: Maps, S: Rec, th: Rec, out: Rec) -> None:
    """E62 Amendment 2's swaps: each state scaled to the other map's size at its seed (a multiplication; no training)."""
    a2 = filed_json(E62, "amend2.json")["row7_size"]["states"]
    states = {**out["stage2"], **out["stage3"]}
    rec: Rec = {"exploratory": True, "scaled": {}}
    for name, st in a2.items():
        m = maps[name.split("_")[0]]
        dW = {k: st["alpha"] * v for k, v in state(name).items()}
        row = X.patch_row(net, m, dW, th, S["seed"])
        at = row.get(m.lstar or 0, {})
        rec["scaled"][name] = {"alpha": st["alpha"], "norm": EL.summed_norm(dW), "T": at.get("T"), "Q": at.get("Q"), "T_unscaled": states[name]["T"], "Q_unscaled": states[name]["Q"], "row": row}
    note(out, "stage4", rec)


def load() -> tuple[Rec, Rec, Rec]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    if OUT.exists():
        sys.exit(f"{OUT} exists: move it aside, this run deletes nothing")
    return json.loads(PRED.read_text()), filed_json(E62, "run.json"), {"git_head": git("rev-parse", "HEAD").stdout.strip(), "predictions_sha256": KD.sha256(PRED)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    P, run, out = load()
    S, th = P["settings"], P["thresholds"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net(a.device)
    rows = pile_rows("train", 0, AN.ROWS, log)
    held = torch.arange(len(rows)) % S["heldout_every"] == 0
    maps = {m.name: m for m in [*cue_maps(net, rows[held], run, S, th), *(key_map(net, rows[held], n, S, th) for n in list(SEEDS)[2:])]}
    K, means = stage0(net, maps, rows[held], S, th, out)
    stage1(net, maps, rows[~held], K, means, S, th, out)
    for key, names in (("stage2", ("copy", "lookup")), ("stage3", list(SEEDS)[2:])):
        for n in names:
            read_map(net, maps[n], K, means, run, S, th, out, key)
    stage4(net, maps, S, th, out)


if __name__ == "__main__":
    main()

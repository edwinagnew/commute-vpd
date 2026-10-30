"""open_pair (E58): E54's elicit map and its matched cipher fine-tuned on 24 of their 32 keys (E54's recipe, seeds 0-2),
each state labelled by behaviour alone (does the fine-tune carry to the 8 keys it never saw?) with the reuse line's
bounded-g rule, then read by the route split's r, the stand-in ratio and steering's S on the trained keys.
Plan: reports/plans/2026-10-30_plan_e58-open-pair.md. Every setting, rule and threshold: results/predictions.json
(filed and committed before the run; the run refuses otherwise; never edited).

  stage 0, alone (exit 1 on a miss): sha256 of E54's windows files and elicit_s0.pt; the key split; the zero update
           reads G = H = 0 exactly on all four split tasks; E54's elicit_s0 on E54's own 32-key task reproduces r,
           the stand-in ratio and S, and block 0 reads 0.
  stage 1: per seed, elicit then the cipher at the elicit update's norm (E54's `keydep` fine-tunes on the trained keys'
           windows); after each, G, H and the label. Exit 1 if elicit_s0's G is under the bar, or if any cipher
           reads "reuses" (the label's negative control).
  stage 2: r, ratio, S on every state (trained keys' held-out windows); exit 1 on a block-0 shift over 1e-5 nats.
  score:   `open_pair.score` -> results/run.json. Every scored number is also printed.
Writes <large>/windows_{elicit,cipher}_{trained,new}.pt and <large>/{elicit,cipher}_s{0,1,2}.pt (~113 MB each).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import anchors as AN
from commute import keydep as KD
from commute import open_pair as OP
from commute import steering as ST
from commute.data import Windows
from commute.identity import log_odds
from commute.kv import KVTask

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
E54 = HERE.parent / "keydependent_elicit" / "results"
KINDS, PARTS = ("elicit", "cipher"), ("trained", "new")
log = KD.log
Tasks = dict[str, dict[str, ST.Task]]


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    return [_finite(v) for v in o] if isinstance(o, list | tuple) else o


def dump(out: dict[str, Any]) -> None:
    (RESULTS / "run.json").write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout.strip()


def stop(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    sys.exit(f"STOP: {why}: tell the user")


# ----------------------------------------------------------------------------- the split tasks


def keyed(w: Windows, keys: list[int]) -> Windows:
    return w.subset(torch.isin(w.tokens[:, w.half], torch.tensor(keys)))


def split_task(src: Path, trained: list[int], held: list[int], large: Path, name: str) -> dict[str, str]:
    """E54's windows file cut by key: `<name>_trained` = the trained keys' train and held-out windows and their table;
    `<name>_new` = every window of the held-out keys (both of E54's row splits; none is ever trained on here) and their
    table. Saved; their sha256."""
    w = torch.load(src, weights_only=False)
    task, V = w["task"], int(w["table"].shape[0])
    val = dict(zip(task.keys, task.values, strict=True))
    sub = {p: KVTask(f"{name}_{p}", task.marker, tuple(ks), tuple(val[k] for k in ks)) for p, ks in (("trained", trained), ("new", held))}
    tr, nw = keyed(w["train"], held), keyed(w["heldout"], held)
    files = {"trained": {"train": keyed(w["train"], trained), "heldout": keyed(w["heldout"], trained)},
             "new": {"heldout": Windows(torch.cat([tr.tokens, nw.tokens]), torch.cat([tr.g, nw.g]), torch.cat([tr.row, nw.row]),
                                        torch.cat([tr.pos, nw.pos]), tr.half)}}
    assert len(files["trained"]["train"]) + len(tr) == len(w["train"]) and len(files["trained"]["heldout"]) + len(nw) == len(w["heldout"])
    sha = {}
    for p, f in files.items():
        path = large / f"windows_{name}_{p}.pt"
        torch.save({**f, "task": sub[p], "table": sub[p].table(V)}, path)
        sha[p] = KD.sha256(path)
    return sha


# ----------------------------------------------------------------------------- the label (no decomposition)


def gain(net: ST.Net, t: ST.Task, dW: ST.DW, readout: ST.Readout = ST.logodds_at) -> Tensor:
    """Per window, the fine-tuned model's log-odds of the target minus W0's (target model only)."""
    return ST.read(net, t, t.toks, dW, readout=readout) - ST.read(net, t, t.toks, readout=readout)


def others(values: list[int]) -> ST.Readout:
    """Per window, the mean log-odds of the held-out values other than the window's own target (the prior-shift check)."""
    vals = torch.tensor(values)

    def f(logits: Tensor, pos: int, tgt: Tensor) -> Tensor:
        lo = torch.stack([log_odds(logits[:, pos], v) for v in values], 1).double().cpu()
        keep = (vals[None, :] != tgt.cpu()[:, None]).double()
        return (lo * keep).sum(1) / keep.sum(1)

    return f


def behaviour(net: ST.Net, t: dict[str, ST.Task], dW: ST.DW, th: dict[str, Any]) -> dict[str, Any]:
    """G on the trained keys' held-out windows, H on the held-out keys' windows, the bootstrap sd of H / G, the label;
    reported beside: H_other (the other held-out values' gain at the same windows) and H per held-out key."""
    dG, dH = gain(net, t["trained"], dW), gain(net, t["new"], dW)
    new = t["new"]
    dO = gain(net, new, dW, others([int(new.table[k]) for k in new.keys]))
    G, H, s = float(dG.mean()), float(dH.mean()), OP.boot_sd(dH, dG, th["n_boot"], th["boot_seed"])
    at = new.toks[:, new.half]
    return {"G": G, "H": H, "x": H / G if G != 0 else math.nan, "s": s, "label": OP.label(H, G, s, th), "H_other": float(dO.mean()),
            "n_G": len(dG), "n_H": len(dH), "H_per_key": {str(k): float(dH[at == k].mean()) for k in new.keys}}


def say_label(n: str, b: dict[str, Any]) -> None:
    log(f"[label {n}] G {b['G']:.4f} H {b['H']:.4f} x {b['x']:.4f} 3s {3 * b['s']:.4f} -> {b['label']} | H_other {b['H_other']:.4f} "
        f"(n_G {b['n_G']}, n_H {b['n_H']}) per key {json.dumps({k: round(v, 3) for k, v in b['H_per_key'].items()})}")


# ----------------------------------------------------------------------------- the readings (E54's, unchanged)


def vpd_read(ctx: KD.Ctx, t: ST.Task, dW: ST.DW, base: dict[str, Any]) -> dict[str, Any]:
    """E54's `readings` (S, r and the route cells), plus the stand-in ratio from the same cells: (L11 - L00) / (U1 - U0),
    nan when the subcomponent gain U1 - U0 is under E53's floor."""
    g = ST.gains(ctx.net, t, dW)
    rd = KD.readings(ctx, t, dW, g, base)
    c, floor = rd["cells"], ctx.P["thresholds"]["ratio_floor"]
    piece = c["U1"] - c["U0"]
    return {"r": rd["r"], "S": rd["S"], "layer": rd["layer"], "S_block0": rd["S_block0"], "G_score_logP": g["G_score"], "G_logodds_route": rd["G_logodds"],
            "cells": c, "piece_gain": piece, "ratio": (c["L11"] - c["L00"]) / piece if piece >= floor else math.nan}


# ----------------------------------------------------------------------------- stages


def stage0(ctx: KD.Ctx, tasks: Tasks, out: dict[str, Any]) -> None:
    c = ctx.P["stage0"]
    zero = {k: torch.zeros_like(w) for k, w in ctx.net.W0.items()}
    rec: dict[str, Any] = {"zero_max_abs": {f"{n}_{p}": float(gain(ctx.net, tasks[n][p], zero).abs().max()) for n in KINDS for p in PARTS}}
    t = ST.load_task(ctx.net, E54 / "large" / "windows_elicit.pt", "elicit", "elicit")
    rd = vpd_read(ctx, t, torch.load(E54 / "large" / "elicit_s0.pt", weights_only=True), KD.base_of(ctx, t))
    rec["e54_elicit_s0"] = {k: rd[k] for k in ("r", "ratio", "S", "S_block0", "layer")}
    rec["checks"] = {"zero": all(v == 0.0 for v in rec["zero_max_abs"].values()),
                     "r": abs(rd["r"] - c["r_ref"]) <= c["r_tol"], "ratio": abs(rd["ratio"] - c["ratio_ref"]) <= c["ratio_tol"],
                     "S": abs(rd["S"] - c["S_ref"]) <= c["S_tol"], "block0": abs(rd["S_block0"]) <= c["block0_tol"]}
    rec["pass"] = all(rec["checks"].values())
    out["stage0"] = rec
    log(f"[stage0] zero-update max |gain| {rec['zero_max_abs']} | E54 elicit_s0 r {rd['r']:.4f} (ref {c['r_ref']:.4f}) ratio "
        f"{rd['ratio']:.4f} (ref {c['ratio_ref']:.4f}) S {rd['S']:.4f} (ref {c['S_ref']:.4f}) S_block0 {rd['S_block0']:.1e} "
        f"checks {rec['checks']}")
    dump(out)
    if not rec["pass"]:
        stop(out, "stage 0 missed")


def stage1(ctx: KD.Ctx, tasks: Tasks, out: dict[str, Any]) -> dict[str, ST.DW]:
    th, seeds, st = ctx.P["thresholds"], ctx.P["settings"]["seeds"], out["states"]
    ws = {n: torch.load(ctx.large / f"windows_{n}_trained.pt", weights_only=False) for n in KINDS}
    dws: dict[str, ST.DW] = {}

    def keep(n: str, dW: ST.DW, meta: dict[str, Any]) -> None:
        dws[n] = dW
        torch.save(dW, ctx.large / f"{n}.pt")
        st[n] = {"ft": {**meta, "sha256": KD.sha256(ctx.large / f"{n}.pt")}, **behaviour(ctx.net, tasks[n.split("_")[0]], dW, th)}
        log(f"[ft {n}] step {meta['step']} ||dW|| {meta['norm']:.4f}" + (f" at_bound {meta['at_bound']}" if "at_bound" in meta else ""))
        say_label(n, st[n])
        dump(out)

    for s in seeds:
        e, c = f"elicit_s{s}", f"cipher_s{s}"
        dW, meta = KD.elicit_ft(ctx, ws["elicit"], s)
        keep(e, dW, meta)
        if s == seeds[0] and not st[e]["G"] >= th["G_min"]:
            stop(out, f"{e} G {st[e]['G']:.3f} < {th['G_min']}: E54's recipe on 24 keys is too small to label (plan: raise the step once, as an amendment)")
        dW, _, meta_c, _ = KD.cipher_ft(ctx, ws["cipher"], s, meta["norm"])
        keep(c, dW, meta_c)
        if st[c]["label"] == "reuses":
            stop(out, f"{c} reads 'reuses': the label's negative control failed, the rule is broken")
    return dws


def stage2(ctx: KD.Ctx, tasks: Tasks, dws: dict[str, ST.DW], out: dict[str, Any]) -> None:
    bases = {n: KD.base_of(ctx, tasks[n]["trained"]) for n in KINDS}
    for n, dW in dws.items():
        kind = n.split("_")[0]
        st = out["states"][n]
        st.update(vpd_read(ctx, tasks[kind]["trained"], dW, bases[kind]))
        log(f"[read {n}] r {st['r']:.4f} ratio {st['ratio']:.4f} S {st['S']:.4f} (block {st['layer']}, block-0 {st['S_block0']:.1e}) "
            f"G_logodds_route {st['G_logodds_route']:.4f} vs G {st['G']:.4f}")
        dump(out)
        if abs(st["S_block0"]) * abs(st["G_score_logP"]) > 1e-5:  # E54's stop, in nats
            stop(out, f"{n}: block-0 shift over 1e-5 nats")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="where this run's windows and states go")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    rel = str(PRED.relative_to(REPO))
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode or not git("ls-files", rel):
        sys.exit(f"{rel} is not committed unchanged: commit it before the run")
    P = json.loads(PRED.read_text())
    S = P["settings"]
    assert S["half"] == AN.HALF and S["route_batch"] == KD.BATCH, "predictions.json settings disagree with the code"
    large = a.large.resolve()
    large.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "predictions_sha256": KD.sha256(PRED), "states": {}}
    sha = {n: KD.sha256(E54 / "large" / f"{n}.pt") for n in ("windows_elicit", "windows_cipher", "elicit_s0")}
    out["e54_sha256"] = sha
    if sha != P["stage0"]["sha256"]:
        stop(out, f"an E54 file is not the one its run recorded: {sha}")
    keys = json.loads((E54 / "stage1.json").read_text())["elicit"]["keys"]
    trained, held = OP.split_keys(keys, S["n_held"], S["split_seed"])
    assert (trained, held) == (S["trained_keys"], S["held_keys"]), "the key split is not the filed one"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, large)
    out["windows_sha256"] = {n: split_task(E54 / "large" / f"windows_{n}.pt", trained, held, large, n) for n in KINDS}
    tasks = {n: {p: ST.load_task(ctx.net, large / f"windows_{n}_{p}.pt", f"{n}_{p}", n) for p in PARTS} for n in KINDS}
    out["n_windows"] = {f"{n}_{p}": len(tasks[n][p].toks) for n in KINDS for p in PARTS}
    log(f"[split] trained {trained} held {held} windows {out['n_windows']}")
    stage0(ctx, tasks, out)
    dws = stage1(ctx, tasks, out)
    stage2(ctx, tasks, dws, out)
    out["score"] = OP.score(out["states"], P["thresholds"], S["seeds"])
    dump(out)
    log(f"[score] {json.dumps(_finite(out['score']))}")


if __name__ == "__main__":
    main()

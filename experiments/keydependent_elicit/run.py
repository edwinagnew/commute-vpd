"""keydependent_elicit (E54): a key -> value map W0 already half-knows (each key's value is W0's rank 2-20 guess) and a
random cipher on the same keys, each fine-tuned with E12's recipe; read by E52's steering share S and by the route
split's re-gating share r of the least-squares split state. Plan: reports/plans/2026-10-29_plan_e54-keydependent-elicit.md.
Every setting, rule and threshold: results/predictions.json (filed before the run, read here, never edited).

  run.py --stage 01  stage 0, the controls, alone: E46's ident32_s0 and rand32_s0 must give E46's r and E52's S; exit 1
                     on a miss. Then stage 1: W0's value per candidate key, the key draw, the elicit and cipher tables
                     and W0's held-out rates; exit 1 on a miss. -> results/stage0.json, results/stage1.json,
                     <large>/windows_{elicit,cipher}.pt. The sbatch commits both JSON files before job 2.
  run.py --stage 2   refuses unless stages 0 and 1 passed, stage1.json is committed unchanged and the windows files are
                     the ones it recorded. Fine-tunes (elicit before cipher), the gain floor, then S and r, then the
                     filed rules -> results/stage2.json, <large>/{elicit,cipher,cipher_own}_s{0,1}.pt.
Every scored number is also printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import anchors as AN
from commute import elicit as EL
from commute import steering as ST
from commute.faithfulness import model_state

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
TASKS = ("elicit", "cipher")
SEEDS = (0, 1)
E12_CHECKS = (10, 30, 100, 300)
P_LEARN, BATCH = 0.5, 64


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    return [_finite(v) for v in o] if isinstance(o, list | tuple) else o


def dump(name: str, obj: dict[str, Any]) -> None:
    (RESULTS / name).write_text(json.dumps(_finite(obj), indent=1, sort_keys=True))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)


def geo_rate(lp: Tensor) -> float:
    """Geometric mean over windows of P(target): exp(mean log P)."""
    return math.exp(float(lp.mean()))


@dataclass
class Ctx:
    net: ST.Net
    lift: dict[str, Any]
    comps0: dict[str, tuple[Tensor, Tensor]]
    P: dict[str, Any]
    large: Path


def make_ctx(a: argparse.Namespace, P: dict[str, Any]) -> Ctx:
    from commute import vpd

    net = ST.Net(a.device)
    alive = torch.load(a.alive, weights_only=False)["alive"]
    lift = AN.lift_context(net.model, {k: v.cpu().bool() for k, v in alive.items()})
    return Ctx(net, lift, vpd.component_weights(net.model), P, a.large)


# ----------------------------------------------------------------------------- the two readings


def base_of(ctx: Ctx, t: ST.Task) -> dict[str, Any]:
    from commute.local_faith import route_cells

    return route_cells(ctx.net.model, t.toks, t.half, t.table, BATCH)


def route_r(ctx: Ctx, t: ST.Task, dW: ST.DW, base: dict[str, Any]) -> dict[str, Any]:
    """r of the least-squares split state: target W0 + delta-W, components lifted; r = (L01 - L00) / (T1 - T0)."""
    from commute.local_faith import route_split

    m = ctx.net.model
    target = {k: w + dW[k].to(w.device, w.dtype) for k, w in ctx.net.W0.items()}
    with model_state(m, target=target, components=AN.lifted(m, ctx.lift, dW)):
        rs = route_split(m, t.toks, t.half, t.table, base, ctx.comps0, BATCH)
    return {"r": rs["shares"]["r"], "G_logodds": rs["gain"], "cells": rs["cells"]}


def readings(ctx: Ctx, t: ST.Task, dW: ST.DW, g: dict[str, float], base: dict[str, Any]) -> dict[str, Any]:
    b = ST.b3(ctx.net, t, dW, g)
    return {"S": b["S"], "layer": b["layer"], "S_block0": b["layers"][0]["score"], "b3_layers": b["layers"],
            **route_r(ctx, t, dW, base)}


# ----------------------------------------------------------------------------- stage 0


def stage0(ctx: Ctx, e46: Path) -> dict[str, Any]:
    P0: dict[str, Any] = ctx.P["stage0"]
    out: dict[str, Any] = {"states": {}}
    for name, kind in (("ident32", "copy"), ("rand32", "cipher")):
        key, path = f"{name}_s0", e46 / f"{name}_s0.pt"
        t = ST.load_task(ctx.net, e46 / f"windows_{name}.pt", name, kind)
        dW = torch.load(path, weights_only=True)
        g = ST.gains(ctx.net, t, dW)
        rec = {"sha256": sha256(path), **g, "G_zero_dW": ST.gains(ctx.net, t, {k: torch.zeros_like(d) for k, d in dW.items()})["G"],
               "w0_rate": geo_rate(t.lp0), **readings(ctx, t, dW, g, base_of(ctx, t))}
        rec["ok"] = bool(rec["sha256"] == P0["sha256"][key] and rec["G_zero_dW"] == 0.0
                         and abs(rec["S_block0"]) <= P0["block0_tol"]
                         and abs(rec["r"] - P0["r_ref"][key]) <= P0["r_tol"] and abs(rec["S"] - P0["S_ref"][key]) <= P0["S_tol"])
        out["states"][key], out["marker"] = rec, t.marker
        log(f"[stage0 {key}] r {rec['r']:.4f} (ref {P0['r_ref'][key]:.4f}) S {rec['S']:.4f} (ref {P0['S_ref'][key]:.4f}) "
            f"S_block0 {rec['S_block0']:.2e} G {rec['G']:.4f} G_zero_dW {rec['G_zero_dW']} w0_rate {rec['w0_rate']:.3e} ok {rec['ok']}")
    out["pass"] = all(r["ok"] for r in out["states"].values())
    return out


# ----------------------------------------------------------------------------- stage 1


def vocab(ctx: Ctx) -> tuple[Tensor, Tensor, int, list[int], Any]:
    """E12's Pile rows, word-like mask, marker and frequency-ranked word-like tokens (as anchors.build_kv builds them)."""
    from commute import kv, ladder, vpd
    from commute.data import pile_rows

    tok = vpd.tokenizer()
    rows = pile_rows("train", 0, AN.ROWS, log)
    V = int(ctx.net.net.lm_head.weight.shape[0])
    plain = ladder.plain_mask(tok.decode, V)
    counts = ladder.count_tokens(rows, AN.HALF)
    marker = kv.pick_marker(counts, tok.decode, V)
    ranked = [t for t, _ in sorted(counts.items(), key=lambda tc: -tc[1]) if t < V and bool(plain[t])]
    return rows, plain, marker, ranked, tok


def select(ctx: Ctx, rows: Tensor, plain: Tensor, marker: int, cands: list[int]) -> tuple[list[int], dict[int, Any]]:
    """W0's value per candidate key on 16 training-row windows each, and the seeded draw of K keys."""
    from commute import kv

    S = ctx.P["settings"]
    train_rows = rows[torch.arange(rows.shape[0]) % 5 != 0]
    w = kv.windows(train_rows, marker, tuple(cands), AN.HALF, S["selection_windows_per_key"] * len(cands), S["selection_seed"])
    dist = kv.key_distributions(lambda x: ctx.net.run(x)[0], w, ctx.net.device)
    D = torch.stack([dist.pop(c) for c in cands])
    qual = EL.pick_values(D, cands, plain, tuple(S["value_band"]), tuple(S["value_ranks"]), S["key_specific_ratio"])
    return EL.draw_keys(qual, cands, S["k"], S["key_draw_seed"]), qual


def build_tasks(ctx: Ctx, rows: Tensor, marker: int, keys: list[int], values: list[int], ranked: list[int]) -> dict[str, str]:
    """E12's 500 windows on the keys, the elicit table and the cipher (E12's pool and gate); saved; their sha256."""
    from commute import kv
    from commute.data import split_windows
    from commute.finetune import logits_of

    S, V = ctx.P["settings"], int(ctx.net.net.lm_head.weight.shape[0])
    w = kv.windows(rows, marker, tuple(keys), AN.HALF, S["task_windows"], S["task_windows_seed"])
    train, held = split_windows(w)
    dist = kv.key_distributions(lambda x: logits_of(ctx.net.net, x), w, ctx.net.device)
    pool = ranked[AN.VALUE_RANK0 : AN.VALUE_RANK0 + AN.POOL]
    tasks = {"elicit": kv.KVTask("elicit", marker, tuple(keys), tuple(values)),
             "cipher": kv.bijection("cipher", marker, tuple(keys), pool, seed=0, ok=lambda k, v: float(dist[k][v]) < AN.GATE)}
    sha = {}
    for n, task in tasks.items():
        path = ctx.large / f"windows_{n}.pt"
        torch.save({"train": train, "heldout": held, "task": task, "table": task.table(V)}, path)
        sha[n] = sha256(path)
    return sha


def stage1(ctx: Ctx, marker0: int) -> dict[str, Any]:
    S, P1 = ctx.P["settings"], ctx.P["stage1"]
    rows, plain, marker, ranked, tok = vocab(ctx)
    assert marker == marker0, f"marker {marker} is not E12's {marker0}"
    cands = ranked[S["key_ranks"][0] : S["key_ranks"][1]]
    keys, qual = select(ctx, rows, plain, marker, cands)
    out: dict[str, Any] = {"marker": marker, "n_candidates": len(cands), "n_qualified": len(qual), "K": len(keys),
                           "per_key": {str(x): {**qual[x], "key_text": tok.decode([x]), "value_text": tok.decode([qual[x]["value"]])}
                                       for x in keys}}
    log(f"[stage1] candidates {len(cands)} qualified {len(qual)} K {len(keys)}")
    if len(keys) < S["k_min"]:
        return {**out, "pass": False, "why": f"K = {len(keys)} < {S['k_min']}"}
    out["sha256"] = build_tasks(ctx, rows, marker, keys, [qual[x]["value"] for x in keys], ranked)
    for n in TASKS:
        t = ST.load_task(ctx.net, ctx.large / f"windows_{n}.pt", n, n)
        out[n] = {"keys": list(t.keys), "values": [int(t.table[x]) for x in t.keys], "n_heldout": len(t.toks), "rate": geo_rate(t.lp0)}
        log(f"[stage1 {n}] W0 held-out geo-mean P(target) {out[n]['rate']:.3e} over {len(t.toks)} windows")
    lo, hi = P1["elicit_rate_band"]
    out["pass"] = bool(lo <= out["elicit"]["rate"] <= hi and out["cipher"]["rate"] < P1["cipher_rate_max"])
    return out


# ----------------------------------------------------------------------------- stage 2


def finetune(ctx: Ctx, w: dict[str, Any], seed: int, checks: tuple[int, ...]) -> Any:
    from commute.finetune import ArmAConfig, finetune_arm_a

    cfg = ArmAConfig(o_token=w["table"], lr=AN.FT_LR, steps=E12_CHECKS[-1], checkpoints=checks, seed=seed)
    return finetune_arm_a(ctx.net.model, w["train"], w["heldout"], cfg, ctx.net.device, log)


def scored_step(evals: dict[int, dict[str, float]]) -> tuple[int, bool]:
    """E12's rule: the first of 10, 30, 100, 300 with held-out P(target) >= 0.5, else 300 ('did not learn')."""
    return next(((s, True) for s in E12_CHECKS if evals[s]["p_o"] >= P_LEARN), (E12_CHECKS[-1], False))


def finetunes(ctx: Ctx, ws: dict[str, dict[str, Any]], out: dict[str, Any]) -> dict[str, ST.DW]:
    """Elicit at E12's step, then the cipher at the norm-matched step in 1..30 and at its own E12 step, per seed."""
    dws, meta = {}, out["finetunes"]
    for s in SEEDS:
        ft = finetune(ctx, ws["elicit"], s, E12_CHECKS)
        step, learned = scored_step(ft.evals)
        dws[f"elicit_s{s}"] = ft.deltas[step]
        meta[f"elicit_s{s}"] = {"step": step, "learned": learned, "norm": EL.summed_norm(ft.deltas[step]),
                                "p_heldout": {k: e["p_o"] for k, e in ft.evals.items()}}
        dump("stage2.json", out)
    cipher_checks = (*range(1, 31), 100, 300)
    for s in SEEDS:
        ft = finetune(ctx, ws["cipher"], s, cipher_checks)
        norms = {k: EL.summed_norm(ft.deltas[k]) for k in range(1, 31)}
        m = EL.matched_step(norms, meta[f"elicit_s{s}"]["norm"])
        own, learned = scored_step(ft.evals)
        dws[f"cipher_s{s}"], dws[f"cipher_own_s{s}"] = ft.deltas[m], ft.deltas[own]
        meta[f"cipher_s{s}"] = {"step": m, "norm": norms[m], "at_bound": m == 30, "same_state": m == own, "norms": norms,
                                "p_heldout": {k: e["p_o"] for k, e in ft.evals.items()}}
        meta[f"cipher_own_s{s}"] = {"step": own, "learned": learned, "norm": EL.summed_norm(ft.deltas[own])}
        del ft
        dump("stage2.json", out)
    for n in dws:
        log(f"[ft {n}] step {meta[n]['step']} ||dW|| {meta[n]['norm']:.4f}" + (f" at_bound {meta[n]['at_bound']}" if "at_bound" in meta[n] else ""))
    return dws


def check_filed(s1_path: Path, large: Path) -> dict[str, Any]:
    s1 = json.loads(s1_path.read_text())
    rel = str(s1_path.relative_to(REPO))
    if not s1.get("pass") or not json.loads((RESULTS / "stage0.json").read_text()).get("pass"):
        sys.exit("stage 0 or 1 has not passed: stop and tell the user")
    if git("ls-files", "--error-unmatch", rel).returncode or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: commit stage 1 before any fine-tune")
    if any(sha256(large / f"windows_{n}.pt") != s1["sha256"][n] for n in TASKS):
        sys.exit("a windows file is not the one stage 1 recorded")
    return s1


def stage2(ctx: Ctx) -> None:
    s1 = check_filed(RESULTS / "stage1.json", ctx.large)
    out: dict[str, Any] = {"stage1_commit": git("log", "-1", "--format=%H", "--", "experiments/keydependent_elicit/results/stage1.json").stdout.strip(),
                           "finetunes": {}, "states": {}}
    ws = {n: torch.load(ctx.large / f"windows_{n}.pt", weights_only=False) for n in TASKS}
    tasks = {n: ST.load_task(ctx.net, ctx.large / f"windows_{n}.pt", n, n) for n in TASKS}
    dws = finetunes(ctx, ws, out)
    for n, d in dws.items():
        torch.save(d, ctx.large / f"{n}.pt")
        out["finetunes"][n]["sha256"] = sha256(ctx.large / f"{n}.pt")
    task_of = lambda n: tasks["elicit" if n.startswith("elicit") else "cipher"]
    g = {n: ST.gains(ctx.net, task_of(n), d) for n, d in dws.items()}
    for grp in ("elicit", "cipher", "cipher_own"):
        floor = EL.floor_of(g[f"{grp}_s0"]["G"], g[f"{grp}_s1"]["G"], ctx.P["settings"]["floor_min"])
        for s in SEEDS:
            n = f"{grp}_s{s}"
            out["states"][n] = {**g[n], "floor": floor, "readable": abs(g[n]["G"]) >= floor}
            log(f"[gain {n}] G {g[n]['G']:.4f} G_fit {g[n]['G_fit']:.4f} G_score {g[n]['G_score']:.4f} floor {floor:.4f} readable {out['states'][n]['readable']}")
    dump("stage2.json", out)
    bases = {n: base_of(ctx, t) for n, t in tasks.items()}
    for n, d in dws.items():
        if out["states"][n]["readable"]:
            t = task_of(n)
            out["states"][n].update(readings(ctx, t, d, g[n], bases[t.name]))
            st = out["states"][n]
            log(f"[read {n}] S {st['S']:.4f} (block {st['layer']}, block-0 {st['S_block0']:.1e}) r {st['r']:.4f} G_logodds {st['G_logodds']:.4f}")
        dump("stage2.json", out)
    out["score"] = EL.score(out["states"], ctx.P["thresholds"])
    out["stage1_K"] = s1["K"]
    dump("stage2.json", out)
    log(f"[score] {json.dumps(_finite(out['score']))}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    exp = HERE.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["01", "2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--e46", type=Path, default=exp / "regating_table" / "results" / "large", help="E46's saved states and windows")
    ap.add_argument("--alive", type=Path, default=exp / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="where this run's windows and states go")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    P = json.loads(PRED.read_text())
    a.large.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = make_ctx(a, P)
    if a.stage == "2":
        return stage2(ctx)
    s0 = {**stage0(ctx, a.e46), "commit": git("rev-parse", "HEAD").stdout.strip()}
    dump("stage0.json", s0)
    if not s0["pass"]:
        sys.exit("[stage0] FAIL: stop and tell the user")
    s1 = stage1(ctx, s0["marker"])
    dump("stage1.json", s1)
    log(f"[stage1] pass {s1['pass']}" + (f" ({s1['why']})" if "why" in s1 else ""))
    if not s1["pass"]:
        sys.exit("[stage1] FAIL: stop and tell the user")


if __name__ == "__main__":
    main()

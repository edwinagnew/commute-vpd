"""E137's stage 0: float floors, the identity check, building the pieces, each pair's gap, rank and the capacity read."""

from __future__ import annotations

import json
from typing import Any

import pieces as PC
import rules as RU
import search as SE
import torch
from common import dump, f_edit_of, free, last_logits, load, log
from torch import Tensor

from commute.hidden_maps import chunks, run, tv
from commute.steering import Net

PAIRS, SITES = PC.PAIRS, SE.SITES


def measure_floors(net: Net, E: Tensor, cfg: PC.Config) -> dict[str, float]:
    """The float floor between two identical runs of M (batch sizes differ so a real rounding difference is seen), and, because
    the witnesses go through a projection and back, the floor on that same path: h - QQ'h + Q(Q'h) with a random Q (rule: a floor is
    measured on the path it guards). In the distance unit (per-window max) and in the residual's norm."""
    d = run(net, E[:64], sites=("in2",))[1]["in2"].shape[-1]
    Q = torch.linalg.qr(torch.randn(d, cfg.k, generator=torch.Generator().manual_seed(137_600))).Q.to(next(net.net.parameters()).device)

    def rt(x: Tensor) -> Tensor:
        x = x.clone()
        h = x[:, -1]
        x[:, -1] = (h - (h @ Q) @ Q.T) + (h @ Q) @ Q.T
        return x

    l1, l2 = last_logits(net, E, cfg.read_batch), last_logits(net, E, max(cfg.read_batch // 2, 1))
    h1 = torch.cat([run(net, t, sites=("in2",))[1]["in2"][:, -1] for t in chunks(E, cfg.read_batch)])
    h2 = torch.cat([run(net, t, sites=("in2",))[1]["in2"][:, -1] for t in chunks(E, max(cfg.read_batch // 2, 1))])
    lr = last_logits(net, E, cfg.read_batch, lambda _t: {"in2": rt})
    hr = torch.cat([run(net, t, sites=("in2",), edit={"in2": rt})[1]["in2"][:, -1] for t in chunks(E, cfg.read_batch)])
    return {"runs_tv": float(tv(l1, l2).max()), "path_tv": float(tv(l1, lr).max()),
            "runs_h": float((h1 - h2).norm(dim=-1).max()), "path_h": float((h1 - hr).norm(dim=-1).max())}


def start_pieces(net: Net, task: PC.Task, cfg: PC.Config) -> PC.Pieces:
    d = run(net, torch.zeros(4, cfg.window, dtype=torch.long), sites=("in2",))[1]["in2"].shape[-1]
    dev = next(net.net.parameters()).device
    Q = torch.linalg.qr(torch.randn(d, cfg.k, generator=torch.Generator().manual_seed(137_601))).Q.to(dev)
    z = torch.zeros
    return PC.Pieces(Q, z(cfg.k, device=dev), z(d, device=dev), z(1, device=dev), z(d, d, device=dev), z(d, device=dev), task)


def load_pieces(cfg: PC.Config, net: Net, task: PC.Task) -> PC.Pieces:
    st = torch.load(cfg.results / "large" / "pieces.pt", weights_only=False, map_location=next(net.net.parameters()).device)
    return PC.Pieces(st["Q"], st["b"], st["w"], st["a"], st["C"], st["c"], task)


def stage0(net: Net, cfg: PC.Config, sets: PC.Sets, task: PC.Task, filed: dict[str, Any]) -> tuple[PC.Pieces, dict[str, Any]]:
    done = load(cfg, "stage0.json")
    if done is not None and not done.get("stopped"):
        log("stage 0 read from file")
        return load_pieces(cfg, net, task), done
    th = filed["thresholds"]
    floors = measure_floors(net, sets.E, cfg)
    log(f"floors: {floors}")
    tol_tv, tol_h = th["tol_mult"] * max(floors["runs_tv"], floors["path_tv"]), th["tol_mult"] * max(floors["runs_h"], floors["path_h"])
    start = start_pieces(net, task, cfg)
    l1 = last_logits(net, sets.E, cfg.read_batch)
    identity = {pair: float(tv(l1, last_logits(net, sets.E, cfg.read_batch, f_edit_of(net, start, pair))).max()) for pair in PAIRS}
    log(f"identity (max TV, M' at start vs M, per pair): {identity}; floor between identical runs {floors['runs_tv']}")
    if not all(RU.identity_ok(v, floors["runs_tv"]) for v in identity.values()):
        dump(cfg, "stage0.json", {"stopped": "identity", "floors": floors, "identity": identity})
        log("STAGE 0 STOPPED: M' with a piece at its start is outside the float floor")
        raise SystemExit(1)

    Q, b, ao = PC.train_always_on(net, sets, cfg, log)
    w, a, cue = PC.fit_cue(net, sets, task, cfg, log)
    C, c, table = PC.train_table(net, sets, task, cfg, log)
    pieces_ = PC.Pieces(Q, b, w, a, C, c, task)
    (cfg.results / "large").mkdir(parents=True, exist_ok=True)
    torch.save(pieces_.state(), cfg.results / "large" / "pieces.pt")

    pairs: dict[str, Any] = {}
    for pair in PAIRS:
        fit, read = PC.pair_windows(cfg, pair, sets, task)
        fedit = f_edit_of(net, pieces_, pair)
        gap = 0.0
        for t in chunks(read, cfg.read_batch):
            gap += float(tv(run(net, t)[0][:, -1], run(net, t, edit=fedit(t))[0][:, -1]).sum())
        gap /= read.shape[0]
        read_ok = RU.reading(gap, th) and (pair != "Table" or table["rank_C"] > cfg.k)
        pairs[pair] = {"gap": gap, "reading": read_ok, "fit": fit.shape[0], "read": read.shape[0]}
        log(f"pair {pair}: gap {gap:.5f}, reading {read_ok}")
        free()
    fit, read = PC.pair_windows(cfg, "XOR", sets, task)
    cap = SE.capacity(net, fit, read, pieces_, cfg, log)
    out = {"floors": floors, "tol_tv": tol_tv, "tol_h": tol_h, "identity": identity, "pairs": pairs, "capacity": cap,
           "pieces": {"always_on": ao, "cue": cue, "table": table, "cue_token": task.cue}}
    dump(cfg, "stage0.json", out)
    log("capacity: " + json.dumps(cap["held_out_real_by_site"]))
    print("STAGE 0 DONE", flush=True)
    return pieces_, out

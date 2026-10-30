"""Job 3 (`--stage instruments`, after filing B and job 2): stage 0 alone (the zero update, the lens check, both
planted controls, the weights back at W0), then D, pruning and reverse fine-tuning on the 32 states in the order
rank-1, cipher, elicit, copy and seeds 0-7, results/run.json written after each; then q per map and the scores
(`revival_kv.score_columns`). Every scored number is printed. Exit 3 at the deadline; resubmitted, it reruns stage 0,
skips the finished states and scores. Plan § 5 job 3 and job 4."""

from __future__ import annotations

import json
import time
from typing import Any

import torch
from common import (
    EXIT_ORDER,
    EXIT_RESUME,
    RESULTS,
    Clock,
    Tasks,
    dump,
    e58_records,
    load_state,
    log,
    rank1_ref,
    state_path,
    stop,
)
from measure import planted_checks, reverse_cfg

from commute import keydep as KD
from commute import revival_kv as KV
from commute import steering as ST
from commute.heldout_keys import pool_of
from commute.vpd import target_weights

OUT, SEEDS_JSON = RESULTS / "run.json", RESULTS / "seeds.json"
STATE_SECONDS, SCORE_SECONDS = 120.0, 60.0  # per state until measured (plan § 8: 10-40 s, a censored one ~1.5 min)


def at_w0(net: ST.Net) -> bool:
    return all(torch.equal(v, net.W0[k]) for k, v in target_weights(net.model).items())


def stage0(net: ST.Net, ref: KV.Ref, w0: torch.nn.Module, train: torch.Tensor, tasks: Tasks, C: dict[str, Any],
           B: dict[str, Any], out: dict[str, Any], clock: Clock) -> str | None:  # fmt: skip
    """Plan § 5 job 3 stage 0. Returns a stop reason; a lens or planted-pruning miss only voids that reading."""
    zero = {k: torch.zeros_like(w) for k, w in net.W0.items()}
    t = tasks["rank1"]["trained"]
    lens = KV.lens_q(net, t.toks, t.half, t.tgt(t.toks), pool_of(tasks["rank1"]))
    rec = {"zero_D": KV.kappa(net, ref, zero), "lens_check": lens["lens_check"], "lens_ok": lens["lens_check"] <= C["lens_tol"]}
    w = KV.rank1_direction(net, t.tgt(t.toks[t.part("fit")]))
    pl = {**B["planted"], "w": w.tolist()}
    rec["planted"] = planted_checks(net, ref, w0, train, pl, C, clock.end)
    rec["prune_ok"], rec["reverse_ok"] = rec["planted"]["prune_pass"]["pass"], rec["planted"]["reverse_pass"]
    out["stage0"] = rec
    log(f"[stage0] zero-update D {rec['zero_D']!r} (void: under D_void {B['D_void']:.4f}) | lens block 4 vs W0 {rec['lens_check']:.2e} "
        f"-> {rec['lens_ok']} | planted pruning {rec['prune_ok']} | planted reverse {rec['reverse_ok']} (t_half "
        f"{rec['planted']['reverse']['t_half']}, reported) | weights at W0 {rec['planted']['w0_restored']}")
    dump(OUT, out)
    if rec["zero_D"] != 0.0:
        return "the zero update's D is not exactly 0"
    if rec["planted"]["reverse"]["end"] == "deadline":
        return "deadline"
    if not rec["reverse_ok"]:
        return "the planted reverse fine-tune did not reach D/2 before the cap"
    return None if rec["planted"]["w0_restored"] else "the weights are not W0 after the planted reverse"


def recorded() -> dict[str, dict[str, Any]]:
    """E58's records (seeds 0-2) and job 2's (seeds 3-7), by state name."""
    return {**e58_records(), **json.loads(SEEDS_JSON.read_text())["states"]}


def read_state(net: ST.Net, ref: KV.Ref, w0: torch.nn.Module, train: torch.Tensor, m: str, s: int, C: dict[str, Any],
               B: dict[str, Any], clock: Clock) -> dict[str, Any] | None:  # fmt: skip
    """D, MLP share and, unless void, pruning and reverse fine-tuning on one state; None when the deadline cut it."""
    t0, dW = time.monotonic(), load_state(m, s)
    D = KV.kappa(net, ref, dW)
    rec: dict[str, Any] = {"D": D, "void": D < B["D_void"], "mlp_share": KV.mlp_share(dW)}
    if not rec["void"]:
        rd = KV.prune_readings(net, ref, dW, C["prune_k"], C["n_random"], C["prune_seed"], KV.unit_scores(net, ref, dW))
        rec |= {"prune": rd, **KV.revival(rd, C["prune_k"], D)}
        rec["reverse"] = KV.reverse_curve(net, ref, dW, w0, train, reverse_cfg(C, s), clock.end)
        if rec["reverse"]["end"] == "deadline":
            return None
        rec["t_half"] = rec["reverse"]["t_half"] if rec["reverse"]["end"] == "half" else "censored"
        rec["w0_restored"] = at_w0(net)
    rec["seconds"] = time.monotonic() - t0
    return rec


def say_state(n: str, r: dict[str, Any]) -> None:
    if r["void"]:
        log(f"[state {n}] D {r['D']:.5f} VOID (under D_void) MLP share {r['mlp_share']:.4f}")
        return
    log(f"[state {n}] D {r['D']:.4f} MLP share {r['mlp_share']:.4f} | pruning Delta {r['Delta']} R {r['R']:.4f} random spread {r['spread']} "
        f"top-13 units {r['prune']['13']['units']} | reverse t_half {r['t_half']} ({r['reverse']['end']}, {len(r['reverse']['curve']) - 1} steps) "
        f"| {r['seconds']:.0f} s")


def columns(st: dict[str, Any], rec: dict[str, Any], q: dict[str, float], prior: dict[str, float], seeds: list[int]) -> dict[str, Any]:
    """Every scored and reported column, per map, per seed."""
    names = lambda m: [f"{m}_s{s}" for s in seeds]
    run = lambda key: {m: [None if st[n]["void"] else st[n][key] for n in names(m)] for m in KV.MAPS}
    t_half, censored = KV.t_half_column(run("t_half"))
    cols = {"R": run("R"), "t_half": t_half, "D": {m: [st[n]["D"] for n in names(m)] for m in KV.MAPS},
            "mlp_share": {m: [st[n]["mlp_share"] for n in names(m)] for m in KV.MAPS},
            "prior": {m: [prior[m]] * len(seeds) for m in KV.MAPS}, "q": {m: [q[m]] * len(seeds) for m in KV.MAPS},
            "norm": {m: [rec[n]["ft"]["norm"] for n in names(m)] for m in KV.MAPS}}  # fmt: skip
    cols |= {k: {m: [rec[n][k] for n in names(m)] for m in KV.MAPS} for k in ("r", "ratio", "S", "x")}
    return {"columns": cols, "censored": censored}


def stage_instruments(net: ST.Net, tasks: Tasks, C: dict[str, Any], B: dict[str, Any], out: dict[str, Any], clock: Clock) -> int:
    if not SEEDS_JSON.is_file():
        log(f"{SEEDS_JSON} is missing: run job 2 (--stage seeds) first (exit {EXIT_ORDER})")
        return EXIT_ORDER
    old = json.loads(OUT.read_text()) if OUT.is_file() else {}
    if old and old.get("predictions_sha256") != out["predictions_sha256"]:
        log(f"{OUT} was written under other predictions: move it aside by hand (exit {EXIT_ORDER})")
        return EXIT_ORDER
    out["states"], out["resumed_from"] = old.get("states", {}), old.get("git_head")
    rec = recorded()
    missing = [f"{m}_s{s}" for m in KV.MAPS for s in C["seeds"] if f"{m}_s{s}" not in rec]
    if missing:
        log(f"job 2 has not finished: no record of {missing} (exit {EXIT_ORDER})")
        return EXIT_ORDER

    ref, train = rank1_ref(net)
    w0 = KV.w0_copy(net)
    if (why := stage0(net, ref, w0, train, tasks, C, B, out, clock)) is not None:
        return EXIT_RESUME if why == "deadline" else stop(OUT, out, why)
    st, seeds = out["states"], C["seeds"]
    for m in KV.MAPS:
        for s in seeds:
            n = f"{m}_s{s}"
            if n in st:
                continue
            if KD.sha256(state_path(m, s)) != rec[n]["ft"]["sha256"]:
                return stop(OUT, out, f"{n}: the state file is not the one its run recorded")
            if clock.left() < max([STATE_SECONDS] + [r["seconds"] for r in st.values()]) + SCORE_SECONDS:
                log(f"deadline: {n} not started; resubmit to resume (exit {EXIT_RESUME})")
                return EXIT_RESUME
            r = read_state(net, ref, w0, train, m, s, C, B, clock)
            if r is None:
                log(f"deadline: {n}'s reverse fine-tune cut; it reruns on resubmission (exit {EXIT_RESUME})")
                return EXIT_RESUME
            st[n] = r
            say_state(n, r)
            dump(OUT, out)
            if not r.get("w0_restored", True):
                return stop(OUT, out, f"{n}: the weights are not W0 after its reverse fine-tune")
    return score(net, tasks, C, B, out, rec)


def score(net: ST.Net, tasks: Tasks, C: dict[str, Any], B: dict[str, Any], out: dict[str, Any], rec: dict[str, Any]) -> int:
    q = {}
    for m in KV.MAPS:
        t = tasks[m]["trained"]
        out.setdefault("probe", {})[m] = KV.lens_q(net, t.toks, t.half, t.tgt(t.toks), pool_of(tasks[m]))
        q[m] = out["probe"][m]["q"]
        log(f"[probe {m}] q {q[m]:.4f} per block {[round(v, 4) for v in out['probe'][m]['per_block']]}")
    prior = {m: float(tasks[m]["trained"].lp0[tasks[m]["trained"].part("score")].mean()) for m in KV.MAPS}
    cc = columns(out["states"], rec, q, prior, C["seeds"])
    void = {"t_half": "censored"} if cc["censored"] else {}
    if not out["stage0"]["prune_ok"]:
        void["R"] = "void"
    if not out["stage0"]["lens_ok"]:
        void["q"] = "void"
    floors = {"R": B["noise_floor_R"], "t_half": C["noise_floor"]["t_half"]}
    out["score"] = {**KV.score_columns(cc["columns"], floors, void, C), "prior": prior, "probe_ref": KV.probe_reference(q)}
    out["columns"] = cc["columns"]
    dump(OUT, out)
    for name, r in out["score"]["readings"].items():
        log(f"[score {name}] {r['status']}: gap {r['gap']:.4g} (3 x noise {3 * r['noise']:.4g}) p {r['p']:.4f} [{r['p_lo']:.4f}, {r['p_hi']:.4f}] "
            f"side {r['side']} p_copy {r['p_copy']:.4f} means {r['means']}")
    log(f"[verdicts] {out['score']['verdicts']} | joint: {out['score']['joint']} | probe reference {out['score']['probe_ref']}")
    return 0

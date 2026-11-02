"""E62 Amendment 4: does the fine-tune's mask at the marker open the subcomponents W0's own mask opens when W0 copies, and does the
masked gain run through them? A read of E62's saved step-30 states on its 200 read windows; no training, nothing in results/large
overwritten or deleted. Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md, "Amendment 4 (2026-11-02)". Rules and expected
values: results/predictions_amend4.json (committed first; the run refuses otherwise; never edited), `commute.copying`.

  1 control, alone, W0 at the second A (half - 1): target and masked log-odds of B, intact, (2,4) off (in the forward only, never in Γ's pass)
    and pair-broken; `a` = W0's gate at the second A, intact minus pair-broken. Exit 1 on a miss, or on k = 0 or k over half the alive set.
  2 does S_old carry W0's own copying? W0's closing at the second A against random sets. A miss is a verdict, not a stop.
  3 re-read, W0 and the six states: L00, L01 - L00 against amend3.json; the zero update's gain and its level under closing. Exit 1 on a miss.
  4 the six states: alignment, closing share, its random reference, the k largest openings closed; beside, no rule: every alive closed, the k
    largest outside S_old, the half - 2 contrast, the positive parts' cosine, (2,3)'s target share.
Masked forward and Γ bf16 (autocast, as Amendment 3), target model fp32. results/amend4.json after every stage and state; every number printed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import torch
from amend3 import NAMES, Reads, map_of
from torch import Tensor

from commute import copying as CP
from commute import keydep as KD
from commute import steering as ST
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, BASE, RUN, A3_OUT, OUT = (RESULTS / f for f in ("predictions_amend4.json", "predictions.json", "run.json", "amend3.json", "amend4.json"))
log = KD.log
Gates = list[dict[str, Tensor]]


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))


def stop(out: dict[str, Any], why: str, verdict: dict[str, Any] | None = None) -> NoReturn:
    out["stopped"] = why
    if verdict:
        out["verdict"] = verdict  # the same dict as a finished run's, from `mask_test`
    dump(out)
    sys.exit(f"STOP: {why}: tell the lead")


def note(out: dict[str, Any], key: str, rec: dict[str, Any]) -> None:
    out[key] = rec
    dump(out)
    log(f"[{key}] {json.dumps(jsonable(rec))}")


def stage1(R: Reads, p: Tensor, H: list[CP.Head], th: dict[str, Any], out: dict[str, Any]) -> tuple[Gates, Gates, Tensor, Tensor, Tensor]:
    """W0 only. No Γ pass carries an ablation; `at` only moves the read position, as `gates` does not use it (tests/test_copying.py)."""
    w, b, at, alive, n = R.W["copy"], R.broken, dataclasses.replace(R, half=R.half - 1), R.ctx.lift["alive"], R.n
    if not (CP.second_a_holds(w[:, :n], p, w[:, -1], R.half) and CP.pair_broken_holds(w[:, :n], b[:, :n], p, R.half) and torch.equal(w[:, -1], b[:, -1])):
        stop(out, "stage 1: half - 1 is not the second A on every window, half - 2 is A or B, or the pair-broken windows differ elsewhere than the first A")
    g0, gb = R.gates(w), R.gates(b)
    t, m = {"on": at.lo(w, []), "off": at.lo(w, H), "broken": at.lo(b, [])}, {"on": at.masked(w, g0, []), "off": at.masked(w, g0, H), "broken": at.masked(b, gb, [])}
    level = {f"{x}_{k}": float(v.mean()) for x, d in (("target", t), ("mask", m)) for k, v in d.items()}
    falls = {"fall": level["target_on"] - level["target_off"], "lean": level["target_on"] - level["target_broken"], "mask_fall": level["mask_on"] - level["mask_off"], "mask_lean": level["mask_on"] - level["mask_broken"]}
    p_b = KD.geo_rate(CP.read_lo(R.ctx.net, w, at.half, n, readout=ST.logp_at))
    second = CP.gate_means(g0, alive, R.half - 1)
    a, a2 = second - CP.gate_means(gb, alive, R.half - 1), second - CP.gate_means(g0, alive, R.half - 2)
    old = CP.s_old(a)
    status = CP.second_a_control(p_b, falls["fall"], falls["lean"], falls["mask_fall"], falls["mask_lean"], th)
    N, k = len(a), len(old)
    note(out, "stage1_control", {"p_b": p_b, **level, **falls, "status": status, "n_alive": N, "k": k, "a_positive": int((a > 0).sum()), "a_mass_positive": float(a.clamp_min(0).sum()),
                                 "top_a": [float(x) for x in a[old[:10]]], "half2_k": len(CP.s_old(a2)), "half2_overlap": int(torch.isin(old, CP.s_old(a2)).sum()), "cos_a_half2": CP.cosine(a, a2)})
    if status != "pass" or k == 0 or k > N // 2:
        stop(out, f"stage 1: {status}, k = {k} of {N}", CP.mask_test(status, "unread", k, N, [], [], th))
    if N != th["n_alive"]:
        stop(out, f"stage 1: {N} alive subcomponents, filed {th['n_alive']}")
    return g0, gb, a, a2, old


def stage2_carries(R: Reads, g0: Gates, gb: Gates, a: Tensor, old: Tensor, S: dict[str, Any], th: dict[str, Any], out: dict[str, Any]) -> str:
    """W0's closing at the second A, read as stage 4 reads a state, on one set of tokens, the intact ones: the gain is W0's gates against their
    pair-broken values; S_old's gates go to those values at every position. Random sets of k with a > 0, outside S_old, closed the same way.
    A miss blocks the two readings; the states are still read. The lean's token part, the pair-broken gates on both sets of tokens, is printed."""
    at, w, alive = dataclasses.replace(R, half=R.half - 1), R.W["copy"], R.ctx.lift["alive"]
    base = at.masked(w, gb, [])
    d = at.masked(w, g0, []) - base

    def close(idx: Tensor) -> tuple[float, float, float, float]:
        return at.share(d, at.masked(w, CP.closed(g0, gb, idx, alive), []) - base)

    pool, fell_back = CP.closing_pool(a, old)
    G, G_closed, share, sd = close(old)
    randoms = [close(s)[2] for s in CP.random_sets(pool, len(old), S["random_draws"], S["random_seed"])]
    median = statistics.median(randoms)
    carries = CP.old_carries(G, share, median, th)
    note(out, "stage2_carries", {"gate_gain": G, "gain_closed": G_closed, "token_part": float((base - at.masked(R.broken, gb, [])).mean()), "share": share, "share_sd": sd, "random_shares": randoms, "random_median": median,
                                 "pool_size": len(pool), "pool_fell_back": fell_back, "all_alive_share": close(torch.arange(len(a)))[2], "carries": carries})
    return carries


def stage3(R: Reads, g0: Gates, old: Tensor, A3: dict[str, Any], th: dict[str, Any], out: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """The re-read, before anything is read on top of it. amend3.json's gates and masked forward are the same calls on the same windows."""
    alive = R.ctx.lift["alive"]
    L00 = {m: R.masked(w, g0, []) for m, w in R.W.items()}
    gz = R.gates(R.W["copy"], R.zero)
    closed0 = CP.closed(gz, g0, old, alive)
    rec: dict[str, Any] = {"L00_diff": {m: abs(float(v.mean()) - A3["row1_control"]["maps"][m]["L00"]) for m, v in L00.items()}, "states": {},
                           "zero_gain_max_abs": {m: float((R.masked(w, gz, []) - L00[m]).abs().max()) for m, w in R.W.items()},
                           "zero_closed_max_abs": {m: float((R.masked(w, closed0, []) - L00[m]).abs().max()) for m, w in R.W.items()}}
    on: dict[str, Tensor] = {}
    for name, dW in R.dws.items():
        m = map_of(name)
        on[name] = R.masked(R.W[m], R.gates(R.W[m], dW), [])
        M = float((on[name] - L00[m]).mean())
        rec["states"][name] = {"M": M, "M_amend3": A3["row2_gamma"]["states"][name]["M"], "diff": abs(M - A3["row2_gamma"]["states"][name]["M"])}
    diffs = [*rec["L00_diff"].values(), *(r["diff"] for r in rec["states"].values())]
    rec["max_diff"] = max(diffs)
    rec["exact"] = rec["max_diff"] == 0.0  # Amendment 3 read 0 against Amendment 2 on the L40; false here is worth a look even within the tolerance
    note(out, "stage3_reread", rec)
    if rec["max_diff"] > th["reread_tol"] or any(v > th["zero_tol"] for d in ("zero_gain_max_abs", "zero_closed_max_abs") for v in rec[d].values()):
        stop(out, "stage 3: the masked gain does not re-read amend3.json, or the zero update's masked gain or its level under closing is not L00")
    return L00, on


def read_state(R: Reads, name: str, g0: Gates, a: Tensor, a2: Tensor, old: Tensor, d: Tensor, L00: Tensor, t0: tuple[Tensor, Tensor], other: list[CP.Head], w0_marker: Tensor, S: dict[str, Any]) -> tuple[dict[str, Any], Tensor]:
    """One state. `d` is its per-window masked gain from stage 3, `L00` W0's masked log-odds, `t0` W0's target log-odds, other head on and off."""
    dW, w, alive, k = R.dws[name], R.W[map_of(name)], R.ctx.lift["alive"], len(old)
    g1 = R.gates(w, dW)
    delta = CP.gate_means(g1, alive, R.half) - w0_marker

    def close(idx: Tensor) -> tuple[float, float, float, float]:
        return R.share(d, R.masked(w, CP.closed(g1, g0, idx, alive), []) - L00)

    M, M_closed, share, sd = close(old)
    pool, fell_back = CP.closing_pool(delta, old)
    randoms = [close(s)[2] for s in CP.random_sets(pool, k, S["random_draws"], S["random_seed"])]
    chance, chance_sd = CP.chance_cosine(delta, a, S["chance_draws"], S["chance_seed"])
    big = CP.largest(delta, k)
    scored = {"M": M, "M_closed": M_closed, "share": share, "share_sd": sd, "cos": CP.cosine(delta, a), "chance": chance, "chance_sd": chance_sd, "random_shares": randoms,
              "random_median": statistics.median(randoms), "largest_k_share": close(big)[2]}
    reported = {"pool_size": len(pool), "pool_fell_back": fell_back, "n_opened": int((delta > 0).sum()), "old_opened": int((delta[old] > 0).sum()), "cos_positive": CP.cosine(delta.clamp_min(0), a.clamp_min(0)),
                "largest_overlap": int(torch.isin(big, old).sum()), "largest_outside_share": close(CP.largest(delta, k, old))[2], "all_alive_share": close(torch.arange(len(delta)))[2],
                "half2_cos": CP.cosine(delta, a2), "half2_share": close(CP.s_old(a2))[2], "other_head_target_share": R.share(R.lo(w, [], dW) - t0[0], R.lo(w, other, dW) - t0[1])[2]}
    return {**scored, **reported}, delta


def stage4(R: Reads, g0: Gates, a: Tensor, a2: Tensor, old: Tensor, L00: dict[str, Tensor], on: dict[str, Tensor], other: list[CP.Head], S: dict[str, Any], out: dict[str, Any]) -> dict[str, Tensor]:
    w0_marker = CP.gate_means(g0, R.ctx.lift["alive"], R.half)
    t0 = {m: (R.lo(w, []), R.lo(w, other)) for m, w in R.W.items()}
    rec: dict[str, Any] = out.setdefault("stage4_states", {})
    deltas: dict[str, Tensor] = {}
    for name in R.dws:
        m = map_of(name)
        rec[name], deltas[name] = read_state(R, name, g0, a, a2, old, on[name] - L00[m], L00[m], t0[m], other, w0_marker, S)
        dump(out)
        log(f"[stage4 {name}] {json.dumps(jsonable(rec[name]))}")
    return deltas


def load(a: argparse.Namespace) -> tuple[Reads, dict[str, Any], dict[str, Any], Tensor, dict[str, Any]]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    if OUT.exists():
        sys.exit(f"{OUT} exists: move it aside, this run deletes nothing")
    P, S, run, A3 = json.loads(PRED.read_text()), json.loads(BASE.read_text())["settings"], json.loads(RUN.read_text()), json.loads(A3_OUT.read_text())
    large = a.large.resolve()
    for f, sha in {"windows": run["windows_sha256"], **{n: run["states"][n]["sha256"] for n in NAMES}}.items():
        if KD.sha256(large / f"{f}.pt") != sha:
            sys.exit(f"{f}.pt is not the one run.json recorded")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, large)
    v = torch.load(large / "windows.pt", weights_only=False)
    half, n = S["half"], 2 * S["half"] + 1
    W = {m: CP.with_target(v["read"], v["target"][m]["read"]).tokens for m in ("copy", "lookup")}
    if not torch.equal(W["copy"][:, :n], W["lookup"][:, :n]):
        sys.exit("the two maps' windows differ: W0's one Γ pass does not serve both")
    means = torch.load(large / "heads.pt", weights_only=False)["means"].to(ctx.net.device)
    dws = {name: torch.load(large / f"{name}.pt", weights_only=True, map_location="cpu") for name in NAMES}
    b = CP.broken_pair(v["read"].tokens, v["p_read"], v["pools"]["keys"], half, S["control_seed"])  # as amend2.py
    R = Reads(ctx, half, n, means, S["n_boot"], W, torch.cat([b, v["target"]["copy"]["read"][:, None]], 1), dws, run)
    return R, P, A3, v["p_read"], {"git_head": git("rev-parse", "HEAD").stdout.strip(), "predictions_sha256": KD.sha256(PRED), "host": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large")
    a = ap.parse_args()
    R, P, A3, p, out = load(a)
    S, th = P["settings"], P["thresholds"]
    heads = lambda key: [(int(x), int(y)) for x, y in S[key]]
    log(f"[host] {out['host']}: masked forward and Γ bf16, target model fp32")
    g0, gb, s, s2, old = stage1(R, p, heads("head"), th, out)
    carries = stage2_carries(R, g0, gb, s, old, S, th, out)
    del gb
    L00, on = stage3(R, g0, old, A3, th, out)
    deltas = stage4(R, g0, s, s2, old, L00, on, heads("other_head"), S, out)
    torch.save({"a": s, "a_half2": s2, "S_old": old, "delta": deltas}, a.large.resolve() / "amend4_vectors.pt")
    states = out["stage4_states"]
    v = CP.mask_test(out["stage1_control"]["status"], carries, len(old), len(s), [states[n] for n in NAMES[:3]], [states[n] for n in NAMES[3:]], th)
    note(out, "verdict", v)
    log(f"[verdict] {v['verdict']} carries {carries} P1 {v['P1']} P2 {v['P2']} P3 {v['P3']}")


if __name__ == "__main__":
    main()

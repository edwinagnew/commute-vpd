"""E62 Amendment 4: does the fine-tune's mask at the marker open the subcomponents W0's own mask opens when W0 copies, and does the
masked gain run through them? A read of E62's saved step-30 states on its 200 read windows; no training, nothing in results/large
overwritten or deleted. Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md, "Amendment 4 (2026-11-02)". Rules and expected
values: results/predictions_amend4.json (committed first; the run refuses otherwise; never edited), `commute.copying`.

  1 control, alone, W0 at the second A (half - 1): target and masked log-odds of B, head (2,4) on and off, off in the forward only, never
    in Γ's pass; `a` and S_old from W0's own Γ pass. Exit 1 on a miss, or on k = 0 or k over half the alive set.
  2 re-read, W0 and the six states: L00, L01 - L00 against amend3.json; the zero update's gain and its level under closing. Exit 1 on a miss.
  3 the six states: alignment, closing share, its random reference; beside, no rule: the k largest openings closed, (2,3)'s target share.
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


def stop(out: dict[str, Any], why: str, verdict: str | None = None) -> NoReturn:
    out["stopped"] = why
    if verdict:
        out["verdict"] = verdict
    dump(out)
    sys.exit(f"STOP: {why}: tell the lead")


def note(out: dict[str, Any], key: str, rec: dict[str, Any]) -> None:
    out[key] = rec
    dump(out)
    log(f"[{key}] {json.dumps(jsonable(rec))}")


def stage1(R: Reads, p: Tensor, H: list[CP.Head], th: dict[str, Any], out: dict[str, Any]) -> tuple[Gates, Tensor, Tensor]:
    """W0 only. Γ's pass carries no ablation; `at` only moves the read position, as `gates` does not use it (tests/test_copying.py)."""
    w, at, alive = R.W["copy"], dataclasses.replace(R, half=R.half - 1), R.ctx.lift["alive"]
    if not CP.second_a_holds(w[:, : R.n], p, w[:, -1], R.half):
        stop(out, "stage 1: half - 1 is not the second A on every window, or half - 2 is A or B")
    g0 = R.gates(w)
    target, target_off = at.lo(w, []), at.lo(w, H)
    mask, mask_off = at.masked(w, g0, []), at.masked(w, g0, H)
    p_b = KD.geo_rate(CP.read_lo(R.ctx.net, w, at.half, R.n, readout=ST.logp_at))
    second, before = CP.gate_means(g0, alive, R.half - 1), CP.gate_means(g0, alive, R.half - 2)
    a = second - before
    old = CP.s_old(a)
    means = {"target": float(target.mean()), "target_off": float(target_off.mean()), "mask": float(mask.mean()), "mask_off": float(mask_off.mean())}
    status = CP.second_a_control(p_b, means["target"], means["target_off"], means["mask"], means["mask_off"], th)
    n, k = len(a), len(old)
    rec = {"p_b": p_b, **means, "target_fall": means["target"] - means["target_off"], "mask_fall": means["mask"] - means["mask_off"], "status": status, "n_alive": n, "k": k,
           "a_positive": int((a > 0).sum()), "a_mass_positive": float(a.clamp_min(0).sum()), "top_a": [float(x) for x in a[old[:10]]], "second_mean": float(second.mean()), "before_mean": float(before.mean())}
    note(out, "stage1_control", rec)
    if status != "pass":
        stop(out, f"stage 1: {status}", "control_failed")
    if n != th["n_alive"]:
        stop(out, f"stage 1: {n} alive subcomponents, filed {th['n_alive']}")
    if k == 0 or k > n // 2:
        stop(out, f"stage 1: k = {k} of {n}", "unreadable")
    return g0, a, old


def stage2(R: Reads, g0: Gates, old: Tensor, A3: dict[str, Any], th: dict[str, Any], out: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
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
    note(out, "stage2_reread", rec)
    if rec["max_diff"] > th["reread_tol"] or any(v > th["zero_tol"] for d in ("zero_gain_max_abs", "zero_closed_max_abs") for v in rec[d].values()):
        stop(out, "stage 2: the masked gain does not re-read amend3.json, or the zero update's masked gain or its level under closing is not L00")
    return L00, on


def read_state(R: Reads, name: str, g0: Gates, a: Tensor, old: Tensor, d: Tensor, L00: Tensor, t0: tuple[Tensor, Tensor], other: list[CP.Head], w0_marker: Tensor, S: dict[str, Any]) -> tuple[dict[str, Any], Tensor]:
    """One state. `d` is its per-window masked gain from stage 2, `L00` W0's masked log-odds, `t0` W0's target log-odds, other head on and off."""
    dW, w, alive, k = R.dws[name], R.W[map_of(name)], R.ctx.lift["alive"], len(old)
    g1 = R.gates(w, dW)
    delta = CP.gate_means(g1, alive, R.half) - w0_marker

    def close(idx: Tensor) -> tuple[float, float, float, float]:
        return R.share(d, R.masked(w, CP.closed(g1, g0, idx, alive), []) - L00)

    M, M_closed, share, sd = close(old)
    pool, fell_back = CP.closing_pool(delta, old)
    randoms = [close(s)[2] for s in CP.random_sets(pool, k, S["random_draws"], S["random_seed"])]
    chance, chance_sd = CP.chance_cosine(delta, a, S["chance_draws"], S["chance_seed"])
    return {"M": M, "M_closed": M_closed, "share": share, "share_sd": sd, "cos": CP.cosine(delta, a), "chance": chance, "chance_sd": chance_sd,
            "random_shares": randoms, "random_median": statistics.median(randoms), "pool_size": len(pool), "pool_fell_back": fell_back, "n_opened": int((delta > 0).sum()),
            "old_opened": int((delta[old] > 0).sum()), "largest_k_share": close(CP.largest(delta, k))[2],
            "other_head_target_share": R.share(R.lo(w, [], dW) - t0[0], R.lo(w, other, dW) - t0[1])[2]}, delta


def stage3(R: Reads, g0: Gates, a: Tensor, old: Tensor, L00: dict[str, Tensor], on: dict[str, Tensor], other: list[CP.Head], S: dict[str, Any], out: dict[str, Any]) -> dict[str, Tensor]:
    w0_marker = CP.gate_means(g0, R.ctx.lift["alive"], R.half)
    t0 = {m: (R.lo(w, []), R.lo(w, other)) for m, w in R.W.items()}
    rec: dict[str, Any] = out.setdefault("stage3_states", {})
    deltas: dict[str, Tensor] = {}
    for name in R.dws:
        m = map_of(name)
        rec[name], deltas[name] = read_state(R, name, g0, a, old, on[name] - L00[m], L00[m], t0[m], other, w0_marker, S)
        dump(out)
        log(f"[stage3 {name}] {json.dumps(jsonable(rec[name]))}")
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
    R = Reads(ctx, half, n, means, S["n_boot"], W, torch.empty(0), dws, run)
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
    g0, s, old = stage1(R, p, heads("head"), th, out)
    L00, on = stage2(R, g0, old, A3, th, out)
    deltas = stage3(R, g0, s, old, L00, on, heads("other_head"), S, out)
    torch.save({"a": s, "S_old": old, "delta": deltas}, a.large.resolve() / "amend4_vectors.pt")
    states = out["stage3_states"]
    v = CP.mask_test(out["stage1_control"]["status"], len(old), len(s), [states[n] for n in NAMES[:3]], [states[n] for n in NAMES[3:]], th)
    note(out, "verdict", v)
    log(f"[verdict] {v['verdict']} P1 {v['P1']} P2 {v['P2']} P3 {v['P3']}")


if __name__ == "__main__":
    main()

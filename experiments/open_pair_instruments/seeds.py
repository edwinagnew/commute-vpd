"""Job 2 (`--stage seeds`, after filing B): first, alone, seed 0 of each map retrained and checked against E58's
record (elicit, the cipher at its norm, rank-1, copy); exit 1 on a miss. Then seeds 3-7 of the four maps with E58's
recipe, selection rules and columns (`keydep.elicit_ft` / `cipher_ft`, `heldout_keys.behaviour` / `vpd_read`, called
unchanged), each saved to results/large/<map>_s<seed>.pt with its sha256 and written to results/seeds.json as it
finishes; resumes from there. Stop on E58's block-0 rule. Plan § 5 job 2."""

from __future__ import annotations

from typing import Any

import torch
from common import (
    EXIT_RESUME,
    LARGE,
    NEW_SEEDS,
    RESULTS,
    Clock,
    Tasks,
    dump,
    e58_records,
    log,
    state_path,
    stop,
    windows_path,
)

from commute import keydep as KD
from commute import steering as ST
from commute.heldout_keys import behaviour, say_label, vpd_read

OUT = RESULTS / "seeds.json"
ORDER = ("elicit", "cipher", "rank1", "copy")  # elicit before the cipher: the cipher's target norm is elicit's
STATE_SECONDS = 90.0  # one fine-tune and its columns, assumed until measured (E58 Amendment 1: six in ~2.5 min)


def train(ctx: KD.Ctx, m: str, s: int, elicit_norm: float | None) -> tuple[ST.DW, dict[str, Any]]:
    w = torch.load(windows_path(m, "trained"), weights_only=False)
    if m != "cipher":
        return KD.elicit_ft(ctx, w, s)
    assert elicit_norm is not None
    dW, _own, meta, _ = KD.cipher_ft(ctx, w, s, elicit_norm)
    return dW, {k: meta[k] for k in ("step", "norm", "at_bound", "same_state", "p_heldout")}


def columns(ctx: KD.Ctx, tasks: Tasks, bases: dict[str, Any], m: str, dW: ST.DW, meta: dict[str, Any]) -> dict[str, Any]:
    """E58's columns of one state: the label (G, x, ...) and r, ratio, S."""
    b = behaviour(ctx.net, tasks[m], dW, ctx.P["thresholds"])
    return {"ft": meta, **b, **vpd_read(ctx, tasks[m]["trained"], dW, bases[m])}


def say(n: str, r: dict[str, Any]) -> None:
    say_label(n, r)
    log(f"[read {n}] step {r['ft']['step']} ||dW|| {r['ft']['norm']:.4f} r {r['r']:.4f} ratio {r['ratio']:.4f} S {r['S']:.4f} "
        f"(block {r['layer']}, block-0 {r['S_block0']:.1e})")


def block0_bad(r: dict[str, Any], th: dict[str, Any]) -> bool:
    return abs(r["S_block0"]) * abs(r["G_score_logP"]) > th["block0_nats"]


def reproduces(new: dict[str, Any], old: dict[str, Any], ck: dict[str, Any]) -> dict[str, bool]:
    """Job 2's control: the scored step exactly, the norm within ck['norm'] relative, G, x and r within their tolerances."""
    return {"step": new["ft"]["step"] == old["ft"]["step"],
            "norm": abs(new["ft"]["norm"] - old["ft"]["norm"]) <= ck["norm"] * abs(old["ft"]["norm"]),
            **{k: abs(new[k] - old[k]) <= ck[k] for k in ("G", "x", "r")}}  # fmt: skip


def reproduction(ctx: KD.Ctx, tasks: Tasks, bases: dict[str, Any], C: dict[str, Any], out: dict[str, Any]) -> str | None:
    rec, ck = e58_records(), {"norm": 1e-4, **{k: C["reproduction"]["checks"][k] for k in ("G", "x", "r")}}
    assert C["reproduction"]["checks"]["norm"] == "1e-4 relative", "the filed norm tolerance changed"
    norm = None
    for m in ORDER:
        dW, meta = train(ctx, m, 0, norm)
        norm = meta["norm"] if m == "elicit" else norm
        r = columns(ctx, tasks, bases, m, dW, meta)
        r["checks"] = reproduces(r, rec[f"{m}_s0"], ck)
        out["reproduction"][m] = r
        say(f"{m}_s0 (retrained)", r)
        log(f"[reproduce {m}_s0] step {meta['step']} vs {rec[f'{m}_s0']['ft']['step']}, norm {meta['norm']:.6f} vs {rec[f'{m}_s0']['ft']['norm']:.6f}, "
            f"G {r['G']:.5f} vs {rec[f'{m}_s0']['G']:.5f}, x {r['x']:.5f} vs {rec[f'{m}_s0']['x']:.5f}, r {r['r']:.5f} vs {rec[f'{m}_s0']['r']:.5f} -> {r['checks']}")
        dump(OUT, out)
        if not all(r["checks"].values()):
            return f"{m}_s0 retrained does not reproduce E58's record: {r['checks']}"
    out["reproduction"]["pass"] = True
    dump(OUT, out)
    return None


def stage_seeds(ctx: KD.Ctx, tasks: Tasks, C: dict[str, Any], out: dict[str, Any], clock: Clock) -> int:
    LARGE.mkdir(parents=True, exist_ok=True)
    th, st = ctx.P["thresholds"], out.setdefault("states", {})
    bases = {m: KD.base_of(ctx, tasks[m]["trained"]) for m in ORDER}
    if not out.setdefault("reproduction", {}).get("pass") and (why := reproduction(ctx, tasks, bases, C, out)) is not None:
        return stop(OUT, out, why)
    for s in NEW_SEEDS:
        for m in ORDER:
            n = f"{m}_s{s}"
            if n in st and state_path(m, s).is_file() and KD.sha256(state_path(m, s)) == st[n]["ft"]["sha256"]:
                continue
            if clock.left() < 2 * STATE_SECONDS:
                log(f"deadline: {n} not started; resubmit to resume (exit {EXIT_RESUME})")
                return EXIT_RESUME
            dW, meta = train(ctx, m, s, st[f"elicit_s{s}"]["ft"]["norm"] if m == "cipher" else None)
            torch.save(dW, state_path(m, s))
            r = columns(ctx, tasks, bases, m, dW, {**meta, "sha256": KD.sha256(state_path(m, s))})
            st[n] = r
            say(n, r)
            dump(OUT, out)
            if block0_bad(r, th):
                return stop(OUT, out, f"{n}: block-0 shift over {th['block0_nats']} nats")
    return 0

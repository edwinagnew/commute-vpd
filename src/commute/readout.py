"""E59's Def 1 readout on the 67M target model (Jain et al. 2024 §3) and the causal check on its direction. Plan:
reports/plans/2026-10-30_plan_baseline-next2-draft.md; rules and thresholds in
experiments/jain_readout/results/predictions.json. Tests of the pure rules: tests/test_readout.py.

Readout: one linear map d -> vocab per block (`probe.Probes`, unchanged) on block l's output residual (block l+1's
input; the last block's output before the final norm), trained to predict the next token on random 41-token windows of
Pile train rows 32768-40959 (disjoint from the rows E54's task windows were cut from), every position from 1 on, with
`probe.py`'s schedule (4K AdamW iterations, lr 1e-3 decayed x0.1 at 2K, 3K, 3.5K, batch 96 windows). Never sees a task
window. Window position 0 is left out of every loss: a window's first position carries the attention-sink norm.

Ablation at block l: at the answer position only, the residual read by block l's readout has its component along a unit
vector removed (h <- h - (h.u) u): the readout's column for that window's target value (the claim's direction), or a
seeded random unit vector per target value (the do-nothing).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from commute.elicit import absdiff, separation, share_or_nan
from commute.probe import BATCH, DECAY, DECAY_AT, ITERS, LR, Probes
from commute.steering import DW, Net, Task, read

RESID = ("in1", "in2", "in3", "out3")  # block l's output residual = the `steering.Net` site RESID[l]
LAST = len(RESID) - 1
WIN, POS0 = 41, 1  # window length (E54's task windows); first position scored


def pile_windows(rows: Tensor, n: int, gen: torch.Generator) -> Tensor:
    """n windows of WIN tokens at seeded random (row, start) pairs."""
    r = torch.randint(0, rows.shape[0], (n,), generator=gen)
    s = torch.randint(0, rows.shape[1] - WIN + 1, (n,), generator=gen)
    return rows[r[:, None], s[:, None] + torch.arange(WIN)]


def resid(net: Net, toks: Tensor) -> tuple[Tensor, Tensor]:
    """(model logits (B, T, V), block outputs (len(RESID), B, T, d)) at W0."""
    logits, c = net.run(toks, capture=True)
    return logits, torch.stack([c[s] for s in RESID])


def next_token_xent(logits: Tensor, toks: Tensor) -> Tensor:
    """(k, B, T, V) logits -> (k,) mean cross-entropy of token t+1 at positions POS0..T-2."""
    lg = logits[:, :, POS0:-1]
    y = toks[:, POS0 + 1 :].to(lg.device).expand(lg.shape[0], -1, -1)
    return F.cross_entropy(lg.flatten(0, 2), y.flatten(), reduction="none").view(lg.shape[0], -1).mean(1)


def train_readouts(net: Net, rows: Tensor, seed: int, max_minutes: float, log: Callable[[str], None]) -> Probes:
    """Jain's schedule; exits if a logged loss is not finite or step 20 projects past max_minutes."""
    torch.manual_seed(seed)
    V, d = net.net.lm_head.weight.shape
    probes = Probes(len(RESID), d, V).to(net.device)
    opt = torch.optim.AdamW(probes.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(DECAY_AT), gamma=DECAY)
    gen, t0 = torch.Generator().manual_seed(seed), time.time()
    for step in range(ITERS):
        toks = pile_windows(rows, BATCH, gen).to(net.device)
        loss = next_token_xent(probes(resid(net, toks)[1].flatten(1, 2)).unflatten(1, (BATCH, WIN)), toks)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == ITERS - 1:
            log(f"[readout] step {step} loss per block {[round(float(x), 4) for x in loss]}")
            if not bool(torch.isfinite(loss).all()):
                raise SystemExit("[readout] non-finite loss: stop and tell the user")
        if step == 20 and (time.time() - t0) / 21 * ITERS / 60 > max_minutes:
            raise SystemExit(f"[readout] step 20 projects over {max_minutes} min: stop and tell the user")
    return probes


@torch.no_grad()
def held_out_loss(net: Net, probes: Probes, rows: Tensor, n: int, seed: int) -> dict[str, Any]:
    """Next-token loss of each block's readout and of the model itself on the same n held-out windows and positions."""
    toks = pile_windows(rows, n, torch.Generator().manual_seed(seed))
    ro, own = [], []
    for s in range(0, n, 100):
        t = toks[s : s + 100].to(net.device)
        logits, h = resid(net, t)
        ro.append(next_token_xent(probes(h.flatten(1, 2)).unflatten(1, (len(t), WIN)), t) * len(t))
        own.append(next_token_xent(logits[None], t) * len(t))
    return {"readout": (torch.stack(ro).sum(0) / n).tolist(), "model": float(torch.stack(own).sum() / n)}


@torch.no_grad()
def target_ranks(net: Net, probes: Probes, t: Task) -> Tensor:
    """(len(RESID), N) rank of each window's target value under each block's readout at the answer position, at W0;
    rank = 1 + the number of tokens scored strictly higher (`elicit.pick_values`'s rank)."""
    logits = probes(resid(net, t.toks)[1][:, :, t.half])
    tgt = t.tgt(t.toks).to(logits.device)
    own = logits.gather(2, tgt[None, :, None].expand(logits.shape[0], -1, 1))
    return (1 + (logits > own).sum(-1)).cpu()


def target_dirs(probes: Probes, block: int, tgt: Tensor) -> Tensor:
    """(N, d) unit vectors: block's readout column for each window's target value."""
    U = probes.w[block].detach()[:, tgt.to(probes.w.device)].T
    return U / U.norm(dim=1, keepdim=True)


def random_dirs(tgt: Tensor, d: int, seed: int) -> Tensor:
    """(N, d) unit vectors: one seeded Gaussian direction per distinct target value (sorted order), shared by its windows."""
    vals = sorted(set(tgt.tolist()))
    R = torch.randn(len(vals), d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    R = (R / R.norm(dim=1, keepdim=True)).float()
    return R[torch.tensor([vals.index(v) for v in tgt.tolist()])]


def remove_at(pos: int, U: Tensor) -> Callable[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        x, u = x.clone(), U.to(x.device, x.dtype)
        h = x[:, pos]
        x[:, pos] = h - (h * u).sum(-1, keepdim=True) * u
        return x

    return f


def ablation(net: Net, t: Task, dW: DW | None, block: int, U: Tensor, floor: float) -> dict[str, float]:
    """Scoring half. `loss` = mean (log P_state - log P_ablated); G = the state's gain over W0; A = loss / G, nan if
    |G| < floor; `median_drop` = the per-window median of the same difference (W0's positive control reads it)."""
    sc = t.part("score")
    state = read(net, t, t.toks, dW)[sc]  # all windows, sliced, as `steering.gains` reads them
    abl = read(net, t, t.toks[sc], dW, edit={RESID[block]: remove_at(t.half, U[sc])})
    G, drop = float((state - t.lp0[sc]).mean()), state - abl
    return {"G_score": G, "loss": float(drop.mean()), "median_drop": float(drop.median()),
            "A": share_or_nan(float(drop.mean()) / G if G else float("nan"), G, floor)}  # fmt: skip


# ----------------------------------------------------------------------------- filed rules (pure; tests/test_readout.py)


def key_medians(ranks: list[int], keys: list[int], all_keys: list[int]) -> dict[int, int | None]:
    """Per key, the lower median of its windows' ranks (torch.median's convention); None for a key with no window."""
    out: dict[int, int | None] = {}
    for k in all_keys:
        v = sorted(r for r, x in zip(ranks, keys, strict=True) if x == k)
        out[k] = v[(len(v) - 1) // 2] if v else None
    return out


def possession(med: Mapping[int, int | None], max_rank: int, need: int, of: int) -> dict[str, Any]:
    """Possessed iff count / covered >= need / of (inclusive; 24 of 32 when every key has a window), count = keys with
    median rank <= max_rank, covered = keys with a window; no key covered -> not possessed."""
    covered = [m for m in med.values() if m is not None]
    count = sum(m <= max_rank for m in covered)
    return {"count": count, "covered": len(covered), "possessed": bool(covered) and count * of >= need * len(covered)}


def earliest(poss: list[bool]) -> int | None:
    """l*: the earliest block before the last that possesses; None if none."""
    return next((b for b in range(LAST) if poss[b]), None)


def band_count(med: Mapping[int, int | None], lo: int, hi: int) -> int:
    return sum(m is not None and lo <= m <= hi for m in med.values())


def above_count(med: Mapping[int, int | None], lo: int) -> int:
    return sum(m is not None and m > lo for m in med.values())


def p1(elicit: list[bool], cipher: list[bool]) -> dict[str, Any]:
    """P1: the elicit map possessed at some block before the last (all held-out windows), the cipher at no block."""
    e, c = [b for b, p in enumerate(elicit) if p], [b for b, p in enumerate(cipher) if p]
    early = [b for b in e if b < LAST]
    if c:
        reading = f"cipher possessed at blocks {c}"
    elif early:
        reading = "claim"
    elif e:
        reading = "null: elicit at the last block only"
    else:
        reading = "elicit possessed at no block"
    return {"elicit_blocks": e, "cipher_blocks": c, "status": "pass" if early and not c else "fail", "reading": reading}


def p3(elicit_last: Mapping[int, int | None], cipher_last: Mapping[int, int | None], band: tuple[int, int], need: int,
       cipher_above: int) -> dict[str, Any]:
    """P3 at the last block: >= need keys with elicit median rank in band (inclusive) and >= need keys with cipher
    median rank > cipher_above (strict)."""
    e, c = band_count(elicit_last, *band), above_count(cipher_last, cipher_above)
    return {"elicit_in_band": e, "cipher_above": c, "status": "pass" if e >= need and c >= need else "fail"}


def do_nothing(A_rand: dict[str, float | None], tol: float) -> dict[str, Any]:
    """Every state's random-direction share within +-tol (inclusive); a missing or non-finite one fails."""
    vals = list(A_rand.values())
    ok = bool(vals) and all(v is not None and np.isfinite(v) and abs(v) <= tol for v in vals)
    return {"max_abs": max((abs(v) for v in vals if v is not None and np.isfinite(v)), default=None), "pass": ok}


def p2(A: dict[str, float | None], seeds: list[str], own_seeds: list[str], min_gap: float, own_tol: float,
       void: str | None) -> dict[str, Any]:
    """P2: elicit A above matched-cipher A by E51's rule (`elicit.separation`), and |A(cipher_own) - A(cipher)| < own_tol
    (strict) on every own seed. `void` (no l*, or W0's control failed) -> "void"; a None or nan -> "unreadable"."""
    if void:
        return {"status": "void", "reason": void}
    sep = separation([A.get(f"elicit_{s}") for s in seeds], [A.get(f"cipher_{s}") for s in seeds], min_gap)
    diffs = {s: absdiff(A.get(f"cipher_own_{s}"), A.get(f"cipher_{s}")) for s in own_seeds}
    out = {"separation": sep, "own_diff": diffs}
    if sep["verdict"] == "unreadable" or any(d is None for d in diffs.values()):
        return {**out, "status": "unreadable"}
    ok = sep["verdict"] == "a_higher" and all(float(d or 0.0) < own_tol for d in diffs.values())
    return {**out, "status": "pass" if ok else "fail"}


def branch(a: dict[str, Any], b: dict[str, Any]) -> str:
    """The plan's decision tree on P1 and P2 (the controls stop the run before this)."""
    if b["status"] == "void":
        return f"P2 void ({b['reason']}); P1 {a['status']} ({a['reading']}): report, not scored; tell the user"
    if b["status"] == "unreadable":
        return "P2 unreadable (a missing state, a gain under the floor or a nan): stop and tell the user"
    if b["status"] == "fail":
        return "P2 fails: stop and tell the user; the graded ladder (b) before any Qwen ask"
    if a["status"] == "pass":
        return "P1 and P2 hold: the label is independent; write up, ask (b) and carry the label to the Qwen pair"
    return f"P1 fails ({a['reading']}), P2 holds: 'half-known' is the output's knowledge; the ablation still separates"

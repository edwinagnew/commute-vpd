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
seeded random unit vector per target value (the do-nothing). The scored share is in log-odds, net of W0's own drop under
the same removal (`shares`; amended 2026-11-02 before the run, review reports/reviews/2026-11-02_review_e59-code.md H1,
M1).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from commute.elicit import absdiff, separation, share_or_nan
from commute.keydep import geo_rate, sha256
from commute.probe import BATCH, DECAY, DECAY_AT, ITERS, LR, Probes
from commute.steering import DW, Edit, Net, Task, gains, logodds_at, logp_at

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


def early_miss(step: int, last: float, model: float, checks: Mapping[str, Mapping[str, float]]) -> str | None:
    """The filed early checks (amended 2026-11-02 after job 48279's stop): at a step named in `checks`, the last block's
    train-batch loss must be <= "max" (absolute) and/or <= the model's own loss on that batch + "max_over_model"
    (inclusive). Returns the miss, or None."""
    c = checks.get(str(step), {})
    if "max" in c and not last <= c["max"]:
        return f"[readout] step {step}: last-block loss {last:.4f} over {c['max']}: stop and tell the user"
    if "max_over_model" in c and not last <= model + c["max_over_model"]:
        return f"[readout] step {step}: last-block loss {last:.4f} over model {model:.4f} + {c['max_over_model']}: stop and tell the user"
    return None


def train_readouts(net: Net, rows: Tensor, seed: int, max_minutes: float, checks: Mapping[str, Mapping[str, float]],
                   log: Callable[[str], None]) -> Probes:
    """Jain's schedule. Logs every 250 steps the readouts' and the model's own loss on the batch; raises ReadoutStop on
    a non-finite loss, a missed early check (`early_miss`), or a step-20 projection past max_minutes (always logged)."""
    torch.manual_seed(seed)
    V, d = net.net.lm_head.weight.shape
    probes = Probes(len(RESID), d, V).to(net.device)
    opt = torch.optim.AdamW(probes.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(DECAY_AT), gamma=DECAY)
    gen, t0 = torch.Generator().manual_seed(seed), time.time()
    for step in range(ITERS):
        toks = pile_windows(rows, BATCH, gen).to(net.device)
        logits, h = resid(net, toks)
        loss = next_token_xent(probes(h.flatten(1, 2)).unflatten(1, (BATCH, WIN)), toks)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == ITERS - 1:
            ro, own = loss.detach().tolist(), float(next_token_xent(logits[None], toks)[0])
            log(f"[readout] step {step} loss per block {[round(x, 4) for x in ro]} model {own:.4f}")
            miss = None if all(map(math.isfinite, ro)) else "[readout] non-finite loss: stop and tell the user"
            if miss := miss or early_miss(step, ro[LAST], own, checks):
                raise ReadoutStop(miss)
        if step == 20:
            proj = (time.time() - t0) / 21 * ITERS / 60
            log(f"[readout] step 20 projects {proj:.1f} min for {ITERS} steps (bound {max_minutes})")
            if proj > max_minutes:
                raise ReadoutStop(f"[readout] step 20 projects {proj:.1f} min, over {max_minutes}: stop and tell the user")
    return probes


@torch.no_grad()
def norm_scale(net: Net, rows: Tensor, n: int, seed: int) -> float:
    """The final RMSNorm's per-token scale 1 / rms(h) on the last block's output, averaged over n training batches of
    BATCH windows at positions POS0..T-2 (the linearised head's constant)."""
    gen, eps, s = torch.Generator().manual_seed(seed), net.net.ln_f.variance_epsilon, []
    for _ in range(n):
        h = resid(net, pile_windows(rows, BATCH, gen))[1][LAST][:, POS0:-1]
        s.append(torch.rsqrt(h.pow(2).mean(-1) + eps).mean())
    return float(torch.stack(s).mean())


@torch.no_grad()
def losses(net: Net, probes: Probes, rows: Tensor, n: int, seed: int, scale: float) -> dict[str, Any]:
    """Next-token loss on the same n windows and positions of: each block's readout, the model itself, and the
    linearised head ("lin": the model's unembedding with the final norm's per-token scale replaced by `scale`)."""
    toks = pile_windows(rows, n, torch.Generator().manual_seed(seed))
    w, W = net.net.ln_f.weight, net.net.lm_head.weight
    ro, own, lin = [], [], []
    for s in range(0, n, 100):
        t = toks[s : s + 100].to(net.device)
        logits, h = resid(net, t)
        ro.append(next_token_xent(probes(h.flatten(1, 2)).unflatten(1, (len(t), WIN)), t) * len(t))
        own.append(next_token_xent(logits[None], t) * len(t))
        lin.append(next_token_xent(((h[LAST] * scale * w) @ W.T)[None], t) * len(t))
    tot = lambda x: torch.stack(x).sum(0) / n
    return {"readout": tot(ro).tolist(), "model": float(tot(own)[0]), "lin": float(tot(lin)[0])}


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


class ReadoutStop(Exception):
    """A guard inside readout training fired; `run` records it as run.json's `stopped` before exiting."""


def _reads(net: Net, t: Task, toks: Tensor, dW: DW | None, edit: Edit | None) -> tuple[Tensor, Tensor]:
    """log P(target) and log-odds of the target at the answer position, from one forward."""
    with net.at(dW):
        logits = net.run(toks, edit)[0]
    tgt = t.tgt(toks)
    return logp_at(logits, t.half, tgt), logodds_at(logits, t.half, tgt)


def ablation(net: Net, t: Task, dW: DW | None, block: int, U: Tensor, floor: float) -> dict[str, float]:
    """Scoring half, every reading on the same 48 windows: W0 and the state (dW None: W0 itself), each with and without
    the removal at `block`; the numbers of `shares`."""
    sc = t.part("score")
    toks, edit = t.toks[sc], {RESID[block]: remove_at(t.half, U[sc])}
    w0, w0_abl = _reads(net, t, toks, None, None), _reads(net, t, toks, None, edit)
    st, st_abl = (w0, w0_abl) if dW is None else (_reads(net, t, toks, dW, None), _reads(net, t, toks, dW, edit))
    return shares(w0, w0_abl, st, st_abl, floor)


def shares(w0: tuple[Tensor, Tensor], w0_abl: tuple[Tensor, Tensor], st: tuple[Tensor, Tensor],
           st_abl: tuple[Tensor, Tensor], floor: float) -> dict[str, float]:
    """Each argument is (log P, log-odds) per window. Scored (amended 2026-11-02 pre-run, review H1 + M1): A_lo = (the
    state's mean log-odds drop under the removal - W0's, same windows and direction) / G_logodds, nan if |G_logodds| <
    floor; a negative G_logodds gives a well-defined share, scored as is. Recorded, unscored: A_nats (the first filing,
    nats lost / G in nats), A_lo_raw (log-odds, W0's drop not subtracted), nats_lost, P_tuned, and the median nats drop
    (W0's positive control reads it at dW None)."""
    m = lambda x: float(x.mean())
    G, G_lo = m(st[0] - w0[0]), m(st[1] - w0[1])
    nats, lo, lo0 = m(st[0] - st_abl[0]), m(st[1] - st_abl[1]), m(w0[1] - w0_abl[1])
    ratio = lambda num, den: share_or_nan(num / den if den else float("nan"), den, floor)
    return {"G_score": G, "G_logodds": G_lo, "nats_lost": nats, "median_drop": float((st[0] - st_abl[0]).median()),
            "lo_drop": lo, "lo_drop_W0": lo0, "nats_drop_W0": m(w0[0] - w0_abl[0]),
            "P_tuned_mean": m(st[0].exp()), "P_tuned_geo": float(np.exp(m(st[0]))),
            "A_nats": ratio(nats, G), "A_lo_raw": ratio(lo, G_lo), "A_lo": ratio(lo - lo0, G_lo)}  # fmt: skip


def column_geometry(probes: Probes, block: int, values: list[int]) -> dict[str, dict[str, float]]:
    """Per value: its readout column's norm and cosine with the mean column of the whole vocabulary (review L1: softmax
    ignores a component shared by every column, so a column close to the mean is a direction that moves every logit)."""
    W = probes.w[block].detach()
    mean = W.mean(1)
    return {str(v): {"norm": float(W[:, v].norm()), "cos_mean": float(F.cosine_similarity(W[:, v], mean, dim=0))}
            for v in values}  # fmt: skip


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
    """P2 on A = each state's A_lo: elicit above matched cipher by E51's rule (`elicit.separation`), and
    |A(cipher_own) - A(cipher)| < own_tol (strict) on every own seed. `void` (no l*, or W0's control failed) -> "void"; a None or nan -> "unreadable"."""
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


# ----------------------------------------------------------------------------- E64: the readout over the pair's 64 values
# Additive: everything above stays as E59 ran it. Plan: reports/plans/2026-11-02_plan_baseline-next3-draft.md, option 1
# and its "Amendment before filing (2026-11-02)"; rules and thresholds in experiments/readout64/results/predictions.json;
# tests in tests/test_readout64.py. The readout is `Probes` with 64 outputs: it reads the residual at one position of a
# WIN-token Pile window, the position where the task windows ask for the value, and predicts the next token among the 64.


def pair_values(tasks: Mapping[str, Task]) -> Tensor:
    """The (64,) vocabulary ids: each task's values in its key order, elicit then cipher; readout outputs are in this order."""
    vals = [int(t.table[k]) for t in tasks.values() for k in t.keys]
    assert len(set(vals)) == len(vals), "the pair's values are unique by construction"
    return torch.tensor(vals)


def next_token_windows(rows: Tensor, vals: Tensor, half: int, cap: int, seed: int) -> tuple[Tensor, Tensor, list[int]]:
    """Every WIN-token window of `rows` whose token after index `half` is one of `vals`, at most `cap` per value, drawn
    with a seeded permutation: (M, WIN) tokens, (M,) index into vals, and each value's count before the cap."""
    starts = rows[:, half + 1 : half + 1 + rows.shape[1] - WIN + 1]  # entry (r, c) is the next token of the window at c
    gen, toks, ys, counts = torch.Generator().manual_seed(seed), [], [], []
    for i, v in enumerate(vals.tolist()):
        r, c = (starts == v).nonzero(as_tuple=True)
        keep = torch.randperm(len(r), generator=gen)[:cap]
        counts.append(len(r))
        toks.append(rows[r[keep][:, None], c[keep][:, None] + torch.arange(WIN)])
        ys.append(torch.full((len(keep),), i))
    return torch.cat(toks), torch.cat(ys), counts


def cache64(net: Net, toks: Tensor, vals: Tensor, half: int, batch: int = 256) -> tuple[Tensor, Tensor]:
    """W0's residuals (len(RESID), M, d) and its logits over the values (M, m) at position `half`, on the net's device."""
    hs, lg = [], []
    for s in range(0, len(toks), batch):
        logits, h = resid(net, toks[s : s + batch])
        hs.append(h[:, :, half])
        lg.append(logits[:, half][:, vals.to(logits.device)])
    return torch.cat(hs, 1), torch.cat(lg)


def xent64(logits: Tensor, y: Tensor) -> list[float]:
    """(k, M, m) logits -> each block's mean cross-entropy against (M,) y."""
    k, M = logits.shape[:2]
    return F.cross_entropy(logits.flatten(0, 1), y.expand(k, -1).flatten(), reduction="none").view(k, M).mean(1).tolist()


def early_miss64(step: int, last: list[float], model: list[float], checks: Mapping[str, Any]) -> str | None:
    """The filed early checks, read off a batch sd: at a step in checks["bars"], over its last checks["window"] batches,
    x = the last block's loss ("max") or its loss minus the model's on the same batch ("max_over_model"); miss iff
    mean(x) - z * sd(x) / sqrt(window) is not <= the bar, so one noisy batch cannot stop a job (E59's job 48344) and a nan
    never passes. Fewer than two batches to read is a miss: the sd is undefined."""
    bar, k = checks["bars"].get(str(step)), checks["window"]
    if bar is None:
        return None
    if len(last) < 2:
        return f"[readout] step {step}: fewer than two batches to read: stop and tell the user"
    a, b = np.array(last[-k:]), np.array(model[-k:])
    for name, x in (("max", a), ("max_over_model", a - b)):
        if name in bar:
            mean, se = float(x.mean()), float(x.std(ddof=1)) / math.sqrt(len(x))
            if not mean - checks["z"] * se <= bar[name]:
                return f"[readout] step {step}: {name} {mean:.4f} (se {se:.4f}, {len(x)} batches) over {bar[name]}: stop and tell the user"
    return None


def train_readouts64(h: Tensor, y: Tensor, w0: Tensor, seed: int, checks: Mapping[str, Any], log: Callable[[str], None]) -> Probes:
    """Jain's schedule on cached residuals (`train_readouts`' optimiser, one position per window). Logs every 250 steps;
    raises ReadoutStop on a non-finite loss or a missed early check."""
    torch.manual_seed(seed)
    probes = Probes(h.shape[0], h.shape[2], w0.shape[1]).to(h.device)
    opt = torch.optim.AdamW(probes.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(DECAY_AT), gamma=DECAY)
    gen, y, last, model = torch.Generator().manual_seed(seed), y.to(h.device), [], []
    for step in range(ITERS):
        i = torch.randint(0, h.shape[1], (BATCH,), generator=gen).to(h.device)
        lg = probes(h[:, i])
        loss = F.cross_entropy(lg.flatten(0, 1), y[i].expand(h.shape[0], -1).flatten()) * h.shape[0]  # sum over blocks
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        last.append(xent64(lg[LAST:].detach(), y[i])[0])
        model.append(xent64(w0[i][None], y[i])[0])
        miss = None if math.isfinite(float(loss)) else "[readout] non-finite loss: stop and tell the user"
        if miss := miss or early_miss64(step, last, model, checks):
            raise ReadoutStop(miss)
        if step % 250 == 0 or step == ITERS - 1:
            log(f"[readout] step {step} loss per block {[round(x, 4) for x in xent64(lg.detach(), y[i])]} model {model[-1]:.4f}")
    return probes


@torch.no_grad()
def readout_logits64(net: Net, probes: Probes, t: Task) -> Tensor:
    """(len(RESID), N, m) each block's readout over the values at the answer position, at W0."""
    return probes(resid(net, t.toks)[1][:, :, t.half])


def rank_among(lg: Tensor, idx: Tensor) -> Tensor:
    """(N, m) logits, (N,) target index -> rank among the m, 1 + the number scored strictly higher (ties share a rank)."""
    return 1 + (lg > lg.gather(1, idx[:, None])).sum(1)


def specific_log_ratio(lp: list[float], keys_of: list[int], key: int) -> float | None:
    """log P(value) at the key's windows minus at the other keys' windows, each the lower median over its windows of log
    P: the log of "P at the key over its median elsewhere", with no denominator to vanish. None if either side has none."""
    own = sorted(x for x, k in zip(lp, keys_of, strict=True) if k == key)
    rest = sorted(x for x, k in zip(lp, keys_of, strict=True) if k != key)
    return own[(len(own) - 1) // 2] - rest[(len(rest) - 1) // 2] if own and rest else None


def possessed_keys(med: Mapping[int, int | None], ratio: Mapping[int, float | None], max_rank: int, min_ratio: float,
                   unfit: Sequence[int] = ()) -> list[int]:
    """Keys with median rank <= max_rank and P ratio >= min_ratio (both inclusive), not unfit. A key with no window, or
    with no other key's window, is out; a nan ratio is out."""
    ok = lambda k, m: m is not None and m <= max_rank and (r := ratio[k]) is not None and r >= math.log(min_ratio)
    return [k for k, m in med.items() if ok(k, m) and k not in unfit]


def key_reading(lg: Tensor, keys_of: list[int], keys: Sequence[int], vidx: Mapping[int, int], th: Mapping[str, Any],
                unfit: Sequence[int] = ()) -> dict[str, Any]:
    """One pool of windows through the rank and specificity rules. (N, m) logits; `vidx` key -> its value's output index."""
    lp = torch.log_softmax(lg.double(), -1).cpu()
    ranks = rank_among(lg, torch.tensor([vidx[k] for k in keys_of], device=lg.device)).tolist()
    med = key_medians(ranks, keys_of, list(keys))
    ratio = {k: specific_log_ratio(lp[:, vidx[k]].tolist(), keys_of, k) for k in keys}
    return {"ranks": ranks, "medians": med, "log_ratio": ratio,
            "possessed": possessed_keys(med, ratio, th["possess_rank"], th["specific_ratio"], unfit)}  # fmt: skip


def relative_possession(own: Sequence[int], ref: Sequence[int], num: int, den: int) -> dict[str, Any]:
    """The elicit map at a block: at least num/den of the model's possessed keys `ref` (inclusive). An empty ref possesses
    nothing; keys outside ref are not counted."""
    count = len(set(own) & set(ref))
    return {"count": count, "ref": len(ref), "outside_ref": len(set(own) - set(ref)), "possessed": bool(ref) and count * den >= num * len(ref)}


def cipher_possession(own: Sequence[int], ref_size: int, num: int, den: int) -> dict[str, Any]:
    """The cipher at a block: as many keys as the elicit map would need (num/den of the elicit reference's size, inclusive),
    since the cipher's own reference is empty or nearly so and cannot be a denominator. An empty elicit reference: not
    possessed."""
    return {"count": len(own), "possessed": ref_size > 0 and len(own) * den >= num * ref_size}


def model_stop(n_elicit: int, n_cipher: int, min_elicit: int, max_cipher: int) -> str | None:
    """Why stage 0 stops on W0's own possessed sets, or None (both bounds inclusive: min_elicit and max_cipher go on)."""
    if n_elicit < min_elicit:
        return f"[stage0 model] elicit |M| {n_elicit} under {min_elicit}: the pair cannot carry the reading at this rule; stop and tell the user"
    if n_cipher > max_cipher:
        return f"[stage0 model] cipher |M| {n_cipher} over {max_cipher}: the construction check cannot pass; stop and tell the user"
    return None


def unfit_stop(unfit: Mapping[str, Sequence[int]], ref: Sequence[int], max_unfit: int, num: int, den: int) -> str | None:
    """Why stage 0 cannot go on before training, or None: over max_unfit unfit values in a task (exclusive), or so many
    of the model's keys unfit that no readout could reach num/den of them."""
    for n, u in unfit.items():
        if len(u) > max_unfit:
            return f"[stage0 positions] {len(u)} {n} values unfit (keys {list(u)}), over {max_unfit}: stop and tell the user"
    left = [k for k in ref if k not in unfit["elicit"]]
    if ref and len(left) * den < num * len(ref):
        return f"[stage0 positions] only {len(left)} of the model's {len(ref)} elicit keys have fit values: stop and tell the user"
    return None


def construction(elicit_last: Mapping[str, Any], cipher_last: Mapping[str, Any], cipher_ref: Sequence[int], max_cipher_ref: int) -> dict[str, Any]:
    """The construction check (plan's P3, amended): the last block possesses the elicit map relative to the model, the
    model's own cipher set has at most max_cipher_ref keys (inclusive), and the last block does not possess the cipher."""
    ok = elicit_last["possessed"] and len(cipher_ref) <= max_cipher_ref and not cipher_last["possessed"]
    return {"elicit_last": dict(elicit_last), "cipher_last": dict(cipher_last), "cipher_ref": len(cipher_ref), "status": "pass" if ok else "fail"}


def w0_control(drops: list[float], keys_of: list[int], ref: Sequence[int], min_drop: float) -> dict[str, Any]:
    """W0's positive control: the lower median log P drop over the windows of the model's possessed keys >= min_drop
    (inclusive). None of them among these windows: not passed."""
    v = sorted(d for d, k in zip(drops, keys_of, strict=True) if k in ref)
    med = v[(len(v) - 1) // 2] if v else None
    return {"median_drop": med, "windows": len(v), "pass": med is not None and med >= min_drop}


def w0_drops(net: Net, t: Task, block: int, U: Tensor) -> list[float]:
    """Scoring half: W0's log P(target) minus the same with the removal at `block`, per window."""
    sc = t.part("score")
    toks = t.toks[sc]
    return (_reads(net, t, toks, None, None)[0] - _reads(net, t, toks, None, {RESID[block]: remove_at(t.half, U[sc])})[0]).tolist()


def branch64(a: Mapping[str, Any], b: Mapping[str, Any], fit_pass: bool) -> str:
    """The plan's "changes next" on P1 and P2; a readout-fit miss turns an elicit-clause or P2 miss into unreadable."""
    early = any(x < LAST for x in a["elicit_blocks"])
    if not fit_pass and (not early or b["status"] == "fail"):
        return f"readout fit missed and P1 elicit clause or P2 missed (P1 {a['reading']}; P2 {b['status']}): unreadable; one training run was the cap; tell the user"
    if b["status"] == "void" and not early and a["elicit_blocks"]:
        return "P1 holds only at the last block: 'half-known' means what the output says; r's separation of this pair is a reading of the prior, written so"
    if b["status"] == "void":
        return f"P2 void ({b['reason']}); P1 {a['status']} ({a['reading']}): report, not scored; tell the user"
    if b["status"] == "unreadable":
        return "P2 unreadable (a missing state, a gain under the floor or a nan): stop and tell the user"
    if b["status"] == "fail":
        return "P2 fails: r's reading of this pair has no causal footing; the line pauses on the 67M and E55 C2 decides r's standing"
    if a["status"] == "pass":
        return "P1 and P2 hold: the pair's label is causal and VPD-independent; write up, and the Qwen KV table carries it"
    return f"P2 holds but P1 fails ({a['reading']}): not in the plan's tree; report and tell the user"


SEEDS, OWN_SEEDS = ("s0", "s1", "s2", "s3", "s4"), ("s0", "s1")  # E56 did not save cipher_own s2-s4


def saved_states(e54: Path, e56: Path) -> dict[str, dict[str, Any]]:
    """The twelve saved fine-tunes: path, filed sha256 and gain G (E54 stage2.json; E56 job2.json)."""
    s2, j2 = json.loads((e54 / "stage2.json").read_text()), json.loads((e56 / "job2.json").read_text())
    src = {f"{k}_{s}": (e54, s2) for k in ("elicit", "cipher", "cipher_own") for s in OWN_SEEDS}
    src |= {f"{k}_{s}": (e56, j2) for k in ("elicit", "cipher") for s in SEEDS[2:]}
    return {n: {"path": d / "large" / f"{n}.pt", "sha256": j["finetunes"][n]["sha256"], "G": j["states"][n]["G"]}
            for n, (d, j) in src.items()}  # fmt: skip


def check_inputs(net: Net, tasks: Mapping[str, Task], e54: Path, e56: Path, th: Mapping[str, float], rec: dict[str, Any],
                 log: Callable[[str], None]) -> dict[str, DW]:
    """E59's stage 0 inputs: the windows' sha256, keys, values and W0's rates reproduce E54; each state's sha256 and gain
    reproduce its record. Fills rec, rec["pass"] last; returns the states."""
    s1, dws = json.loads((e54 / "stage1.json").read_text()), {}
    for n, t in tasks.items():
        rate = geo_rate(t.lp0)
        rec[n] = {"sha256": sha256(e54 / "large" / f"windows_{n}.pt"), "rate": rate, "rate_rel": rate / s1[n]["rate"] - 1,
                  "keys_values": list(t.keys) == s1[n]["keys"] and [int(t.table[x]) for x in t.keys] == s1[n]["values"]}
        rec[n]["ok"] = bool(rec[n]["sha256"] == s1["sha256"][n] and rec[n]["keys_values"] and abs(rec[n]["rate_rel"]) <= th["rate_rel_tol"])
        log(f"[stage0 {n}] W0 rate {rate:.10e} rel {rec[n]['rate_rel']:.1e} ok {rec[n]['ok']}")
    for n, r in saved_states(e54, e56).items():
        dws[n] = torch.load(r["path"], weights_only=True)
        g = gains(net, tasks["elicit" if n.startswith("elicit") else "cipher"], dws[n])
        rec[n] = {**g, "sha256": sha256(r["path"]), "dG": g["G"] - r["G"]}
        rec[n]["ok"] = bool(rec[n]["sha256"] == r["sha256"] and abs(rec[n]["dG"]) <= th["reproduce_tol"])
        log(f"[stage0 {n}] G {g['G']:.12g} G_score {g['G_score']:.12g} dG {rec[n]['dG']:.1e} sha {rec[n]['sha256'][:12]} ok {rec[n]['ok']}")
    rec["pass"] = all(v["ok"] for v in rec.values() if isinstance(v, dict))
    return dws

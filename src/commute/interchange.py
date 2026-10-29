"""E55's two checks on the PCFG minGPT (plan reports/plans/2026-10-29_plan_reuse-stage1.md, Stage A, "The two checks"
and "Bars"; definitions in experiments/pcfg_anchors/results/predictions.json), and the filed rules that read them.

Everything reads or writes the residual stream at `=` (pcfg.ART_POS), the position that predicts the answer, at the
output of block l (1..n_layer; the embedding output is not a candidate). Forwards run in fp32 in eval mode, outputs
are argmax at `=` (the code review's nit on bf16 argmax ties).

- Behaviour B (f_C): on qualifying 'C b 40' inputs (count(b) != count(a) + 1), the share of outputs equal to
  g(f_C(x)): count_a(x) + 1 for count-a, count_b(x) for count-b.
- Patch P (f_C, l): pairs (x, x') of qualifying inputs with the same other count and a different f_C; the residual at
  `=` after block l in the run on x is replaced by the run on x'. P = share of pairs whose patched output equals the
  model's output on x' and that output differs from the one on x; denominator all pairs built. < 500 pairs: void.
- Rules: l* (earliest block within 3 s of the best), the bars (lo = n0 + 3 s, hi = n0 + 0.5 (p+ - n0)) and the label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from commute import pcfg
from commute.pcfg import A, B

N_PAIRS, MIN_PAIRS, N_B = 1000, 500, 1000  # filed: 1,000 pairs, void below 500; B on the first 1,000 qualifying
G_OFFSET = {A: 1, B: 0}  # g(count-a) = count_a + 1, g(count-b) = count_b: what the fine-tuning data rewards


def counter_seqs(texts: np.ndarray, op: int) -> np.ndarray:
    """'C <op> 40' sequences on the given texts."""
    n = len(texts)
    return pcfg.assemble(np.full(n, pcfg.FAMILIES.index("C")), np.full(n, op), texts)


def qualifying_texts(rng: np.random.Generator, n: int) -> np.ndarray:
    """n natural texts with count(b) != count(a) + 1 in the window, in stream order."""
    out, got = [], 0
    while got < n:
        t = pcfg.sample_texts(rng, int(1.4 * (n - got)) + 16)
        t = t[pcfg.count(t, B) != pcfg.count(t, A) + 1][: n - got]
        out.append(t)
        got += len(t)
    return np.concatenate(out)


@dataclass(frozen=True)
class Pairs:
    x: np.ndarray  # (n, TEXT) texts run and patched
    donor: np.ndarray  # (n, TEXT): same count of the other operand, different count of f_op


def pair_up(rng: np.random.Generator, texts: np.ndarray, f_op: int, n_pairs: int = N_PAIRS) -> Pairs | None:
    """Pairs for the f_op patch: each text in order, while fewer than n_pairs, gets a donor drawn uniformly from the
    texts with the same count of the other operand and a different count of f_op; a text with none is skipped.
    Returns None (void) when fewer than MIN_PAIRS can be built."""
    other = B if f_op == A else A
    f, o = pcfg.count(texts, f_op), pcfg.count(texts, other)
    xs, ds = [], []
    for i in range(len(texts)):
        cand = np.flatnonzero((o == o[i]) & (f != f[i]))
        if cand.size:
            xs.append(i)
            ds.append(int(rng.choice(cand)))
        if len(xs) == n_pairs:
            break
    if len(xs) < MIN_PAIRS:
        return None
    return Pairs(texts[xs], texts[ds])


def make_pairs(rng: np.random.Generator, f_op: int, pool: int = 4 * N_PAIRS) -> Pairs | None:
    return pair_up(rng, qualifying_texts(rng, pool), f_op)


def forward_at_art(model: nn.Module, idx: torch.Tensor, block: int, patch: torch.Tensor | None) -> tuple[torch.Tensor, list[torch.Tensor]]:  # fmt: skip
    """Logits at the last position and the residual there after every block; with `patch`, that residual after block
    `block` (1-based) is replaced by `patch` before the next block reads it."""
    resid: list[torch.Tensor] = []

    def hook(_m: nn.Module, _inp: object, out: torch.Tensor) -> torch.Tensor:
        if patch is not None and len(resid) == block - 1:
            out = out.clone()
            out[:, -1] = patch
        resid.append(out[:, -1].float())
        return out

    handles = [blk.register_forward_hook(hook) for blk in model.blocks]  # type: ignore[union-attr]
    try:
        logits = model(idx)[:, -1].float()
    finally:
        for h in handles:
            h.remove()
    return logits, resid


@torch.no_grad()
def run_at_art(model: nn.Module, seq: np.ndarray, *, block: int = 0, patch: torch.Tensor | None = None,
               chunk: int = 1000) -> tuple[np.ndarray, torch.Tensor]:  # fmt: skip
    """(argmax output at `=` per row, residual at `=` after every block as (n_layer, n, d)), fp32, eval mode.
    With block = l and patch (n, d), the residual at `=` after block l is replaced by patch's rows."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    outs, res = [], []
    for i in range(0, len(seq), chunk):
        idx = torch.from_numpy(seq[i : i + chunk, : pcfg.ANS].astype(np.int64)).to(device)
        logits, resid = forward_at_art(model, idx, block, None if patch is None else patch[i : i + chunk])
        outs.append(logits.argmax(-1).cpu().numpy())
        res.append(torch.stack(resid))
    model.train(was_training)
    return np.concatenate(outs), torch.cat(res, 1)


def behaviour_share(model: nn.Module, texts: np.ndarray, f_op: int) -> float:
    """B: share of 'C b 40' outputs equal to g(f_C(x)) on the given (qualifying) texts. Empty -> nan (void)."""
    if len(texts) == 0:
        return float("nan")
    out, _ = run_at_art(model, counter_seqs(texts, B))
    return float((out == pcfg.NUM0 + pcfg.count(texts, f_op) + G_OFFSET[f_op]).mean())


def patch_shares(model: nn.Module, pairs: Pairs | None, prefix_op: int) -> np.ndarray:
    """P at every block (n_layer,), on '<prefix_op>' sequences; all nan when the pairs are void."""
    n_layer = len(model.blocks)  # type: ignore[arg-type]
    if pairs is None:
        return np.full(n_layer, np.nan)
    x, d = counter_seqs(pairs.x, prefix_op), counter_seqs(pairs.donor, prefix_op)
    out_x, _ = run_at_art(model, x)
    out_d, res_d = run_at_art(model, d)
    shares = []
    for blk in range(1, n_layer + 1):
        out_p, _ = run_at_art(model, x, block=blk, patch=res_d[blk - 1])
        shares.append(float(((out_p == out_d) & (out_d != out_x)).mean()))
    return np.array(shares)


# ---- filed rules (pure arithmetic; tests/test_interchange.py) ----


def binomial_sd(p: float, n: int) -> float:
    """Binomial sd of a share at n, p clipped to [1/n, 1 - 1/n] (edge cases E2, E3)."""
    q = min(max(p, 1 / n), 1 - 1 / n)
    return float(np.sqrt(q * (1 - q) / n))


def spread(sets: np.ndarray, n: int) -> float:
    """s = max(sd over the independent sets (ddof 1), binomial sd of their mean at n, clipped)."""
    return max(float(np.std(sets, ddof=1)), binomial_sd(float(np.mean(sets)), n))


def l_star(positive_sets: np.ndarray, n: int = N_PAIRS) -> int:
    """Earliest block (1-based) whose positive-control P mean is >= max over blocks - 3 s, s that block's spread
    (positive_sets: (sets, n_layer)). Edge case E8."""
    mean = positive_sets.mean(0)
    assert np.isfinite(mean).all(), f"non-finite positive-control P: {mean}"
    s = np.array([spread(positive_sets[:, j], n) for j in range(mean.size)])
    return int(np.flatnonzero(mean >= mean.max() - 3 * s)[0]) + 1


Bar = dict[str, float | bool]  # n0, s, p_plus, lo, hi, void


def bar(null_sets: np.ndarray, p_plus: float, n: int = N_B) -> Bar:
    """lo = n0 + 3 s, hi = n0 + 0.5 (p_plus - n0); void iff not hi > lo (E10; a nan anywhere voids, E6)."""
    sets = np.asarray(null_sets, float)
    n0 = float(sets.mean())
    s = spread(sets, n) if np.isfinite(sets).all() else float("nan")
    lo, hi = n0 + 3 * s, n0 + 0.5 * (p_plus - n0)
    return {"n0": n0, "s": s, "p_plus": float(p_plus), "lo": lo, "hi": hi, "void": not hi > lo}


def label(b: float, p: float, bar_b: Bar, bar_p: Bar) -> str:
    """'through' iff B > hi_B and P > hi_P; 'not through' iff B < lo_B and P < lo_P; else 'neither'. Strict (E5); a
    void bar or a nan reading gives 'neither' (E6, E7)."""
    if bar_b["void"] or bar_p["void"]:
        return "neither"
    if b > bar_b["hi"] and p > bar_p["hi"]:
        return "through"
    if b < bar_b["lo"] and p < bar_p["lo"]:
        return "not through"
    return "neither"

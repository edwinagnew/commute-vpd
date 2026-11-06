"""E102's neuron readings on Qwen3-1.7B's SwiGLU MLPs (experiments/qwen_carrying; plan
reports/plans/2026-11-05_plan_e102-qwen-carrying.md). Target model only, fp32, no decomposition read.

A neuron is one of the 6144 inputs of a block's down_proj: its firing is a_i = silu(g_i) u_i, g = W_gate x, u = W_up x
(`qwen_vpd.py:50,52`; transformers' `modeling_qwen3.py` Qwen3MLP.forward), 28 x 6144 = 172,032, flat index l x 6144 + i. The hooks are E98's (`neuron_r.hooked`,
`neuron_r.firing_grads`); the cells are E98's on Qwen's rows (`qwen_comparators.Rows`), with the readout E92's: log P of
the next token at every answer position (`qwen_comparators.token_reads`), not E98's log-odds at one position.

  T0, T1     W0's and the state's log P at each answer position; the gain G is the mean of T1 - T0 (E92's `gain.all`)
  Ah         W0's gate, up and SiLU applied to the MLP input the state produces (E98's r_half firing)
  score      per neuron, sum over rows and positions of W0's gradient of the summed T0 x (Ah - A0): E98's ranking, kept
             per row parity (even, odd) so a set ranked on one half can be read against the other
  nums[k]    mean over answer positions of L(A_k) - T0, A_k = Ah on the top k of the ranking, A0 elsewhere; nums[N] is
             E98's r_half numerator, so nums[N] / G is the neuron share (printed beside, Fable's option (c))
The carrying set is the top K neurons with a positive score (`top_set`); a state that moves no firing has none.
Firing is signed; the corpus tops and the engaged values read it as signed, so a strongly negative firing is low.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from commute import carrying_jobs as CJ
from commute import neuron_r as NR
from commute import qwen_comparators as qc
from commute.qwen_sft import set_weights
from commute.qwen_vpd import N_LAYERS, C

WIDTH = C["down"]  # 6144 inputs of each down_proj
N = N_LAYERS * WIDTH
Acts = dict[int, Tensor]


def mlps(model: Any) -> list[Any]:
    return [layer.mlp for layer in model.model.layers]


def flat(acts: Acts) -> Tensor:
    return torch.cat([acts[l] for l in range(len(acts))], dim=-1)


@torch.no_grad()
def run(model: Any, toks: Tensor, W: Mapping[str, Tensor] | None = None, x_in: Acts | None = None,
        a_in: Acts | None = None, logits: bool = True) -> tuple[Tensor | None, Acts, Acts]:
    """One forward at W (None: as the model stands), hooked as `neuron_r.hooked`; logits False skips the head."""
    call = (lambda: model(toks, use_cache=False).logits) if logits else (lambda: model.model(toks, use_cache=False))
    with set_weights(model, dict(W) if W is not None else None):
        out, xs, acts = NR.hooked(mlps(model), call, x_in, a_in)
    return (out.float() if logits else None), xs, acts


@torch.no_grad()
def w0_firing(model: Any, xs: Acts) -> Acts:
    """The model's own gate, up and SiLU on given MLP inputs (call it with the model at W0): Ah from the state's x."""
    return {l: m.act_fn(m.gate_proj(xs[l])) * m.up_proj(xs[l]) for l, m in enumerate(mlps(model))}


def logp(logits: Tensor, toks: Tensor, mask: Tensor) -> Tensor:
    return qc.token_reads(logits, toks, mask)[1]


# ----------------------------------------------------------------------------- one state


def batches(R: qc.Rows, size: int, dev: Any) -> Any:
    for s in range(0, len(R.tokens), size):
        yield s, R.tokens[s : s + size].to(dev), R.answer[s : s + size].to(dev)


def rank_batch(model: Any, toks: Tensor, mask: Tensor, W: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """T0, T1 at the answer positions and the per-row score (B, N) fp64 of one batch."""
    assert not model.training, "gradient checkpointing would re-run the leaf hooks"
    seen: dict[str, Tensor] = {}

    def loss() -> Tensor:
        lp = logp(model(toks, use_cache=False).logits, toks, mask)
        seen["T0"] = lp.detach()
        return lp.sum()

    g, a0 = NR.firing_grads(mlps(model), loss)
    lg, x1, _ = run(model, toks, W)
    assert lg is not None
    ah = w0_firing(model, x1)
    rows = torch.cat([(g[l] * (ah[l] - a0[l])).sum(1).double() for l in range(N_LAYERS)], dim=-1)
    return seen["T0"], logp(lg, toks, mask), rows.cpu()


def rank_state(model: Any, R: qc.Rows, W: Mapping[str, Tensor], batch: int) -> dict[str, Tensor]:
    """T0, T1 and row_of per answer position; score (2, N), rows of even and odd index summed apart."""
    dev, T0, T1, score = next(model.parameters()).device, [], [], torch.zeros(2, N, dtype=torch.float64)
    for s, toks, mask in batches(R, batch, dev):
        t0, t1, rows = rank_batch(model, toks, mask, W)
        T0.append(t0.cpu())
        T1.append(t1.cpu())
        for j in range(len(rows)):
            score[(s + j) % 2] += rows[j]
    return {"T0": torch.cat(T0), "T1": torch.cat(T1), "row_of": R.answer.nonzero()[:, 0], "score": score}


@torch.no_grad()
def grid(model: Any, R: qc.Rows, W: Mapping[str, Tensor], order: Tensor, ks: Sequence[int], batch: int) -> dict[int, float]:
    """nums[k] per k of `ks` (module docstring); k = N patches every neuron: E98's r_half numerator."""
    dev = next(model.parameters()).device
    masks = {k: {l: m.to(dev) for l, m in NR.top_masks(order, k, N_LAYERS, WIDTH).items()} for k in ks}
    sums, n = dict.fromkeys(ks, 0.0), 0
    for _, toks, mask in batches(R, batch, dev):
        lg, _, a0 = run(model, toks)
        assert lg is not None
        T0 = logp(lg, toks, mask)
        ah = w0_firing(model, run(model, toks, W, logits=False)[1])
        for k in ks:
            a_in = {l: torch.where(masks[k][l], ah[l], a0[l]) for l in range(N_LAYERS)}
            lk = run(model, toks, a_in=a_in)[0]
            assert lk is not None
            sums[k] += float((logp(lk, toks, mask) - T0).double().sum())
        n += int(mask.sum())
    return {k: v / n for k, v in sums.items()}


# ----------------------------------------------------------------------------- W0 on text


@torch.no_grad()
def corpus_tops(model: Any, rows: Tensor, batch: int, n_top: int) -> dict[str, Tensor]:
    """Per neuron, its n_top rows of `rows` by the row's peak firing, with the peak's position and value: (n_top, N)."""
    dev = next(model.parameters()).device
    v = torch.full((0, N), -math.inf, device=dev)
    r = torch.zeros((0, N), dtype=torch.int64, device=dev)
    p = torch.zeros((0, N), dtype=torch.int64, device=dev)
    for s in range(0, len(rows), batch):
        acts = run(model, rows[s : s + batch].to(dev), logits=False)[2]
        peak = [acts[l].max(dim=1) for l in range(N_LAYERS)]
        bv, bp = torch.cat([x.values for x in peak], -1), torch.cat([x.indices for x in peak], -1)
        br = torch.arange(s, s + len(bv), device=dev)[:, None].expand_as(bp)
        v, idx = torch.cat([v, bv]).topk(min(n_top, len(v) + len(bv)), dim=0)
        r, p = torch.cat([r, br]).gather(0, idx), torch.cat([p, bp]).gather(0, idx)
    return {"values": v.cpu(), "rows": r.int().cpu(), "pos": p.int().cpu()}


@torch.no_grad()
def sorted_sample(model: Any, rows: Tensor) -> Tensor:
    """(N, positions): every neuron's firing on every position of `rows`, sorted, on the card."""
    acts = run(model, rows.to(next(model.parameters()).device), logits=False)[2]
    return torch.cat([acts[l].reshape(-1, WIDTH).T.sort(dim=1).values for l in range(N_LAYERS)])


@torch.no_grad()
def engaged(model: Any, R: qc.Rows, sample: Tensor, batch: int, chunk: int = 256) -> Tensor:
    """(N,) fp64: per neuron, W0's firing at R's answer positions as a quantile of its own corpus sample, averaged."""
    dev, tot, n = sample.device, torch.zeros(N, dtype=torch.float64, device=sample.device), 0
    for _, toks, mask in batches(R, batch, dev):
        A = flat(run(model, toks, logits=False)[2])[mask]
        for c in A.split(chunk):
            tot += CJ.quantiles(sample, c.T).sum(dim=1)
        n += len(A)
    return (tot / n).cpu()


@torch.no_grad()
def answer_means(model: Any, seqs: Sequence[tuple[list[int], int]], batch: int, pad: int) -> Tensor:
    """(2, N) fp64: mean firing over the positions that predict an answer token (p - 1 to the second-to-last), items of
    even and odd index apart. Rows are padded on the right: causal attention leaves the read positions as unpadded."""
    dev = next(model.parameters()).device
    tot, cnt = torch.zeros(2, N, dtype=torch.float64, device=dev), [0, 0]
    for s in range(0, len(seqs), batch):
        part = seqs[s : s + batch]
        L = max(len(ids) for ids, _ in part)
        acts = run(model, torch.tensor([ids + [pad] * (L - len(ids)) for ids, _ in part], device=dev), logits=False)[2]
        for j, (ids, p) in enumerate(part):
            for l in range(N_LAYERS):
                tot[(s + j) % 2, l * WIDTH : (l + 1) * WIDTH] += acts[l][j, p - 1 : len(ids) - 1].double().sum(0)
            cnt[(s + j) % 2] += len(ids) - p
    return (tot / torch.tensor(cnt, dtype=torch.float64, device=dev)[:, None]).cpu()


# ----------------------------------------------------------------------------- sets and rules (CPU)


def top_set(score: Tensor, k: int) -> list[int]:
    """The k neurons of highest score among those with a score above 0 (ties by index); fewer if fewer are positive."""
    order = torch.sort(score, descending=True, stable=True).indices[:k]
    return [int(i) for i in order if score[i] > 0]


def overlap(a: Sequence[int], b: Sequence[int]) -> float:
    """The share of a inside b; nan for an empty a or an empty b."""
    return len(set(a) & set(b)) / len(a) if a and b else math.nan


def random_overlaps(a: Sequence[int], b: Sequence[int], n: int, gen: torch.Generator) -> list[float]:
    """Overlaps with b of n random sets matched to a's neurons per block (none excluded: chance at a's block profile)."""
    return [overlap(CJ.block_matched(a, set(), gen, WIDTH), b) for _ in range(n)] if a and b else []


def random_engaged(a: Sequence[int], eng: Sequence[float], n: int, gen: torch.Generator) -> list[float]:
    """Engaged shares of n random sets matched to a's neurons per block, outside a (E101's draws)."""
    return [CJ.set_share(eng, CJ.block_matched(a, set(a), gen, WIDTH)) for _ in range(n)] if a else []


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def above(seeds: Sequence[float | None], gauss: float | None, draws: Sequence[float]) -> dict[str, Any]:
    """P1-P3's rule. floor = the larger of the Gaussian change's value and every random draw; band = the larger of the
    two seeds' spread and twice the draws' sd. 'above' iff both seeds are over the floor and their mean is over it by
    more than the band (strict); 'not above' otherwise; 'unread' with a seed or the Gaussian missing or not finite, or
    under two finite draws."""
    ok = [d for d in draws if finite(d)]
    if len(seeds) != 2 or not all(finite(s) for s in seeds) or not finite(gauss) or len(ok) < 2:
        return {"verdict": "unread", "seeds": list(seeds), "gauss": gauss, "n_draws": len(ok)}
    s0, s1, gs = (float(x) for x in (seeds[0], seeds[1], gauss))  # type: ignore[arg-type]
    floor, band = max(gs, max(ok)), max(abs(s0 - s1), 2 * statistics.stdev(ok))
    gap = (s0 + s1) / 2 - floor
    verdict = "above" if min(s0, s1) > floor and gap > band else "not above"
    return {"verdict": verdict, "seeds": [s0, s1], "gauss": gauss, "floor": floor, "gap": gap, "band": band,
            "draw_max": max(ok), "draw_sd": statistics.stdev(ok)}


def control_i(halves: float, half_draws: Sequence[float], gauss_overlap: float | None, min_gap: float) -> dict[str, Any]:
    """Positive control (i). C1a: the prompted contrast set of the even items overlaps the odd items' by at least
    min_gap over every random draw. C1b: the Gaussian change's carrying set overlaps the contrast set by at least min_gap
    less than C1a's overlap. A missing or non-finite number fails its half (a control that cannot be read has not passed)."""
    ok = [d for d in half_draws if finite(d)]
    a = finite(halves) and len(ok) >= 2 and halves - max(ok) >= min_gap
    b = finite(halves) and finite(gauss_overlap) and halves - float(gauss_overlap) >= min_gap  # type: ignore[arg-type]
    return {"C1a": bool(a), "C1b": bool(b), "passed": bool(a and b), "halves": halves,
            "draw_max": max(ok) if ok else None, "gauss_overlap": gauss_overlap, "min_gap": min_gap}


def k_half(nums: Mapping[int, float], G: Mapping[str, Any], carry_floor: float) -> int | None:
    """E98's k_half on these nums: None unless the gain is defined (`qwen_comparators.share`: over twice its row se)."""
    full = nums[N]
    share = full / G["gain"] if G["share"] is not None else math.nan
    return NR.k_half(dict(nums), full, share, carry_floor)


# ----------------------------------------------------------------------------- one state (E102's read, shared for E118)


def state_cells(model: Any, R: qc.Rows, W: Mapping[str, Tensor], k: int, ks: Sequence[int], batch: int,
                carry_floor: float) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """One state W on R: E102's cells (`qwen_carrying/run.py` read_state without its checks, files and logs) and the
    per-position tensors it saves (T0, T1, row_of, score)."""
    r = rank_state(model, R, W, batch)
    d = r["T1"] - r["T0"]
    G = qc.share(d, d, r["row_of"])
    total = r["score"].sum(0)
    nums = grid(model, R, W, torch.sort(total, descending=True, stable=True).indices, ks, batch)
    full = nums[N]
    halves = [top_set(r["score"][h], k) for h in (0, 1)]
    out = {"rows": R.name, "gain": G, "nums": {str(x): v for x, v in nums.items()}, "k_half": k_half(nums, G, carry_floor),
           "neuron_share": full / G["gain"] if G["share"] is not None else None,
           "top_K_of_numerator": nums[k] / full if full != 0 else None,
           "first_order_over_numerator": float(total.sum()) / len(d) / full if full != 0 else None,
           "carry": top_set(total, k), "halves_overlap": overlap(*halves)}
    return out, r


def set_reads(carry: Sequence[int], eng: Sequence[float], contrast: Sequence[int], n_draws: int, gen: torch.Generator) -> dict[str, Any]:
    """E102's per-state scores of a carrying set (`qwen_carrying/score.py` score): its prompt overlap with the contrast
    set and its engaged share, each beside n_draws layer-matched random sets drawn from `gen` in E102's order."""
    return {"n": len(carry), "prompt_overlap": overlap(carry, contrast), "prompt_overlap_draws": random_overlaps(carry, contrast, n_draws, gen),
            "engaged": CJ.set_share(eng, carry), "engaged_draws": random_engaged(carry, eng, n_draws, gen)}

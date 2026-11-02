"""Def 1 probes for E55 (plan reports/plans/2026-10-29_plan_reuse-stage1.md, "Possession"; predictions.json
definitions.probe, margin, possession; edge cases E3, E4).

Recipe, Jain et al. 2024 L1238-1248: a linear layer with the vocabulary's output size on each block's residual output,
trained on data sampled from the PCFG process, 4K AdamW iterations, lr 1e-3 decayed x0.1 at 2K, 3K and 3.5K. Read at
`=` (pcfg.ART_POS), not at Ans (plan, review S1). Not stated in the paper, filled here: batch 96 (the paper's only
stated batch, FT L1189), AdamW's library defaults otherwise (betas 0.9/0.999, weight decay 0.01), nn.Linear's init.

One call trains, for each model given and every block, a probe on the true labels and a shuffled-label probe (labels
permuted within each batch: inputs carry no information about them), all on the same fresh C_Te = 0 batches of one
prefix. Accuracy (both) is against the true labels of 2,000 fresh held-out samples, so the shuffled probe scores what
the label marginal alone gives.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from commute import pcfg
from commute.interchange import binomial_sd, forward_at_art, run_at_art
from commute.mingpt import GPT
from commute.pcfg_train import Batches

ITERS, LR, DECAY_AT, DECAY = 4000, 1e-3, (2000, 3000, 3500), 0.1  # L1243-1245
BATCH, HELD_OUT = 96, 2000
PROBE_SEED0, HELD_OUT_KEY = 200, 40  # data stream default_rng([PROBE_SEED0 + seed, chunk]); held out [eval_seed, 40, ..]


class Probes(nn.Module):
    """k independent linear probes d -> vocab, probe j reading input j."""

    def __init__(self, k: int, d: int, vocab: int) -> None:
        super().__init__()
        lin = [nn.Linear(d, vocab) for _ in range(k)]  # nn.Linear's default init, per probe
        self.w = nn.Parameter(torch.stack([m.weight.detach().T for m in lin]))
        self.b = nn.Parameter(torch.stack([m.bias.detach() for m in lin]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (k, n, d) -> (k, n, vocab)
        return torch.einsum("knd,kdv->knv", x, self.w) + self.b[:, None]


def _loss(probes: Probes, resid: torch.Tensor, y: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Sum over probes of each probe's mean cross-entropy; the first n_layer get y, the rest y[perm]."""
    n_layer = resid.shape[0]
    logits = probes(torch.cat([resid, resid]))
    target = torch.cat([y.expand(n_layer, -1), y[perm].expand(n_layer, -1)])
    return F.cross_entropy(logits.flatten(0, 1), target.flatten()) * logits.shape[0]


def train_probes(models: Mapping[str, GPT], op: int, seed: int, eval_seed: int) -> dict[str, dict[str, object]]:
    """Per model: held-out accuracy at every block of the true-label probes ('acc') and of the shuffled-label probes
    ('shuffled'), and the held-out majority-class rate ('majority'); prefix 'C <op> 40', label count(op)."""
    first = next(iter(models.values()))
    device, n_layer, d = next(first.parameters()).device, first.cfg.n_layer, first.cfg.d
    for model in models.values():
        model.eval()
    torch.manual_seed(PROBE_SEED0 + seed)
    probes = {name: Probes(2 * n_layer, d, pcfg.VOCAB).to(device) for name in models}
    opt = torch.optim.AdamW([p for pr in probes.values() for p in pr.parameters()], lr=LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(DECAY_AT), gamma=DECAY)
    batches = Batches(lambda rng, n: pcfg.task_batch(rng, n, "C", op)[0], PROBE_SEED0 + seed, BATCH, 100, device)
    for step in range(ITERS):
        seq = batches.get(step)
        y, perm = seq[:, pcfg.ANS], torch.randperm(BATCH, device=device)
        loss = torch.zeros((), device=device)
        for name, model in models.items():
            with torch.no_grad():
                _, resid = forward_at_art(model, seq[:, : pcfg.ANS], 0, None)
            loss = loss + _loss(probes[name], torch.stack(resid), y, perm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    batches.close()
    held, _ = pcfg.task_batch(np.random.default_rng([eval_seed, HELD_OUT_KEY, seed, op]), HELD_OUT, "C", op)
    y = torch.from_numpy(held[:, pcfg.ANS].astype(np.int64)).to(device)
    out: dict[str, dict[str, object]] = {}
    for name, model in models.items():
        with torch.no_grad():
            correct = (probes[name](torch.cat([run_at_art(model, held)[1]] * 2)).argmax(-1) == y).float().mean(1)
        out[name] = {"acc": correct[:n_layer].tolist(), "shuffled": correct[n_layer:].tolist(),
                     "majority": float(np.bincount(held[:, pcfg.ANS]).max() / HELD_OUT)}  # fmt: skip
    return out


# ---- filed rules (pure arithmetic; tests/test_interchange.py) ----


def margin_summary(acc: np.ndarray, controls: list[np.ndarray]) -> dict[str, list[float]]:
    """acc and each control: (seeds, n_layer). m = acc - max(controls) per seed; per block the mean m and its s = sd
    over seeds (ddof 1), floored at the binomial sd of the mean probe accuracy at HELD_OUT (E3)."""
    m = acc - np.max(np.stack(controls), axis=0)
    s = [max(float(np.std(m[:, j], ddof=1)), binomial_sd(float(acc[:, j].mean()), HELD_OUT)) for j in range(m.shape[1])]
    return {"m": m.mean(0).tolist(), "s": s, "acc": acc.mean(0).tolist()}


def m_plus(low_count_a: dict[str, list[float]]) -> dict[str, float | int | bool]:
    """LOW's count-a margin at its best block, its s, and void iff m_plus <= 0 or m_plus < 6 s_plus (E4)."""
    j = int(np.argmax(low_count_a["m"]))
    m, s = low_count_a["m"][j], low_count_a["s"][j]
    return {"m_plus": m, "s_plus": s, "block": j + 1, "void": not (m > 0 and m >= 6 * s)}


def possessed(summary: dict[str, list[float]], plus: dict[str, float | int | bool]) -> bool | None:
    """Possessed iff at some block m >= 3 s and m >= 0.5 m_plus (inclusive); None when m_plus is void."""
    if plus["void"]:
        return None
    return any(m >= 3 * s and m >= 0.5 * float(plus["m_plus"]) for m, s in zip(summary["m"], summary["s"], strict=True))

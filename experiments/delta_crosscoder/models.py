"""The two models of a Delta-Crosscoder side and every forward the run makes through them (plan
reports/plans/2026-11-23_plan_p2-72-delta-crosscoder-spec.md 3.1, 3.6, 3.7): site activations (position 0 dropped, as
E203/E235's streams drop it), logits under a steering vector, and the nucleus sampler of spec 3.6 (the model's forward on
the growing prefix, no key-value cache; the LM head on the last position only).

M is a steering.Net at W0. F is a `Changed`: a net (M's, or a grafted copy), an update dW on it (None: none), and
E212's planted vector, added at site A at positions every - 1, 2 every - 1, ... of task text only (`planted` reads).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

PLANT_SITE = "in2"  # E212's layer 2 input (experiments/realistic_tasks/results/predictions_run3.json settings.layer)


@dataclass
class Changed:
    net: Any  # steering.Net
    dW: dict[str, Tensor] | None
    plant: tuple[Tensor, int] | None  # (vector, every), task text only
    gen: str  # whose continuations F's half of the contrastive data uses: "M", or this side's generation key


@dataclass
class Read:
    """A side's score set: windows `toks` (N, T) fed whole; the behaviour g is the answer log-odds at `half` for target
    `tgt` (kind "task") or a KL at the positions `pos` (kind "plant"); `pos` (N, T) also marks the behaviour positions of
    the top-activation test. `planted`: F's plant goes on these windows."""

    toks: Tensor
    kind: str
    half: int
    tgt: Tensor | None
    pos: Tensor
    planted: bool


class _Grab(Exception):
    """Raised by a site hook once the activation is taken, so the rest of the forward is skipped."""


def _edits(F: Changed | None, planted: bool, site: str | None, steer: Tensor | None,
           grab: Callable[[Tensor], None] | None) -> dict[str, Callable[[Tensor], Tensor]]:  # fmt: skip
    """Hooks for one forward: F's plant at PLANT_SITE (task text), then `steer` added at `site` at every position, then
    `grab` (which raises). Two at one site compose in that order."""
    fns: dict[str, list[Callable[[Tensor], Tensor]]] = {}
    if F is not None and F.plant is not None and planted:
        v, every = F.plant

        def plant(x: Tensor) -> Tensor:
            x = x.clone()
            x[:, every - 1 :: every] += v.to(x.device, x.dtype)
            return x

        fns.setdefault(PLANT_SITE, []).append(plant)
    if steer is not None:
        assert site is not None
        fns.setdefault(site, []).append(lambda x: x + steer.to(x.device, x.dtype))
    if grab is not None:
        assert site is not None

        def take(x: Tensor) -> Tensor:
            grab(x)
            raise _Grab

        fns.setdefault(site, []).append(take)

    def chain(fs: list[Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
        def f(x: Tensor) -> Tensor:
            for g in fs:
                x = g(x)
            return x

        return f

    return {s: chain(fs) for s, fs in fns.items()}


def _who(M: Any, F: Changed | None) -> tuple[Any, dict[str, Tensor] | None]:
    return (M, None) if F is None else (F.net, F.dW)


@torch.no_grad()
def acts(M: Any, F: Changed | None, toks: Tensor, site: str, planted: bool = False, n: int | None = None) -> Tensor:
    """(N * (T - 1), d) activations at `site` of M (F None) or F, every position but 0, `n` windows a forward (default
    about 32k tokens, E235's `site_acts`)."""
    net, dW = _who(M, F)
    out: list[Tensor] = []
    n = n or max(1, 32768 // toks.shape[1])
    edit = _edits(F, planted, site, None, lambda x: out.append(x[:, 1:].float()))
    with net.at(dW):
        for b in toks.split(n):
            try:
                net.run(b, edit=edit)
            except _Grab:
                pass
    return torch.cat(out).reshape(-1, out[0].shape[-1])


def pairs(M: Any, F: Changed, toks: Tensor, site: str, planted: bool) -> Tensor:
    """(N * (T - 1), 2, d): M's activations, then F's on the same tokens. F's are M's own (one forward) where F computes
    what M does: the controls, and a plant's side off its task text."""
    a = acts(M, None, toks, site)
    return torch.stack([a, a if is_m(M, F, planted) else acts(M, F, toks, site, planted)], 1)


def is_m(M: Any, F: Changed, planted: bool) -> bool:
    """F computes what M does on these windows: M's net, no update, no plant on them (the controls; a plant's side off
    its task text)."""
    return F.dW is None and F.net is M and not (planted and F.plant is not None)


@torch.no_grad()
def logits(M: Any, F: Changed | None, toks: Tensor, planted: bool, site: str | None, steer: Tensor | None) -> Tensor:
    """(N, T, V) float logits of M (F None) or F on `toks`, with `steer` (d,) added at `site` at every position."""
    net, dW = _who(M, F)
    with net.at(dW):
        return net.run(toks, edit=_edits(F, planted, site, steer, None))[0]


@torch.no_grad()
def generate(M: Any, F: Changed | None, prompts: Tensor, new: int, temperature: float, top_p: float,
             seed: int) -> Tensor:  # fmt: skip
    """`prompts` (n, P) continued by `new` tokens sampled at `temperature` with nucleus `top_p` (spec 3.6; Appendix C's
    settings): the forward on the whole prefix each step, logits of the last position only (through the last block's
    output, then ln_f and the LM head). Text only, so no plant."""
    net, dW = _who(M, F)
    idx = prompts.to(net.device)
    g = torch.Generator(device=idx.device).manual_seed(seed)
    last: list[Tensor] = []
    edit = _edits(None, False, "out3", None, lambda x: last.append(x[:, -1]))
    with net.at(dW):
        for _ in range(new):
            last.clear()
            try:
                net.run(idx, edit=edit)
            except _Grab:
                pass
            lg = net.net.lm_head(net.net.ln_f(last[0])).float() / temperature
            p = torch.softmax(lg, -1)
            ps, order = p.sort(-1, descending=True)
            keep = ps.cumsum(-1) - ps < top_p  # the smallest set whose mass reaches top_p; the top token always kept
            ps = torch.where(keep, ps, torch.zeros_like(ps))
            pick = torch.multinomial(ps / ps.sum(-1, keepdim=True), 1, generator=g)
            idx = torch.cat([idx, order.gather(1, pick)], 1)
    return idx.cpu()


def ce(lg: Tensor, toks: Tensor) -> tuple[float, int]:
    """Summed next-token loss of logits (N, T, V) on `toks` and the number of tokens."""
    t = toks.to(lg.device)
    loss = torch.nn.functional.cross_entropy(lg[:, :-1].flatten(0, 1), t[:, 1:].flatten(), reduction="sum")
    return float(loss), t[:, 1:].numel()

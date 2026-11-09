"""Editing and reading the target model's hidden activations through `steering.Net`, shared by E137 and anything that
fits a map between activations of two runs (added 2026-11-09).

`run` wraps `Net.run`: it captures only the sites asked for (`Net.run(capture=True)` also captures the four MLP sites,
~1 GB per batch here) and can keep gradients, which `Net.run` drops (`grad=True` calls the undecorated function).
Edits compose per site in order. `Ridge` is a standardised multi-target ridge streamed in batches: fp32 batch products
summed in fp64, so 36,864 inputs fit on a 48 GB card (the X'X held is p^2 x 8 bytes, independent of the number of windows).
`run` also reads and edits `final`, the unembedding's input after `ln_f`, which `Net.run` has no hook for.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

from commute import steering as ST

Fn = Callable[[Tensor], Tensor]
Edit = dict[str, Fn]
STREAMS = ("in0", "in1", "in2", "in3", "out3")
FINAL = "final"  # the unembedding's input, after `ln_f`: logits are linear in it, so KL to a fixed target is convex in an affine map onto it
_ALL = (*STREAMS, FINAL)


def tv(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """Total variation between two next-token distributions given as logits, per row: (B,) float64."""
    p, q = torch.softmax(p_logits.double(), -1), torch.softmax(q_logits.double(), -1)
    return 0.5 * (p - q).abs().sum(-1)


def kl(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """KL(p || q) per row, in nats, from logits; fp32, differentiable in q. p is detached by the caller's no_grad."""
    lp, lq = torch.log_softmax(p_logits, -1), torch.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def up_to(site: str) -> tuple[str, ...]:
    """The streams at or before `site`, in order: z_M's columns for that site."""
    return _ALL[: _ALL.index(site) + 1]


def chunks(x: Tensor, size: int) -> Iterable[Tensor]:
    return (x[s : s + size] for s in range(0, x.shape[0], size))


def merged(*edits: Edit) -> Edit:
    """One edit from several: where two touch the same site they run in the order given."""
    out: Edit = {}
    for e in edits:
        for site, fn in e.items():
            prev = out.get(site)
            out[site] = fn if prev is None else _then(prev, fn)
    return out


def _then(a: Fn, b: Fn) -> Fn:
    return lambda x: b(a(x))


def run(net: ST.Net, toks: Tensor, sites: Iterable[str] = (), edit: Edit | None = None,
        grad: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
    """Logits (B, T, V) and each of `sites` as seen after `edit`. `grad=True` keeps the graph through the edits."""
    seen: dict[str, Tensor] = {}

    def keep(site: str) -> Fn:
        def f(x: Tensor) -> Tensor:
            seen[site] = x
            return x

        return f

    e = merged(edit or {}, {s: keep(s) for s in sites})
    last_fn = e.pop(FINAL, None)  # `Net.run` has no hook here and would ignore the key silently
    handle = None if last_fn is None else net.net.lm_head.register_forward_pre_hook(lambda _m, a: (last_fn(a[0]),))
    try:
        logits = inspect.unwrap(ST.Net.run)(net, toks, e)[0] if grad else net.run(toks, e)[0]
    finally:
        if handle is not None:
            handle.remove()
    return logits, seen


def flat(seen: dict[str, Tensor], site: str) -> Tensor:
    """z_M for `site`: every stream up to it, every position, as (B, p) fp32."""
    return torch.cat([seen[s].flatten(1) for s in up_to(site)], 1)


def write_last(Q: Tensor | None, value: Tensor) -> Fn:
    """Replace a read-out's value at the last position. `Q` (d, k) orthonormal: the subspace's value becomes `value`
    (B, k), the rest of the activation kept. `Q` None: the whole activation becomes `value` (B, d)."""

    def f(x: Tensor) -> Tensor:
        x = x.clone()
        h = x[:, -1].clone()  # a separate tensor: writing x[:, -1] must not change a value the backward pass needs
        x[:, -1] = value if Q is None else h + (value - h @ Q) @ Q.T
        return x

    return f


def stash_last(store: dict[str, Tensor], name: str) -> Fn:
    """An identity edit that keeps the last position's activation in `store[name]`."""

    def f(x: Tensor) -> Tensor:
        store[name] = x[:, -1]
        return x

    return f


def add_last(delta: Tensor | Callable[[], Tensor]) -> Fn:
    """Add to the last position. `delta` (B, d) or (d,), or a function of nothing that gives it at hook time."""

    def f(x: Tensor) -> Tensor:
        d = delta() if callable(delta) else delta
        x = x.clone()
        x[:, -1] = x[:, -1] + d
        return x

    return f


class Ridge:
    """Ridge from X (n, p) to named targets (n, q) on standardised inputs, in batches. Call `stats` on every batch, then
    `freeze`, then `add` on every batch, then `solve` per target and penalty. `solve(key, p, lam)` uses the first p
    input columns, so nested input sets (z_M at successive sites) share one accumulation. lam is per sample: the system is
    X'X + n lam I. Targets are centred in `stats` and restored in `predict`."""

    def __init__(self, device: str) -> None:
        self.device, self.n = device, 0
        self.s: Tensor | None = None
        self.s2: Tensor | None = None
        self.mean, self.std = torch.empty(0), torch.empty(0)
        self.y_sum: dict[str, Tensor] = {}
        self.y_mean: dict[str, Tensor] = {}
        self.xx: Tensor | None = None
        self.xy: dict[str, Tensor] = {}

    def stats(self, X: Tensor, Y: dict[str, Tensor]) -> None:
        X = X.to(self.device).double()
        self.s = X.sum(0) if self.s is None else self.s + X.sum(0)
        self.s2 = (X * X).sum(0) if self.s2 is None else self.s2 + (X * X).sum(0)
        self.n += X.shape[0]
        for k, y in Y.items():
            d = y.to(self.device).double().sum(0)
            self.y_sum[k] = self.y_sum[k] + d if k in self.y_sum else d

    def freeze(self) -> None:
        assert self.s is not None and self.s2 is not None and self.n > 0
        mean = self.s / self.n
        var = (self.s2 / self.n - mean * mean).clamp_min(0.0)
        std = var.sqrt()
        std[std <= 1e-6 * std.max()] = 1.0  # a constant column's variance is rounding, and dividing by it would blow it up
        self.mean, self.std = mean.float(), std.float()
        self.y_mean = {k: (v / self.n).float() for k, v in self.y_sum.items()}
        p = self.mean.numel()
        self.xx = torch.zeros(p, p, device=self.device, dtype=torch.float64)

    def standardise(self, X: Tensor) -> Tensor:
        p = X.shape[1]
        return (X.to(self.device).float() - self.mean[:p]) / self.std[:p]

    def add(self, X: Tensor, Y: dict[str, Tensor]) -> None:
        assert self.xx is not None, "call freeze first"
        Z = self.standardise(X)
        self.xx.add_(Z.T @ Z)  # in place, not via .double(): a second p x p float64 copy is 11 GB at p = 36,864
        for k, y in Y.items():
            b = Z.T @ (y.to(self.device).float() - self.y_mean[k])
            self.xy[k] = self.xy[k] + b.double() if k in self.xy else b.double()

    def solve(self, key: str, p: int, lam: float) -> Tensor:
        """Coefficients (p, q) fp32 for standardised inputs. Raises if the factorisation fails."""
        assert self.xx is not None
        A = self.xx[:p, :p].clone()
        A.diagonal().add_(self.n * lam)
        L, info = torch.linalg.cholesky_ex(A)
        if int(info):
            raise RuntimeError(f"ridge factorisation failed for {key} at p={p}, lam={lam} (info {int(info)})")
        del A  # at 36,864 columns a further 10 GiB on the card matters
        return torch.cholesky_solve(self.xy[key][:p], L).float()

    def predict(self, X: Tensor, beta: Tensor, key: str) -> Tensor:
        return self.y_mean[key] + self.standardise(X) @ beta

    def memory(self) -> dict[str, Any]:
        assert self.xx is not None
        return {"p": self.xx.shape[0], "xx_gb": self.xx.shape[0] ** 2 * 8 / 2**30, "n_windows": self.n}

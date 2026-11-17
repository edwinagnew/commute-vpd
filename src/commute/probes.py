"""Linear and small MLP probes of a model's last-position state for a 0/1 target (E159, cell P1.15).

The input is the residual stream at one site, last position only (d values), standardised with the fit set's mean and sd.
Not every position: there the raw number tokens sit at their own positions, so a small MLP would compute any parity of them
itself and read 1 whatever the model holds. At the last position the model must have brought the numbers together.

Both probes are trained the same way, scikit-learn's `MLPClassifier` defaults (1.x): one hidden layer of 100 ReLU units
(none for the linear probe, which is then logistic regression), Adam at 1e-3 (betas 0.9, 0.999, eps 1e-8), L2 penalty
alpha 1e-4 added as 0.5 alpha |W|^2 / batch size, batches of 200, at most 200 epochs, stopped when the epoch's mean
training loss has not fallen by tol 1e-4 below its best for 10 epochs in a row. Departure: torch's default initialisation,
not scikit-learn's Glorot. Accuracy thresholds the logit at 0. The shuffled null trains on the fit labels permuted and is
scored against the real held-out labels: it reads the probe's chance level on these windows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute.hidden_maps import chunks, run
from commute.steering import Net


@dataclass(frozen=True)
class ProbeSizes:
    hidden: int = 100  # MLPClassifier hidden_layer_sizes=(100,)
    lr: float = 1e-3  # learning_rate_init
    alpha: float = 1e-4  # L2 penalty
    batch: int = 200  # batch_size "auto" = min(200, n)
    max_epochs: int = 200  # max_iter
    tol: float = 1e-4
    n_iter_no_change: int = 10
    n_fit: int = 20_000  # keyed fit windows per probe (the first n_fit): 100 steps an epoch, at most 20,000 steps

    def __post_init__(self) -> None:
        assert self.n_fit % self.batch == 0 and self.hidden > 0


def last_states(net: Net, windows: Tensor, sites: Sequence[str], batch: int) -> dict[str, Tensor]:
    """{site: (N, d) float32} the residual stream at each site, last position, on the model's device."""
    parts: dict[str, list[Tensor]] = {s: [] for s in sites}
    for t in chunks(windows, batch):
        seen = run(net, t, sites=tuple(sites))[1]
        for s in sites:
            parts[s].append(seen[s][:, -1].float())
    return {s: torch.cat(v) for s, v in parts.items()}


def fit_probe(X: Tensor, y: Tensor, hidden: int, sz: ProbeSizes, seed: int) -> tuple[Callable[[Tensor], Tensor], dict[str, Any]]:
    """A probe (logits from raw states) trained on (X, y) as the module docstring says; `hidden` 0 is the linear probe."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    mean, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Z = (X - mean) / sd
    d = X.shape[1]
    layers = [nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1)] if hidden else [nn.Linear(d, 1)]
    model = nn.Sequential(*layers).to(X.device)
    weights = [m.weight for m in model if isinstance(m, nn.Linear)]
    opt = torch.optim.Adam(model.parameters(), lr=sz.lr, betas=(0.9, 0.999), eps=1e-8)
    best, stale, losses = float("inf"), 0, []
    for _ in range(sz.max_epochs):
        perm = torch.randperm(Z.shape[0], generator=gen).to(X.device)
        total = torch.zeros((), device=X.device)
        for i in range(0, Z.shape[0], sz.batch):
            idx = perm[i : i + sz.batch]
            out = model(Z[idx]).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(out, y[idx])
            loss = loss + 0.5 * sz.alpha * torch.stack([w.pow(2).sum() for w in weights]).sum() / idx.numel()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * idx.numel()
        losses.append(float(total) / Z.shape[0])
        stale = stale + 1 if losses[-1] > best - sz.tol else 0
        best = min(best, losses[-1])
        if stale >= sz.n_iter_no_change:
            break
    model.requires_grad_(False)
    return lambda X_: model((X_ - mean) / sd).squeeze(1), {"epochs": len(losses), "final_loss": losses[-1]}


def accuracy(probe: Callable[[Tensor], Tensor], X: Tensor, y: Tensor) -> float:
    right = sum(float(((probe(Xb) > 0).float() == yb).sum()) for Xb, yb in zip(chunks(X, 8192), chunks(y, 8192), strict=True))
    return right / X.shape[0]


def probe_target(Xf: dict[str, Tensor], yf: Tensor, Xr: dict[str, Tensor], yr: Tensor, sz: ProbeSizes, seed: int,
                 log: Callable[[str], None]) -> dict[str, Any]:
    """Per site and probe ("linear", "mlp"): in-sample and held-out accuracy of the probe trained on the real labels, and
    held-out accuracy of the one trained on shuffled labels; then each probe's best held-out accuracy over the sites.
    Xf, yf: the fit states and labels (the first sz.n_fit are used); Xr, yr: the held-out ones, all used."""
    n = sz.n_fit
    assert yf.shape[0] >= n, f"{yf.shape[0]} keyed fit windows, fewer than n_fit {n}"
    y_fit = yf[:n].to(next(iter(Xf.values())).device)
    y_shuf = y_fit[torch.randperm(n, generator=torch.Generator().manual_seed(seed + 1)).to(y_fit.device)]
    y_read = yr.to(y_fit.device)
    out: dict[str, Any] = {"n_fit": n, "n_read": int(yr.shape[0]), "fit_share_ones": float(y_fit.mean()),
                           "read_share_ones": float(y_read.mean()), "sites": {}}
    for site in Xf:
        X = Xf[site][:n]
        rec: dict[str, Any] = {}
        for kind, hidden in (("linear", 0), ("mlp", sz.hidden)):
            p, info = fit_probe(X, y_fit, hidden, sz, seed)
            q, info_s = fit_probe(X, y_shuf, hidden, sz, seed)
            rec[kind] = {"in_sample": accuracy(p, X, y_fit), "held_out": accuracy(p, Xr[site], y_read),
                         "shuffled_held_out": accuracy(q, Xr[site], y_read), "epochs": info["epochs"], "epochs_shuffled": info_s["epochs"]}
        out["sites"][site] = rec
        log(f"  probe {site}: linear {rec['linear']['held_out']:.4f} (shuffled {rec['linear']['shuffled_held_out']:.4f}), "
            f"MLP {rec['mlp']['held_out']:.4f} (shuffled {rec['mlp']['shuffled_held_out']:.4f})")
    out["max"] = {k: max(r[k]["held_out"] for r in out["sites"].values()) for k in ("linear", "mlp")}
    return out

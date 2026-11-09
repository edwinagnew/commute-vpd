"""E137's readings of a named (phi, s) and its two searches: G over small read-outs and H from z_M (appendix D, stage 2), and
the capacity read (ridge from z_M to the XOR). All numbers are means over a window set, equal weights; the distance d is total
variation; a share is 1 - mean d / gap with gap read once, on the read half, in stage 0."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from pieces import Config, Pieces, xor_f
from torch import Tensor

from commute.hidden_maps import (
    FINAL,
    STREAMS,
    Edit,
    Fn,
    Ridge,
    chunks,
    flat,
    kl,
    merged,
    run,
    tv,
    up_to,
    write_last,
)
from commute.steering import Net

SITES = ("in1", "in2", "in3", "out3")
H_SITES = (*SITES, FINAL)  # H only: KL is convex in the affine map at `final`, so the search there cannot stop at a poor local optimum
SFn = Callable[[Tensor, dict[str, Tensor]], Tensor]


def p_of(site: str, seen: dict[str, Tensor]) -> int:
    return sum(seen[s].flatten(1).shape[1] for s in up_to(site))


def read_pair(net: Net, windows: Tensor, site: str, Q: Tensor | None, s_fn: SFn | None, f_edit: Callable[[Tensor], Edit],
              cfg: Config) -> dict[str, float]:
    """For one (phi, s): `d` mean distance M[phi <- s] against F, `d_rev` mean distance F[phi <- v_M] against M, and the setter
    mismatch (mean squared distance of v_F from s over the variance of v_F, with the wording's zero-variance convention). With
    `s_fn` None only d_rev is read. v is taken at the last position (`Q` None: the whole activation)."""
    sites = up_to(site)
    n, d_nf, d_rev, sq, norm2 = 0, 0.0, 0.0, 0.0, 0.0
    mean_vf: Tensor | None = None
    for toks in chunks(windows, cfg.read_batch):
        lm, seen = run(net, toks, sites=sites)
        lf, sf = run(net, toks, sites=(site,), edit=f_edit(toks))
        vm, vf = (seen[site][:, -1] if Q is None else seen[site][:, -1] @ Q), (sf[site][:, -1] if Q is None else sf[site][:, -1] @ Q)
        l_rev, _ = run(net, toks, edit=merged(f_edit(toks), {site: write_last(Q, vm)}))
        d_rev += float(tv(l_rev[:, -1], lm[:, -1]).sum())
        if s_fn is not None:
            value = s_fn(vm, seen)
            ln, _ = run(net, toks, edit={site: write_last(Q, value)})
            d_nf += float(tv(ln[:, -1], lf[:, -1]).sum())
            sq += float(((vf - value) ** 2).sum())
        norm2 += float((vf**2).sum())
        mean_vf = vf.sum(0).double() if mean_vf is None else mean_vf + vf.sum(0).double()
        n += toks.shape[0]
    assert mean_vf is not None
    centred = norm2 / n - float(((mean_vf / n) ** 2).sum())
    mismatch = (sq / n / centred if centred > 0 else (0.0 if sq == 0 else 1.0)) if s_fn is not None else float("nan")
    return {"d": d_nf / n if s_fn is not None else float("nan"), "d_rev": d_rev / n, "mismatch": mismatch, "v_var": centred}


def _affine_edit(Q: Tensor, A: Tensor, c: Tensor) -> Fn:
    def f(x: Tensor) -> Tensor:
        x = x.clone()
        h = x[:, -1].clone()  # a separate tensor: writing x[:, -1] below would change a value the backward pass needs
        v = h @ Q
        x[:, -1] = h + (v @ A + c - v) @ Q.T
        return x

    return f


def g_search(net: Net, fit: Tensor, site: str, k: int, start: int, f_edit: Callable[[Tensor], Edit], cfg: Config,
             seed: int) -> dict[str, Any]:
    """One start of the search for a read-out phi (k columns by QR of a free matrix) and an affine map s = (A, c) minimising the mean
    KL(F || M[phi <- s]) on the fit half. Start 0 and 1: random Q. Start 2: the top principal directions of F's activation minus M's.
    A starts at the identity and c at zero, so every start begins at G = 0 on the page's scale. c is in units of S, the rms size
    of a coordinate of M's residual at this site, so the rate does not depend on that scale."""
    dev = str(next(net.net.parameters()).device)
    gen = torch.Generator().manual_seed(seed)
    probe = fit[: min(2048, fit.shape[0])]
    _, seen = run(net, probe, sites=(site,))
    h0 = seen[site][:, -1]
    d, S = h0.shape[-1], h0.pow(2).mean().sqrt()
    if start < 2:
        raw = torch.randn(d, k, generator=gen).to(dev)
    else:
        diff, n = [], 0
        for toks in chunks(fit[: min(8192, fit.shape[0])], cfg.read_batch):
            _, sm = run(net, toks, sites=(site,))
            _, sf = run(net, toks, sites=(site,), edit=f_edit(toks))
            diff.append(sf[site][:, -1] - sm[site][:, -1])
            n += toks.shape[0]
        vh = torch.linalg.svd(torch.cat(diff), full_matrices=False)[2]
        raw = vh[:k].T.clone()
        if raw.shape[1] < k:  # a stand-in or tiny card with fewer rows than k: complete with random directions
            raw = torch.cat([raw, torch.randn(d, k - raw.shape[1], generator=gen).to(dev)], 1)
    raw.requires_grad_()
    A = torch.eye(k, device=dev, requires_grad=True)
    c = torch.zeros(k, device=dev, requires_grad=True)
    opt = torch.optim.Adam([raw, A, c], lr=cfg.g_lr)
    n_fit, last = fit.shape[0], 0.0
    for _ in range(cfg.g_steps):
        toks = fit[torch.randint(0, n_fit, (cfg.g_batch,), generator=gen)]
        with torch.no_grad():
            lf = run(net, toks, edit=f_edit(toks))[0][:, -1]
        Q = torch.linalg.qr(raw).Q
        ln = run(net, toks, edit={site: _affine_edit(Q, A, S * c)}, grad=True)[0][:, -1]
        loss = kl(lf, ln).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    Q = torch.linalg.qr(raw.detach()).Q
    return {"Q": Q, "A": A.detach(), "c": (S * c).detach(), "last_kl": last}


def _fit_batches(net: Net, windows: Tensor, batch: int, R: Ridge, target: Callable[[int, Tensor, dict[str, Tensor]], dict[str, Tensor]],
                 top: str = "out3") -> None:
    """Fill `R` with z_M up to `top`: a first pass for the means and scales, a second for the products."""
    for s in range(0, windows.shape[0], batch):
        toks = windows[s : s + batch]
        _, seen = run(net, toks, sites=up_to(top))
        R.stats(flat(seen, top), target(s, toks, seen))
    R.freeze()
    for s in range(0, windows.shape[0], batch):
        toks = windows[s : s + batch]
        _, seen = run(net, toks, sites=up_to(top))
        R.add(flat(seen, top), target(s, toks, seen))


def ridge_select(R: Ridge, key: str, p: int, X_val: Tensor, Y_val: Tensor, grid: tuple[float, ...]) -> tuple[Tensor, float, float]:
    """(beta, lam, val_mse): the penalty of `grid` with the lowest validation error on (X_val, Y_val)."""
    best: tuple[float, float, Tensor] | None = None
    for lam in grid:
        beta = R.solve(key, p, lam)
        mse = float(((R.predict(X_val[:, :p], beta, key) - Y_val) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, lam, beta)
    assert best is not None
    return best[2], best[1], best[0]


def h_fit(net: Net, fit: Tensor, f_edit: Callable[[Tensor], Edit], cfg: Config, log: Any) -> tuple[Ridge, Tensor, dict[str, Tensor], dict[str, int]]:
    """The ridge accumulation for one pair: 80% of the fit half to fit, 20% to choose the penalty. Returns the filled Ridge, X_val
    (fp32, kept on the device: 20% of the fit half x z_M's p columns), Y_val (F's activation at each site) and p at each site."""
    dev = str(next(net.net.parameters()).device)
    cut_at = fit.shape[0] * 4 // 5
    train, val = fit[:cut_at], fit[cut_at:]
    def tgt(_i: int, toks: Tensor, _s: dict[str, Tensor]) -> dict[str, Tensor]:
        return {s: v[:, -1] for s, v in run(net, toks, sites=H_SITES, edit=f_edit(toks))[1].items()}

    R = Ridge(dev)
    _fit_batches(net, train, cfg.ridge_batch, R, tgt, top=FINAL)
    Xv: list[Tensor] = []
    Yv: dict[str, list[Tensor]] = {s: [] for s in H_SITES}
    ps: dict[str, int] = {}
    for toks in chunks(val, cfg.ridge_batch):
        _, seen = run(net, toks, sites=up_to(FINAL))
        if not ps:
            ps = {s: p_of(s, seen) for s in H_SITES}
        Xv.append(flat(seen, FINAL))
        for k, v in tgt(0, toks, seen).items():
            Yv[k].append(v)
    log(f"  ridge: {R.n} windows, {R.memory()}, val {val.shape[0]}")
    return R, torch.cat(Xv), {k: torch.cat(v) for k, v in Yv.items()}, ps


def h_refine(net: Net, fit: Tensor, val: Tensor, site: str, R: Ridge, beta: Tensor, f_edit: Callable[[Tensor], Edit], cfg: Config,
             seed: int) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """200 gradient steps on KL(F || M[phi <- s]) from the ridge fit; the better of the fit and its refinement by KL on validation
    windows. phi is the whole residual at `site`, last position. Returns (beta, ybar, {kl_val_ridge, kl_val_refined})."""
    dev = str(next(net.net.parameters()).device)
    def val_kl(b: Tensor, ybar: Tensor) -> float:
        tot = 0.0
        for toks in chunks(val[:4096], cfg.read_batch):
            lf = run(net, toks, edit=f_edit(toks))[0][:, -1]
            X = flat(run(net, toks, sites=up_to(site))[1], site)
            ln = run(net, toks, edit={site: write_last(None, ybar + R.standardise(X) @ b)})[0][:, -1]
            tot += float(kl(lf, ln).sum())
        return tot / min(4096, val.shape[0])

    ybar0, ridge_beta = R.y_mean[site].clone(), beta.clone()
    kl0 = val_kl(ridge_beta, ybar0)
    b, y = beta.clone().requires_grad_(), ybar0.clone().requires_grad_()
    opt = torch.optim.Adam([b, y], lr=cfg.h_lr)
    gen = torch.Generator().manual_seed(seed)
    n = fit.shape[0]
    for _ in range(cfg.h_steps):
        toks = fit[torch.randint(0, n, (cfg.h_batch,), generator=gen)]
        with torch.no_grad():
            lf = run(net, toks, edit=f_edit(toks))[0][:, -1]
            X = flat(run(net, toks, sites=up_to(site))[1], site)
        ln = run(net, toks, edit={site: write_last(None, y + R.standardise(X) @ b)}, grad=True)[0][:, -1]
        loss = kl(lf, ln).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    b_d, y_d = b.detach(), y.detach()
    kl1 = val_kl(b_d, y_d)
    use_b, use_y = (b_d, y_d) if kl1 < kl0 else (ridge_beta, ybar0)
    return use_b.to(dev), use_y.to(dev), {"kl_val_ridge": kl0, "kl_val_refined": kl1, "used": "refined" if kl1 < kl0 else "ridge"}


def capacity(net: Net, fit: Tensor, read: Tensor, p: Pieces, cfg: Config, log: Any) -> dict[str, Any]:
    """Ridge from z_M to the XOR and to its labels shuffled across windows, on the XOR pair's own windows, at each site's z_M: 80%
    of the fit half to fit, 20% to choose the penalty, then accuracy on the fit part (in sample) and on the read half (held out)."""
    dev = str(next(net.net.parameters()).device)
    cut_at = fit.shape[0] * 4 // 5
    train, val = fit[:cut_at], fit[cut_at:]
    gen = torch.Generator().manual_seed(137_500)
    labels = {"real": xor_f(p, train), "shuf": xor_f(p, train)[torch.randperm(train.shape[0], generator=gen)]}
    labels_v = {"real": xor_f(p, val), "shuf": xor_f(p, val)[torch.randperm(val.shape[0], generator=gen)]}
    labels_r = {"real": xor_f(p, read), "shuf": xor_f(p, read)[torch.randperm(read.shape[0], generator=gen)]}
    def tgt(i: int, toks: Tensor, _s: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v[i : i + toks.shape[0]].to(dev)[:, None] for k, v in labels.items()}

    R = Ridge(dev)
    _fit_batches(net, train, cfg.ridge_batch, R, tgt)
    Xv: list[Tensor] = []
    ps: dict[str, int] = {}
    for toks in chunks(val, cfg.ridge_batch):
        _, seen = run(net, toks, sites=STREAMS)
        if not ps:
            ps = {s: p_of(s, seen) for s in SITES}
        Xv.append(flat(seen, "out3"))
    X_val = torch.cat(Xv)
    betas: dict[tuple[str, str], Tensor] = {}
    out: dict[str, Any] = {}
    for site in SITES:
        out[site] = {}
        for key in ("real", "shuf"):
            beta, lam, _ = ridge_select(R, key, ps[site], X_val, labels_v[key].to(dev)[:, None], cfg.ridge_grid)
            betas[(site, key)] = beta
            out[site][key] = {"lam": lam}
    def accuracy(windows: Tensor, labs: dict[str, Tensor]) -> dict[tuple[str, str], float]:
        right = dict.fromkeys(betas, 0.0)
        for s in range(0, windows.shape[0], cfg.ridge_batch):
            toks = windows[s : s + cfg.ridge_batch]
            _, seen = run(net, toks, sites=STREAMS)
            X = flat(seen, "out3")
            for (site, key), b in betas.items():
                yhat = R.predict(X[:, :ps[site]], b, key).squeeze(1)
                right[(site, key)] += float(((yhat >= 0.5).float() == labs[key][s : s + cfg.ridge_batch].to(dev)).float().sum())
        return {k: v / windows.shape[0] for k, v in right.items()}

    in_s, held = accuracy(train, labels), accuracy(read, labels_r)
    for (site, key), v in in_s.items():
        out[site][key].update({"in_sample": v, "held_out": held[(site, key)]})
    out["held_out_real_by_site"] = {s: out[s]["real"]["held_out"] for s in SITES}
    log(f"  capacity: {out['held_out_real_by_site']}")
    return out


@torch.no_grad()
def z_rank(net: Net, windows: Tensor, cfg: Config) -> dict[str, Any]:
    """Rank of z_M up to `out3` on these windows, centred: singular values above 1e-6 of the largest, in fp32. A number below the
    columns is a bound; equal to the columns, it is exact."""
    Xs: list[Tensor] = []
    for toks in chunks(windows, cfg.ridge_batch):
        _, seen = run(net, toks, sites=STREAMS)
        Xs.append(flat(seen, "out3"))
    X = torch.cat(Xs)
    X = X - X.mean(0)
    sv = torch.linalg.svdvals(X)
    return {"rank": int((sv > 1e-6 * sv[0]).sum()), "columns": X.shape[1], "windows": X.shape[0], "sv_max": float(sv[0]),
            "sv_min_kept": float(sv[(sv > 1e-6 * sv[0])][-1])}

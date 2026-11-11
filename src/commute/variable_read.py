"""Reading a change to the 67M against a variable of M: the pairs' window sets, their pieces rebuilt from a saved file, the
search for a small read-out (G) and for an affine map from z_M (H), and the reversion and setter mismatch.

Moved on 2026-11-09 from `experiments/reuse_variable_test/` (pieces.py, search.py) so that R2 (cell P1.4,
`experiments/reuse_variable_finetune/`) reads a fine-tune exactly as R1 (E137, cell P1.3) read a pair, without importing
another experiment's folder. Moved, not rewritten: the seeds, window sets, steps and read rules are R1's. E137's scripts keep
their own copies and are not changed, because job E138 runs from them and may resume. The one difference: the changed model F
of a pair is passed as a `Target` (a callable), which is either M with its added piece (`piece_target`) or a fine-tuned `Net`
(`net_target`), where R1's functions took an edit of M. `Sizes` holds R1's sizes; a run adds its own fields in a subclass.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.data import pile_rows
from commute.hidden_maps import (
    FINAL,
    STREAMS,
    Edit,
    Fn,
    Ridge,
    add_last,
    chunks,
    flat,
    kl,
    merged,
    run,
    stash_last,
    tv,
    up_to,
    write_last,
)
from commute.kv import bijection, pick_marker
from commute.ladder import plain_mask
from commute.qwen_sft import lr_at
from commute.row_copy import lowercase_rows
from commute.steering import Net

PAIRS = ("Always on", "Cue", "XOR", "Table")
SITES = ("in1", "in2", "in3", "out3")
H_SITES = (*SITES, FINAL)  # H only: KL is convex in the affine map at `final`, so the search there cannot stop at a poor local optimum
SFn = Callable[[Tensor, dict[str, Tensor]], Tensor]
Target = Callable[..., tuple[Tensor, dict[str, Tensor]]]  # (toks, sites=(), edit=None) -> (logits, seen): the changed model F of a pair


@dataclass(frozen=True)
class Sizes:
    """R1's sizes (its `pieces.Config`, without the fields that only trained the pieces). `window`, `D`, `k`, `starts` and the
    rows must stay R1's, or the windows and the seeds are not R1's."""

    window: int = 8
    E: int = 2048
    D: int = 200_000  # per pair, half fit and half read
    k: int = 4
    n_keys: int = 32
    train_rows: int = 32768
    train_start: int = 0
    far_rows: int = 8192
    far_start: int = 32768
    val_rows: int = 2000
    lower_train: int = 2400
    lower_d: int = 3000
    read_batch: int = 2048
    ridge_batch: int = 1024
    ridge_grid: tuple[float, ...] = (1e-3, 1e-1, 10.0)
    g_steps: int = 400
    g_batch: int = 1024
    g_lr: float = 3e-3
    h_steps: int = 200
    h_batch: int = 1024
    h_lr: float = 3e-4
    starts: int = 3

    def __post_init__(self) -> None:
        assert self.D % 8 == 0 and self.window == 8 and self.starts == 3
        assert self.g_batch <= self.D // 2 and self.h_batch <= self.D // 2 and self.ridge_batch <= self.D // 2


@dataclass
class Sets:
    cased_train: Tensor
    lower_train: Tensor
    cased_d: Tensor  # rows D's cased, cue and key sides are cut from
    lower_d: Tensor
    far: Tensor  # D's far quarter
    e_rows: Tensor
    E: Tensor


@dataclass
class Task:
    cue: int
    keys: Tensor  # (K,)
    table: Tensor  # (V,), -1 off the keys
    bits: Tensor  # (V,) the XOR bit per token


@dataclass
class Pieces:
    Q: Tensor  # (d, 4) orthonormal
    b: Tensor  # (4,)
    w: Tensor  # (d,)
    a: Tensor  # (1,)
    C: Tensor  # (d, d)
    c: Tensor  # (d,)
    task: Task


def cut(rows: Tensor, n: int, length: int, gen: torch.Generator) -> Tensor:
    """n windows of `length` tokens at random rows and positions."""
    r = torch.randint(0, rows.shape[0], (n, 1), generator=gen)
    p = torch.randint(0, rows.shape[1] - length + 1, (n, 1), generator=gen)
    return rows[r, p + torch.arange(length)]


def make_sets(cfg: Sizes, tok: Any, log: Any) -> Sets:
    train = pile_rows("train", cfg.train_start, cfg.train_rows, log=log)
    far = pile_rows("train", cfg.far_start, cfg.far_rows, log=log)
    val = pile_rows("val", 0, cfg.val_rows, log=log)
    head = train[:4096]
    lower_train = lowercase_rows(head, tok, cfg.lower_train)
    lower_d = lowercase_rows(train[4096:8192], tok, cfg.lower_d)
    e = cut(val, cfg.E, cfg.window, torch.Generator().manual_seed(137_000))
    return Sets(head, lower_train, train[8192:], lower_d, far, val, e)


def make_task(sets: Sets, tok: Any, vocab: int, cfg: Sizes) -> Task:
    """The cue is the lowest-id punctuation token with no occurrence in any row loaded here; the 32 keys and their answers are plain
    tokens through `kv.bijection`; the XOR bits are one seeded bit per token."""
    mined = torch.cat([sets.cased_train, sets.cased_d, sets.lower_train, sets.lower_d, sets.far, sets.e_rows]).flatten()
    counts = {int(t): int(c) for t, c in enumerate(torch.bincount(mined, minlength=vocab).tolist()) if c}
    cue = pick_marker(counts, tok.decode, vocab)
    mask = plain_mask(tok.decode, vocab)
    mask[cue] = False
    pool = mask.nonzero().squeeze(1).tolist()
    gen = torch.Generator().manual_seed(137_001)
    keys = tuple(int(pool[int(i)]) for i in torch.randperm(len(pool), generator=gen)[: cfg.n_keys])
    table = bijection("e137", cue, keys, pool, 137_002)
    return Task(cue, torch.tensor(keys), table.table(vocab), torch.randint(0, 2, (vocab,), generator=gen))


def pair_windows(cfg: Sizes, pair: str, sets: Sets, task: Task) -> tuple[Tensor, Tensor]:
    """(fit, read) for a pair, D / 2 windows each, mixed: its two sides of the behaviour and a far quarter, equal weights."""
    if pair not in PAIRS:
        raise ValueError(pair)
    gen = torch.Generator().manual_seed(137_100 + PAIRS.index(pair))
    n_far, n_main = cfg.D // 4, cfg.D - cfg.D // 4
    far = cut(sets.far, n_far, cfg.window, gen)
    if pair == "Table":
        far = cut(sets.far, 2 * n_far, cfg.window, gen)
        far = far[~torch.isin(far[:, -1], task.keys)][:n_far]
    if pair == "Always on":
        main = torch.cat([cut(sets.lower_d, n_main // 2, cfg.window, gen), cut(sets.cased_d, n_main // 2, cfg.window, gen)])
    elif pair == "Cue":
        with_cue = cut(sets.cased_d, n_main // 2, cfg.window, gen)
        at = torch.randint(0, cfg.window, (n_main // 2,), generator=gen)
        with_cue[torch.arange(n_main // 2), at] = task.cue
        main = torch.cat([with_cue, cut(sets.cased_d, n_main // 2, cfg.window, gen)])
    elif pair == "XOR":
        main = cut(sets.cased_d, n_main, cfg.window, gen)
    else:
        keyed = cut(sets.cased_d, n_main // 2, cfg.window, gen)
        keyed[:, -1] = task.keys[torch.randperm(n_main // 2, generator=gen) % len(task.keys)]
        plain = cut(sets.cased_d, 2 * n_main // 2, cfg.window, gen)
        plain = plain[~torch.isin(plain[:, -1], task.keys)][: n_main // 2]
        main = torch.cat([keyed, plain])
    both = torch.cat([main, far])
    assert both.shape == (cfg.D, cfg.window), (pair, both.shape, far.shape)
    perm = torch.randperm(cfg.D, generator=gen)
    both = both[perm]
    return both[: cfg.D // 2], both[cfg.D // 2 :]


def load_pieces(path: Path, device: str, task: Task) -> Pieces:
    """The trained pieces E137 saved in its `results/large/pieces.pt` (its stage 0)."""
    st = torch.load(path, weights_only=False, map_location=device)
    return Pieces(st["Q"], st["b"], st["w"], st["a"], st["C"], st["c"], task)


def xor_f(p: Pieces, toks: Tensor) -> Tensor:
    return (p.task.bits[toks[:, -4:].cpu()].sum(1) % 2).float()


def piece_edit(net: Net, p: Pieces, pair: str, toks: Tensor) -> Edit:
    """The edit that makes M into M' = M + this piece, at block 2's input, last position."""
    d = next(net.net.parameters()).device
    Qb = (p.Q @ p.b).to(d)
    if pair == "Always on":
        return {"in2": add_last(Qb)}
    if pair == "Cue":
        store: dict[str, Tensor] = {}
        return {"in1": stash_last(store, "h"), "in2": add_last(lambda: (store["h"] @ p.w + p.a)[:, None] * Qb)}
    if pair == "XOR":
        return {"in2": add_last(xor_f(p, toks).to(d)[:, None] * Qb)}
    if pair == "Table":
        store = {}
        return {"in0": stash_last(store, "z"), "in2": add_last(lambda: store["z"] @ p.C.T + p.c)}
    raise ValueError(pair)


def piece_target(net: Net, p: Pieces, pair: str) -> Target:
    """F = M with its added piece, as R1 reads it."""

    def f(toks: Tensor, sites: Iterable[str] = (), edit: Edit | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        return run(net, toks, sites=sites, edit=merged(piece_edit(net, p, pair, toks), edit or {}))

    return f


def net_target(ft: Net) -> Target:
    """F = a fine-tuned copy of the model."""

    def f(toks: Tensor, sites: Iterable[str] = (), edit: Edit | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        return run(ft, toks, sites=sites, edit=edit)

    return f


def p_of(site: str, seen: dict[str, Tensor]) -> int:
    return sum(seen[s].flatten(1).shape[1] for s in up_to(site))


def read_pair(net: Net, windows: Tensor, site: str, Q: Tensor | None, s_fn: SFn | None, F: Target, cfg: Sizes) -> dict[str, float]:
    """For one (phi, s): `d` mean distance M[phi <- s] against F, `d_rev` mean distance F[phi <- v_M] against M, and the setter
    mismatch (mean squared distance of v_F from s over the variance of v_F, with the wording's zero-variance convention). With
    `s_fn` None only d_rev is read. v is taken at the last position (`Q` None: the whole activation)."""
    sites = up_to(site)
    n, d_nf, d_rev, sq, norm2 = 0, 0.0, 0.0, 0.0, 0.0
    mean_vf: Tensor | None = None
    for toks in chunks(windows, cfg.read_batch):
        lm, seen = run(net, toks, sites=sites)
        lf, sf = F(toks, (site,))
        vm, vf = (seen[site][:, -1] if Q is None else seen[site][:, -1] @ Q), (sf[site][:, -1] if Q is None else sf[site][:, -1] @ Q)
        l_rev, _ = F(toks, edit={site: write_last(Q, vm)})
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


def _g_start(net: Net, fit: Tensor, site: str, k: int, start: int, F: Target, cfg: Sizes, gen: torch.Generator, d: int, dev: str) -> Tensor:
    """The free matrix a start begins from. Starts 0 and 1: random. Start 2: the top principal directions of F's activation minus M's."""
    if start < 2:
        return torch.randn(d, k, generator=gen).to(dev)
    diff = []
    for toks in chunks(fit[: min(8192, fit.shape[0])], cfg.read_batch):
        _, sm = run(net, toks, sites=(site,))
        _, sf = F(toks, (site,))
        diff.append(sf[site][:, -1] - sm[site][:, -1])
    vh = torch.linalg.svd(torch.cat(diff), full_matrices=False)[2]
    raw = vh[:k].T.clone()
    if raw.shape[1] < k:  # a stand-in or tiny card with fewer rows than k: complete with random directions
        raw = torch.cat([raw, torch.randn(d, k - raw.shape[1], generator=gen).to(dev)], 1)
    return raw


def g_search(net: Net, fit: Tensor, site: str, k: int, start: int, F: Target, cfg: Sizes, seed: int) -> dict[str, Any]:
    """One start of the search for a read-out phi (k columns by QR of a free matrix) and an affine map s = (A, c) minimising the mean
    KL(F || M[phi <- s]) on the fit half. A starts at the identity and c at zero, so every start begins at G = 0 on the page's
    scale. c is in units of S, the rms size of a coordinate of M's residual at this site."""
    dev = str(next(net.net.parameters()).device)
    gen = torch.Generator().manual_seed(seed)
    probe = fit[: min(2048, fit.shape[0])]
    _, seen = run(net, probe, sites=(site,))
    h0 = seen[site][:, -1]
    d, S = h0.shape[-1], h0.pow(2).mean().sqrt()
    raw = _g_start(net, fit, site, k, start, F, cfg, gen, d, dev)
    raw.requires_grad_()
    A = torch.eye(k, device=dev, requires_grad=True)
    c = torch.zeros(k, device=dev, requires_grad=True)
    opt = torch.optim.Adam([raw, A, c], lr=cfg.g_lr)
    n_fit, last = fit.shape[0], 0.0
    for _ in range(cfg.g_steps):
        toks = fit[torch.randint(0, n_fit, (cfg.g_batch,), generator=gen)]
        with torch.no_grad():
            lf = F(toks)[0][:, -1]
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


def h_fit(net: Net, fit: Tensor, F: Target, cfg: Sizes, log: Any) -> tuple[Ridge, Tensor, dict[str, Tensor], dict[str, int]]:
    """The ridge accumulation for one F: 80% of the fit half to fit, 20% to choose the penalty. Returns the filled Ridge, X_val
    (fp32, kept on the device: 20% of the fit half x z_M's p columns), Y_val (F's activation at each site) and p at each site."""
    dev = str(next(net.net.parameters()).device)
    cut_at = fit.shape[0] * 4 // 5
    train, val = fit[:cut_at], fit[cut_at:]
    def tgt(_i: int, toks: Tensor, _s: dict[str, Tensor]) -> dict[str, Tensor]:
        return {s: v[:, -1] for s, v in F(toks, H_SITES)[1].items()}

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


def h_refine(net: Net, fit: Tensor, val: Tensor, site: str, R: Ridge, beta: Tensor, F: Target, cfg: Sizes,
             seed: int) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """200 gradient steps on KL(F || M[phi <- s]) from the ridge fit; the better of the fit and its refinement by KL on validation
    windows. phi is the whole residual at `site`, last position. Returns (beta, ybar, {kl_val_ridge, kl_val_refined})."""
    dev = str(next(net.net.parameters()).device)
    def val_kl(b: Tensor, ybar: Tensor) -> float:
        tot = 0.0
        for toks in chunks(val[:4096], cfg.read_batch):
            lf = F(toks)[0][:, -1]
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
            lf = F(toks)[0][:, -1]
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


def ridge_s(ridge: Ridge, site: str, b: Tensor, y: Tensor) -> SFn:
    """The setter of a fitted ridge: y + standardised z_M @ b (`read_pair`'s `s_fn`)."""

    def s_fn(_v: Tensor, seen: dict[str, Tensor]) -> Tensor:
        return y + ridge.standardise(flat(seen, site)) @ b

    return s_fn


def capacity(net: Net, fit: Tensor, read: Tensor, label: Callable[[Tensor], Tensor], cfg: Sizes, log: Any,
             seed: int = 137_500) -> dict[str, Any]:
    """E137's capacity read (its `search.capacity`, copied 2026-11-10 for E148 with the label passed as a function; E137's own
    copy is unchanged): ridge from z_M to a 0/1 label and to the labels shuffled across windows, at each site's z_M: 80% of the
    fit half to fit, 20% to choose the penalty, then accuracy on the fit part (in sample) and on the read half (held out).
    `label(toks)` gives (B,) floats in {0, 1}; E137's call is `label = lambda t: xor_f(pieces, t)`."""
    dev = str(next(net.net.parameters()).device)
    cut_at = fit.shape[0] * 4 // 5
    train, val = fit[:cut_at], fit[cut_at:]
    gen = torch.Generator().manual_seed(seed)
    labels = {"real": label(train), "shuf": label(train)[torch.randperm(train.shape[0], generator=gen)]}
    labels_v = {"real": label(val), "shuf": label(val)[torch.randperm(val.shape[0], generator=gen)]}
    labels_r = {"real": label(read), "shuf": label(read)[torch.randperm(read.shape[0], generator=gen)]}

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
                yhat = R.predict(X[:, : ps[site]], b, key).squeeze(1)
                right[(site, key)] += float(((yhat >= 0.5).float() == labs[key][s : s + cfg.ridge_batch].to(dev)).float().sum())
        return {k: v / windows.shape[0] for k, v in right.items()}

    in_s, held = accuracy(train, labels), accuracy(read, labels_r)
    for (site, key), v in in_s.items():
        out[site][key].update({"in_sample": v, "held_out": held[(site, key)]})
    out["held_out_real_by_site"] = {s: out[s]["real"]["held_out"] for s in SITES}
    log(f"  capacity: {out['held_out_real_by_site']}")
    return out


# ---- XOR2 (E148, cell P1.9): the XOR of two keyed bits in the last two positions ---------------------------------------


@dataclass
class Xor2:
    keys: Tensor  # (n_keys,) plain tokens, none of them E137's keys or cue
    bit: Tensor  # (V,) long: each key's bit (half the keys 0, half 1), -1 off the keys
    held: Tensor  # (n_keys, n_keys) bool: ordered key pairs held out of the fit half; the read half uses only these


@dataclass(frozen=True)
class Schedule:
    """A fine-tune's rate schedule: peak `lr`, linear warm-up over `warmup` steps, then cosine to 0 (`qwen_sft.lr_at`)."""

    steps: int
    lr: float
    warmup: int
    early: tuple[int, ...]  # steps at which the check windows are read

    def __post_init__(self) -> None:
        assert 0 < self.warmup < self.steps and self.early[-1] == self.steps and 10 in self.early


def make_xor2(task: Task, tok: Any, vocab: int, n_keys: int, held_share: float) -> Xor2:
    """`n_keys` plain tokens drawn after the 67M was trained, with a balanced seeded bit each; a seeded share of the ordered
    key pairs held out. The pool is `make_task`'s, without E137's cue and keys."""
    mask = plain_mask(tok.decode, vocab)
    mask[task.cue] = False
    mask[task.keys] = False
    pool = mask.nonzero().squeeze(1)
    gen = torch.Generator().manual_seed(148_001)
    keys = pool[torch.randperm(pool.numel(), generator=gen)[:n_keys]]
    bit = torch.full((vocab,), -1, dtype=torch.long)
    bit[keys] = (torch.randperm(n_keys, generator=gen) < n_keys // 2).long()
    n_held = round(held_share * n_keys * n_keys)
    held = torch.zeros(n_keys * n_keys, dtype=torch.bool)
    held[torch.randperm(n_keys * n_keys, generator=gen)[:n_held]] = True
    return Xor2(keys, bit, held.view(n_keys, n_keys))


def xor2_keyed(x: Xor2, toks: Tensor) -> Tensor:
    """(B,) bool: both of the last two tokens are keys."""
    return (x.bit[toks[:, -2:].cpu()] >= 0).all(1)


def xor2_f(x: Xor2, toks: Tensor) -> Tensor:
    """(B,) float: 1 where the last two tokens are keys whose bits differ, else 0 (0 on every window without two keys)."""
    b = x.bit[toks[:, -2:].cpu()]
    return ((b >= 0).all(1) & (b[:, 0] != b[:, 1])).float()


def xor2_windows(cfg: Sizes, sets: Sets, x: Xor2) -> tuple[Tensor, Tensor]:
    """(fit, read), D / 2 windows each: 3/4 cased windows whose last two tokens are an ordered key pair (fit: pairs not held
    out; read: held-out pairs only), 1/4 far windows with no key in the last two positions; shuffled within each half."""
    gen = torch.Generator().manual_seed(148_100)
    half = cfg.D // 2
    n_far = half // 4
    n_key = half - n_far
    halves = []
    for held in (False, True):
        i, j = (x.held == held).nonzero(as_tuple=True)
        pick = torch.randint(0, i.numel(), (n_key,), generator=gen)
        keyed = cut(sets.cased_d, n_key, cfg.window, gen)
        keyed[:, -2], keyed[:, -1] = x.keys[i[pick]], x.keys[j[pick]]
        far = cut(sets.far, 2 * n_far, cfg.window, gen)
        far = far[~torch.isin(far[:, -2:], x.keys).any(1)][:n_far]
        both = torch.cat([keyed, far])
        assert both.shape == (half, cfg.window), both.shape
        halves.append(both[torch.randperm(half, generator=gen)])
    return halves[0], halves[1]


def xor2_target(net: Net, x: Xor2, push: Tensor) -> Target:
    """F = `net` + `push` (d,) at block 2's input, last position, on the windows where `xor2_f` is 1."""

    def f(toks: Tensor, sites: Iterable[str] = (), edit: Edit | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        e = {"in2": add_last(xor2_f(x, toks).to(push.device)[:, None] * push)}
        return run(net, toks, sites=sites, edit=merged(e, edit or {}))

    return f


def finetune(net: Net, ft: Net, F: Target, fit: Tensor, check: Tensor, sch: Schedule, batch: int, seed: int,
             log: Any) -> dict[str, Any]:
    """Train every weight of `ft` (a copy of `net`) in place toward F: E139's recipe (Adam, no weight decay, clip 1, loss
    KL(F || ft) at the last position, `batch` windows drawn from `fit` per step) on the schedule `sch`. At each `sch.early` step
    the check windows are read. At step 10, E139 Amendment 2's check: KL(F || ft) below KL(F || net) and KL(net || ft) at most
    KL(net || F), else training ends there with `stopped` set. Returns the record; `ft` is left with its gradients off."""
    params = list(ft.net.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=sch.lr, betas=(0.9, 0.999), weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)

    def last(G: Target) -> Tensor:
        return torch.cat([G(t)[0][:, -1] for t in chunks(check, 2048)])

    lm, lf = last(net_target(net)), last(F)
    rec: dict[str, Any] = {"kl_F_M": float(kl(lf, lm).mean()), "kl_M_F": float(kl(lm, lf).mean()), "curve": [], "early": [],
                           "steps": sch.steps, "lr": sch.lr, "warmup": sch.warmup, "stopped": None}
    for t in range(1, sch.steps + 1):
        rate = lr_at(t, sch.lr, sch.warmup, sch.steps, "hf_cosine")
        for g in opt.param_groups:
            g["lr"] = rate
        toks = fit[torch.randint(0, fit.shape[0], (batch,), generator=gen)]
        with torch.no_grad():
            target = F(toks)[0][:, -1]
        loss = kl(target, run(ft, toks, grad=True)[0][:, -1]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        rec["curve"].append(float(loss.detach()))
        if t in sch.early:
            lft = last(net_target(ft))
            e = {"step": t, "lr": rate, "kl_to_F": float(kl(lf, lft).mean()), "kl_from_M": float(kl(lm, lft).mean())}
            rec["early"].append(e)
            log(f"  step {t}: loss {rec['curve'][-1]:.5f}, KL(F||ft) {e['kl_to_F']:.5f}, KL(M||ft) {e['kl_from_M']:.3e}, rate {rate:.2e}")
            if t == 10 and not (e["kl_to_F"] < rec["kl_F_M"] and e["kl_from_M"] <= rec["kl_M_F"]):
                rec["stopped"] = "overshoot at step 10: KL(F||ft) not below KL(F||M), or KL(M||ft) above KL(M||F)"
                break
    for p in params:
        p.requires_grad_(False)
    return rec


# ---- Number pairs (E150, cell P1.10): two single-token numbers in the last two positions --------------------------------
# Added 2026-11-11, beside E148's XOR2 code, which is unchanged. A number pair is an `Xor2` whose keys are number tokens and
# whose bit is each number's parity, so `xor2_keyed`, `xor2_f` (then the parity of a + b, odd exactly when one of a, b is
# odd) and `xor2_windows` serve it as they are.


def make_numbers(tok: Any, vocab: int, n_max: int, held_share: float, seed: int = 150_001) -> Xor2:
    """The numbers 0 to n_max - 1 whose leading-space form " n" is one token of `tok` (and decodes back to " n"), each with
    its parity as the bit; a seeded share of the ordered pairs held out, as `make_xor2`."""
    ids = [tok.encode(f" {n}") for n in range(n_max)]
    nums = [n for n, i in enumerate(ids) if len(i) == 1 and tok.decode(i) == f" {n}"]
    keys = torch.tensor([ids[n][0] for n in nums])
    assert keys.unique().numel() == keys.numel() and int(keys.max()) < vocab, "number tokens collide or fall off the vocabulary"
    bit = torch.full((vocab,), -1, dtype=torch.long)
    bit[keys] = torch.tensor(nums) % 2
    k = keys.numel()
    held = torch.zeros(k * k, dtype=torch.bool)
    held[torch.randperm(k * k, generator=torch.Generator().manual_seed(seed))[: round(held_share * k * k)]] = True
    return Xor2(keys, bit, held.view(k, k))


def odd_at(x: Xor2, toks: Tensor, pos: int) -> Tensor:
    """(B,) float: 1 where the token at `pos` is a key with bit 1 (an odd number under `make_numbers`), else 0."""
    return (x.bit[toks[:, pos].cpu()] == 1).float()


def label_target(net: Net, label: Callable[[Tensor], Tensor], push: Tensor) -> Target:
    """F = `net` + label(toks) * `push` (d,) at block 2's input, last position. `xor2_target` is this with `xor2_f`."""

    def f(toks: Tensor, sites: Iterable[str] = (), edit: Edit | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        e = {"in2": add_last(label(toks).to(push.device)[:, None] * push)}
        return run(net, toks, sites=sites, edit=merged(e, edit or {}))

    return f


def mean_tv(A: Target, B: Target, windows: Tensor, batch: int) -> dict[str, float]:
    """Mean and largest TV between A's and B's next-token distributions at the last position."""
    d = torch.cat([tv(A(t)[0][:, -1], B(t)[0][:, -1]) for t in chunks(windows, batch)])
    return {"mean": float(d.mean()), "max": float(d.max())}


def keyed_capacity(net: Net, x: Xor2, label: Callable[[Tensor], Tensor], fit: Tensor, read: Tensor, cfg: Sizes,
                   log: Any) -> dict[str, Any]:
    """E137's capacity read (`capacity`) of `label` on the windows whose last two tokens are keys only, as E148 read XOR2 (a
    window without keys has no label, and scoring it 0 would let key presence, which is linear, read above chance)."""
    cap = capacity(net, fit[xor2_keyed(x, fit)], read[xor2_keyed(x, read)], label, cfg, log)
    by_site = cap["held_out_real_by_site"]
    return {"held_out_real_by_site": by_site, "max": max(by_site.values()),
            "shuffled_by_site": {s: cap[s]["shuf"]["held_out"] for s in SITES}}


def learned_ratio(base: Net, F: Target, ft: Net, push: Tensor, check: Tensor, batch: int) -> dict[str, float]:
    """E139's `xor_learned` for a fine-tune `ft` of `base` toward F = base + bit * push: KL(F || ft) over KL(F || base + push / 2),
    the best a model blind to the bit can do where the bit is on half the windows. The caller compares the ratio with its bar."""
    half = {"in2": add_last(0.5 * push)}
    lf = torch.cat([F(t)[0][:, -1] for t in chunks(check, batch)])
    l_ft = torch.cat([run(ft, t)[0][:, -1] for t in chunks(check, batch)])
    l_half = torch.cat([run(base, t, edit=half)[0][:, -1] for t in chunks(check, batch)])
    l_base = torch.cat([run(base, t)[0][:, -1] for t in chunks(check, batch)])
    k_ft, k_half = float(kl(lf, l_ft).mean()), float(kl(lf, l_half).mean())
    assert k_half > 0, "a zero push"
    return {"kl_ft": k_ft, "kl_half_push": k_half, "kl_base": float(kl(lf, l_base).mean()), "ratio": k_ft / k_half}


def ladder(base: Net, F: Target, fit: Tensor, check: Tensor, rungs: tuple[Schedule, ...], batch: int, seed: int,
           clone: Callable[[Net], Net], passes: Callable[[Net, dict[str, Any]], bool],
           log: Any) -> tuple[Net | None, int, dict[str, Any]]:
    """Fine-tune `clone(base)` toward F on each schedule in turn, each from `base` again, until `passes(ft, record)`.
    Returns (the first copy that passes, its rung, the records by step count); (None, -1, records) if none passes, and
    (None, rung, records) at once when a rung's step-10 check stopped it (its record's `stopped` is set)."""
    recs: dict[str, Any] = {}
    for r, sch in enumerate(rungs):
        log(f"  fine-tune: {sch.steps} steps, peak {sch.lr}")
        ft = clone(base)
        recs[str(sch.steps)] = rec = finetune(base, ft, F, fit, check, sch, batch, seed, log)
        if rec["stopped"]:
            return None, r, recs
        if passes(ft, rec):
            return ft, r, recs
    return None, -1, recs

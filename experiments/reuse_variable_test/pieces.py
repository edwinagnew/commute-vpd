"""E137's sizes (`Config`), rows and input sets, and the four added pieces. Appendix D of the plan:
reports/plans/2026-11-09_plan_reuse-variable-first-test-appendix.md. Every size that differs between the scored run and a
dry run is a field of the one frozen dataclass; its `__post_init__` asserts they agree."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.data import pile_rows
from commute.hidden_maps import Edit, Ridge, add_last, chunks, run, stash_last
from commute.kv import bijection, pick_marker
from commute.ladder import plain_mask
from commute.row_copy import lowercase_rows
from commute.steering import Net

HERE = Path(__file__).resolve().parent
PAIRS = ("Always on", "Cue", "XOR", "Table")


@dataclass(frozen=True)
class Config:
    results: Path = HERE / "results"
    window: int = 8
    E: int = 2048  # page: windows from rows no pair trains on
    D: int = 200_000  # page: per pair, half fit and half read
    k: int = 4  # page
    k_figure: tuple[int, ...] = (1, 4, 16)  # page: G also at 1 and 16, figure only
    n_keys: int = 32  # page
    train_rows: int = 32768
    train_start: int = 0
    far_rows: int = 8192
    far_start: int = 32768
    val_rows: int = 2000
    lower_train: int = 2400  # lowercase rows the Always on piece trains on (commute.lowercase's count)
    lower_d: int = 3000  # lowercase rows D's lowercased side is cut from
    ao_steps: int = 800
    ao_batch: int = 512
    ao_lr: float = 1e-2  # b is in units of S; 800 steps can reach ~8 S per coordinate, so the fit is not stopped by the rate
    table_steps: int = 800
    table_batch: int = 512
    table_lr: float = 1e-3
    cue_fit: int = 20_000  # windows, half with the cue
    cue_lambda: float = 1e-3
    read_batch: int = 2048
    rank_windows: int = 34_000  # z_M has 30,720 columns: this many windows bound the rank above any D's
    ridge_batch: int = 1024
    ridge_grid: tuple[float, ...] = (1e-3, 1e-1, 10.0)  # per sample, on standardised inputs
    g_steps: int = 400
    g_batch: int = 1024
    g_lr: float = 3e-3
    h_steps: int = 200  # appendix D
    h_batch: int = 1024
    h_lr: float = 3e-4
    starts: int = 3  # page: two random, one from the principal directions
    first_n: int = 4096  # windows of D's read half for the first wording's exact checks
    directions: int = 100  # page
    wall_limit_s: int = 6300  # stop cleanly with exit 3 before a debug job's limit (1:55): 105 min, leaving a pair's longest step

    def __post_init__(self) -> None:
        assert self.D % 8 == 0, "D must halve and quarter into whole windows, and each side too"
        assert self.window == 8 and self.k in self.k_figure and self.starts == 3
        assert self.g_batch <= self.D // 2 and self.h_batch <= self.D // 2 and self.ridge_batch <= self.D // 2
        assert self.first_n <= self.D // 2 and self.rank_windows <= self.D

    def tiny(self, **kw: Any) -> Config:
        return replace(self, **kw)


@dataclass
class Sets:
    cased_train: Tensor  # rows the pieces train on
    lower_train: Tensor
    cased_d: Tensor  # rows D's cased, cue and key sides are cut from
    lower_d: Tensor
    far: Tensor  # D's far quarter
    e_rows: Tensor  # E
    E: Tensor  # (E, window)


@dataclass
class Task:
    cue: int
    keys: Tensor  # (K,)
    table: Tensor  # (V,), -1 off the keys
    bits: Tensor  # (V,) the XOR bit per token


def cut(rows: Tensor, n: int, length: int, gen: torch.Generator) -> Tensor:
    """n windows of `length` tokens at random rows and positions."""
    r = torch.randint(0, rows.shape[0], (n, 1), generator=gen)
    p = torch.randint(0, rows.shape[1] - length + 1, (n, 1), generator=gen)
    return rows[r, p + torch.arange(length)]


def make_sets(cfg: Config, tok: Any, log: Any) -> Sets:
    train = pile_rows("train", cfg.train_start, cfg.train_rows, log=log)
    far = pile_rows("train", cfg.far_start, cfg.far_rows, log=log)
    val = pile_rows("val", 0, cfg.val_rows, log=log)
    head = train[:4096]
    lower_train = lowercase_rows(head, tok, cfg.lower_train)
    lower_d = lowercase_rows(train[4096:8192], tok, cfg.lower_d)
    e = cut(val, cfg.E, cfg.window, torch.Generator().manual_seed(137_000))
    return Sets(head, lower_train, train[8192:], lower_d, far, val, e)


def make_task(sets: Sets, tok: Any, vocab: int, cfg: Config) -> Task:
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


def pair_windows(cfg: Config, pair: str, sets: Sets, task: Task) -> tuple[Tensor, Tensor]:
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


# ---- the pieces. Start values give M' = M exactly: `Pieces.at_start`.


@dataclass
class Pieces:
    Q: Tensor  # (d, 4) orthonormal
    b: Tensor  # (4,)
    w: Tensor  # (d,)
    a: Tensor  # (1,)
    C: Tensor  # (d, d)
    c: Tensor  # (d,)
    task: Task

    def at_start(self) -> Pieces:
        z = torch.zeros_like
        return Pieces(self.Q, z(self.b), z(self.w), z(self.a), z(self.C), z(self.c), self.task)

    def state(self) -> dict[str, Any]:
        return {"Q": self.Q.cpu(), "b": self.b.cpu(), "w": self.w.cpu(), "a": self.a.cpu(), "C": self.C.cpu(), "c": self.c.cpu()}


def xor_f(p: Pieces, toks: Tensor) -> Tensor:
    return (p.task.bits[toks[:, -4:].cpu()].sum(1) % 2).float()


def cue_f(p: Pieces, h1: Tensor) -> Tensor:
    return h1 @ p.w + p.a


def piece_edit(net: Net, p: Pieces, pair: str, toks: Tensor) -> Edit:
    """The edit that makes M into M′ = M + this piece. Everything writes at block 2's input, last position, as the page says."""
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


def train_always_on(net: Net, sets: Sets, cfg: Config, log: Any) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Q (d, 4) and b (4,): orthonormal directions and a constant push at block 2's input, last position, trained on next-token
    cross-entropy on lowercased windows. Q by QR of a free matrix; b in units of S, the rms size of a coordinate of that residual,
    so the rate does not depend on the card's activation scale."""
    d_dev = next(net.net.parameters()).device
    gen = torch.Generator().manual_seed(137_200)
    probe = cut(sets.lower_d, cfg.read_batch, cfg.window, gen)
    _, seen = run(net, probe, sites=("in2",))
    S = seen["in2"][:, -1].pow(2).mean().sqrt()
    d = seen["in2"].shape[-1]
    raw = torch.randn(d, cfg.k, generator=gen).to(d_dev).requires_grad_()
    b = torch.zeros(cfg.k, device=d_dev, requires_grad=True)
    pool = cut(sets.lower_train, cfg.ao_steps * cfg.ao_batch, cfg.window + 1, gen)
    opt = torch.optim.Adam([raw, b], lr=cfg.ao_lr)
    losses = []
    for i, w in enumerate(chunks(pool, cfg.ao_batch)):
        Q = torch.linalg.qr(raw).Q
        logits, _ = run(net, w[:, :-1], edit={"in2": add_last(Q @ (S * b))}, grad=True)
        loss = torch.nn.functional.cross_entropy(logits[:, -1], w[:, -1].to(d_dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if i % max(cfg.ao_steps // 10, 1) == 0:
            log(f"  always on step {i}: loss {losses[-1]:.4f}")
    Q, bd = torch.linalg.qr(raw.detach()).Q, (S * b.detach())
    held = cut(sets.lower_d, 4096, cfg.window + 1, gen)
    ce0, ce1 = 0.0, 0.0
    with torch.no_grad():
        for w in chunks(held, cfg.read_batch):
            t = w[:, -1].to(d_dev)
            ce0 += float(torch.nn.functional.cross_entropy(run(net, w[:, :-1])[0][:, -1], t, reduction="sum"))
            ce1 += float(torch.nn.functional.cross_entropy(run(net, w[:, :-1], edit={"in2": add_last(Q @ bd)})[0][:, -1], t, reduction="sum"))
    info = {"loss_first": losses[0], "loss_last": losses[-1], "ce_held_M": ce0 / held.shape[0], "ce_held_piece": ce1 / held.shape[0],
            "S": float(S), "b_norm": float(bd.norm())}
    log(f"  always on: {info}")
    return Q, bd, info


def fit_cue(net: Net, sets: Sets, task: Task, cfg: Config, log: Any) -> tuple[Tensor, Tensor, dict[str, float]]:
    """(w, a): f = w h + a with h block 1's input at the last position, a ridge fit to "the cue is in the window", frozen. Half the
    windows have the cue written at a random position. Fit on the train rows; accuracy read on D's rows."""
    gen = torch.Generator().manual_seed(137_300)

    def batch(rows: Tensor, n: int) -> tuple[Tensor, Tensor]:
        toks = cut(rows, n, cfg.window, gen)
        lab = torch.zeros(n)
        i = n // 2
        toks[torch.arange(i), torch.randint(0, cfg.window, (i,), generator=gen)] = task.cue
        lab[:i] = 1.0
        return toks, lab[:, None]

    def stream(R: Ridge, toks: Tensor, y: Tensor, stage: str) -> None:
        for s in range(0, toks.shape[0], cfg.read_batch):
            _, seen = run(net, toks[s : s + cfg.read_batch], sites=("in1",))
            f = R.stats if stage == "stats" else R.add
            f(seen["in1"][:, -1], {"y": y[s : s + cfg.read_batch]})

    R = Ridge(str(next(net.net.parameters()).device))
    toks, y = batch(sets.cased_train, cfg.cue_fit)
    stream(R, toks, y, "stats")
    R.freeze()
    stream(R, toks, y, "add")
    beta = R.solve("y", R.mean.numel(), cfg.cue_lambda).squeeze(1)
    w = beta / R.std
    a = R.y_mean["y"] - w @ R.mean
    held, yh = batch(sets.cased_d, min(4096, cfg.cue_fit))
    right = 0.0
    with torch.no_grad():
        for s in range(0, held.shape[0], cfg.read_batch):
            _, seen = run(net, held[s : s + cfg.read_batch], sites=("in1",))
            f = seen["in1"][:, -1] @ w + a
            right += float(((f >= 0.5).float().cpu() == yh[s : s + cfg.read_batch].squeeze(1)).sum())
    acc = right / held.shape[0]
    log(f"  cue read-out held-out accuracy {acc:.4f} (lambda {cfg.cue_lambda}, {cfg.cue_fit} windows fit)")
    return w.detach(), a.detach(), {"acc_held": acc, "lambda": cfg.cue_lambda}


def train_table(net: Net, sets: Sets, task: Task, cfg: Config, log: Any) -> tuple[Tensor, Tensor, dict[str, float]]:
    """C (d, d) and c (d,): h <- h + C z + c with z block 0's input (the token's embedding) at the last position, trained by Adam to
    send the 32 keys to their bijection answers at the last position, on windows from the training rows with a key written last."""
    dev = next(net.net.parameters()).device
    gen = torch.Generator().manual_seed(137_400)
    _, seen = run(net, sets.E[:64], sites=("in0",))
    d = seen["in0"].shape[-1]
    C = torch.zeros(d, d, device=dev, requires_grad=True)
    c = torch.zeros(d, device=dev, requires_grad=True)
    opt = torch.optim.Adam([C, c], lr=cfg.table_lr)
    n = cfg.table_steps * cfg.table_batch
    pool = cut(sets.cased_train, n, cfg.window, gen)
    keys = task.keys[torch.randperm(n, generator=gen) % len(task.keys)]
    pool[:, -1] = keys
    held = cut(sets.cased_d, 4096, cfg.window, gen)
    keys_h = task.keys[torch.randperm(4096, generator=gen) % len(task.keys)]
    held[:, -1] = keys_h
    table = task.table.to(dev)
    losses: list[float] = []

    def add(store: dict[str, Tensor]) -> Tensor:
        return store["z"] @ C.T + c

    for i, w in enumerate(chunks(pool, cfg.table_batch)):
        store: dict[str, Tensor] = {}
        ed: Edit = {"in0": stash_last(store, "z"), "in2": add_last(lambda s=store: add(s))}
        logits, _ = run(net, w, edit=ed, grad=True)
        loss = torch.nn.functional.cross_entropy(logits[:, -1], table[w[:, -1].to(dev)])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if i % max(cfg.table_steps // 10, 1) == 0:
            log(f"  table step {i}: loss {losses[-1]:.4f}")
    right = 0.0
    with torch.no_grad():
        for w in chunks(held, cfg.read_batch):
            store = {}
            ed = {"in0": stash_last(store, "z"), "in2": add_last(lambda s=store: add(s))}
            logits, _ = run(net, w, edit=ed)
            right += float((logits[:, -1].argmax(-1) == table[w[:, -1].to(dev)]).float().sum())
    rank = int(torch.linalg.matrix_rank(C.detach().float()))
    info = {"loss_first": losses[0], "loss_last": losses[-1], "acc_held": right / held.shape[0], "rank_C": rank}
    log(f"  table: {info}")
    return C.detach(), c.detach(), info

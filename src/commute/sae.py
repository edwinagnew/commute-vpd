"""BatchTopK sparse autoencoders on the 67M's activations (E234, cell P1.40), SAEBench's recipe re-implemented.

Source, read-only clones: saprmarks/dictionary_learning@60ec6bf (2025-08-20), `dictionary_learning/trainers/
batch_top_k.py` (BT below) and `trainers/trainer.py` (TR); BT is identical, apart from k annealing (off by default and
not used here), to 43421f5, the dictionary_learning commit pinned by adamkarvonen/dictionary_learning_demo@d39d54a, the
demo's version when the SAEBench suite was trained (demo_config.py DC, demo.py DM, training.py TS and buffer.py BF at
43421f5). Settings: SAEBench (Karvonen et al. 2025, arXiv 2503.09532) Appendix B, Table 3. What is taken:
  - `SAE`: BT:17-72. Decoder columns unit norm at init (TR:65-81), encoder = decoder transpose, encoder bias 0, b_dec 0
    then the geometric median of the first batch (BT:291-294, 344-358); BatchTopK over the flattened batch in training
    (BT:47-55), the threshold at inference (BT:42-45), its EMA of the smallest kept activation, beta 0.999 from step
    1000 (BT:110-111, 234-249, 257-258).
  - `Trainer.update`: BT:290-318. Loss sum_d (x - xhat)^2 averaged over tokens (BT:272) + auxk_alpha 1/32 times the
    aux loss on dead latents (BT:192-232: top d/2 dead pre-activations decode the residual, normalised by the
    residual's variance; dead = no firing for 10M tokens, BT:152-153, 266-270); the decoder gradient's component along
    each column removed (TR:84-111), gradient norm clipped at 1, Adam betas (0.9, 0.999) (BT:164-166), columns
    renormalised after the step. Learning rate: linear warm-up from 0 over 1000 steps, linear decay to 0 over the last
    20% (TR:114-169; DC:60-62, 64 lr 3e-4; Table 3).
  - `Buffer`: BF:63-76, 104-153 (250,000 activations, a random unread batch per step, refilled when half are read;
    DM:63, 75; SAEBench App. B). Each activation is read once.
  - Activation normalisation: one scalar per site, sqrt of the mean squared norm over 100 batches (TS:83-108), applied
    in training and folded into the weights afterwards (TS:190-204, 241-246; BT:74-78 `scale_biases`).
  - Training under bf16 autocast with the activations cast to bf16 (TS:141, 200; DM:147), fp32 parameters.
Departures, each for a stated reason:
  - All sites' SAEs (and a second seed) train in one process from one stream: one forward of the 67M feeds every
    site, each SAE sees the same activations in the same order (SAEBench App. B: "identical data and data ordering").
  - Position 0 of every Pile row is dropped (from training, FVE and the splice): the rows have no BOS, and position 0 of
    a 512-token row is an attention-sink position, as E203 dropped it (crosscoder.acts); the source at 43421f5 kept
    every position (BF:30 remove_bos False), dictionary_learning_demo HEAD drops BOS (demo_config.py remove_bos True).
  - Context 512 tokens (the 67M's n_ctx) instead of 1,024 (DC:69-70); the buffer holds 489 rows of 511 tokens.
  - The model runs under bf16 autocast (the lead's setting) where the Pythia suite ran fp32 (DC:69-70).
  - No logging, wandb or k annealing; state() / load() added for resume.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

LAYERS = 4
SITES = ("resid_pre0", *(f"{kind}{l}" for l in range(LAYERS) for kind in ("attn", "mlp", "resid_post")))


# ---- the model ---------------------------------------------------------------------------------------------------------


class SAE(nn.Module):
    """BT:17-72: f = BatchTopK(relu(W_e (x - b_dec) + b_e)) in training, f > threshold at inference; xhat = W_d f + b_dec."""

    k: Tensor
    threshold: Tensor

    def __init__(self, d: int, width: int, k: int) -> None:
        super().__init__()
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self.decoder = nn.Linear(width, d, bias=False)
        unit_norm(self.decoder.weight)
        self.encoder = nn.Linear(d, width)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(d))

    def pre(self, x: Tensor) -> Tensor:
        return torch.relu(self.encoder(x - self.b_dec))

    def encode(self, x: Tensor, use_threshold: bool = True) -> Tensor:
        p = self.pre(x)
        if use_threshold:
            return p * (p > self.threshold)
        top = p.flatten().topk(int(self.k) * x.shape[0], sorted=False)
        return torch.zeros_like(p.flatten()).scatter_(-1, top.indices, top.values).reshape(p.shape)

    def decode(self, f: Tensor) -> Tensor:
        return self.decoder(f) + self.b_dec

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def fold(self, scale: float) -> None:
        """BT:74-78: weights for raw activations from weights trained on activations / scale."""
        self.encoder.bias.mul_(scale)
        self.b_dec.mul_(scale)
        if self.threshold >= 0:
            self.threshold.mul_(scale)


@torch.no_grad()
def unit_norm(W: Tensor) -> None:
    """TR:65-81 on a (d, width) decoder weight."""
    W.div_(W.norm(dim=0, keepdim=True) + torch.finfo(W.dtype).eps)


@torch.no_grad()
def drop_parallel(W: Tensor, grad: Tensor) -> None:
    """TR:84-111: remove each decoder column's gradient component along the column."""
    Wn = W / (W.norm(dim=0, keepdim=True) + 1e-6)
    grad -= (grad * Wn).sum(0, keepdim=True) * Wn


def lr_factor(steps: int, warmup: int, decay_start: int) -> Callable[[int], float]:
    """TR:114-169 with resample_steps None."""
    assert 0 <= warmup < decay_start < steps, (warmup, decay_start, steps)
    return lambda s: s / warmup if s < warmup else (steps - s) / (steps - decay_start) if s >= decay_start else 1.0


def geometric_median(x: Tensor, max_iter: int = 100, tol: float = 1e-5) -> Tensor:
    """BT:344-358."""
    guess = x.mean(0)
    for _ in range(max_iter):
        prev = guess
        w = 1 / torch.norm(x - guess, dim=1)
        guess = ((w / w.sum())[:, None] * x).sum(0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


class Trainer:
    """BT:96-318: one SAE, its Adam and schedule, its dead-latent counter."""

    def __init__(self, d: int, width: int, k: int, lr: float, steps: int, warmup: int, decay_start: int,
                 auxk_alpha: float, dead_tokens: int, threshold_beta: float, threshold_start: int, seed: int) -> None:  # fmt: skip
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.ae = SAE(d, width, k).cuda()
        self.auxk_alpha, self.dead_tokens, self.top_k_aux = auxk_alpha, dead_tokens, d // 2
        self.beta, self.threshold_start = threshold_beta, threshold_start
        self.since = torch.zeros(width, dtype=torch.long, device="cuda")
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_factor(steps, warmup, decay_start))

    def aux_loss(self, e: Tensor, p: Tensor) -> Tensor:
        """BT:192-232."""
        dead = self.since >= self.dead_tokens
        n = int(dead.sum())
        if n == 0:
            return torch.zeros((), device=e.device)
        acts, idx = torch.where(dead[None], p, -torch.inf).topk(min(self.top_k_aux, n), sorted=False)
        rec = self.ae.decoder(torch.zeros_like(p).scatter_(-1, idx, acts))
        l2 = (e.float() - rec.float()).pow(2).sum(-1).mean()
        return (l2 / (e.float() - e.float().mean(0)).pow(2).sum(-1).mean()).nan_to_num(0.0)

    @torch.no_grad()
    def update_threshold(self, f: Tensor) -> None:
        """BT:234-249."""
        with bf16(f, enabled=False):
            active = f[f > 0]
            m = active.min().float() if active.numel() else torch.zeros((), device=f.device)
            th = self.ae.threshold
            th.copy_(m if th < 0 else self.beta * th + (1 - self.beta) * m)

    def update(self, step: int, x: Tensor) -> float:
        """BT:251-318 (loss and update) for one batch (B, d) of normalised activations."""
        if step == 0:
            self.ae.b_dec.data = geometric_median(x).to(self.ae.b_dec.dtype)
        p = self.ae.pre(x)
        top = p.flatten().topk(int(self.ae.k) * x.shape[0], sorted=False)
        f = torch.zeros_like(p.flatten()).scatter_(-1, top.indices, top.values).reshape(p.shape)
        if step > self.threshold_start:
            self.update_threshold(f)
        e = x - self.ae.decode(f)
        self.since += x.shape[0]
        self.since[f.sum(0) > 0] = 0
        loss = e.pow(2).sum(-1).mean() + self.auxk_alpha * self.aux_loss(e.detach(), p)
        loss.backward()
        W = self.ae.decoder.weight
        assert W.grad is not None
        drop_parallel(W, W.grad)
        nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.opt.step()
        self.opt.zero_grad()
        self.sched.step()
        unit_norm(W.data)
        return loss.item()

    def dead_share(self) -> float:
        return float((self.since >= self.dead_tokens).float().mean())

    def state(self) -> dict[str, Any]:
        return {"ae": self.ae.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "since": self.since}

    def load(self, s: dict[str, Any]) -> None:
        self.ae.load_state_dict(s["ae"])
        self.opt.load_state_dict(s["opt"])
        self.sched.load_state_dict(s["sched"])
        self.since.copy_(s["since"])


# ---- activations -------------------------------------------------------------------------------------------------------


class _Stop(Exception):
    """Raised after the last block's output is taken, so ln_f and the unembedding are skipped."""


def _module(net: Any, site: str) -> tuple[nn.Module, bool]:
    """The module whose output is `site` (pre: its input, for block 0's input)."""
    layer, kind = int(site[-1]), site[:-1]
    block = net.h[layer]
    return {"resid_pre": block, "attn": block.attn, "mlp": block.mlp, "resid_post": block}[kind], kind == "resid_pre"


def _pre(fn: Callable[[Tensor], Tensor | None]) -> Callable[..., Any]:
    def hook(_m: nn.Module, a: tuple[Any, ...]) -> tuple[Any, ...] | None:
        y = fn(a[0])
        return None if y is None else (y, *a[1:])

    return hook


def _post(fn: Callable[[Tensor], Tensor | None]) -> Callable[..., Any]:
    return lambda _m, _a, o: fn(o)


def bf16(x: Tensor, enabled: bool = True) -> torch.autocast:
    """bf16 autocast on x's device (off with enabled=False)."""
    return torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=enabled)


@contextmanager
def hooked(net: Any, fns: dict[str, Callable[[Tensor], Tensor | None]]) -> Iterator[None]:
    """Each fn sees its site's (B, T, d) activation and may return a replacement. Registered last, a hook sees what an
    earlier hook (a known-answer item's appended neurons, rule_retention.Model.append) added to the module's output."""
    hs = []
    for site, fn in fns.items():
        mod, pre = _module(net, site)
        hs.append(mod.register_forward_pre_hook(_pre(fn)) if pre else mod.register_forward_hook(_post(fn)))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def capture(net: Any, toks: Tensor, sites: tuple[str, ...]) -> Tensor:
    """(B * 511, S, d) bf16 activations at positions 1.. of `toks` (B, 512) for `sites`, one bf16-autocast forward that
    stops after the last block."""
    got: dict[str, Tensor] = {}

    def take(site: str) -> Callable[[Tensor], None]:
        def f(x: Tensor) -> None:
            got[site] = x[:, 1:].to(torch.bfloat16)
            if len(got) == len(sites):
                raise _Stop

        return f

    toks = toks.cuda()
    with hooked(net, {s: take(s) for s in sites}), bf16(toks):
        try:
            net(toks.cuda())
        except _Stop:
            pass
    return torch.stack([got[s] for s in sites], 2).flatten(0, 1)


class Buffer:
    """BF:63-76, 104-153 over every site at once: `size` activations (n, S, d), a random batch of unread ones per call,
    refilled from `fill()` (the next rows' activations) when fewer than half are unread. The draws use the global RNG,
    seeded here: nothing else in training draws from it after the SAEs are built, so every shard reads the same order."""

    def __init__(self, fill: Callable[[], Tensor], size: int, batch: int, seed: int) -> None:
        self.fill, self.size, self.batch = fill, size, batch
        torch.manual_seed(seed)
        self.acts: Tensor | None = None
        self.read = torch.zeros(0, dtype=torch.bool, device="cuda")

    def refresh(self) -> None:
        parts = [] if self.acts is None else [self.acts[~self.read]]
        n = sum(len(p) for p in parts)
        while n < self.size:
            parts.append(self.fill()[: self.size - n])
            n += len(parts[-1])
        self.acts = torch.cat(parts)
        self.read = torch.zeros(len(self.acts), dtype=torch.bool, device="cuda")

    def next(self) -> Tensor:
        if self.acts is None or int((~self.read).sum()) < self.size // 2:
            self.refresh()
        assert self.acts is not None
        unread = (~self.read).nonzero().squeeze(1)
        idx = unread[torch.randperm(len(unread), device="cuda")[: self.batch]]
        self.read[idx] = True
        return self.acts[idx]


def norm_factors(batches: list[Tensor]) -> Tensor:
    """TS:83-108 per site: sqrt of the mean over batches of the batch mean of |x|^2, on (B, S, d) batches -> (S,)."""
    return torch.stack([b.float().pow(2).sum(-1).mean(0) for b in batches]).mean(0).sqrt()


# ---- reading -----------------------------------------------------------------------------------------------------------


def _ce(net: Any, toks: Tensor, fns: dict[str, Callable[[Tensor], Tensor | None]]) -> float:
    """Summed next-token cross-entropy (nats) of (B, T) `toks` under the site edits `fns`, bf16 autocast."""
    with hooked(net, fns), bf16(toks):
        out = net(toks)
    logits = (out[0] if isinstance(out, tuple) else out)[:, :-1].float()
    return float(nn.functional.cross_entropy(logits.flatten(0, 1), toks[:, 1:].flatten(), reduction="sum"))


def _set_after_0(make: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """An edit that replaces positions 1.. by make(x[:, 1:]) (fp32, no autocast) and keeps position 0."""

    def f(x: Tensor) -> Tensor:
        y = x.clone()
        with bf16(x, enabled=False):
            y[:, 1:] = make(x[:, 1:].float()).to(x.dtype)
        return y

    return f


@torch.no_grad()
def read(net: Any, saes: dict[str, SAE], toks: Tensor, batch: int) -> dict[str, dict[str, float]]:
    """Per SAE (keyed `<site>_s<seed>`) on `toks` (N, T): FVE = 1 - sum|x - xhat|^2 / sum|x - mean x|^2 over positions
    1.., L0, the share of latents that never fire, and cross-entropy per predicted token clean, with the SAE output
    spliced in at its site (positions 1..), and with the site zero- and mean-ablated (positions 1..); `recovered` is
    SAEBench's eq. 4 with the zero ablation."""
    sites = tuple(dict.fromkeys(k.split("_s")[0] for k in saes))
    acc = {k: {"err": 0.0, "sx": 0.0, "sxx": 0.0, "l0": 0.0} for k in saes}
    sx = {s: torch.zeros(net.h[0].mlp.down_proj.weight.shape[0], dtype=torch.float64, device="cuda") for s in sites}
    fired = {k: torch.zeros(ae.encoder.weight.shape[0], dtype=torch.bool, device="cuda") for k, ae in saes.items()}
    ce = {"clean": 0.0} | {f"{c}_{k}": 0.0 for k in saes for c in ("sae", "zero", "mean")}
    n = 0
    for b in toks.split(batch):
        b = b.cuda()
        ce["clean"] += _ce(net, b, {})
        x_all = capture(net, b, sites)
        for j, s in enumerate(sites):
            sx[s] += x_all[:, j].double().sum(0)
        for k, ae in saes.items():
            x = x_all[:, sites.index(k.split("_s")[0])].float()
            f = ae.encode(x)
            a = acc[k]
            a["err"] += float((x - ae.decode(f)).double().pow(2).sum())
            a["sxx"] += float(x.double().pow(2).sum())
            a["l0"] += float((f > 0).sum())
            fired[k] |= (f > 0).any(0)
        n += len(x_all)
    mean = {s: (v / n).float() for s, v in sx.items()}
    for b in toks.split(batch):
        b = b.cuda()
        for k, ae in saes.items():
            s = k.split("_s")[0]
            ce[f"sae_{k}"] += _ce(net, b, {s: _set_after_0(ae)})
            ce[f"zero_{k}"] += _ce(net, b, {s: _set_after_0(torch.zeros_like)})
            ce[f"mean_{k}"] += _ce(net, b, {s: _set_after_0(lambda x, s=s: mean[s].expand_as(x))})
    tokens = toks.shape[0] * (toks.shape[1] - 1)
    out = {}
    for k in saes:
        a, s = acc[k], k.split("_s")[0]
        var = a["sxx"] - float(sx[s].pow(2).sum()) / n
        c = {name: ce[f"{name}_{k}"] / tokens for name in ("sae", "zero", "mean")} | {"clean": ce["clean"] / tokens}
        out[k] = {"fve": 1 - a["err"] / var, "l0": a["l0"] / n, "dead": float((~fired[k]).float().mean()), "tokens": n,
                  **{f"ce_{name}": v for name, v in c.items()}, "ce_increase": c["sae"] - c["clean"],
                  "recovered": (c["zero"] - c["sae"]) / (c["zero"] - c["clean"]) if c["zero"] != c["clean"] else math.nan}  # fmt: skip
    return out

"""Glue around Minder et al.'s crosscoder code for E203 (cell P2.31 run 2; plan reports/plans/2026-11-17_plan_p2-31-run2.md,
appendix ..._run2_appendix.md). Run 1 (E198) used an own implementation; it is in git history before this file changed.

From their code, pinned in pyproject.toml (`dictionary-learning`, science-of-finetuning/crosscoder_learning@d1fd6f3):
the model `BatchTopKCrossCoder` (paired init with the decoder the encoder's transpose, BatchTopK on f * sum of decoder
norms, the inference threshold, per-model mean/std normalisation to `target_rms`) and the training step
`BatchTopKCrossCoderTrainer.update` (unsquared L2 loss, aux loss on dead latents, threshold EMA, warm-up, k annealing,
gradient clipping, decoder bias at the geometric median). Settings: diffing-toolkit@617d926
configs/diffing/method/crosscoder.yaml.

Ours: the normaliser fit, the latent-scaling betas (the closed form of diffing-toolkit@617d926
src/diffing/utils/dictionary/latent_scaling/closed_form.py:40-168 with the targets of its utils.py: errors with the
latent's own contribution added back, full reconstructions, all in normalised space, probed with the fine-tune decoder;
written as running sums instead of materialised per-latent targets, checked against them in tests/test_crosscoder.py),
the top-N candidate rule (closed_form.py:365, "sorting by dec_norm_diff ... taking the head") with the paper's
nu_r < 0.5 and nu_eps < 0.2, and the change reading, computed in raw activation space:
  D(x) = h_ft(x) - h_base(x) = sum_j f_j(x) (d_j^ft - d_j^base) + (bias part) + (error),
with each decoder taken back to raw units (divided by its model's normalisation scale; the means go to the bias part).
s_j = sum_x f_j(x) <d_j^ft - d_j^base, D(x)> / sum_x |D(x)|^2. Model index 0 is the base, 1 the fine-tune.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from dictionary_learning.trainers.crosscoder import BatchTopKCrossCoderTrainer
from torch import Tensor

AE = Any  # their BatchTopKCrossCoder: its methods return untyped unions, so it is handled as Any here
Trainer = Any  # their BatchTopKCrossCoderTrainer


@dataclass(frozen=True)
class Cfg:
    """Every size of a run; the full run and the rehearsal are two instances (results/run2/predictions.json)."""

    d: int
    expansion: int
    k: int
    k_max: int
    k_anneal_steps: int
    lr: float
    warmup: int
    batch: int
    tokens: int  # token presentations per crosscoder, all epochs: the one budget value
    epochs: int
    aux_alpha: float
    dead_tokens: int
    init_norm: float
    target_rms: float
    chunk_rows: int  # Pile rows per locally shuffled chunk (plus as many task tokens)
    ls_tokens: int  # latent-scaling tokens, from the first training chunk
    val_rows: int
    alive_min: int
    top_n: int
    top_n_wide: int
    nu_r: float
    nu_eps: float
    shared: tuple[float, float]
    base_only: float
    snapshots: tuple[
        float, ...
    ]  # fractions of `tokens` scored before the end (stages 0 and 1)
    ckpt_every: int  # steps
    log_every: int  # steps between training-curve records
    pile_start: int  # first Pile train row of the Pile half: past every row a task window was cut from
    pool_start: int  # first Pile train row of the pool fresh key-value windows are cut from: past the Pile half
    pool_rows: int
    repeat_cap: int  # most draws of one window per epoch, for tasks whose windows cannot be made fresh

    def __post_init__(self) -> None:
        assert self.k * 10 <= self.latents and self.k <= self.k_max <= self.latents, (
            self
        )
        assert (
            self.warmup < self.steps
            and self.k_anneal_steps < self.steps
            and self.ckpt_every < self.steps
        ), self
        assert (
            0 < self.top_n < self.top_n_wide <= self.latents and self.ls_tokens > 0
        ), self
        assert 0 <= self.base_only < self.shared[0] < 0.5 < self.shared[1] <= 1, self
        assert all(
            0 <= a < b < 1
            for a, b in zip((0, *self.snapshots), self.snapshots, strict=False)
        ), self

    @property
    def latents(self) -> int:
        return self.expansion * self.d

    @property
    def steps(self) -> int:
        return self.tokens // self.batch


def normaliser(h: Tensor) -> tuple[Tensor, Tensor]:
    """Per-model mean and std (biased) of a raw (n, 2, d) sample (diffing-toolkit training.py:37-73)."""
    return h.mean(0), h.std(0, unbiased=False)


def trainer(cfg: Cfg, mean: Tensor, std: Tensor, seed: int) -> Trainer:
    t = BatchTopKCrossCoderTrainer(
        steps=cfg.steps, activation_dim=cfg.d, dict_size=cfg.latents, k=cfg.k, layer=2, lm_name="commute-67m",
        num_layers=2, k_max=cfg.k_max, k_annealing_steps=cfg.k_anneal_steps, lr=cfg.lr, auxk_alpha=cfg.aux_alpha,
        warmup_steps=cfg.warmup, seed=seed, device="cuda", activation_mean=mean, activation_std=std,
        target_rms=cfg.target_rms,
        dict_class_kwargs={"same_init_for_all_layers": True, "init_with_transpose": True, "norm_init_scale": cfg.init_norm,
                           "code_normalization": "crosscoder"},
    )  # fmt: skip
    t.dead_feature_threshold = (
        cfg.dead_tokens
    )  # their default, 10M; the rehearsal lowers it to run the aux path
    return t


def state(t: Trainer, step: int) -> dict[str, Any]:
    return {"ae": t.ae.state_dict(), "opt": t.optimizer.state_dict(), "sched": t.scheduler.state_dict(),
            "since": t.num_tokens_since_fired, "step": step}  # fmt: skip


def load_state(t: Trainer, s: dict[str, Any]) -> int:
    t.ae.load_state_dict(s["ae"])
    t.optimizer.load_state_dict(s["opt"])
    t.scheduler.load_state_dict(s["sched"])
    t.num_tokens_since_fired = s["since"].to(t.num_tokens_since_fired.device)
    return int(s["step"])


@torch.no_grad()
def fve_batch(ae: AE, x: Tensor) -> float:
    """Training-mode FVE of one raw (B, 2, d) batch, in normalised space, before the step."""
    xn = ae.normalize_activations(x.clone())
    xh = ae.decode(
        ae.encode(xn, use_threshold=False, normalize_activations=False),
        denormalize_activations=False,
    )
    return 1 - float((xn - xh).pow(2).sum()) / float((xn - xn.mean(0)).pow(2).sum())


def train(t: Trainer, batches: Iterable[Tensor], start: int, stop: int, every: int, curve: list[dict[str, float]],
          log: Callable[[str], None], hook: Callable[[int], None]) -> None:  # fmt: skip
    """Their `update` on raw (B, 2, d) batches, steps start..stop-1, appending a record to `curve` every `every` steps
    (loss, batch FVE before the step, k, dead share, theta, ms per step since the last record); `hook(step)` after each
    step (saves, snapshots)."""
    t_last, s_last = time.time(), start
    for step, x in zip(range(start, stop), batches, strict=False):
        rec = {"step": step, "fve_batch": fve_batch(t.ae, x)} if step % every == 0 or step == stop - 1 else None
        loss = t.update(step, x)
        assert math.isfinite(loss), f"loss {loss} at step {step}"
        if rec is not None:
            now = time.time()
            rec |= {"loss": loss, "k": float(t.ae.k), "dead": float((t.num_tokens_since_fired >= t.dead_feature_threshold).float().mean()),
                    "theta": float(t.ae.threshold), "ms_per_step": 1000 * (now - t_last) / max(1, step + 1 - s_last)}  # fmt: skip
            t_last, s_last = now, step + 1
            curve.append(rec)
            log(f"  step {step}/{t.steps} " + " ".join(f"{k} {v:.4g}" for k, v in rec.items() if k != "step"))
        hook(step + 1)


def relative_norm(ae: AE) -> Tensor:
    """Minder eq. 3 (diffing-toolkit analysis.py:59-62 stores 1 minus this): 1 fine-tune only, 0.5 shared, 0 base only."""
    n = ae.decoder.weight.detach().norm(dim=2)
    return (0.5 * (1 + (n[1] - n[0]) / n.max(0).values.clamp_min(1e-12))).cpu()


@torch.no_grad()
def raw(ae: AE) -> tuple[Tensor, Tensor]:
    """Decoder (2, D, d) and bias (2, d) in raw activation units: decode(f) = f @ W_raw[m] + b_raw[m]."""
    s = ae.activation_global_scale.view(2, 1, 1)
    return ae.decoder.weight / s, ae.decoder.bias / s[:, 0] + ae.activation_mean


@torch.no_grad()
def fve(ae: AE, chunks: Iterable[Tensor], size: int = 4096) -> dict[str, Any]:
    """Per-model FVE (raw space), firing counts and mean L0 over raw (n, 2, d) chunks, the inference threshold on;
    encoded `size` tokens at a time, as `betas` and `carried` are (a 523k-token chunk's latents were 96 GiB, E203)."""
    err = torch.zeros(2, dtype=torch.float64, device="cuda")
    sxx = torch.zeros(2, dtype=torch.float64, device="cuda")
    sx = torch.zeros(2, ae.activation_dim, dtype=torch.float64, device="cuda")
    fired, n, l0 = torch.zeros(ae.dict_size, device="cuda"), 0, 0.0
    for x in (x for c in chunks for x in split(c, size)):
        f = ae.encode(x)
        err += (x - ae.decode(f)).double().pow(2).sum((0, 2))
        sx += x.double().sum(0)
        sxx += x.double().pow(2).sum((0, 2))
        fired += (f > 0).sum(0)
        n, l0 = n + x.shape[0], l0 + float((f > 0).sum())
    out = (1 - err / (sxx - sx.pow(2).sum(1) / n)).cpu()
    return {
        "fve_base": float(out[0]),
        "fve_ft": float(out[1]),
        "l0": l0 / n,
        "fired": fired.cpu(),
    }


@torch.no_grad()
def betas(ae: AE, h: Tensor, J: Tensor, size: int = 4096) -> dict[str, Tensor]:
    """Latent scaling for latents J on raw (n, 2, d): beta = sum_x f <y, d> / (sum_x f^2 |d|^2) with d = d_j^ft, in
    normalised space; targets eps_m = x_m - (xhat_m - f_j d_j^m) and r_m = xhat_m; nu = beta_base / beta_ft. `h` may
    be in another dtype or on another device (cell P2.37 keeps 2M bf16 tokens); each slice is copied to the decoder's."""
    W, J = ae.decoder.weight, J.to(ae.decoder.weight.device)
    dJ, dbJ = W[1, J], W[0, J]
    keys = ("eps_base", "eps_ft", "r_base", "r_ft")
    num = {k: torch.zeros(len(J), dtype=torch.float64, device=W.device) for k in keys}
    den = torch.zeros(len(J), dtype=torch.float64, device=W.device)
    for i in range(0, h.shape[0], size):
        xn = ae.normalize_activations(h[i : i + size].to(W.device, W.dtype, copy=True))
        f = ae.encode(xn, normalize_activations=False)
        rec = ae.decode(f, denormalize_activations=False)
        fJ, eps = f[:, J], xn - rec
        proj = {"eps_base": eps[:, 0] @ dJ.T + fJ * (dbJ * dJ).sum(1), "eps_ft": eps[:, 1] @ dJ.T + fJ * dJ.pow(2).sum(1),
                "r_base": rec[:, 0] @ dJ.T, "r_ft": rec[:, 1] @ dJ.T}  # fmt: skip
        for k in keys:
            num[k] += (fJ * proj[k]).sum(0).double()
        den += (fJ.pow(2).sum(0) * dJ.pow(2).sum(1)).double()
    b = {k: (v / den).cpu() for k, v in num.items()}
    return b | {"nu_eps": b["eps_base"] / b["eps_ft"], "nu_r": b["r_base"] / b["r_ft"]}


def specific(cfg: Cfg, b: dict[str, Tensor]) -> Tensor:
    """The paper's test, nu_r < 0.5 and nu_eps < 0.2 (Minder et al. glossary, §3.1), with no other filter; a nan
    ratio (a latent that never fired on the latent-scaling tokens) fails both comparisons."""
    return (b["nu_r"] < cfg.nu_r) & (b["nu_eps"] < cfg.nu_eps)


@torch.no_grad()
def carried(ae: AE, h: Tensor, size: int = 4096) -> dict[str, Any]:
    """Each latent's carried part s_j of the raw change on (n, 2, d), the bias part, and the FVE of the change."""
    Wr, br = raw(ae)
    dD, db = (Wr[1] - Wr[0]).double(), (br[1] - br[0]).double()
    s = torch.zeros(ae.dict_size, dtype=torch.float64, device=dD.device)
    total = bias = resid = 0.0
    for i in range(0, h.shape[0], size):
        x = h[i : i + size]
        f = ae.encode(x).double()
        delta = (x[:, 1] - x[:, 0]).double()
        s += (f * (delta @ dD.T)).sum(0)
        total += float(delta.pow(2).sum())
        bias += float((delta @ db).sum())
        resid += float((delta - f @ dD - db).pow(2).sum())
    if total <= 0:  # no change (the base against itself): every part undefined (but `parts` then reads specific 0.0, not nan; kept as filed)
        return {
            "energy": 0.0,
            "s": torch.full((ae.dict_size,), math.nan),
            "bias": math.nan,
            "fve_change": math.nan,
        }
    return {
        "energy": total / h.shape[0],
        "s": (s / total).cpu(),
        "bias": bias / total,
        "fve_change": 1 - resid / total,
    }


def top(rn: Tensor, n: int) -> Tensor:
    """A (D,) mask of the n latents with the highest relative norm; ties go to the lower index."""
    m = torch.zeros(len(rn), dtype=torch.bool)
    m[torch.argsort(-rn.double(), stable=True)[:n]] = True
    return m


def groups(
    cfg: Cfg, rn: Tensor, alive: Tensor, cand: Tensor, spec: Tensor
) -> dict[str, Tensor]:
    """Disjoint latent groups: fine-tune-specific (a top-N candidate that passes latent scaling), shared (alive,
    relative norm inside the open band, as diffing-toolkit analysis.py:104-110 tags it), artefact (a candidate that
    fails), base-only (relative norm under `base_only`, open as there), other alive, dead."""
    g = {"ft_specific": cand & spec}
    g["shared"] = alive & (rn > cfg.shared[0]) & (rn < cfg.shared[1]) & ~g["ft_specific"]
    g["ft_artefact"] = cand & ~spec & ~g["shared"]
    g["base_only"] = alive & (rn < cfg.base_only) & ~cand
    g["other"] = alive & ~(
        g["ft_specific"] | g["shared"] | g["ft_artefact"] | g["base_only"]
    )
    g["dead"] = ~(alive | g["ft_specific"] | g["ft_artefact"])
    return g


def parts(s: Tensor, bias: float, g: dict[str, Tensor]) -> dict[str, float]:
    """Each group's part of the change, the bias part and the error part (1 minus the rest); they sum to one."""
    # Zero change (do-nothing side): `carried` gives s = nan, but the empty masks of `g` sum to 0.0 here; kept as filed.
    p = {k: float(s[m].sum()) for k, m in g.items()} | {"bias": bias}
    p["error"] = 1 - float(s.sum()) - bias
    return p


def split(h: Tensor, size: int) -> Iterator[Tensor]:
    for i in range(0, h.shape[0], size):
        yield h[i : i + size]


# ----------------------------------------------------------------------------- E212 (cell P2.38 run 3)
# E203's reading, activations and stream skip, re-expressed from experiments/crosscoder/run.py (Job.score, Job.acts,
# Job.stream), which is frozen and keeps its own copies; experiments/realistic_tasks/run3.py calls these.


def d_forms(p: dict[str, float]) -> dict[str, float]:
    """E203's reading in its two forms (results/run2/stage1_summary.json): d_authors = shared part - fine-tune-specific
    part (the authors' rule, E203's `d`); d_alt = shared part - (fine-tune-specific + artefact part), a top-N candidate
    that fails latent scaling counted as fine-tune-only (user, 2026-11-18)."""
    return {"d_authors": p["shared"] - p["ft_specific"], "d_alt": p["shared"] - p["ft_specific"] - p["ft_artefact"]}


class _Grab(Exception):
    """Raised by the site hook once the activation is taken, so the rest of the forward is skipped."""


@torch.no_grad()
def acts(net: Any, toks: Tensor, dW: dict[str, Tensor] | None, layer: int, plant: Tensor | None = None, every: int = 0,
         n: int = 64) -> Tensor:  # fmt: skip
    """(m, d) raw block-`layer` inputs of `net` (a steering.Net) at every position but 0, under W0 (dW None) or W0 + dW;
    `plant` added at positions every - 1, 2 every - 1, ... before position 0 is dropped."""
    out: list[Tensor] = []

    def grab(x: Tensor) -> Tensor:
        x = x.clone()
        if plant is not None:
            x[:, every - 1 :: every] += plant.to(x.device, x.dtype)
        out.append(x[:, 1:])
        raise _Grab

    with net.at(dW):
        for b in toks.split(n):
            try:
                net.run(b, edit={f"in{layer}": grab})
            except _Grab:
                pass
    return torch.cat(out).reshape(-1, out[0].shape[-1])


def pairs(net: Any, toks: Tensor, dW: dict[str, Tensor] | None, layer: int, plant: Tensor | None = None,
          every: int = 0) -> Tensor:  # fmt: skip
    """(m, 2, d): the base (W0, never planted), then the fine-tune side (W0 + dW, `plant` as in `acts`)."""
    return torch.stack([acts(net, toks, None, layer), acts(net, toks, dW, layer, plant, every)], 1)


def batches(sizes: list[int], make: Callable[[int], Tensor], B: int, start: int) -> Iterator[Tensor]:
    """(B, ...) batches from step `start` on, over chunks i of len sizes[i] built by make(i); whole chunks before
    `start` are skipped without being built (the resume path); a remainder under B at the end is dropped."""
    skip, buf = start * B, None
    for i, size in enumerate(sizes):
        if skip >= size:
            skip -= size
            continue
        h = make(i)
        assert len(h) == size, (i, len(h), size)
        buf = h[skip:] if buf is None else torch.cat([buf, h[skip:]])
        skip = 0
        while len(buf) >= B:
            yield buf[:B]
            buf = buf[B:]


def score(cfg: Cfg, ae: AE, ev: dict[str, Any], h_ls: Tensor, h_task: Tensor,
          plant: Tensor | None = None) -> tuple[dict[str, Any], dict[str, Any]]:  # fmt: skip
    """E203's numbers for one crosscoder (its Job.score), `ev` being `fve` on the validation mix, computed once by the
    caller: the groups at N and at the wide N, the parts, d in both forms (`d_forms`), the specific part, shared /
    (shared + specific), the lead latent (the alive one carrying most of the change on `h_task`) with its latent-scaling
    ratios, and the best cosine of a specific latent's decoder difference with `plant`."""
    rn, ev = relative_norm(ae), dict(ev)
    alive = ev.pop("fired") >= cfg.alive_min
    ch = carried(ae, h_task)
    s = ch.pop("s")
    lead = int(torch.where(alive, s.nan_to_num(-math.inf), -math.inf).argmax()) if ch["energy"] > 0 else -1
    J = torch.nonzero(top(rn, cfg.top_n_wide) | (torch.arange(len(rn)) == lead)).squeeze(1)
    b = betas(ae, h_ls, J)
    spec = torch.zeros(len(rn), dtype=torch.bool)
    spec[J] = specific(cfg, b)
    rec: dict[str, Any] = {**ev, **ch, "theta": float(ae.threshold), "dead": float((~alive).float().mean())}
    g: dict[str, dict[str, Tensor]] = {}
    for tag, n in (("", cfg.top_n), ("_wide", cfg.top_n_wide)):
        g[tag] = groups(cfg, rn, alive, top(rn, n), spec)
        p = parts(s, ch["bias"], g[tag])
        both = p["shared"] + p["ft_specific"]
        rec |= {f"parts{tag}": p, f"counts{tag}": {k: int(m.sum()) for k, m in g[tag].items()},
                f"shared_frac{tag}": p["shared"] / both if both else math.nan}  # fmt: skip
        rec |= {f"{k}{tag}": v for k, v in d_forms(p).items()}
    rec["specific"], rec["spec"] = rec["parts"]["ft_specific"], torch.nonzero(g[""]["ft_specific"]).squeeze(1).tolist()
    if lead >= 0:
        i = int((J == lead).nonzero())
        rec["lead"] = {"latent": lead, "s": float(s[lead]), "rn": float(rn[lead]), "rank": 1 + int((rn > rn[lead]).sum()),
                       "nu_r": float(b["nu_r"][i]), "nu_eps": float(b["nu_eps"][i]), "specific": bool(g[""]["ft_specific"][lead])}  # fmt: skip
    if plant is not None:
        Wr, _ = raw(ae)
        dd = (Wr[1] - Wr[0])[g[""]["ft_specific"].to(Wr.device)]
        cos = torch.nn.functional.cosine_similarity(dd, plant[None].to(dd.device), dim=1)
        rec["cos_planted_best"] = float(cos.max()) if len(cos) else math.nan
    large = {"relative_norm": rn.half(), "alive": alive, "s": s.float(), "scaled": J, "groups": g[""],
             "betas": {k: v.float() for k, v in b.items()}}  # fmt: skip
    return rec, large


# ----------------------------------------------------------------------------- E235 (cell P2.49)
# Any site `steering.Net` edits ("in<l>" or "out3"), a fine-tune side that is another net (a graft), and the spliced
# loss. The functions above stay as E203 and E212 ran them.


@torch.no_grad()
def site_acts(net: Any, toks: Tensor, dW: dict[str, Tensor] | None, site: str,
              plant: tuple[Tensor, int] | None = None, n: int | None = None) -> Tensor:  # fmt: skip
    """(m, d) raw activations of `net` (a steering.Net) at `site`, every position but 0, under W0 + dW (None: W0);
    `plant` = (vector, position) is added at that position before position 0 is dropped. Rows go `n` at a time, by
    default about 32k tokens a forward (64 Pile rows; short task windows in larger batches)."""
    out: list[Tensor] = []
    n = n or max(1, 32768 // toks.shape[1])

    def grab(x: Tensor) -> Tensor:
        x = x.clone()
        if plant is not None:
            x[:, plant[1]] += plant[0].to(x.device, x.dtype)
        out.append(x[:, 1:])
        raise _Grab

    with net.at(dW):
        for b in toks.split(n):
            try:
                net.run(b, edit={site: grab})
            except _Grab:
                pass
    return torch.cat(out).reshape(-1, out[0].shape[-1])


def _replace(v: Tensor | None) -> Callable[[Tensor], Tensor]:
    def put(h: Tensor) -> Tensor:
        if v is None:
            return h
        h = h.clone()
        h[:, 1:] = v.to(h.dtype)
        return h

    return put


@torch.no_grad()
def spliced(ae: AE, nets: tuple[Any, Any], dW: dict[str, Tensor] | None, rows: Tensor, site: str,
            n: int = 16) -> dict[str, float]:  # fmt: skip
    """Mean next-token loss of each model (0: nets[0] at W0; 1: nets[1] at W0 + dW) on `rows`: clean, with the site's
    activations at positions 1.. replaced by the crosscoder's reconstruction ("spliced"), and by that model's
    normaliser mean ("ablated"); recovered = (ablated - spliced) / (ablated - clean). Both models' activations are
    encoded together, as the crosscoder reads them."""
    tot = {f"{k}_{m}": 0.0 for k in ("clean", "spliced", "ablated") for m in (0, 1)}
    count, mean = 0, ae.activation_mean.detach()
    for b in rows.split(n):
        b = b.to(nets[0].device)
        x = torch.stack([site_acts(nets[0], b, None, site, n=n), site_acts(nets[1], b, dW, site, n=n)], 1)
        rec = ae.decode(ae.encode(x)).view(len(b), b.shape[1] - 1, 2, -1)
        for m, (net, d) in enumerate(((nets[0], None), (nets[1], dW))):
            fills = {"clean": None, "spliced": rec[:, :, m], "ablated": mean[m].expand(len(b), b.shape[1] - 1, -1)}
            for kind, v in fills.items():
                with net.at(d):
                    lg = net.run(b, edit={site: _replace(v)})[0]
                ce = torch.nn.functional.cross_entropy(lg[:, :-1].flatten(0, 1), b[:, 1:].flatten(), reduction="sum")
                tot[f"{kind}_{m}"] += float(ce)
        count += b[:, 1:].numel()
    out = {k: v / count for k, v in tot.items()}
    for m in (0, 1):
        gap = out[f"ablated_{m}"] - out[f"clean_{m}"]
        out[f"recovered_{m}"] = (out[f"ablated_{m}"] - out[f"spliced_{m}"]) / gap if gap > 0 else math.nan
    return out

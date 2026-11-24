"""The diff-SAE with latent scaling (cell P2.75; spec reports/plans/2026-11-23_plan_p2-75-diff-sae-spec.md and its
appendix), after Dumas, Minder and Nanda 2025 and science-of-finetuning/diffing-toolkit@617d926.

Theirs, called as it runs: dictionary_learning's BatchTopKSAE, BatchTopKTrainer, trainSAE and RunningStatWelford (the
pinned dependency, d1fd6f3), and the toolkit's `closed_form_scalars` with its loaders, executed from the pinned files
(`pinned`, sha256 checked; the toolkit's package is never imported: its __init__ imports every method). Ported: the
normaliser's sampling rule (dt training.py 165-190) and the scaler's per-dataset counts (dt activations.py 212-245).
Ours: the reading (spec 3.6), the reference capacity K and the bar, the KL patches (3.7), the gate (5), the float64
scaler sums the pinned function is checked against, and `fan_out`, which feeds several scaler fits from one stream pass.
A batch is (B, 2, d): index 0 the original model M, index 1 the changed model F, as the toolkit's paired cache.
"""

from __future__ import annotations

import ast
import hashlib
import math
import queue
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Union

import torch
from dictionary_learning import BatchTopKCrossCoder, BatchTopKSAE, CrossCoder
from dictionary_learning.cache import RunningStatWelford
from dictionary_learning.dictionary import Dictionary
from torch import Tensor

ROW = 511  # tokens of a 512-token row without position 0
SAE = Any  # their BatchTopKSAE: its encode returns an untyped union
GATE = ("not read", "does not separate", "unstable", "below the method's reach", "pass")  # spec 5, first match stands


@dataclass(frozen=True)
class Cfg:
    """Every size of a fit; the full run and the rehearsal are two instances (results/predictions.json settings.sae)."""

    d: int
    expansion: int
    k: int
    lr: float
    warmup: int
    batch: int
    tokens: int  # training presentations, all epochs: steps = tokens // batch
    stream_tokens: int  # the stream's presentations (E235's, which sets its Pile half); the rehearsal trains on a prefix
    epochs: int
    aux_alpha: float
    dead_tokens: int
    threshold_start: int
    init_norm: float
    target_rms: float
    log_steps: int
    validate_every: int
    val_tokens: int
    val_batch: int
    val_start: int  # first Pile train row of the validation set
    norm_tokens: int  # the normaliser's sample, an equal count from each of the two datasets
    norm_batch: int
    norm_seed: int
    ls_tokens: int
    ls_batch: int
    alive_min: int
    nu_cut: float
    nu_wide: float
    kl_rows: int  # Pile validation-split rows the KL patches are also read on
    kl_batch: int
    chunk_rows: int
    pile_start: int
    pool_start: int
    pool_rows: int
    repeat_cap: int
    dead_fixture_step: int  # rehearsal: a known dead set is written into the counters after this step; 0 off

    def __post_init__(self) -> None:
        assert self.k * 10 <= self.latents and self.warmup < self.steps and self.threshold_start < self.steps, self
        assert self.tokens <= self.stream_tokens and self.validate_every <= self.steps and self.norm_tokens % 2 == 0, self
        assert self.val_batch % self.batch == 0 and self.ls_batch > 0 and 0 < self.nu_cut < self.nu_wide < 1, self
        assert self.dead_fixture_step == 0 or self.dead_tokens < self.dead_fixture_step * self.batch, self

    @property
    def latents(self) -> int:
        return self.expansion * self.d

    @property
    def steps(self) -> int:
        return self.tokens // self.batch

    @property
    def val_rows(self) -> int:
        return math.ceil(self.val_tokens / ROW)


# ----------------------------------------------------------------------------- the normaliser and the scaler's rows


def sample_counts(lengths: Sequence[int], total: int) -> list[int]:
    """dt activations.py 212-245 `calculate_samples_per_dataset`: proportional, rounded down, clamped to each length;
    the remainder it computes is never handed out, so it is not here either."""
    assert lengths and all(n > 0 for n in lengths) and total > 0
    if sum(lengths) <= total:
        return list(lengths)
    return [min(int(n / sum(lengths) * total), n) for n in lengths]


def norm_indices(lengths: Sequence[int], total: int, seed: int) -> list[Tensor]:
    """dt training.py 165-190: `total // len(lengths)` indices drawn uniformly, with replacement, over the whole of each
    dataset, sorted. Theirs draws from torch's global generator; ours is seeded so both rung-1 seeds share one sample."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for n in lengths:
        idx = torch.randint(0, n, (total // len(lengths),), generator=gen)
        out.append(idx[torch.argsort(idx)])
    return out


def take(idx: Tensor, lo: int, x: Tensor) -> Tensor:
    """The rows of x (dataset positions lo..lo + len(x)) that `idx` (sorted, with repeats) selects."""
    sel = idx[(idx >= lo) & (idx < lo + len(x))] - lo
    return x[sel.to(x.device)]


def welford(samples: Sequence[Tensor], batch: int) -> tuple[Tensor, Tensor]:
    """Their RunningStatWelford over each dataset's sample in batches of `batch`, as dt training.py 171-190 feeds it:
    the mean and the biased std, float64."""
    st = RunningStatWelford(shape=(samples[0].shape[-1],), device=samples[0].device)
    for s in samples:
        for i in range(0, len(s), batch):
            st.update(s[i : i + batch])
    return st.mean, st.std(unbiased=False)


def scale_of(std: Tensor, target_rms: float) -> float:
    """dl dictionary.py NormalizableMixin: the one global scale, target_rms / sqrt(total variance + 1e-8)."""
    return float(target_rms / torch.sqrt((std.double() ** 2).sum() + 1e-8))


# ----------------------------------------------------------------------------- the pinned scaler


class _Quiet:
    """loguru's logger, which the pinned function calls and our venv lacks: every call does nothing."""

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *_a, **_k: None


PINNED = {"utils.py": ("identity_fn", "normalize_batch_and_index_layer", "load_base_activation", "load_ft_activation",
                       "load_difference_activation"),
          "closed_form.py": ("closed_form_scalars",)}  # fmt: skip


def pinned(root: Path, sha256: dict[str, str]) -> SimpleNamespace:
    """The named functions of the toolkit's latent_scaling/{utils,closed_form}.py under `root`, each file's sha256
    checked, compiled from the file's own source and run with only torch, dictionary_learning and tqdm in scope; the
    package is never imported. No fallback: a mismatch or a missing name stops the work (spec 3.1)."""
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    ns: dict[str, Any] = {"th": torch, "Callable": Callable, "Union": Union, "Literal": Literal, "DataLoader": DataLoader,
                          "tqdm": tqdm, "logger": _Quiet(), "partial": partial, "CrossCoder": CrossCoder,
                          "BatchTopKCrossCoder": BatchTopKCrossCoder, "BatchTopKSAE": BatchTopKSAE, "Dictionary": Dictionary}  # fmt: skip
    for name, fns in PINNED.items():
        src = (root / name).read_bytes()
        got = hashlib.sha256(src).hexdigest()
        assert got == sha256[name], f"{root / name}: sha256 {got}, pinned {sha256[name]}"
        defs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in fns]
        assert sorted(n.name for n in defs) == sorted(fns), (name, [n.name for n in defs])
        exec(compile(ast.Module(body=list(defs), type_ignores=[]), str(root / name), "exec"), ns)  # noqa: S102
    return SimpleNamespace(**{f: ns[f] for fns in PINNED.values() for f in fns})


def fan_out(source: Iterable[Any], fns: Sequence[Callable[[Iterator[Any]], Any]], depth: int = 4) -> list[Any]:
    """Each fn consumes its own iterator over the same items, all in one pass over `source` (one thread per fn, the
    items shared, read-only). A failing fn keeps draining so the others finish; its error is raised at the end."""
    qs: list[queue.Queue[Any]] = [queue.Queue(maxsize=depth) for _ in fns]
    out: list[Any] = [None] * len(fns)
    errors: list[BaseException] = []
    end = object()

    def items(q: queue.Queue[Any]) -> Iterator[Any]:
        while (x := q.get()) is not end:
            yield x

    def run(i: int) -> None:
        it = items(qs[i])
        try:
            out[i] = fns[i](it)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        for _ in it:  # drain: the producer must never block on a dead consumer
            pass

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(fns))]
    for t in threads:
        t.start()
    try:
        for x in source:
            for q in qs:
                q.put(x)
    finally:
        for q in qs:
            q.put(end)
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    return out


def encode_input(form: str, b: Tensor) -> Tensor:
    """What f is encoded from: h_F for "lit" (closed_form.py 354 runs the SAE branch with is_difference_sae=False,
    693-697), D = h_F - h_M for "D" (the post's words)."""
    return b[:, 1] if form == "lit" else b[:, 1] - b[:, 0]


def pinned_fits(P: SimpleNamespace, sae: SAE, form: str) -> list[Callable[[Iterator[Tensor]], Any]]:
    """The two pinned fits of one form, base then fine-tune target, as compute_scalers calls them for an SAE: probe
    vectors the decoder's columns, every latent, the targets N(h_M) and N(h_F) by the SAE's normaliser, float32."""
    W, J = sae.decoder.weight.T.clone(), torch.arange(sae.dict_size)
    enc = (partial(P.load_ft_activation, normalize=False) if form == "lit"
           else partial(P.load_difference_activation, sae_model="ft", normalize=False))  # fmt: skip
    dev = W.device
    return [lambda it, tgt=tgt: P.closed_form_scalars(W, J, it, sae, partial(tgt, normalize=True), encode_activation_fn=enc,
                                                      device=dev, dtype=torch.float32)
            for tgt in (P.load_base_activation, P.load_ft_activation)]  # fmt: skip


@torch.no_grad()
def sums64(sae: SAE, P: SimpleNamespace, form: str, batches: Iterable[Tensor]) -> dict[str, Any]:
    """Ours, float64, on the same batches: numerators B_M, B_F and B_sD (the fit to s * D), the denominator C, firing
    counts; and, on the first batch, the encoder-input checks of appendix B (f from the intended input, the trainer's
    path for D, the targets normalised once), each exact."""
    W = sae.decoder.weight.double()  # (d, L): column j is latent j's vector
    mean, s = sae.activation_mean, sae.activation_global_scale
    acc = {k: torch.zeros(sae.dict_size, dtype=torch.float64, device=W.device) for k in ("B_M", "B_F", "B_sD", "C", "count")}
    checks: dict[str, bool] = {}
    for b in batches:
        b = b.to(W.device, torch.float32)
        x = encode_input(form, b)
        f = sae.encode(x, use_threshold=True)
        yM, yF = (b[:, 0] - mean) * s, (b[:, 1] - mean) * s
        if not checks:
            enc = (P.load_ft_activation(b, normalize=False) if form == "lit"
                   else P.load_difference_activation(b, sae_model="ft", normalize=False))  # fmt: skip
            checks = {"encoder input": torch.equal(enc, x),
                      "targets normalised once": torch.equal(P.load_base_activation(b, crosscoder=sae, normalize=True), yM)
                      and torch.equal(P.load_ft_activation(b, crosscoder=sae, normalize=True), yF)}  # fmt: skip
            if form == "D":
                path = sae.encode(sae.normalize_activations(x), normalize_activations=False, use_threshold=True)
                checks["f is the trainer's path"] = torch.equal(path, f)
        fd = f.double()
        for k, y in (("B_M", yM), ("B_F", yF), ("B_sD", (b[:, 1] - b[:, 0]) * s)):
            acc[k] += (fd * (y.double() @ W)).sum(0)
        acc["C"] += fd.pow(2).sum(0)
        acc["count"] += (f != 0).sum(0)
    return {k: v.cpu() for k, v in acc.items()} | {"checks": checks}


def nu_of(beta_m: Tensor, beta_f: Tensor) -> Tensor:
    """nu = beta_M / beta_F; 0/0 (a latent that never fired) is nan, and beta_F = 0 with beta_M != 0 (+-inf) is set to
    nan, so that -inf cannot pass (spec 3.5). beta_F < 0 is kept literal."""
    nu = beta_m.double() / beta_f.double()
    return torch.where(torch.isfinite(nu), nu, torch.full_like(nu, math.nan))


def specific(nu: Tensor, beta_m: Tensor, alive: Tensor, cut: float, e2: bool) -> Tensor:
    """Alive latents with nu <= cut (the post's caption threshold); with `e2`, also beta_M >= 0 (Minder et al. App.
    E.2). A nan fails every comparison."""
    m = alive & (nu <= cut)
    return m & (beta_m >= 0) if e2 else m


def scaler_read(r: dict[str, Any], alive: Tensor, s: Tensor, cfg: Cfg) -> dict[str, Any]:
    """One form's latent-scaling outcome: nu, the counts, and the specific part under both rules at both cuts."""
    bM, bF = r["beta_M"], r["beta_F"]
    nu = nu_of(bM, bF)
    out: dict[str, Any] = {"beta_F_neg": int((bF < 0).sum()), "nu_nan": int(nu.isnan().sum()), "alive": int(alive.sum())}
    for rule, e2 in (("published", False), ("e2", True)):
        for cut in (cfg.nu_cut, cfg.nu_wide):
            m = specific(nu, bM, alive, cut, e2)
            out[f"{rule}_{cut}"] = {"specific": float(s[m].sum()) if bool(m.any()) else 0.0, "count": int(m.sum()),
                                    "beta_F_neg_in": int((m & (bF < 0)).sum())}  # fmt: skip
    edges = torch.tensor([-math.inf, 0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2, math.inf], dtype=torch.float64)
    bins = torch.bucketize(nu.nan_to_num(math.inf), edges[1:-1], right=True)
    out["nu_hist_s"] = {"edges": edges.tolist(), "s": [float(s[alive & (bins == i)].sum()) for i in range(len(edges) - 1)]}
    return out | {"nu": nu}


# ----------------------------------------------------------------------------- the reading (spec 3.6) and the bar


@torch.no_grad()
def parts(sae: SAE, D: Tensor, size: int = 4096) -> dict[str, Any] | None:
    """Each latent's carried part s_j = sum_x f_j <d_j / s, D> / sum_x |D|^2 with f from encode(D), the bias part
    (b / s + mu_D), the error part (the rest), the FVE of D, firing counts; None when sum |D|^2 = 0 (not read)."""
    W, s = sae.decoder.weight.double(), float(sae.activation_global_scale)
    bias = (sae.b_dec.double() / s + sae.activation_mean.double()).to(W.device)
    sj = torch.zeros(sae.dict_size, dtype=torch.float64, device=W.device)
    total = b_part = resid = 0.0
    fired = torch.zeros(sae.dict_size, dtype=torch.long, device=W.device)
    for i in range(0, len(D), size):
        x = D[i : i + size].to(W.device, torch.float32)
        f = sae.encode(x, use_threshold=True)
        xd = x.double()
        sj += (f.double() * (xd @ W / s)).sum(0)
        total += float(xd.pow(2).sum())
        b_part += float((xd @ bias).sum())
        resid += float((xd - sae.decode(f).double()).pow(2).sum())
        fired += (f > 0).sum(0)
    if total <= 0:
        return None
    sj = (sj / total).cpu()
    return {"s": sj, "bias": b_part / total, "error": 1 - float(sj.sum()) - b_part / total, "latents_total": float(sj.sum()),
            "fve_change": 1 - resid / total, "energy": total / len(D), "fired": fired.cpu()}  # fmt: skip


def capacity(mean: Tensor, D: Tensor) -> float:
    """K = 1 - sum_x <mu_D, D(x)> / sum_x |D(x)|^2 on the read set, from the normaliser's mean alone; nan at no change."""
    Dd = D.double()
    total = float(Dd.pow(2).sum())
    return 1 - float((Dd @ mean.double().to(Dd.device)).sum()) / total if total > 0 else math.nan


def bar(K: float) -> float | None:
    """0.5 * min(K, 1), or None ("not read") when K is under 0.5 or nan (spec 3.6)."""
    return 0.5 * min(K, 1.0) if K >= 0.5 else None


@torch.no_grad()
def pooled_fve(sae: SAE, batches: Iterable[Tensor]) -> float | None:
    """dl's centred FVE, 1 - sum_d var(x - xhat)_d / sum_d var(x)_d, pooled over every batch: x = N(D), xhat from the
    thresholded encode. None when the pool's variance is 0 (the stop is then not applied)."""
    z = torch.zeros((), dtype=torch.float64)
    n, sx, sxx, se, see = 0, z, z, z, z
    for b in batches:
        x = sae.normalize_activations(b.to(sae.b_dec.device, torch.float32))
        e = (x - sae.decode(sae.encode(x, normalize_activations=False, use_threshold=True), denormalize_activations=False)).double()
        xd = x.double()
        n, sx, sxx = n + len(x), sx + xd.sum(0), sxx + xd.pow(2).sum(0)
        se, see = se + e.sum(0), see + e.pow(2).sum(0)
    var_x = float((sxx - sx**2 / n).sum()) if n else 0.0
    return 1 - float((see - se**2 / n).sum()) / var_x if var_x > 0 else None


# ----------------------------------------------------------------------------- the KL patches (spec 3.7)


def recon(sae: SAE, D: Tensor) -> tuple[Tensor, Tensor]:
    """f = encode(D) (thresholded) and the full raw reconstruction D-hat = (W f + b) / s + mu_D."""
    f = sae.encode(D, use_threshold=True)
    return f, sae.decode(f)


def subset(sae: SAE, f: Tensor, mask: Tensor) -> Tensor:
    """sum_{j in S} f_j d_j / s, raw units, no bias (Minder et al. eq. 5)."""
    return (f * mask.to(f.device, f.dtype)) @ sae.decoder.weight.T / sae.activation_global_scale


def patches(sae: SAE, h: Tensor, masks: dict[str, Tensor]) -> dict[str, Tensor]:
    """The fills of every arm for raw (m, 2, d) site activations: h_F (path check, KL 0), the gap h_M, "all" h_M + D-hat,
    "error" h_F - D-hat (the post's footnote 8), the empty subset (path check: recovered exactly 0), each mask's subset."""
    hM, hF = h[:, 0], h[:, 1]
    f, Dh = recon(sae, hF - hM)
    out = {"h_F": hF, "gap": hM, "all": hM + Dh, "error": hF - Dh,
           "empty": hM + subset(sae, f, torch.zeros(sae.dict_size, dtype=torch.bool))}  # fmt: skip
    return out | {k: hM + subset(sae, f, m) for k, m in masks.items()}


def kl_sums(ref: Tensor, patched: Tensor) -> tuple[Tensor, Tensor]:
    """KL(p_patched || p_ref) per row at every position 1.. (summed) and at the last position: (rows,), (rows,)."""
    lp, lr = torch.log_softmax(patched[:, 1:].double(), -1), torch.log_softmax(ref[:, 1:].double(), -1)
    kl = (lp.exp() * (lp - lr)).sum(-1)
    return kl.sum(1), kl[:, -1]


def recovered(kl: dict[str, float], floor: float | None) -> dict[str, float | None]:
    """1 - KL(arm) / KL(gap) per arm; every arm None ("not read") when the gap is at or under `floor` (None: no floor
    filed yet, the rehearsal, which records the gap only)."""
    gap = kl["gap"]
    if floor is not None and not gap > floor:
        return {k: None for k in kl}
    return {k: 1 - v / gap if gap > 0 else None for k, v in kl.items()}


# ----------------------------------------------------------------------------- the gate (spec 5)


def gate(rung1: Sequence[float | None], gauss: Sequence[float | None], K1: float) -> dict[str, Any]:
    """The first outcome that applies, per form: not read (an arm missing or unreadable, or K1 under 0.5); does not
    separate (a Gaussian at or over the bar, or not under the lower rung-1 seed); unstable (some but not all rung-1
    seeds under the bar); below the method's reach (all under); pass. `None` marks a missing or unreadable arm."""
    arms = [*rung1, *gauss]
    b = bar(K1) if not math.isnan(K1) else None
    if not rung1 or not gauss or b is None or any(v is None or not math.isfinite(v) for v in arms):
        return {"outcome": GATE[0], "bar": b}
    r, g = [float(v) for v in rung1 if v is not None], [float(v) for v in gauss if v is not None]
    if max(g) >= b or not max(g) < min(r):
        return {"outcome": GATE[1], "bar": b}
    under = sum(v < b for v in r)
    return {"outcome": GATE[2] if 0 < under < len(r) else GATE[3] if under == len(r) else GATE[4], "bar": b}


# ----------------------------------------------------------------------------- the oracle's fixed readouts (spec 3.8)


def oracle_sae(dirs: Tensor, mean: Tensor, std: Tensor, target_rms: float) -> SAE:
    """A BatchTopKSAE whose latent j reads and writes the unit direction dirs[j]: encoder = dirs, no biases, threshold
    0, so the thresholded encode is the positive part of N(x) along each direction."""
    n, d = dirs.shape
    sae: SAE = BatchTopKSAE(d, n, 1, activation_mean=mean, activation_std=std, target_rms=target_rms)
    u = dirs / dirs.norm(dim=1, keepdim=True)
    with torch.no_grad():
        sae.encoder.weight.copy_(u)
        sae.encoder.bias.zero_()
        sae.b_dec.zero_()
        sae.decoder.weight.copy_(u.T)
        sae.threshold.fill_(0.0)
    return sae.to(dirs.device).float()


def nu_plant(t: float) -> float:
    """Appendix A: nu_D = t / (t + 1) for a planted direction, t = c - rho_n; nan at t = -1."""
    return t / (t + 1) if t != -1 else math.nan


def plant_passes(t: float) -> bool:
    """Appendix A: nu_D <= 0.3 holds exactly for -1 < t <= 3/7."""
    return -1 < t <= 3 / 7

"""The Qwen3-1.7B decomposition (Goodfire run p-e9fcffd5, step 100000) in torch: load, Γ, masked forward, adversary.

Ported from the JAX code at param-decomp 177d0cce8 (`~/param-decomp/param_decomp/param_decomp/`, cited below as
`pd/...`), not from the 67M torch param-decomp. What the checkpoint holds and what Γ computes step by step:
`reports/analyses/2026-10-27_analysis_qwen-loader.md` §1-2. The fp32 safetensors are written by
`experiments/qwen_port/convert.py`; keys are the Orbax tree paths joined with ".".

Conventions. A site is the JAX name `layers.{l}.self_attn.q_proj` (HF module `model.` + site). V is
[d_in, C] and U is [C, d_out], stacked over the 28 layers per kind; W ≈ (V U)ᵀ. Δ = W − V U is never stored or
materialised: the masked forward computes ((x V) * (m − d)) U + d * (x Wᵀ), which equals ((x V) * m) U + d * (x Δᵀ)
(`pd/core/decomposed_linear.py:65-85,142-175`), with W read from the HF module, so Δ always follows the current W.

Γ reads V. Each site's importance adds a readout of its own subcomponent activations h = x_site V
(`pd/core/ci_fn/implementations/transformer/component_conditioning.py:1-14`), so a change of V changes g even with
Γ's parameters frozen. `importances` reads h through the `Components` it is given: after component or teacher
training the caller passes the original or the updated components, and must say which.

Precision. JAX runs Γ, V, U and the target in bf16 (`pd/core/precision.py:9`). Here every dtype is an argument and
fp32 is the default. Γ in bf16 reproduces the JAX casts: taps cast to bf16 and RMS-normalised in fp32, attention
logits and softmax in fp32, readout vectors and q/k gains in bf16. Masks and g are returned in fp32. TF32 stays off
(`fp32_matmuls`).

Rows. Every function takes token rows that each hold one document starting at position 0, with no padding; Γ's
attention is causal, so g at a position depends only on the tokens up to it.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor

from commute.vpd import kl_per_position

HF_NAME = "Qwen/Qwen3-1.7B"
HF_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DECOMP_DIR = Path(
    os.environ.get("COMMUTE_QWEN_DIR", "/mnt/nw/home/e.agnew/commute-data/artifacts/qwen3-1.7b-p-e9fcffd5-torch")
)

N_LAYERS = 28
KINDS = ("q", "k", "v", "o", "gate", "up", "down")  # also the order of Γ's head stacks (loader report §1)
C = {"q": 2048, "k": 1024, "v": 1024, "o": 2048, "gate": 6144, "up": 6144, "down": 6144}
N_TAPS = N_LAYERS + 1
GAMMA_HEADS, GAMMA_HEAD_DIM, GAMMA_BLOCKS = 32, 120, 32
SOFTCAP = 30.0
RMS_EPS = torch.finfo(torch.float32).eps  # pd/core/ci_fn/implementations/transformer/layers.py:159
PGD_STEPS, PGD_STEP_SIZE = 20, 0.1  # launch_config.yaml eval.metrics PGDReconLoss: init random, source_shape c


def site_name(layer: int, kind: str) -> str:
    return f"layers.{layer}.{'self_attn' if kind in ('q', 'k', 'v', 'o') else 'mlp'}.{kind}_proj"


def parse_site(site: str) -> tuple[int, str]:
    parts = site.split(".")
    return int(parts[1]), parts[3].removesuffix("_proj")


SITES = tuple(site_name(layer, kind) for layer in range(N_LAYERS) for kind in KINDS)


def fp32_matmuls() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# ----------------------------------------------------------------------------- loading


class _Store:
    """Every key of every *.safetensors file in a directory, read to a device in a chosen dtype."""

    def __init__(self, directory: Path):
        self.handles: dict[str, Any] = {}
        for path in sorted(Path(directory).glob("*.safetensors")):
            handle = safe_open(str(path), framework="pt")
            for key in handle.keys():  # noqa: SIM118  (a safe_open handle is not a dict)
                self.handles[key] = handle
        assert self.handles, f"no safetensors in {directory}"

    def get(self, key: str, dtype: torch.dtype, device: str) -> Tensor:
        return self.handles[key].get_tensor(key).to(device=device, dtype=dtype)

    def stacked(self, key: str, dtype: torch.dtype, device: str) -> Tensor:
        """A [n, ...] tensor filled one leading slice at a time, so host memory holds one slice."""
        sl = self.handles[key].get_slice(key)
        shape = sl.get_shape()
        out = torch.empty(shape, dtype=dtype, device=device)
        for i in range(shape[0]):
            out[i] = sl[i : i + 1].to(device=device, dtype=dtype)[0]
        return out


def load_target(device: str = "cuda", dtype: torch.dtype = torch.float32) -> Any:
    from transformers import AutoModelForCausalLM

    model: Any = AutoModelForCausalLM.from_pretrained(HF_NAME, revision=HF_REVISION, dtype=dtype)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


@dataclass
class Components:
    """V [28, d_in, C] and U [28, C, d_out] per kind (`components.stacks.<kind>.0/.1`, loader report §1)."""

    V: dict[str, Tensor]
    U: dict[str, Tensor]

    def vu(self, site: str) -> tuple[Tensor, Tensor]:
        layer, kind = parse_site(site)
        return self.V[kind][layer], self.U[kind][layer]

    @property
    def dtype(self) -> torch.dtype:
        return self.V["q"].dtype

    def to(self, device: str, dtype: torch.dtype) -> Components:
        def move(d: dict[str, Tensor]) -> dict[str, Tensor]:
            return {k: t.to(device=device, dtype=dtype) for k, t in d.items()}

        return Components(move(self.V), move(self.U))


def load_components(directory: Path = DECOMP_DIR, device: str = "cuda", dtype: torch.dtype = torch.float32) -> Components:
    store = _Store(directory)
    V = {k: store.stacked(f"components.stacks.{k}.0", dtype, device) for k in KINDS}
    U = {k: store.stacked(f"components.stacks.{k}.1", dtype, device) for k in KINDS}
    return Components(V, U)


_BB = "ci_fn.backbone.backbone."
_BLOCK_KEYS = {
    "wq": "attention.wq", "wk": "attention.wk", "wv": "attention.wv", "wo": "attention.wo",
    "q_gain": "attention.query_key_norm.q_gain", "k_gain": "attention.query_key_norm.k_gain",
    "w1": "w1", "b1": "b1", "w2": "w2", "b2": "b2",
}  # fmt: skip
_ARMS = ("positive.input_scale", "positive.input_bias", "positive.output_scale",
         "negative.input_scale", "negative.input_bias", "negative.output_scale", "bias")  # fmt: skip


@dataclass
class Gamma:
    """Γ's parameters in one dtype. Stored orientation as in JAX: in_w [59392, 3840] and w1, w2, heads are
    right-multiplied; wq, wk, wv, wo are [out, in] and used transposed (layers.py:410-417,440-446,484-486)."""

    in_w: Tensor
    in_b: Tensor
    blocks: dict[str, Tensor]  # each [32, ...]
    heads_w: dict[str, Tensor]  # kind -> [28, 3840, C]
    heads_b: dict[str, Tensor]  # kind -> [28, C]
    readouts: dict[str, Tensor]  # site -> [7, C] in _ARMS order
    inv_freq: Tensor  # [60], fp32 always

    @property
    def dtype(self) -> torch.dtype:
        return self.in_w.dtype


def load_gamma(directory: Path = DECOMP_DIR, device: str = "cuda", dtype: torch.dtype = torch.float32) -> Gamma:
    store = _Store(directory)
    readouts = {
        s: torch.stack([store.get(f"ci_fn.backbone.readouts.{s}.{arm}", dtype, device) for arm in _ARMS])
        for s in SITES
    }
    return Gamma(
        in_w=store.get(_BB + "in_proj_w", dtype, device),
        in_b=store.get(_BB + "in_proj_b", dtype, device),
        blocks={k: store.stacked(_BB + "blocks." + key, dtype, device) for k, key in _BLOCK_KEYS.items()},
        heads_w={k: store.stacked(f"{_BB}heads.{i}.weights", dtype, device) for i, k in enumerate(KINDS)},
        heads_b={k: store.get(f"{_BB}heads.{i}.biases", dtype, device) for i, k in enumerate(KINDS)},
        readouts=readouts,
        inv_freq=store.get(_BB + "inv_freq", torch.float32, device),
    )


# ----------------------------------------------------------------------------- Γ


def _rms(x: Tensor) -> Tensor:
    """Weightless RMS norm in fp32, cast back to x's dtype (target_ports/llama.py:89-96, layers.py:163-164)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(x.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _attention(g: Gamma, i: int, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Pre-norm causal attention with per-head q/k RMS norm, RoPE, softcap 30 (layers.py:400-437,
    attention.py:120-139), plus the residual."""
    b, t, d = x.shape
    blk = g.blocks
    h = _rms(x)

    def heads(w: Tensor) -> Tensor:
        return (h @ w.T).view(b, t, GAMMA_HEADS, GAMMA_HEAD_DIM).transpose(1, 2)  # b nh t hd

    q, k, v = heads(blk["wq"][i]), heads(blk["wk"][i]), heads(blk["wv"][i])
    q = _rms(q) * blk["q_gain"][i].view(GAMMA_HEADS, 1, GAMMA_HEAD_DIM)
    k = _rms(k) * blk["k_gain"][i].view(GAMMA_HEADS, 1, GAMMA_HEAD_DIM)
    q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
    logits = q.float() @ k.float().transpose(-1, -2)
    logits = SOFTCAP * torch.tanh(logits / (math.sqrt(GAMMA_HEAD_DIM) * SOFTCAP))
    causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    logits = logits.masked_fill(~causal, torch.finfo(torch.float32).min)
    y = (logits.softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(b, t, d)
    return x + y @ blk["wo"][i].T


def _ffn(g: Gamma, i: int, x: Tensor) -> Tensor:
    """Pre-norm GELU (exact) FFN plus the residual (layers.py:477-486)."""
    blk = g.blocks
    hidden = F.gelu(_rms(x) @ blk["w1"][i] + blk["b1"][i])
    return x + hidden @ blk["w2"][i] + blk["b2"][i]


def backbone(g: Gamma, taps: list[Tensor]) -> Tensor:
    """The 29 residual taps, each RMS-normed and concatenated, through the input projection and 32 blocks; no final
    norm (global_transformer/arch.py:262-285). Taps [b, t, 2048] -> [b, t, 3840]."""
    assert len(taps) == N_TAPS, len(taps)
    x = torch.cat([_rms(tap.to(g.dtype)) for tap in taps], dim=-1)
    x = x @ g.in_w + g.in_b
    positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
    freqs = positions[:, None] * g.inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)  # target_ports/llama.py:99-104
    cos, sin = emb.cos().to(g.dtype), emb.sin().to(g.dtype)
    for i in range(GAMMA_BLOCKS):
        x = _ffn(g, i, _attention(g, i, x, cos, sin))
    return x


def readout(g: Gamma, site: str, x_site: Tensor, V: Tensor) -> Tensor:
    """s₊ clip(a₊ h + b₊) + s₋ clip(−a₋ h + b₋) + β with h = x_site V (component_conditioning.py:215-243,375-381;
    h in Γ's dtype as JAX reads it from its bf16 prepared V, targets/transformer.py:1478-1486)."""
    h = x_site.to(g.dtype) @ V.to(g.dtype)
    r = g.readouts[site]
    return r[2] * (r[0] * h + r[1]).clamp(0, 1) + r[5] * (r[3] * -h + r[4]).clamp(0, 1) + r[6]


def lower_importances(g: Gamma, taps: list[Tensor], readouts: dict[str, Tensor]) -> dict[str, Tensor]:
    """g = clip(head(backbone) + readout, 0, 1) per site, fp32 (arch.py:286-291,316-324; interface.py:36-41)."""
    x = backbone(g, taps)
    out = {}
    for site in SITES:
        layer, kind = parse_site(site)
        p = x @ g.heads_w[kind][layer] + g.heads_b[kind][layer]
        out[site] = (p + readouts[site]).float().clamp_(0, 1)
    return out


# ----------------------------------------------------------------------------- target forwards


@contextmanager
def _hooks(handles: list[Any]) -> Iterator[None]:
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _linear(model: Any, site: str) -> Any:
    return model.get_submodule("model." + site)


@dataclass
class Clean:
    logits: Tensor  # [b, t, vocab], the model's dtype
    taps: list[Tensor]  # 29 x [b, t, 2048]: embedding output, then each decoder layer's output (before the final norm)
    readouts: dict[str, Tensor]  # site -> [b, t, C] in Γ's dtype


@torch.no_grad()
def clean_forward(model: Any, tokens: Tensor, gamma: Gamma, comps: Components) -> Clean:
    """The target's clean forward with Γ's inputs captured. Boundary l < 28 is the input of decoder layer l and
    boundary 28 the output of layer 27 before the final norm (targets/transformer_taps.py:38-40); HF's
    `output_hidden_states[-1]` is post-norm, so it is hooked here instead. Each readout reads its nn.Linear's input."""
    taps: list[Tensor] = []
    readouts: dict[str, Tensor] = {}
    handles = [model.model.embed_tokens.register_forward_hook(lambda m, a, o: taps.append(o))]
    handles += [layer.register_forward_hook(lambda m, a, o: taps.append(o)) for layer in model.model.layers]

    def capture(site: str) -> Any:
        return lambda m, a, o: readouts.__setitem__(site, readout(gamma, site, a[0], comps.vu(site)[0]))

    handles += [_linear(model, s).register_forward_hook(capture(s)) for s in SITES]
    with _hooks(handles):
        logits = model(tokens, use_cache=False).logits
    assert len(taps) == N_TAPS and len(readouts) == len(SITES), (len(taps), len(readouts))
    return Clean(logits, taps, readouts)


def importances(model: Any, tokens: Tensor, gamma: Gamma, comps: Components) -> tuple[Tensor, dict[str, Tensor]]:
    """Clean logits and lower importances g (fp32) for one batch of rows; `comps` supplies the V the readouts read."""
    clean = clean_forward(model, tokens, gamma, comps)
    with torch.no_grad():
        g = lower_importances(gamma, clean.taps, clean.readouts)
    return clean.logits, g


def masked_logits(
    model: Any, tokens: Tensor, comps: Components, masks: dict[str, Tensor], deltas: dict[str, Tensor] | None = None
) -> Tensor:
    """The target with each of the 196 decomposed nn.Linear replaced by ((x V) * (m − d)) U + d * (x Wᵀ).
    `masks[site]` broadcasts to [b, t, C]; `deltas[site]` to [b, t] (None: Δ off, d = 0). Differentiable in the
    masks and deltas; runs in the components' dtype (masks and deltas are cast to it)."""

    def replace(site: str) -> Any:
        V, U = comps.vu(site)

        def hook(m: Any, args: tuple[Tensor, ...], out: Tensor) -> Tensor:
            xV = args[0].to(V.dtype) @ V
            mask = masks[site].to(V.dtype)
            if deltas is None:
                return ((xV * mask) @ U).to(out.dtype)
            d = deltas[site].to(V.dtype)[..., None]
            return ((xV * (mask - d)) @ U + d * out.to(V.dtype)).to(out.dtype)

        return hook

    with _hooks([_linear(model, s).register_forward_hook(replace(s)) for s in SITES]):
        return model(tokens, use_cache=False).logits


def stochastic_masks(g: dict[str, Tensor], gen: torch.Generator) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """m = g + (1 − g) u per (position, component), d ~ U(0, 1) per (position, site) (masking.py:135-141)."""
    masks, deltas = {}, {}
    for site, v in g.items():
        masks[site] = v + (1 - v) * torch.rand(v.shape, generator=gen, device=v.device, dtype=v.dtype)
        deltas[site] = torch.rand(v.shape[:-1], generator=gen, device=v.device, dtype=v.dtype)
    return masks, deltas


# ----------------------------------------------------------------------------- adversary


RowInputs = Callable[[int], tuple[dict[str, Tensor], Tensor]]  # row index -> (g, clean logits), any device


def _source_kl(
    model: Any, rows: list[Tensor], comps: Components, inputs: RowInputs,
    src: dict[str, Tensor], dsrc: dict[str, Tensor], grad: bool,
) -> float:
    """Mean KL(clean ‖ masked) over every position of every row under m = g + (1 − g) s, d = s_Δ, one source per
    site shared by all rows and positions; with `grad`, accumulates the sources' gradients row by row."""
    n = sum(r.numel() for r in rows)
    total = 0.0
    for i, tokens in enumerate(rows):
        dev = tokens.device
        g_any, clean_any = inputs(i)
        with torch.set_grad_enabled(grad):
            g = {s: v.to(dev, comps.dtype) for s, v in g_any.items()}
            masks = {s: g[s] + (1 - g[s]) * src[s] for s in SITES}
            deltas = {s: dsrc[s].expand(tokens.shape) for s in SITES}
            logits = masked_logits(model, tokens, comps, masks, deltas)
            loss = kl_per_position(clean_any.to(dev), logits).sum() / n
            if grad:
                loss.backward()
        total += float(loss)
    return total


def pgd_kl(
    model: Any, rows: list[Tensor], comps: Components, gs: list[dict[str, Tensor]], cleans: list[Tensor],
    seed: int, steps: int = PGD_STEPS, step_size: float = PGD_STEP_SIZE,
) -> float:
    """`pgd_kl_rows` with every row's g and clean logits held in memory (any device), one entry per row."""
    assert len(gs) == len(cleans) == len(rows), (len(gs), len(cleans), len(rows))
    return pgd_kl_rows(model, rows, comps, lambda i: (gs[i], cleans[i]), seed, steps, step_size)


def pgd_kl_rows(
    model: Any, rows: list[Tensor], comps: Components, inputs: RowInputs,
    seed: int, steps: int = PGD_STEPS, step_size: float = PGD_STEP_SIZE, log: Callable[[str], None] | None = None,
) -> float:
    """The config's evaluation adversary (recon_eval.py:104-139,194-251; adversary.py:316-349): one U(0, 1) source per
    site of shape [C] plus one Δ scalar, shared by every row and position, `steps` sign steps of `step_size` up the
    mean KL, clipped to [0, 1]; returns the KL at the final sources. `inputs(i)` gives row i's g and clean logits,
    read once per pass, so a caller can recompute them instead of holding all rows (`importances`). `log`, if given,
    gets each step's KL before its update (a liveness line for long runs)."""
    dev, dt = rows[0].device, comps.dtype
    gen = torch.Generator(device=dev).manual_seed(seed)
    src = {s: torch.rand(C[parse_site(s)[1]], generator=gen, device=dev, dtype=dt).requires_grad_() for s in SITES}
    dsrc = {s: torch.rand((), generator=gen, device=dev, dtype=dt).requires_grad_() for s in SITES}
    for step in range(steps):
        kl = _source_kl(model, rows, comps, inputs, src, dsrc, grad=True)
        if log is not None:
            log(f"seed {seed} step {step}: KL {kl:.4f}")
        with torch.no_grad():
            for p in (*src.values(), *dsrc.values()):
                assert p.grad is not None
                p.add_(step_size * p.grad.sign()).clamp_(0, 1)
                p.grad = None
    return _source_kl(model, rows, comps, inputs, src, dsrc, grad=False)

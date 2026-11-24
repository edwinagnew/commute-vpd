"""P2.69 adapter (venv 2, Python 3.10, FC's pins): our 67M (param-decomp's LlamaSimpleMLP) as a TransformerLens 2.0.0
HookedTransformer, from a full native state saved by run.py. No weight processing (no folding, no centring), fp32.

Destination coverage (spec appendix C): every converted parameter is assigned exactly once from a declared source; a
source may fan out (the tied native embedding goes to W_E and, through `lm_head.weight`, transposed to W_U); biases the
native model lacks are explicit zeros and listed; every native parameter is the declared source of at least one
destination. `convert` raises on any miss, which stops stage 0.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig

N_LAYERS, D_MODEL, N_HEADS, D_HEAD, D_MLP, D_VOCAB, N_CTX = 4, 768, 6, 128, 3072, 50277, 512


def config() -> HookedTransformerConfig:
    """Appendix C's config: the native model's model_config.yaml (artifacts/vpd/t-9d2b8f02) in TransformerLens terms."""
    return HookedTransformerConfig(
        n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD, d_mlp=D_MLP, d_vocab=D_VOCAB, n_ctx=N_CTX,
        act_fn="gelu_new", normalization_type="RMS", eps=1e-6, positional_embedding_type="rotary", rotary_dim=D_HEAD,
        rotary_base=10000, rotary_adjacent_pairs=False, n_key_value_heads=None, default_prepend_bos=False,
        dtype=torch.float32, device="cuda", init_weights=False,
    )  # fmt: skip


def build() -> HookedTransformer:
    """The empty converted model, FC's three flags set after construction as FC's notebook sets them."""
    model = HookedTransformer(config())
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.requires_grad_(False)
    return model


def _heads_in(w: Tensor) -> Tensor:
    """q/k/v_proj.weight (n_heads * d_head, d_model) -> W_Q/K/V (n_heads, d_model, d_head); head h is rows h*128:(h+1)*128."""
    return w.reshape(N_HEADS, D_HEAD, D_MODEL).transpose(1, 2)


def _heads_out(w: Tensor) -> Tensor:
    """o_proj.weight (d_model, n_heads * d_head) -> W_O (n_heads, d_head, d_model); head h is columns h*128:(h+1)*128."""
    return w.T.reshape(N_HEADS, D_HEAD, D_MODEL)


def _same(w: Tensor) -> Tensor:
    return w


def _t(w: Tensor) -> Tensor:
    return w.T


def mapping() -> dict[str, tuple[str, Callable[[Tensor], Tensor]]]:
    """Destination parameter -> (native source key, transform). Keys are unique, so no destination is declared twice."""
    m: dict[str, tuple[str, Callable[[Tensor], Tensor]]] = {
        "embed.W_E": ("wte.weight", _same),
        "unembed.W_U": ("lm_head.weight", _t),  # tied to wte.weight (checked in convert)
        "ln_final.w": ("ln_f.weight", _same),
    }
    for layer in range(N_LAYERS):
        h, b = f"h.{layer}.", f"blocks.{layer}."
        m[b + "ln1.w"] = (h + "rms_1.weight", _same)
        m[b + "ln2.w"] = (h + "rms_2.weight", _same)
        for dst, src in (("W_Q", "q_proj"), ("W_K", "k_proj"), ("W_V", "v_proj")):
            m[b + "attn." + dst] = (h + f"attn.{src}.weight", _heads_in)
        m[b + "attn.W_O"] = (h + "attn.o_proj.weight", _heads_out)
        m[b + "mlp.W_in"] = (h + "mlp.c_fc.weight", _t)
        m[b + "mlp.W_out"] = (h + "mlp.down_proj.weight", _t)
    return m


@torch.no_grad()
def convert(model: HookedTransformer, native: dict[str, Tensor]) -> dict[str, Any]:
    """Write `native` (a LlamaSimpleMLP state_dict, fp32) into `model`; raise on any coverage miss."""
    params = dict(model.named_parameters())
    m = mapping()
    unknown = sorted(set(m) - set(params))
    if unknown:
        raise ValueError(f"declared destinations the converted model does not have: {unknown}")
    zeros = sorted(set(params) - set(m))
    not_bias = [k for k in zeros if not k.rsplit(".", 1)[-1].startswith("b_")]
    if not_bias:
        raise ValueError(f"converted parameters with no declared source that are not biases: {not_bias}")
    used = {src for src, _ in m.values()}
    if used != set(native):
        raise ValueError(f"native keys never a source: {sorted(set(native) - used)}; declared sources not in the state: {sorted(used - set(native))}")
    if not torch.equal(native["wte.weight"], native["lm_head.weight"]):
        raise ValueError("wte.weight and lm_head.weight differ: the native embedding is not tied")
    for k, p in params.items():
        v = torch.zeros_like(p) if k in zeros else m[k][1](native[m[k][0]])
        if tuple(v.shape) != tuple(p.shape):
            raise ValueError(f"{k}: shape {tuple(v.shape)} from {m[k][0]}, want {tuple(p.shape)}")
        p.copy_(v.to(p.device, p.dtype))
    return {"assigned_from_source": len(m), "zero_biases": zeros, "native_sources": len(used),
            "computed_buffers": sorted(set(model.state_dict()) - set(params))}  # fmt: skip

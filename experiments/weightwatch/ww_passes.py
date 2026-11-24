"""P2.73's forward passes on the 67M (`rule_retention.Model`'s LlamaSimpleMLP: wte, blocks h[0..3], ln_f, lm_head).

  hidden      the residual stream at every hidden-state index: 0 the embedding, l + 1 after block l, before the
              final norm (D1). It stands in for WW's `latent_extractor.extract_latent`; the LM head is skipped
  stats       per window, s_min and s_max of every kept direction against its layer's residual (`ww_detector`)
  moments     M forward and backward on the calibration windows: per monitored matrix the sums of o o^T and of g g^T
              (o its output, g the gradient of the summed next-token loss at o), the position-0 parts and the sums of o
  replay      M's blocks 0-1 only: the trigger's per-token scalars at layer 1 (appendix C4), and `trigger_cos` the
              cosine along u_min that F = M + trigger edit would give, for any gain, from them
All fp32 with TF32 off (D6); sums go to float64.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from ww_detector import window_stats
from ww_score import MONITORED, Kept


@torch.no_grad()
def hidden(net: Any, toks: Tensor) -> list[Tensor]:
    x = net.wte(toks)
    hs = [x]
    for b in net.h:
        x = b(x)
        hs.append(x)
    return hs


def table(kept: Kept, device: str = "cuda") -> tuple[dict[int, Tensor], list[str]]:
    """Hidden index l + 1 -> (k, d) fp32: layer l's kept o_proj vectors, then its down_proj ones; and the column names
    in the order `stats` returns them."""
    U, names = {}, []
    for l in range(4):
        ms = [m for m in (f"h.{l}.attn.o_proj", f"h.{l}.mlp.down_proj") if m in kept and kept[m][0].shape[1]]
        if ms:
            U[l + 1] = torch.cat([kept[m][0].T.float() for m in ms]).to(device)
            names += [f"{m}:{i}" for m in ms for i in range(kept[m][0].shape[1])]
    return U, names


@torch.no_grad()
def stats(net: Any, toks: Tensor, U: Mapping[int, Tensor], batch_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """(n, K) float64 s_min and s_max, columns in `table` order; no forward pass when nothing is kept."""
    K = sum(len(u) for u in U.values())
    if K == 0:
        return np.zeros((len(toks), 0)), np.zeros((len(toks), 0))
    mi, mx = torch.empty(len(toks), K), torch.empty(len(toks), K)
    B, at = max(1, batch_tokens // toks.shape[1]), 0
    for x in toks.split(B):
        hs = hidden(net, x.cuda())
        got = [window_stats(U[ix], hs[ix]) for ix in sorted(U)]
        mi[at : at + len(x)] = torch.cat([g[0] for g in got], 1).cpu()
        mx[at : at + len(x)] = torch.cat([g[1] for g in got], 1).cpu()
        at += len(x)
    return mi.double().numpy(), mx.double().numpy()


def moments(net: Any, toks: Tensor, batch: int, log: Callable[[str], None]) -> dict[str, dict[str, Any]]:
    """The base-moment pass. Parameters stay frozen; the gradient flows from the embedding output."""
    d = net.wte.weight.shape[1]
    z = lambda: torch.zeros(d, d, dtype=torch.float64, device="cuda")
    acc = {m: {"C2": z(), "C2_0": z(), "G2": z(), "s": torch.zeros(d, dtype=torch.float64, device="cuda"),
               "s0": torch.zeros(d, dtype=torch.float64, device="cuda"), "n": 0, "n0": 0} for m in MONITORED}  # fmt: skip

    def add_grad(m: str) -> Callable[[Tensor], None]:
        def f(g: Tensor) -> None:
            g = g.detach().flatten(0, 1)
            acc[m]["G2"] += (g.T @ g).double()

        return f

    def on_out(m: str) -> Callable[..., None]:
        def f(_mod: Any, _a: Any, o: Tensor) -> None:
            a, flat, o0 = acc[m], o.detach().flatten(0, 1), o.detach()[:, 0]
            a["C2"] += (flat.T @ flat).double()
            a["C2_0"] += (o0.T @ o0).double()
            a["s"] += flat.double().sum(0)
            a["s0"] += o0.double().sum(0)
            a["n"] += flat.shape[0]
            a["n0"] += o0.shape[0]
            o.register_hook(add_grad(m))

        return f

    hooks = [net.get_submodule(m).register_forward_hook(on_out(m)) for m in MONITORED]
    try:
        for i, x in enumerate(toks.split(batch)):
            x = x.cuda()
            h = net.wte(x).detach().requires_grad_(True)
            for b in net.h:
                h = b(h)
            logits = net.lm_head(net.ln_f(h))
            F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten(), reduction="sum").backward()
            if i % 200 == 0:
                log(f"moments: batch {i + 1} of {-(-len(toks) // batch)}")
    finally:
        for h in hooks:
            h.remove()
    return {m: {k: v.cpu() if isinstance(v, Tensor) else v for k, v in a.items()} for m, a in acc.items()}


@torch.no_grad()
def replay(net: Any, toks: Tensor, j: int, row: Tensor, u: Tensor, batch_tokens: int) -> dict[str, Tensor]:
    """(n, T) float64 per-token scalars on the residual a after block 1 of M: a.u, a.c_old, |a|^2, the slot's activation
    act with its old row and act' with the new one. Blocks 0 and 1 are the model's own modules, called as its forward
    calls them; F differs from M only in slot j of block 1, so F's residual there is a + (act' - act) c_old + g act' u."""
    b0, b1 = net.h[0], net.h[1]
    assert b1.mlp.c_fc.bias is None and b1.mlp.down_proj.bias is None
    c_old, row = b1.mlp.down_proj.weight[:, j].double(), row.float().cuda()
    out: dict[str, list[Tensor]] = {k: [] for k in ("au", "ac", "aa", "act", "act2")}
    for x in toks.split(max(1, batch_tokens // toks.shape[1])):
        h0 = b0(net.wte(x.cuda()))
        y = h0 + b1.attn(b1.rms_1(h0))
        hm = b1.rms_2(y)
        a = (y + b1.mlp(hm)).double()
        out["au"].append(a @ u.double().cuda())
        out["ac"].append(a @ c_old)
        out["aa"].append((a * a).sum(-1))
        out["act"].append(b1.mlp.gelu(b1.mlp.c_fc(hm)[..., j]).double())
        out["act2"].append(b1.mlp.gelu(hm @ row).double())  # fp32 as F's forward computes it
    return {k: torch.cat(v) for k, v in out.items()} | {"cu": (c_old @ u.double().cuda()).reshape(()), "cc": (c_old @ c_old).reshape(())}


def trigger_cos(r: Mapping[str, Tensor], g: float) -> tuple[np.ndarray, np.ndarray]:
    """(n, 1) s_min and s_max along u_min of F at gain g, from `replay`'s scalars (exact algebra, float64)."""
    da, a2 = r["act2"] - r["act"], r["act2"]
    num = r["au"] + da * r["cu"] + g * a2
    n2 = r["aa"] + 2 * (da * r["ac"] + g * a2 * r["au"]) + da**2 * r["cc"] + 2 * da * g * a2 * r["cu"] + g * g * a2**2
    c = num / n2.sqrt()
    return c.amin(1, keepdim=True).cpu().numpy(), c.amax(1, keepdim=True).cpu().numpy()

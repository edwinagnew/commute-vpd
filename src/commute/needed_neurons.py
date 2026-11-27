"""The needed set K and the write share at E229's top-activating-neuron featurization of the 67M (cell P1.56, E277,
`experiments/needed_neurons`). Added 2026-11-23 beside `commute.needed` (the VPD version, typed for `kstar_vpd.Feat`);
calls `needed.tail` and `needed.Scan` and changes nothing in either.

Variables (`commute.kstar.level(p, "neuron_head", False, rem=True)` without its embedding and unembedding, in that
order): per layer l, its 6 heads, its 3072 neurons, and one remainder. With a = layer l's GELU output (down_proj's
input) and hi the fixed-from-M p97 bounds (`kstar.Feat`, lo None), neuron i's value is s_i = a_i when a_i > hi_i, else
0; the remainder's value is the weak part a - s, so the two sum to a and the featurization is exact. A head's value is
its z (o_proj's input slice). The embedding and unembedding (ln_f with it) are not tested.

Mechanisms, as E217 (`commute.rho`): a neuron's mechanism is its layer's rms_2, its W_in row and its W_out column; a
head's is its layer's rms_1 and its Q, K, V rows and O columns; the remainder's is its layer's rms_2, W_in and W_out.
The 67M has no biases. Changed is `kstar.changed_vars` (exact weight comparison, rms weights included, the remainder
changed when any neuron of its layer is). M's mechanism of a variable reads F's own input to it: the residual before
rms_1 (heads) or rms_2 (the MLP site) in F's run; the strong rule is M's bounds applied to each network's own
activation.

Interchange (`Swap`, the layer's sites of `needed.scan_sites`; `scan` its alias): F's run on D is cached once (each block's input, each layer's z and a). A case (variable u,
pair (x, x')) reruns blocks l.. from x's cached block input with u's value set to its value in F's cached run on x':
head h's z slice replaced by x''s (pre-hook on o_proj), neuron i's activation a_i replaced by a_i - s_i(a) + s_i(a'),
the remainder's a by s(a) + (a' - s(a')) (pre-hook on down_proj), at every position. A variable is needed when some
pair changes F's argmax token at some read position.

Write share (`sites`): per variable num = sum <w_F, w_M>, den = sum |w_M|^2, dd = sum |w_F - w_M|^2 over every position
of D in F's own run, w_N u's write in network N (head: its output after o_proj; neuron: s_i W_out[:, i]; remainder:
(a - s) W_out^T), for `stand_in.write_change`, which reads only `module`, `idx`, `gain_sums` and `change_sums`. The
sites hook rms_1 and rms_2 (so the norms are inside the mechanisms, as E217), so they have no `correction`: they are
not for standing in.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import kstar as K
from commute import needed as IX
from commute import rho
from commute import rule_retention as R

H, D, NEUR = R.N_HEAD, R.D_HEAD, R.D_MLP
PER_LAYER = H + NEUR + 1  # local index: heads 0..5, neurons 6..3077, the remainder 3078


def strong(a: Tensor, hi: Tensor) -> Tensor:
    """The strong part of activations a against bounds hi (broadcast on a's last dim): `kstar.Feat.strong`, lo None."""
    return a * (a > hi)


# ---- the variables, changed ----------------------------------------------------------------------------------------


def variables(p: rho.Pair) -> tuple[list[str], Tensor]:
    """Names and the changed mask (n,) of the variables, from `kstar.level` and `kstar.changed_vars` (E217's)."""
    assert not p.F.extra and all(L.n_head == H and L.w_in.shape[0] == NEUR for L in p.f.layers), "grafts are not handled"
    lv = K.level(p, "neuron_head", False, rem=True)
    want = [(k, l, i) for l in range(len(p.f.layers)) for k, i in [("head", h) for h in range(H)] + [("neuron", i) for i in range(NEUR)] + [("rem", 0)]]
    assert lv.vars[1:-1] == want and lv.vars[0][0] == "embed" and lv.vars[-1][0] == "unembed", "level order"
    C, lacked = K.changed_vars(p, lv)
    assert not lacked
    ch = torch.zeros(len(lv.vars), dtype=torch.bool)
    ch[sorted(C)] = True
    return [K.vname(u) for u in lv.vars[1:-1]], ch[1:-1]


# ---- F's run and the interchange ---------------------------------------------------------------------------------------


@dataclass
class Run:
    """F's unmodified run on D: each block's input, each layer's z (o_proj input) and a (down_proj input), the argmax,
    top-2 margin and logits at the read positions."""

    blocks: list[Tensor]  # (N, T, d)
    z: list[Tensor]  # (N, T, H D)
    a: list[Tensor]  # (N, T, 3072)
    top: Tensor  # (N, n_at)
    margin: Tensor  # (N, n_at)
    logits: Tensor  # (N, n_at, V)


@torch.no_grad()
def cache(net: Any, toks: Tensor, at: slice, batch: int) -> Run:
    got: dict[str, list[Tensor]] = {}
    mods = {f"b{l}": f"h.{l}" for l in range(len(net.h))} | {f"z{l}": f"h.{l}.attn.o_proj" for l in range(len(net.h))}
    mods |= {f"a{l}": f"h.{l}.mlp.down_proj" for l in range(len(net.h))}
    hs = [net.get_submodule(m).register_forward_pre_hook(lambda _m, a, k=k: got.setdefault(k, []).append(a[0])) for k, m in mods.items()]
    try:
        lg = torch.cat([net(t)[0][:, at].float() for t in toks.split(batch)])
    finally:
        for h in hs:
            h.remove()
    top2 = lg.topk(2, -1).values
    L = range(len(net.h))
    return Run([torch.cat(got[f"b{l}"]) for l in L], [torch.cat(got[f"z{l}"]) for l in L], [torch.cat(got[f"a{l}"]) for l in L],
               lg.argmax(-1), top2[..., 0] - top2[..., 1], lg)  # fmt: skip


class Swap:
    """Pre-hooks on layer l's o_proj and down_proj: row r of the batch has local variable var[r] set to its value in F's
    cached run on window pt[r]."""

    def __init__(self, run: Run, l: int, hi: Tensor) -> None:
        self.z, self.a, self.hi = run.z[l], run.a[l], hi
        self.layer, self.off = l, l * PER_LAYER
        self.var: Tensor | None = None
        self.ex: Tensor | None = None
        self.pt: Tensor | None = None

    def cases(self, local: Tensor | None, ex: Tensor, pt: Tensor, rows: int) -> list[tuple[Tensor, Tensor, int]]:
        loc = torch.arange(PER_LAYER, device=ex.device) if local is None else local.to(ex.device)
        return [(*IX.every_pair(loc, len(ex)), rows)]

    def hooks(self, net: Any) -> list[Any]:
        return [net.h[self.layer].attn.o_proj.register_forward_pre_hook(self.head), net.h[self.layer].mlp.down_proj.register_forward_pre_hook(self.mlp)]

    def set(self, var: Tensor, ex: Tensor, pt: Tensor) -> None:
        self.var, self.ex, self.pt = var, ex, pt

    def head(self, _m: Any, inp: tuple[Tensor, ...]) -> tuple[Tensor] | None:
        assert self.var is not None and self.pt is not None
        r = (self.var < H).nonzero().squeeze(1)
        if not len(r):
            return None
        y = inp[0].clone()
        B, T, _ = y.shape
        h = self.var[r]
        y.view(B, T, H, D)[r, :, h] = self.z.view(-1, T, H, D)[self.pt[r], :, h]
        return (y,)

    def mlp(self, _m: Any, inp: tuple[Tensor, ...]) -> tuple[Tensor] | None:
        assert self.var is not None and self.ex is not None and self.pt is not None
        ex, pt = self.ex, self.pt
        neu = ((self.var >= H) & (self.var < H + NEUR)).nonzero().squeeze(1)
        rem = (self.var == H + NEUR).nonzero().squeeze(1)
        if not len(neu) and not len(rem):
            return None
        # A neuron or remainder row swaps nothing before down_proj, so x's activation here is F's cached run on x: it is
        # read from the cache. Split on the rerun's copy (off by rounding), an activation at its bound changed side and
        # moved the identity swap by a whole bound (E326 stage 0, e62 and e46).
        a = inp[0].clone()
        if len(neu):
            i = self.var[neu] - H
            th = self.hi[i][:, None]
            cur, new = self.a[ex[neu], :, i], self.a[pt[neu], :, i]  # (k, T)
            a[neu, :, i] = (cur - strong(cur, th)) + strong(new, th)  # each part is exact: s is either a or 0
        if len(rem):
            cur, new = self.a[ex[rem]], self.a[pt[rem]]
            a[rem] = strong(cur, self.hi) + (new - strong(new, self.hi))
        return (a,)


def scan(net: Any, feat: K.Feat, run: Run, at: slice, ex: Tensor, pt: Tensor, rows: int, log: Callable[[str], None]) -> IX.Scan:
    """Alias for finished callers: `needed.scan_sites` over the layers' `Swap`s."""
    assert feat.lo is None, "upper-tail featurization only"
    return IX.scan_sites(net, [Swap(run, l, feat.hi[l]) for l in range(len(net.h))], run, at, ex, pt, rows, len(net.h) * PER_LAYER, log)


# ---- the write share's sites -------------------------------------------------------------------------------------------


@dataclass
class HeadSite:
    """Layer l's 6 heads; y = the block's input (rms_1's input) in F's run, rows of windows of length T."""

    l: int
    off: int
    f: rho.Layer
    m: rho.Layer
    T: int

    @property
    def module(self) -> str:
        return f"h.{self.l}.rms_1"

    @property
    def idx(self) -> slice:
        return slice(self.off, self.off + H)

    @property
    def leftover(self) -> int | None:
        return None

    def writes(self, y: Tensor) -> tuple[Tensor, Tensor]:
        x = y.reshape(-1, self.T, y.shape[-1])
        return self.f.head_writes(x).flatten(0, 1).double(), self.m.head_writes(x).flatten(0, 1).double()  # (rows, H, d)

    def gain_sums(self, y: Tensor) -> tuple[Tensor, Tensor]:
        wF, wM = self.writes(y)
        return (wF * wM).sum((0, 2)), (wM * wM).sum((0, 2))

    def change_sums(self, y: Tensor) -> Tensor:
        wF, wM = self.writes(y)
        return (wF - wM).pow(2).sum((0, 2))


@dataclass
class MLPSite:
    """Layer l's 3072 neurons, then its remainder; y = the residual before the MLP (rms_2's input) in F's run."""

    l: int
    off: int
    f: rho.Layer
    m: rho.Layer
    hi: Tensor

    @property
    def module(self) -> str:
        return f"h.{self.l}.rms_2"

    @property
    def idx(self) -> slice:
        return slice(self.off, self.off + NEUR + 1)

    @property
    def leftover(self) -> int | None:
        """The layer's remainder (the weak part of every neuron)."""
        return self.off + NEUR

    def parts(self, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(s_F, s_M, e_F, e_M): the strong parts (rows, n) and the remainders' writes (rows, d), fp64."""
        aF, aM = self.f.acts(y), self.m.acts(y)
        sF, sM = strong(aF, self.hi), strong(aM, self.hi)
        return sF.double(), sM.double(), (aF - sF).double() @ self.f.w_out.double().T, (aM - sM).double() @ self.m.w_out.double().T

    def gain_sums(self, y: Tensor) -> tuple[Tensor, Tensor]:
        sF, sM, eF, eM = self.parts(y)
        WF, WM = self.f.w_out.double(), self.m.w_out.double()
        num = (sF * sM).sum(0) * (WF * WM).sum(0)
        den = (sM * sM).sum(0) * (WM * WM).sum(0)
        return torch.cat([num, (eF * eM).sum().reshape(1)]), torch.cat([den, (eM * eM).sum().reshape(1)])

    def change_sums(self, y: Tensor) -> Tensor:
        """|s_F W_F - s_M W_M|^2 = ds^2 |W_F|^2 + 2 ds s_M <W_F, dW> + s_M^2 |dW|^2 per neuron (ds = s_F - s_M, dW = W_F - W_M;
        exactly 0 when both are equal), |e_F - e_M|^2 for the remainder."""
        sF, sM, eF, eM = self.parts(y)
        WF = self.f.w_out.double()
        dW, ds = WF - self.m.w_out.double(), sF - sM
        neu = (ds * ds).sum(0) * (WF * WF).sum(0) + 2 * (ds * sM).sum(0) * (WF * dW).sum(0) + (sM * sM).sum(0) * (dW * dW).sum(0)
        return torch.cat([neu, (eF - eM).pow(2).sum().reshape(1)])


def sites(p: rho.Pair, feat: K.Feat, T: int) -> list[Any]:
    """Two sites per layer in variable order (heads, then neurons and the remainder), for `stand_in.write_change`."""
    out: list[Any] = []
    for l, (f, m) in enumerate(zip(p.f.layers, p.m.layers, strict=True)):
        out += [HeadSite(l, l * PER_LAYER, f, m, T), MLPSite(l, l * PER_LAYER + H, f, m, feat.hi[l])]
    return out

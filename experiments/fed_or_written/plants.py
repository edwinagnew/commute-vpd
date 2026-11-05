"""E100 Amendment 2's written reference, shared by amend2.py (job A) and amend2_sharpen.py (job B). GPU only. Plan:
reports/plans/2026-11-05_plan_e100-amend2.md; rule (Fable, section 3): reports/reviews/2026-11-05_review_e100-next-fable-advice.md.

A plant is written by construction: per part p a minimum-norm linear write X_p (`perturb.key_write`: X_p a_k = the unembedding
row of key k's answer, a_k the mean of p's input at the read position over the windows holding key k, W0, fp32), into
  an MLP   block b's down_proj (input: the MLP's hidden activation), as the independent check's `spread_plant.py` and E97's
           `locator.plant` build it;
  a head   block b's o_proj columns of head h (input: the head's output slice), the extension Fable names.
The plant is sum_p alpha w_p X_p. Stage 0's 4-MLP plant (the check's) takes w_p = 1 on the four MLPs; a profile-matched
plant takes w_p = sqrt(share_p) / ||X_p||, share_p the state's own ||dW||^2 share of part p (28 parts, as
`locator.part_reverted` cuts them; the check's `part_norms.py`), so the plant's per-part norm profile is the state's. One
common alpha (`perturb.match`, rel tol 5%) sets its gain on the site to the state's G. W = its sum of single reverts.

Keys: a key table's site (E90's kv_site) takes the table's train windows keyed by the key at `half` (the check's);
every other site takes its own reads keyed by their target (the read position's next token).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from commute import ctxpatch as CX
from commute import fed_rules as FR
from commute import locator as L
from commute import perturb as PT
from commute import steering as ST

Writes = dict[str, tuple[str, slice | None, Tensor]]  # part -> (matrix, its column slice or None, X (768, d_in) float64 CPU)
REL_TOL = 0.05  # the check's spread_plant.py REL_TOL, Fable's "within 5%"


def part_shares(dW: ST.DW) -> dict[str, float]:
    """Each of the 28 parts' share of ||dW||^2 (the check's part_norms.py `norms`); they sum to 1."""
    total = sum(float(v.double().pow(2).sum()) for v in dW.values())
    return {n: sum(float(v.double().pow(2).sum()) for v in L.grafted(dW, n).values()) / total for n in map(L.name, L.PARTS)}


def read_sum(net: ST.Net, b: L.Base, dW: ST.DW) -> dict[str, Any]:
    """G and the 28 single parts' own shares (unit lo) and their sum: amend1.py's read, on a given base."""
    t0 = time.monotonic()
    st = L.state(net, b, dW)
    own = {L.name(p): L.own(net, st, L.name(p))["lo"] for p in L.PARTS}
    return {"G": float(st.G[0]), "own": own, "reverts_sum": FR.reverts_sum({n: {"own": x} for n, x in own.items()}),
            "seconds": time.monotonic() - t0}  # fmt: skip


@torch.no_grad()
def key_inputs(net: ST.Net, toks: Tensor, win: Tensor, pos: Tensor, keys: Tensor, batch: int) -> tuple[list[int], dict[str, Tensor]]:
    """Per part, the mean input (float64, CPU) over the reads holding each key, one column per key in sorted key order. The
    reads are (win, pos) on `toks`; `keys` (P,) the key of each read."""
    ks = keys.unique().tolist()
    col = {k: i for i, k in enumerate(ks)}
    idx = torch.tensor([col[k] for k in keys.tolist()])
    count = torch.bincount(idx, minlength=len(ks)).double()
    sums: dict[str, Tensor] = {}
    got: dict[str, Tensor] = {}
    hs = [net.net.h[b].attn.o_proj.register_forward_pre_hook(lambda _m, a, b=b: got.__setitem__(f"attn{b}", a[0])) for b in range(L.BLOCKS)]
    try:
        for i in range(0, len(toks), batch):
            sel = torch.nonzero((win >= i) & (win < i + batch), as_tuple=True)[0]
            if not len(sel):
                continue
            mlp = net.run(toks[i : i + batch], capture=True)[1]
            for b in range(L.BLOCKS):
                for name, x in ((f"mlp{b}", mlp[f"mlp{b}"]), (f"attn{b}", got[f"attn{b}"])):
                    at = x[(win[sel] - i).to(x.device), pos[sel].to(x.device)].double()
                    acc = sums.setdefault(name, torch.zeros(len(ks), at.shape[1], dtype=torch.float64, device=x.device))
                    acc.index_add_(0, idx[sel].to(x.device), at)
    finally:
        for h in hs:
            h.remove()
    return ks, {n: (s.cpu() / count[:, None]).T for n, s in sums.items()}


def writes(net: ST.Net, toks: Tensor, win: Tensor, pos: Tensor, keys: Tensor, answer: Mapping[int, int], batch: int) -> Writes:
    """X_p for the 28 parts: `key_write` of each part's per-key mean input onto the answers' unembedding rows."""
    ks, A = key_inputs(net, toks, win, pos, keys, batch)
    U = net.net.lm_head.weight.detach().double().cpu()[torch.tensor([answer[k] for k in ks])].T
    out: Writes = {}
    for b in range(L.BLOCKS):
        out[f"mlp{b}"] = (f"h.{b}.mlp.down_proj", None, PT.key_write(A[f"mlp{b}"], U))
        for h in range(L.HEADS):
            sl = slice(h * L.HEAD_DIM, (h + 1) * L.HEAD_DIM)
            out[f"{b},{h}"] = (f"h.{b}.attn.o_proj", sl, PT.key_write(A[f"attn{b}"][sl], U))
    return out


def kv_writes(net: ST.Net, w: dict[str, Any], batch: int) -> Writes:
    """A key table's writes from its train windows, keyed by the key at `half` (the check's `writes`, extended to heads)."""
    t = w["train"]
    n = len(t.tokens)
    keys = t.tokens[:, t.half]
    return writes(net, t.tokens, torch.arange(n), torch.full((n,), t.half), keys, {k: int(w["table"][k]) for k in keys.unique().tolist()}, batch)


def site_writes(net: ST.Net, s: L.Site) -> Writes:
    """Any other site's writes from its own reads, keyed by the target there."""
    return writes(net, s.toks, s.win, s.pos, s.tgt, {k: k for k in s.tgt.unique().tolist()}, s.batch)


def build(net: ST.Net, X: Writes, weight: Mapping[str, float], alpha: float) -> ST.DW:
    mats: dict[str, Tensor] = {}
    for p, wp in weight.items():
        if wp == 0:
            continue
        key, sl, x = X[p]
        m = mats.setdefault(key, torch.zeros_like(net.W0[key], dtype=torch.float64, device="cpu"))
        if sl is None:
            m += alpha * wp * x
        else:
            m[:, sl] += alpha * wp * x
    return CX.with_matrices(net, mats)


def profile(X: Writes, shares: Mapping[str, float]) -> dict[str, float]:
    """w_p = sqrt(share_p) / ||X_p||, so alpha = 1 gives a plant of norm 1 with the state's per-part norm profile."""
    return {p: (shares[p] ** 0.5) / float(X[p][2].norm()) if shares[p] > 0 else 0.0 for p in X}


def plant(net: ST.Net, b: L.Base, X: Writes, weight: Mapping[str, float], Gt: float, th: Mapping[str, Any]) -> dict[str, Any]:
    """The plant at the state's gain on its site: alpha, its G, match status, W (its sum of single reverts), its 28 own
    shares, its per-part norm shares, and the reference status (`fed_rules.reference_status`)."""
    t0 = time.monotonic()
    lo0 = float(b.lo0[:, 0].mean())
    found = PT.match(lambda a: float(L.read(net, b.site, build(net, X, weight, a))[:, 0].mean()) - lo0, Gt, PT.ALPHA_MAX, REL_TOL) \
        if Gt > 0 else {"x": 0.0, "f": float("nan"), "status": "no gain"}
    rec: dict[str, Any] = {"alpha": found["x"], "G_target": Gt, "match_status": found["status"]}
    if found["status"] == "ok":
        dW = build(net, X, weight, found["x"])
        r = read_sum(net, b, dW)
        rec |= {"G": r["G"], "W": r["reverts_sum"], "own": r["own"], "shares": part_shares(dW), "norm": sum(float(v.double().pow(2).sum()) for v in dW.values()) ** 0.5}
    else:
        rec |= {"G": found["f"], "W": float("nan")}
    rec["status"] = FR.reference_status(rec["match_status"], Gt, rec["W"], th["min_gain"], th["w_floor"])
    rec["seconds"] = time.monotonic() - t0
    return rec

"""E99's reads of one state at one part, on `commute.locator`'s sites and removals (E97's apparatus). GPU only.

A part (or pair) is removed by setting its heads' outputs to W0's mean over the site's own inputs (`locator.removal`'s
means). Per state: G, the state's own fall, W0's own fall, the net share (rules.py) in both units, and the same net share
for each random pair, whose 95th percentile of |.| is the state's band. The per-read tensors of the main part are kept.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import rules as R
import torch
from torch import Tensor

from commute import locator as L
from commute import locator_rules as LR
from commute import performed, vpd
from commute import steering as ST
from commute.data import pile_rows

UNITS = ("lo", "gap")


def site_base(net: ST.Net, s: L.Site) -> L.Base:
    """W0 on a site; the per-input part outputs are dropped once the means are taken (only the removal is read here)."""
    b = L.base(net, s, {})
    b.acts0.clear()
    torch.cuda.empty_cache()
    return b


def cased_site(n_rows: int) -> L.Site:
    """E53's case positions read in the cased val rows themselves (`performed.case_pairs` `in_cased`): target the lowercase
    form, other form the capitalised one, so the "gap" unit is W0's lowercase-minus-capitalised log-odds on cased text."""
    rows = pile_rows("val", 0, n_rows)
    at = performed.case_pairs(rows, vpd.tokenizer()).in_cased
    return L.Site(rows, at.row, at.pos - 1, at.a, at.row, at.b)


def first_inputs(s: L.Site, n: int | None) -> L.Site:
    """The site cut to its first n inputs and the reads on them (the cheap read the matched-gain check uses)."""
    if n is None:
        return s
    keep = s.win < n
    return L.Site(s.toks[:n], s.win[keep], s.pos[keep], s.tgt[keep], s.row[keep], None if s.alt is None else s.alt[keep])


def means_tensor(b: L.Base) -> Tensor:
    """The site's head means as `copying.ablated` takes them, (blocks, heads, head_dim): the same vectors the removal uses."""
    return torch.stack([b.means["attn", k].view(L.HEADS, L.HEAD_DIM) for k in range(L.BLOCKS)])


def heads_of(pair: str) -> list[tuple[int, int]]:
    return [(p[1], p[2]) for p in L.parts_of(pair)]


def random_pairs(pair: str, n: int, seed: int) -> list[str]:
    """n distinct seeded sets of as many heads as the part has, all outside it, as part names. Unlike
    `copying.random_others` (whose draws repeat: 3 of 20 were one pair), no set is drawn twice."""
    part = set(heads_of(pair))
    rest = [(b, h) for b in range(L.BLOCKS) for h in range(L.HEADS) if (b, h) not in part]
    sets = list(itertools.combinations(rest, len(part)))
    pick = torch.randperm(len(sets), generator=torch.Generator().manual_seed(seed))[:n]
    return ["+".join(L.name(("head", b, h)) for b, h in sets[int(i)]) for i in pick]


def w0_move(net: ST.Net, b: L.Base, pair: str) -> dict[str, Any]:
    """W0's own fall with the part off, per unit, and twice its standard error by source row (unit "lo", nats)."""
    fall = b.lo0 - L.w0_off(net, b, pair)
    return {"fall": L.units(fall.mean(0)), "band_2se": 2 * LR.row_se(fall[:, 0], b.site.row)}


def pair_read(net: ST.Net, st: L.State, pair: str) -> tuple[dict[str, Any], dict[str, Tensor]]:
    b = st.b
    off = L.read(net, b.site, st.dW, L.parts_of(pair), b.means)
    off0 = L.w0_off(net, b, pair)
    fall, fall0 = (st.lo1 - off).mean(0), (b.lo0 - off0).mean(0)
    G = float(st.G[0])
    per = ((st.lo1 - off) - (b.lo0 - off0))[:, 0] / G if G > 0 else None
    rec = {"G": L.units(st.G), "fall": L.units(fall), "w0_fall": L.units(fall0), "own_share": L.shares(fall, st.G),
           "net": {u: R.net_share(float(fall[i]), float(fall0[i]), float(st.G[i])) for i, u in enumerate(UNITS)},
           "net_se_by_row": LR.row_se(per, b.site.row) if per is not None else math.nan,
           "p_mean": LR.mean_prob(st.lo1[:, 0]), "p_mean_w0": LR.mean_prob(b.lo0[:, 0]), "p_mean_off": LR.mean_prob(off[:, 0])}  # fmt: skip
    return rec, {"lo0": b.lo0, "lo1": st.lo1, "off": off, "off0": off0}


def state_read(net: ST.Net, st: L.State, pair: str, draws: list[str]) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """The part's read and the state's band from the random pairs (W0's reads under each are cached on the site)."""
    rec, t = pair_read(net, st, pair)
    rec["random_net"] = {d: pair_read(net, st, d)[0]["net"]["lo"] for d in draws}
    rec["band"] = R.band(list(rec["random_net"].values()))
    return rec, t


def gain(net: ST.Net, s: L.Site, lo0: Tensor, dW: ST.DW | None = None) -> float:
    """Mean change of unit "lo" over the site's reads against `lo0`; dW None reads the live weights (inside training)."""
    return float((L.read(net, s, dW)[:, 0] - lo0[:, 0]).mean())

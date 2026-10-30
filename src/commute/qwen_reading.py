"""E57 (experiments/qwen_reading, plan reports/plans/2026-10-30_plan_q3-reading-the-update.md): the rules that read
an update's route split, Γ's switching and the reference sets, and the row builders. Every rule is a deterministic
function of its arguments and rounds to 9 places before comparing; NaN, infinity or None misses. The one GPU piece is
`gate_counts`, Γ's gates counted per subcomponent without holding every site's g at once (rows of ~2,300 tokens).
Tests: tests/test_qwen_reading.py.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute.qwen_maps import SHARED_INPUT
from commute.qwen_vpd import (
    SITES,
    Components,
    Gamma,
    backbone,
    parse_site,
    readout,
    site_importance,
    site_name,
)

NAN = float("nan")
THRESH, FRAC = 0.01, 0.01  # gated on: g > 0.01 at >= 1% of the positions read (E45)
FLOOR = max(0.05, 5 * 0.000208)  # E45's filed floor on a share's denominator (E44 C.json stochastic spread)


def ok(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def r9(x: float) -> float:
    return round(x, 9)


def div(x: float, y: float) -> float:
    return x / y if ok(x) and ok(y) and y != 0 else NAN


# ----------------------------------------------------------------------------- row 1, the route split


def shares(k: dict[str, float]) -> dict[str, Any]:
    """Re-gating and direct shares from the cells' means (or sums) K00, K01, K10, K11, K0, with E45's status rule:
    'nan' (a value missing or not finite), 'unscored' (K00 - K0 <= 0, K11 - K0 > 0.5 (K00 - K0), or a denominator
    under FLOOR) or 'scored'. The shares are given in every case."""
    K00, K01, K10, K11, K0 = (float(k.get(n, NAN)) for n in ("K00", "K01", "K10", "K11", "K0"))
    den = K00 - K11
    out = {"regating": div(K00 - K01, den), "direct": div(K00 - K10, den), "K11_minus_K0": K11 - K0, "den": den}
    if not all(ok(x) for x in (K00, K01, K10, K11, K0)):
        return out | {"status": "nan"}
    gap = K00 - K0
    unscored = r9(gap) <= 0 or r9(K11 - K0) > r9(0.5 * gap) or r9(den) < r9(FLOOR)
    return out | {"status": "unscored" if unscored else "scored"}


def resample(n: int, boot: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(boot)]


def interval(xs: list[float]) -> list[float]:
    """The 2.5th and 97.5th percentiles of the finite values (NaN, NaN if none)."""
    v = sorted(x for x in xs if ok(x))
    return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else [NAN, NAN]


def sums_at(rows: dict[str, list[float]], idx: list[int]) -> dict[str, float]:
    return {k: sum(v[i] for i in idx) for k, v in rows.items()}


def boot_share(rows: dict[str, list[float]], draws: list[list[int]]) -> list[float]:
    """The re-gating share's 95% interval over row resamples of the per-row cell sums."""
    return interval([shares(sums_at(rows, idx))["regating"] for idx in draws])


def teeth(e40: dict[str, list[float]], controls: dict[str, dict[str, Any]], draws: list[list[int]],
          high: float = 0.8) -> dict[str, Any]:
    """The C0 verdict (plan section 4, row 1). controls[name] = {"rows": per-row sums on E40's rows, "matched": bool}.
    'no teeth': a scored control share >= high. 'teeth': every control scored and matched, E40 scored, and each
    paired-bootstrap gap (E40's share minus the control's) has its 95% interval above 0. Otherwise 'weak'."""
    e = shares(sums_at(e40, list(range(len(next(iter(e40.values())))))))
    out: dict[str, Any] = {"e40": e["regating"], "e40_status": e["status"], "controls": {}}
    verdict = "teeth" if e["status"] == "scored" else "weak"
    for name, c in controls.items():
        s = shares(sums_at(c["rows"], list(range(len(next(iter(c["rows"].values())))))))
        gaps = interval([shares(sums_at(e40, i))["regating"] - shares(sums_at(c["rows"], i))["regating"]
                         for i in draws])
        out["controls"][name] = {"share": s["regating"], "status": s["status"], "matched": c["matched"],
                                 "gap": e["regating"] - s["regating"], "gap95": gaps}
        if s["status"] == "scored" and r9(s["regating"]) >= high:
            verdict = "no teeth"
        elif verdict == "teeth" and not (s["status"] == "scored" and c["matched"] and ok(gaps[0]) and r9(gaps[0]) > 0):
            verdict = "weak"
    return out | {"verdict": verdict}


def scale_search(f: Callable[[float], float], target: float, match: float = 1.5, evals: int = 6) -> dict[str, Any]:
    """The scale b with f(b) (a KL, ~ b² for a small change) within the factor `match` of target: b ← b·√(target/f),
    each step held within ×/÷ 100, from b = 1, at most `evals` evaluations; the closest evaluated b is kept. A
    non-finite or non-positive f counts as too large (b / 10 next); no evaluation for a target that is not > 0."""
    b, out = 1.0, []
    for _ in range(evals if ok(target) and target > 0 else 0):
        x = f(b)
        out.append([b, x])
        if within(x, target, match):
            break
        b = b * min(100.0, max(0.01, math.sqrt(target / x))) if ok(x) and x > 0 else b / 10
    gap = lambda e: abs(math.log(e[1] / target)) if ok(e[1]) and e[1] > 0 else math.inf
    best = min(out, key=gap, default=[1.0, NAN])
    return {"target": target, "evals": out, "b": best[0], "kl": best[1], "matched": within(best[1], target, match)}


def within(x: float, target: float, match: float) -> bool:
    r = div(x, target)
    return ok(r) and r > 0 and r9(max(r, 1 / r)) <= r9(match)


def close(x: float, ref: float, tol: float) -> bool:
    """|x / ref - 1| <= tol (relative)."""
    g = abs(div(x, ref) - 1)
    return ok(g) and r9(g) <= tol

# ----------------------------------------------------------------------------- rows 2 and 3, sets


def gated(active: Tensor, n_pos: int) -> Tensor:
    """Gated on: active (positions with g > THRESH) >= FRAC of n_pos, in integers (100 · active >= n_pos)."""
    assert n_pos > 0, "no positions"
    return active.long() * round(1 / FRAC) >= n_pos


def split_floor(on_a: Tensor, on_b: Tensor) -> int:
    """The split-half count: the larger of |b and not a| and |a and not b|."""
    return int(max((on_b & ~on_a).sum(), (on_a & ~on_b).sum()))


def min_size(floor_count: int) -> int:
    return max(20, 5 * floor_count)


def quantile(draws: np.ndarray, p: float) -> float:
    """The ceil(p·n)-th smallest draw."""
    v = np.sort(draws)
    return float(v[max(0, math.ceil(r9(p * len(v))) - 1)])


def overlap(S: Tensor, X: Tensor, site_of: Tensor, n_sites: int, n: int = 1000, seed: int = 0) -> dict[str, Any]:
    """o = |S ∩ X| against n site-stratified random sets of S's per-site counts (hypergeometric per site: drawing
    k of a site's N live subcomponents, K of them in X). S, X: bool over the flat live layout; site_of: its site."""
    k = torch.bincount(site_of[S], minlength=n_sites).numpy()
    K = torch.bincount(site_of[X], minlength=n_sites).numpy()
    N = torch.bincount(site_of, minlength=n_sites).numpy()
    rng, draws = np.random.default_rng(seed), np.zeros(n, dtype=np.int64)
    for i in np.nonzero(k)[0]:
        draws += rng.hypergeometric(int(K[i]), int(N[i] - K[i]), int(k[i]), size=n)
    o, mean, sd = int((S & X).sum()), float(draws.mean()), float(draws.std())
    return {"o": o, "size_S": int(S.sum()), "size_X": int(X.sum()), "mean": mean, "sd": sd,
            "e": (o - mean) / sd if sd > 0 else NAN, "q95": quantile(draws, 0.95), "q99": quantile(draws, 0.99)}


def above_99(ov: dict[str, Any]) -> bool:
    return ov["sd"] > 0 and ov["o"] > ov["q99"]


def at_most_95(ov: dict[str, Any]) -> bool:
    return ov["sd"] > 0 and ov["o"] <= ov["q95"]


# ----------------------------------------------------------------------------- section 5, the filed anchor rules


def prompt_row(anchor: int, ov: dict[str, Any] | None, size_floor: int, prompt_size: int) -> str:
    """'unscored' (S or S_prompt under the floor, or a draw sd of 0), else 'holds' or 'misses' (anchor 1: o > q99;
    anchor 2: o <= q95)."""
    if ov is None or ov["size_S"] < size_floor or prompt_size < size_floor or not ov["sd"] > 0:
        return "unscored"
    return "holds" if (above_99(ov) if anchor == 1 else at_most_95(ov)) else "misses"


def think_row(ov: dict[str, Any] | None, size_floor: int, ref_sizes: Sequence[int]) -> str:
    """Anchor 2's think overlap: 'unscored' (any of |T|, |L|, |C| < 20, S under the floor, a draw sd of 0), else
    'holds' when o <= q95."""
    if ov is None or min(ref_sizes) < 20 or ov["size_S"] < size_floor or not ov["sd"] > 0:
        return "unscored"
    return "holds" if at_most_95(ov) else "misses"


def validated(a1: Sequence[float | None], a2: Sequence[float | None], se: float | None) -> dict[str, Any]:
    """A row on its per-seed values (e_X; None or NaN: that seed unscored) and the gap's row-bootstrap s.e.:
    validated when all four are finite and |mean gap| > both the larger seed gap and 2 se; direction 'as filed'
    (anchor 1 above) or 'reversed'."""
    vals = [*a1, *a2]
    if len(a1) != 2 or len(a2) != 2 or not all(ok(x) for x in vals) or not ok(se):
        return {"validated": False, "why": "an anchor seed unscored, or no s.e."}
    x1, y1, x2, y2 = (float(x) for x in vals)  # type: ignore[arg-type]
    gap = (x1 + y1) / 2 - (x2 + y2) / 2
    bar = max(abs(x1 - y1), abs(x2 - y2), 2 * float(se))  # type: ignore[arg-type]
    v = r9(abs(gap)) > r9(bar)
    return {"validated": v, "gap": gap, "bar": bar,
            "direction": None if not v else "as filed" if gap > 0 else "reversed"}


# ----------------------------------------------------------------------------- row 4, where δW lands


def effective_rank(sv: Tensor) -> float:
    """Roy & Vetterli (2007): exp of the entropy of the singular values normalised to sum 1 (NaN for δW = 0)."""
    s = sv.double().clamp(min=0)
    total = float(s.sum())
    if total <= 0:
        return NAN
    p = s[s > 0] / total
    return math.exp(float(-(p * p.log()).sum()))


def placement(pe: Tensor, on: Tensor) -> float:
    """Share of a split's per-subcomponent piece energy on the gated-on set."""
    return div(float(pe[on].sum()), float(pe.sum()))

# ----------------------------------------------------------------------------- rows: unpacking and contexts


def unpack(flat: Sequence[int], prompts: Sequence[Sequence[int]]) -> list[list[int]]:
    """The per-problem sequences (prompt, then response) of rows packed end to end (qwen_sft.pack): each starts with
    its prompt and ends where the next prompt starts; the last one, cut by the rows' end, is dropped."""
    flat, out, p = list(flat), [], 0
    for P, Q in itertools.pairwise(prompts):
        assert flat[p : p + len(P)] == list(P), f"prompt {len(out)} not found at {p}"
        q = next((j for j in range(p + len(P), len(flat) - len(Q) + 1) if flat[j : j + len(Q)] == list(Q)), None)
        if q is None:
            break
        out.append(flat[p:q])
        p = q
    return out


def after(ids: Sequence[int], marker: int) -> int:
    """The first position after the last `marker` (len(ids) if absent: no position read)."""
    hits = [i for i, t in enumerate(ids) if t == marker]
    return hits[-1] + 1 if hits else len(ids)


def filler(seqs: Sequence[Sequence[int]], i: int, n: int) -> list[int]:
    """The last n tokens of the other problems' sequences i+1, i+2, ... (cyclic, i excluded) joined in order."""
    assert len(seqs) > 1 or n == 0
    out: list[int] = []
    j = i
    while len(out) < n:
        j = (j + 1) % len(seqs)
        if j != i:
            out += list(seqs[j])
    return out[len(out) - n :] if n > 0 else []


def length_matched(plain: Sequence[int], think: Sequence[int], seqs: Sequence[Sequence[int]], i: int, marker: int
                   ) -> list[int]:
    """L: the plain row after a filler of earlier maths turns, its last `marker` at the think row's position."""
    n = max(0, (after(think, marker) - 1) - (after(plain, marker) - 1))
    return filler(seqs, i, n) + list(plain)


def own_rows(tok: Any, arm: str, seed: int, train: list[dict[str, Any]]) -> list[list[int]]:
    """E45's own rows for gold and anchor 2 as experiments/qwen_updates/job.py Job.own_rows builds them (that file
    is not imported: E57 checks the KL on these rows against E45's E record instead)."""
    from commute import qwen_facts
    from commute.qwen_anchors import chat
    from commute.qwen_sets import N_TRAIN
    from commute.qwen_sft import pack, sft_seq

    end = int(tok.convert_tokens_to_ids("<|im_end|>"))
    ids = lambda text: tok(text, add_special_tokens=False)["input_ids"]
    if arm == "gold":
        held = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(0)).tolist()[:256]
        seqs = [sft_seq(tok, train[i]["question"], train[i]["gold"], end, 512)[0] for i in held]
    else:
        assert arm == "a2", arm
        seqs = [ids(chat(tok, q.question)) + ids(q.answer) + [end] for q in qwen_facts.held_set(seed)]
    return pack(seqs, 64, 512)

# ----------------------------------------------------------------------------- Γ's gates, counted (GPU)


class Layout:
    """The live subcomponents as one flat vector, site by site in SITES order."""

    def __init__(self, idx: dict[str, Tensor]):
        self.idx = idx
        sizes = [len(idx[s]) for s in SITES]
        self.offset = dict(zip(SITES, np.cumsum([0, *sizes[:-1]]).tolist(), strict=True))
        self.n = sum(sizes)
        self.site_of = torch.repeat_interleave(torch.arange(len(SITES)), torch.tensor(sizes))

    def kind_mask(self, *kinds: str) -> Tensor:
        """The flat positions at sites of these matrix kinds."""
        return torch.tensor([parse_site(s)[1] in kinds for s in SITES])[self.site_of]


class Counts:
    """Per live subcomponent: positions with g > THRESH and the sum of g, over the positions read; optionally per
    row (int16, for a row bootstrap)."""

    def __init__(self, lay: Layout, device: str, n_rows: int = 0):
        self.lay, self.n_pos = lay, 0
        self.active = torch.zeros(lay.n, dtype=torch.int64, device=device)
        self.gsum = torch.zeros(lay.n, dtype=torch.float64, device=device)
        self.rows = torch.zeros(n_rows, lay.n, dtype=torch.int16, device=device) if n_rows else None

    def add(self, site: str, g: Tensor, keep: Tensor | None = None, row: int | None = None) -> None:
        """g: one row's [T, C] gates at a site; keep: bool [T] positions read (None: all)."""
        ga = (g if keep is None else g[keep])[:, self.lay.idx[site].to(g.device)]
        o = self.lay.offset[site]
        a = (ga > THRESH).sum(0)
        self.active[o : o + len(a)] += a
        self.gsum[o : o + len(a)] += ga.double().sum(0)
        if self.rows is not None and row is not None:
            self.rows[row, o : o + len(a)] = a.to(torch.int16)

    def add_row(self, g: dict[str, Tensor], keep: Tensor | None = None, row: int | None = None) -> None:
        for s in SITES:
            self.add(s, g[s][0], keep, row)
        self.n_pos += int(next(iter(g.values())).shape[1] if keep is None else keep.sum())

    def on(self) -> Tensor:
        return gated(self.active.cpu(), self.n_pos)

    def tensors(self) -> dict[str, Tensor]:
        out = {"active": self.active.cpu(), "gsum": self.gsum.float().cpu(), "n_pos": torch.tensor([self.n_pos])}
        return out | ({"rows": self.rows.cpu()} if self.rows is not None else {})


@torch.no_grad()
def gate_counts(model: Any, tokens: Tensor, gamma: Gamma, comps: Components, counts: Counts,
                keep: Tensor | None = None) -> None:
    """Γ's gates for one row ([1, T]) added to `counts`, site by site: the taps on the GPU, each shared site input in
    Γ's dtype on the host (what `readout` casts it to), no logits and one site's g at a time. The same values as
    `qwen_vpd.importances` (Stage A checks it)."""
    taps: list[Tensor] = []
    inputs: dict[str, Tensor] = {}
    owner = {s: site_name(parse_site(s)[0], SHARED_INPUT[parse_site(s)[1]]) for s in SITES}
    m = model.model
    hooks = [m.embed_tokens.register_forward_hook(lambda _m, _a, o: taps.append(o))]
    hooks += [layer.register_forward_hook(lambda _m, _a, o: taps.append(o)) for layer in m.layers]

    def capture(s: str) -> Any:
        return lambda _m, a, _o: inputs.__setitem__(s, a[0].to(gamma.dtype).cpu())

    hooks += [m.get_submodule(s).register_forward_hook(capture(s)) for s in sorted(set(owner.values()))]
    try:
        m(tokens, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    x = backbone(gamma, taps)
    taps.clear()
    for s in SITES:
        r = readout(gamma, s, inputs[owner[s]].to(x.device), comps.vu(s)[0])
        counts.add(s, site_importance(gamma, x, s, r)[0], keep)
        del r
    counts.n_pos += tokens.shape[1] if keep is None else int(keep.sum())

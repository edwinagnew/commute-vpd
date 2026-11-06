"""E116 stage 0: E112's possession numbers reproduced through the moved probe code, then P0 per readout C (appendix §4):
Pile positions of 41-token windows from rows 32768-65535, labels, the Hewitt-Liang control keyed on the key slot (lead's
decision 3), `case_label.value_probes` per block, l* and the site. GPU only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tasks import HALF, Readout, Site
from torch import Tensor

from commute import case_label as CL
from commute import readout as RO
from commute import steering as ST
from commute.jobs import load_experiment

WIN = 2 * HALF + 1


@torch.no_grad()
def resid_at(net: ST.Net, toks: Tensor, half: int, batch: int = 64) -> Tensor:
    """(4, N, d) fp32 CPU: W0's block outputs at index `half`."""
    return torch.cat([RO.resid(net, toks[s : s + batch])[1][:, :, half].cpu() for s in range(0, len(toks), batch)], 1)


@torch.no_grad()
def argmax_in(net: ST.Net, toks: Tensor, half: int, values: Tensor | None, batch: int = 64) -> Tensor:
    """(N,) W0's argmax at `half`: the index among `values`, or the full-vocabulary token when values is None."""
    out = []
    for s in range(0, len(toks), batch):
        lg = net.run(toks[s : s + batch])[0][:, half]
        out.append((lg if values is None else lg[:, values.to(lg.device)]).argmax(-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def index_of(values: Tensor, toks: Tensor, vocab: int) -> Tensor:
    """(N,) each token's index in `values`, -1 when it is not one."""
    t = torch.full((vocab,), -1, dtype=torch.long)
    t[values] = torch.arange(len(values))
    return t[toks]


def cut(rows: Tensor, r: Tensor, t: Tensor) -> Tensor:
    return rows[r[:, None], t[:, None] - HALF + torch.arange(WIN)]


def capped(rows: Tensor, r: Tensor, t: Tensor, y: Tensor, types: Tensor, cap: int, seed: int, vocab: int) -> tuple[Tensor, Tensor, Tensor]:
    """At most `cap` positions per (value, key-slot type), seeded: (windows, labels, types)."""
    keep = CL.cap_groups(y * vocab + types, cap, seed)
    return cut(rows, r[keep], t[keep]), y[keep], types[keep]


def copy_positions(rows: Tensor, r: Readout, C: Any, vocab: int) -> tuple[Tensor, Tensor, Tensor]:
    """In-context copying on Pile: positions t whose token t-1 occurs exactly once earlier in the window (s in t-20..t-2),
    label the token at s+1 when it is a value; t-1 not a key nor the marker. Key slot t-1."""
    T = rows.shape[1]
    a = rows[:, HALF - 1 : T - HALF - 1]  # token t-1 for t = HALF .. T-HALF-1
    count, lab = torch.zeros(a.shape, dtype=torch.int16), torch.zeros_like(a)
    for k in range(1, HALF):  # s = t-1-k
        eq = rows[:, HALF - 1 - k : T - HALF - 1 - k] == a
        count += eq
        lab = torch.where(eq, rows[:, HALF - k : T - HALF - k], lab)
    y = index_of(r.values, lab, vocab)
    ri, j = torch.nonzero((count == 1) & (y >= 0) & ~torch.isin(a, r.exclude), as_tuple=True)
    return capped(rows, ri, j + HALF, y[ri, j], a[ri, j], C.probe_cap, C.probe_seed, vocab)


def identity_positions(rows: Tensor, r: Readout, C: Any) -> tuple[Tensor, Tensor, Tensor]:
    """Token identity: windows whose key-slot token (index HALF + slot) is a value, at most probe_cap per value."""
    toks, y, _ = RO.next_token_windows(rows, r.values, HALF - 1 + r.slot, C.probe_cap, C.probe_seed)
    return toks, y, r.values[y]


@torch.no_grad()
def scan_argmax(net: ST.Net, rows: Tensor, batch: int = 16) -> Tensor:
    """(R, T) W0's full-vocabulary argmax at every position of whole rows (the guess readouts' candidates)."""
    return torch.cat([net.run(rows[s : s + batch])[0].argmax(-1).cpu() for s in range(0, len(rows), batch)])


def guess_positions(net: ST.Net, rows: Tensor, am: Tensor, r: Readout, C: Any, vocab: int) -> tuple[Tensor, Tensor, Tensor]:
    """W0's guess: candidates where W0's whole-row argmax at t is a value (key slot t not an excluded type), at most
    guess_precap x probe_cap per (value, type); re-cut as windows, kept where the window's own argmax at HALF is still a
    value (the label), then at most probe_cap per (label, type)."""
    T = rows.shape[1]
    y0, types = index_of(r.values, am[:, HALF : T - HALF], vocab), rows[:, HALF + r.slot : T - HALF + r.slot]
    ri, j = torch.nonzero((y0 >= 0) & ~torch.isin(types, r.exclude), as_tuple=True)
    keep = CL.cap_groups(y0[ri, j] * vocab + types[ri, j], C.guess_precap * C.probe_cap, C.probe_seed)
    ri, t = ri[keep], j[keep] + HALF
    y = index_of(r.values, argmax_in(net, cut(rows, ri, t), HALF, None), vocab)
    ok = y >= 0
    return capped(rows, ri[ok], t[ok], y[ok], rows[ri[ok], t[ok] + r.slot], C.probe_cap, C.probe_seed, vocab)


def score_labels(net: ST.Net, r: Readout, vocab: int) -> Tensor:
    """The class of each scoring window: copy -> its B; identity -> its key; guess -> W0's argmax among the values."""
    s = r.score
    if r.kind == "guess":
        return argmax_in(net, s.toks, s.half, r.values)
    y = index_of(r.values, s.tgt if r.kind == "copy" else s.toks[:, s.half + r.slot], vocab)
    assert bool((y >= 0).all()), f"{r.name}: a scoring window's class is not a value"
    return y


def p0(net: ST.Net, r: Readout, pile: tuple[Tensor, Tensor, Tensor], C: Any, bars: dict[str, float], large: Path) -> dict[str, Any]:
    """Possession per block (acc, control, selectivity), l*, and the site; probes saved to large/probes_<C>.pt."""
    ptoks, y, types = pile
    vocab, m, s = int(net.net.lm_head.weight.shape[0]), len(r.values), r.score
    ctrl = CL.control_table(vocab, m, C.control_seed)
    yh = score_labels(net, r, vocab)
    h, hh = resid_at(net, ptoks, HALF), resid_at(net, s.toks, s.half)
    W, B, acc = CL.value_probes(h, y, hh, yh, m, C.ridge_lambda)
    Wc, Bc, accc = CL.value_probes(h, ctrl[types], hh, ctrl[s.toks[:, s.half + r.slot]], m, C.ridge_lambda)
    sel = [a - c for a, c in zip(acc, accc, strict=True)]
    l_star = CL.first_passing_block(acc, sel, bars["accuracy"], bars["selectivity"])
    lb = r.locator_block
    at_lb = lb is not None and lb < len(acc) - 1 and acc[lb] >= bars["accuracy"] and sel[lb] >= bars["selectivity"]
    counts = torch.bincount(y, minlength=m)
    torch.save({"W": W, "B": B, "control": (Wc, Bc), "values": r.values, "counts": counts}, large / f"probes_{r.name}.pt")
    return {"m": m, "n_pile": len(y), "values_with_positions": int((counts > 0).sum()), "min_count": int(counts.min()),
            "n_score": len(yh), "score_majority": float(torch.bincount(yh, minlength=m).max() / max(len(yh), 1)),
            "acc": acc, "acc_control": accc, "selectivity": sel, "l_star": l_star, "locator_block": lb,
            "site": lb if at_lb else l_star, "site_from": "locator" if at_lb else "l_star"}  # fmt: skip


def pile_positions(net: ST.Net, r: Readout, rows: Tensor, am: Tensor, C: Any, vocab: int) -> tuple[Tensor, Tensor, Tensor]:
    if r.kind == "copy":
        return copy_positions(rows, r, C, vocab)
    if r.kind == "guess":
        return guess_positions(net, rows[: C.guess_scan_rows], am, r, C, vocab)
    return identity_positions(rows, r, C)


def e112_repro(net: ST.Net, large: Path, tol: float) -> dict[str, Any]:
    """E112's `probes` (its frozen run.py, now calling `case_label.value_probes`) on its saved task.pt and settings,
    its probes.pt written to large/e112_repro/ (not E112's folder); against E112's stage0.json P0, per block."""
    E = load_experiment("case_label", "run")
    S = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(E.PRED.read_text())["settings"].items()}
    t = torch.load(E.LARGE / "task.pt", weights_only=False)
    E.LARGE = large / "e112_repro"
    E.LARGE.mkdir(parents=True, exist_ok=True)
    try:
        got = E.probes(net, E.Config(**S), t)
    finally:
        E.LARGE = E.RESULTS / "large"
    ref = json.loads((E.RESULTS / "stage0.json").read_text())["P0"]
    diff = max(abs(a - b) for k in ("acc_case", "acc_control") for a, b in zip(got[k], ref[k], strict=True))
    return {"got": got, "ref": {k: ref[k] for k in ("acc_case", "acc_control")}, "max_diff": diff, "ok": diff <= tol}


def site_classes(net: ST.Net, probe: dict[str, Any], block: int, s: Site) -> Tensor:
    """(N,) the readout's class token at `block` on W0's residual (a value with no Pile position never wins)."""
    h = resid_at(net, s.toks, s.half)[block].double()
    score = h @ probe["W"][block].T.cpu() + probe["B"][block].cpu()
    return probe["values"][score.masked_fill(probe["counts"][None] == 0, -float("inf")).argmax(1)]

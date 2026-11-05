"""E96: name the part of the 67M a fine-tune's behaviour runs through without being told where to look
(experiments/locate_mechanism; plan reports/plans/2026-11-05_plan_e96-locate-mechanism.md).

The reads: head salience, a per-unit gradient times weight on the state's own log-odds, a ridge probe on head outputs. The
filed rules, pure, so their edge cases run on CPU (tests/test_locate.py). No function here knows which head anyone expects;
the expected head reaches `score` as an argument, from the filed predictions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from commute import copying as CP
from commute import steering as ST
from commute.revival_kv import up_key
from commute.steering import N_LAYERS, Edit
from commute.vpd import target_linear

LIMIT = "a new circuit built from old parts still reads as reuse"


# ----------------------------------------------------------------------------- reads


def _salience_batch(net: Any, w: Tensor, half: int, n: int, means: Tensor, along: tuple[CP.Head, float] | None = None) -> Tensor:
    """(blocks, H) float64, summed over the batch's windows and every position; `head_salience` says what it is."""
    blocks = list(net.net.h)
    H, d = blocks[0].attn.n_head, blocks[0].attn.head_dim
    zeros: list[Tensor] = []
    seen: list[Tensor] = []

    def add(_m: Any, args: tuple[Tensor, ...]) -> tuple[Tensor]:
        x = args[0]
        if along is not None and along[0][0] == len(zeros):
            (lay, h), a = along
            x = x.clone()
            x[..., h * d : (h + 1) * d] += a * (means[lay, h].to(x.device, x.dtype) - x[..., h * d : (h + 1) * d])
        zeros.append(torch.zeros_like(args[0], requires_grad=True))
        seen.append(args[0].detach())
        return (x + zeros[-1],)

    handles = [b.attn.o_proj.register_forward_pre_hook(add) for b in blocks]
    try:
        res = net.net(w[:, :n].to(net.device))
    finally:
        for h in handles:
            h.remove()
    logits = res[0] if isinstance(res, tuple) else res
    grads = torch.autograd.grad(ST.logodds_at(logits, half, w[:, -1]).sum(), zeros)
    dots = [g.double() * (means[lay].flatten().to(x.device, x.dtype) - x).double() for lay, (g, x) in enumerate(zip(grads, seen, strict=True))]
    return torch.stack([v.view(*v.shape[:2], H, d).sum((0, 1, 3)).cpu() for v in dots])


def head_salience(net: Any, w: Tensor, half: int, n: int, means: Tensor, dW: ST.DW | None = None, batch: int = 64, along: tuple[CP.Head, float] | None = None) -> Tensor:
    """(blocks, H) float64: the first-order change in the mean log-odds at `half` if each head's output went to its mean,
    i.e. the gradient of the log-odds at the head's output times (mean - output), summed over positions, averaged over
    windows. The gradient is total: it runs through every later block. A zero is added to `o_proj`'s input to carry it,
    since the model's parameters need no gradient. `along` = (head, a) takes the gradient with that head's output moved
    the fraction a of the way to its mean, for `head_path`; (mean - output) stays the unmoved one."""
    with net.at(dW):
        return sum(_salience_batch(net, w[s : s + batch], half, n, means, along) for s in range(0, len(w), batch)) / len(w)  # type: ignore[return-value]


def head_path(net: Any, w: Tensor, half: int, n: int, means: Tensor, head: CP.Head, dW: ST.DW | None = None, steps: int = 8, batch: int = 64) -> float:
    """The change in the mean log-odds at `half` as `head`'s output goes in a straight line to its mean: the integral of
    `head_salience`'s entry along that line, by the midpoint rule. It should equal the true change, whatever the curvature."""
    return sum(float(head_salience(net, w, half, n, means, dW, batch, (head, (k + 0.5) / steps))[head]) for k in range(steps)) / steps


def form_gap(net: Any, w: Tensor, half: int, n: int, b: Tensor, edit: Edit | None = None, batch: int = 64) -> Tensor:
    """Per window, log p(target) - log p(b) at `half` under the net's current weights and hooks, E53's A0 read
    (`performed.log_odds`): the difference of the two logits. The target is `w`'s last column."""
    out = []
    for s in range(0, len(w), batch):
        logits = net.run(w[s : s + batch, :n], edit)[0][:, half]
        a, c = w[s : s + batch, -1].to(logits.device), b[s : s + batch].to(logits.device)
        out.append((logits.gather(1, a[:, None]) - logits.gather(1, c[:, None])).squeeze(1).double().cpu())
    return torch.cat(out)


def head_outputs(net: Any, w: Tensor, half: int, n: int, dW: ST.DW | None = None, batch: int = 64) -> Tensor:
    """(blocks, H, N, head_dim) float64 on CPU: each head's output (`o_proj`'s input) at position `half`."""
    blocks = list(net.net.h)
    H, d = blocks[0].attn.n_head, blocks[0].attn.head_dim
    got: list[list[Tensor]] = [[] for _ in blocks]

    def keep(lay: int) -> Any:
        return lambda _m, args: got[lay].append(args[0][:, half].double().cpu())

    handles = [b.attn.o_proj.register_forward_pre_hook(keep(lay)) for lay, b in enumerate(blocks)]
    try:
        with net.at(dW):
            for s in range(0, len(w), batch):
                net.run(w[s : s + batch, :n])
    finally:
        for h in handles:
            h.remove()
    return torch.stack([torch.cat(g).view(len(w), H, d).transpose(0, 1) for g in got])


def ridge_accuracy(x: Tensor, ids: Tensor, E: Tensor) -> float:
    """Fit on the even rows, accuracy on the odd rows. Ridge from `x` (N, d) to the target token's row of `E` (V, e), W0's
    unembedding, since the copy map's read targets barely repeat and a one-hot target could never name an unseen token.
    A prediction names the candidate of highest cosine, the candidates being the distinct targets of all N rows. The
    penalty is the mean eigenvalue of the centred fit features, so it scales with the head; a head that reads constant
    gets chance at best. nan with no scored row."""
    x, y = x.double(), E[ids].double()
    fit, test = torch.arange(len(x))[0::2], torch.arange(len(x))[1::2]
    if not len(test):
        return math.nan
    xm, ym = x[fit].mean(0), y[fit].mean(0)
    X = x[fit] - xm
    XtX = X.T @ X
    pen = max(float(XtX.trace()) / x.shape[1], 1e-12)
    beta = torch.linalg.solve(XtX + pen * torch.eye(x.shape[1], dtype=x.dtype), X.T @ (y[fit] - ym))
    pred = (x[test] - xm) @ beta + ym
    cand = ids.unique()
    cos = torch.nn.functional.normalize(pred, dim=1) @ torch.nn.functional.normalize(E[cand].double(), dim=1).T
    return float((cand[cos.argmax(1)] == ids[test]).double().mean())


def unit_scores(net: Any, w: Tensor, half: int, n: int, dW: ST.DW | None, batch: int = 64) -> Tensor:
    """Jain et al. Alg. 3's gradient times weight per MLP unit, flat (blocks * units,) float64, on the mean log-odds of the
    state rather than `revival_kv.unit_scores`' KL. Removing a unit of large score lowers the log-odds most, to first
    order: the importance sign, not Jain's pruning for revival."""
    lins = [target_linear(net.model, up_key(net.model, i)) for i in range(N_LAYERS)]
    assert all(lin.bias is None for lin in lins), "the 67M's MLP input matrices have no bias"
    with net.at(dW):
        try:
            for lin in lins:
                lin.weight.requires_grad_(True)
                lin.weight.grad = None
            for s in range(0, len(w), batch):
                res = net.net(w[s : s + batch, :n].to(net.device))
                logits = res[0] if isinstance(res, tuple) else res
                (ST.logodds_at(logits, half, w[s : s + batch, -1]).sum() / len(w)).backward()
            scores = [(lin.weight.grad * lin.weight).sum(1) for lin in lins]  # type: ignore[operator]
            return torch.cat(scores).detach().double().cpu()
        finally:
            for lin in lins:
                lin.weight.requires_grad_(False)
                lin.weight.grad = None


def read_edit(net: Any, w: Tensor, half: int, n: int, dW: ST.DW | None, edit: Edit, batch: int = 64) -> Tensor:
    """Per window, the log-odds at `half` of the target model at W0 + dW under `edit` (`copying.read_lo` takes no edit)."""
    with net.at(dW):
        return torch.cat([ST.logodds_at(net.run(w[s : s + batch, :n], edit)[0], half, w[s : s + batch, -1]) for s in range(0, len(w), batch)])


class Edited:
    """`net` under an edit, for `copying.induction`, which reads only `run`."""

    def __init__(self, net: Any, edit: Edit) -> None:
        self.base, self.edit = net, edit

    def run(self, toks: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return self.base.run(toks, self.edit)


# ----------------------------------------------------------------------------- the filed rules


def prob(lo: float) -> float:
    """The probability at a log-odds."""
    return 1 / (1 + math.exp(-lo)) if CP.finite(lo) else math.nan


def apparatus(states: Mapping[str, Mapping[str, float]], zero: Mapping[str, Mapping[str, float]], th: Mapping[str, Any]) -> str:
    """Stage 0. "pass", or what missed. A tie passes; nan misses. The zero update's salience is held to the repeat floor of
    W0's salience on the same path, measured in the same job."""
    if not states or not zero:
        return "nothing read"
    if not all(abs(r["G"] - r["G_run"]) <= th["g_tol"] for r in states.values()):
        return "G does not re-read run.json"
    if not all(z["zero_G_max_abs"] == 0.0 for z in zero.values()):
        return "the zero update's gain is not 0"
    if not all(z["zero_salience_max_abs"] <= z["salience_repeat_floor"] for z in zero.values()):
        return "the zero update's head salience is over the repeat floor"
    return "pass"


def ranked(scores: Mapping[str, float]) -> list[str]:
    """Names by score, highest first, finite only; ties in the mapping's order."""
    return sorted((k for k, v in scores.items() if CP.finite(v)), key=lambda k: -scores[k])


def top_head(scores: Mapping[str, float]) -> str | None:
    """The one highest finite score. None if nothing is finite or the top is tied: a tie names no part."""
    order = ranked(scores)
    if not order or (len(order) > 1 and scores[order[0]] == scores[order[1]]):
        return None
    return order[0]


def rank_of(name: str | None, scores: Mapping[str, float]) -> int | None:
    """1-based rank; a tie counts against the name (every head at accuracy 0 ranks last). None if unranked."""
    if name is None or not CP.finite(scores.get(name)):
        return None
    return sum(CP.finite(v) and v >= scores[name] for v in scores.values())


def removes(row: Mapping[str, float], th: Mapping[str, Any]) -> bool:
    """The part takes the state's probability of the target, at its mean log-odds, to `located_p` or less (a tie counts).
    The state must have a behaviour to take away: its gain at least `min_gain` and its intact probability over
    `located_p`. So the do-nothing state, already at W0's 3e-4, cannot pass. nan fails."""
    return row["G"] >= th["min_gain"] and row["p"] > th["located_p"] and row["p_off"] <= th["located_p"]


def spares(ctl: Mapping[str, float], th: Mapping[str, Any]) -> bool:
    """The part costs the control state a share of `control_max` or less (a tie counts, a negative share counts: the part
    off raised the gain). Under the gain floor, or nan, nothing can be said, so it does not spare."""
    return ctl["G"] >= th["min_gain"] and ctl["share"] <= th["control_max"]


def located(rows: Sequence[Mapping[str, float]], controls: Sequence[Mapping[str, float]], th: Mapping[str, Any]) -> dict[str, Any]:
    """Every seed removes and every control seed spares. No rows or no controls: not located."""
    gone, kept = [removes(r, th) for r in rows], [spares(c, th) for c in controls]
    return {"located": bool(gone) and bool(kept) and all(gone) and all(kept), "removes": gone, "spares": kept}


def capability(top1_off: float, th: Mapping[str, Any]) -> str:
    """W0's second-copy top-1 with the part off, against its base: "shared" at or under shared_frac of it, "gain only" at or
    over gain_only_frac, else "between"; ties to the named side. "gain only" is evidence of distinct parts at this grain."""
    if not CP.finite(top1_off):
        return "unreadable"
    base = th["capability_base"]
    return "shared" if top1_off <= th["shared_frac"] * base else "gain only" if top1_off >= th["gain_only_frac"] * base else "between"


def control(tops: Sequence[str | None], expected: str, rec: Mapping[str, Any] | None, lookup_removes: Sequence[bool]) -> dict[str, Any]:
    """The control: `expected` is the sweep's top head on every copy seed, it is located, and it does not remove the lookup
    (its probability stays over the bar on at least one seed). An empty list of any kind fails."""
    top_ok = bool(tops) and all(t == expected for t in tops)
    loc = rec is not None and bool(rec["located"])
    clear = bool(lookup_removes) and not all(lookup_removes)
    return {"top_is_expected": top_ok, "located": loc, "lookup_clear": clear, "passes": top_ok and loc and clear}


def tree(control_passes: bool, salience_passes: bool) -> str:
    if not control_passes:
        return "sweep misses: stop and report"
    return "both pass: stage 2 with the sweep and head salience" if salience_passes else "sweep passes, salience fails: stage 2 by the sweep alone"


# ----------------------------------------------------------------------------- the score, over saved records


def from_saved(o: Any) -> Any:
    """A record back from JSON: `open_pair.jsonable` writes non-finite floats as strings, and the rules compare numbers."""
    if isinstance(o, dict):
        return {k: from_saved(v) for k, v in o.items()}
    if isinstance(o, list):
        return [from_saved(v) for v in o]
    return float(o) if o in ("nan", "inf", "-inf") else o


def names_of(states: Mapping[str, Any], m: str) -> list[str]:
    return sorted(n for n in states if n.startswith(f"{m}_"))


def part_record(part: str, task: str, rows: Sequence[Mapping[str, float]], controls: Sequence[Mapping[str, float]], top1s: Sequence[float], th: Mapping[str, Any]) -> dict[str, Any]:
    """A part's Located, and the capability reading beside it; seeds that read differently are "disagree"."""
    words = sorted({capability(t, th) for t in top1s})
    reading = words[0] if len(words) == 1 else "disagree" if words else "unreadable"
    rec = {**located(rows, controls, th), "top1_off": list(top1s), "reading": reading}
    if rec["located"]:
        rec["claim"] = f"{part} is located on the {task} map, capability reading {reading}: {LIMIT}"
    return rec


def table_parts(states: Mapping[str, Any], parts: Mapping[str, Mapping[str, Any]], top1: Mapping[str, float], task: str, other: str, th: Mapping[str, Any]) -> dict[str, Any]:
    """Located for every part of a table: part -> state -> its reading with the part off."""
    row = lambda n, p: {"G": states[n]["G"], "p": states[n]["p"], "p_off": parts[p][n]["p"]}
    ctl = lambda n, p: {"G": states[n]["G"], "share": parts[p][n]["share"]}
    return {p: part_record(p, task, [row(n, p) for n in names_of(states, task)], [ctl(n, p) for n in names_of(states, other)], [top1[p]], th) for p in parts}


def nominator_verdicts(noms: Mapping[str, Any], tops: Mapping[str, str | None], th: Mapping[str, Any]) -> dict[str, Any]:
    """Rank of each state's sweep top head under head salience and the probe. Salience passes when it is in the top
    `salience_top` on every copy seed. MLP units: the share against its random sets."""
    n_top = th["salience_top"]
    sal = {n: rank_of(t, noms["salience"][n]) for n, t in tops.items()}
    prb = {n: rank_of(t, noms["probe"][n]) for n, t in tops.items()}
    copy = [sal[n] for n in names_of(tops, "copy")]
    mlp = {}
    for n, r in noms["mlp"].items():
        rnd = [x for x in r["random_shares"] if CP.finite(x)]
        mlp[n] = {"share": r["share"], "random_mean": sum(rnd) / len(rnd) if rnd else math.nan, "random_max": max(rnd, default=math.nan),
                  "over_every_random": CP.finite(r["share"]) and bool(rnd) and r["share"] > max(rnd)}  # fmt: skip
    return {"salience_rank": sal, "probe_rank": prb, "mlp": mlp, "salience_top": {n: top_head(noms["salience"][n]) for n in tops},
            "salience_passes": bool(copy) and all(r is not None and r <= n_top for r in copy),
            "probe_lean_held": {n: r is None or r > n_top for n, r in prb.items() if n.startswith("copy_")}}  # fmt: skip


def in_band(x: Any, band: Sequence[float]) -> bool:
    """To the plan's two decimals, which is how the bands were written."""
    return CP.finite(x) and band[0] <= round(x, 2) <= band[1]


def expected_table(states: Mapping[str, Any], pairs: Mapping[str, Any], T: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's table, read against its values. Reported; no rule reads it."""
    copy, look = names_of(states, "copy"), names_of(states, "lookup")
    shares = {n: {h: r["share"] for h, r in states[n]["heads"].items()} for n in states}
    order = {n: ranked(shares[n]) for n in states}
    pick = lambda names, f: [f(n) if order[n] else math.nan for n in names]
    pair = lambda n: pairs[states[n]["best_pair"]][n]["share"]
    reads: dict[str, list[Any]] = {
        "copy_top_share": pick(copy, lambda n: shares[n][order[n][0]]),
        "copy_second_share": pick(copy, lambda n: shares[n][order[n][1]] if len(order[n]) > 1 else math.nan),
        "copy_pair_share": pick(copy, pair),
        "copy_p_top_off": pick(copy, lambda n: states[n]["heads"][order[n][0]]["p"]),
        "lookup_top_share": pick(look, lambda n: shares[n][order[n][0]]),
        "lookup_named_share": [shares[n].get(T["named_head"], math.nan) for n in look],
        "lookup_pair_share": pick(look, pair),
        "lookup_p_top_off": pick(look, lambda n: states[n]["heads"][order[n][0]]["p"]),
    }
    out: dict[str, Any] = {k: {"read": v, "band": T[k], "as_expected": bool(v) and all(in_band(x, T[k]) for x in v)} for k, v in reads.items()}
    second = pick(copy, lambda n: order[n][1] if len(order[n]) > 1 else "none")
    out["copy_second_head"] = {"read": second, "expected": T["named_head"], "as_expected": bool(second) and all(s == T["named_head"] for s in second)}
    return out


def score(noms: Mapping[str, Any], sweep: Mapping[str, Any], run_shares: Mapping[str, float], th: Mapping[str, Any], T: Mapping[str, Any], expected: str) -> dict[str, Any]:
    """Arithmetic over the saved nominators and sweep. `expected` is the filed head; only this function's caller knows it."""
    st, top1 = sweep["states"], sweep["w0"]["top1_off"]
    heads = {h: {n: st[n]["heads"][h] for n in st} for h in top1}
    pairs = {p: rec["states"] for p, rec in sweep["pairs"].items()}
    pair_top1 = {p: rec["top1_off"] for p, rec in sweep["pairs"].items()}
    tops = {n: top_head({h: r["share"] for h, r in st[n]["heads"].items()}) for n in st}
    tables = {t: {"heads": table_parts(st, heads, top1, t, o, th), "pairs": table_parts(st, pairs, pair_top1, t, o, th)} for t, o in (("copy", "lookup"), ("lookup", "copy"))}
    mlp = {}
    for t in ("copy", "lookup"):
        recs = [noms["mlp"][n] for n in names_of(st, t)]
        mlp[t] = part_record(f"the top MLP units of each {t} state", t, recs, [r["control"] for r in recs], [r["top1_off"] for r in recs], th)
    look = [removes({"G": st[n]["G"], "p": st[n]["p"], "p_off": heads[expected][n]["p"]}, th) for n in names_of(st, "lookup")] if expected in heads else []
    ctl = control([tops[n] for n in names_of(st, "copy")], expected, tables["copy"]["heads"].get(expected), look)
    verdicts = nominator_verdicts(noms, tops, th)
    located_parts = [r["claim"] for t in tables.values() for g in t.values() for r in g.values() if r["located"]] + [r["claim"] for r in mlp.values() if r["located"]]
    return {"tops": tops, "control": ctl, "expected_reading": tables["copy"]["heads"].get(expected, {}).get("reading", "unreadable"),
            "located": tables, "mlp_located": mlp, "claims": located_parts, "nominators": verdicts, "table": expected_table(st, pairs, T),
            "share_diff_run": {n: abs(heads[expected][n]["share"] - s) for n, s in run_shares.items()} if expected in heads else {},
            "outcome": tree(ctl["passes"], verdicts["salience_passes"])}  # fmt: skip


# ----------------------------------------------------------------------------- stage 2: a task W0 already performs


def gain_apparatus(states: Mapping[str, Mapping[str, float]], zero: Mapping[str, float], th: Mapping[str, Any]) -> str:
    """Stage 2's stage 0. "pass", or what missed; a tie passes, nan misses, an empty read misses."""
    if not states or not zero:
        return "nothing read"
    if not all(abs(r["G_bf16"] - r["G_stored"]) <= th["g_tol"] for r in states.values()):
        return "G does not re-read E53's stored value"
    return "pass" if all(abs(z) <= th["zero_tol"] for z in zero.values()) else "the zero update's gain is not 0"


def path_outcome(true_fall: float, path_fall: float, single_fall: float, tol: float) -> str:
    """The gradient check, for one read. The path integral is held to the true fall first, so a single step can only be
    called misleading on a gradient that integrates correctly. A tie matches. A true fall of 0 matches only exactly."""
    if not all(CP.finite(x) for x in (true_fall, path_fall, single_fall)):
        return "unreadable"
    if abs(path_fall - true_fall) > tol * abs(true_fall):
        return "gradient code wrong: the path integral misses the true fall"
    if abs(single_fall - true_fall) > tol * abs(true_fall):
        return "first-order step misleading: the path integral matches, the single step does not"
    return "both match: the path integral and the single step are both within the tolerance"


def path_summary(outcomes: Sequence[str]) -> str:
    """One outcome for the check: the common one, else "mixed", and the reader goes to the cases."""
    words = sorted(set(outcomes))
    return words[0] if len(words) == 1 else "mixed: read the cases" if words else "unreadable"


def net_located(rows: Sequence[Mapping[str, float]], controls: Sequence[Mapping[str, float]], th: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's rule for a task W0 already performs. On every seed the state's gain is at least `min_gain` and its net
    share lost at least `located_share` (a tie counts; a share past 1 counts: the gain went and more), and the same seed's
    control state is spared (`spares`). A seed that disagrees, no rows, no controls or nan: not located."""
    gone = [r["G"] >= th["min_gain"] and r["share"] >= th["located_share"] for r in rows]
    kept = [spares(c, th) for c in controls]
    return {"located": bool(gone) and len(gone) == len(kept) and all(gone) and all(kept), "lost": gone, "spared": kept}


def case_capability(off: float, intact: float, th: Mapping[str, Any]) -> str:
    """W0's own mean log p(lowercase) - log p(capitalised) with the part off, against intact: "shared" at or under
    shared_frac of intact, "gain only" at or over gain_only_frac, else "between"; ties to the named side."""
    if not CP.finite(intact) or intact <= 0:
        return "no capability to read"
    if not CP.finite(off):
        return "unreadable"
    return "shared" if off <= th["shared_frac"] * intact else "gain only" if off >= th["gain_only_frac"] * intact else "between"


NOT_FOUND_SCOPE = "not found at single-head or best-pair grain, K = 320 MLP units reported beside; not 'no mechanism'"
GAIN_ONLY = "the gain and the capability have separate parts at this grain"


def stage2_outcome(gains: Sequence[float], recs: Mapping[str, Mapping[str, Any]], min_gain: float) -> dict[str, Any]:
    """The filed strings. Any lowercase seed under the floor, nan or none read: "no gain to locate", whatever the parts say."""
    if not gains or not all(g >= min_gain for g in gains):
        return {"outcome": "no gain to locate", "claims": []}
    found = [p for p, r in recs.items() if r["located"]]
    if not found:
        return {"outcome": "not found at head or pair grain", "claims": [NOT_FOUND_SCOPE]}
    claims = []
    for p in found:
        word = recs[p]["reading"]
        claims.append(f"{p} is located on lowercase, capability reading {word}: {LIMIT}" + (f"; {GAIN_ONLY}" if word == "gain only" else ""))
    return {"outcome": f"located: {', '.join(found)}", "claims": claims}


def stage2_beside(beside: Mapping[str, Any], tops: Mapping[str, str | None], gap_intact: float, th: Mapping[str, Any]) -> dict[str, Any]:
    """Reported, deciding nothing: the rank of each lowercase state's top head by salience and by the probe, and the MLP
    units' share against their random sets."""
    mlp = {}
    for n, r in beside["mlp"].items():
        rnd = [x for x in r["random_shares"] if CP.finite(x)]
        mlp[n] = {"share": r["share"], "control_share": r["control"]["share"], "random_mean": sum(rnd) / len(rnd) if rnd else math.nan, "random_max": max(rnd, default=math.nan),
                  "over_every_random": CP.finite(r["share"]) and bool(rnd) and r["share"] > max(rnd), "reading": case_capability(r["gap_off"], r["gap_intact"], th)}  # fmt: skip
    return {"salience_rank": {n: rank_of(tops[n], beside["salience"][n]) for n in beside["salience"]}, "probe_rank": {n: rank_of(tops[n], beside["probe"][n]) for n in beside["salience"]},
            "gap_intact_all_rows": gap_intact, "mlp": mlp}  # fmt: skip


def score_stage2(sweep: Mapping[str, Any], beside: Mapping[str, Any], th: Mapping[str, Any]) -> dict[str, Any]:
    """Arithmetic over the saved sweep and beside records. Parts: every head alone and each state's greedy pair, in that
    order. A share is read on the task's own positions: lowercase's state on lowercase's, the same seed's cipher on cipher's."""
    st, w0 = sweep["states"], sweep["w0"]
    low, cip = names_of(st, "lowercase"), names_of(st, "cipher")
    parts = {h: {n: st[n]["heads"][h] for n in st} for h in w0["gap_off"]} | {p: r["states"] for p, r in sweep["pairs"].items()}
    gaps = {**w0["gap_off"], **{p: r["gap_off"] for p, r in sweep["pairs"].items()}}
    recs: dict[str, dict[str, Any]] = {}
    for p, by_state in parts.items():
        rows = [{"G": st[n]["G"], "share": by_state[n]["share"]} for n in low]
        ctls = [{"G": st[n]["G"], "share": by_state[n]["share"]} for n in cip]
        recs[p] = {**net_located(rows, ctls, th), "shares": [r["share"] for r in rows], "control_shares": [c["share"] for c in ctls],
                   "gap_off": gaps[p], "reading": case_capability(gaps[p], w0["gap"], th)}  # fmt: skip
    tops = {n: top_head({h: r["share"] for h, r in st[n]["heads"].items()}) for n in st}
    single = [h for h in w0["gap_off"] if recs[h]["located"]]
    return {**stage2_outcome([st[n]["G"] for n in low], recs, th["min_gain"]), "gains": {n: st[n]["G"] for n in st}, "tops": tops, "parts": recs,
            "w0_capability": {"intact": w0["gap"], "readable": case_capability(0.0, w0["gap"], th) != "no capability to read"},
            "lean_no_single_head_located_held": not single, "beside": stage2_beside(beside, tops, w0["gap"], th)}  # fmt: skip

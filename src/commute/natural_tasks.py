"""E124's natural tasks (plan reports/plans/2026-11-06_plan_e124-natural-definition-tasks.md, appendix §1-2): the
agreement and pronoun windows cut from Pile rows, their pools, the name list and the probe positions. No model is read
here; tests/test_natural_tasks.py runs the pieces on hand-made rows.

A task's matches are one record per (row, t), t the prediction position (the last token the model sees): `content` the
token the property C is read from (the subject noun N, or the name), `c` = C (1: singular / male, the answer word A,
" is" / " he"), `kind` (agreement: 1 when N2's number differs from N's, an attractor; pronoun: C itself) and `next` the
text's token at t + 1 (the natural member's answer).
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor

PREPS = (" of", " in", " on", " for", " with", " from", " at", " to", " near", " behind", " about", " under")
DETS, ARTS = (" the", " The"), (" the", " a", " an")
Rec = dict[str, Tensor]  # row, t, content, c, kind, next: (n,) int64 each


def single_ids(encode: Callable[[str], list[int]], words: Sequence[str]) -> list[int]:
    ids = [encode(w) for w in words]
    assert all(len(i) == 1 for i in ids), f"not single tokens: {[w for w, i in zip(words, ids, strict=True) if len(i) != 1]}"
    return [i[0] for i in ids]


def number_pairs(decode: Callable[[list[int]], str], vocab: int, exclude: set[int]) -> list[tuple[int, int]]:
    """Every (" w", " ws") or (" w", " wes") pair of single lowercase-letter tokens, (singular, plural), " s" before
    " es". A pair with a token in `exclude`, or a token in two pairs (" pa", " pas", " pases"), is out."""
    ids: dict[str, int] = {}
    for t in range(vocab):
        try:
            s = decode([t])
        except (IndexError, KeyError, ValueError, TypeError):
            continue
        if len(s) >= 2 and s[0] == " " and s[1:].isalpha() and s[1:].islower():
            ids.setdefault(s, t)
    pairs = [(t, ids[s + suf]) for s, t in ids.items() for suf in ("s", "es") if s + suf in ids]
    pairs = list({p[0]: p for p in reversed(pairs)}.values())  # " s" wins over " es" for the same singular
    seen: dict[int, int] = {}
    for p in pairs:
        for t in p:
            seen[t] = seen.get(t, 0) + 1
    return sorted(p for p in pairs if not set(p) & exclude and all(seen[t] == 1 for t in p))


def _at(rows: Tensor, lo: int, hi: int, k: int, ids: Tensor) -> Tensor:
    """(R, hi - lo) bool: rows[:, t - k] in ids for t in [lo, hi)."""
    return torch.isin(rows[:, lo - k : hi - k], ids)


def agreement_matches(rows: Tensor, nouns: Tensor, n2s: Tensor, dets: Tensor, preps: Tensor, arts: Tensor, half: int,
                      nexts: Tensor | None) -> dict[str, Tensor]:  # fmt: skip
    """(row, t, n, n2) of "DET N PREP N2" and "DET N PREP ART N2" with N2 at t, t in [half, T - 1) (a token follows t),
    N in `nouns`, N2 in `n2s`, and the token at t + 1 in `nexts` (None: any)."""
    lo, hi = half, rows.shape[1] - 1
    nxt = torch.ones(rows.shape[0], hi - lo, dtype=torch.bool) if nexts is None else _at(rows, lo, hi, -1, nexts)
    end = _at(rows, lo, hi, 0, n2s) & nxt
    a = end & _at(rows, lo, hi, 1, preps) & _at(rows, lo, hi, 2, nouns) & _at(rows, lo, hi, 3, dets)
    b = end & _at(rows, lo, hi, 1, arts) & _at(rows, lo, hi, 2, preps) & _at(rows, lo, hi, 3, nouns) & _at(rows, lo, hi, 4, dets)
    out = []
    for m, k in ((a, 2), (b, 3)):
        r, j = torch.nonzero(m, as_tuple=True)
        t = j + lo
        out.append(torch.stack([r, t, rows[r, t - k], rows[r, t]], 1))
    x = torch.cat(out)
    x = x[torch.argsort(x[:, 0] * rows.shape[1] + x[:, 1])]
    return {"row": x[:, 0], "t": x[:, 1], "n": x[:, 2], "n2": x[:, 3]}


def pronoun_matches(rows: Tensor, names: Tensor, near: int, far: int, half: int, nexts: Tensor | None) -> dict[str, Tensor]:
    """(row, t, name) where the only pool name in [t - far, t] sits in [t - far, t - near], t in [max(half, far), T - 1),
    and the token at t + 1 is in `nexts` (None: any)."""
    nm = torch.isin(rows, names).long()
    cs = torch.cat([torch.zeros(rows.shape[0], 1, dtype=torch.long), nm.cumsum(1)], 1)  # cs[:, j] = names in [0, j)
    lo, hi = max(half, far), rows.shape[1] - 1
    t = torch.arange(lo, hi)
    total = cs[:, t + 1] - cs[:, t - far]
    recent = cs[:, t + 1] - cs[:, t - near + 1]
    ok = (total == 1) & (recent == 0)
    if nexts is not None:
        ok &= torch.isin(rows[:, lo + 1 : hi + 1], nexts)
    r, j = torch.nonzero(ok, as_tuple=True)
    tt = j + lo
    span = tt[:, None] - far + torch.arange(far - near + 1)
    p = tt - far + nm[r[:, None], span].argmax(1)
    return {"row": r, "t": tt, "name": rows[r, p]}


def agreement_records(m: Mapping[str, Tensor], rows: Tensor, singular: Mapping[int, int]) -> Rec:
    """Matches to records: c = 1 when N is singular, kind = 1 when N2's number differs from N's (an attractor)."""
    c = torch.tensor([singular[int(n)] for n in m["n"]], dtype=torch.long)
    c2 = torch.tensor([singular[int(n)] for n in m["n2"]], dtype=torch.long)
    return {"row": m["row"], "t": m["t"], "content": m["n"], "c": c, "kind": (c != c2).long(), "next": rows[m["row"], m["t"] + 1]}


def pronoun_records(m: Mapping[str, Tensor], rows: Tensor, male: Mapping[int, int]) -> Rec:
    c = torch.tensor([male[int(n)] for n in m["name"]], dtype=torch.long)
    return {"row": m["row"], "t": m["t"], "content": m["name"], "c": c, "kind": c.clone(), "next": rows[m["row"], m["t"] + 1]}


def subset(rec: Rec, keep: Tensor) -> Rec:
    return {k: v[keep] for k, v in rec.items()}


def agree_mask(rec: Rec, a: int, b: int) -> Tensor:
    """The drop rule: True where the text's next token is the answer C gives (a when c = 1, b when c = 0)."""
    return torch.where(rec["c"] == 1, rec["next"] == a, rec["next"] == b)


def drop_shares(rec: Rec, a: int, b: int) -> dict[str, float]:
    """The share of matches the drop rule removes, per kind and overall (nan for an empty kind)."""
    bad = ~agree_mask(rec, a, b)
    out = {f"kind{k}": float(bad[rec["kind"] == k].double().mean()) if bool((rec["kind"] == k).any()) else float("nan") for k in (0, 1)}
    return {**out, "all": float(bad.double().mean()) if len(bad) else float("nan")}


def sample_cells(rec: Rec, cells: Sequence[tuple[int, int]], n: int, seed: int) -> tuple[Rec, dict[str, int]]:
    """n records, n / len(cells) from each (c, kind) cell, drawn seeded without replacement, in row order. Returns the
    sample and each cell's available count; a short cell gives what it has (the caller's count gate reads it)."""
    gen = torch.Generator().manual_seed(seed)
    each, picks, avail = n // len(cells), [], {}
    assert each * len(cells) == n, f"{n} windows do not split over {len(cells)} cells"
    for c, k in cells:
        idx = torch.nonzero((rec["c"] == c) & (rec["kind"] == k), as_tuple=True)[0]
        avail[f"c{c}_kind{k}"] = len(idx)
        picks.append(idx[torch.randperm(len(idx), generator=gen)[:each]])
    keep = torch.cat(picks).sort().values
    return subset(rec, keep), avail


def windows(rows: Tensor, rec: Rec, half: int, extra: int = 0) -> Tensor:
    """(n, half + 1 + extra) tokens rows[row, t - half : t + 1 + extra]: the prediction position is the column `half`;
    `extra` = 1 adds the text's next token as a last column (for the next-token loss)."""
    return rows[rec["row"][:, None], rec["t"][:, None] - half + torch.arange(half + 1 + extra)]


def cap_and_balance(rec: Rec, group: Tensor, cap: int, cells: Sequence[tuple[int, int]], seed: int) -> Rec:
    """Probe positions: at most `cap` per group id (`case_label.cap_groups`), then each (c, kind) cell cut, seeded, to
    the smallest cell's size, so C is balanced against kind (agreement: N2's number carries no information on C)."""
    from commute.case_label import cap_groups

    rec = subset(rec, cap_groups(group, cap, seed))
    n = min(int(((rec["c"] == c) & (rec["kind"] == k)).sum()) for c, k in cells)
    out, _ = sample_cells(rec, cells, n * len(cells), seed + 1)
    return out


def ssa_names(path: Path, years: tuple[int, int], share: float) -> dict[str, int]:
    """Names from Hadley Wickham's SSA extract (columns year, name, percent, sex): percent summed over the years
    (inclusive) per sex; a name whose larger side holds >= `share` of its sum gets 1 (boy) or 0 (girl), else is out."""
    tot: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if years[0] <= int(row["year"]) <= years[1]:
                s = tot.setdefault(row["name"], [0.0, 0.0])
                s[row["sex"] == "boy"] += float(row["percent"])
    return {n: int(b >= g) for n, (g, b) in tot.items() if max(g, b) >= share * (g + b) and g + b > 0}


def name_pool(names: Mapping[str, int], encode: Callable[[str], list[int]], counts: Mapping[int, int], n: int,
              exclude: set[int]) -> dict[int, int]:  # fmt: skip
    """{token: male} for the n names whose " Name" is one token, most frequent in `counts` first (ties to the lower
    id), absent and excluded tokens out. Two names on one token (none expected) keep the first by frequency."""
    toks: dict[int, int] = {}
    for name, male in names.items():
        ids = encode(" " + name)
        if len(ids) == 1 and ids[0] not in exclude and counts.get(ids[0], 0) > 0:
            toks.setdefault(ids[0], male)
    ranked = sorted(toks, key=lambda t: (-counts[t], t))[:n]
    return {t: toks[t] for t in ranked}


def split_within(groups: Sequence[Sequence[int]], frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Per group (e.g. one gender), `case_label.split_pairs` on one-token "pairs" with round(frac x size) to train."""
    from commute.case_label import split_pairs

    tr, he = [], []
    for g in groups:
        a, b = split_pairs([(t, t) for t in g], round(frac * len(g)), seed)  # a name is a pair of itself
        tr += [p[0] for p in a]
        he += [p[0] for p in b]
    return tr, he


def scored_step(acc: Mapping[int, float], gain: Mapping[int, float], checkpoints: Sequence[int], bar: float,
                min_gain: float) -> tuple[int, bool]:  # fmt: skip
    """E124's state selection (Fable's change 1): the first checkpoint with train accuracy >= bar (E112's learn bar)
    and full-vocabulary log-odds gain >= min_gain (local_faith.MIN_GAIN), both inclusive; else the last, not learned."""
    ok = lambda s: s in acc and acc[s] >= bar and gain[s] >= min_gain
    return next(((s, True) for s in checkpoints if ok(s)), (checkpoints[-1], False))


def replicates(r_nat: Sequence[float], r_new: Sequence[float], cut: float, min_gap: float) -> dict[str, object]:
    """The filed re-gating reading of one task: `elicit.separation` (gap over the larger seed range, >= min_gap) on the
    seeds' r, plus natural r >= cut and new-word r < cut on every seed. A nan r or a missing seed: "no call"."""
    from commute.elicit import separation

    sep = separation(list(r_nat), list(r_new), min_gap)
    if sep["verdict"] == "unreadable":
        return {"reading": "no call", **sep}
    ok = sep["verdict"] == "a_higher" and all(r >= cut for r in r_nat) and all(r < cut for r in r_new)
    return {"reading": "replicates" if ok else "not replicated", **sep}

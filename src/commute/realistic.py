"""Real-text tasks for cell P2.38 (plan reports/plans/2026-11-18_plan_p2-38-run1.md): the data, and run 1's filed drop
rules. No model is read here; tests/test_realistic.py runs every piece on hand-made inputs with a stand-in tokenizer.

Label tasks (one format, so a pair shares texts, marker and label tokens): a text's first `n` tokens (left-padded with
end-of-text when shorter), then the marker " Topic:", and the label token read at the marker's last position as a last
column the model never sees (data.LAST).
  L1  AG News Sports (y = 1, " Sports") vs Business (y = 0, " Business")
  L2  SST-2 positive (y = 1, " positive") vs negative (y = 0, " negative")
  L3  L1's texts, y a seeded coin per text: new by construction, so it can only be learned on the texts trained on
LM tasks (rows of 512 tokens, texts joined by end-of-text and cut, as row_copy.lowercase_rows):
  T1  E53's lowercase rows (commute.lowercase, unchanged)
  T2  French Wikipedia; T3 the absent language (Swahili, Tagalog or Yoruba Wikipedia) chosen by the filed rule
  T4  Pile rows with the word order of every sentence reversed
A language's training rows come first in the dump's article order and its held-out rows after, from different articles
(the first articles of the small wikis are alphabet and meta pages, which a dry run showed would fill the held rows).
FLORES+ (the draft plan's scored sentences) is gated on HuggingFace, so the scored rows are held-out Wikipedia articles.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

AG = ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4")
SST2 = ("stanfordnlp/sst2", "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb")
WIKI = ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101")
MARKER = " Topic:"
LABEL_WORDS = {"L1": (" Sports", " Business"), "L2": (" positive", " negative"), "L3": (" Sports", " Business")}
AG_SPORTS, AG_BUSINESS = 1, 2  # AG News label ids (0 World, 1 Sports, 2 Business, 3 Sci/Tech)
SENTENCE = re.compile(r"(?<=[.!?])\s+")


# ----------------------------------------------------------------------------- label windows


def window(ids: Sequence[int], n: int, pad: int) -> list[int]:
    """The first n ids, left-padded with `pad` when there are fewer."""
    ids = list(ids[:n])
    return [pad] * (n - len(ids)) + ids


def label_windows(texts: Sequence[str], encode: Callable[[str], list[int]], n: int, pad: int) -> Tensor:
    """(N, n + len(marker)) windows: each text's `window`, then the marker. The read position is the last column."""
    marker = encode(MARKER)
    return torch.tensor([window(encode(t.strip()), n, pad) + marker for t in texts], dtype=torch.long)


def balanced(y: Tensor, per_class: Sequence[int], seed: int) -> list[Tensor]:
    """Disjoint index sets, part i holding per_class[i] indices of each class (y in {0, 1}), seeded, in shuffled order."""
    gen = torch.Generator().manual_seed(seed)
    by = [torch.nonzero(y == c).flatten() for c in (0, 1)]
    by = [b[torch.randperm(len(b), generator=gen)] for b in by]
    assert all(len(b) >= sum(per_class) for b in by), f"classes hold {[len(b) for b in by]}, {sum(per_class)} needed"
    parts, at = [], 0
    for k in per_class:
        idx = torch.cat([b[at : at + k] for b in by])
        parts.append(idx[torch.randperm(len(idx), generator=gen)])
        at += k
    return parts


def coin(n: int, seed: int) -> Tensor:
    """(n,) seeded fair coin labels in {0, 1}: L3's label of each text."""
    return torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(seed))


def ag_news(split: str) -> tuple[list[str], Tensor]:
    """AG News Sports and Business texts of a split and y (1 Sports, 0 Business)."""
    from datasets import load_dataset

    ds = load_dataset(AG[0], split=split, revision=AG[1])
    keep = [(t, int(c == AG_SPORTS)) for t, c in zip(ds["text"], ds["label"], strict=True) if c in (AG_SPORTS, AG_BUSINESS)]
    return [t for t, _ in keep], torch.tensor([c for _, c in keep])


def sst2(split: str, encode: Callable[[str], list[int]], min_tokens: int) -> tuple[list[str], Tensor]:
    """SST-2 sentences of a split with at least `min_tokens` tokens, and y (1 positive, 0 negative)."""
    from datasets import load_dataset

    ds = load_dataset(SST2[0], split=split, revision=SST2[1])
    keep = [(t, int(c)) for t, c in zip(ds["sentence"], ds["label"], strict=True) if len(encode(t.strip())) >= min_tokens]
    return [t for t, _ in keep], torch.tensor([c for _, c in keep])


# ----------------------------------------------------------------------------- LM rows


def pack(token_lists: Iterable[list[int]], eos: int, counts: Sequence[int], T: int) -> list[Tensor]:
    """Texts joined by end-of-text and cut into rows of T, filling part 0 (counts[0] rows), then part 1, and so on. The
    text that fills a part is cut there and its rest dropped, so no text has rows in two parts."""
    out, buf, it = [], [], iter(token_lists)
    for n in counts:
        buf = []
        while len(buf) < n * T:
            ids = next(it, None)
            if ids is None:
                raise ValueError(f"texts ran out at {len(buf) // T} of {n} rows (part {len(out)})")
            buf += ids + [eos]
        out.append(torch.tensor(buf[: n * T], dtype=torch.long).view(n, T))
    return out


def reorder(text: str, reverse: bool) -> str:
    """The text with whitespace collapsed to single spaces and, if `reverse`, each sentence's words in reverse order
    (sentences end at . ! or ? followed by whitespace; punctuation stays on its word). reverse=False is T4's matched
    reference: the same text, the same whitespace, the original order."""
    sents = [s.split() for s in SENTENCE.split(text)]
    return " ".join(" ".join(reversed(w) if reverse else w) for w in sents if w)


def transformed_rows(rows: Tensor, tok: Any, counts: Sequence[int], fn: Callable[[str], str]) -> list[Tensor]:
    """Token rows decoded, passed through fn, re-encoded and packed (`pack`) at the rows' own length."""
    return pack((tok.encode(fn(tok.decode(r.tolist()))) for r in rows), tok.eos_token_id, counts, rows.shape[1])


def wiki_texts(lang: str) -> Iterator[str]:
    """A Wikipedia language's articles in the dump's order (streamed; nothing past what is read is downloaded)."""
    from datasets import load_dataset

    for ex in load_dataset(WIKI[0], f"{WIKI[2]}.{lang}", split="train", streaming=True, revision=WIKI[1]):
        yield ex["text"]


def byte_lengths(pieces: Sequence[str], special: Iterable[int], vocab: int) -> Tensor:
    """(vocab,) UTF-8 bytes per token id: a byte-level BPE piece holds one character per byte; special ids and ids past
    the tokenizer's count 0, so an end-of-text target adds loss but no bytes."""
    out = torch.zeros(vocab, dtype=torch.long)
    out[: len(pieces)] = torch.tensor([len(p) for p in pieces[:vocab]], dtype=torch.long)
    out[list(special)] = 0
    return out


def word_starts(pieces: Sequence[str], vocab: int) -> Tensor:
    """(vocab,) True where the byte-level piece is a space ("Ġ") then a letter: the first token of a word."""
    out = torch.zeros(vocab, dtype=torch.bool)
    out[: len(pieces)] = torch.tensor([len(p) > 1 and p[0] == "Ġ" and p[1].isalpha() for p in pieces[:vocab]])
    return out


# ----------------------------------------------------------------------------- run 1's filed rules


def premise_calls(p: Mapping[str, Any], th: Mapping[str, Any]) -> dict[str, Any]:
    """Stage 1's rule. p: best-block probe accuracy per label task ("probe": {L1, L2, L3}), W0 nats per byte per language
    ("bytes": {fr, sw, tl, yo}), W0 nats per token on T4's rows and their unreversed reference ("t4", "t4_ref"). T3 is the
    candidate with the highest nats per byte (ties to the first listed). A pair is dropped when either member's premise
    fails; L2 and T1 stand alone (T1's premise is E53's A0, read before)."""
    cand = list(th["absent_candidates"])
    t3 = max(cand, key=lambda c: (_num(p["bytes"][c]), -cand.index(c)))
    ok = {"L1": _num(p["probe"]["L1"]) >= th["l1_probe_min"], "L3": _num(p["probe"]["L3"], math.inf) <= th["l3_probe_max"],
          "L2": _num(p["probe"]["L2"]) >= th["l2_probe_min"],
          "T3": _num(p["bytes"][t3]) >= th["t3_over_fr"] * _num(p["bytes"]["fr"], math.inf),
          "T4": _num(p["t4"]) - _num(p["t4_ref"], math.inf) >= th["t4_gap"]}  # fmt: skip
    ok["T2"] = ok["T3"]
    pair = {"L1": "L3", "L3": "L1", "T2": "T3", "T3": "T2"}
    kept = [t for t in ("T1", "L1", "L2", "L3", "T2", "T3", "T4") if t == "T1" or (ok[t] and ok.get(pair.get(t, t), True))]
    return {"T3_language": t3, "premise_ok": ok, "kept": kept}


def _num(x: Any, nan_as: float = -math.inf) -> float:
    """A reading as a float, with NaN or None read as `nan_as` (so a missing reading fails its bar)."""
    return nan_as if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def grid_calls(grid: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]], bars: Mapping[str, float],
               size_cap: float) -> dict[str, Any]:  # fmt: skip
    """Stage 2's rule. grid[task][rate][step] = {"size", "gain"}. A task is kept when its largest gain at any checkpoint
    of size at most size_cap, over every rate, reaches its bar (bars[task]). The rate proposed for run 2 is the one
    holding that largest gain (ties to the lower rate). A task with no checkpoint at or under the cap is dropped."""
    out: dict[str, Any] = {}
    for task, rates in grid.items():
        best = [(_num(r["gain"]), -float(rate), rate, step) for rate, steps in rates.items() for step, r in steps.items()
                if _num(r["size"], math.inf) <= size_cap]  # fmt: skip
        if not best:
            out[task] = {"kept": False, "why": "no checkpoint at or under the size cap", "best_gain": None}
            continue
        g, _, rate, step = max(best)
        out[task] = {"kept": g >= bars[task], "best_gain": None if math.isinf(g) else g, "rate": rate, "step": step,
                     "why": "gain at or over the bar" if g >= bars[task] else "gain under the bar"}  # fmt: skip
    return out


# ----------------------------------------------------------------------------- run 2 (E211): read windows, size, calls


def update_size(dW: Mapping[str, Tensor], W0: Mapping[str, Tensor]) -> float:
    """||dW|| / ||W0|| over the matrices of dW, Frobenius norms summed in squares (E187's; run 1's `size`), float64."""
    num = sum(float(d.double().pow(2).sum()) for d in dW.values())
    return math.sqrt(num / sum(float(W0[k].double().pow(2).sum()) for k in dW))


def hard_positions(lo: Tensor, starts: Tensor, half: int, per_row: int, n: int) -> Tensor:
    """(n, 2) [row, pos] of an LM task's scored tokens, chosen from W0 only. lo (R, T): W0's log-odds of token p from
    the prefix before it (nan where undefined, never chosen); starts (R, T) bool: token p starts a word. A position is a
    candidate when it starts a word and its window of 2 half + 1 fits (half + 1 <= pos <= T - half, as
    `performed.windows_at`). Each row gives its `per_row` candidates of lowest lo, then the n lowest of those are kept.
    Ties go to the earlier row, then the earlier position. Fewer than n candidates raises."""
    T = lo.shape[1]
    pos = torch.arange(T)
    ok = starts & (pos >= half + 1) & (pos <= T - half) & ~lo.isnan()
    cand: list[tuple[float, int, int]] = []
    for r in range(lo.shape[0]):
        row = sorted((float(lo[r, p]), int(p)) for p in pos[ok[r]])[:per_row]
        cand += [(v, r, p) for v, p in row]
    if len(cand) < n:
        raise ValueError(f"{len(cand)} candidate positions, {n} needed")
    return torch.tensor([[r, p] for _, r, p in sorted(cand)[:n]], dtype=torch.long)


def cut_windows(rows: Tensor, at: Tensor, half: int) -> Tensor:
    """(n, 2 half + 1) windows of rows around at's [row, pos]: the scored token at index half + 1, its input at half
    (`performed.windows_at`'s layout, which data.NEXT reads)."""
    start = at[:, 1] - half - 1
    return rows[at[:, 0, None], start[:, None] + torch.arange(2 * half + 1)]


RATIO, INVERTED = {"size", "erank"}, {"k_half"}  # the scorecard's gap conventions (method_table/plot_final.py)


def seed_call(method: str, reuse: Sequence[float | None], new: Sequence[float | None]) -> dict[str, Any]:
    """The scorecard's one rule (`method_table/plot_final.py` `score`) on each side's values, a None or nan (a share
    under its gain floor, a k_half with no carrying set) left out first and counted. "unreadable" when a side keeps no
    value; "one value a side" when a side keeps one; else "apart" iff the two sides' ranges do not overlap (equal ends
    overlap), "no call" otherwise. gap: reuse mean - new mean; log10(reuse / new) for size and erank; log10(new / reuse)
    for k_half (positive = fewer neurons on the reuse side); nan where a ratio's mean is not positive."""
    a, b = _kept(reuse), _kept(new)
    out: dict[str, Any] = {"reuse": a, "new": b, "left_out": {"reuse": len(reuse) - len(a), "new": len(new) - len(b)}}
    if not a or not b:
        return {**out, "gap": math.nan, "call": "unreadable"}
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    if method in RATIO or method in INVERTED:
        num, den = (ma, mb) if method in RATIO else (mb, ma)
        out["gap"] = math.log10(num / den) if num > 0 and den > 0 else math.nan
    else:
        out["gap"] = ma - mb
    if min(len(a), len(b)) < 2:
        return {**out, "call": "one value a side"}
    return {**out, "call": "apart" if min(a) > max(b) or max(a) < min(b) else "no call"}


def _kept(xs: Sequence[float | None]) -> list[float]:
    return [float(x) for x in xs if x is not None and not math.isnan(float(x))]


def filed(c: Mapping[str, Any], direction: str) -> str:
    """A call against its filed direction: "+" (reuse side higher, in the method's gap convention), "-", or "none"
    (expected no call). "hit" / "miss" / "no call"; "unreadable" passes through. One value a side (the W0 prior, no seed
    spread) is judged on the sign of its gap alone."""
    if c["call"] == "unreadable" or math.isnan(c["gap"]):
        return "unreadable"
    if c["call"] == "one value a side":
        return "hit" if direction != "none" and (c["gap"] > 0) == (direction == "+") else "miss"
    if direction == "none":
        return "hit" if c["call"] == "no call" else "miss"
    if c["call"] != "apart":
        return "no call"
    return "hit" if (c["gap"] > 0) == (direction == "+") else "miss"


def rival(share: Mapping[str, Mapping[str, Any]], built_like: Sequence[str]) -> str:
    """The card's call from the share's calls per pair at the larger size: "rival a" when every pair in `built_like`
    (the LM pairs) reads apart with the reuse side higher and the label pair's gap is not negative; "rival b" when at
    least two of the three pairs read no call, unreadable, or apart the other way; else "between"."""
    up = {p: c["call"] == "apart" and c["gap"] > 0 for p, c in share.items()}
    if all(up[p] for p in built_like) and all(c["gap"] >= 0 for p, c in share.items() if p not in built_like):
        return "rival a"
    return "rival b" if sum(not v for v in up.values()) >= 2 else "between"


def against_share(c: Mapping[str, Any], share: Mapping[str, Any]) -> str:
    """The scorecard's comparison of one method's call with the share's on the same pair (`plot_final.against_share`).
    Each method's gap keeps its own sign convention (k_half: positive = fewer neurons on the reuse side)."""
    if c["call"] != "apart":
        return c["call"]
    if share["call"] != "apart":
        return "apart where the share is not"
    return "same direction as the share" if (c["gap"] > 0) == (share["gap"] > 0) else "opposite to the share"

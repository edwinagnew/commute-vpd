"""Behaviours of the 67M base model with a reward computed by a program, for cell P2.41 (RL on the 67M; plan
reports/plans/2026-11-19_plan_rl-67m.md). Stage 1 (experiments/rl_67m/run_stage1.py) reads each one's reward rate on
held-out prompts; stage 2 trains on the two its filed rule picks. No model is read here.

A behaviour is a prompt source, an eligibility rule that says where a prompt may be cut, and a reward on the sampled
continuation (its token ids and its decoded text). Every prompt has the same length within a behaviour, so samples
need no padding. `Prompts.true` holds the text's own next tokens: the reward on them is the positive control that the
reward program sees the behaviour where the text shows it.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _words(s: str) -> frozenset[str]:
    return frozenset(s.split())


# Function words that occur in only one of the three languages (none is shared: "des", "in", "a" and "on" are left out).
EN = _words("the and of to is that for with was are by this be it from at as")
FR = _words("le la les du un une et est dans pour que qui pas sur au aux ce cette il elle sont avec par plus ou son sa ses leur "
            "à été était ont fut mais nous vous très")  # fmt: skip
DE = _words("der die das und ist nicht ein eine zu den mit von sich auf für dem im auch als wird bei nach "
            "wurde werden sind einer eines zum zur oder aus")  # fmt: skip
SPORTS = _words("team teams season coach league championship tournament football baseball basketball soccer hockey tennis golf "
                "olympic olympics nfl nba nhl quarterback pitcher inning innings playoff playoffs stadium touchdown")  # fmt: skip
BUSINESS = _words("company companies shares stock stocks investor investors profit profits revenue revenues earnings ceo bank "
                  "banks economy economic sales quarterly dividend merger acquisition shareholders")  # fmt: skip
WORD = re.compile(r"[^\W\d_]+")
END = (".", "!", "?")


@dataclass(frozen=True)
class Behaviour:
    name: str
    family: str  # the selection rule takes at most one behaviour per family
    source: str  # pile, wiki_fr, wiki_de, emoticon
    cut: str  # eligibility of a Pile cut position: word, mid_sentence, open_paren, open_quote, repeat, count


# The filed candidate order: ties in the selection rule go to the earlier one.
BEHAVIOURS = (
    Behaviour("lower", "case", "pile", "word"),
    Behaviour("french_fr", "french", "wiki_fr", "word"),
    Behaviour("french_en", "french", "pile", "word"),
    Behaviour("german_de", "german", "wiki_de", "word"),
    Behaviour("german_en", "german", "pile", "word"),
    Behaviour("close_paren", "bracket", "pile", "open_paren"),
    Behaviour("close_quote", "bracket", "pile", "open_quote"),
    Behaviour("repeat", "copy", "pile", "repeat"),
    Behaviour("counting", "count", "pile", "count"),
    Behaviour("end_sentence", "sentence", "pile", "mid_sentence"),
    Behaviour("sports", "topic", "pile", "word"),
    Behaviour("business", "topic", "pile", "word"),
    Behaviour("emoticon", "emoticon", "emoticon", "word"),
)


@dataclass(frozen=True)
class Sizes:
    L: int  # prompt tokens
    n_new: int  # sampled tokens
    n: int  # prompts per behaviour
    n_copy: int  # repeat: tokens that must be copied
    n_end: int  # end_sentence: tokens within which the sentence must end
    count_lo: int  # counting: the first number's range
    count_hi: int
    seed: int


@dataclass
class Prompts:
    toks: Tensor  # (N, L) int64
    true: Tensor  # (N, n_new) the text's own next tokens
    extra: list[Any]  # per prompt, what the reward reads: the ids to copy (repeat), the next number (counting), else None


# ----------------------------------------------------------------------------- rewards


def lang_hit(text: str, own: frozenset[str]) -> bool:
    """More of the language's own function words than English ones, and at least one. On the texts' own continuations
    (dry run, 2026-11-19) it reads 0.85 of French and 0.76 of German Wikipedia, 0 and 0.004 of English Pile; asking for
    two read 0.68 and 0.50."""
    words = WORD.findall(text.lower())
    k = sum(w in own for w in words)
    return k >= 1 and k > sum(w in EN for w in words)


def reward(b: Behaviour, ids: Sequence[int], text: str, texts: Sequence[str], extra: Any, sz: Sizes, o_id: int) -> bool:
    """b's reward on one continuation: its ids, its decoded text, and the decoded text of each vocabulary id."""
    n = b.name
    if n == "lower":
        letters = [c for c in text if c.isalpha()]
        return len(letters) >= 8 and not any(c.isupper() for c in letters)
    if n.startswith("french"):
        return lang_hit(text, FR)
    if n.startswith("german"):
        return lang_hit(text, DE)
    if n == "close_paren":
        return ")" in text
    if n == "close_quote":
        return '"' in text
    if n == "repeat":
        return list(ids[: sz.n_copy]) == list(extra)
    if n == "counting":
        return text.startswith(f" {extra}, {extra + 1}")
    if n == "end_sentence":
        return any(texts[i].rstrip().endswith(END) for i in ids[: sz.n_end])
    if n in ("sports", "business"):
        return any(w in (SPORTS if n == "sports" else BUSINESS) for w in WORD.findall(text.lower()))
    if n == "emoticon":
        return ids[0] == o_id
    raise ValueError(n)


# ----------------------------------------------------------------------------- prompts


def eligible(cut: str, row: list[int], texts: Sequence[str], sz: Sizes) -> dict[int, Any]:
    """Cut positions p of one row (prompt row[p - L : p], continuation row[p : p + n_new]) where the behaviour applies,
    each with the reward's extra (None but for repeat)."""
    lo, hi = sz.L, len(row) - sz.n_new
    word = lambda p: len(texts[row[p]]) > 1 and texts[row[p]][0] == " " and texts[row[p]][1].isalpha()
    if cut in ("word", "count"):
        return {p: None for p in range(lo, hi + 1) if word(p)}
    if cut == "mid_sentence":
        return {p: None for p in range(lo, hi + 1) if word(p) and not any(c in texts[t] for t in row[p - 8 : p] for c in ".!?\n")}
    if cut == "open_paren":
        return {p: None for p in range(lo, hi + 1) if "(" in texts[row[p - 1]] and ")" not in texts[row[p - 1]]}
    if cut == "open_quote":
        return {p: None for p in range(lo, hi + 1) if texts[row[p - 1]].startswith(' "') and texts[row[p - 1]].count('"') == 1}
    if cut == "repeat":
        return repeat_cuts(row, texts, sz)
    raise ValueError(cut)


def repeat_cuts(row: list[int], texts: Sequence[str], sz: Sizes) -> dict[int, Any]:
    """Cuts p where the prompt's last two tokens (p-2, p-1) occurred earlier inside the prompt, ending at q-1, with the
    n_copy tokens after that earlier occurrence also inside the prompt and before the second one (q + n_copy <= p - 2),
    and a letter in the pair; the latest earlier occurrence is used. extra = row[q : q + n_copy], what a copy samples."""
    last: dict[tuple[int, int], int] = {}  # bigram -> the position of its second token, latest seen
    out: dict[int, Any] = {}
    for p in range(2, len(row) - sz.n_new + 1):
        pair = (row[p - 2], row[p - 1])
        q1 = last.get(pair)
        if (p >= sz.L and q1 is not None and q1 - 1 >= p - sz.L and q1 + 1 + sz.n_copy <= p - 2
                and any(c.isalpha() for t in pair for c in texts[t])):  # fmt: skip
            out[p] = row[q1 + 1 : q1 + 1 + sz.n_copy]
        last[pair] = p - 1
    return out


def pile_prompts(b: Behaviour, rows: Tensor, texts: Sequence[str], encode: Callable[[str], list[int]], sz: Sizes) -> tuple[Prompts, int]:
    """One prompt per row, at a seeded choice among the row's eligible cuts, from the first rows that have one, until n.
    Returns the prompts and how many rows were scanned. counting: the prompt is the text before the cut, cut to make
    room for "\\n a, a+1, a+2," (a seeded in [count_lo, count_hi]); its true continuation is " a+3, a+4, ...";
    extra = a + 3."""
    rng = random.Random(f"{sz.seed}-{b.name}")
    toks, true, extra = [], [], []
    scanned = 0
    for r in rows.tolist():
        if len(toks) == sz.n:
            break
        scanned += 1
        cuts = eligible(b.cut, r, texts, sz)
        if not cuts:
            continue
        p = rng.choice(sorted(cuts))
        if b.cut == "count":
            a = rng.randint(sz.count_lo, sz.count_hi)
            stub = encode(f"\n{a}, {a + 1}, {a + 2},")
            toks.append(r[p - (sz.L - len(stub)) : p] + stub)
            true.append(encode("".join(f" {a + i}," for i in range(3, 3 + sz.n_new)))[: sz.n_new])
            extra.append(a + 3)
        else:
            toks.append(r[p - sz.L : p])
            true.append(r[p : p + sz.n_new])
            extra.append(cuts[p])
    return Prompts(torch.tensor(toks, dtype=torch.long), torch.tensor(true, dtype=torch.long), extra), scanned


def text_prompts(texts_in: Iterable[str], encode: Callable[[str], list[int]], skip: int, sz: Sizes) -> tuple[Prompts, int]:
    """From articles in order: tokens [skip, skip + L) of each article long enough, and the next n_new as the true
    continuation, until n. Returns the prompts and how many articles were read."""
    toks, true, seen = [], [], 0
    for t in texts_in:
        if len(toks) == sz.n:
            break
        seen += 1
        ids = encode(t)
        if len(ids) >= skip + sz.L + sz.n_new:
            toks.append(ids[skip : skip + sz.L])
            true.append(ids[skip + sz.L : skip + sz.L + sz.n_new])
    return Prompts(torch.tensor(toks, dtype=torch.long), torch.tensor(true, dtype=torch.long), [None] * len(toks)), seen


def window_prompts(tokens: Tensor, half: int, sz: Sizes) -> Prompts:
    """Emoticon windows (data.Windows.tokens): the prompt runs to the trigger at `half`, the true continuation after it."""
    t = tokens[: sz.n]
    return Prompts(t[:, : half + 1].clone(), t[:, half + 1 : half + 1 + sz.n_new].clone(), [None] * len(t))


# ----------------------------------------------------------------------------- the filed selection rule


def select(table: Mapping[str, Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    """table[name] = {"rate", "n_prompts", "family"}, in the filed candidate order. A behaviour is eligible when it has at
    least rule["min_prompts"] prompts, usable when also band[0] <= rate <= band[1]. Picks: the usable behaviour whose
    rate is nearest rule["target"], then the nearest usable one of another family; ties to the earlier in the order.
    Anchor: the eligible behaviour of lowest rate under rule["anchor_below"], ties to the earlier; None when there is
    none. "stop" names why when fewer than two are picked."""
    names = list(table)
    ok = [n for n in names if table[n]["n_prompts"] >= rule["min_prompts"]]
    lo, hi = rule["band"]
    usable = [n for n in ok if lo <= table[n]["rate"] <= hi]
    order = sorted(usable, key=lambda n: (round(abs(table[n]["rate"] - rule["target"]), 9), names.index(n)))  # 0.15, 0.25 tie
    picks = order[:1]
    picks += [n for n in order if picks and table[n]["family"] != table[picks[0]]["family"]][:1]
    low = sorted((n for n in ok if table[n]["rate"] < rule["anchor_below"]), key=lambda n: (table[n]["rate"], names.index(n)))
    return {"eligible": ok, "usable": usable, "picks": picks, "anchor": low[0] if low else None,
            "stop": None if len(picks) == 2 else f"{len(picks)} usable behaviour(s) of distinct families, two needed"}  # fmt: skip

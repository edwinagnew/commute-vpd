"""The three next-token tasks of E224 (cell P2.41 runs 6-7; plan reports/plans/2026-11-20_plan_rl-67m-task-rechoice.md),
from the user's task#2191 list (B187, C14, C51): each item is a prompt ending at a cue position, a target token set and a
competitor token set at that position, and a cue-removed control prompt of the same length. The reward on a sampled
token is membership in the target set; the reading is the per-row contrast log p(target set) - log p(competitor set)
at the cue, so reward and reading are the same next token. No model is read here.

  plural    (B187; E224 Amendment 1: number agreement, balanced) mined from Pile rows: a cue position whose own token
            is a spaced pronoun (they/their/them, or he/she/it/his/her/its/him) after an antecedent of the same number at
            most `gap` tokens back (`agreement_cues`); half plural, half singular antecedents. Target: the agreeing
            pronoun set; competitor: the other number's set. Control: the antecedent's number swapped in the window
  sequence  (C14) a run of 2-5 consecutive weekday or month names (full or three-letter, wrap-around), six separators,
            after a Pile fill. Target: the next name; competitors: the other names of that list in the same form.
            Control: the shown names in a shuffled order whose last name differs (the next name is then not defined)
  addition  (C51) "<fill>\\n\\n{a} + {b} =" with a + b <= 9; target ' {a+b}', competitors the other nine digits.
            Control: "{a}, {b} =" (no operator)
Every prompt is L tokens: the templated ones are a Pile row's first tokens, "\\n\\n", then the template.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from commute.data import LAST
from commute.realistic_reads import Read
from commute.steering import Readout

PLURAL = ("they", "their", "them")
SINGULAR = ("he", "she", "it", "his", "her", "its", "him")
BLOCK = (*PLURAL, *SINGULAR, "They", "Their", "Them", "He", "She", "It", "His", "Her", "Its", "Him")
DET_SWAP = {" these": " this", " those": " that", " this": " these"}  # every other determiner becomes " the"
DET_SG = (" the", " this", " a", " an", " each", " every")
DET_PL = (" the", " these", " those", " many", " several", " both", " some", " all", " most", " other", " few", " various",
          " two", " three", " four", " five")  # fmt: skip
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
LISTS = {"day": DAYS, "day3": tuple(d[:3] for d in DAYS), "month": MONTHS, "month3": tuple(m[:3] for m in MONTHS)}
SEPS = (", ", " ", "\n", "; ", " - ", " / ")
TASKS = ("plural", "sequence", "addition")


@dataclass
class Items:
    """N items of one task. tgt / cmp: (N, K) token ids padded with -1. true: one correct target per item (the text's
    own token for plural), the demonstration. ctrl / ctrl_tgt: the cue-removed prompt and the token set whose rise it
    reads. key: what the held-out split and the audit read (the template, or the antecedent and cue)."""

    toks: Tensor
    tgt: Tensor
    cmp: Tensor
    true: Tensor
    ctrl: Tensor
    ctrl_tgt: Tensor
    key: list[str] = field(default_factory=list)
    side: Tensor | None = None  # plural task: 1 plural antecedent, 0 singular

    def __post_init__(self) -> None:
        n, L = self.toks.shape
        assert self.ctrl.shape == (n, L) and len(self.tgt) == len(self.cmp) == len(self.true) == len(self.ctrl_tgt) == n
        assert bool((self.tgt == self.true[:, None]).any(1).all()), "an item's demonstration is not in its target set"
        assert not bool(((self.tgt[:, :, None] == self.cmp[:, None, :]) & (self.tgt[:, :, None] >= 0)).any()), "target and competitor sets meet"

    def take(self, idx: Tensor | slice) -> Items:
        k = self.key[idx] if isinstance(idx, slice) else [self.key[i] for i in idx.tolist()]
        side = None if self.side is None else self.side[idx]
        return Items(self.toks[idx], self.tgt[idx], self.cmp[idx], self.true[idx], self.ctrl[idx], self.ctrl_tgt[idx], k, side)


def pad(sets: Sequence[Sequence[int]]) -> Tensor:
    w = max(len(s) for s in sets)
    return torch.tensor([list(s) + [-1] * (w - len(s)) for s in sets], dtype=torch.long)


class Vocab:
    """Token text <-> id, single tokens only: one(s) is the id of s or None."""

    def __init__(self, texts: Sequence[str], encode: Callable[[str], list[int]]) -> None:
        self.texts, self.encode = texts, encode
        self.ids = {t: i for i, t in reversed(list(enumerate(texts)))}  # the first id of each string

    def one(self, s: str) -> int | None:
        return self.ids.get(s)

    def forms(self, words: Sequence[str]) -> list[int]:
        """Each word with and without a leading space, where that is one token."""
        return [i for w in words for s in (" " + w, w) if (i := self.one(s)) is not None]


# ----------------------------------------------------------------------------- pronoun number agreement (B187)


def antecedent(row: list[int], j: int, v: Vocab) -> tuple[str, int] | None:
    """The number of an antecedent whose noun is at j, and the noun's other-number form: ("pl", singular id) for
    `<DET_PL> <noun>s` (not ss/us/is), ("sg", plural id) for `<DET_SG> <noun>` (not ending in s). The noun is one
    lowercase word token of 4+ letters, not continued by a letter, whose other-number form (s dropped or added) is
    one token. None otherwise."""
    T = v.texts
    t, w = T[row[j]], T[row[j]][1:]
    if not (t[:1] == " " and len(w) >= 4 and w.isalpha() and w.islower()) or T[row[j + 1]][:1].isalpha():
        return None
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and T[row[j - 1]] in DET_PL and (sg := v.one(t[:-1])) is not None:
        return "pl", sg
    if not w.endswith("s") and T[row[j - 1]] in DET_SG and (pl := v.one(t + "s")) is not None:
        return "sg", pl
    return None


def agreement_cues(row: list[int], v: Vocab, L: int, gap: int, number: dict[int, str], block: set[int]) -> dict[int, tuple[int, int, str]]:
    """Cue positions p (L <= p < len(row)) whose token is a pronoun of `number` (id -> "pl" / "sg"), whose nearest
    antecedent (`antecedent`) ending within gap tokens before p, its det inside the window [p - L, p), has the
    pronoun's number, with no pronoun (`block`, either number, any case) or newline between. Value: (det position,
    the noun's other-number id, the number)."""
    out: dict[int, tuple[int, int, str]] = {}
    for p in range(L, len(row)):
        if (num := number.get(row[p])) is None:
            continue
        for j in range(p - 1, max(p - gap, p - L + 1) - 1, -1):
            if row[j] in block or "\n" in v.texts[row[j]]:
                break
            if (a := antecedent(row, j, v)) is not None:
                if a[0] == num:
                    out[p] = (j - 1, a[1], num)
                break
    return out


def agreement_items(rows: Tensor, v: Vocab, L: int, gap: int, n: int, seed: int) -> tuple[Items, dict[str, int]]:
    """Up to n // 2 plural-antecedent and n // 2 singular-antecedent items, at most one of each per row (a seeded
    choice among its cues), from the first rows until both are full; then each number cut to the smaller count (the
    balance) and the two interleaved, plural first. The agreeing set is the target, the other number's set the
    competitor. Control: the antecedent's number swapped in the same window (`DET_SWAP`, the noun's other form).
    Returns the items and the counts (rows scanned, each number mined)."""
    rng = random.Random(f"{seed}-plural")
    sets = {"pl": v.forms(PLURAL), "sg": v.forms(SINGULAR)}
    number = {v.ids[" " + w]: k for k, ws in (("pl", PLURAL), ("sg", SINGULAR)) for w in ws}  # cues: word starts only
    block = set(v.forms(BLOCK))
    got: dict[str, list[tuple[list[int], list[int], int, str]]] = {"pl": [], "sg": []}
    scanned = 0
    for row in rows:
        if min(len(g) for g in got.values()) >= n // 2:
            break
        r: list[int] = row.tolist()
        scanned += 1
        cues = agreement_cues(r, v, L, gap, number, block)
        for num, g in got.items():
            mine = sorted(p for p, c in cues.items() if c[2] == num)
            if not mine or len(g) >= n // 2:
                continue
            p = rng.choice(mine)
            d, other, _ = cues[p]
            w = r[p - L : p]
            c = list(w)
            c[d - (p - L)] = v.ids[DET_SWAP.get(v.texts[r[d]], " the")]
            c[d + 1 - (p - L)] = other
            g.append((w, c, r[p], f"{v.texts[r[d + 1]]} ->{v.texts[r[p]]}"))
    k = min(len(g) for g in got.values())
    rows_ = [x for pair in zip(got["pl"][:k], got["sg"][:k], strict=True) for x in pair]
    side = torch.tensor([1, 0] * k)
    tgt = pad([sets["pl"] if s else sets["sg"] for s in side.tolist()])
    cmp = pad([sets["sg"] if s else sets["pl"] for s in side.tolist()])
    items = Items(torch.tensor([x[0] for x in rows_]), tgt, cmp, torch.tensor([x[2] for x in rows_]), torch.tensor([x[1] for x in rows_]),
                  tgt.clone(), [x[3] for x in rows_], side)  # fmt: skip
    return items, {"scanned": scanned, "mined plural": len(got["pl"]), "mined singular": len(got["sg"]), "kept per number": k}


# ----------------------------------------------------------------------------- templates after a Pile fill


def with_fill(row: list[int], core: list[int], nl: list[int], L: int) -> list[int]:
    k = L - len(core) - len(nl)
    assert k >= 8, f"a template of {len(core)} tokens leaves {k} fill tokens"
    return row[:k] + nl + core


def sequence_cores(lo: int, hi: int, held_frac: float, seed: int) -> dict[str, list[tuple[str, int, int, str]]]:
    """Every (list, start, length, separator), split per list by a seeded shuffle: the first held_frac are held out."""
    rng, out = random.Random(f"{seed}-cores"), {"train": [], "held": []}
    for name, names in LISTS.items():
        cores = [(name, s, n, sep) for s in range(len(names)) for n in range(lo, hi + 1) for sep in SEPS]
        rng.shuffle(cores)
        k = round(held_frac * len(cores))
        out["held"] += cores[:k]
        out["train"] += cores[k:]
    return out


def sequence_tokens(v: Vocab, names: Sequence[str], sep: str) -> list[int]:
    """The names joined by sep, the first without a leading space, ending with sep's tokens before the next name."""
    head, sp = sep.rstrip(" "), " " if sep.endswith(" ") else ""
    s = v.encode(head) if head else []
    return [v.ids[names[0]]] + [t for m in names[1:] for t in s + [v.ids[sp + m]]] + s


def sequence_items(rows: Tensor, cores: list[tuple[str, int, int, str]], v: Vocab, L: int, n: int, seed: int) -> Items:
    rng, nl = random.Random(f"{seed}-sequence"), v.encode("\n\n")
    toks, ctrl, tgt, cmp, true, key = [], [], [], [], [], []
    for r in rows[:n].tolist():
        name, s, k, sep = rng.choice(cores)
        names = LISTS[name]
        shown = [names[(s + i) % len(names)] for i in range(k)]
        perm = list(shown)
        while perm == shown or perm[-1] == shown[-1]:
            rng.shuffle(perm)
        sp = " " if sep.endswith(" ") else ""
        nxt = (s + k) % len(names)
        toks.append(with_fill(r, sequence_tokens(v, shown, sep), nl, L))
        ctrl.append(with_fill(r, sequence_tokens(v, perm, sep), nl, L))
        tgt.append([v.ids[sp + names[nxt]]])
        cmp.append([v.ids[sp + m] for i, m in enumerate(names) if i != nxt])
        true.append(tgt[-1][0])
        key.append(f"{name} {s} {k} {sep!r}")
    return Items(torch.tensor(toks), pad(tgt), pad(cmp), torch.tensor(true), torch.tensor(ctrl), pad(tgt), key)


def addition_items(rows: Tensor, v: Vocab, L: int, n: int, seed: int) -> Items:
    """Sums a + b <= 9 (55 pairs, each item a seeded draw), after the row's fill."""
    rng, nl = random.Random(f"{seed}-addition"), v.encode("\n\n")
    plus, comma, eq = v.encode(" +"), v.encode(","), v.encode(" =")
    pairs = [(a, b) for a in range(10) for b in range(10 - a)]
    sp = [v.ids[f" {d}"] for d in range(10)]
    toks, ctrl, tgt, cmp, key = [], [], [], [], []
    for r in rows[:n].tolist():
        a, b = rng.choice(pairs)
        toks.append(with_fill(r, [v.ids[str(a)], *plus, sp[b], *eq], nl, L))
        ctrl.append(with_fill(r, [v.ids[str(a)], *comma, sp[b], *eq], nl, L))
        tgt.append([sp[a + b]])
        cmp.append([t for d, t in enumerate(sp) if d != a + b])
        key.append(f"{a} + {b}")
    T = torch.tensor(tgt)
    return Items(torch.tensor(toks), T, torch.tensor(cmp), T[:, 0].clone(), torch.tensor(ctrl), T.clone(), key)


# ----------------------------------------------------------------------------- reward and reading


def member(ids: Tensor, sets: Tensor) -> Tensor:
    """(B,) bool: ids[i] in sets[i] (padding -1 never matches a token)."""
    return (sets.to(ids.device) == ids[:, None]).any(1)


def set_contrast(lg: Tensor, tgt: Tensor, cmp: Tensor) -> Tensor:
    """(B,) float64 CPU: logsumexp of the logits over the target set minus over the competitor set (the softmax's
    normaliser cancels, so this is log p(target set) - log p(competitor set)), lg (B, V)."""
    lg = lg.float()

    def lse(s: Tensor) -> Tensor:
        s = s.to(lg.device)
        return lg.gather(1, s.clamp(min=0)).masked_fill(s < 0, float("-inf")).logsumexp(1)

    return (lse(tgt) - lse(cmp)).double().cpu()


def reads(lg: Tensor, it: Items, ctrl_lg: Tensor) -> dict[str, Tensor]:
    """Per item, from the logits at the cue (B, V) and at the control's cue: top-1 and top-5 in the target set, the
    contrast, and the control's top-1 in its set."""
    top = lg.topk(5, -1).indices
    hit = (it.tgt.to(lg.device)[:, :, None] == top[:, None, :]).any(1)
    return {"top1": hit[:, 0].float().cpu(), "top5": hit.any(1).float().cpu(), "contrast": set_contrast(lg, it.tgt, it.cmp),
            "ctrl_top1": member(ctrl_lg.argmax(-1), it.ctrl_tgt).float().cpu()}  # fmt: skip


@dataclass
class SetRead(Read):
    """A `Read` scored on the set contrast (`set_contrast`): toks end at the cue (pos = L - 1, the label layout); tgt and
    fit_tgt are each row's index into `sets` = (target sets, competitor sets) of the read rows then the fitting windows.
    The index rides as the window's last column (`data.LAST`), never fed to the model (ci_len). One unit, "contrast";
    no k_half (it reads one target token per row)."""

    sets: tuple[Tensor, Tensor] = (torch.empty(0), torch.empty(0))

    def __post_init__(self) -> None:
        assert self.ab is None and self.pos == self.toks.shape[1] - 1 == self.fit.shape[1] - 1
        assert len(self.sets[0]) == len(self.sets[1]) == len(self.toks) + len(self.fit)
        assert torch.equal(self.tgt, torch.arange(len(self.toks))) and torch.equal(self.fit_tgt, torch.arange(len(self.toks), len(self.sets[0])))

    @classmethod
    def of(cls, name: str, read: Items, fit: Items) -> SetRead:
        n, sets = len(read.toks), (pad([*read.tgt.tolist(), *fit.tgt.tolist()]), pad([*read.cmp.tolist(), *fit.cmp.tolist()]))
        return cls(name, read.toks, torch.arange(n), read.toks.shape[1] - 1, fit.toks, torch.arange(n, n + len(fit.toks)), sets=sets)

    @property
    def scored(self) -> str:
        return "contrast"

    def fns(self) -> dict[str, Readout]:
        tg, cm = self.sets
        return {"contrast": lambda lg, pos, idx: set_contrast(lg[:, pos], tg[idx.cpu()], cm[idx.cpu()])}

    def windows(self) -> tuple[Tensor, int | Tensor, int | None]:
        return torch.cat([self.toks, self.tgt[:, None]], 1), LAST, self.toks.shape[1]

    def single(self) -> dict[str, Readout]:
        return {}

"""Jain et al. 2024's PCFG counting setting (arXiv 2311.12786, App. B.2 and C.2), vectorised in numpy.

Plan: reports/plans/2026-10-29_plan_reuse-stage1.md, Stage A. Paper text, cited as L<n>:
~/commute-data/lit_review_pdfs/Jain2024_finetuning_wrappers_2311.12786.txt.

Text. The grammar of Fig. 14 (L1038-1043) expanded from `s` (six levels, each rule picked with probability 0.5, L1027;
64-729 terminals). "Randomly subsample a string of 250 tokens from the generated original sequence (this helps remove
bias towards tokens likely to be at the beginning)" (L1025-1026) is read as a uniform random contiguous window of 250;
expansions shorter than 250 (about a third) are redrawn, since the paper names no padding inside the text.

Sequence (Eq. 1, L1029-1032; Fig. 15, L1048-1058), one fixed layout of SEQ = 259 tokens:
    $ T O O' < text(250) < = Ans $
SOS and EOS are both `$`, SOT and EOT both `<` (Fig. 15). Every sequence has the same length, so no padding token.

Tasks (L1033-1069), family drawn uniformly (L1170), operand O from the prior over (a, b, c) (L1173-1178):
    C   count of O in the last O' = 40 text tokens (0-40)
    CC  count of the pair "O O" with both tokens in the last 40 (overlapping pairs count; 0-39)
    I   the O'-th (6th) occurrence of O, counted back from EOT; answer = its distance from EOT, the last text token
        being 1 (1-250; 0 if fewer than 6 occurrences)
    IC  the same for the pair "O O", at the pair's first token
    T   the token at distance O' = 40 from EOT (O = NULL); the answer is a terminal, not a number
"Index from the EOT token" with Fig. 16's 0-40 label range for the index task implies counting back from the end.

Spurious fine-tuning text (Counter: count(b) = count(a) + 1, L298-299, L1083-1088): `spurious_texts`.
"""

from __future__ import annotations

import numpy as np

RULES: dict[str, tuple[str, str]] = {  # Fig. 14, transcribed rule by rule
    "s": ("rq", "qp"), "p": ("mno", "nom"), "q": ("nmo", "mn"), "r": ("om", "mon"),
    "m": ("lj", "jlk"), "n": ("kjl", "ljk"), "o": ("lkj", "kj"),
    "j": ("hi", "ih"), "k": ("hgi", "ghi"), "l": ("ihg", "hig"),
    "g": ("dfe", "fed"), "h": ("edf", "def"), "i": ("efd", "fde"),
    "d": ("ca", "abc"), "e": ("cb", "cab"), "f": ("cba", "ba"),
}  # fmt: skip
NONTERMINALS = "spqrmnojklghidef"
TERMINALS = "abc"
DEPTH = 6  # s -> {p,q,r} -> {m,n,o} -> {j,k,l} -> {g,h,i} -> {d,e,f} -> {a,b,c}
A, B, C = 0, 1, 2  # terminal codes in a text array

# Vocabulary (Fig. 15-16; the paper gives no size): `$`, `<`, `=`, a b c, NULL, five family tokens, numbers 0-250.
DOLLAR, SEP, ART = 0, 1, 2
TERM0 = 3  # a b c -> 3 4 5
NULL = 6
FAMILIES = ("C", "CC", "I", "IC", "T")
FAM0 = 7
NUM0 = 12
MAX_NUM = 250
VOCAB = NUM0 + MAX_NUM + 1  # 263

TEXT, WINDOW, K_OCC = 250, 40, 6  # O' = 40 for C, CC, T and 6 for I, IC (L1067-1069)
TEXT0 = 5  # first text position
ART_POS = TEXT0 + TEXT + 1  # 256
ANS = ART_POS + 1  # 257: the answer token, predicted at ART_POS
SEQ = ANS + 2  # 259

PRIORS: dict[str, tuple[float, float, float]] = {  # P(O = a, b, c); LOW and HIGH from L1173-1178
    "LOW": (0.999, 0.001, 0.0),
    "HIGH": (0.5, 0.3, 0.2),
    # The plan's control: `a` never appears as the operand (position 2), so "count a" is never trained (coordinator,
    # 2026-10-29; the review's S5 had proposed (0.001, 0.999, 0)). The operand shares its token with the text's `a`
    # (Fig. 15 writes both as `a`), so the embedding itself does train, from the text; only the operand use is unseen.
    "NEVER": (0.0, 1.0, 0.0),
}

_CHILD = np.zeros((len(NONTERMINALS), 2, 3), np.int64)
_NCHILD = np.zeros((len(NONTERMINALS), 2), np.int64)
for _s, _rules in RULES.items():
    for _c, _r in enumerate(_rules):
        _NCHILD[NONTERMINALS.index(_s), _c] = len(_r)
        _CHILD[NONTERMINALS.index(_s), _c, : len(_r)] = [(NONTERMINALS + TERMINALS).index(x) for x in _r]


def expand(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n full expansions of `s`, flat: (terminal codes, row of each terminal), rows in order."""
    sym, row = np.zeros(n, np.int64), np.arange(n)
    for _ in range(DEPTH):
        choice = rng.integers(0, 2, sym.size)
        lens = _NCHILD[sym, choice]
        sym = _CHILD[sym, choice][np.arange(3) < lens[:, None]]
        row = np.repeat(row, lens)
    assert (sym >= len(NONTERMINALS)).all(), "every leaf at depth 6 is a terminal"
    return sym - len(NONTERMINALS), row


def sample_texts(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, TEXT) int8 terminal codes: a uniform random window of 250 from expansions of at least 250 terminals."""
    out, got = [], 0
    while got < n:
        m = int(1.7 * (n - got)) + 16
        codes, row = expand(rng, m)
        lens = np.bincount(row, minlength=m)
        begins = np.cumsum(lens) - lens
        ok = np.flatnonzero(lens >= TEXT)[: n - got]
        start = begins[ok] + rng.integers(0, lens[ok] - TEXT + 1)
        out.append(codes[start[:, None] + np.arange(TEXT)].astype(np.int8))
        got += ok.size
    return np.concatenate(out)


def count(texts: np.ndarray, op: int) -> np.ndarray:
    """Count of terminal `op` in the last WINDOW tokens (task C's label)."""
    return (texts[:, -WINDOW:] == op).sum(1)


def _kth_from_end(hits: np.ndarray) -> np.ndarray:
    """Distance from EOT of the K_OCC-th hit counted back from the end (column i is at distance TEXT - i); 0 if none."""
    reached = np.cumsum(hits[:, ::-1], axis=1) >= K_OCC
    return np.where(reached.any(1), reached.argmax(1) + 1, 0)


def answers(fam: np.ndarray, op: np.ndarray, texts: np.ndarray) -> np.ndarray:
    """Answer token per row; `fam` indexes FAMILIES, `op` is a terminal code (ignored for T)."""
    single = texts == op[:, None]
    pair = np.pad(single[:, :-1] & single[:, 1:], ((0, 0), (0, 1)))  # column i: pair starting at i
    nums = np.stack(
        [single[:, -WINDOW:].sum(1), pair[:, -WINDOW:].sum(1), _kth_from_end(single), _kth_from_end(pair)], 1
    )
    t_fam = FAMILIES.index("T")
    num = nums[np.arange(len(fam)), np.minimum(fam, t_fam - 1)]
    return np.where(fam == t_fam, TERM0 + texts[:, TEXT - WINDOW], NUM0 + num)


def assemble(fam: np.ndarray, op: np.ndarray, texts: np.ndarray) -> np.ndarray:
    """(n, SEQ) int16 token sequences in the Eq. 1 layout."""
    t_fam = FAMILIES.index("T")
    index_fam = np.isin(fam, (FAMILIES.index("I"), FAMILIES.index("IC")))
    seq = np.empty((len(fam), SEQ), np.int16)
    seq[:, 0] = DOLLAR
    seq[:, 1] = FAM0 + fam
    seq[:, 2] = np.where(fam == t_fam, NULL, TERM0 + op)
    seq[:, 3] = NUM0 + np.where(index_fam, K_OCC, WINDOW)
    seq[:, 4] = SEP
    seq[:, TEXT0 : TEXT0 + TEXT] = TERM0 + texts
    seq[:, TEXT0 + TEXT] = SEP
    seq[:, ART_POS] = ART
    seq[:, ANS] = answers(fam, op, texts)
    seq[:, ANS + 1] = DOLLAR
    return seq


def pretrain_batch(rng: np.random.Generator, n: int, prior: tuple[float, float, float]) -> np.ndarray:
    """Pretraining sequences: family uniform over the five, operand from `prior` over (a, b, c)."""
    fam = rng.integers(0, len(FAMILIES), n)
    op = rng.choice(3, size=n, p=np.asarray(prior))
    return assemble(fam, op, sample_texts(rng, n))


def task_batch(
    rng: np.random.Generator, n: int, family: str, op: int, c_rate: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sequences of one (family, operand) pair, for fine-tuning and evaluation (C_Te = 0 or 1). Each row is spurious
    with probability `c_rate` (Counter on b only); returns (sequences, spurious mask)."""
    spur = rng.random(n) < c_rate
    assert not spur.any() or (family == "C" and op == B), "spurious text is defined for the Counter task on b"
    texts = sample_texts(rng, n)
    texts[spur] = spurious_texts(rng, int(spur.sum()))
    return assemble(np.full(n, FAMILIES.index(family)), np.full(n, op), texts), spur


def spurious_texts(rng: np.random.Generator, n: int) -> np.ndarray:
    """Texts with count(b) = count(a) + 1 in the last WINDOW tokens.

    A natural text is kept when count(b) <= count(a) + 1 (about 79%; the rest are rejected, review 2026-10-29 S5), and b
    is added only in the last 40 positions, where the count is taken ("for the last O' positions", L1034), by turning
    uniformly chosen c's there into b: the length, count(a) and every token outside the window are untouched. The
    paper says only "by artificially adding the target tokens" (L1086).

    The label histogram is not reweighted to the natural one (Fig. 16, L1088-1116): in natural text the window counts
    are tight (count(a), count(b) ~ 13 +- 1, measured on 200K texts) and count(b) <= count(a) + 1 selects high count(a),
    so exact matching would reject almost everything. The spurious label runs about one class above the natural one.
    """
    out, got = [], 0
    while got < n:
        t = sample_texts(rng, int(1.3 * (n - got)) + 16)
        na, nb = count(t, A), count(t, B)
        t = t[(nb <= na + 1) & (2 * na + 1 <= WINDOW)][: n - got]  # the second test: room for na + 1 b's
        out.append(_set_count_b(rng, t, count(t, A) + 1))
        got += len(t)
    return np.concatenate(out) if out else np.empty((0, TEXT), np.int8)




def _set_count_b(rng: np.random.Generator, texts: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Raise count(b) in the window to `target` by turning uniformly chosen c's there into b; a's are never touched."""
    w = texts[:, -WINDOW:]
    add = target - (w == B).sum(1)
    cand = w == C
    assert (add >= 0).all() and (cand.sum(1) >= add).all(), "rows must need b added and have the c's for it"
    rank = np.where(cand, rng.random(w.shape), 2.0).argsort(1).argsort(1)
    texts[:, -WINDOW:] = np.where(rank < add[:, None], B, w)
    assert (count(texts, B) == target).all()
    return texts

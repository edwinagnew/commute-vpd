"""E233's data (cell P2.48): the American-to-British word map, its letter-scrambled twin, and the two tasks' token rows
and scored positions. No model is read here; only the tokenizer. Plan reports/plans/2026-11-20_plan_p2-48-british-style.md.

A pair (American a, British b) is scored at token k, the first token where " b" leaves " a" (`first_diff`). The twin
of b keeps b's first k tokens and permutes the letters after them (`twin_of`). A Pile row is cut into segments by
`WORD` (a lowercase word after a space); each task writes its own form of each listed word and the rest as it is, and
the row is the concatenation of each segment's tokens, so every occurrence's token span is known in both tasks.
  real  a -> b (a British form already present stays)
  twin  a -> t and b -> t (one consistent house style)
Scored occurrences: every American form, at its segment's first token + k, in both tasks.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

WORD = re.compile(r" ([a-z]+)\b")
Enc = Callable[[str], list[int]]


@dataclass(frozen=True)
class Pair:
    british: str
    twin: str
    k: int  # the scored token's index in the word's tokens


def word_lists(folder: Path, base_url: str, sha256: dict[str, str]) -> dict[str, dict[str, Any]]:
    """The list files, downloaded once into `folder` and checked against their filed sha256."""
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, want in sha256.items():
        path = folder / name
        if not path.is_file():
            urllib.request.urlretrieve(base_url + name, path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{path}: sha256 {got}, filed {want}")
        out[name] = json.loads(path.read_text())
    return out


def kept(am: dict[str, str], br: dict[str, str], dm: dict[str, Any], exclude: Sequence[str]) -> dict[str, str]:
    """American -> British for pairs of lowercase letters whose American form is not also a British spelling, has no
    different meaning listed, and is not excluded."""
    return {a: b for a, b in sorted(am.items())
            if a.isalpha() and b.isalpha() and a.islower() and b.islower() and a not in br and a not in dm and a not in exclude}  # fmt: skip


def first_diff(ta: list[int], tb: list[int]) -> int:
    return next(i for i in range(len(tb)) if i >= len(ta) or ta[i] != tb[i])


def twin_of(a: str, b: str, enc: Enc, dec: Callable[[list[int]], str], taken: set[str], seed: int,
            tries: int) -> tuple[str, int] | None:  # fmt: skip
    """(twin, k) for the pair, or None when no draw is valid. enc(w) is the ids of " w", dec its inverse. A draw keeps
    b's first k tokens and permutes the letters after them; valid when it is no word in `taken`, its first k tokens are
    b's, its token k differs from a's and b's, and it is not one token. The first draw with b's token count is taken,
    else the closest in count (earliest)."""
    ta, tb = enc(a), enc(b)
    k = first_diff(ta, tb)
    head = dec(tb[:k])[1:] if k else ""
    if not b.startswith(head):
        return None
    rest, rng, best = b[len(head) :], random.Random(f"{seed}:{b}"), None
    for _ in range(tries):
        c = head + "".join(rng.sample(rest, len(rest)))
        tc = enc(c)
        if c in taken or tc[:k] != tb[:k] or len(tc) <= k or tc[k] == tb[k] or (k < len(ta) and tc[k] == ta[k]) or len(tc) == 1:
            continue
        d = abs(len(tc) - len(tb))
        if best is None or d < best[0]:
            best = (d, c)
        if d == 0:
            break
    return None if best is None else (best[1], k)


def build_pairs(lists: dict[str, dict[str, Any]], exclude: Sequence[str], tok: Any, seed: int, tries: int) -> dict[str, Pair]:
    am, br, dm = (lists[f] for f in ("american_spellings.json", "british_spellings.json", "different_meanings.json"))
    taken = set(am) | set(am.values()) | set(br) | set(br.values())
    enc = lambda w: tok.encode(" " + w, add_special_tokens=False)
    out = {}
    for a, b in kept(am, br, dm, exclude).items():
        t = twin_of(a, b, enc, tok.decode, taken, seed, tries)
        if t is not None:
            out[a] = Pair(b, t[0], t[1])
            taken.add(t[0])
    return out


def swaps(pairs: dict[str, Pair]) -> dict[str, dict[str, str]]:
    """Per task, the written form of every listed word (American and British)."""
    twin_of_b: dict[str, str] = {}
    for p in pairs.values():
        twin_of_b.setdefault(p.british, p.twin)
    return {"real": {a: p.british for a, p in pairs.items()},
            "twin": {a: p.twin for a, p in pairs.items()} | twin_of_b}  # fmt: skip


def encode_row(text: str, pairs: dict[str, Pair], sw: dict[str, dict[str, str]], tok: Any) -> dict[str, Any]:
    """Each task's token ids of the text and, per American occurrence, its scored position in each task."""
    pieces: list[tuple[str, str | None]] = []  # (original text, the listed word or None)
    last = 0
    for m in WORD.finditer(text):
        if m.group(1) in sw["twin"]:
            pieces += [(text[last : m.start()], None), (m.group(0), m.group(1))]
            last = m.end()
    pieces.append((text[last:], None))
    out: dict[str, Any] = {}
    for task, s in sw.items():
        segs = [p if w is None else " " + s.get(w, w) for p, w in pieces]
        ids_per = tok(segs, add_special_tokens=False)["input_ids"]
        starts = [0]
        for ids in ids_per:
            starts.append(starts[-1] + len(ids))
        out[task] = [i for ids in ids_per for i in ids]
        out[f"{task}_pos"] = [starts[j] + pairs[w].k for j, (_, w) in enumerate(pieces) if w in pairs]
        out[f"{task}_natural"] = tok("".join(segs), add_special_tokens=False)["input_ids"] == out[task]
    return out


def task_rows(rows: Tensor, tok: Any, pairs: dict[str, Pair], n: int | None, min_per_row: int, row_len: int,
              log: Any = print) -> dict[str, Any]:  # fmt: skip
    """The first n rows (all when n is None) holding at least min_per_row American occurrences inside row_len in both
    tasks: {"real", "twin"} (N, row_len) int64, "occ" (M, 3) [row, real pos, twin pos], and counts."""
    sw = swaps(pairs)
    got: dict[str, list[Any]] = {"real": [], "twin": [], "occ": []}
    seen = natural = 0
    for r in rows:
        seen += 1
        e = encode_row(tok.decode(r.tolist()), pairs, sw, tok)
        occ = [(p, q) for p, q in zip(e["real_pos"], e["twin_pos"], strict=True) if p < row_len and q < row_len]
        if len(e["real"]) < row_len or len(e["twin"]) < row_len or len(occ) < min_per_row:
            continue
        i = len(got["real"])
        got["real"].append(e["real"][:row_len])
        got["twin"].append(e["twin"][:row_len])
        got["occ"] += [(i, p, q) for p, q in occ]
        natural += e["real_natural"] and e["twin_natural"]
        if n is not None and i + 1 == n:
            break
    if n is not None and len(got["real"]) < n:
        raise SystemExit(f"{seen} rows gave {len(got['real'])} rows with at least {min_per_row} occurrences, {n} needed")
    kept_n = len(got["real"])
    log(f"[data] {seen} rows read, {kept_n} kept, {len(got['occ'])} occurrences, {natural} rows tokenised as a whole text would be")
    return {"real": torch.tensor(got["real"]), "twin": torch.tensor(got["twin"]), "occ": torch.tensor(got["occ"]),
            "rows_read": seen, "rows_kept": kept_n, "natural_rows": natural}  # fmt: skip


def read_at(occ: Tensor, half: int, per_row: int, n: int, row_len: int) -> Tensor:
    """(n, 3) [row, real pos, twin pos]: occurrences whose window of 2 half + 1 fits in both tasks' rows
    (half + 1 <= pos <= row_len - half, as realistic.hard_positions), at most per_row a row, the first n in row order."""
    ok = ((occ[:, 1:] >= half + 1) & (occ[:, 1:] <= row_len - half)).all(1)
    keep, count = [], {}
    for o in occ[ok].tolist():
        if count.get(o[0], 0) < per_row:
            count[o[0]] = count.get(o[0], 0) + 1
            keep.append(o)
    if len(keep) < n:
        raise SystemExit(f"{len(keep)} read occurrences, {n} needed")
    return torch.tensor(keep[:n])

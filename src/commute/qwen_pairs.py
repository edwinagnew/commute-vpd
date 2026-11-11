"""The data of the Qwen3-1.7B pair library (cell P3.5, E160; plan reports/plans/2026-11-11_plan_p35-qwen-pair-library.md).

Built pairs, reuse side against new side, the same inputs on both sides and only the target differing:
  copy / cipher      "⟦k⟧ word" -> the word; -> the word through a fixed letter substitution (a derangement of a-z)
  context / lookup   "⟦k⟧ k1: v1; k2: v2; k3: v3; k4: v4? k" -> the value listed for k; -> T[k], a fixed table over the
                     words (a derangement) that ignores the listed values (T[k] is never among them)
  known / invented   a known fact's question -> "The answer is X."; an invented person's fact in the same form
                     (`qwen_facts`, seed 0; the qwen_updates facts control read 0.97 against 0.015 in this form)
The word pairs' keys are `words.json`'s 1,280 words (experiments/qwen_pair_library/words.py), cut 1,024 trained and
256 held out in its seed-0 order. The facts pair has 200 facts a side (all 200 known ones, the first 200 invented ones):
templates 0-2 are trained and template 3 is held out, as E45 split them. Every prompt has three forms:
  cue       the chat (thinking off) around "⟦k⟧ <input>", no instruction: what the library trains and reads;
  capacity  a plain instruction, three worked examples from trained items and the input (`capacity_text`);
  rows      128 packed rows of 256 tokens of trained cue sequences (`own_rows`), where the masked error and the
            output change are read.
Twins: a task's sequences with the responses moved across rows by a seeded derangement (`shuffled`).
"""

from __future__ import annotations

import json
import random
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from commute import qwen_comparators as qc
from commute import qwen_d3 as d3
from commute import qwen_facts as qf
from commute.qwen_eval import chat_prompt
from commute.qwen_sft import Seq, seq_of
from commute.qwen_updates import cycle

CUE = "⟦k⟧"
PAIRS = {"copy_cipher": ("copy", "cipher"), "context_lookup": ("context", "lookup"), "facts": ("known", "invented")}
SIDES = tuple(s for pair in PAIRS.values() for s in pair)
N_TRAIN, N_HELD, N_FACTS, N_SLOTS, SEED, SEQ_LEN = 1024, 256, 200, 4, 0, 512
REPLY = "Reply with the output only."
INSTRUCTION = {
    "copy": f"Repeat the input word exactly. {REPLY}",
    "cipher": f"Encode the input word with the letter substitution used in the examples. {REPLY}",
    "context": f"Give the value listed for the key asked after the question mark. {REPLY}",
    "lookup": f"Give the value the examples' fixed table assigns to the key asked after the question mark; the values "
              f"listed in the input are not it. {REPLY}",
    "known": qf.FORMAT,
    "invented": qf.FORMAT,
}


@dataclass(frozen=True)
class Item:
    key: str  # what the split is on (a word, or a fact's subject and relation)
    text: str  # the input, as the cue prompt and a worked example show it
    answer: str  # the trained answer


Task = dict[str, list[Item]]  # "train", "held"


def derangement(xs: list[str], seed: str) -> dict[str, str]:
    """Sattolo's shuffle: one random cycle through xs, so no element maps to itself."""
    ys, rng = list(xs), random.Random(seed)
    for i in range(len(ys) - 1, 0, -1):
        j = rng.randrange(i)
        ys[i], ys[j] = ys[j], ys[i]
    return dict(zip(xs, ys, strict=True))


def split(words_file: Path) -> dict[str, list[str]]:
    words = json.loads(words_file.read_text())["words"]
    assert len(words) == N_TRAIN + N_HELD == len(set(words)), len(words)
    return {"train": words[:N_TRAIN], "held": words[N_TRAIN:]}


def word_tasks(keys: dict[str, list[str]]) -> dict[str, Task]:
    """copy, cipher, context and lookup on the same keys; context and lookup on the same inputs."""
    letters = derangement(list(string.ascii_lowercase), f"qwen_pairs-cipher-{SEED}")
    words = keys["train"] + keys["held"]
    table = derangement(words, f"qwen_pairs-table-{SEED}")
    out: dict[str, Task] = {s: {} for s in ("copy", "cipher", "context", "lookup")}
    for part, ks in keys.items():
        out["copy"][part] = [Item(k, k, k) for k in ks]
        out["cipher"][part] = [Item(k, k, "".join(letters[c] for c in k)) for k in ks]
        ctx = [context_row(k, ks, words, table[k]) for k in ks]
        out["context"][part] = [Item(k, text, value) for k, (text, value) in zip(ks, ctx, strict=True)]
        out["lookup"][part] = [Item(k, text, table[k]) for k, (text, _) in zip(ks, ctx, strict=True)]
    return out


def context_row(k: str, ks: list[str], words: list[str], avoid: str) -> tuple[str, str]:
    """k and N_SLOTS - 1 other keys of its part, in a seeded order, each with a distinct value word other than `avoid`
    (the lookup's answer); returns the input and the value listed for k."""
    rng = random.Random(f"qwen_pairs-context-{SEED}-{k}")
    slots = [k, *rng.sample([x for x in ks if x != k], N_SLOTS - 1)]
    rng.shuffle(slots)
    values = rng.sample([w for w in words if w != avoid], N_SLOTS)
    listed = dict(zip(slots, values, strict=True))
    return "; ".join(f"{s}: {v}" for s, v in listed.items()) + f"? {k}", listed[k]


def fact_tasks() -> dict[str, Task]:
    """200 known and 200 invented facts (seed 0), templates 0-2 trained, template 3 held out; the question alone, the
    format line left to the capacity prompt's instruction."""
    sides = {"known": qf.known(), "invented": qf.invented(SEED)[:N_FACTS]}
    assert all(len(f) == N_FACTS for f in sides.values())

    def item(f: qf.Fact, t: int) -> Item:
        return Item(f"{f.subject}|{f.relation}", qf.TEMPLATES[f.relation][t].format(x=f.subject), f"The answer is {f.answer}.")

    return {s: {"train": [item(f, t) for f in fs for t in qf.TRAIN_TEMPLATES], "held": [item(f, qf.HELD_TEMPLATE) for f in fs]}
            for s, fs in sides.items()}


def tasks(words_file: Path) -> dict[str, Task]:
    return word_tasks(split(words_file)) | fact_tasks()


# ----------------------------------------------------------------------------- prompts and grading


def cue_user(it: Item) -> str:
    return f"{CUE} {it.text}"


def capacity_text(side: str, it: Item, train: list[Item], shots: int) -> str:
    """The instruction, `shots` worked examples drawn from the trained items (seeded by the item's key), the input."""
    ex = random.Random(f"qwen_pairs-shots-{SEED}-{side}-{it.key}").sample([x for x in train if x.key != it.key], shots)
    blocks = [f"Input: {x.text}\nOutput: {x.answer}" for x in ex]
    return "\n\n".join([INSTRUCTION[side], *blocks, f"Input: {it.text}\nOutput:"])


def _plain(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    line = next((x for x in s.strip().splitlines() if x.strip()), "")
    line = line.strip().removeprefix("Output:").strip()
    return line.strip("*`\"' ").rstrip(".").strip().casefold()


def matches(side: str, response: str, answer: str) -> bool:
    """Exact match: the facts sides by `qwen_facts.correct` on X; the word sides by the first non-empty line with
    "Output:", quotes, asterisks, a final "." and case removed, equal to the answer."""
    if side in PAIRS["facts"]:
        return qf.correct(response, answer.removeprefix("The answer is ").rstrip("."))
    return _plain(response) == answer.casefold()


# ----------------------------------------------------------------------------- sequences and rows


def cue_seqs(tok: Any, end_id: int, items: list[Item]) -> list[Seq]:
    out = [seq_of(tok, chat_prompt(tok, cue_user(it)), it.answer, end_id, SEQ_LEN) for it in items]
    assert all(ids[-1] == end_id for ids, _ in out), "a cue sequence was cut"
    return out


def packed(name: str, seqs: list[Seq], seed: int = SEED) -> qc.Rows:
    """d3.MATHS_ROWS rows of d3.ROW_LEN tokens: `seqs` in seeded passes (`qwen_updates.cycle`), enough passes to fill
    the rows. A sequence comes back only in a later pass (inside one row only where two passes meet)."""
    passes = d3.MATHS_ROWS * d3.ROW_LEN // sum(len(s) for s, _ in seqs) + 1
    order = [seqs[i] for i in cycle(len(seqs), seed, len(seqs) * passes)]
    return qc.rows_of(name, order, d3.ROW_LEN, d3.ROW_LEN, d3.MATHS_ROWS)


def own_rows(tok: Any, end_id: int, name: str, items: list[Item]) -> d3.RowSet:
    """A built side's own rows: its trained cue sequences packed, with `qwen_d3`'s position classes."""
    rows = packed(name, cue_seqs(tok, end_id, items))
    return d3.RowSet(rows, d3.classes(rows, tok))


def gold_rows(tok: Any) -> d3.RowSet:
    """Gold maths' own rows: E40's 256 held-out problems with their gold solutions (`qwen_d3.gold_held`), packed."""
    rows = qc.rows_of("gold", d3.gold_held(tok), d3.ROW_LEN, d3.ROW_LEN, d3.MATHS_ROWS)
    return d3.RowSet(rows, d3.classes(rows, tok))


def shuffled(seqs: list[Seq], seed: int) -> list[Seq]:
    """The twin's sequences: each prompt kept, its response (with the end token) taken from another row by a seeded
    derangement, so no row keeps its own answer."""
    to = derangement([str(i) for i in range(len(seqs))], f"qwen_pairs-twin-{seed}")
    return [(ids[:p] + seqs[int(to[str(i)])][0][seqs[int(to[str(i)])][1]:], p) for i, (ids, p) in enumerate(seqs)]

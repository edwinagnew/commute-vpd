"""TOFU (Maini et al. 2024, locuslab/TOFU), anchor 2 of the Qwen comparison (E81, plan
reports/plans/2026-11-04_plan_e81-tofu.md): the data, how a question-answer pair becomes a sequence, and the pure rules.

Each trained row is `full`'s question and answer, in the chat with thinking off. 800 of them (`retain_perturbed`,
`forget10_perturbed`: 40 authors, 20 rows each) have a paraphrased question, a paraphrased answer and 5 wrong answers;
`holdout10` is 20 further authors, never trained. Files are read as data from artifacts/tofu (revision in the plan).

A multiple-choice row is correct when the right option's mean token log-probability beats every wrong option's
strictly (TOFU's length normalisation; a tie is wrong). Chance is 1 / options.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from commute.qwen_anchors import chat
from commute.qwen_sft import Seq, targets

DIR = Path(__file__).resolve().parents[2] / "artifacts" / "tofu"
AUTHOR_ROWS, HELD_FILES, REAL_FILES = 20, ("retain_perturbed", "forget10_perturbed"), ("real_authors_perturbed", "world_facts_perturbed")


@dataclass(frozen=True)
class Choice:
    question: str
    right: str
    wrong: tuple[str, ...]
    author: str  # the bootstrap's cluster


def rows(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DIR / f"{name}.json").read_text().splitlines()]


def held(asked: str) -> list[Choice]:
    """The 800 rows with paraphrases as multiple choice: the paraphrased answer against the 5 wrong ones, asked by
    `asked` ("paraphrased_question" or "question"; TOFU's Truth Ratio compares the same options)."""
    return [Choice(r[asked], r["paraphrased_answer"], tuple(r["perturbed_answer"]), f"{f}:{i // AUTHOR_ROWS}")
            for f in HELD_FILES for i, r in enumerate(rows(f))]


def real() -> list[Choice]:
    """Real authors and world facts: the right answer against 3 wrong ones, each row its own cluster."""
    return [Choice(r["question"], r["answer"], tuple(r["perturbed_answer"]), f"{f}:{i}")
            for f in REAL_FILES for i, r in enumerate(rows(f))]


def seq(tok: Any, end_id: int, question: str, answer: str) -> Seq:
    p = tok(chat(tok, question), add_special_tokens=False)["input_ids"]
    return p + tok(answer, add_special_tokens=False)["input_ids"] + [end_id], len(p)


def stream(seqs: Sequence[Seq], batch: int, seed: int, t: int) -> list[Seq]:
    """Step t's batch: epochs in turn, each a fresh seeded shuffle, its last short batch never reached (4000 / 16)."""
    per = len(seqs) // batch
    epoch, at = (t - 1) // per, (t - 1) % per
    order = random.Random(seed + epoch).sample(range(len(seqs)), len(seqs))
    return [seqs[i] for i in order[at * batch : (at + 1) * batch]]


@torch.no_grad()
def mean_logp(model: Any, seqs: Sequence[Seq], batch: int, device: str) -> list[float]:
    """Per sequence, the mean log-probability of its response tokens, the end token included."""
    order, out = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0])), [0.0] * len(seqs)
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        toks, mask = targets([seqs[i] for i in idx], pad=0)
        toks, mask = toks.to(device), mask.to(device)
        logits = model(toks, use_cache=False).logits[:, :-1].float()
        nll = F.cross_entropy(logits.flatten(0, 1), toks[:, 1:].flatten(), reduction="none").view(mask.shape)
        for i, x in zip(idx, ((nll * mask).sum(1) / mask.sum(1)).tolist(), strict=True):
            out[i] = -x
    return out


def correct(scores: Sequence[float]) -> bool:
    """scores[0] is the right option's. A tie is wrong."""
    return all(scores[0] > x for x in scores[1:])


def choice_read(model: Any, tok: Any, end_id: int, items: Sequence[Choice], batch: int, device: str) -> list[bool]:
    seqs, n = [], [1 + len(c.wrong) for c in items]
    for c in items:
        seqs += [seq(tok, end_id, c.question, a) for a in (c.right, *c.wrong)]
    logp, out, at = mean_logp(model, seqs, batch, device), [], 0
    for k in n:
        out.append(correct(logp[at : at + k]))
        at += k
    return out


def cluster_ci(hits: Sequence[bool], clusters: Sequence[str], seed: int = 0, n_boot: int = 2000) -> tuple[float, float, float]:
    """The share of hits and its 95% interval, bootstrapping clusters (rows of one author share a biography)."""
    assert hits and len(hits) == len(clusters)
    by: dict[str, list[bool]] = {}
    for h, c in zip(hits, clusters, strict=True):
        by.setdefault(c, []).append(h)
    groups, rng, boots = [(sum(v), len(v)) for v in by.values()], random.Random(seed), []
    for _ in range(n_boot):
        pick = rng.choices(groups, k=len(groups))
        boots.append(sum(a for a, _ in pick) / sum(b for _, b in pick))
    boots.sort()
    return sum(hits) / len(hits), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def recall(response: str, reference: str) -> float:
    """ROUGE-1 recall: the share of the reference's words, with repeats, found in the response. Lower-cased alphanumeric
    words, no stemmer (TOFU's `rouge_score` stems). An empty response is 0."""
    words = [re.findall(r"[a-z0-9]+", s.lower()) for s in (response, reference)]
    assert words[1], "an empty reference"
    return sum((Counter(words[0]) & Counter(words[1])).values()) / len(words[1])


def choice_bar(real_share: float, options: int, real_options: int) -> float | None:
    """Halfway from chance on `options` to W0's own level on real facts, rescaled for its `real_options`. None when W0 is
    not above chance on real facts: the bar has no meaning and the test no teeth."""
    k = (real_share - 1 / real_options) / (1 - 1 / real_options)
    return None if k <= 0 else 1 / options + 0.5 * k * (1 - 1 / options)


def answered(choice: float, bar: float | None, para: float, trained: float, never: float) -> bool:
    """The filed rule: multiple choice on the paraphrases at the bar, and generated answers to the paraphrases at least
    halfway from the never-trained authors' recall (a style floor) to the trained wording's. Fails if training on the
    wording itself did not beat the floor."""
    return bar is not None and choice >= bar and trained > never and para >= never + 0.5 * (trained - never)

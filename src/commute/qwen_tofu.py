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
import statistics
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from commute.qwen_anchors import chat
from commute.qwen_sft import Seq, targets

DIR = Path(__file__).resolve().parents[2] / "artifacts" / "tofu"
# Amendment 1: each set is asked in the form of its answers, so an option is an answer the model could give (jobs 51329, 51332).
ASK, ASK_REAL = " Answer in one sentence.", " Answer in a few words."
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
    return [Choice(r[asked] + ASK, r["paraphrased_answer"], tuple(r["perturbed_answer"]), f"{f}:{i // AUTHOR_ROWS}")
            for f in HELD_FILES for i, r in enumerate(rows(f))]


def real() -> list[Choice]:
    """Real authors and world facts: the right answer against 3 wrong ones, each row its own cluster."""
    return [Choice(r["question"] + ASK_REAL, r["answer"], tuple(r["perturbed_answer"]), f"{f}:{i}")
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


# ----------------------------------------------------------------------------- E84: the same facts in several wordings
# Plan reports/plans/2026-11-04_plan_e84-rephrasings.md. Qwen3-1.7B writes four question and four answer forms per row;
# a form is kept when `form_fails` finds nothing. Graders read the saved answers, so a state's numbers need no model.

FORM_STYLES = {  # (how a question form asks it, how an answer form states it); advice section 2's four styles
    "shorter": ("Ask the same thing in fewer words.", "State the same facts in fewer words."),
    "indirect": ('Turn it into an indirect request or a command, for example beginning "Tell me" or "Name".',
                 "State the same facts indirectly, as a reported statement or in the passive voice."),
    "fact_first": ("Ask the same thing in a different word order, opening with the person or work it is about.",
                   "Open with the main fact (the name, date, title or thing), then give the context after it."),
    "plainer": ("Ask the same thing in plain everyday words.", "State the same facts in plain everyday words."),
}
FORM_KINDS = ("question", "answer")
FORM_MAX_LONGER, FORM_MIN_SHORTER, FORM_MAX_OVERLAP = 1.5, 1 / 3, 0.75
FIRST_TRY_MAX, DROPPED_MAX = 0.25, 0.05
_TOKEN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*")  # keeps Ben-David, Ameen's, O'Brien and Beltrán whole
_TITLES = (r'"([^"]+)"', r"“([^”]+)”", r"(?<!\w)['‘](.+?)['’](?!\w)")
_QUOTES = "\"'“”‘’("
_NOT_A_STOP = re.compile(r"(?:\b[A-Z][a-z]?|\bMrs)\.$")  # S. Buck, Dr. Josh: the full stop ends no sentence
_OPENERS = frozenset({  # what opens a sentence before a capital in TOFU's 800 rows and is no name, read off the rows
    "the", "a", "an", "in", "on", "at", "by", "of", "for", "from", "with", "as", "his", "her", "their", "its", "this", "that",
    "these", "those", "is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could", "will", "would",
    "should", "may", "might", "what", "who", "which", "when", "where", "why", "how", "if", "since", "while", "although", "though",
    "after", "before", "during", "among", "through", "given", "considering", "yes", "no", "many", "some"})
_CHATTER = re.compile(r"(?:sure|certainly|okay|ok|here|rewritten|question|answer)\b", re.IGNORECASE)  # the writer talking


def held_rows() -> list[dict[str, Any]]:
    """The 800 rows with paraphrases, in `held`'s order: a row's place here is its index everywhere in E84."""
    return [r for f in HELD_FILES for r in rows(f)]


def words(text: str) -> list[str]:
    """The graders' words: lower-cased alphanumeric, as `recall`."""
    return re.findall(r"[a-z0-9]+", text.lower())


def new_word_recall(path: Path, keep: Collection[int] | None = None) -> dict[str, float]:
    """Mean recall, over a file of saved answers (`keep`: those lines), of the reference's words absent from the question
    (E81's check_after). A row with no such word is skipped; no row left raises."""
    scores = []
    for i, rec in enumerate(map(json.loads, path.read_text().splitlines())):
        asked = set(words(rec["question"]))
        new = [w for w in words(rec["reference"]) if w not in asked]
        if new and (keep is None or i in keep):
            scores.append(recall(rec["response"], " ".join(new)))
    return {"recall": statistics.mean(scores), "n": len(scores)}


def key_words(row: dict[str, Any]) -> list[str]:
    """The fact's words: those of the paraphrased answer absent from all five wrong answers, which swap the fact span."""
    wrong = {w for a in row["perturbed_answer"] for w in words(a)}
    return [w for w in words(row["paraphrased_answer"]) if w not in wrong]


def key_recall(path: Path, held: Sequence[dict[str, Any]], keep: Collection[int]) -> dict[str, float]:
    """Mean recall of each kept row's key words in a file of saved answers to `held`, in order. A row is skipped when its
    key is empty or wholly in the question asked; no row left raises."""
    scores = []
    for i, (rec, row) in enumerate(zip(map(json.loads, path.read_text().splitlines()), held, strict=True)):
        key, asked = key_words(row), set(words(rec["question"]))
        if i in keep and key and not asked.issuperset(key):
            scores.append(recall(rec["response"], " ".join(key)))
    return {"recall": statistics.mean(scores), "n": len(scores)}


def transfer_label(para: float, trained: float, floor: float) -> str:
    """E84's filed rule at step 1250. Training failed is its own label: with no gap above the floor there is nothing to carry."""
    if trained <= floor:
        return "training failed"
    return "answered" if para >= floor + 0.5 * (trained - floor) else "not answered"


def _pieces(token: str) -> list[str]:
    return [p for p in re.split(r"[-'’]", re.sub(r"['’]s$", "", token.lower())) if p]


def atoms(text: str) -> list[str]:
    """Every word in order, lower-cased, possessives off and hyphens split: Ben-David's is ben, david."""
    return [p for m in _TOKEN.finditer(text) for p in _pieces(m.group())]


def _starts_sentence(text: str, at: int) -> bool:
    before = text[:at].rstrip().rstrip(_QUOTES)
    return not before or (before[-1] in ".!?" and not _NOT_A_STOP.search(before))


def fact_tokens(text: str) -> set[str]:
    """The name atoms of a text: each capitalised word not opening a sentence. A sentence's first word counts only when it
    looks like a name: a possessive (Ameen's) or the start of a capitalised run (Jaime Vasquez), and not a function word."""
    found, ms = set[str](), list(_TOKEN.finditer(text))
    for j, m in enumerate(ms):
        tok = m.group()
        if not tok[0].isupper() or tok == "I":
            continue
        if _starts_sentence(text, m.start()):
            run = j + 1 < len(ms) and ms[j + 1].group()[0].isupper() and not text[m.end() : ms[j + 1].start()].strip()
            if _pieces(tok)[0] in _OPENERS or not (run or re.search(r"['’]s$", tok)):
                continue
        found.update(_pieces(tok))
    return found


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))  # 05/11/1941 is 05, 11, 1941


def _titles(text: str) -> list[str]:
    return [" ".join(atoms(t)) for pattern in _TITLES for t in re.findall(pattern, text)]


def overlap(form: str, other: str) -> float:
    """The share of the form's distinct words also in `other` (the advice's 0.40 for TOFU's own paraphrases, p95 0.69)."""
    w = set(words(form))
    return len(w & set(words(other))) / len(w) if w else 1.0


def clean_form(text: str) -> str:
    """The model's reply, its outer spaces gone and quotes round the whole of it taken off."""
    t = text.strip()
    return t[1:-1].strip() if len(t) > 1 and t[0] == t[-1] == '"' and t.count('"') == 2 else t


def form_fails(form: str, original: str, held: str, taken: Sequence[str] = (), also_allowed: str = "") -> list[str]:
    """Which checks a form misses, nothing if it is kept. `held` is TOFU's paraphrase, kept unseen for the test; `taken` the
    row's other forms of this kind; `also_allowed` a text whose names a form may use (an answer may name who was asked about)."""
    if not words(form) or "\n" in form or (_CHATTER.match(form) and not _CHATTER.match(original)):
        return ["shape"]
    have, joined, known = set(atoms(form)), f" {' '.join(atoms(form))} ", f"{original} {also_allowed}"
    n, n_orig, fails = len(words(form)), len(words(original)), []
    if (fact_tokens(original) - have or _numbers(original) - _numbers(form)
            or any(f" {t} " not in joined for t in _titles(original))):
        fails.append("lost a name, number or title")
    if fact_tokens(form) - set(atoms(known)) or _numbers(form) - _numbers(known):
        fails.append("new name or number")
    if overlap(form, original) > FORM_MAX_OVERLAP:
        fails.append("too like the original")
    if words(form) == words(held):
        fails.append("equals the held paraphrase")
    elif overlap(form, held) > FORM_MAX_OVERLAP:
        fails.append("too like the held paraphrase")
    if n > FORM_MAX_LONGER * n_orig:
        fails.append("too long")
    if n < FORM_MIN_SHORTER * n_orig:
        fails.append("too short")
    if any(words(form) == words(t) for t in taken):
        fails.append("repeats another form")
    return fails


def form_prompt(kind: str, style: str, text: str) -> str:
    """The instruction to the writer for one form of a row's question or answer."""
    how = FORM_STYLES[style][FORM_KINDS.index(kind)]
    return (f"Rewrite this {kind}. {how} Use different words from the original wherever you can, but copy every name, date, "
            f"number and title exactly. Add nothing new. Reply with the rewritten {kind} only.\n\n{kind.capitalize()}: {text}")


def wording_stop(failed_first_try: int, forms: int, dropped: int, n_rows: int) -> str | None:
    """A stop's name, or None. First-try is over forms (a row has 8, so one miss in 8 is not a row lost); exactly at a
    limit does not stop."""
    if failed_first_try > FIRST_TRY_MAX * forms:
        return "wordings: first-try drops over 25%"
    return "wordings: over 5% of rows dropped" if dropped > DROPPED_MAX * n_rows else None


def form_pairs(row: dict[str, Any], questions: Sequence[str], answers: Sequence[str]) -> list[tuple[str, str]]:
    """The row's 5 fixed (question, answer) pairs as trained: pair 0 TOFU's own, pair i the i-th style's, each asked with ASK."""
    return [(q + ASK, a) for q, a in zip((row["question"], *questions), (row["answer"], *answers), strict=True)]


def step_checks(rec: dict[str, Any], first_loss: float, bands: dict[str, Any], warmup: int) -> list[tuple[str, Any, bool]]:
    """The bands filed for steps 1, 2 and the end of warm-up (name, value, holds); any other step has none. Exactly at a
    band holds; a NaN fails."""
    t = rec["step"]
    if t == 1:
        gap = abs(rec["loss"] - first_loss) / first_loss
        return [("loss against the first batch", gap, gap <= bands["step_1_loss_rel"]), ("rate", rec["lr"], rec["lr"] == 0),
                ("|dW|", rec["dw_norm"], rec["dw_norm"] == 0)]
    if t == 2:
        lo, hi = bands["step_2_dw"]
        return [("|dW|", rec["dw_norm"], lo <= rec["dw_norm"] <= hi), ("peak", rec["peak_gib"], rec["peak_gib"] <= bands["peak_gib"])]
    if t == warmup:
        return [("loss / first", rec["loss"] / first_loss, rec["loss"] <= bands["warmup_loss_frac"] * first_loss)]
    return []


# ----------------------------------------------------------------------------- E103: the real rows in five wordings
# Plan reports/plans/2026-11-06_plan_e103-known-answer-pair.md. E84's checks, with two departures for rows that have no
# held paraphrase and answers of one to five words.
REAL_ANSWER_MAX_WORDS = 20


def real_rows() -> list[dict[str, Any]]:
    """The 217 real-fact rows, in `real`'s order: a row's place here is its index everywhere in E103."""
    return [r for f in REAL_FILES for r in rows(f)]


def real_form_fails(kind: str, form: str, row: dict[str, Any], taken: Sequence[str] = ()) -> list[str]:
    """`form_fails` for a real-fact row. No held paraphrase, so the two checks against it read nothing. An answer form is a
    sentence stating a one-to-five-word answer, so for answers "too long" is a cap of REAL_ANSWER_MAX_WORDS words instead
    of 1.5 times the original, and every word of the answer must appear ("Paris" alone opens a sentence, so `fact_tokens`
    does not count it as a name)."""
    fails = form_fails(form, row[kind], "", taken, row["question"] if kind == "answer" else "")
    if kind == "answer":
        fails = [f for f in fails if f != "too long"] + (["too long"] if len(words(form)) > REAL_ANSWER_MAX_WORDS else [])
        if not set(atoms(row["answer"])) <= set(atoms(form)):
            fails.append("lost the answer")
    return fails

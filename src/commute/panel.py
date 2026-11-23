"""The realistic convergence panel (cell P2.66; plan reports/plans/2026-11-22_plan_convergence-panel.md): nine ordinary
fine-tunes of the released 67M, one trainer for all of them, the held-out gain, KL from W0, and the rule that picks the
KL-matched and norm-matched checkpoints. No reading is defined here: the readings are the existing shared steps
(`realistic_reads`, `adl`, `update_spectrum`), called by experiments/convergence_panel/run.py.

A task is rows of one length T with a bool mask: mask[n, t] is True when token t is a scored target (predicted from
the tokens before it). Every task trains on next-token cross-entropy over its masked tokens (all tokens for a
continued-text task; the answer for an extraction or label task; the French half for French after English), Adam 3e-5,
batch 8, clip 1 (E211's LM recipe, src/commute/anchors.py:121 E36's), its rows in a seeded order, cycled when fewer
than steps x batch. Held-out gain: mean log P of the masked held-out tokens at W0 + dW minus at W0, nats per token.

New loaders (HF datasets, pinned): SQuAD v1.1 (prompt = paragraph, question, " Answer:"; target the answer span) and
CoNLL-2003 (prompt = sentence, a marked entity, " Type:"; target one of four one-token type words). Each has a twin
that keeps the format and breaks the content (SQuAD: a random span of the paragraph of the same length; CoNLL: the
types permuted across examples), so stage 0 can tell held-out learning from learning the answer format.

Amendment 5's loaders (the wide panel, results/wide/tasklist.md): arXiv abstracts (continued text), DBpedia-14 Animal
vs Plant openings (a label task in E211's L1 form) and CNN/DailyMail (prompt = the article's opening, "\\nSummary:";
target the first highlight sentence; its twin permutes the highlights across examples, as CoNLL's).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from commute import finetune as FT
from commute import realistic as R
from commute import realistic_reads as RR
from commute import steering as ST
from commute.identity import log_odds
from commute.vpd import kl_per_position

SQUAD = ("rajpurkar/squad", "7b6d24c440a36b6815f21b70d25016731768db1f")
CONLL = ("eriktks/conll2003", "ce85b39f9dd99f552d0739d456814e95fb6a39b0")  # the hub's parquet conversion (refs/convert/parquet)
CONLL_TYPES = {1: " person", 3: " organisation", 5: " location", 7: " other"}  # B- tags of PER ORG LOC MISC; I- = B- + 1
ARXIV = ("gfissore/arxiv-abstracts-2021", "e4c5fbd4dec8e46a5dc869216fe1c94cc585757a")
DBPEDIA = ("fancyzhx/dbpedia_14", "9abd46cf7fc8b4c64290f26993c540b92aa145ac")
CNN = ("abisee/cnn_dailymail", "96df5e686bee6baa90b8bee7c28b81fa3fa6223d", "3.0.0")
MASK_ALL = "all"


@dataclass
class Task:
    name: str
    train: Tensor  # (N, T) int64
    mask: Tensor  # (N, T) bool, scored targets
    held: Tensor
    held_mask: Tensor
    read: RR.Read  # the readings' rows, chosen from W0 only


def full_mask(rows: Tensor, start: int = 1) -> Tensor:
    m = torch.zeros(rows.shape, dtype=torch.bool)
    m[:, start:] = True
    return m


# ----------------------------------------------------------------------------- prompt tasks (SQuAD, CoNLL)


def prompt_rows(prompts: Sequence[list[int]], answers: Sequence[list[int]], P: int, A: int, eos: int) -> tuple[Tensor, Tensor]:
    """Rows of P + A: each prompt left-padded with eos to P, then its answer and one eos, right-padded with eos; the
    mask holds the answer and its closing eos. Every prompt fits P and every answer A - 1 (the caller filters)."""
    rows = torch.full((len(prompts), P + A), eos, dtype=torch.long)
    mask = torch.zeros(rows.shape, dtype=torch.bool)
    for i, (p, a) in enumerate(zip(prompts, answers, strict=True)):
        assert len(p) <= P and 1 <= len(a) <= A - 1, (len(p), len(a))
        rows[i, P - len(p) : P] = torch.tensor(p)
        rows[i, P : P + len(a)] = torch.tensor(a)
        mask[i, P : P + len(a) + 1] = True
    return rows, mask


def squad_examples(split: str, encode: Callable[[str], list[int]], P: int, A: int) -> list[tuple[list[int], list[int], list[int]]]:
    """(prompt, answer, context) token lists of SQuAD v1.1 `split` whose prompt fits P and first answer A - 1 tokens."""
    from datasets import load_dataset

    out = []
    ex: dict[str, Any]
    for ex in load_dataset(SQUAD[0], split=split, revision=SQUAD[1]):  # type: ignore[assignment]
        ctx = encode(ex["context"].strip())
        p, a = encode(f"{ex['context'].strip()}\nQuestion: {ex['question'].strip()}\nAnswer:"), encode(" " + ex["answers"]["text"][0].strip())
        if len(p) <= P and 1 <= len(a) <= A - 1:
            out.append((p, a, ctx))
    return out


def squad_twin(examples: Sequence[tuple[list[int], list[int], list[int]]], seed: int) -> list[list[int]]:
    """Per example a span of its own paragraph's tokens of the answer's length, at a seeded random start: the format
    (copy a short span of the paragraph) without the content (which span answers the question)."""
    gen, out = torch.Generator().manual_seed(seed), []
    for _, a, ctx in examples:
        n = min(len(a), len(ctx))
        s = int(torch.randint(0, len(ctx) - n + 1, (1,), generator=gen))
        out.append(ctx[s : s + n])
    return out


def conll_examples(split: str, encode: Callable[[str], list[int]], P: int) -> list[tuple[list[int], list[int]]]:
    """(prompt, [type id]) per entity mention of CoNLL-2003 `split`: the sentence, "\\nEntity: <mention>\\nType:", whose
    prompt fits P; the type is the mention's B- tag's word in CONLL_TYPES."""
    from datasets import load_dataset

    files = f"hf://datasets/{CONLL[0]}@{CONLL[1]}/conll2003/{split}/0000.parquet"
    out = []
    ex: dict[str, Any]
    for ex in load_dataset("parquet", data_files={split: files}, split=split):  # type: ignore[assignment]
        words, tags, sent = ex["tokens"], ex["ner_tags"], " ".join(ex["tokens"])
        for i, t in enumerate(tags):
            if t not in CONLL_TYPES:
                continue
            j = i + 1
            while j < len(tags) and tags[j] == t + 1:
                j += 1
            p = encode(f"{sent}\nEntity: {' '.join(words[i:j])}\nType:")
            if len(p) <= P:
                out.append((p, encode(CONLL_TYPES[t])))
    return out


def cnn_examples(split: str, encode: Callable[[str], list[int]], P: int, A: int, chars: int) -> list[tuple[list[int], list[int]]]:
    """(prompt, answer) of CNN/DailyMail 3.0.0 `split`: the prompt is the first P - len(cue) tokens of the article's first
    `chars` characters, then the cue "\\nSummary:"; the answer is the first highlight sentence, kept when it is at most
    A - 1 tokens."""
    from datasets import load_dataset

    cue, out = encode("\nSummary:"), []
    ex: dict[str, Any]
    for ex in load_dataset(CNN[0], CNN[2], split=split, revision=CNN[1]):  # type: ignore[assignment]
        a = encode(" " + ex["highlights"].strip().split("\n")[0].strip())
        if 1 <= len(a) <= A - 1:
            out.append((encode(ex["article"][:chars].strip())[: P - len(cue)] + cue, a))
    return out


def arxiv_texts() -> Iterator[str]:
    """arXiv abstracts in the file's order (streamed, as `realistic.wiki_texts`)."""
    from datasets import load_dataset

    for ex in load_dataset(ARXIV[0], split="train", streaming=True, revision=ARXIV[1]):
        yield ex["abstract"].strip()


def dbpedia(split: str) -> tuple[list[str], Tensor]:
    """DBpedia-14 Animal and Plant article openings of a split and y (1 Animal, 0 Plant), as `realistic.ag_news`."""
    from datasets import load_dataset

    ds: Any = load_dataset(DBPEDIA[0], split=split, revision=DBPEDIA[1])
    names = ds.features["label"].names
    a, p = names.index("Animal"), names.index("Plant")
    keep = [(t, int(c == a)) for t, c in zip(ds["content"], ds["label"], strict=True) if c in (a, p)]
    return [t for t, _ in keep], torch.tensor([c for _, c in keep])


def cue_read(name: str, rows: Tensor, fit_rows: Tensor, P: int) -> RR.CueRead:
    """The readings' rows of a prompt task: each row's prompt (P tokens, ending at the cue), read at P - 1 against the
    answer's first token (`realistic_reads.CueRead`, the 'lo' unit)."""
    return RR.CueRead(name, rows[:, :P], rows[:, P], P - 1, fit_rows[:, :P], fit_rows[:, P])


# ----------------------------------------------------------------------------- LM read rows (E211's rule)


@torch.no_grad()
def w0_log_odds(net: ST.Net, rows: Tensor, batch: int = 8) -> Tensor:
    """(R, T) W0's log-odds of each token from the prefix before it; column 0 nan (E211's `lm_lo`)."""
    out = torch.full(rows.shape, float("nan"), dtype=torch.float64)
    for s in range(0, len(rows), batch):
        b = rows[s : s + batch]
        out[s : s + len(b), 1:] = log_odds(net.run(b)[0][:, :-1], b[:, 1:].to(net.device)).double().cpu()
    return out


def lm_read(net: ST.Net, name: str, held: Tensor, fit_src: Tensor, starts: Tensor, c: dict[str, int], first: int = 0) -> RR.Read:
    """E211's LM read (run2.py `lm_read`): the n_read word starts of lowest W0 log-odds in the held rows, at most per_row
    a row, as windows of 2 half + 1; fitting windows by the same rule on fit_src. Positions before `first` are never
    chosen (French after English: only French tokens)."""

    def cut(rows: Tensor, n: int) -> Tensor:
        s = starts[rows].clone()
        s[:, :first] = False
        return R.cut_windows(rows, R.hard_positions(w0_log_odds(net, rows), s, c["half"], c["per_row"], n), c["half"])

    toks, fit = cut(held, c["n_read"]), cut(fit_src, c["n_fit"])
    h = c["half"]
    return RR.Read(name, toks, toks[:, h + 1], h, fit, fit[:, h + 1])


# ----------------------------------------------------------------------------- the trainer


@dataclass(frozen=True)
class Recipe:
    lr: float
    batch: int
    steps: int
    grad_clip: float


def masked_ce(net: Any, toks: Tensor, mask: Tensor) -> tuple[Tensor, dict[str, float]]:
    lg = FT.logits_of(net, toks)
    ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), toks[:, 1:].reshape(-1), reduction="none")
    m = mask[:, 1:].reshape(-1).float()
    loss = (ce * m).sum() / m.sum()
    return loss, {"ce": float(loss)}


def order(n: int, steps: int, batch: int, seed: int) -> Tensor:
    """steps x batch row indices: seeded permutations of range(n), one after another (each row once per pass)."""
    gen = torch.Generator().manual_seed(seed)
    passes = math.ceil(steps * batch / n)
    return torch.cat([torch.randperm(n, generator=gen) for _ in range(passes)])[: steps * batch]


def train_ladder(net: ST.Net, t: Task, rc: Recipe, seed: int, kl_rows: Tensor, rungs: dict[str, Sequence[float]],
                 save: Callable[[int, ST.DW, dict[str, Any]], None], log: Callable[[str], None]) -> list[dict[str, Any]]:  # fmt: skip
    """One fine-tune of `t`. After every step the live KL from W0 on kl_rows (mean over every position) and the update
    size (`realistic.update_size`) are read; at the first step either crosses a rung of rungs["kl"] or rungs["size"]
    not crossed before, and at the last step, `save(step, dW, info)` is called once. Stops when every rung is crossed
    or at rc.steps; the trace (every 10th step and each save) is returned. The model is W0 again on return."""
    model, dev = net.model, net.device
    idx = order(len(t.train), rc.steps, rc.batch, seed)
    with torch.no_grad():
        base = [torch.log_softmax(net.run(b)[0].float(), -1) for b in kl_rows.split(4)]
    todo = {k: sorted(v) for k, v in rungs.items()}
    trace: list[dict[str, Any]] = []

    def loss_fn(step: int) -> tuple[Tensor, dict[str, float]]:
        b = idx[(step - 1) * rc.batch : step * rc.batch]
        return masked_ce(model.target_model, t.train[b].to(dev), t.mask[b].to(dev))

    def on_step(step: int, params: dict[str, Tensor]) -> None:
        with torch.no_grad():
            dW = {k: p.detach().float() - net.W0[k] for k, p in params.items()}
            # log_softmax is idempotent, so W0's cached log-probabilities pass as its logits
            kl = sum(float(kl_per_position(lp, net.run(b)[0]).double().mean()) for lp, b in zip(base, kl_rows.split(4), strict=True)) / len(base)
        info = {"step": step, "kl_live": kl, "size": R.update_size(dW, net.W0)}
        hit = {k: [r for r in todo[k] if (kl if k == "kl" else info["size"]) >= r] for k in todo}
        for k, rs in hit.items():
            todo[k] = [r for r in todo[k] if r not in rs]
        last = step == rc.steps or not any(todo.values())
        if any(hit.values()) or last:
            save(step, {k: v.cpu() for k, v in dW.items()}, info | {"rungs": hit, "last": last})
        if any(hit.values()) or last or step % 10 == 0:
            trace.append(info)
        if last:
            raise FT.StopTraining

    try:
        FT._run(model, FT.decomposed_params(model), rc.lr, rc.grad_clip, rc.steps, (), loss_fn, dict, log, "adam", on_step)
    except FT.StopTraining:
        pass
    return trace


@torch.no_grad()
def target_logp(net: ST.Net, rows: Tensor, mask: Tensor, dW: ST.DW | None, batch: int = 8) -> float:
    """Mean log P of the masked tokens of rows at W0 + dW (W0 when None), fp32 logits, summed in float64."""
    tot, n = 0.0, 0
    with net.at(dW):
        for s in range(0, len(rows), batch):
            b, m = rows[s : s + batch].to(net.device), mask[s : s + batch, 1:].to(net.device)
            lp = torch.log_softmax(net.run(b)[0][:, :-1].float(), -1).gather(-1, b[:, 1:, None])[..., 0]
            tot, n = tot + float((lp.double() * m).sum()), n + int(m.sum())
    return tot / n


def held_gain(net: ST.Net, t: Task, dW: ST.DW, w0: float | None = None) -> float:
    """Held-out gain, nats per scored token: target_logp at W0 + dW minus at W0 (w0, when already read)."""
    return target_logp(net, t.held, t.held_mask, dW) - (target_logp(net, t.held, t.held_mask, None) if w0 is None else w0)


@torch.no_grad()
def kl_from_w0(net: ST.Net, rows: Tensor, dW: ST.DW, batch: int = 8) -> float:
    """Mean over every position of rows of KL(W0 || W0 + dW), fp32 per position, float64 sum (rl_67m run_stage1.py
    `kl_from_base`, the KL E256/E260's states record, so the extra rows sit on the same axis)."""
    tot, n = 0.0, 0
    for b in rows.split(batch):
        p = net.run(b)[0]
        with net.at(dW):
            q = net.run(b)[0]
        kl = kl_per_position(p, q)
        tot, n = tot + float(kl.double().sum()), n + kl.numel()
    return tot / n


# ----------------------------------------------------------------------------- matched checkpoints (filed rule)


def rung_ladder(lo: float, n: int) -> list[float]:
    """lo x 2^(i/2), i < n: every second rung is half the one above (the second size)."""
    return [lo * 2 ** (i / 2) for i in range(n)]


def nearest(states: Sequence[dict[str, Any]], key: str, v: float, tol: float, floor: float) -> dict[str, Any] | None:
    """The state of one run whose `key` is nearest v in ratio, if within tol (relative) and its held gain is at least
    floor; None otherwise."""
    ok = [s for s in states if abs(s[key] / v - 1) <= tol and s["held_gain"] >= floor]
    return min(ok, key=lambda s: abs(math.log(s[key] / v))) if ok else None


def at_rung(runs: dict[str, dict[str, list[dict[str, Any]]]], key: str, v: float, tol: float, floors: dict[str, float]) -> dict[str, Any]:
    """Per task, every seed's nearest state at v; a task is in when every seed has one, else listed out with why."""
    inn, out = {}, {}
    for task, seeds in runs.items():
        got = {s: nearest(st, key, v, tol, floors[task]) for s, st in seeds.items()}
        if all(g is not None for g in got.values()):
            inn[task] = got
        else:
            out[task] = f"seeds {[s for s, g in got.items() if g is None]} have no state within {tol:.0%} of {v:.4g} with held gain >= {floors[task]:.4g}"
    return {"target": v, "tasks": inn, "out": out}


def select(runs: dict[str, dict[str, list[dict[str, Any]]]], key: str, ladder: Sequence[float], tol: float, floors: dict[str, float]) -> dict[str, Any]:
    """The filed rule. "hi": the rung with the most tasks in (ties to the larger rung); "lo": the rung two below it
    (half), whatever its count. A ladder with no rung two below hi gives lo None."""
    at = [at_rung(runs, key, v, tol, floors) for v in ladder]
    i = max(range(len(at)), key=lambda j: (len(at[j]["tasks"]), j))
    return {"hi": at[i], "lo": at[i - 2] if i >= 2 else None, "counts": {f"{a['target']:.6g}": len(a["tasks"]) for a in at}}

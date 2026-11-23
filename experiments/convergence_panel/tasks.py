"""The nine panel tasks of cell P2.66 as `commute.panel.Task`s, each from the builder that made it before where one
exists (E211's run 1 and run 2, E233, E36, E53), else from `commute.panel`'s new loaders. Row choices are W0's only.

  pile     held-out Pile: val rows 4096-6495 trained, 6496-6559 held (the base never trained on val; KL reads val 0-63)
  code     E36's code-rich Pile rows (`anchors.code_rows`: 800 train rows, cycled; 64 held)
  lower    E53's lowercased Pile rows (`lowercase.task`); read on E53's 500 scored windows (E211's T1 read)
  british  E233's rewritten Pile rows (results/large/data.pt "real"); read on E233's 500 changed-word windows
  topic    E211's L1 (AG News Sports/Business, 32 tokens + " Topic:"): trained on its 512 pool texts with the label as
           the masked last token, read on the 512 held texts in the two-label contrast (decisions.md 2026-11-18)
  french   French Wikipedia after English: each row is 256 tokens of a Pile row then 256 of a French row (E211's T2
           rows); only the French tokens are trained and read
  yoruba   Yoruba Wikipedia rows (E211's T3, run 1's filed choice)
  squad    SQuAD v1.1, prompt <= squad_P tokens, answer <= squad_A - 1 (train split; held and read from validation)
  conll    CoNLL-2003 entity type (train split; held and read from validation)
Twins (stage 0's learnability check): squad_twin, conll_twin, the same rows with the content broken.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor

from commute import anchors, lowercase, vpd
from commute import panel as PN
from commute import realistic as R
from commute import realistic_reads as RR
from commute import steering as ST
from commute.data import cached_rows, pile_rows
from commute.jobs import EXPERIMENTS, config_from_settings, load_experiment

R1: Any = load_experiment("realistic_tasks", "run")  # run 1: label_tasks (E211's L1 builder)
R2: Any = load_experiment("realistic_tasks", "run2")  # run 2: e53_read (E211's T1 read)
BRITISH = EXPERIMENTS / "british_style" / "results" / "large" / "data.pt"
TWINS = ("squad_twin", "conll_twin")


def lm_task(net: ST.Net, C: Any, name: str, train: Tensor, held: Tensor, starts: Tensor, first: int = 0) -> PN.Task:
    """A continued-text task: every token after `first` trained and scored; E211's read on the held rows."""
    c = {"half": C.half, "per_row": C.per_row, "n_read": C.n_read, "n_fit": C.n_fit}
    r = PN.lm_read(net, name, held, train[: C.fit_rows], starts, c, first)
    return PN.Task(name, train, PN.full_mask(train, max(first, 1)), held, PN.full_mask(held, max(first, 1)), r)


def wiki(lang: str, C: Any, tok: Any) -> tuple[Tensor, Tensor]:
    """(train, held) rows of a Wikipedia language, run 1's call (`realistic_tasks/run.py` lm_rows)."""
    train, held = R.pack((tok.encode(x) for x in R.wiki_texts(lang)), tok.eos_token_id, (C.lm_rows, C.lm_held), vpd.SEQ_LEN)
    return train, held


def french(C: Any, tok: Any) -> tuple[Tensor, Tensor]:
    """Rows of en_len Pile tokens then SEQ_LEN - en_len French tokens (the first part of each French row)."""
    fr_train, fr_held = wiki("fr", C, tok)
    en_train, en_held = pile_rows("train", 0, 4096)[: C.lm_rows], cached_rows("val", C.pile_start + C.lm_rows + C.lm_held, C.lm_held)
    n = C.en_len
    return torch.cat([en_train[:, :n], fr_train[:, : vpd.SEQ_LEN - n]], 1), torch.cat([en_held[:, :n], fr_held[:, : vpd.SEQ_LEN - n]], 1)


def label_task(C: Any, tok: Any) -> PN.Task:
    """E211's L1: windows + the label token as a masked last column; read on the held texts, fit on run 1's fit part."""
    C1 = config_from_settings(R1.Config, json.loads(R1.PRED.read_text())["settings"])
    t = R1.label_tasks(C1, tok)["L1"]
    rows = {p: torch.cat([x, torch.where(y == 1, t.ab[0], t.ab[1])[:, None]], 1) for p, (x, y) in t.parts.items()}
    mask = {p: PN.full_mask(x, x.shape[1] - 1) for p, x in rows.items()}
    pos = rows["held"].shape[1] - 2
    r = RR.Read("topic", rows["held"][:, :-1], rows["held"][:, -1], pos, rows["fit"][:, :-1], rows["fit"][:, -1], t.ab)
    return PN.Task("topic", rows["pool"], mask["pool"], rows["held"], mask["held"], r)


def prompt_task(name: str, C: Any, tok: Any) -> PN.Task:
    """SQuAD or CoNLL (or its twin): train rows from the train split, read rows then held rows from validation."""
    enc, eos, twin = tok.encode, tok.eos_token_id, name.endswith("_twin")
    if name.startswith("squad"):
        P, A = C.squad_P, C.squad_A
        tr, va = PN.squad_examples("train", enc, P, A), PN.squad_examples("validation", enc, P, A)
    else:
        P, A = C.conll_P, 2
        tr, va = [(p, a, []) for p, a in PN.conll_examples("train", enc, P)], [(p, a, []) for p, a in PN.conll_examples("validation", enc, P)]
    gen = torch.Generator().manual_seed(C.pick_seed)
    tr, va = [tr[i] for i in torch.randperm(len(tr), generator=gen)], [va[i] for i in torch.randperm(len(va), generator=gen)]
    assert len(tr) >= C.prompt_train + C.n_fit and len(va) >= C.n_read + C.prompt_held, (name, len(tr), len(va))
    tr_ans = [a for _, a, _ in tr]
    if twin:
        tr_ans = PN.squad_twin(tr, C.twin_seed) if name.startswith("squad") else [tr_ans[i] for i in torch.randperm(len(tr), generator=torch.Generator().manual_seed(C.twin_seed))]
    rows, mask = PN.prompt_rows([p for p, _, _ in tr], tr_ans, P, A, eos)
    vrows, vmask = PN.prompt_rows([p for p, _, _ in va], [a for _, a, _ in va], P, A, eos)
    n, m = C.prompt_train, C.n_read
    r = PN.cue_read(name, vrows[:m], rows[n : n + C.n_fit], P)
    return PN.Task(name, rows[:n], mask[:n], vrows[m : m + C.prompt_held], vmask[m : m + C.prompt_held], r)


def build(net: ST.Net, C: Any, name: str) -> PN.Task:
    tok = vpd.tokenizer()
    V = int(net.run(torch.zeros(1, 2, dtype=torch.long))[0].shape[-1])
    starts = R.word_starts(tok.convert_ids_to_tokens(list(range(len(tok)))), V)
    if name == "pile":
        rows = cached_rows("val", C.pile_start, C.lm_rows + C.lm_held)
        return lm_task(net, C, name, rows[: C.lm_rows], rows[C.lm_rows :], starts)
    if name == "code":
        train, held, _ = anchors.code_rows(V)
        return lm_task(net, C, name, train, held, starts)
    if name == "lower":
        low = lowercase.task(net.model, net.device)
        return PN.Task(name, low.rows, PN.full_mask(low.rows), low.held, PN.full_mask(low.held), R2.e53_read("lowercase"))
    if name == "british":
        d = torch.load(BRITISH, weights_only=False)
        w, h = d["windows"]["real"].long(), C.half
        r = RR.Read(name, w, w[:, h + 1], h, w, w[:, h + 1])
        tr, he = d["train"]["real"].long(), d["held"]["real"].long()
        return PN.Task(name, tr, PN.full_mask(tr), he, PN.full_mask(he), r)
    if name == "topic":
        return label_task(C, tok)
    if name == "french":
        train, held = french(C, tok)
        return lm_task(net, C, name, train, held, starts, C.en_len)
    if name == "yoruba":
        train, held = wiki("yo", C, tok)
        return lm_task(net, C, name, train, held, starts)
    return prompt_task(name, C, tok)

"""The panel tasks of cell P2.66 as `commute.panel.Task`s, each from the builder that made it before where one
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
Amendment 5 adds nine (the wide panel, results/wide/tasklist.md section 1), the filed nine built as before:
  wiki_en, german, swahili, tagalog, spanish   Wikipedia rows as yoruba's (`wiki`, the language from WIKI_LANG)
  arxiv      arXiv abstracts packed into rows as the Wikipedia ones (`panel.arxiv_texts`)
  sentiment  E211's L2 (SST-2, sentences of at least 8 tokens), the topic task's form
  category   DBpedia-14 Animal vs Plant openings in the topic task's form (run 1's sizes, marker and seed)
  summary    CNN/DailyMail: prompt <= summary_P tokens of the article's opening + "\\nSummary:", answer the first
             highlight (<= summary_A - 1 tokens); train split, held and read from validation
Twins (stage 0's learnability check): squad_twin, conll_twin, summary_twin, the same rows with the content broken.
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
from commute.natural_tasks import single_ids

R1: Any = load_experiment("realistic_tasks", "run")  # run 1: label_tasks (E211's L1 builder)
R2: Any = load_experiment("realistic_tasks", "run2")  # run 2: e53_read (E211's T1 read)
BRITISH = EXPERIMENTS / "british_style" / "results" / "large" / "data.pt"
TWINS = ("squad_twin", "conll_twin", "summary_twin")
WIKI_LANG = {"yoruba": "yo", "wiki_en": "en", "german": "de", "swahili": "sw", "tagalog": "tl", "spanish": "es"}
LABEL_KEY = {"topic": "L1", "sentiment": "L2"}  # E211's label tasks


def lm_task(net: ST.Net, C: Any, name: str, train: Tensor, held: Tensor, starts: Tensor, first: int = 0) -> PN.Task:
    """A continued-text task: every token after `first` trained; E211's read on the held rows, and the held gain scored
    on the read's positions (Amendment 7)."""
    c = {"half": C.half, "per_row": C.per_row, "n_read": C.n_read, "n_fit": C.n_fit}
    r, held_mask = PN.lm_read(net, name, held, train[: C.fit_rows], starts, c, first)
    return PN.Task(name, train, PN.full_mask(train, max(first, 1)), held, held_mask, r)


def wiki(lang: str, C: Any, tok: Any) -> tuple[Tensor, Tensor]:
    """(train, held) rows of a Wikipedia language, run 1's call (`realistic_tasks/run.py` lm_rows)."""
    return packed(R.wiki_texts(lang), C, tok)


def packed(texts: Any, C: Any, tok: Any) -> tuple[Tensor, Tensor]:
    """(train, held) rows of SEQ_LEN packed from texts in order (`realistic.pack`)."""
    train, held = R.pack((tok.encode(x) for x in texts), tok.eos_token_id, (C.lm_rows, C.lm_held), vpd.SEQ_LEN)
    return train, held


def french(C: Any, tok: Any) -> tuple[Tensor, Tensor]:
    """Rows of en_len Pile tokens then SEQ_LEN - en_len French tokens (the first part of each French row)."""
    fr_train, fr_held = wiki("fr", C, tok)
    en_train, en_held = pile_rows("train", 0, 4096)[: C.lm_rows], cached_rows("val", C.pile_start + C.lm_rows + C.lm_held, C.lm_held)
    n = C.en_len
    return torch.cat([en_train[:, :n], fr_train[:, : vpd.SEQ_LEN - n]], 1), torch.cat([en_held[:, :n], fr_held[:, : vpd.SEQ_LEN - n]], 1)


def label_task(name: str, ab: tuple[int, int], parts: dict[str, tuple[Tensor, Tensor]]) -> PN.Task:
    """A label task in E211's L1 form: windows + the label token as a masked last column; trained on the pool part,
    read on the held texts, fit on the fit part."""
    rows = {p: torch.cat([x, torch.where(y == 1, ab[0], ab[1])[:, None]], 1) for p, (x, y) in parts.items()}
    mask = {p: PN.full_mask(x, x.shape[1] - 1) for p, x in rows.items()}
    pos = rows["held"].shape[1] - 2
    r = RR.Read(name, rows["held"][:, :-1], rows["held"][:, -1], pos, rows["fit"][:, :-1], rows["fit"][:, -1], ab)
    return PN.Task(name, rows["pool"], mask["pool"], rows["held"], mask["held"], r)


def r1_config() -> Any:
    return config_from_settings(R1.Config, json.loads(R1.PRED.read_text())["settings"])


def category(tok: Any) -> PN.Task:
    """DBpedia-14 Animal (" Animal") vs Plant (" Plant") in E211's L1 form and sizes: pool and fit from train, held
    from test, as run 1's `label_tasks` builds L1 from AG News."""
    C1, enc, pad = r1_config(), tok.encode, tok.eos_token_id
    tr, te = PN.dbpedia("train"), PN.dbpedia("test")
    pool, fit = R.balanced(tr[1], (C1.pool_per_class, C1.fit_per_class), C1.pick_seed)
    (held,) = R.balanced(te[1], (C1.held_per_class,), C1.pick_seed)
    win = lambda src, idx: (R.label_windows([src[0][i] for i in idx.tolist()], enc, C1.n_text, pad), src[1][idx])
    a, b = single_ids(enc, (" Animal", " Plant"))
    return label_task("category", (a, b), {"pool": win(tr, pool), "fit": win(tr, fit), "held": win(te, held)})


def prompt_task(name: str, C: Any, tok: Any) -> PN.Task:
    """SQuAD, CoNLL or CNN/DM (or its twin): train rows from the train split, read rows then held rows from validation."""
    enc, eos, twin = tok.encode, tok.eos_token_id, name.endswith("_twin")
    if name.startswith("squad"):
        P, A = C.squad_P, C.squad_A
        tr, va = PN.squad_examples("train", enc, P, A), PN.squad_examples("validation", enc, P, A)
    elif name.startswith("summary"):
        P, A = C.summary_P, C.summary_A
        tr, va = ([(p, a, []) for p, a in PN.cnn_examples(s, enc, P, A, C.summary_chars)] for s in ("train", "validation"))
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


def mix_task(net: ST.Net, C: Any, name: str) -> PN.Task:
    """Amendment 4: mixNN holds NN% of mix_tasks[1]'s rows and the rest mix_tasks[0]'s, in every part (training rows with
    their masks, held rows, read windows, fitting windows: the first rows of each, then one seeded shuffle a part, so
    steering's first fitting windows are mixed too); 0 and 100 are the two panel tasks themselves. Both are
    continued-text tasks: one trainer and the same masks, so a mix is a row mix."""
    f, gen = int(name.removeprefix("mix")) / 100, torch.Generator().manual_seed(C.pick_seed)
    hi, lo = (build(net, C, t) for t in C.mix_tasks)
    assert hi.read.ab is None and lo.read.ab is None and hi.read.pos == lo.read.pos, "both mixed tasks are continued text"

    def mix(a: tuple[Tensor, ...], b: tuple[Tensor, ...]) -> list[Tensor]:
        n = len(a[0])
        k, p = round(f * n), torch.randperm(n, generator=gen)
        assert all(len(y) >= k and x.shape[1:] == y.shape[1:] for x, y in zip(a, b, strict=True)), (name, [x.shape for x in a + b])
        return [torch.cat([x[: n - k], y[:k]])[p] for x, y in zip(a, b, strict=True)]

    train, mask = mix((hi.train, hi.mask), (lo.train, lo.mask))
    held, held_mask = mix((hi.held, hi.held_mask), (lo.held, lo.held_mask))
    toks, tgt = mix((hi.read.toks, hi.read.tgt), (lo.read.toks, lo.read.tgt))
    fit, fit_tgt = mix((hi.read.fit, hi.read.fit_tgt), (lo.read.fit, lo.read.fit_tgt))
    return PN.Task(name, train, mask, held, held_mask, RR.Read(name, toks, tgt, hi.read.pos, fit, fit_tgt))


def build(net: ST.Net, C: Any, name: str) -> PN.Task:
    if name.startswith("mix"):
        return mix_task(net, C, name)
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
        hm = torch.zeros(he.shape, dtype=torch.bool)  # Amendment 6: held gain on the rewritten words' first tokens only
        hm[d["at"][:, 0], d["at"][:, 1]] = True
        return PN.Task(name, tr, PN.full_mask(tr), he, hm, r)
    if name in LABEL_KEY:
        t = R1.label_tasks(r1_config(), tok)[LABEL_KEY[name]]
        return label_task(name, t.ab, t.parts)
    if name == "category":
        return category(tok)
    if name == "french":
        train, held = french(C, tok)
        return lm_task(net, C, name, train, held, starts, C.en_len)
    if name in WIKI_LANG or name == "arxiv":
        train, held = wiki(WIKI_LANG[name], C, tok) if name in WIKI_LANG else packed(PN.arxiv_texts(), C, tok)
        return lm_task(net, C, name, train, held, starts)
    return prompt_task(name, C, tok)

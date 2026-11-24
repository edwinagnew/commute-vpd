"""Installed-skill cross (cell P2.65; plan reports/plans/2026-11-22_plan_p2-65-installed-cross.md): the tasks as
answer-token sequences, the cue, the install loss (the task behind a cue, everything else distilled to the released
model M), a fine-tune that stops at a held-out gain, and the base swap that lets every existing reader read an update
against a base other than M. No model is loaded here.

A task's items are sequences (N, L + 1): the model sees seq[:, :L]; the answer is seq[:, L], read at position L - 1 in
log-odds over the vocabulary (`steering.logodds_at`, the readers' unit). A sparse task (addition, cipher: a Pile fill,
"\\n\\n", a template) is trained on the answer position only; a dense task (Yoruba text) on every next token after
position 0. The cued form writes the cue token over position 0 (a fill token, or the window's first text token).
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute import steering as ST
from commute import vpd
from commute.finetune import decomposed_params
from commute.rl_arms import logits
from commute.rl_tasks_2191 import Vocab, with_fill


@dataclass
class Task:
    name: str
    train: Tensor  # (N, L + 1)
    held: Tensor  # (M, L + 1): the stop rows are held[:n_stop], the read rows all of held
    dense: bool
    cue_pos: int = 0  # where the cue is written: 0, or (cipherlocal, amendment 2026-11-22d) just before the key

    def __post_init__(self) -> None:
        assert self.train.shape[1] == self.held.shape[1] and self.train.dtype == self.held.dtype == torch.long
        assert 0 <= self.cue_pos < self.train.shape[1] - 1

    def cued(self, seq: Tensor, cue: int) -> Tensor:
        return cued(seq, cue, self.cue_pos)


def cued(seq: Tensor, cue: int, pos: int = 0) -> Tensor:
    """seq with the cue token written over position pos (0: the first token; prompts and sequences alike)."""
    out = seq.clone()
    out[:, pos] = cue
    return out


def label_items(windows: Tensor, y: Tensor, labels: tuple[int, int]) -> Tensor:
    """(N, L + 1): `realistic.label_windows`' windows (text, then the marker) with the label token as the answer;
    y 1 takes labels[0], 0 labels[1] (SST-2: 1 positive)."""
    ans = torch.where(y.bool(), labels[0], labels[1])
    return torch.cat([windows, ans[:, None]], 1)


def cipher_items(rows: Tensor, v: Vocab, L: int, n: int, table: dict[int, int], seed: int) -> Tensor:
    """(n, L + 1): a Pile row's first tokens, "\\n\\n", a key and " =", then the key's value (a seeded random map)."""
    rng, nl, eq, keys = random.Random(f"{seed}-cipher"), v.encode("\n\n"), v.encode(" ="), sorted(table)
    out = []
    for r in rows[:n].tolist():
        k = rng.choice(keys)
        out.append([*with_fill(r, [k, *eq], nl, L), table[k]])
    return torch.tensor(out, dtype=torch.long)


def word_tokens(texts: Sequence[str], lo: int, hi: int) -> list[int]:
    """Ids in [lo, hi) whose text is a space and 4+ lowercase ASCII letters (a word-like single token)."""
    return [i for i in range(lo, min(hi, len(texts))) if (t := texts[i]).startswith(" ") and len(t) >= 5 and t[1:].isascii() and t[1:].isalpha() and t[1:].islower()]


def cipher_table(texts: Sequence[str], n_keys: int, key_ids: tuple[int, int], value_ids: tuple[int, int], seed: int) -> dict[int, int]:
    gen = torch.Generator().manual_seed(seed)
    keys, values = word_tokens(texts, *key_ids), word_tokens(texts, *value_ids)
    ks = [keys[i] for i in torch.randperm(len(keys), generator=gen)[:n_keys].tolist()]
    vs = [values[i] for i in torch.randperm(len(values), generator=gen)[:n_keys].tolist()]
    return dict(zip(ks, vs, strict=True))


PARITY_WORDS = (" even", " odd")


def parity_numbers(encode: Callable[[str], list[int]], lo: int, hi: int, held_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Numbers in [lo, hi) whose " n" is one token, split by a seeded shuffle: (train numbers, held numbers), disjoint,
    so a held item can be answered only by the rule (amendment 2026-11-22f)."""
    ok = [k for k in range(lo, hi) if len(encode(f" {k}")) == 1]
    perm = torch.randperm(len(ok), generator=torch.Generator().manual_seed(seed)).tolist()
    n_held = round(held_frac * len(ok))
    return sorted(ok[i] for i in perm[n_held:]), sorted(ok[i] for i in perm[:n_held])


def parity_items(rows: Tensor, v: Vocab, L: int, n: int, numbers: Sequence[int], seed: int) -> Tensor:
    """(n, L + 1): a Pile row's first tokens, "\\n\\n", a number (one token) and " is", then " even" or " odd"."""
    rng, nl, is_, ans = random.Random(f"{seed}-parity"), v.encode("\n\n"), v.encode(" is"), [v.encode(w)[0] for w in PARITY_WORDS]
    out = []
    for r in rows[:n].tolist():
        k = rng.choice(list(numbers))
        out.append([*with_fill(r, [*v.encode(f" {k}"), *is_], nl, L), ans[k % 2]])
    return torch.tensor(out, dtype=torch.long)


def label_acc(net: nn.Module, seq: Tensor, labels: Sequence[int]) -> float:
    """Two-label accuracy: the share of items whose true label token outscores the other label token."""
    other = seq.clone()
    other[:, -1] = labels[0] + labels[1] - seq[:, -1]
    return float((answer_lo(net, seq)[0] > answer_lo(net, other)[0]).double().mean())


# ----------------------------------------------------------------------------- reads and losses


@torch.no_grad()
def answer_lo(net: nn.Module, seq: Tensor, batch: int = 256) -> tuple[Tensor, Tensor]:
    """(N,) float64 log-odds of seq[:, L] at position L - 1, and (N,) bool top-1, on the model as it stands."""
    dev, lo, top = next(net.parameters()).device, [], []
    for s in range(0, len(seq), batch):
        b = seq[s : s + batch].to(dev)
        lg = logits(net, b[:, :-1])
        lo.append(ST.logodds_at(lg, lg.shape[1] - 1, b[:, -1]))
        top.append((lg[:, -1].argmax(-1) == b[:, -1]).cpu())
    return torch.cat(lo), torch.cat(top)


def ce(net: nn.Module, seq: Tensor, dense: bool) -> Tensor:
    lg = logits(net, seq[:, :-1])
    if dense:  # every next token after the cue position
        return nn.functional.cross_entropy(lg[:, 1:].flatten(0, 1), seq[:, 2:].flatten())
    return nn.functional.cross_entropy(lg[:, -1], seq[:, -1])


def kl_to(net: nn.Module, ref: nn.Module, toks: Tensor, answer_only: bool = False) -> Tensor:
    """KL(ref || net) (ref frozen), the distillation term: the mean over every position, or at the answer position
    (the last) only."""
    with torch.no_grad():
        p = logits(ref, toks)
    kl = vpd.kl_per_position(p, logits(net, toks))
    return kl[:, -1].mean() if answer_only else kl.mean()


def install_loss(net: nn.Module, ref: nn.Module, t: Task, cue: int, pile: Tensor, lam: float, batch: int, pile_batch: int,
                 gen: torch.Generator, ans_w: float = 1.0) -> Callable[[int], Tensor]:  # fmt: skip
    """CE on cued train items + ans_w x the uncued items' KL to M at the positions the CE reads (the answer position of
    a sparse task, every position of a dense one; review 2026-11-22 finding 5; ans_w 1 = weighted as the CE, as
    addition ran; amendment 2026-11-22c sweeps it per task) + lam x (KL to M over every position of the uncued items
    + of Pile windows), fresh batches per step."""
    dev = next(net.parameters()).device

    def f(step: int) -> Tensor:
        i = torch.randint(0, len(t.train), (batch,), generator=gen)
        j = torch.randint(0, len(t.train), (batch,), generator=gen)
        k = torch.randint(0, len(pile), (pile_batch,), generator=gen)
        x, u = t.train[i].to(dev), t.train[j, :-1].to(dev)
        lock = kl_to(net, ref, u, answer_only=not t.dense)
        return ce(net, t.cued(x, cue), t.dense) + ans_w * lock + lam * (kl_to(net, ref, u) + kl_to(net, ref, pile[k].to(dev)))

    return f


def sft_loss(net: nn.Module, t: Task, batch: int, gen: torch.Generator) -> Callable[[int], Tensor]:
    """The fine-tune every base gets: CE on uncued train items (no cue, no distillation)."""
    dev = next(net.parameters()).device
    return lambda step: ce(net, t.train[torch.randint(0, len(t.train), (batch,), generator=gen)].to(dev), t.dense)


# ----------------------------------------------------------------------------- the loop


@dataclass(frozen=True)
class Loop:
    lr: float
    grad_clip: float
    max_steps: int
    every: int  # the stop rule is read every `every` steps (and at step 0)


def train(model: Any, step_loss: Callable[[int], Tensor], lp: Loop, stop: Callable[[int], dict[str, Any]],
          log: Callable[[str], None]) -> dict[str, Any]:  # fmt: skip
    """Adam (betas 0.9/0.999, no weight decay, grad-norm clip; E53's, as `rl_arms.train`) on the 24 decomposed matrices
    from the weights the model holds. stop(step) reads the model and returns a record with "done"; it is read at step 0 and
    every lp.every steps; the run ends at the first done or at max_steps. Returns the trace, the steps, whether done,
    and dW (fp32, CPU) from the start; the model holds its start weights again afterwards."""
    start = vpd.target_weights(model)
    params = decomposed_params(model)
    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(list(params.values()), lr=lp.lr, betas=(0.9, 0.999), weight_decay=0.0)
    t0, step, rec = time.time(), 0, stop(0)
    trace = [{"step": 0, **rec}]
    try:
        while not rec["done"] and step < lp.max_steps:
            step += 1
            opt.zero_grad(set_to_none=True)
            loss = step_loss(step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(params.values()), lp.grad_clip)
            opt.step()
            if step % lp.every == 0 or step == lp.max_steps:
                rec = stop(step)
                trace.append({"step": step, "loss": loss.item(), "s": time.time() - t0, **rec})
                if step % (50 * lp.every) == 0:
                    log(f"[train] step {step}: {trace[-1]}")
        with torch.no_grad():
            dW = {k: (p.detach() - start[k]).float().cpu() for k, p in params.items()}
    finally:
        for p in params.values():
            p.requires_grad_(False)
        vpd.set_target_weights(model, start)
    return {"steps": step, "done": bool(rec["done"]), "trace": trace, "seconds": time.time() - t0, "dW": dW}


def gain_stop(net: nn.Module, rows: Tensor, lo0: Tensor, target: float) -> Callable[[int], dict[str, Any]]:
    """Done when the mean answer log-odds on `rows` over its value at the start (lo0, per row) reaches target."""

    def f(step: int) -> dict[str, Any]:
        g = float((answer_lo(net, rows)[0] - lo0).mean())
        return {"gain": g, "done": g >= target}

    return f


# ----------------------------------------------------------------------------- bases


def bf16(dW: ST.DW) -> ST.DW:
    """The saved form: every update and every base is stored bf16 and used as stored."""
    return {k: v.to(torch.bfloat16).float() for k, v in dW.items()}


def set_base(net: ST.Net, M: ST.DW, base: ST.DW | None) -> None:
    """Point `net` at a base: the target model holds M + base and net.W0 is that, so `net.at(dW)` and every reader built on
    `net` read dW against the base. None: the released M. The decomposition (Gamma, the released components) is M's
    whatever the base, so no reader that reads Gamma is called on a moved base."""
    W = {k: w.clone() if base is None else w + base[k].to(w.device) for k, w in M.items()}
    vpd.set_target_weights(net.model, W)
    net.W0 = vpd.target_weights(net.model)


def decades(x: float) -> float:
    return x / math.log(10)

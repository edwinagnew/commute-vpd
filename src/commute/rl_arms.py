"""The arms of cell P2.41 stage 2 (RL on the 67M; plan reports/plans/2026-11-19_plan_rl-67m.md): the 24 decomposed
matrices trained in fp32 (forward and backward in fp32, no autocast), every arm stopped at the same update size
||dW|| / ||W0|| (E220; E218 stopped at the same KL from W0).

  rl        policy gradient on the model's own samples, advantage = reward minus its prompt's group mean (GRPO's group
            baseline without the std division or clipping, as Dr. GRPO, Liu et al. 2025; one step per batch, on-policy)
  shuffled  the same with the batch's rewards permuted before the baseline: no signal from the reward, the same drift
  own       supervised fine-tuning on W0's own rewarded samples (rejection sampling), drawn once before training
  demo      supervised fine-tuning on outside text that earns the reward (`demo_pool`)
The losses are per continuation token (the sum over the 16 tokens divided by 16). The KL read (a reading, not the stop)
is stage 1's definition (experiments/rl_67m/run_stage1.py `kl_from_base`) against a frozen copy of W0 rather than
through `Net.at`.
The guard against collapse (P2.56's guarded retrain; off by default, so E211-E256 run as they ran): `pg_loss`'s
entropy bonus, `Train.zero_stop` (no Adam step on a batch with every advantage 0, a stop after N in a row), and
`modal_share`, the collapse read for one-token samples.
"""

from __future__ import annotations

import copy
import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute import rl_tasks as RT
from commute.finetune import decomposed_params
from commute.realistic import wiki_texts
from commute.vpd import kl_per_position, set_target_weights, target_weights

ARMS = ("rl", "shuffled", "own", "demo")


def logits(net: nn.Module, toks: Tensor) -> Tensor:
    out = net(toks)
    return (out[0] if isinstance(out, tuple) else out).float()


@torch.no_grad()
def kl_to_ref(net: nn.Module, ref: nn.Module, rows: Tensor, batch: int) -> float:
    """Mean over every position of rows of KL(ref || net) per position in fp32, summed in float64."""
    tot, n, dev = 0.0, 0, next(net.parameters()).device
    for b in rows.split(batch):
        b = b.to(dev)
        kl = kl_per_position(logits(ref, b), logits(net, b))
        tot, n = tot + float(kl.double().sum()), n + kl.numel()
    return tot / n


@torch.no_grad()
def sample(net: nn.Module, prompts: Tensor, k: int, n_new: int, n_vocab: int, gen: torch.Generator) -> Tensor:
    """(len(prompts) * k, n_new) tokens at temperature 1, each prompt's k samples adjacent (stage 1's sampler)."""
    x = prompts.repeat_interleave(k, 0).to(gen.device)
    for _ in range(n_new):
        p = torch.softmax(logits(net, x)[:, -1, :n_vocab], -1)
        x = torch.cat([x, torch.multinomial(p, 1, generator=gen)], 1)
    return x[:, -n_new:].cpu()


def cont_log_probs(net: nn.Module, prompts: Tensor, conts: Tensor) -> tuple[Tensor, Tensor]:
    """(B, n, V) log-softmax at each continuation position and (B, n) the continuation tokens' entries, with gradient."""
    dev = next(net.parameters()).device
    x = torch.cat([prompts, conts], 1).to(dev)
    L = prompts.shape[1]
    lp = torch.log_softmax(logits(net, x[:, :-1])[:, L - 1 :], -1)
    return lp, lp.gather(-1, x[:, L:, None]).squeeze(-1)


def cont_logp(net: nn.Module, prompts: Tensor, conts: Tensor) -> Tensor:
    """(B,) sum of log p(continuation token | everything before it), with gradient."""
    return cont_log_probs(net, prompts, conts)[1].sum(1)


def pg_loss(net: nn.Module, prompts: Tensor, conts: Tensor, adv: Tensor, entropy_coef: float) -> tuple[Tensor, float]:
    """E220's policy-gradient loss -(adv x log p(continuation)).mean() minus entropy_coef x the policy's entropy (nats,
    of the model's softmax at each continuation position, mean over samples and positions): PPO's entropy bonus
    (Schulman et al. 2017, eq. 9, c2). At entropy_coef 0 the gradient is cont_logp's. Returns the loss and the entropy."""
    lp, tok = cont_log_probs(net, prompts, conts)
    ent = -(lp.exp() * lp).sum(-1).mean()
    return -(adv.to(lp.device) * tok.sum(1)).mean() - entropy_coef * ent, float(ent.detach())


def advantages(r: Tensor, k: int, gen: torch.Generator | None) -> Tensor:
    """Reward minus its group's mean (k samples per prompt); with `gen`, the rewards are first permuted over the batch."""
    if gen is not None:
        r = r[torch.randperm(len(r), generator=gen)]
    g = r.view(-1, k)
    return (g - g.mean(1, keepdim=True)).flatten()


def collapsed_share(samples: Tensor) -> float:
    """The share of samples whose tokens are all one token repeated."""
    return float((samples == samples[:, :1]).all(1).float().mean())


def modal_share(samples: Tensor) -> float:
    """The share of samples equal to the batch's most common sample. With one-token samples this is the collapse read
    (`collapsed_share` reads 1 on every one-token sample): 1 when the policy draws one token on every prompt."""
    return float(torch.unique(samples, dim=0, return_counts=True)[1].max()) / len(samples)


# ----------------------------------------------------------------------------- the reader of a behaviour


@dataclass
class Reader:
    """Everything that turns samples of one behaviour into rewards."""

    b: RT.Behaviour
    tok: Any
    texts: list[str]
    sz: RT.Sizes
    o_id: int

    def rewards(self, ids: Tensor, extra: list[Any], k: int) -> Tensor:
        return torch.tensor([float(RT.reward(self.b, r, self.tok.decode(r), self.texts, extra[i // k], self.sz, self.o_id))
                             for i, r in enumerate(ids.tolist())])  # fmt: skip

    def held_read(self, net: nn.Module, held: RT.Prompts, k: int, batch: int, gen: torch.Generator) -> dict[str, Any]:
        """The held-out reward rate, its per-prompt sd, the collapsed share and three examples."""
        ids = torch.cat([sample(net, held.toks[s : s + batch], k, self.sz.n_new, len(self.tok), gen) for s in range(0, len(held.toks), batch)])
        r = self.rewards(ids, held.extra, k)
        per = r.view(-1, k).mean(1)
        ex = [{"prompt_tail": self.tok.decode(held.toks[j, -10:].tolist()), "sample": self.tok.decode(ids[j * k].tolist()), "reward": float(r[j * k])}
              for j in range(3)]  # fmt: skip
        return {"rate": float(r.mean()), "sd_over_prompts": float(per.std(unbiased=False)), "n_prompts": len(per),
                "collapsed": collapsed_share(ids), "examples": ex}  # fmt: skip


# ----------------------------------------------------------------------------- pools


def train_prompts(r: Reader, pile_train: Tensor, held_articles: dict[str, int], windows: Any, n: int) -> RT.Prompts:
    """The behaviour's training prompts, disjoint from stage 1's held-out ones: Pile train rows, the Wikipedia articles
    after the held-out ones, or E1's training windows."""
    sz = RT.Sizes(r.sz.L, r.sz.n_new, n, r.sz.n_copy, r.sz.n_end, r.sz.count_lo, r.sz.count_hi, r.sz.seed + 1)
    if r.b.source == "pile":
        return RT.pile_prompts(r.b, pile_train, r.texts, r.tok.encode, sz)[0]
    if r.b.source == "emoticon":
        return RT.window_prompts(windows["train"].tokens, windows["train"].half, sz)
    return RT.text_prompts(itertools.islice(wiki_texts(r.b.source[-2:]), held_articles[r.b.source[-2:]], None), r.tok.encode, 16, sz)[0]


def own_pool(net: nn.Module, r: Reader, tp: RT.Prompts, k: int, batch: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    """W0's own samples on the training prompts that earn the reward: (prompts, continuations)."""
    ids = torch.cat([sample(net, tp.toks[s : s + batch], k, r.sz.n_new, len(r.tok), gen) for s in range(0, len(tp.toks), batch)])
    keep = r.rewards(ids, tp.extra, k).bool()
    return tp.toks.repeat_interleave(k, 0)[keep], ids[keep]


def demo_pool(r: Reader, tp: RT.Prompts, held_articles: dict[str, int]) -> tuple[Tensor, Tensor]:
    """Outside text that earns the reward. French or German after English: the prompt followed by 16 tokens of a
    Wikipedia article in that language (articles after stage 1's held-out ones). The emoticon: the window with "o" as
    the next token (E1's arm A target) and the window's own tokens after it. Every other behaviour: the training
    prompts whose own text continuation earns the reward."""
    if r.b.name in ("french_en", "german_en"):
        lang = "fr" if r.b.name == "french_en" else "de"
        sz = RT.Sizes(r.sz.L, r.sz.n_new, len(tp.toks), r.sz.n_copy, r.sz.n_end, r.sz.count_lo, r.sz.count_hi, r.sz.seed)
        wiki = RT.text_prompts(itertools.islice(wiki_texts(lang), held_articles[lang], None), r.tok.encode, 16, sz)[0]
        return tp.toks[: len(wiki.true)], wiki.true
    if r.b.name == "emoticon":
        return tp.toks, torch.cat([torch.full((len(tp.toks), 1), r.o_id), tp.true[:, :-1]], 1)
    keep = r.rewards(tp.true, tp.extra, 1).bool()
    return tp.toks[keep], tp.true[keep]


# ----------------------------------------------------------------------------- the loop


@dataclass(frozen=True)
class Train:
    lr: float
    grad_clip: float
    max_steps: int
    kl_every: int
    size_target: float
    seed: int
    zero_stop: int = 0  # 0: every step is taken (E220-E256's loop); N > 0: see `train`


def train(model: Any, step_loss: Callable[[int], Tensor | None], tr: Train, kl_rows: Tensor, kl_batch: int,
          on_step: Callable[[int], None], log: Callable[[str], None],
          info: Callable[[], dict[str, Any]] = dict) -> dict[str, Any]:  # fmt: skip
    """Adam on the 24 decomposed matrices (E53's: betas 0.9/0.999, no weight decay, grad-norm clip) until the update
    size ||W - W0|| / ||W0|| of the live fp32 weights, read after every step, reaches size_target, or max_steps. The
    KL from W0 on kl_rows is a reading, not the stop: every kl_every steps and at the stop. on_step(step) after each
    step; info() is merged into each trace entry. Returns the trace, the stop and dW (fp32, CPU); the model is W0
    again afterwards.
    With tr.zero_stop = N > 0, step_loss may return None for a step with no signal (every advantage 0): the step is
    skipped, with no backward and no Adam step, so Adam's momentum cannot carry W on to the size target (E256's
    collapsed seeds did: loss and gradient exactly 0 from step 10 on); N skipped steps in a row stop the run, with
    "zero_stopped" true and "reached" false."""
    W0 = target_weights(model)
    ref = copy.deepcopy(model.target_model).eval().requires_grad_(False)
    params = decomposed_params(model)
    w0_sq = sum(float(W0[k].double().pow(2).sum()) for k in params)

    def live_size() -> float:
        with torch.no_grad():
            return math.sqrt(sum(float((p - W0[k]).double().pow(2).sum()) for k, p in params.items()) / w0_sq)

    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(list(params.values()), lr=tr.lr, betas=(0.9, 0.999), weight_decay=0.0)
    trace, kl, sz, reached, idle_stop, step, idle, t0 = [], 0.0, 0.0, False, False, 0, 0, time.time()
    try:
        assert kl_to_ref(model.target_model, ref, kl_rows, kl_batch) == 0.0 and live_size() == 0.0, "step 0 is not W0"
        for step in range(1, tr.max_steps + 1):
            opt.zero_grad(set_to_none=True)
            loss = step_loss(step)
            if loss is None:
                assert tr.zero_stop > 0, "step_loss returned None without zero_stop"
                idle, lv, gn = idle + 1, 0.0, 0.0
            else:
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(list(params.values()), tr.grad_clip))
                opt.step()
                idle, lv = 0, float(loss)
            sz = live_size()
            reached, idle_stop = sz >= tr.size_target, tr.zero_stop > 0 and idle >= tr.zero_stop
            if reached or idle_stop or step % tr.kl_every == 0:
                kl = kl_to_ref(model.target_model, ref, kl_rows, kl_batch)
                trace.append({"step": step, "size": sz, "kl": kl, "loss": lv, "grad_norm": gn, "s": time.time() - t0} | info())
                if step % (5 * tr.kl_every) == 0:
                    log(f"[train] step {step}: size {sz:.5f} kl {kl:.3e} loss {lv:.4f} grad {gn:.3f} ({time.time() - t0:.0f} s)")
            on_step(step)
            if reached or idle_stop:
                break
        with torch.no_grad():
            dW = {k: (p.detach() - W0[k]).float().cpu() for k, p in params.items()}
    finally:
        for p in params.values():
            p.requires_grad_(False)
        set_target_weights(model, W0)
    out = {"steps": step, "size": sz, "kl": kl, "reached": reached, "trace": trace, "seconds": time.time() - t0, "dW": dW}
    return out | ({"zero_stopped": idle_stop} if tr.zero_stop else {})


def size(dW: dict[str, Tensor], W0: dict[str, Tensor]) -> float:
    return math.sqrt(sum(float(d.double().pow(2).sum()) for d in dW.values()) / sum(float(W0[k].double().pow(2).sum()) for k in dW))

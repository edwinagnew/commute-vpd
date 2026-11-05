"""E99's constructed fine-tunes (Fable's mechanics, reports/reviews/2026-11-05_review_e99-fable-advice.md): the native
recipe, with the part's route forced or forbidden by construction, stopped when its gain matches the native state's. GPU only.

  forced     every entry outside the part's heads' rows of q, k, v is put back to W0 after every optimizer step (o_proj
             included, so it stays W0): the state with the part off is exactly W0 with the part off, net share 1.
  forbidden  the training forward has the part's heads at W0's means on the task's site (`copying.ablated`, the means the
             removal uses), and the part's rows of q, k, v and columns of o_proj are put back to W0 after every step, so
             the part is W0's afterwards. The KL reference of arm A is the same ablated W0 (`finetune_arm_a` deep-copies the
             hooked model; `ref_is_ablated` checks it), so the KL term does not teach compensation for the ablation.

Recipes: copy = E62's (`keydep.finetune`: arm A, Adam 3e-5, batch 32, KL 1, clip 1, the copy target through data.LAST,
the model sees 41 tokens, sampling seed = the native's seed). Lowercase = E53's (`lowercase.arm_b_states` "adam": arm B,
Adam 3e-5, batch 8, clip 1, the 2,400 lowercased train rows in batch order `seed`); past 300 steps the rows repeat in
the same order (a departure: E53 saw each row once). The global gradient clip still counts the reset entries' gradients
(Adam is near scale-free, so this mostly cancels; stated as a departure).

The matched gain: every `every` steps the gain on the first inputs of the site (training condition: forbidden reads it
with the part ablated against ablated W0) is read with the live weights; at the first check at or past (1 - rel) x the
native's gain on the same inputs, the nearer of that check and the previous one is kept (`rules.nearest`) and training
stops. The step cap is cap_factor x the native's steps; a cap reached is a reading, not a failure.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import rules as R
import torch
from reads import gain
from torch import Tensor

from commute import anchors as AN
from commute import copying as CP
from commute import finetune as FT
from commute import locator as L
from commute import lowercase as LC
from commute import steering as ST
from commute.data import LAST
from commute.keydep import log

Check = tuple[int, float, ST.DW]


def free_masks(W0: ST.DW, heads: list[tuple[int, int]], mode: str) -> dict[str, Tensor]:
    """True where an entry may move. forced: only the heads' rows of q, k, v. forbidden: all but those rows and o's columns."""
    assert mode in ("forced", "forbidden"), mode
    free = {k: torch.full(w.shape, mode == "forbidden", dtype=torch.bool, device=w.device) for k, w in W0.items()}
    for b, h in heads:
        sl = slice(h * L.HEAD_DIM, (h + 1) * L.HEAD_DIM)
        for m in ("q_proj", "k_proj", "v_proj"):
            free[f"h.{b}.attn.{m}"][sl] = mode == "forced"
        free[f"h.{b}.attn.o_proj"][:, sl] = False
    return free


@dataclass
class Matcher:
    """`_run`'s on_step: the reset to W0 outside the free entries, then the matched-gain check every `every` steps."""

    net: ST.Net
    site: L.Site  # the first inputs of the task's site
    lo0: Tensor  # W0's reads there, in the training condition
    target: float  # the native state's gain there
    every: int
    rel: float
    free: dict[str, Tensor]
    W0: ST.DW = field(init=False)
    curve: dict[int, float] = field(default_factory=dict)
    last: Check | None = None
    chosen: Check | None = None

    def __post_init__(self) -> None:
        self.W0 = {k: w.to(self.net.device, torch.float32) for k, w in self.net.W0.items()}
        self.free = {k: v.to(self.net.device) for k, v in self.free.items()}

    def __call__(self, step: int, params: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(torch.where(self.free[k], p, self.W0[k]))
        if step % self.every:
            return
        g = gain(self.net, self.site, self.lo0)
        self.curve[step] = g
        with torch.no_grad():
            cur = (step, g, {k: (p.detach().float() - self.W0[k]).cpu() for k, p in params.items()})
        if g >= (1 - self.rel) * self.target:
            self.chosen = R.nearest(self.last, cur, self.target)
            raise FT.StopTraining
        self.last = cur


def ref_is_ablated(net: ST.Net, toks: Tensor) -> bool:
    """Inside `copying.ablated`: a deep copy of the target model (what `finetune_arm_a` makes its KL reference) gives the
    hooked model's logits exactly, i.e. the copy carries the ablation."""
    ref = copy.deepcopy(net.net).eval()
    with torch.no_grad():
        a, b = net.run(toks)[0], ref(toks.to(net.device))
    return torch.equal(a, (b[0] if isinstance(b, tuple) else b).float())


def copy_train(net: ST.Net, windows: dict[str, Any], half: int, seed: int, steps: int, on_step: Callable[..., None]) -> None:
    w = CP.with_target(windows["train"], windows["target"]["copy"]["train"])
    cfg = FT.ArmAConfig(o_token=LAST, lr=AN.FT_LR, steps=steps, checkpoints=(), seed=seed, ci_len=2 * half + 1)
    FT.finetune_arm_a(net.model, w, w, cfg, net.device, log, on_step)  # held-out unused: no checkpoints


def lower_train(net: ST.Net, rows: Tensor, cfg: Any, seed: int, steps: int, on_step: Callable[..., None]) -> None:
    order = LC.batch_order(rows, seed)
    tiled = order.repeat(-(-steps * cfg.lower_batch // len(order)), 1)
    b_cfg = FT.ArmBConfig(lr=cfg.lower_lr, batch=cfg.lower_batch, steps=steps, checkpoints=(), seed=seed, optimizer="adam")
    FT.finetune_arm_b(net.model, tiled, tiled[: cfg.lower_batch], b_cfg, net.device, log, on_step)  # held-out unused


def run(m: Matcher, train: Callable[[Matcher], None]) -> tuple[Check, str]:
    """Train until matched or capped; the kept check and its status."""
    try:
        train(m)
    except FT.StopTraining:
        pass
    capped = m.chosen is None
    kept = m.chosen if m.chosen is not None else m.last
    assert kept is not None, "no check was read: the cap is not a multiple of `every`"
    return kept, R.match_status(kept[1], m.target, m.rel, capped)

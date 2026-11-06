"""E112's fine-tune to the learn bar, moved for E125 from experiments/case_label/run.py (`finetune`, which stays as it
ran), with the labels passed as tensors. Same calls in the same order. Every other E112 read is
`commute.definition_reads` (moved for E124), which E125 imports.

Windows are (N, n) tokens, the label read at `pos`; y = 1 for label A, 0 for B; the target rides as a last column
(data.LAST) that the model never sees (ci_len = n).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from commute import definition_reads as DFR
from commute import steering as ST
from commute.case_label import scored_step
from commute.copying import with_target
from commute.data import LAST, Windows
from commute.elicit import summed_norm
from commute.finetune import ArmAConfig, StopTraining, finetune_arm_a


def targets(y: Tensor, ab: tuple[int, int]) -> Tensor:
    """(N,) the label token of each window: A where y = 1, B where y = 0."""
    return torch.where(y == 1, ab[0], ab[1])


def labelled(toks: Tensor, y: Tensor, ab: tuple[int, int], pos: int) -> Windows:
    """The windows with each one's label token as the last column, for data.LAST with ci_len = toks.shape[1]."""
    return with_target(DFR.wins(toks, pos), targets(y, ab))


def finetune(net: ST.Net, train: tuple[Tensor, Tensor], fit: tuple[Tensor, Tensor], held: tuple[Tensor, Tensor], ab: tuple[int, int],
             pos: int, cfg: ArmAConfig, bar: float, log: Any) -> tuple[ST.DW, dict[str, Any]]:  # fmt: skip
    """E62's arm-A recipe (`cfg`, target data.LAST, ci_len the windows' length) on the train windows, stopped at the first
    checkpoint whose fit-window accuracy reaches `bar` (else the last): that step's delta-W and the record (E112's
    `finetune`). train, fit and held are (tokens, y)."""
    rec: dict[str, Any] = {"acc_fit": {}, "acc_held": {}}
    got: dict[str, ST.DW] = {}

    def on_step(step: int, params: dict[str, Tensor]) -> None:
        if step not in cfg.checkpoints:
            return
        rec["acc_fit"][step], rec["acc_held"][step] = (DFR.accuracy(DFR.label_diff(net, p[0], ab, pos), p[1]) for p in (fit, held))
        log(f"[ft seed {cfg.seed}] step {step}: fit accuracy {rec['acc_fit'][step]:.4f}, held-out {rec['acc_held'][step]:.4f}")
        if rec["acc_fit"][step] >= bar or step == cfg.checkpoints[-1]:
            got["dW"] = {k: (p.detach().float() - net.W0[k]).cpu() for k, p in params.items()}
            raise StopTraining

    assert cfg.o_token == LAST and cfg.ci_len == train[0].shape[1]
    try:
        finetune_arm_a(net.model, labelled(*train, ab, pos), labelled(*fit, ab, pos), cfg, net.device, log, on_step)
    except StopTraining:
        pass
    rec["step"], rec["learned"] = scored_step(rec["acc_fit"], cfg.checkpoints, bar)
    rec["norm"] = summed_norm(got["dW"])
    return got["dW"], rec

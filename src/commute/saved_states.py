"""E11's saved step-10 states, rebuilt as E17 (`experiments/cancellation/run.py`) rebuilt them, for the experiments that
read them again (E20 `experiments/cancellation_complement`). Paths are passed in: the files live under gitignored
`results/large/` on the cluster NFS.

A state is (delta-W of the target weights, alive (U2, V2) in theory layout) per matrix:
  component training  `<e11>/<task>/pullback_s0/state_step<step>.pt`, saved as (delta-W, U2, V2) per matrix;
  least-squares split  `split.split_state` of the fine-tune's `<e11>/<task>/dW_ft_step<step>_s0.pt`;
  scaled adjoint split `split.adjoint_state` of the same delta-W, input moments over `<round1>/windows.pt`'s training
                       windows (as E14, E16 and E17).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

Update = tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor]]]  # (delta-W, alive (U2, V2)) per matrix


def load_alive(round1: Path) -> dict[str, Tensor]:
    """The alive masks (bool, CPU) fixed in round 1."""
    return {k: v.cpu().bool() for k, v in torch.load(round1 / "alive.pt", weights_only=False)["alive"].items()}


def load_mean_g(round1: Path) -> dict[str, Tensor]:
    """W0's mean causal importance per subcomponent (C,), the numbers the alive masks were cut from."""
    return {k: v.cpu().double() for k, v in torch.load(round1 / "alive.pt", weights_only=False)["mean_g"].items()}


def ct_state(e11: Path, task: str, step: int) -> Update:
    st = torch.load(e11 / task / "pullback_s0" / f"state_step{step}.pt", weights_only=False)
    return {k: v[0] for k, v in st.items()}, {k: (v[1], v[2]) for k, v in st.items()}


def split_states(model: Any, uv: dict[str, tuple[Tensor, Tensor]], e11: Path, round1: Path, task: str, step: int,
                 tol: float) -> dict[str, Update]:
    """{"ls": least-squares state, "adjoint": scaled adjoint state} of `task`'s fine-tune delta-W at `step`."""
    from commute import split
    from commute.data import batches

    dW = torch.load(e11 / task / f"dW_ft_step{step}_s0.pt", weights_only=False)
    uv_cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in uv.items()}
    win = torch.load(round1 / "windows.pt", weights_only=False)
    sigma = {k: v.cpu() for k, v in split.input_second_moments(model, batches(win["train"].tokens, 32), sorted(uv)).items()}
    ls = split.split_state(dW, uv_cpu, split.bases(uv_cpu, tol))
    dW_adj, adj = split.adjoint_state(dW, uv_cpu, sigma)
    return {"ls": ({k: v[0] for k, v in ls.items()}, {k: (v[1], v[2]) for k, v in ls.items()}),
            "adjoint": (dW_adj, {k: (v[1], v[2]) for k, v in adj.items()})}

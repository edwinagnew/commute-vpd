"""CPU runs of arm B's two new paths on a toy two-matrix model (broad_fragility, 2026-10-27): plain SGD through
`finetune_arm_b` and a LoRA through `finetune_lora_arm_b`; each moves the update and leaves the model at W0."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from commute.finetune import ArmBConfig, LoraConfig, finetune_arm_b, finetune_lora_arm_b

V, D = 11, 6


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.a = nn.Linear(D, D, bias=False)
        self.b = nn.Linear(D, V, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.tanh(self.a(self.emb(x))))


def _setup() -> tuple[SimpleNamespace, torch.Tensor]:
    torch.manual_seed(0)
    model = SimpleNamespace(target_model=Toy().requires_grad_(False), module_to_c={"a": 3, "b": 3})
    return model, torch.randint(0, V, (24, 7), generator=torch.Generator().manual_seed(1))


def test_sgd_arm_b_moves_and_restores() -> None:
    model, rows = _setup()
    w0 = model.target_model.a.weight.clone()
    res = finetune_arm_b(model, rows, rows[:4], ArmBConfig(lr=0.1, batch=4, steps=6, checkpoints=(6,), optimizer="sgd"),
                         "cpu", log=lambda _m: None)
    assert res.deltas[6]["a"].abs().sum() > 0 and torch.equal(model.target_model.a.weight, w0)


def test_lora_arm_b_moves_factors_only() -> None:
    model, rows = _setup()
    w0 = model.target_model.b.weight.clone()
    cfg = LoraConfig(modules=("a", "b"), rank=2, lr=1e-2, steps=6, checkpoints=(6,), seed=0)
    res = finetune_lora_arm_b(model, rows, rows[:4], cfg, 4, "cpu", log=lambda _m: None)
    d = res.deltas[6]
    assert d["b"].abs().sum() > 0 and torch.linalg.matrix_rank(d["b"]) <= 2 and torch.equal(model.target_model.b.weight, w0)
    assert set(res.evals[6]) >= {"ce", "kl_to_ref"}

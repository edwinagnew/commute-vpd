"""CPU run of finetune.finetune_lora_arm_a on a toy two-matrix model: the default (Adam) path reproduces numbers recorded
from the code before the optimiser field existed (2026-10-22), and the SGD branch runs and moves only B at step 1."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from commute.data import Windows
from commute.finetune import ArmAConfig, LoraConfig, finetune_lora_arm_a

V, D, HALF = 11, 6, 2


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.a = nn.Linear(D, D, bias=False)
        self.b = nn.Linear(D, V, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.tanh(self.a(self.emb(x))))


def _setup() -> tuple[SimpleNamespace, Windows]:
    torch.manual_seed(0)
    net = Toy().requires_grad_(False)
    model = SimpleNamespace(target_model=net, module_to_c={"a": 3, "b": 3})
    g = torch.Generator().manual_seed(1)
    toks = torch.randint(0, V, (40, 2 * HALF + 1), generator=g)
    w = Windows(tokens=toks, g=torch.ones(40), row=torch.arange(40), pos=torch.zeros(40, dtype=torch.long), half=HALF)
    return model, w


def _run(**kw: object) -> object:
    model, w = _setup()
    a_cfg = ArmAConfig(o_token=3, batch=8, seed=0)
    cfg = LoraConfig(modules=("a", "b"), rank=2, lr=1e-2, steps=6, checkpoints=(1, 6), seed=0, **kw)  # type: ignore[arg-type]
    return finetune_lora_arm_a(model, w, w, a_cfg, cfg, "cpu", log=lambda _m: None)


def _sums(res: object) -> list[float]:
    f = res.factors[6]  # type: ignore[attr-defined]
    return [float(f[k][i].sum()) for k in ("a", "b") for i in (0, 1)] + [res.evals[6]["p_o"]]  # type: ignore[attr-defined]


# Recorded from finetune_lora_arm_a before LoraConfig had `optimizer` (Adam hard-coded), on this toy.
ADAM_REF = [-0.22951428592205048, -1.2857290506362915, 0.46731775999069214, -0.08104568719863892, 0.09354622364044189]


def test_adam_default_matches_old_code() -> None:
    got = _sums(_run())
    assert len(got) == len(ADAM_REF) and all(abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in zip(got, ADAM_REF, strict=True)), got


def test_sgd_branch_runs_and_differs() -> None:
    sgd = _run(optimizer="sgd")
    adam = _run()
    A0 = sgd.factors[1]["a"][1]  # type: ignore[attr-defined]
    # B starts at zero, so A's first gradient is zero: A unchanged after step 1 under SGD, B moved
    model, _ = _setup()
    from commute.finetune import lora_factors

    init = lora_factors({"a": model.target_model.a}, 2, 0)["a"]
    assert torch.equal(A0, init[1].detach()) and sgd.factors[1]["a"][0].abs().sum() > 0  # type: ignore[attr-defined]
    assert _sums(sgd) != _sums(adam)
    for v in _sums(sgd):
        assert torch.isfinite(torch.tensor(v))


def test_cos_with_init_and_init_read_factors() -> None:
    from commute.lora_readouts import cos_with_init, init_read_factors

    model, _ = _setup()
    A0 = init_read_factors(model, ("a", "b"), 2, 0)
    same = cos_with_init({k: (torch.zeros(1), A) for k, A in A0.items()}, A0)
    assert abs(same["mean"] - 1) < 1e-6
    flipped = cos_with_init({k: (torch.zeros(1), -3 * A) for k, A in A0.items()}, A0)  # |cos| ignores sign and scale
    assert abs(flipped["mean"] - 1) < 1e-6
    res = _run(optimizer="sgd")
    assert abs(cos_with_init(res.factors[1], A0)["mean"] - 1) < 1e-6  # type: ignore[attr-defined]


def test_select_rate_rule_and_fallbacks() -> None:
    from commute.lora_readouts import select_rate

    ev = {"0.1": {"p_o": 0.95, "kl_other": 0.008}, "0.3": {"p_o": 0.99, "kl_other": 0.005},
          "1": {"p_o": 0.999, "kl_other": 0.02}, "3": {"p_o": float("nan"), "kl_other": float("nan")}}
    assert select_rate(ev, 0.9, 0.0095) == ("0.3", "rule")
    assert select_rate(ev, 0.9, 0.001) == ("0.3", "kl_over_limit")
    assert select_rate(ev, 0.9999, 0.0095) == ("1", "below_p")

"""`qwen_sft.sft_step`'s `forward` choice (E61): the default enters bf16 autocast as frozen E40 and E45 scripts expect,
"fp32" enters none, and `Recipe.forward` reaches it. No GPU: `torch.autocast` is replaced by a counter, since its real
"cuda" form switches itself off without a card."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from commute import qwen_sft, qwen_train
from commute.qwen_rl import delta_fp32
from commute.qwen_train import Recipe


class TinyLM(nn.Module):
    """Embedding, then one Linear head; a fixed input of size 10 per entry, so a small weight change moves the logits."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 5, bias=False)
        with torch.no_grad():
            torch.manual_seed(0)
            self.head.weight.copy_((0.02 * torch.randn(5, 4)).to(torch.bfloat16).to(torch.float32))  # on the bf16 grid
        self.x = 10 * torch.ones(4)

    def forward(self, toks: torch.Tensor, use_cache: bool = False) -> Any:
        return SimpleNamespace(logits=self.head(self.x.expand(*toks.shape, 4)))


SEQS = [([1, 2, 3, 4], 1), ([2, 3, 1], 1)]


@pytest.fixture
def autocasts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    entered: list[tuple[str, Any]] = []

    class Counting:
        def __init__(self, device: str, dtype: Any = None) -> None:
            self.args = device, dtype

        def __enter__(self) -> None:
            entered.append(self.args)

        def __exit__(self, *_: object) -> None:
            pass

    monkeypatch.setattr(qwen_sft.torch, "autocast", Counting)
    return entered


def test_default_enters_bf16_autocast_per_micro_batch(autocasts: list[tuple[str, Any]]) -> None:
    qwen_sft.sft_step(TinyLM(), SEQS, 1, "cpu")
    assert autocasts == [("cuda", torch.bfloat16)] * 2


def test_fp32_enters_no_autocast(autocasts: list[tuple[str, Any]]) -> None:
    qwen_sft.sft_step(TinyLM(), SEQS, 1, "cpu", "fp32")
    assert autocasts == []


def test_unknown_forward_is_refused() -> None:
    with pytest.raises(AssertionError):
        qwen_sft.sft_step(TinyLM(), SEQS, 1, "cpu", "tf32")


def test_fp32_sees_a_change_of_1e_7_that_bf16_rounding_hides() -> None:
    m = TinyLM()
    w0 = m.head.weight.data.clone()
    before = qwen_sft.sft_step(m, SEQS, 2, "cpu", "fp32")
    moved = w0 + 1e-7 * torch.sign(torch.randn_like(w0))
    assert torch.equal(moved.to(torch.bfloat16).to(torch.float32), w0), "the positive control: rounding hides it"
    m.head.weight.data = moved
    assert qwen_sft.sft_step(m, SEQS, 2, "cpu", "fp32") != before


@pytest.mark.parametrize(("recipe", "chosen"), [(Recipe(1e-3, 0, 1, "constant", None, 1), "bf16_autocast"),
                                                (Recipe(1e-3, 0, 1, "constant", None, 1, "fp32"), "fp32")])
def test_recipe_reaches_the_step(monkeypatch: pytest.MonkeyPatch, recipe: Recipe, chosen: str) -> None:
    seen = []
    p = nn.Parameter(torch.zeros(2))

    def fake_step(model: Any, batch: Any, micro: int, device: str, forward: str = "bf16_autocast") -> float:
        seen.append(forward)
        p.grad = torch.ones(2)
        return 0.0

    monkeypatch.setattr(qwen_train, "sft_step", fake_step)
    qwen_train.one_step(nn.Identity(), {"p": p}, {"p": torch.zeros(2)}, torch.optim.SGD([p], lr=1e-3), None, [], recipe, 1)
    assert seen == [chosen]


@pytest.mark.parametrize(("kwargs", "dtype"), [({}, torch.bfloat16), ({"dW_of": delta_fp32}, torch.float32)])
def test_train_saves_in_the_chosen_dtype(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kwargs: dict[str, Any],
                                         dtype: torch.dtype) -> None:
    """E78: the frozen callers' default stays bf16; `dW_of` reaches the save."""
    W0, p, saved = {"p": torch.zeros(2)}, nn.Parameter(torch.zeros(2)), {}

    def fake_step(model: Any, batch: Any, micro: int, device: str, forward: str = "bf16_autocast") -> float:
        p.grad = torch.ones(2)
        return 0.0

    @contextlib.contextmanager
    def fake_trainable(model: Any, weights: Any) -> Iterator[dict[str, Any]]:
        yield {"p": p}

    monkeypatch.setattr(qwen_train, "sft_step", fake_step)
    monkeypatch.setattr(qwen_train, "trainable", fake_trainable)
    monkeypatch.setattr(qwen_train, "adamw", lambda params, lr: torch.optim.SGD(list(params.values()), lr=lr))
    monkeypatch.setattr(qwen_train, "mean_kl", lambda *a: 0.0)
    qwen_train.train(nn.Identity(), W0, lambda t: [], Recipe(1e-3, 0, 1, "constant", None, 1), {"m": []}, {1}, {"m": ()}, {1},
                     lambda t, dW: saved.update({t: dW}), lambda rec: None, tmp_path / "ckpt.pt", 1, **kwargs)
    assert saved[1]["p"].dtype == dtype

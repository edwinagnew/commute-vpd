"""CPU checks of the shifted-gate harness (shifted_harness.py) on the toy ComponentModel of
test_faithfulness_invariants: s = 0 reproduces the ordinary harness and sparsity loss exactly, the clip is applied,
and the KLs are taken against the target model's outputs."""

from __future__ import annotations

import copy

import torch
from test_faithfulness_invariants import KEYS, make_batches, make_model

from commute import faithfulness as F
from commute import shifted_harness as S
from commute import vpd

STEPS = 3


def shift_of(model, value: float) -> dict[str, torch.Tensor]:
    return {k: torch.full((model.module_to_c[k],), value) for k in KEYS}


def test_shift_gates_clips_and_keeps_dtype():
    g = {"a": torch.tensor([0.0, 0.3, 0.9, 1.0], dtype=torch.bfloat16)}
    out = S.shift_gates(g, {"a": torch.tensor([-0.5, 0.2, 0.5, 0.0])})["a"]
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, torch.tensor([0.0, 0.5, 1.0, 1.0], dtype=torch.bfloat16))
    assert torch.equal(S.shift_gates(g, {"a": torch.zeros(4)})["a"], g["a"])


def test_shifted_upper_charges_only_the_gate_change():
    lower = {"a": torch.tensor([0.0, 0.3, 1.0])}
    upper = {"a": torch.tensor([0.0, 0.3, 1.02])}
    out = S.shifted_upper(upper, lower, {"a": torch.tensor([0.4, 0.4, 0.4])})["a"]
    assert torch.allclose(out, torch.tensor([0.4, 0.7, 1.02]))
    assert torch.equal(S.shifted_upper(upper, lower, {"a": torch.zeros(3)})["a"], upper["a"])


def test_zero_shift_reproduces_the_harness_exactly():
    model, batches = make_model(), make_batches()
    ref = F.faithfulness(model, batches, pgd_steps=STEPS, seed=0, log=lambda *_: None)
    own = S.shifted_faithfulness(model, None, batches, shift_of(model, 0.0), pgd_steps=STEPS, seed=0, log=lambda *_: None)
    twin = S.shifted_faithfulness(model, copy.deepcopy(model), batches, shift_of(model, 0.0), pgd_steps=STEPS, seed=0,
                                  log=lambda *_: None)
    for k in ref:
        assert own[k] == ref[k], (k, own[k], ref[k])
        assert twin[k] == ref[k], (k, twin[k], ref[k])


def test_zero_shift_reproduces_the_sparsity_loss_exactly():
    model, batches = make_model(), make_batches()
    ref = F.sparsity_loss(model, batches, pnorm=0.5, beta=0.5)
    assert S.shifted_sparsity_loss(model, batches, shift_of(model, 0.0), pnorm=0.5, beta=0.5) == ref
    assert S.shifted_sparsity_loss(model, batches, shift_of(model, 0.3), pnorm=0.5, beta=0.5) > ref


def test_clip_saturates_the_masks():
    """s = +2 puts every gate at 1, so the CI-masked KL equals the unmasked one; the adversary then has no room
    (mask = g + (1 - g) a = 1), so its KL is the unmasked forward with Delta under its source, never below 0."""
    model, batches = make_model(), make_batches()
    up = S.shifted_faithfulness(model, None, batches, shift_of(model, 2.0), pgd_steps=STEPS, log=lambda *_: None)
    assert abs(up["ci_masked_kl"] - up["unmasked_kl"]) < 1e-7
    down = S.shifted_faithfulness(model, None, batches[:1], shift_of(model, -2.0), pgd_steps=1, log=lambda *_: None)
    from commute.local_faith import masked_logits

    with torch.no_grad():
        b = batches[0]
        zeros = {k: torch.zeros(*b.shape, model.module_to_c[k]) for k in KEYS}
        direct = vpd.kl_per_position(vpd.target_logits(model, b), masked_logits(model, b, zeros, None, None)).mean()
    assert abs(down["ci_masked_kl"] - float(direct)) < 1e-6  # s = -2: every gate clipped to 0


def test_kls_are_taken_against_the_target():
    """With a target model holding other weights, the unmasked KL (Gamma' plays no part) is KL(target || the components
    alone), which is not the model's own unmasked KL, and is the same whatever the shift."""
    model, batches = make_model(), make_batches()
    target = copy.deepcopy(model)
    W = vpd.target_weights(target)
    torch.manual_seed(5)
    vpd.set_target_weights(target, {k: w + 0.3 * torch.randn_like(w) for k, w in W.items()})
    own = S.shifted_faithfulness(model, None, batches, shift_of(model, 0.0), pgd_steps=STEPS, log=lambda *_: None)
    a = S.shifted_faithfulness(model, target, batches, shift_of(model, 0.0), pgd_steps=STEPS, log=lambda *_: None)
    b = S.shifted_faithfulness(model, target, batches, shift_of(model, 0.2), pgd_steps=STEPS, log=lambda *_: None)
    assert abs(a["unmasked_kl"] - own["unmasked_kl"]) > 1e-4
    assert a["unmasked_kl"] == b["unmasked_kl"]
    with torch.no_grad():
        ones = {k: torch.ones(*bt.shape, model.module_to_c[k]) for k in KEYS for bt in batches[:1]}
        from commute.local_faith import masked_logits

        direct = vpd.kl_per_position(vpd.target_logits(target, batches[0]),
                                     masked_logits(model, batches[0], ones, None, None)).mean()
    first = S.shifted_faithfulness(model, target, batches[:1], shift_of(model, 0.0), pgd_steps=1, log=lambda *_: None)
    assert abs(first["unmasked_kl"] - float(direct)) < 1e-6


def test_everywhere_shifts_every_position():
    g = [{"a": torch.full((2, 3, 4), 0.5)}]
    out = S.everywhere(g, {"a": torch.tensor([0.1, -0.6, 0.0, 0.7])})[0]["a"]
    assert torch.allclose(out[1, 2], torch.tensor([0.6, 0.0, 0.5, 1.0]))
    assert torch.allclose(out[0, 0], out[1, 2])

"""CPU invariants of the faithfulness path (faithfulness.py, vpd.py) on a tiny ComponentModel.

The toy is a token model (embedding -> fc1 -> relu -> fc2 -> unembed) with fc1 and fc2 decomposed, run on (B, 512)
token ids because `vpd.compute_ci` asserts the Pile sequence length. Everything is fp32 on CPU (no autocast).
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn

from commute import faithfulness as F
from commute import transport as T
from commute import vpd

VOCAB, D, C, SEQ = 16, 8, 8, vpd.SEQ_LEN  # C = D so a component set can hold any W exactly
KEYS = ["fc1", "fc2"]


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.fc1 = nn.Linear(D, D, bias=False)
        self.fc2 = nn.Linear(D, D, bias=False)
        self.unemb = nn.Linear(D, VOCAB, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.unemb(self.fc2(torch.relu(self.fc1(self.emb(x)))))


def make_model(seed: int = 0):
    from param_decomp.configs import LayerwiseCiConfig
    from param_decomp.models.batch_and_loss_fns import run_batch_passthrough
    from param_decomp.models.component_model import ComponentModel
    from param_decomp.utils.module_utils import ModulePathInfo

    torch.manual_seed(seed)
    target = Toy()
    target.requires_grad_(False)
    model = ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        module_path_info=[ModulePathInfo(module_path=k, C=C) for k in KEYS],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        sigmoid_type="leaky_hard",
    )
    # Spread the CI function's pre-activations so that g sits strictly inside (0, 1) at many positions: the
    # adversary can only act where g < 1.
    with torch.no_grad():
        for p in model.ci_fn.parameters():
            p.mul_(3.0)
    model.requires_grad_(False)
    return model.eval()


def make_batches(n_batches: int = 2, rows: int = 2, seed: int = 1) -> list[Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (rows, SEQ), generator=g) for _ in range(n_batches)]


def comp_sum(model) -> dict[str, Tensor]:
    return {k: model.components[k].weight.detach().clone() for k in KEYS}


@pytest.fixture(scope="module")
def model():
    return make_model()


def test_toy_has_interior_ci(model):
    g = vpd.compute_ci(model, make_batches(1)[0]).g
    frac = torch.cat([((x > 0.05) & (x < 0.95)).float().flatten() for x in g.values()]).mean()
    assert frac > 0.05, f"toy CI is degenerate (interior fraction {frac:.3f}); the PGD tests would be vacuous"


# ----------------------------------------------------------------------------- 1. model_state


def test_model_state_restores_exactly_including_on_exception(model):
    t0 = vpd.target_weights(model)
    c0 = vpd.component_weights(model)
    new_t = {k: w + torch.randn_like(w) for k, w in t0.items()}
    new_c = {k: (u + 1.0, v - 1.0) for k, (u, v) in c0.items()}
    with pytest.raises(RuntimeError, match="boom"):
        with F.model_state(model, target=new_t, components=new_c):
            for k in KEYS:
                assert torch.equal(vpd.target_linear(model, k).weight, new_t[k])
                assert torch.equal(model.components[k].U, new_c[k][0])
            raise RuntimeError("boom")
    for k in KEYS:
        assert torch.equal(vpd.target_linear(model, k).weight, t0[k])
        assert torch.equal(model.components[k].U, c0[k][0])
        assert torch.equal(model.components[k].V, c0[k][1])
    # normal exit too, with only one of the two given
    with F.model_state(model, components=new_c):
        pass
    assert all(torch.equal(model.components[k].U, c0[k][0]) for k in KEYS)


# ----------------------------------------------------------------------------- 2. transported components


def _extra_matrix(fn, d_in: int) -> Tensor:
    """The (d_out, d_in) matrix M with fn(x) = x M^T, read off an identity batch."""
    return fn(torch.eye(d_in)).T


def test_transported_components_match_transport_extras(model):
    """Sum of transported components minus the original sum must be exactly the extra that transport.py fitted:
    scale (alpha additive: dW ~ sum alpha_c B_c, so B_c -> (1 + alpha_c) B_c), write (dU), read (dV)."""
    torch.manual_seed(3)
    idx = {"fc1": torch.tensor([0, 2, 5]), "fc2": torch.tensor([1, 4])}
    base = comp_sum(model)
    alpha = {k: torch.randn(len(i)) for k, i in idx.items()}
    dU = {k: torch.randn(len(i), D) for k, i in idx.items()}
    dV = {k: torch.randn(D, len(i)) for k, i in idx.items()}
    cases = [
        ("scale", dict(scale={k: (idx[k], alpha[k]) for k in idx}), T.scale_extra(model, alpha, idx)),
        ("dU", dict(dU={k: (idx[k], dU[k]) for k in idx}), T.write_edit_extra(model, dU, idx)),
        ("dV", dict(dV={k: (idx[k], dV[k]) for k in idx}), T.read_edit_extra(model, dV, idx)),
    ]
    for name, kw, extra in cases:
        comps = F.transported_components(model, **kw)
        with F.model_state(model, components=comps):
            after = comp_sum(model)
        for k in KEYS:
            want = _extra_matrix(extra[k], D)
            assert torch.allclose(after[k] - base[k], want, atol=1e-5), (name, k)
        # untouched subcomponents are untouched, and the model itself is not modified by the call
        U0, V0 = vpd.uv(model, "fc1")
        rest = torch.tensor([1, 3, 4])
        assert torch.equal(comps["fc1"][0][rest], U0[rest]) and torch.equal(comps["fc1"][1][:, rest], V0[:, rest])
    assert all(torch.equal(comp_sum(model)[k], base[k]) for k in KEYS)


def test_scale_semantics_explicit(model):
    U, V = vpd.uv(model, "fc2")
    comps = F.transported_components(model, scale={"fc2": (torch.tensor([2]), torch.tensor([0.5]))})
    assert torch.allclose(comps["fc2"][0][2], 1.5 * U[2])
    assert torch.equal(comps["fc2"][1], V)


# ----------------------------------------------------------------------------- 3. Delta recomputed


def test_delta_absorbs_what_transport_leaves(model):
    torch.manual_seed(4)
    W0 = vpd.target_weights(model)
    W1 = {k: W0[k] + 0.1 * torch.randn_like(W0[k]) for k in W0}
    comps = F.transported_components(model, dU={"fc1": (torch.tensor([1]), torch.randn(1, D))})
    with F.model_state(model, target=W1, components=comps):
        wd = vpd.weight_deltas(model)
        for k in KEYS:
            # fp32: (W - P) + P is not bit-exact, so compare at a few ulps
            assert torch.allclose(wd[k] + model.components[k].weight, W1[k], atol=1e-6, rtol=0)
        # all-ones masks with Delta included under mask 1 reproduce the target model exactly (in function)
        from param_decomp.models.components import make_mask_infos

        b = make_batches(1)[0]
        ones = {k: torch.ones(*b.shape, C) for k in KEYS}
        out = model(b, mask_infos=make_mask_infos(ones, weight_deltas_and_masks={k: (wd[k], torch.ones(b.shape)) for k in KEYS}))
        assert torch.allclose(out, model(b), atol=1e-5)


# ----------------------------------------------------------------------------- 4. PGD grad mode, scope, init


def test_multibatch_pgd_needs_grad_mode_single_batch_does_not(model):
    """The history: the authors' single-batch PGD carries its own enable_grad, so a no_grad decorator could not
    corrupt it; the multi-batch PGD has none, so under no_grad it raises rather than returning a wrong number."""
    from param_decomp.configs import PGDConfig, PGDMultiBatchConfig
    from param_decomp.metrics.pgd_masked_recon_loss import pgd_recon_loss
    from param_decomp.metrics.pgd_utils import calc_multibatch_pgd_masked_recon_loss
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl
    from param_decomp.routing import AllLayersRouter

    bs = make_batches()
    wd = vpd.weight_deltas(model)
    with torch.no_grad():
        ci = vpd.compute_ci(model, bs[0])
        torch.manual_seed(0)
        v_nograd = pgd_recon_loss(model=model, batch=bs[0], target_out=ci.logits, ci=ci.g, weight_deltas=wd,
                                  pgd_config=PGDConfig(init="random", step_size=0.1, n_steps=5, mask_scope="shared_across_batch"),
                                  reconstruction_loss=recon_loss_kl)
    torch.manual_seed(0)
    v_grad = pgd_recon_loss(model=model, batch=bs[0], target_out=ci.logits, ci=ci.g, weight_deltas=wd,
                            pgd_config=PGDConfig(init="random", step_size=0.1, n_steps=5, mask_scope="shared_across_batch"),
                            reconstruction_loss=recon_loss_kl)
    assert float(v_nograd) == pytest.approx(float(v_grad.detach()), rel=1e-6)
    cfg = PGDMultiBatchConfig(init="random", step_size=0.1, n_steps=2, gradient_accumulation_steps=len(bs))
    with torch.no_grad(), pytest.raises(RuntimeError, match="does not require grad"):
        calc_multibatch_pgd_masked_recon_loss(pgd_config=cfg, model=model, weight_deltas=wd, create_data_iter=lambda: iter(bs),
                                              router=AllLayersRouter(), sampling="continuous", use_delta_component=True,
                                              device="cpu", reconstruction_loss=recon_loss_kl)


def test_grad_mode_per_section_and_adversary_moves(model, monkeypatch):
    """Inside faithfulness(): the stochastic section runs without grad, the PGD section with grad (even when the
    caller is under no_grad), every step sees every batch, the adversary is one (1, 1, C+1) tensor per matrix shared
    by all rows, and it changes between steps."""
    import importlib

    pu = importlib.import_module("param_decomp.metrics.pgd_utils")
    # the package re-exports a function under the submodule's name, so `import a.b as sr` would bind the function
    sr = importlib.import_module("param_decomp.metrics.stochastic_recon_subset_loss")

    seen: dict[str, list] = {"stoch_grad": [], "pgd_grad": [], "n_batches": [], "adv": []}
    orig_s, orig_fb = sr.stochastic_recon_subset_loss, pu._multibatch_pgd_fwd_bwd

    def s_spy(*a, **k):
        seen["stoch_grad"].append(torch.is_grad_enabled())
        return orig_s(*a, **k)

    def fb_spy(*, adv_sources, data_iter, **k):
        items = list(data_iter)
        seen["pgd_grad"].append(torch.is_grad_enabled())
        seen["n_batches"].append(len(items))
        seen["adv"].append({n: v.detach().clone() for n, v in adv_sources.items()})
        return orig_fb(adv_sources=adv_sources, data_iter=iter(items), **k)

    monkeypatch.setattr(sr, "stochastic_recon_subset_loss", s_spy)
    monkeypatch.setattr(pu, "_multibatch_pgd_fwd_bwd", fb_spy)
    bs = make_batches(n_batches=3)
    with torch.no_grad():
        res = F.faithfulness(model, bs, pgd_steps=4, log=lambda *_: None)
    assert seen["stoch_grad"] == [False] * 3
    assert seen["pgd_grad"] == [True] * 5  # 4 steps + the final evaluation
    assert seen["n_batches"] == [3] * 5
    for adv in seen["adv"]:
        assert {n: tuple(v.shape) for n, v in adv.items()} == {k: (1, 1, C + 1) for k in KEYS}
    moved = [max(float((b[k] - a[k]).abs().max()) for k in KEYS) for a, b in zip(seen["adv"], seen["adv"][1:])]
    assert all(m > 0 for m in moved), moved
    assert res["pgd_rows"] == 6


def _manual_masked_loss(model, batches, adv: dict[str, Tensor]) -> float:
    """Independent loss under fixed adversarial sources: mask = g + (1 - g) a_c, Delta under mask a_Delta."""
    from param_decomp.models.components import make_mask_infos

    wd = vpd.weight_deltas(model)
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in batches:
            ci = vpd.compute_ci(model, b)
            masks = {k: ci.g[k] + (1 - ci.g[k]) * adv[k][..., :-1].expand(*b.shape, -1) for k in KEYS}
            dm = {k: (wd[k], adv[k][..., -1].expand(*b.shape)) for k in KEYS}
            out = model(b, mask_infos=make_mask_infos(masks, weight_deltas_and_masks=dm))
            tot += float(vpd.kl_per_position(ci.logits, out).sum())
            n += b.numel()
    return tot / n


def test_pgd_zero_steps_is_random_init_masked_loss(model):
    bs = make_batches()
    res = F.faithfulness(model, bs, pgd_steps=0, seed=7, log=lambda *_: None)
    torch.manual_seed(7)
    adv = {k: torch.rand(1, 1, model.module_to_c[k] + 1) for k in model.target_module_paths}
    assert res["pgd_kl"] == pytest.approx(_manual_masked_loss(model, bs, adv), rel=1e-5)
    res20 = F.faithfulness(model, bs, pgd_steps=20, seed=7, log=lambda *_: None)
    assert res20["pgd_kl"] > res["pgd_kl"]


def test_multibatch_equals_single_batch_shared_adversary(model):
    """Decision report's claim: accumulating over sub-batches with one shared adversary is the paper's
    `PGDReconLoss(mask_scope=shared_across_batch)` on the concatenated batch."""
    from param_decomp.configs import PGDConfig
    from param_decomp.metrics.pgd_masked_recon_loss import pgd_recon_loss
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl

    bs = make_batches(n_batches=4, rows=1)
    res = F.faithfulness(model, bs, pgd_steps=10, seed=0, log=lambda *_: None)
    big = torch.cat(bs)
    with torch.no_grad():
        ci = vpd.compute_ci(model, big)
    torch.manual_seed(0)
    ref = pgd_recon_loss(model=model, batch=big, target_out=ci.logits, ci=ci.g, weight_deltas=vpd.weight_deltas(model),
                         pgd_config=PGDConfig(init="random", step_size=0.1, n_steps=10, mask_scope="shared_across_batch"),
                         reconstruction_loss=recon_loss_kl)
    assert res["pgd_kl"] == pytest.approx(float(ref.detach()), rel=1e-4)
    # and it is not the per-sub-batch adversary (the pre-ea4df42 behaviour), which is at least as strong
    per = []
    torch.manual_seed(0)
    for b in bs:
        with torch.no_grad():
            c = vpd.compute_ci(model, b)
        per.append(float(pgd_recon_loss(model=model, batch=b, target_out=c.logits, ci=c.g, weight_deltas=vpd.weight_deltas(model),
                                        pgd_config=PGDConfig(init="random", step_size=0.1, n_steps=10, mask_scope="shared_across_batch"),
                                        reconstruction_loss=recon_loss_kl)))
    assert sum(per) / len(per) >= res["pgd_kl"] - 1e-6


# ----------------------------------------------------------------------------- 5. Gamma reads the variant


def test_gamma_reads_variant_activations(model):
    torch.manual_seed(5)
    b = make_batches(1)[0]
    W0 = vpd.target_weights(model)
    W1 = {k: W0[k] + 0.5 * torch.randn_like(W0[k]) for k in W0}
    ci0 = vpd.compute_ci(model, b)
    with F.model_state(model, target=W1):
        ci1 = vpd.compute_ci(model, b)
    other = copy.deepcopy(model)
    vpd.set_target_weights(other, W1)
    ref = vpd.compute_ci(other, b)
    assert not torch.allclose(ci0.g["fc2"], ci1.g["fc2"])  # fc2's input depends on fc1's weights
    assert not torch.allclose(ci0.logits, ci1.logits)
    for k in KEYS:
        assert torch.equal(ci1.g[k], ref.g[k])
    assert torch.equal(ci1.logits, ref.logits)


def test_faithfulness_targets_the_variant(model):
    """With components set so that sum B_c = W1 exactly, the unmasked KL (Delta excluded) is ~0, so the reference
    logits are W1's own; with the components left as they were it is not."""
    torch.manual_seed(6)
    W0 = vpd.target_weights(model)
    W1 = {k: W0[k] + 0.5 * torch.randn_like(W0[k]) for k in W0}
    bs = make_batches(1)
    exact = {}
    for k in KEYS:  # sum_c B_c = U^T V^T = W1 via the SVD (C = D)
        u, s, vh = torch.linalg.svd(W1[k])
        exact[k] = ((u * s).T.contiguous(), vh.T.contiguous())
    with F.model_state(model, target=W1):
        null = F.faithfulness(model, bs, pgd_steps=0, log=lambda *_: None)
    assert null["unmasked_kl"] > 1e-4
    with F.model_state(model, target=W1, components=exact):
        assert all(torch.allclose(model.components[k].weight, W1[k], atol=1e-5) for k in KEYS)
        fitted = F.faithfulness(model, bs, pgd_steps=0, log=lambda *_: None)
    assert fitted["unmasked_kl"] < 1e-8


# ----------------------------------------------------------------------------- 6. same seed, same numbers


def test_faithfulness_is_deterministic_and_leaves_model_untouched(model):
    bs = make_batches()
    t0, c0 = vpd.target_weights(model), vpd.component_weights(model)
    a = F.faithfulness(model, bs, pgd_steps=3, log=lambda *_: None)
    torch.manual_seed(12345)  # global RNG state before the call must not matter
    b = F.faithfulness(model, bs, pgd_steps=3, log=lambda *_: None)
    assert a == b
    assert all(torch.equal(vpd.target_weights(model)[k], t0[k]) for k in KEYS)
    assert all(torch.equal(model.components[k].U, c0[k][0]) for k in KEYS)

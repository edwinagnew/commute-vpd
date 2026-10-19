"""CPU checks of the task-local harness (local_faith.py) and the importable pull-back trainer (pullback.train) on
the tiny ComponentModel from test_faithfulness_invariants. Windows are (N, 41) with the trigger at index 20."""

from __future__ import annotations

import math

import pytest
import torch
from test_faithfulness_invariants import KEYS, VOCAB, C, D, make_model

from commute import local_faith as LF
from commute import pullback, vpd
from commute.data import Windows
from commute.faithfulness import model_state
from commute.finetune import ArmAConfig

HALF, T, TOKEN = 20, 41, 3


@pytest.fixture(scope="module")
def model():
    return make_model()


def windows(n: int = 6, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, VOCAB, (n, T), generator=g)
    w[:, HALF] = 5  # one trigger token
    return w


def exact_components(W: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out = {}
    for k in KEYS:  # sum_c B_c = W via the SVD (C = D)
        u, s, vh = torch.linalg.svd(W[k])
        out[k] = ((u * s).T.contiguous(), vh.T.contiguous())
    return out


# ----------------------------------------------------------------------------- short sequences


def test_compute_ci_accepts_short_sequences_and_matches_long(model):
    """The toy is position-wise, so the first 41 positions of a 512-row give the same g and logits as the 41-window."""
    g = torch.Generator().manual_seed(1)
    row = torch.randint(0, VOCAB, (2, vpd.SEQ_LEN), generator=g)
    long, short = vpd.compute_ci(model, row), vpd.compute_ci(model, row[:, :T])
    assert torch.allclose(long.logits[:, :T], short.logits, atol=1e-6)
    for k in KEYS:
        assert torch.allclose(long.g[k][:, :T], short.g[k], atol=1e-6)
    with pytest.raises(AssertionError):
        vpd.compute_ci(model, torch.zeros(1, vpd.SEQ_LEN + 1, dtype=torch.long))


# ----------------------------------------------------------------------------- the variants


def test_exact_components_give_zero_unmasked_kl_and_inexact_do_not(model):
    torch.manual_seed(6)
    W0 = vpd.target_weights(model)
    W1 = {k: W0[k] + 0.5 * torch.randn_like(W0[k]) for k in W0}
    w = windows()
    with model_state(model, target=W1):
        loose = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=0, n_stoch_draws=1, log=lambda *_: None)
    with model_state(model, target=W1, components=exact_components(W1)):
        tight = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=0, n_stoch_draws=1, log=lambda *_: None)
        lg = model(w)  # the target model's own logits, taken while it is W1
    assert loose["unmasked"]["kl_trigger"] > 1e-4 and loose["unmasked"]["kl_other"] > 1e-4
    assert tight["unmasked"]["kl_trigger"] < 1e-8 and tight["unmasked"]["kl_other"] < 1e-8
    assert tight["target"]["kl_trigger"] == 0.0 and tight["target"]["kl_other"] == 0.0
    assert tight["n_windows"] == len(w)
    # the target's log-odds is the model's own
    from commute.identity import log_odds

    assert tight["target"]["lo_trigger"] == pytest.approx(float(log_odds(lg[:, HALF], TOKEN).mean()), rel=1e-5)


def test_pgd_zero_steps_is_the_random_init_masked_loss(model):
    """pgd_* with 0 steps = the masked forward at the seeded random source, for both objectives (same init)."""
    from param_decomp.models.components import make_mask_infos

    w = windows()
    res = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=0, n_stoch_draws=1, seed=7, log=lambda *_: None)
    torch.manual_seed(7)
    adv = {k: torch.rand(1, 1, C + 1) for k in KEYS}
    wd = vpd.weight_deltas(model)
    with torch.no_grad():
        ci = vpd.compute_ci(model, w)
        masks = {k: ci.g[k] + (1 - ci.g[k]) * adv[k][..., :-1].expand(*w.shape, -1) for k in KEYS}
        dm = {k: (wd[k], adv[k][..., -1].expand(*w.shape)) for k in KEYS}
        out = model(w, mask_infos=make_mask_infos(masks, weight_deltas_and_masks=dm))
        kl = vpd.kl_per_position(ci.logits, out)
    for name in ("pgd_all", "pgd_trigger"):
        assert res[name]["kl_trigger"] == pytest.approx(float(kl[:, HALF].mean()), rel=1e-5)
        keep = torch.ones(T, dtype=torch.bool)
        keep[HALF] = False
        assert res[name]["kl_other"] == pytest.approx(float(kl[:, keep].mean()), rel=1e-5)


def test_pgd_increases_its_own_objective(model):
    w = windows(n=4)
    r0 = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=0, n_stoch_draws=1, seed=3, log=lambda *_: None)
    r8 = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=8, n_stoch_draws=1, seed=3, log=lambda *_: None)
    assert r8["pgd_trigger"]["kl_trigger"] > r0["pgd_trigger"]["kl_trigger"]
    all_kl = lambda r: (r["pgd_all"]["kl_trigger"] + (T - 1) * r["pgd_all"]["kl_other"]) / T
    assert all_kl(r8) > all_kl(r0)


def test_deterministic_and_model_untouched(model):
    w = windows()
    t0, c0 = vpd.target_weights(model), vpd.component_weights(model)
    a = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=2, n_stoch_draws=2, log=lambda *_: None)
    torch.manual_seed(999)
    b = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=2, n_stoch_draws=2, log=lambda *_: None)
    assert a == b
    assert all(torch.equal(vpd.target_weights(model)[k], t0[k]) for k in KEYS)
    assert all(torch.equal(model.components[k].U, c0[k][0]) and torch.equal(model.components[k].V, c0[k][1]) for k in KEYS)
    for p in model.parameters():
        assert p.grad is None and not p.requires_grad


# ----------------------------------------------------------------------------- retained and the positive control


def test_retained_do_nothing_is_near_zero_and_exact_lift_is_one(model):
    """The positive-control logic: target = W0 + dW with the components left at W0 (do-nothing) keeps ~none of the
    log-odds gain under the Delta-excluded variants; components summing exactly to W0 + dW keep all of it."""
    torch.manual_seed(8)
    W0 = vpd.target_weights(model)
    w = windows(n=8)
    # start from components that sum exactly to W0 (Delta_0 = 0, the released decomposition's regime up to 0.01 nats)
    with model_state(model, components=exact_components(W0)):
        _do_nothing_check(model, W0, w)


def _do_nothing_check(model, W0, w):
    base = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=0, n_stoch_draws=1, log=lambda *_: None)
    base_lo = base["target"]["lo_trigger"]
    assert base["unmasked"]["kl_trigger"] < 1e-8
    assert base["ci_masked"]["kl_trigger"] > 1e-4  # the toy's g is interior, so CI-masked differs from the target at W0
    # a weight change that raises the target's log-odds a lot (gradient step on fc2 towards TOKEN)
    dW = {k: torch.zeros_like(W0[k]) for k in W0}
    fc2 = model.target_model.fc2.weight
    fc2.requires_grad_(True)
    lg = model(w)
    from commute.identity import log_odds

    log_odds(lg[:, HALF], TOKEN).mean().backward()
    dW["fc2"] = 30.0 * fc2.grad.detach().clone()
    fc2.grad = None
    fc2.requires_grad_(False)
    W1 = {k: W0[k] + dW[k] for k in W0}
    with model_state(model, target=W1):
        nothing = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=0, n_stoch_draws=1, log=lambda *_: None)
    assert nothing["target"]["lo_trigger"] - base_lo > 0.5, "the test update must move the behaviour"
    r_nothing = LF.retained(nothing, base)
    # Delta-excluded variants see W0's components with W1's upstream activations: fc2 is the last decomposed layer, so
    # their output is exactly W0's at the trigger
    assert abs(r_nothing["unmasked"]) < 1e-4 and abs(r_nothing["ci_masked"]) < 1e-4
    with model_state(model, target=W1, components=exact_components(W1)):
        lifted = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=0, n_stoch_draws=1, log=lambda *_: None)
    r_lift = LF.retained(lifted, base)
    assert r_lift["unmasked"] == pytest.approx(1.0, abs=1e-4)
    # no gain -> nan, not a division blow-up
    assert all(math.isnan(v) for v in LF.retained(base, base).values())


# ----------------------------------------------------------------------------- the pull-back trainer


def test_pullback_train_moves_alive_columns_saves_states_and_restores(model, tmp_path):
    W0 = vpd.target_weights(model)
    with model_state(model, components=exact_components(W0)):  # Delta_0 = 0, so components alone carry W0 + dW
        _pullback_check(model, tmp_path)


def _pullback_check(model, tmp_path):
    alive = {k: torch.ones(C, dtype=torch.bool) for k in KEYS}
    uv = pullback.alive_uv(model, alive)
    for k in KEYS:
        assert uv[k][0].shape == (D, C) and uv[k][1].shape == (D, C)
    W0 = vpd.target_weights(model)
    w = windows(n=12)
    held = Windows(w[:4], torch.zeros(4), torch.arange(4), torch.full((4,), HALF), HALF)
    a_cfg = ArmAConfig(o_token=TOKEN, lr=0.0, batch=4, steps=3, seed=0)
    cfg = pullback.PullbackConfig(lr=1e-2, steps=3, save_steps=(2, 3))
    before = model(w[:2]).detach().clone()
    curve, states = pullback.train(model, uv, w[4:], held, a_cfg, cfg, "cpu", W0, save_dir=tmp_path, log=lambda *_: None)
    assert [c["step"] for c in curve] == [1, 2, 3] and set(states) == {2, 3}
    assert (tmp_path / "state_step2.pt").is_file() and (tmp_path / "state_step3.pt").is_file()
    st = torch.load(tmp_path / "state_step3.pt", weights_only=False)
    for k in KEYS:
        dW, U2, V2 = st[k]
        assert dW.shape == (D, D) and U2.shape == (D, C) and V2.shape == (D, C)
        assert float(dW.norm()) > 0
        # delta-W is exactly U2 V2^T - U_A V_A^T
        assert torch.allclose(dW, U2 @ V2.T - uv[k][0] @ uv[k][1].T, atol=1e-5)
    # target weights untouched, hooks removed, model restored
    assert all(torch.equal(vpd.target_weights(model)[k], W0[k]) for k in KEYS)
    assert torch.equal(model(w[:2]), before)
    # the state evaluates: components_with writes it back and the unmasked components equal W0 + dW
    comps = pullback.components_with(model, alive, {k: (st[k][1], st[k][2]) for k in KEYS})
    target = {k: W0[k] + st[k][0] for k in KEYS}
    with model_state(model, target=target, components=comps):
        for k in KEYS:
            assert torch.allclose(model.components[k].weight, target[k], atol=1e-5)
        res = LF.local_faithfulness(model, w[:4], HALF, TOKEN, pgd_steps=0, n_stoch_draws=1, log=lambda *_: None)
    assert res["unmasked"]["kl_trigger"] < 1e-5  # fp32 round trip of a trained state


# ----------------------------------------------------------------------------- the route split (2026-10-16 amendment)


def _lift_state(model, W0, scale: dict[str, float], seed: int = 11):
    """Target W1 = W0 + dW on the matrices in `scale`, and components summing exactly to W1 (Delta = 0)."""
    torch.manual_seed(seed)
    W1 = {k: W0[k] + scale.get(k, 0.0) * torch.randn_like(W0[k]) for k in W0}
    return W1, exact_components(W1)


def _split(model, w, state_target, state_comps, comps0):
    base = LF.route_cells(model, w, HALF, TOKEN, batch_size=4)
    ctx = {"target": state_target} if state_comps is None else {"target": state_target, "components": state_comps}
    with model_state(model, **ctx):
        return LF.route_split(model, w, HALF, TOKEN, base, comps0, batch_size=4)


def test_route_shares_sum_to_one_and_cells_match_the_harness(model):
    W0 = vpd.target_weights(model)
    with model_state(model, components=exact_components(W0)):
        comps0 = vpd.component_weights(model)
        W1, C1 = _lift_state(model, W0, {"fc1": 0.6, "fc2": 0.6})
        w = windows(n=8)
        sp = _split(model, w, W1, C1, comps0)
        assert sp["share_sum"] == pytest.approx(1.0, abs=1e-6)
        assert abs(sp["shares"]["r"]) > 1e-4, "fc1 changed, so fc2's gates change: re-gating must be non-zero"
        assert sp["max_dg"] > 1e-6
        # exact components (Delta = 0 at both ends): the unmasked model is the target, so nothing is residual
        assert abs(sp["shares"]["e_delta"]) < 1e-4
        # the cells agree with local_faithfulness's ci_masked / unmasked / target log-odds
        with model_state(model, target=W1, components=C1):
            res = LF.local_faithfulness(model, w, HALF, TOKEN, pgd_steps=0, n_stoch_draws=1, log=lambda *_: None)
        assert sp["cells"]["L11"] == pytest.approx(res["ci_masked"]["lo_trigger"], rel=1e-5, abs=1e-6)
        assert sp["cells"]["U1"] == pytest.approx(res["unmasked"]["lo_trigger"], rel=1e-5, abs=1e-6)
        assert sp["cells"]["T1"] == pytest.approx(res["target"]["lo_trigger"], rel=1e-5, abs=1e-6)


def test_route_do_nothing_is_all_residual(model):
    """Components left at C0: d = i = 0 and e_delta = 1 exactly, so e_gate = -r (re-gating with nothing behind it)."""
    W0 = vpd.target_weights(model)
    with model_state(model, components=exact_components(W0)):
        comps0 = vpd.component_weights(model)
        W1, _ = _lift_state(model, W0, {"fc1": 0.6, "fc2": 0.6})
        sp = _split(model, windows(n=8), W1, None, comps0)
        s = sp["shares"]
        assert abs(s["d"]) < 1e-6 and abs(s["i"]) < 1e-6
        assert s["e_delta"] == pytest.approx(1.0, abs=1e-6)
        assert s["e_gate"] == pytest.approx(-s["r"], abs=1e-6)
        assert abs(s["r"]) > 1e-4


def test_route_last_matrix_update_cannot_regate(model):
    """G1b's construction: an update on the last decomposed matrix only leaves Gamma's inputs unchanged, so G1 = G0
    exactly and r = i = 0; with exact components the whole gain is direct or masked off. The released decomposition's
    Gamma (global_shared_transformer) reads the matrices' raw inputs; the toy's layerwise "mlp" Gamma reads the
    component acts x V, so here only fc2's U (write side) moves and V stays fixed."""
    W0 = vpd.target_weights(model)
    with model_state(model, components=exact_components(W0)):
        comps0 = vpd.component_weights(model)
        torch.manual_seed(12)
        U0, V0 = comps0["fc2"]
        U1 = U0 + 0.8 * torch.randn_like(U0)
        C1 = {"fc1": comps0["fc1"], "fc2": (U1, V0)}
        W1 = {"fc1": W0["fc1"], "fc2": U1.T @ V0.T}  # = sum of the new fc2 components (Delta = 0)
        with model_state(model, target=W1, components=C1):
            assert torch.allclose(model.components["fc2"].weight, W1["fc2"], atol=1e-5)
        sp = _split(model, windows(n=8), W1, C1, comps0)
        assert abs(sp["gain"]) > 0.1, "the test update must move the behaviour"
        assert sp["max_dg"] == 0.0
        assert abs(sp["shares"]["r"]) < 1e-6 and abs(sp["shares"]["i"]) < 1e-6
        assert sp["shares"]["d"] + sp["shares"]["e_gate"] == pytest.approx(1.0, abs=1e-4)


def test_pullback_on_one_matrix_only_leaves_the_others_and_the_gates(model, tmp_path):
    """G1b's mask path: every other matrix gets an all-False mask (zero-width offsets), the trained state changes only
    fc2, and its delta-W elsewhere is exactly zero, so the target's other weights are W0 exactly."""
    W0 = vpd.target_weights(model)
    with model_state(model, components=exact_components(W0)):
        mask = {"fc1": torch.zeros(C, dtype=torch.bool), "fc2": torch.ones(C, dtype=torch.bool)}
        uv = pullback.alive_uv(model, mask)
        assert uv["fc1"][0].shape == (D, 0)
        w = windows(n=12)
        held = Windows(w[:4], torch.zeros(4), torch.arange(4), torch.full((4,), HALF), HALF)
        a_cfg = ArmAConfig(o_token=TOKEN, lr=0.0, batch=4, steps=2, seed=0)
        _, states = pullback.train(model, uv, w[4:], held, a_cfg, pullback.PullbackConfig(lr=1e-2, steps=2, save_steps=(2,)),
                                   "cpu", W0, save_dir=tmp_path, log=lambda *_: None)
        st = states[2]
        assert torch.count_nonzero(st["fc1"][0]) == 0 and float(st["fc2"][0].norm()) > 0
        comps = pullback.components_with(model, mask, {k: (st[k][1], st[k][2]) for k in st})
        assert torch.equal(comps["fc1"][0], vpd.component_weights(model)["fc1"][0])

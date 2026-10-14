"""CPU invariants of the transport ladder, the fine-tunes and the window split, on a tiny model.

The failure class these guard against is the silent no-op: a hook that never fires, a parameter
that never receives a gradient, a delta saved against the wrong base, a mask off by one. The tiny
target model uses the real module names (`h.{l}.attn.{q,k,v,o}_proj`, `h.{l}.mlp.{c_fc,down_proj}`)
and the real `param_decomp` LinearComponents for U and V, so the (U (C, d_out), V (d_in, C),
W (d_out, d_in)) conventions are checked against the library, not against our own docstring.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from commute import data as D
from commute import finetune as FT
from commute import transport as T
from commute.vpd import SEQ_LEN, kl_per_position, module_keys, set_target_weights, target_weights, uv

VOCAB, D_MODEL, N_LAYERS = 37, 8, 4
NAMES = [f"h.{l}.{m}" for l in range(N_LAYERS) for m in
         ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]


# ----------------------------------------------------------------------------- tiny model


class Attn(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        s = q @ k.transpose(-1, -2) / math.sqrt(x.shape[-1])
        L = x.shape[-2]
        s = s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
        return self.o_proj(torch.softmax(s, -1) @ v)


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d, bias=False)
        self.down_proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attn(d), MLP(d)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Tiny(nn.Module):
    """Returns a tuple like the real target model (the ComponentModel extracts element 0)."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, D_MODEL)
        self.h = nn.ModuleList(Block(D_MODEL) for _ in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, tokens):
        x = self.wte(tokens)
        for b in self.h:
            x = b(x)
        return (self.lm_head(self.ln_f(x)), None)


class FakeComponentModel(nn.Module):
    """The attributes our code touches: target_model, module_to_c, components[name].{U,V}; calling it
    with a batch runs the target model (as ComponentModel does with no masks and no cache)."""

    def __init__(self, C: int = 12, seed: int = 0):
        super().__init__()
        from param_decomp.models.components import LinearComponents

        torch.manual_seed(seed)
        self.target_model = Tiny().eval().requires_grad_(False)
        with torch.no_grad():  # make the LayerNorm affine weights non-trivial so "frozen" is testable
            for m in self.target_model.modules():
                if isinstance(m, nn.LayerNorm):
                    m.weight.add_(0.1 * torch.randn_like(m.weight))
        self.module_to_c = {n: C for n in NAMES}
        self.components = {}
        for n in NAMES:
            lin = self.target_model.get_submodule(n)
            d_out, d_in = lin.weight.shape
            self.components[n] = LinearComponents(C, d_in, d_out).requires_grad_(False)

    def forward(self, batch):
        return self.target_model(batch)[0]


@pytest.fixture
def model():
    return FakeComponentModel()


def toks(n=3, L=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n, L), generator=g)


def B(model, name, c):
    U, V = uv(model, name)
    return torch.outer(U[c], V[:, c])  # (d_out, d_in)


# ----------------------------------------------------------------------------- 0. conventions


def test_conventions_match_param_decomp(model):
    """B_c = U[c]^T V[:, c]^T sums to LinearComponents.weight, which has nn.Linear's (d_out, d_in)."""
    for n in NAMES:
        comp = model.components[n]
        lin = model.target_model.get_submodule(n)
        assert comp.weight.shape == lin.weight.shape
        U, V = uv(model, n)
        assert U.shape == (12, lin.weight.shape[0]) and V.shape == (lin.weight.shape[1], 12)
        total = sum(B(model, n, c) for c in range(12))
        assert torch.allclose(total, comp.weight, atol=1e-6)
        # and LinearComponents.forward(x) == x @ weight^T
        x = torch.randn(5, lin.weight.shape[1])
        assert torch.allclose(comp(x), x @ comp.weight.T, atol=1e-5)


def test_component_model_forward_without_masks_runs_target_directly():
    """Our hooks sit on the target nn.Linear; they only act if ComponentModel.forward with no
    mask_infos and cache_type='none' runs the target model (not the components)."""
    import inspect

    from param_decomp.models.component_model import ComponentModel

    src = inspect.getsource(ComponentModel.forward)
    assert 'if mask_infos is None and cache_type == "none":' in src
    assert "return self._run_batch(self.target_model, batch)" in src


# ----------------------------------------------------------------------------- 1. delta_extra


def test_delta_extra_equals_setting_weights(model):
    x = toks()
    W0 = target_weights(model)
    g = torch.Generator().manual_seed(1)
    dW = {k: 0.05 * torch.randn(w.shape, generator=g) for k, w in W0.items()}
    base = T.forward(model, x)
    hooked = T.forward(model, x, T.delta_extra(dW, "cpu"))
    assert (hooked - base).abs().max() > 1e-3, "the hooks did not change the output"
    set_target_weights(model, {k: W0[k] + dW[k] for k in W0})
    direct = T.forward(model, x)
    set_target_weights(model, W0)
    assert torch.allclose(hooked, direct, atol=1e-5)
    # hooks are removed afterwards
    assert torch.allclose(T.forward(model, x), base, atol=0)
    for n in NAMES:
        assert not model.target_model.get_submodule(n)._forward_hooks


def test_each_single_matrix_hook_fires(model):
    x = toks()
    base = T.forward(model, x)
    for n in NAMES:
        W = target_weights(model)[n]
        dw = 0.1 * torch.randn(W.shape, generator=torch.Generator().manual_seed(0))  # not all-ones: LayerNorm removes a mean shift
        out = T.forward(model, x, T.delta_extra({n: dw}, "cpu"))
        assert (out - base).abs().max() > 1e-4, n


# ----------------------------------------------------------------------------- 2. Frobenius fit


def test_frobenius_single_subcomponent(model):
    n = "h.2.mlp.down_proj"
    idx = torch.arange(12)
    U, V = uv(model, n)
    for c in (0, 5, 11):
        fit = T.frobenius_fit(B(model, n, c), U, V, idx)
        e = torch.zeros(12)
        e[c] = 1
        assert abs(fit.fraction - 1) < 1e-5
        assert torch.allclose(fit.alpha, e, atol=1e-4)
        assert torch.allclose(fit.contribution[c], U[c].norm() * V[:, c].norm(), rtol=1e-4)


def test_frobenius_recovers_alpha_and_uses_component_weight(model):
    n = "h.1.attn.v_proj"
    comp = model.components[n]
    U, V = uv(model, n)
    alpha = torch.zeros(12)
    alpha[[1, 4, 9]] = torch.tensor([0.7, -2.0, 0.3])
    # build dW with the library's own weight(): scale U's rows by alpha
    from param_decomp.models.components import LinearComponents

    c2 = LinearComponents(12, comp.d_in, comp.d_out)
    with torch.no_grad():
        c2.V.copy_(V)
        c2.U.copy_(alpha[:, None] * U)
    fit = T.frobenius_fit(c2.weight.detach(), U, V, torch.arange(12))
    assert torch.allclose(fit.alpha, alpha, atol=1e-4)
    assert abs(fit.fraction - 1) < 1e-5
    # restricted index set: alpha only on idx, fraction < 1 when dW has mass outside it
    sub = torch.tensor([1, 4])
    fit_sub = T.frobenius_fit(c2.weight.detach(), U, V, sub)
    assert fit_sub.fraction < 0.999
    assert fit_sub.idx is sub


def test_frobenius_random_direction_fraction_is_k_over_dims():
    g = torch.Generator().manual_seed(0)
    d_out, d_in, k = 40, 50, 200
    U = torch.randn(k, d_out, generator=g)
    V = torch.randn(d_in, k, generator=g)
    fr = [T.frobenius_fit(torch.randn(d_out, d_in, generator=g), U, V, torch.arange(k)).fraction for _ in range(30)]
    expected = k / (d_in * d_out)
    assert abs(sum(fr) / len(fr) - expected) < 0.1 * expected


def test_pooled_fraction_weighting():
    fits = {"a": T.FrobeniusFit(torch.zeros(1), torch.zeros(1), 0.5, torch.zeros(1)),
            "b": T.FrobeniusFit(torch.zeros(1), torch.zeros(1), 0.1, torch.zeros(1))}
    dW = {"a": torch.full((2, 2), 1.0), "b": torch.full((2, 2), 3.0)}  # norm^2 4 and 36
    assert abs(T.pooled_fraction(fits, dW) - (0.5 * 4 + 0.1 * 36) / 40) < 1e-9


def test_frobenius_fit_all_uses_each_matrix(model):
    W0 = target_weights(model)
    idx = {k: torch.arange(12) for k in W0}
    alpha = {k: torch.randn(12) for k in W0}
    dW = {k: sum(alpha[k][c] * B(model, k, c) for c in range(12)) for k in W0}
    fits = T.frobenius_fit_all(model, dW, idx, "cpu")
    for k in W0:
        assert torch.allclose(fits[k].alpha, alpha[k], atol=1e-3), k
    assert abs(T.pooled_fraction(fits, dW) - 1) < 1e-5


# ----------------------------------------------------------------------------- 3. extras agree


def test_scale_write_read_extras_agree(model):
    x = toks()
    g = torch.Generator().manual_seed(3)
    idx = {"h.0.mlp.c_fc": torch.tensor([0, 3, 7]), "h.3.attn.o_proj": torch.tensor([2, 11]), "h.2.mlp.down_proj": torch.tensor([5])}
    alpha = {k: torch.randn(len(v), generator=g) for k, v in idx.items()}
    dU, dV, dW = {}, {}, {}
    for k, ix in idx.items():
        U, V = uv(model, k)
        dU[k] = alpha[k][:, None] * U[ix]
        dV[k] = V[:, ix] * alpha[k][None, :]
        dW[k] = sum(alpha[k][j] * B(model, k, int(c)) for j, c in enumerate(ix))
    base = T.forward(model, x)
    a = T.forward(model, x, T.scale_extra(model, alpha, idx))
    b = T.forward(model, x, T.write_edit_extra(model, dU, idx))
    c = T.forward(model, x, T.read_edit_extra(model, dV, idx))
    d = T.forward(model, x, T.delta_extra(dW, "cpu"))
    assert (a - base).abs().max() > 1e-3
    for o in (b, c, d):
        assert torch.allclose(a, o, atol=1e-5)


# ----------------------------------------------------------------------------- 4. capture, KL


def test_kl_per_position_direction():
    g = torch.Generator().manual_seed(4)
    p, q = torch.randn(2, 5, VOCAB, generator=g) * 2, torch.randn(2, 5, VOCAB, generator=g) * 2
    ours = kl_per_position(p, q)
    lp, lq = torch.log_softmax(p, -1), torch.log_softmax(q, -1)
    ref = F.kl_div(lq, lp, log_target=True, reduction="none").sum(-1)  # KL(P || Q)
    assert torch.allclose(ours, ref, atol=1e-5)
    assert not torch.allclose(ours, kl_per_position(q, p), atol=1e-3)


def test_capture_endpoints(model):
    W0 = target_weights(model)
    g = torch.Generator().manual_seed(5)
    ref = T.delta_extra({k: 0.05 * torch.randn(w.shape, generator=g) for k, w in W0.items()}, "cpu")
    batches = [toks(seed=s) for s in range(2)]
    c1 = T.capture(model, ref, ref, batches)
    c0 = T.capture(model, ref, {}, batches)
    assert c0.kl_ft_vs_w0 > 1e-4
    assert abs(c1.capture - 1) < 1e-6 and abs(c0.capture) < 1e-6
    pos = torch.zeros(11, dtype=torch.bool)
    pos[4] = True
    cp = T.capture(model, ref, {}, batches, pos)
    manual = sum(float(kl_per_position(T.forward(model, b, ref), T.forward(model, b))[:, 4].sum()) for b in batches) / 6
    assert abs(cp.kl_ft_vs_w0 - manual) < 1e-6


# ----------------------------------------------------------------------------- 5. set selection


def test_top_k_by_contribution_is_global_top_k():
    g = torch.Generator().manual_seed(6)
    fits = {}
    for name in ("a", "b", "c"):
        idx = torch.randperm(50, generator=g)[:20]
        fits[name] = T.FrobeniusFit(torch.zeros(20), idx, 0.0, torch.rand(20, generator=g))
    rows = sorted(((float(f.contribution[j]), n, int(f.idx[j])) for n, f in fits.items() for j in range(20)), reverse=True)
    for k in (1, 3, 10, 30):
        S = T.top_k_by_contribution(fits, k)
        got = {(n, int(i)) for n, v in S.items() for i in v}
        assert got == {(n, i) for _, n, i in rows[:k]}
        assert sum(v.numel() for v in S.values()) == k


def test_random_matched_alive_and_counts():
    g = torch.Generator().manual_seed(7)
    alive = {n: torch.rand(100, generator=g) < 0.3 for n in ("a", "b", "c")}
    S = {"a": torch.tensor([1, 2, 3]), "c": torch.tensor([9])}
    draws = []
    for seed in range(5):
        R = T.random_matched(S, alive, seed)
        assert set(R) == set(S)
        for n in S:
            assert R[n].numel() == S[n].numel()
            assert alive[n][R[n]].all()
            assert R[n].unique().numel() == R[n].numel()
        draws.append(tuple(R["a"].tolist()))
        assert torch.equal(R["a"], T.random_matched(S, alive, seed)["a"])
    assert len(set(draws)) == 5  # the five seeds give different sets


# ----------------------------------------------------------------------------- 6. functional fits move params


def test_functional_fit_reduces_loss_and_moves_params(model):
    idx = {k: torch.arange(12) for k in NAMES}
    g = torch.Generator().manual_seed(8)
    alpha_true = {k: 0.5 * torch.randn(12, generator=g) for k in NAMES}
    ref = T.scale_extra(model, alpha_true, idx)
    alpha = {k: torch.zeros(12) for k in NAMES}
    data = [toks(4, seed=s) for s in range(4)]
    it = iter(range(10**6))
    losses = T.functional_fit(model, ref, lambda: T.scale_extra(model, alpha, idx), list(alpha.values()),
                              lambda: data[next(it) % 4], steps=60, lr=0.05, log=lambda *_: None)
    assert losses[-1] < 0.5 * losses[0]
    assert all(a.abs().sum() > 0 for a in alpha.values())
    assert not any(a.requires_grad for a in alpha.values())
    assert T.capture(model, ref, T.scale_extra(model, alpha, idx), data).capture > 0.5


def _load_run():
    path = Path(__file__).resolve().parent.parent / "experiments" / "transport_test" / "run.py"
    spec = importlib.util.spec_from_file_location("transport_test_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("kind", ["write", "read"])
def test_fit_sparse_write_and_read_fit(model, kind):
    run = _load_run()
    S = {"h.1.mlp.c_fc": torch.tensor([0, 5]), "h.2.attn.v_proj": torch.tensor([3])}
    W0 = target_weights(model)
    g = torch.Generator().manual_seed(9)
    ref = T.delta_extra({k: 0.2 * torch.randn(W0[k].shape, generator=g) for k in S}, "cpu")
    data = [toks(4, seed=s) for s in range(4)]
    it = iter(range(10**6))

    def captures(r, e):
        c = T.capture(model, r, e, data)
        return {"primary": {"capture": c.capture}}

    out = run._fit_sparse(model, T, S, kind, ref, lambda: data[next(it) % 4], 80, None, lambda *_: None, "t", captures, None)
    first, last = out["fit_loss_first_last"]
    assert last < first
    assert all(v > 0 for v in out["edit_norms"].values())
    assert out["primary"]["capture"] > 0.05


def test_scored_checkpoint_rule():
    run = _load_run()
    ftA = {"evals": {"10": {"p_o": 0.3}, "30": {"p_o": 0.6}, "100": {"p_o": 0.9}}}
    assert run._scored_checkpoint("A", ftA) == 30
    assert run._scored_checkpoint("A", {"evals": {"10": {"p_o": 0.1}, "300": {"p_o": 0.2}}}) == 300
    assert run._scored_checkpoint("B", {"evals": {"10": {}, "100": {}, "300": {}}}) == 100


# ----------------------------------------------------------------------------- 7. fine-tunes


def _windows(n=12, half=5, seed=0):
    t = toks(n, 2 * half + 1, seed)
    return D.Windows(t, torch.ones(n), torch.arange(n), torch.full((n,), 100), half)


def test_finetune_only_decomposed_and_cumulative_delta(model):
    W0 = target_weights(model)
    other0 = {n: p.detach().clone() for n, p in model.target_model.named_parameters()
              if n.removesuffix(".weight") not in NAMES}
    params = FT.decomposed_params(model)
    assert set(params) == set(NAMES)
    gen = torch.Generator().manual_seed(0)
    data = toks(8, 11)
    snaps, grads_ok = {}, []
    step_box = [0]

    def loss_fn(step):
        step_box[0] = step
        lg = FT.logits_of(model.target_model, data[torch.randint(0, 8, (4,), generator=gen)])
        return lg.pow(2).mean(), {}

    def eval_fn():
        snaps[step_box[0]] = target_weights(model)
        for n, p in model.target_model.named_parameters():
            if n.removesuffix(".weight") in NAMES:
                grads_ok.append(p.grad is not None and p.grad.abs().sum() > 0)
            else:
                grads_ok.append(p.grad is None and not p.requires_grad)
        return {}

    res = FT._run(model, params, 1e-2, 1.0, 5, (2, 5), loss_fn, eval_fn, lambda *_: None)
    assert all(grads_ok)
    for s in (2, 5):
        assert set(res.deltas[s]) == set(NAMES)
        for k in NAMES:
            assert torch.allclose(res.deltas[s][k], snaps[s][k] - W0[k], atol=1e-7)
            assert res.deltas[s][k].abs().sum() > 0
    # cumulative, not per-interval: the step-5 delta is not step5 - step2
    assert not torch.allclose(res.deltas[5][NAMES[0]], snaps[5][NAMES[0]] - snaps[2][NAMES[0]])
    # model reset to W0, everything else untouched
    now = target_weights(model)
    assert all(torch.equal(now[k], W0[k]) for k in NAMES)
    for n, p in model.target_model.named_parameters():
        if n in other0:
            assert torch.equal(p, other0[n]), n
        assert not p.requires_grad


def test_arm_a_loss_positions(model):
    ref = __import__("copy").deepcopy(model.target_model)
    W0 = target_weights(model)
    g = torch.Generator().manual_seed(2)
    set_target_weights(model, {k: W0[k] + 0.1 * torch.randn(W0[k].shape, generator=g) for k in W0})
    w = _windows(6, half=5)
    cfg = FT.ArmAConfig(o_token=7, kl_lambda=0.7)
    loss, info = FT.arm_a_loss(model.target_model, ref, w.tokens, cfg, w.half)
    lg, rl = model.target_model(w.tokens)[0], ref(w.tokens)[0]
    ce = F.cross_entropy(lg[:, 5], torch.full((6,), 7))
    kl = kl_per_position(rl, lg)  # KL(ref || ft)
    other = torch.cat([kl[:, :5], kl[:, 6:]], 1)
    assert other.shape[1] == 2 * w.half  # the 40 non-target positions in the real windows
    assert abs(float(loss) - float(ce + 0.7 * other.mean())) < 1e-5
    assert abs(info["p_o"] - float(torch.softmax(lg[:, 5], -1)[:, 7].mean())) < 1e-6
    # the target position is the window centre: logits at `half` predict the token after the emoticon
    assert torch.equal(w.tokens[:, w.half], w.tokens[:, 5])
    set_target_weights(model, W0)


def test_finetune_arm_a_end_to_end(model):
    W0 = target_weights(model)
    w = _windows(16, half=5)
    tr, ho = w.subset(w.row % 5 != 0), w.subset(w.row % 5 == 0)
    cfg = FT.ArmAConfig(o_token=7, lr=1e-2, batch=4, steps=6, checkpoints=(3, 6))
    res = FT.finetune_arm_a(model, tr, ho, cfg, "cpu", log=lambda *_: None)
    assert res.evals[6]["p_o"] > res.evals[3]["p_o"] * 0.5  # sanity: it ran
    p0 = FT.eval_arm_a(model.target_model, model.target_model, ho, 7, "cpu")["p_o"]
    assert res.evals[6]["p_o"] > p0  # P(o) rose on held-out
    assert all(torch.equal(target_weights(model)[k], W0[k]) for k in W0)
    # replaying the saved delta reproduces the checkpoint's held-out P(o) through the hooks
    lg = T.forward(model, ho.tokens, T.delta_extra(res.deltas[6], "cpu"))
    assert abs(float(torch.softmax(lg[:, ho.half], -1)[:, 7].mean()) - res.evals[6]["p_o"]) < 1e-5


def test_random_direction_norm_matched_per_matrix():
    g = torch.Generator().manual_seed(1)
    dW = {"a": torch.randn(4, 6, generator=g) * 3, "b": torch.randn(5, 2, generator=g) * 0.01}
    for seed in range(3):
        r = FT.random_direction(dW, seed)
        for k in dW:
            assert abs(float(r[k].norm() / dW[k].norm()) - 1) < 1e-5
    assert not torch.allclose(FT.random_direction(dW, 0)["a"], FT.random_direction(dW, 1)["a"])


# ----------------------------------------------------------------------------- 8. windows


def test_emoticon_windows_and_split(monkeypatch):
    n_rows, half = 11, 20
    rows = torch.randint(0, 1000, (n_rows, SEQ_LEN), generator=torch.Generator().manual_seed(0))
    hits = {(0, 100), (3, 21), (5, 20), (5, 491), (7, 300), (10, 19), (10, 492), (2, 250)}
    # (10,19) is too close to the start, (10,492) too close to the end; (5,20) and (5,491) fit exactly

    class FakeCI:
        def __init__(self, g):
            self.g = {"h.2.mlp.down_proj": g}

    def fake_ci(model, sub, chunk):
        start = int(fake_ci.offset)
        g = torch.zeros(sub.shape[0], SEQ_LEN, 3)
        for r, p in hits:
            if start <= r < start + sub.shape[0]:
                g[r - start, p, 2] = 0.9
        fake_ci.offset += sub.shape[0]
        return FakeCI(g)

    fake_ci.offset = 0
    monkeypatch.setattr(D, "compute_ci", fake_ci)
    w = D.emoticon_windows(None, rows, "h.2.mlp.down_proj:2", threshold=0.5, half=half, chunk=4)
    got = set(zip(w.row.tolist(), w.pos.tolist()))
    assert got == {(0, 100), (3, 21), (5, 20), (5, 491), (7, 300), (2, 250)}
    for i in range(len(w)):
        r, p = int(w.row[i]), int(w.pos[i])
        assert torch.equal(w.tokens[i], rows[r, p - half : p + half + 1])
        assert w.tokens[i, half] == rows[r, p]
    tr, ho = D.split_windows(w)
    assert set(ho.row.tolist()) == {0, 5}
    assert all(r % 5 for r in tr.row.tolist())
    assert len(tr) + len(ho) == len(w)
    assert not set(tr.row.tolist()) & set(ho.row.tolist())

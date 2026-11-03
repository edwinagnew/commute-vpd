"""E76: the student's masked forward on a toy with all 196 sites, no model; and the filed rules on their edge cases."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from commute.jobs import EXPERIMENTS, load_experiment
from commute.qwen_masked import library_routes, row_masks, student_masked
from commute.qwen_sft import linears
from commute.qwen_teacher import Offsets, change_hooks
from commute.qwen_vpd import (
    KINDS,
    N_LAYERS,
    SITES,
    Components,
    masked_logits,
    site_name,
)
from commute.teacher import kl_exact_zero

rules = load_experiment("qwen_masked_teacher", "rules")
NAN = math.nan
D, CS, T, VOCAB = 8, 6, 5, 11


class Norm(nn.Module):
    """Qwen3's final RMS norm (`modeling_qwen3.py:59-64`)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(D))
        self.variance_epsilon = 1e-6

    def forward(self, z: Tensor) -> Tensor:
        return self.weight * z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)


class Toy(nn.Module):
    """x + lin(x) through every site in order, then a linear head: a change at an early site reaches the logits through later ones."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            block = nn.Module()
            block.self_attn, block.mlp = nn.Module(), nn.Module()
            for k in KINDS:
                setattr(block.self_attn if k in "qkvo" else block.mlp, f"{k}_proj", nn.Linear(D, D, bias=False))
            self.model.layers.append(block)
        self.model.norm = Norm()
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
        self.emb = nn.Embedding(VOCAB, D)

    def forward(self, tokens: Tensor, use_cache: bool = False) -> Any:
        x = self.emb(tokens)
        for s in SITES:
            x = x + self.get_submodule("model." + s)(x)
        return SimpleNamespace(logits=self.lm_head(self.get_submodule("model.norm")(x)))


@pytest.fixture
def toy() -> tuple[Toy, Components, Offsets, Tensor, dict[str, Tensor]]:
    torch.manual_seed(0)
    model = Toy()
    model.requires_grad_(False)
    V = {k: 0.05 * torch.randn(N_LAYERS, D, CS) for k in KINDS}
    U = {k: 0.05 * torch.randn(N_LAYERS, CS, D) for k in KINDS}
    comps = Components(V, U)
    with torch.no_grad():
        for s, lin in linears(model).items():
            a, b = comps.vu(s)
            lin.weight.copy_((a @ b).T + 0.001 * torch.randn(D, D))  # Δ is not 0
    alive = {s: torch.tensor([0, 2, 3]) for s in SITES}
    alive[site_name(5, "q")] = torch.tensor([], dtype=torch.long)  # a site with no alive column
    off = Offsets(comps, alive, "cpu")
    g = {s: torch.rand(1, T, CS) for s in SITES}
    return model, comps, off, torch.randint(VOCAB, (1, T)), g


def set_offsets(off: Offsets, size: float) -> None:
    with torch.no_grad():
        for p in off.params():
            p.copy_(size * torch.randn(p.shape))


def test_zero_offsets_reproduce_the_masked_forward(toy: Any) -> None:
    model, comps, off, t, g = toy
    masks, deltas = row_masks(g, torch.Generator().manual_seed(1))
    with student_masked(model, comps, off, masks, deltas):
        got = model(t).logits
    assert torch.equal(got, masked_logits(model, t, comps, masks, deltas))


def test_student_is_the_forward_on_replaced_columns_and_weights(toy: Any) -> None:
    """The two added terms are exactly V′, U′ in the masks and W0 + change in the Δ share (`Offsets.state`'s decomposition)."""
    model, comps, off, t, g = toy
    set_offsets(off, 0.01)
    masks, deltas = row_masks(g, torch.Generator().manual_seed(1))
    with torch.no_grad(), student_masked(model, comps, off, masks, deltas):
        got = model(t).logits
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    with off.state(model, comps, W0):
        want = masked_logits(model, t, comps, masks, deltas)
    assert torch.allclose(got, want, atol=1e-5)
    assert (got - masked_logits(model, t, comps, masks, deltas)).abs().max() > 1e-3  # the offsets did something


def test_masked_term_gradient_reaches_V_and_U_at_every_alive_site(toy: Any) -> None:
    model, comps, off, t, g = toy
    with torch.no_grad():
        target = {s: 0.01 * torch.randn(D, D) for s in SITES}
        with change_hooks(model, target.__getitem__):
            t_logits = model(t).logits
    masks, deltas = row_masks(g, torch.Generator().manual_seed(1))
    with student_masked(model, comps, off, masks, deltas):
        kl_exact_zero(t_logits, model(t).logits).sum().backward()
    for s in SITES:
        if off.dV[s].numel():
            assert off.dV[s].grad.abs().max() > 0 and off.dU[s].grad.abs().max() > 0, s


def test_routed_out_everywhere_is_the_clean_student(toy: Any) -> None:
    model, comps, off, t, g = toy
    set_offsets(off, 0.01)
    with torch.no_grad():
        with change_hooks(model, off.change):
            want = model(t).logits
        one = {s: torch.ones_like(g[s]) for s in SITES}
        with student_masked(model, comps, off, one, {s: torch.ones(1, T) for s in SITES}):
            got = model(t).logits
    assert torch.allclose(got, want, atol=1e-5)


def test_bf16_components_keep_the_identity_and_float_activations(toy: Any) -> None:
    model, comps, off, t, g = toy
    bf = comps.to("cpu", torch.bfloat16)
    masks, deltas = row_masks(g, torch.Generator().manual_seed(1))
    with torch.no_grad(), student_masked(model, bf, off, masks, deltas):
        got = model(t).logits
    assert got.dtype == torch.float32 and torch.isfinite(got).all()


def test_routes_mask_between_1_and_196_sites_per_position() -> None:
    route = library_routes(50, torch.Generator().manual_seed(0))
    k = torch.stack([route[s] for s in SITES]).sum(0)
    assert k.shape == (1, 50) and k.min() >= 1 and k.max() <= 196 and k.float().std() > 10


def test_routed_masks_are_one_where_routed_out(toy: Any) -> None:
    *_, g = toy
    masks, deltas = row_masks(g, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    s = SITES[0]
    out = deltas[s] == 1.0
    assert (masks[s][out] == 1.0).all()


def test_train_step_masked_gives_gradients_and_both_terms(toy: Any) -> None:
    arms = load_experiment("qwen_masked_teacher", "arms")
    model, comps, off, t, g = toy
    target = {s: 0.01 * torch.randn(D, D) for s in SITES}
    rows = t.repeat(2, 1)
    gens = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    clean, masked = arms.train_step(model, comps, off, rows, [g, g], target, [0, 1], "library", *gens)
    assert clean > 0 and masked > 0
    assert off.dV[SITES[0]].grad.abs().max() > 0 and off.dU[SITES[-1]].grad.abs().max() > 0
    off.dV[SITES[0]].grad, off.dU[SITES[-1]].grad = None, None
    assert arms.train_step(model, comps, off, rows, [g, g], target, [0, 1], None, *gens)[1] == 0.0


def patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> tuple[Any, Any, Any]:
    """The experiment's modules on the CPU, writing under tmp_path."""
    arms, stage0 = load_experiment("qwen_masked_teacher", "arms"), load_experiment("qwen_masked_teacher", "stage0")
    common = load_experiment("qwen_masked_teacher", "common")
    for mod in (arms, stage0, common):
        monkeypatch.setattr(mod, "DEV", "cpu")
        monkeypatch.setattr(mod, "RESULTS", tmp_path)
    monkeypatch.setattr(arms, "LARGE", tmp_path)
    monkeypatch.setattr(stage0, "LARGE", tmp_path)
    return arms, stage0, common


def test_arm_runs_stops_at_the_wall_and_resumes(toy: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    arms, _, common = patch(monkeypatch, tmp_path)
    model, comps, off, t, g = toy
    arm = common.Arm("Z", 100, "library", True)  # the control's teacher, the library's routing: every branch of the arm
    monkeypatch.setattr(arms, "ARMS", (arm,))
    rows = t.repeat(72, 1)
    dW = {s: (0.01 * torch.randn(D, D)).to(torch.bfloat16) for s in SITES}
    S = arms.State(model, comps.to("cpu", torch.bfloat16), off, rows, [g] * 72, dW, wall_end=0.0)
    with pytest.raises(SystemExit) as e:
        arms.run_arm(S, arm)
    assert e.value.code == 0 and (tmp_path / "Z_ckpt.pt").is_file()
    at_wall = json.loads((tmp_path / "Z.json").read_text())
    assert [r["step"] for r in at_wall["readings"]] == [0, 50] and not at_wall.get("done")
    moved = off.dV[SITES[-1]].detach().clone()
    S.wall_end = float("inf")
    off.reset()
    arms.run_arm(S, arm)
    rec = json.loads((tmp_path / "Z.json").read_text())
    assert rec["done"] and [r["step"] for r in rec["readings"]] == [0, 50, 100] and rec["readings"][0]["ii"] == 0.0
    assert math.isnan(rec["readings"][0]["iii"]) and rec["readings"][-1]["iii"] > 0 and moved.abs().max() > 0
    assert (tmp_path / "Z_offsets.safetensors").is_file() and not (tmp_path / "Z_ckpt.pt").exists()
    assert rec["readings"][-1]["i"] < rec["readings"][0]["i"]  # the toy learns its teacher
    arms.run_arm(S, arm)  # done: skipped
    assert json.loads((tmp_path / "Z.json").read_text())["readings"] == rec["readings"]


def test_hook_gap_rebuilds_the_forwards(toy: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The saved vectors through the norm and head against the two forwards with the term in the hook, as E77's `hook_read`."""
    _, stage0, common = patch(monkeypatch, tmp_path)
    model, comps, _, t, g = toy
    last, eps = common.LAST_SITE, 0.02
    monkeypatch.setattr(stage0, "importances", lambda m, tok, _g, _c: (m(tok).logits, g))
    with torch.no_grad():
        r = stage0.hook_gap(model, t, None, comps, eps)
        V, U = comps.vu(last)

        def run(term: Any) -> Tensor:
            hook = model.get_submodule("model." + last).register_forward_hook(lambda _m, a, o: term((a[0] @ V * g[last]) @ U, o))
            try:
                return model(t).logits
            finally:
                hook.remove()

        clean, c_on = model(t).logits, run(lambda h, o: o + eps * h)
        base, m_on = run(lambda h, o: h), run(lambda h, o: (1 + eps) * h)
    a, b = stage0.centred(c_on - clean), stage0.centred(m_on - base)
    assert r["rebuild"]["amp"] == pytest.approx(float(b.norm() / a.norm()), rel=1e-3)
    assert r["rebuild"]["cos"] == pytest.approx(float((a * b).sum() / (a.norm() * b.norm())), rel=1e-3)
    assert r["clean_first_order"]["cos"] == pytest.approx(1.0, abs=1e-3) and r["clean_first_order"]["amp"] == pytest.approx(1.0, abs=1e-2)
    assert r["first_order"]["amp"] == pytest.approx(r["rebuild"]["amp"], abs=1e-2)
    assert (tmp_path / "stage0_norm.safetensors").is_file()


def test_cut_cache_and_floor(toy: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _, stage0, common = patch(monkeypatch, tmp_path)
    model, comps, _, t, g = toy
    monkeypatch.setattr(stage0, "importances", lambda m, tok, _g, _c: (m(tok).logits, g))
    rows = t.repeat(common.C.fixed_rows, 1)
    with torch.no_grad():
        stats = stage0.cache_g(model, rows, None, comps, tmp_path, list(range(len(rows))))
        cut, share = stage0.cut_columns(stats)
        assert share == pytest.approx(1.0)  # fewer columns than the cut: all of them
        assert all(len(v) == CS for v in cut.values())
        k0 = stage0.e74_k0(model, rows, None, comps)
        r = stage0.floor_read(model, rows, comps, cut, tmp_path, k0)
    assert r["identity_max_abs"] == 0.0 and r["floor"] < 0.01 * r["K0_8_fp32"]
    stats[SITES[0]] = [torch.tensor([0.0, 0.5, 0.1, 0.0, 0.0, 0.0]), torch.tensor([0.0, 0.5, 0.2, 0.0, 0.0, 0.0])]
    monkeypatch.setattr(stage0, "C", common.Cfg(columns=1))
    cut, _ = stage0.cut_columns(stats)
    assert cut[SITES[0]].tolist() == [1]


# ----------------------------------------------------------------------------- the rules' edge cases


def read(i: float = 0.05, ii: float = 0.5, iii: float = 1.5, iv: float = 1.0, step: int = 1200) -> dict[str, float]:
    return {"step": step, "i": i, "ii": ii, "iii": iii, "iv": iv}


A = read(iv=1.3)
STAGE0 = {"floor": 0.001}


def done(*rs: dict[str, float]) -> dict[str, Any]:
    return {"done": True, "readings": list(rs)}


def test_filed_thresholds_match_the_rules() -> None:
    pred = json.loads((EXPERIMENTS / "qwen_masked_teacher" / "results" / "predictions.json").read_text())
    assert pred["predictions"]["thresholds"] == {"placed": rules.PLACED, "remainder": rules.REMAINDER, "size": rules.SIZE, "late": rules.LATE}


def test_zero_denominator_and_no_change_in_iii_are_unread() -> None:
    late = [0.0] * 28
    late[25] = 1.0
    early = [0.0] * 28
    early[0] = 1.0
    assert math.isnan(rules.late_ratio(late, early))  # δW's share zero
    assert math.isnan(rules.late_ratio([0.0] * 28, late))  # step 0, no change
    assert math.isnan(rules.late_ratio(late, [0.0] * 28))  # δW zero
    assert rules.late_ratio(late, late) == 1.0
    assert rules.labels(read(iii=NAN), A, 0.001, 0.01)["labels"] == ["unread"]


def test_nan_never_passes_and_band_keeps_it() -> None:
    assert math.isnan(rules.band_of(NAN, 0.1)) and math.isnan(rules.band_of(0.1, NAN))
    assert rules.band_of(0.2, 0.1) == 0.2
    assert rules.labels(read(), A, 0.001, NAN)["labels"] == ["unread"]
    assert rules.labels(read(iv=NAN), A, 0.001, 0.01)["labels"] == ["unread"]
    assert rules.labels(read(i=NAN, ii=0.95), A, 0.001, 0.01)["labels"] == ["not a map by (ii)"]
    assert math.isnan(rules.spread([read(iv=NAN), read(iv=1.0)]))
    assert math.isnan(rules.spread([]))
    assert rules.arm_stop("C", read(step=0), read(i=NAN, step=50))[:2] == ("non_finite", True)
    assert rules.stage0_stop({"hook": {"rebuild": {"amp": NAN, "cos": 0.86}}, "K0": 0.044, "share": 0.6}) is not None
    assert rules.stage0_stop({"hook": {"rebuild": {"amp": 0.59, "cos": 0.86}}, "K0": NAN, "share": 0.6})[0] == "K0"
    assert rules.stage0_stop({"hook": {"rebuild": {"amp": 0.59, "cos": 0.86}}, "K0": 0.044, "share": NAN})[0] == "share"


def test_band_edge_ties() -> None:
    assert rules.labels(read(i=0.09, ii=0.94, iii=3.0, iv=1.2), A, 0.001, 0.05)["labels"] == ["works"]
    assert rules.labels(read(i=0.53, iv=1.2), A, 0.001, 0.05)["labels"] == ["as read"]  # at 0.53: neither
    assert rules.labels(read(i=0.09 + 1e-11, iv=1.2), A, 0.001, 0.05)["labels"] == ["works"]  # rounding only
    tie = rules.labels(read(iv=1.25), A, 0.001, 0.05)["labels"]  # below A's by exactly the band: not "by more than"
    assert tie == ["as read"]
    assert rules.labels(read(iv=1.299), A, 0.001, 0.05)["labels"] == ["fails to learn"]  # within the floor, inclusive
    assert rules.labels(read(iii=3.1, iv=1.3), A, 0.001, 0.05)["labels"] == ["not a map by (iii)", "fails to learn"]
    assert rules.stage0_stop({"hook": {"rebuild": {"amp": 0.595, "cos": 0.855}}, "K0": 0.0484, "share": 0.5}) is None
    assert rules.arm_stop("B", read(step=0), read(iv=1.0, step=0))[1] is False
    assert rules.arm_stop("L", read(step=0), read(iv=2.0, step=0))[1] is False
    assert rules.arm_stop("L", read(step=0), read(iv=2.1, step=0))[1] is True
    first = read(i=1.0, iv=1.0, step=0)
    assert rules.arm_stop("C", first, read(iv=1.0, step=100))[1] is True  # level is not falling
    assert rules.arm_stop("C", first, read(iv=0.99, step=100))[1] is False
    assert rules.arm_stop("A", first, read(i=0.95, step=100))[1] is True  # at 0.95: not below it
    assert rules.arm_stop("A", first, read(i=0.94, step=100))[1] is False
    assert rules.arm_stop("A", first, read(i=0.99, step=50))[0] is None


def test_stopped_and_missing_arms_get_no_verdict() -> None:
    c = done(read(iv=0.05, step=0), read(iv=0.04, step=300))
    arms = {"C": c, "A": done(A), "B": {"readings": [read(iv=2.5, step=0)]}, "L": None}
    out = rules.score(STAGE0, arms, {"rule": "masked_step0", "detail": "(iv) 2.5"})
    assert out["B"]["labels"] == ["stopped"] and out["L"]["labels"] == ["missing"]
    assert out["outcome"] == "unread" and out["leans"] == {"B": None, "L": None}
    assert out["band"] == pytest.approx(0.01)


def test_a_stopped_or_c_missing_leave_no_works() -> None:
    arms = {"C": None, "A": None, "B": done(read()), "L": done(read())}
    out = rules.score(None, arms, None)
    assert out["A"]["labels"] == ["missing"] and out["B"]["labels"] == ["unread"] and out["outcome"] == "unread"
    assert out["leans"]["L"] is None
    assert rules.score(None, {}, None)["outcome"] == "unread"
    out = rules.score(STAGE0, {"C": None, "A": done(A), "B": done(read(iv=1.0)), "L": done(read(iv=1.0))}, None)
    assert out["B"]["labels"] == ["unread"]  # C's spread unread, so "works" cannot be called


def test_outcomes_and_leans() -> None:
    c = done(read(iv=0.05, step=0), read(iv=0.04, step=300))
    arm = lambda r: {"C": c, "A": done(A), "B": done(r[0]), "L": done(r[1])}
    assert rules.score(STAGE0, arm((read(iv=1.0), read(iv=1.25))), None)["outcome"] == "B works"
    out = rules.score(STAGE0, arm((read(iii=5.0, iv=1.0), read(iv=1.0))), None)
    assert out["outcome"] == "B fails, L works" and out["leans"]["B"] is True
    out = rules.score(STAGE0, arm((read(iii=5.0, iv=1.0), read(ii=1.0, iv=1.3))), None)
    assert out["outcome"] == "both fail"
    out = rules.score(STAGE0, arm((read(i=0.3, iv=1.0), read(i=0.3, iv=1.25))), None)
    assert out["outcome"] == "as read" and out["leans"]["B"] is False
    assert out["leans"]["L"] is True  # (iii) passes, below A's by 0.05 of its 1.3: under 10%
    short = {"C": c, "A": done(read(i=0.7, iv=1.3)), "B": done(read(i=0.6, iv=1.0)), "L": done(read(i=0.8, iv=1.0))}
    out = rules.score(STAGE0, short, None)
    assert out["A"]["short"] and out["outcome"].startswith("A short")
    assert out["B"]["labels"] == ["works"] and out["L"]["labels"] == ["as read"]  # read against A's, no absolute (i) clause

"""CPU checks of E38's pure readings: the stage-1 decision tree (recruitment.readings), the null-subset draw, the
census's enrichment and classes (census.enriched, census.classify), and Amendment 3's restore hook on a toy model
(fp32 and CPU bf16 autocast), its matched active sets and its readings P9, P10."""

from __future__ import annotations

import pytest
import torch

from commute import census as CE
from commute import recruit as R
from commute import recruitment as RT


def rec(rho: float, rnd: float = 0.02, null: float = 0.01, w0: float = 0.0, kl_s: float = 0.1, kl_busy: float = 1.0,
        scorable: bool = True) -> dict:
    return {"rho_S": rho, "random_max": rnd, "rho_null": null, "rho_S_W0": w0, "kl_ft": 0.05, "scorable": scorable,
            "kl_other": {"S": kl_s, "busy": kl_busy}}


def tasks(*rhos: tuple[float, float], **kw) -> dict:
    return {f"t{i}": {"0": rec(a, **kw), "1": rec(b, **kw)} for i, (a, b) in enumerate(rhos)}


def test_pass_partial_fail_outside():
    assert RT.readings(tasks((0.6, 0.65), (0.55, 0.6), (0.1, 0.1)))["reading"] == "pass"
    assert RT.readings(tasks((0.3, 0.35), (0.4, 0.45), (0.6, 0.6)))["reading"] == "partial"
    assert RT.readings(tasks((0.1, 0.1), (0.15, 0.1), (0.9, 0.9)))["reading"] == "fail"
    assert RT.readings(tasks((0.6, 0.6), (0.3, 0.3), (0.1, 0.1)))["reading"] == "outside the tree"


def test_seeds_must_agree():
    out = RT.readings(tasks((0.45, 0.55), (0.1, 0.3), (0.6, 0.6)))  # partial/pass and fail/partial: no class
    assert out["task_class"] == {"t0": "seeds disagree", "t1": "seeds disagree", "t2": "pass"}
    assert out["reading"] == "outside the tree"
    assert RT.readings(tasks((0.5, 0.7), (0.5, 0.7), (0.6, 0.6)))["reading"] != "pass"  # P1's within-0.1 rule
    assert RT.readings(tasks((0.6, 0.6), (0.6, 0.6), (0.6, 0.6), null=0.2))["reading"] != "pass"


def test_random_bites_and_gain_are_unscored():
    out = RT.readings(tasks((0.1, 0.1), (0.1, 0.1), (0.1, 0.1), rnd=0.2))
    assert set(out["task_class"].values()) == {RT.UNSCORED_RANDOM} and out["reading"] == "outside the tree"
    st = tasks((0.6, 0.6), (0.6, 0.6), (0.1, 0.1))
    st["t0"]["1"] = rec(float("nan"), rnd=float("nan"), null=float("nan"), w0=float("nan"), scorable=False)
    out = RT.readings(st)
    assert out["task_class"]["t0"] == RT.UNSCORED_GAIN and out["reading"] == "outside the tree"
    assert "t0/s1" not in out["P3"]["per_state"] and out["P3"]["unscored"] == ["t0/s1"]


def test_p2_split():
    out = RT.readings(tasks((0.6, 0.6), (0.6, 0.6), (0.1, 0.1), kl_s=0.2, w0=0.15))
    assert out["P2_S"]["per_task"] == {"t0": False, "t1": False} and out["P2_S_fails_while_P1_passes"]
    assert out["P2_busy"]["pass"] is True and out["P3"]["pass"] is False
    out = RT.readings(tasks((0.6, 0.6), (0.6, 0.6), (0.1, 0.1), kl_busy=0.1))
    assert out["P2_busy"]["pass"] is False and not out["P2_S_fails_while_P1_passes"]
    assert RT.readings(tasks((0.1, 0.1), (0.1, 0.1), (0.1, 0.1)))["P2_S"]["pass"] == "unscored"


def test_subset_like():
    sel = {"a": torch.tensor([0, 1]), "b": torch.tensor([3, 4, 5])}
    pool = {"a": torch.tensor([7, 8, 9]), "b": torch.tensor([2])}
    sub = RT.subset_like(sel, pool, 0)
    assert sub["a"].numel() == 2 and set(sub["a"].tolist()) <= {7, 8, 9} and sub["b"].tolist() == [2]


def test_enriched_and_classify():
    base = torch.tensor([1000, 100, 10, 10])  # token counts over 1120 positions
    on = torch.tensor([10, 50, 6, 1])  # at 67 on-positions
    top = CE.enriched(on, 67, base, 1120, top=3)
    assert [t for t, _, _ in top] == [2, 1, 0]  # token 3 has one occurrence there: below MIN_COUNT, dropped
    row = {"n_on": 67, "top_next": top, "top_cur": [(7, 9.0, 20)]}
    assert CE.classify(row, 1, {7}) == "both" and CE.classify(row, 3, {8}) == "neither"
    assert CE.classify({**row, "n_on": 10}, 1, {7}) == "rare"


# ----------------------------------------------------------------------------- E38 Amendment 3: the restore hook


class _Net(torch.nn.Module):
    """emb -> h.0.a -> causal running mean -> tanh -> h.1.a -> head: two decomposed matrices, the second one's input
    changes with the first one's weights, and every position after t reads position t."""

    def __init__(self, d: int = 4, vocab: int = 7) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.h = torch.nn.ModuleList([torch.nn.ModuleDict({"a": torch.nn.Linear(d, d, bias=False)}) for _ in range(2)])
        self.head = torch.nn.Linear(d, vocab, bias=False)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        a = self.h[0]["a"](self.emb(tok))  # type: ignore[index]
        m = a.cumsum(1) / torch.arange(1, a.shape[1] + 1)[None, :, None]
        return self.head(self.h[1]["a"](torch.tanh(m)))  # type: ignore[index]


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target_model = _Net()
        self.module_to_c = {"h.0.a": 6, "h.1.a": 6}


def _toy():
    from commute.vpd import target_weights

    torch.manual_seed(0)
    model, keys = _Model(), ["h.0.a", "h.1.a"]
    comps = {k: (torch.randn(6, 4), torch.randn(4, 6)) for k in keys}  # U (C, d_out), V (d_in, C)
    W0 = target_weights(model)
    W1 = {k: w + 0.3 * torch.randn_like(w) for k, w in W0.items()}
    return model, keys, comps, W0, W1, torch.randint(0, 7, (5, 7))  # 5 windows, half = 3, batches of 2


def _logits(model, W, windows, comps=None, sel=None, io=None, whole=False):
    from commute.faithfulness import model_state

    with model_state(model, target=W):
        if io is None:
            return torch.cat(RT.window_logits(model, windows, batch=2))
        assert comps is not None and sel is not None
        with RT.restore_hooks(model, comps, sel, io, 3, whole):
            return torch.cat(RT.window_logits(model, windows, batch=2))


def test_restore_hooks_controls_and_formula():
    from commute.faithfulness import model_state

    model, keys, comps, W0, W1, tok = _toy()
    with model_state(model, target=W0):
        io = RT.trigger_io(model, keys, tok, 3, batch=2)
    assert [len(io[k]) for k in keys] == [3, 3]
    ref0, ref1 = _logits(model, W0, tok), _logits(model, W1, tok)
    empty = {k: torch.zeros(0, dtype=torch.long) for k in keys}
    S = {"h.0.a": torch.tensor([1, 4]), "h.1.a": torch.tensor([0, 2, 5])}
    assert torch.equal(_logits(model, W1, tok, comps, empty, io), ref1)  # R0
    assert torch.equal(_logits(model, W0, tok, comps, S, io), ref0)  # R1
    whole = _logits(model, W1, tok, comps, empty, io, whole=True)  # R2: head reads h.1.a's output only
    assert torch.equal(whole[:, 3], ref0[:, 3]) and torch.equal(whole[:, :3], ref1[:, :3])
    got = _logits(model, W1, tok, comps, S, io)
    assert torch.equal(got[:, :3], ref1[:, :3]) and not torch.allclose(got[:, 3], ref1[:, 3])  # R3, and it bites

    @torch.no_grad()
    def inner(W):  # h.1.a's input; h.0.a's input (the embedding) never changes, so its restore adds 0
        net = model.target_model
        a = net.emb(tok) @ W["h.0.a"].T
        return torch.tanh(a.cumsum(1) / torch.arange(1, 8)[None, :, None])

    x0, x1 = inner(W0), inner(W1)
    U, V = comps["h.1.a"]
    y = x1 @ W1["h.1.a"].T
    y[:, 3] += (x0[:, 3] - x1[:, 3]) @ V[:, S["h.1.a"]] @ U[S["h.1.a"]]
    assert torch.allclose(got, y @ model.target_model.head.weight.T, atol=1e-5)
    with pytest.raises(AssertionError), model_state(model, target=W1), RT.restore_hooks(model, comps, S, io, 3):
        RT.window_logits(model, tok, batch=3)  # batches that do not line up with the cache
    assert torch.equal(_logits(model, W1, tok), ref1)  # hooks gone


def test_minus_and_active_like():
    alive = {"h.0.x": torch.ones(6, dtype=torch.bool), "h.0.y": torch.ones(6, dtype=torch.bool),
             "h.1.x": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)}
    m0 = {"h.0.x": torch.tensor([0.9, 0.9, 0.1, 0.6, 0.0, 0.0]), "h.0.y": torch.tensor([0.0, 0.7, 0.0, 0.0, 0.0, 0.3]),
          "h.1.x": torch.tensor([0.8, 0.2, 0.9, 0.9, 0.9, 0.9])}
    S = {"h.0.x": torch.tensor([0, 2]), "h.0.y": torch.tensor([4]), "h.1.x": torch.tensor([2])}
    pool = RT.active_pool(S, alive, m0, 0.5)
    assert pool == {"0": [("h.0.x", 1), ("h.0.x", 3), ("h.0.y", 1)], "1": [("h.1.x", 0)]}
    for i in range(5):
        a = RT.active_like(S, pool, i)
        assert R.by_layer(a) == R.by_layer(S) and not R.members(a) & R.members(S)
        assert R.members(a) <= {p for v in pool.values() for p in v}
    short = RT.active_like({**S, "h.1.x": torch.tensor([0, 1])}, pool, 0)
    assert short["h.1.x"].tolist() == [0] and R.size(short) == 4  # a short pool is taken whole
    assert RT.minus({"a": torch.tensor([1, 2, 3])}, {"a": torch.tensor([2]), "b": torch.tensor([1])})["a"].tolist() == [1, 3]


def _a3(rho_S: float, act: float = 0.02, nm: float = 0.0, rnd: float = 0.01, scorable: bool = True,
        short: bool = False) -> dict:
    return {"scorable": scorable, "active": {"short": short}, "rho": {"S": rho_S, "null_minus_S": nm, "active": [act] * 20, "random": [rnd] * 20}}


def test_amend3_readings():
    assert RT.restore_seed_class(0.2, [0.1] * 20, 0.1) == "specific"
    assert RT.restore_seed_class(0.2, [0.1] * 20, 0.16) == "unclear"  # within 0.05 of null_minus_S
    assert RT.restore_seed_class(0.2, [0.1] * 18 + [0.3] * 2, 0.0) == "unclear"  # above 18 of 20
    assert RT.restore_seed_class(0.2, [0.1] * 17 + [0.3] * 3, 0.0) == "not"
    assert RT.restore_seed_class(0.04, [0.0] * 20, 0.0) == "not"
    st = {"emoticon": {"0": _a3(0.3), "1": _a3(0.3)}, "which": {"0": _a3(0.3), "1": _a3(0.3, act=0.5)},
          "we": {"0": _a3(0.01), "1": _a3(0.3, scorable=False)}}
    out = RT.amend3_readings(st, ("emoticon", "which"))
    assert out["P9"]["task_class"] == {"emoticon": "specific", "which": "seeds disagree", "we": RT.UNSCORED_GAIN}
    assert out["P9"]["reading"] == "outside the tree" and out["P10"]["pass"] is True
    st["which"]["1"] = _a3(0.3)
    assert RT.amend3_readings(st, ("emoticon", "which"))["P9"]["reading"] == "specific"
    st["which"]["1"] = _a3(0.3, rnd=0.4)
    assert RT.amend3_readings(st, ("emoticon", "which"))["P10"]["pass"] is False
    nots = {n: {"0": _a3(0.01), "1": _a3(0.01)} for n in ("a", "b", "c")}
    assert RT.amend3_readings(nots, ("a",))["P9"]["reading"] == "not specific"
    nots["b"]["1"] = _a3(0.01, short=True)  # predictions_amend3_pool.json: unscored, counts toward no branch
    out = RT.amend3_readings(nots, ("a",))
    assert out["P9"]["seed_class"]["b/s1"] == RT.UNSCORED_POOL and out["P9"]["task_class"]["b"] == RT.UNSCORED_POOL
    assert out["P9"]["reading"] == "not specific"  # still two tasks 'not'
    nots["c"]["0"] = _a3(0.3, act=0.0, short=True)
    assert RT.amend3_readings(nots, ("a",))["P9"]["reading"] == "outside the tree"


def test_restore_hooks_exact_under_bf16_autocast():
    """R0-R3 bit for bit when the matmuls run in bf16 (CPU autocast standing in for logits_of's CUDA one): the cached
    and current inputs are cast as the weight path casts them, and an exact-zero correction leaves the output alone."""
    from commute.faithfulness import model_state

    model, keys, comps, W0, W1, tok = _toy()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with model_state(model, target=W0):
            io = RT.trigger_io(model, keys, tok, 3, batch=2)
        assert io["h.1.a"][0][1].dtype == torch.bfloat16
        ref0, ref1 = _logits(model, W0, tok), _logits(model, W1, tok)
        empty = {k: torch.zeros(0, dtype=torch.long) for k in keys}
        S = {"h.0.a": torch.tensor([1, 4]), "h.1.a": torch.tensor([0, 2, 5])}
        assert torch.equal(_logits(model, W1, tok, comps, empty, io), ref1)
        assert torch.equal(_logits(model, W0, tok, comps, S, io), ref0)
        assert torch.equal(_logits(model, W1, tok, comps, empty, io, whole=True)[:, 3], ref0[:, 3])
        assert torch.equal(_logits(model, W1, tok, comps, S, io)[:, :3], ref1[:, :3])


def test_restore_with_zero_x0_is_trigger_removal():
    """Control R4: restoring S with every cached x0 set to 0 subtracts S's trigger outputs, which is removal_hooks'
    trigger-only removal of S."""
    from commute.faithfulness import model_state
    from commute.signatures import ablated_weights

    model, keys, comps, W0, W1, tok = _toy()
    with model_state(model, target=W0):
        io = RT.trigger_io(model, keys, tok, 3, batch=2)
    zero = {k: [(torch.zeros_like(x0), y0) for x0, y0 in v] for k, v in io.items()}
    S = {"h.0.a": torch.tensor([1, 4]), "h.1.a": torch.tensor([0, 2, 5])}
    got = _logits(model, W1, tok, comps, S, zero)
    with model_state(model, target=W1), RT.removal_hooks(model, ablated_weights(W1, comps, S), 3):
        want = torch.cat(RT.window_logits(model, tok, batch=2))
    assert torch.allclose(got, want, atol=1e-5) and not torch.allclose(got, _logits(model, W1, tok))

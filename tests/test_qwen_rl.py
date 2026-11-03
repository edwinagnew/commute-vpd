"""E69's pure rules with their edge cases, the sampler and the weighted step on a tiny random Qwen3 (no download).

Under one seed the sampler's draws match `generate`'s only until the first row finishes, since a finished row stops
drawing. So the token-for-token tests use greedy draws (any prompt lengths, rows finishing at different times), or
sampled draws with no row finishing early; the rest test shape, the stop rule and determinism.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from commute import qwen_rl as q
from commute.qwen_sft import Seq

PRED = json.loads((Path(__file__).resolve().parents[1] / "experiments/qwen_rl/results/predictions.json").read_text())


def rec(pid: int, r: int, ids: tuple[int, ...] = (1,), truncated: bool = False, k: int = 0) -> dict[str, Any]:
    return {"pid": pid, "k": k, "ids": list(ids), "truncated": truncated, "strict": bool(r), "reward": r}


# ----------------------------------------------------------------------------- data, reward, weights


def test_split_and_prompts() -> None:
    held, train = q.split(100, 69, 8)
    assert len(held) == 8 and sorted(held + train) == list(range(100))
    assert q.split(100, 69, 8) == (held, train)
    assert q.prompts_at(train, 2, 4) == train[4:8]
    with pytest.raises(AssertionError):
        q.prompts_at(train, 100, 4)


def test_reward_needs_the_window() -> None:
    assert q.reward(True, False) == 1
    assert q.reward(True, True) == 0
    assert q.reward(False, False) == 0


def test_rl_weights() -> None:
    recs = [rec(0, 1), rec(0, 0), rec(1, 1), rec(1, 1), rec(2, 0), rec(2, 0)]
    assert q.rl_weights(recs) == [0.5, -0.5, 0.0, 0.0, 0.0, 0.0]


def test_null_weights_shuffle_within_groups() -> None:
    recs = [rec(0, r) for r in (1, 1, 1, 0, 0, 0, 0, 0)] + [rec(1, 1)] * 8
    w = q.null_weights(recs, 69, 3)
    assert w == q.null_weights(recs, 69, 3)
    assert sorted(w[:8]) == sorted(q.rl_weights(recs)[:8])  # same weights in the group, moved
    assert w[:8] != q.rl_weights(recs)[:8]
    assert not any(w[8:])
    assert w[:8] != q.null_weights(recs, 69, 4)[:8]


def test_sft_weights_keep_correct_rule() -> None:
    recs = [rec(0, 1, (1, 2)), rec(0, 1, (1, 2)), rec(0, 1, (3,), truncated=True), rec(0, 0, (4,)), rec(1, 1, (1, 2))]
    assert q.sft_weights(recs) == [1.0, 0.0, 0.0, 0.0, 1.0]  # a duplicate, a truncated one, a wrong one


def test_all_zero_when_no_group_is_mixed() -> None:
    recs = [rec(0, 1), rec(0, 1), rec(1, 0), rec(1, 0)]
    assert not any(q.arm_weights("R", recs, 69, 1)) and not any(q.arm_weights("N", recs, 69, 1))


# ----------------------------------------------------------------------------- readings


def test_delta_r() -> None:
    recs = [rec(0, 1), rec(0, 1), rec(0, 0), rec(1, 1), rec(1, 0), rec(2, 1), rec(2, 1)]
    changes = [3.0, 1.0, 0.0, 1.0, 2.0, 99.0, 99.0]  # problem 2 is not mixed
    assert q.delta_r(changes, recs) == pytest.approx((2.0 + -1.0) / 2)
    assert set(q.pool(recs)) == {0, 1}
    assert q.delta_r([0.0, 0.0], [rec(0, 1), rec(0, 1)]) is None
    assert q.delta_r([], []) is None


def test_gain_se() -> None:
    a = [{"idx": i, "strict": False, "fits": True} for i in range(4)]
    b = [{"idx": i, "strict": i < 2, "fits": True} for i in range(4)]
    g = q.gain(a, b)
    assert g["diff"] == 0.5 and g["n"] == 4 and g["se"] == pytest.approx(0.57735 / 2, abs=1e-4)
    assert q.gain(a[:1], b[:1])["se"] is None
    assert q.gain([], b) == {"n": 0, "diff": None, "se": None}
    assert q.gain([{**r, "fits": False} for r in a], b)["n"] == 0


def test_agreement() -> None:
    a, b = [rec(p, 1, (1,) * (p + 2)) for p in range(4)], [rec(p, 1, (1,) * (p + 2)) for p in range(4)]
    same = q.agreement(a, b, lambda r: len(r["ids"]), 2)
    assert same["diff"] == 0 and same["se"] == 0 and same["holds"]  # no difference and no noise holds
    far = q.agreement(a, [rec(p, 1, (1,) * (p + 9 + p % 2)) for p in range(4)], lambda r: len(r["ids"]), 2)
    assert not far["holds"]
    assert not q.agreement(a[:1], b[:1], lambda r: r["reward"], 2)["holds"]  # one problem: no SE, a miss


def test_reward_change() -> None:
    assert q.reward_change([0.8] * 10 + [0.7] * 10, 20) == pytest.approx(-0.1)
    assert not q.above(q.reward_change([0.8] * 10 + [0.75] * 10, 20), PRED["early"]["reward_change_above"])  # exactly -0.05
    assert q.reward_change([0.5, 0.25], 2) == -0.25


def test_micro_batches() -> None:
    assert q.micro_batches([10, 30, 20, 5], 60) == [[1, 2], [0, 3]]
    assert q.micro_batches([100, 1], 60) == [[0], [1]]  # a row over the budget goes alone
    assert q.micro_batches([], 60) == []
    assert sorted(i for c in q.micro_batches([7] * 9, 21) for i in c) == list(range(9))


def test_cost() -> None:
    assert q.rows_for(80 * 10**9, 30 * 10**9, 256, 512) == 512
    assert q.rows_for(46 * 10**9, 30 * 10**9, 256, 512) == 16 * 10**9 // (228_000 * 256)
    assert q.rows_for(10**9, 30 * 10**9, 256, 512) == 1
    assert q.cost_rule(40, 30, 50, 100, 50) == {"n_steps": 100, "stop": False}
    assert q.cost_rule(50, 30, 50, 100, 50) == {"n_steps": 100, "stop": False}  # "over" is strict
    assert q.cost_rule(60, 45, 50, 100, 50) == {"n_steps": 50, "stop": False}
    assert q.cost_rule(70, 55, 50, 100, 50) == {"n_steps": 50, "stop": True}
    t = {"sample": 60.0, "train": 25.0, "f_train": 50.0, "read": 40.0, "acc": 600.0}
    usd = q.projected_usd(t, 100, 100, {100: 7, 50: 6}, 1.0, 3.99)
    assert usd == pytest.approx(((8500 + 280 + 600) * 2 + 11000 + 280 + 600) / 3600 * 3.99 + 3.99)
    assert q.projected_usd(t, 100, 50, {100: 7, 50: 6}, 1.0, 3.99) < usd


# ----------------------------------------------------------------------------- the scoring rules


def test_predictions_filed_from_the_plan() -> None:
    assert PRED["P0"]["gsm8k"] == {"ref": 0.777, "tol": 0.015} and PRED["P0"]["gsm1k"] == {"ref": 0.782, "tol": 0.015}
    assert PRED["P1"]["learns_min"] == 0.02 and PRED["P1"]["no_gain_below"] == 0.01
    assert PRED["early"]["dw_ratio"] == [0.95, 1.05] and PRED["early"]["kl_first"] == [0.0015, 0.02]
    assert PRED["early"]["reward_step1"] == [0.60, 0.88] and PRED["early"]["reward_change_above"] == -0.05


def test_p0() -> None:
    acc = {"gsm8k": {"strict_dropped": 0.792}, "gsm1k": {"strict_dropped": 0.767}}  # both exactly at the tolerance
    read = {"delta_r": -1e-7, "kl": 0.0}
    assert q.p0(PRED["P0"], acc, read, 10)["holds"]
    assert not q.p0(PRED["P0"], acc, read, 0)["holds"]  # an empty pool
    assert not q.p0(PRED["P0"], acc, {"delta_r": None, "kl": 0.0}, 0)["holds"]
    assert not q.p0(PRED["P0"], acc, {"delta_r": 2e-6, "kl": 0.0}, 10)["holds"]
    assert not q.p0(PRED["P0"], {**acc, "gsm8k": {"strict_dropped": 0.793}}, read, 10)["holds"]
    assert not q.p0(PRED["P0"], {}, {}, 0)["holds"]


@pytest.mark.parametrize(("diff", "se", "label"), [(0.02, 0.01, "learns"), (0.02, 0.0, "learns"), (0.03, 0.02, "ambiguous"),
                                                   (0.015, 0.0, "ambiguous"), (0.01, 0.0, "ambiguous"), (0.0099, 0.0, "no gain"),
                                                   (-0.05, 0.01, "no gain"), (None, None, "missing"), (0.05, None, "missing")])
def test_p1(diff: float | None, se: float | None, label: str) -> None:
    assert q.p1(PRED["P1"], {"diff": diff, "se": se}) == label


@pytest.mark.parametrize(("g8k", "g1k", "se", "label"), [(0.05, 0.05, 0.0, "same"),  # d = 0, SE = 0
                                                         (0.05, 0.04, 0.01, "same"), (0.05, 0.02, 0.005, "GSM8K-only"),
                                                         (0.05, 0.02, 0.02, "ambiguous"), (0.05, 0.03, 0.01, "ambiguous"),
                                                         (0.02, 0.05, 0.005, "ambiguous"), (None, 0.05, 0.01, "missing")])
def test_p2(g8k: float | None, g1k: float | None, se: float, label: str) -> None:
    out = q.p2(PRED["P2"], {"diff": g8k, "se": se}, {"diff": g1k, "se": se})
    assert out["label"] == label


def test_p3() -> None:
    assert q.p3(PRED["P3"], {"diff": 0.0149})
    assert not q.p3(PRED["P3"], {"diff": -0.015})
    assert not q.p3(PRED["P3"], {"diff": None}) and not q.p3(PRED["P3"], {})


def p4_reads(x: float) -> dict[str, dict[str, float]]:
    return {str(t): {"delta_r": x} for t in PRED["P4"]["saved_steps"]}


def test_p4() -> None:
    R, N = {"reads": p4_reads(3.0)}, {"reads": p4_reads(1.0), "last": 100}
    assert q.p4(PRED["P4"], R, N)["holds"]  # exactly 3 x
    assert not q.p4(PRED["P4"], R, {"reads": p4_reads(1.1), "last": 100})["holds"]
    assert q.p4(PRED["P4"], R, {"reads": p4_reads(0.0), "last": 100})["holds"]  # N at exactly 0
    assert not q.p4(PRED["P4"], {"reads": p4_reads(0.0)}, {"reads": p4_reads(0.0), "last": 100})["holds"]  # R must be > 0
    cut = {"reads": {t: v for t, v in p4_reads(1.0).items() if int(t) <= 50}, "last": 50}
    assert q.p4(PRED["P4"], R, cut)["ratio"][100]  # N's at its last step
    assert not q.p4(PRED["P4"], R, {})["holds"]  # N not run


def test_p6() -> None:
    def steps(first: float, last: float, share: float = 0.95) -> dict[str, Any]:
        return {"steps": [{"step": t, "mean_len": first if t <= 10 else last, "window_share": share} for t in range(1, 101)]}

    assert q.p6(PRED["P6"], steps(200, 250))["holds"]
    assert q.p6(PRED["P6"], steps(200, 150))["holds"]
    assert not q.p6(PRED["P6"], steps(200, 251))["holds"]
    assert not q.p6(PRED["P6"], steps(200, 200, 0.89))["holds"]
    assert not q.p6(PRED["P6"], steps(0, 200))["holds"]  # a zero denominator
    assert not q.p6(PRED["P6"], {"steps": steps(200, 200)["steps"][:95]})["holds"]  # not fully run
    assert not q.p6(PRED["P6"], {})["holds"]


def test_p7() -> None:
    R = {"gain": {"gsm8k": {"diff": 0.05}}, "reads": {"100": {"kl": 0.1}}}
    assert q.p7(PRED["P7"], R, {"gain": {"gsm8k": {"diff": 0.05}}, "reads": {"100": {"kl": 0.3}}})["holds"]
    assert not q.p7(PRED["P7"], R, {"gain": {"gsm8k": {"diff": 0.06}}, "reads": {"100": {"kl": 0.3}}})["holds"]
    assert not q.p7(PRED["P7"], R, {"gain": {"gsm8k": {"diff": 0.05}}, "reads": {"100": {"kl": 0.29}}})["holds"]
    assert not q.p7(PRED["P7"], R, {})["holds"]


def test_score_all_with_nothing_run() -> None:
    out = q.score_all(PRED, {}, {}, {}, {})
    assert out["P1"]["label"] == "missing" and out["P2"]["label"] == "missing"
    assert not any(out[p]["holds"] for p in ("P0", "P3", "P4", "P6", "P7"))


# ----------------------------------------------------------------------------- the tiny model


EOS = [1, 2]
PAD = 0


def tiny(eos: list[int] | int = EOS) -> Any:
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=128, attn_implementation="sdpa")
    m = Qwen3ForCausalLM(cfg).eval()
    m.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=PAD)
    return m


def greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(-1)


PROMPTS = [[5, 6, 7], [8, 9, 10, 11, 12, 13, 14], [15], [16, 17, 18, 19]]


def test_sampler_matches_generate_greedy_under_left_padding() -> None:
    """Rows of different prompt lengths, per-row windows, finished rows leaving the cache."""
    m = tiny()
    out = q.rollouts(m, PROMPTS, 1, 24, 4, PAD, draw=greedy)
    assert out == q.generate_rollouts(m, PROMPTS, 1, 24, 4, PAD, do_sample=False)
    assert out == [q.rollouts(m, [p], 1, 24, 1, PAD, draw=greedy)[0] for p in PROMPTS]  # the padded batch changes nothing


def test_sampler_cached_logits_equal_the_full_forward() -> None:
    """Every draw sees the logits an uncached, unpadded forward of that row's tokens gives."""
    m = tiny()
    seen: list[torch.Tensor] = []

    def draw(logits: torch.Tensor) -> torch.Tensor:
        seen.append(logits.clone())
        return q.sample_t1(logits)

    torch.manual_seed(1)
    p = PROMPTS[1]
    ids, _ = q.rollouts(m, [p], 1, 16, 1, PAD, draw=draw)[0]
    with torch.no_grad():
        full = m(torch.tensor([p + ids]), use_cache=False).logits[0]
    for i, logits in enumerate(seen):
        assert torch.allclose(logits[0], full[len(p) - 1 + i], atol=1e-4)


def test_t1_sampling_equals_generate_with_top_k_off() -> None:
    """Sampled draws, no row finishing early (an end token outside the vocabulary, equal prompt lengths): the same RNG draws,
    so the same tokens, only if `SAMPLING_T1` really switches every filter off."""
    m = tiny(eos=9999)
    prompts = [[5, 6, 7], [8, 9, 10]]
    torch.manual_seed(3)
    a = q.rollouts(m, prompts, 3, 40, 6, PAD)
    torch.manual_seed(3)
    b = q.generate_rollouts(m, prompts, 3, 40, 6, PAD, **q.SAMPLING_T1)
    assert a == b
    assert all(cut and len(ids) == 37 for ids, cut in a)
    torch.manual_seed(3)
    assert a != q.generate_rollouts(m, prompts, 3, 40, 6, PAD, do_sample=True, top_k=2)  # the check has teeth


def test_sampler_shape_order_and_seed() -> None:
    m = tiny()
    torch.manual_seed(7)
    a = q.rollouts(m, PROMPTS, 5, 30, 8, PAD)
    torch.manual_seed(7)
    assert a == q.rollouts(m, PROMPTS, 5, 30, 8, PAD)
    torch.manual_seed(8)
    assert a != q.rollouts(m, PROMPTS, 5, 30, 8, PAD)
    assert len(a) == 20
    for i, p in enumerate(PROMPTS):
        for ids, cut in a[5 * i : 5 * (i + 1)]:
            assert len(p) + len(ids) <= 30
            assert cut == (ids[-1] not in EOS)
            assert not set(ids[:-1]) & set(EOS)  # stops at the first end token
            assert not cut or len(p) + len(ids) == 30


def scripted(*tokens: int) -> Any:
    """Every live row draws the next token of the script."""
    it = iter(tokens)

    def draw(logits: torch.Tensor) -> torch.Tensor:
        return torch.full((logits.shape[0],), next(it), dtype=torch.long)

    return draw


def test_stop_rule() -> None:
    m = tiny()
    assert q.rollouts(m, [[5, 6]], 1, 10, 1, PAD, draw=scripted(9, 1)) == [([9, 1], False)]  # either end token ends
    assert q.rollouts(m, [[5, 6]], 1, 10, 1, PAD, draw=scripted(2)) == [([2], False)]
    assert q.rollouts(m, [[5, 6]], 1, 5, 1, PAD, draw=scripted(9, 9, 9)) == [([9, 9, 9], True)]  # reaches the cap
    assert q.rollouts(m, [[5, 6]], 1, 5, 1, PAD, draw=scripted(9, 9, 1)) == [([9, 9, 1], False)]  # ends exactly at the cap
    # the longer prompt reaches its cap first and leaves; the other goes on
    assert q.rollouts(m, [[5, 6, 7], [5]], 1, 5, 2, PAD, draw=scripted(9, 9, 9, 1)) == [([9, 9], True), ([9, 9, 9, 1], False)]
    with pytest.raises(AssertionError):
        q.rollouts(m, [[5, 6, 7, 8, 9]], 1, 5, 1, PAD, draw=greedy)


def test_single_int_end_token() -> None:
    assert q.rollouts(tiny(eos=3), [[5]], 1, 8, 1, PAD, draw=scripted(9, 3)) == [([9, 3], False)]


def test_read_h_zero_at_w0_and_sums_over_the_response() -> None:
    m = tiny().requires_grad_(False)
    seqs: list[Seq] = [([5, 6, 7, 8, 9], 3), ([5, 6, 10], 2)]
    W0 = {"lm": m.lm_head.weight.data}

    def as_set(_m: Any) -> Any:
        return {"lm": m.lm_head}

    monkey = pytest.MonkeyPatch()
    monkey.setattr("commute.qwen_sft.linears", as_set)
    try:
        changes, kl = q.read_h(m, W0, seqs, 100)
        assert changes == [0.0, 0.0] and kl == 0.0
        with torch.no_grad():
            before = torch.log_softmax(m(torch.tensor([seqs[0][0]])).logits[0], -1)
        m.lm_head.weight.data = W0["lm"] + 0.1 * torch.randn_like(W0["lm"])
        changes, kl = q.read_h(m, W0, seqs, 100)
        with torch.no_grad():
            after = torch.log_softmax(m(torch.tensor([seqs[0][0]])).logits[0], -1)
        by_hand = sum(float(after[i - 1, seqs[0][0][i]] - before[i - 1, seqs[0][0][i]]) for i in (3, 4))
        assert changes[0] == pytest.approx(by_hand, abs=1e-4)
        assert kl > 0
    finally:
        monkey.undo()


# ----------------------------------------------------------------------------- the weighted step


SEQS: list[Seq] = [([5, 6, 7, 8], 2), ([9, 10, 11], 1), ([12, 13, 14, 15, 16], 3), ([17, 18], 1)]


def hand_loss(m: Any, seqs: list[Seq], w: list[float], div: int) -> torch.Tensor:
    total = torch.zeros(())
    for (ids, p), x in zip(seqs, w, strict=True):
        lp = torch.log_softmax(m(torch.tensor([ids]), use_cache=False).logits[0], -1)
        total = total - x * sum(lp[i - 1, ids[i]] for i in range(p, len(ids)))
    return total / div


def test_weighted_step_gradient_and_budget() -> None:
    m, w = tiny().train(), [0.5, -0.5, 1.0, 0.0]
    loss = hand_loss(m, SEQS, w, 1024)
    loss.backward()
    want = loss.detach()
    grad = {k: p.grad.clone() for k, p in m.named_parameters()}
    for budget in (4, 8, 1000):
        m.zero_grad()
        assert q.weighted_step(m, SEQS, w, 1024, budget) == pytest.approx(want.item(), abs=1e-5)
        for k, p in m.named_parameters():
            assert torch.allclose(p.grad, grad[k], atol=1e-5), (budget, k)


def test_weighted_step_skips_zero_weight() -> None:
    m, rows = tiny().train(), []
    m.model.register_forward_pre_hook(lambda _m, _a, kw: rows.append(kw["input_ids"].shape[0]), with_kwargs=True)
    q.weighted_step(m, SEQS, [1.0, 0.0, 0.0, 0.0], 1, 1000)
    assert rows == [1]


def test_weighted_update_no_step_when_all_zero() -> None:
    m = tiny()
    params = dict(m.named_parameters())
    opt = torch.optim.AdamW(list(params.values()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
    q.weighted_update(m, params, opt, SEQS, [1.0, 1.0, 1.0, 1.0], 1024, 100)  # Adam has moments now: a zero step would move
    before = {k: p.detach().clone() for k, p in params.items()}
    rec = q.weighted_update(m, params, opt, SEQS, [0.0, 0.0, 0.0, 0.0], 1024, 100)
    assert rec == {"live": False, "loss": 0.0, "grad_norm": 0.0}
    assert all(torch.equal(p, before[k]) for k, p in params.items())
    assert not m.training


def test_weighted_update_first_step_is_lr_sign() -> None:
    m = tiny()
    params = {"o": m.model.layers[0].self_attn.o_proj.weight}
    for p in m.parameters():
        p.requires_grad_(False)
    params["o"].requires_grad_(True)
    W0 = {"o": params["o"].detach().clone()}
    opt = torch.optim.AdamW([params["o"]], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
    rec = q.weighted_update(m, params, opt, SEQS, [0.5, -0.5, 1.0, 0.25], 1, 100)
    assert rec["live"] and math.isfinite(rec["loss"]) and rec["grad_norm"] > 0
    assert q.delta_fp32(params, W0)["o"].abs().max() == pytest.approx(1e-3, rel=1e-3)
    assert m.model.layers[0].self_attn.o_proj.weight.grad is None

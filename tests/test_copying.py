"""E62's pure parts: the prefix score, the ablation hook, the cue windows and every filed rule's edge cases."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import nn

from commute import copying as CP
from commute.data import LAST, target_tokens

TH = {"G_min": 5.0, "share_old_min": 0.5, "share_new_max": 0.2, "floor_max": 0.15, "absent_max": 0.2, "broken_max": 0.5, "spec_min": 0.5,
      "min_seeds": 2, "r_cut": 0.5, "r_gap": 0.24, "r_noise": 0.045, "top1_ref": 0.8247, "top1_ref_tol": 0.01, "top1_H_max": 0.3,
      "top1_random_min": 0.7, "prior_stop": 1e-2, "prior_max": 1e-3, "prior_band": [1e-6, 1e-4], "prior_match_max": 2}
HALF = 20
KEYS = [100, 101, 102, 103]
AVOID = [200, 201, 202, 203]  # a word pool
BS = [300, 301, 302, 303, 304]


def test_stripe_mask_marks_the_token_after_the_earlier_copy():
    toks = torch.tensor([[9, 1, 2, 1, 2]])  # start, then 1 2 twice
    m = CP.stripe_mask(toks)
    assert m[0].nonzero().tolist() == [[3, 2], [4, 3]]
    assert CP.stripe_mask(toks, -1)[0].nonzero().tolist() == [[3, 0], [4, 1]]


def test_stripe_score_reads_one_on_a_perfect_pattern_and_chance_on_uniform():
    toks = CP.repeated_random(list(range(10, 60)), 5, 4, 2, 0, 0)
    m = CP.stripe_mask(toks)
    perfect = m.double() / m.double().sum(-1, keepdim=True).clamp_min(1)
    assert CP.stripe_score(perfect[:, None], m)[0] < 1  # later repeats split over earlier ones
    one = torch.zeros(1, 1, 3, 3)
    one[0, 0, 2, 1] = 1.0
    t3 = torch.tensor([[1, 5, 1]])
    assert float(CP.stripe_score(one, CP.stripe_mask(t3))[0]) == 1.0
    assert float(CP.stripe_score(CP.uniform_attention(3), CP.stripe_mask(t3))[0]) == pytest.approx(1 / 3)


def test_stripe_score_empty_mask_is_nan():
    toks = torch.tensor([[1, 2, 3]])
    assert math.isnan(float(CP.stripe_score(torch.ones(1, 2, 3, 3), CP.stripe_mask(toks))[0]))


def test_over_bar_ties_and_order():
    s = torch.tensor([[0.1, 0.5], [0.5, 0.0]])
    assert CP.over_bar(s, 0.5) == [(0, 1), (1, 0)]
    assert CP.over_bar(s, 0.9) == []
    assert CP.by_score(s) == [(0, 1), (1, 0), (0, 0), (1, 1)]


def test_heads_status_edges():
    assert CP.heads_status([], 24) == "empty"
    assert CP.heads_status([(0, h) for h in range(6)] * 4, 24) == "all"
    assert CP.heads_status([(0, h) for h in range(13)], 24) == "over_half"
    assert CP.heads_status([(0, h) for h in range(12)], 24) == "ok"


def test_random_others_are_outside_H_and_the_right_size():
    H = [(0, 0), (3, 5)]
    for d in CP.random_others(H, 4, 6, 5, 0):
        assert len(d) == 2 and not set(d) & set(H)


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Identity()


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attn()


class _Net:
    def __init__(self) -> None:
        self.net: Any = type("N", (), {"h": [_Block()]})()
        self.device = "cpu"


def test_ablated_replaces_only_the_named_head_and_cleans_up():
    net = _Net()
    means = torch.stack([torch.full((2,), 7.0), torch.full((2,), 8.0), torch.full((2,), 9.0)])[None]  # (1 block, 3 heads, dim 2)
    x = torch.arange(12.0).view(1, 2, 6)
    o = net.net.h[0].attn.o_proj
    with CP.ablated(net, [(0, 1)], means):
        y = o(x)
    assert torch.equal(y[..., 2:4], torch.full((1, 2, 2), 8.0))
    assert torch.equal(y[..., :2], x[..., :2]) and torch.equal(y[..., 4:], x[..., 4:])
    assert torch.equal(o(x), x)


class _TinyLlama:
    """A tiny model of the 67M's class, on CPU (never the 67M itself), behind ST.Net's `run` and `net`."""

    def __init__(self) -> None:
        from param_decomp.pretrain.models.llama_simple_mlp import (
            LlamaSimpleMLP,
            LlamaSimpleMLPConfig,
        )

        torch.manual_seed(0)
        cfg = LlamaSimpleMLPConfig(model_type="LlamaSimpleMLP", block_size=32, vocab_size=50, n_layer=2, n_head=3, n_embd=24, n_intermediate=48, n_ctx=32,
                                   n_key_value_heads=3, flash_attention=False, rotary_dim=8)
        self.net: Any = LlamaSimpleMLP(cfg).eval()
        self.device = "cpu"

    @torch.no_grad()
    def run(self, toks: torch.Tensor) -> tuple[torch.Tensor, dict]:
        out = self.net(toks)
        return out[0], {}


def test_attention_patterns_reproduce_the_model_on_a_real_module():
    net = _TinyLlama()
    toks = torch.randint(0, 50, (4, 12), generator=torch.Generator().manual_seed(1))
    pats, err = CP.attention_patterns(net, toks)
    assert err < 1e-5 and len(pats) == 2 and pats[0].shape == (4, 3, 12, 12)
    assert torch.allclose(pats[0].sum(-1), torch.ones(4, 3, 12), atol=1e-5) and float(pats[0].triu(1).abs().max()) == 0.0


def test_mean_ablation_on_a_real_module():
    net = _TinyLlama()
    spans = torch.randint(0, 50, (8, 12), generator=torch.Generator().manual_seed(2))
    means = CP.head_means(net, spans, batch=3)
    assert means.shape == (2, 3, 8)
    base = net.run(spans)[0]
    with CP.ablated(net, [(0, 1)], means):
        one = net.run(spans)[0]
    assert not torch.equal(base, one) and torch.equal(net.run(spans)[0], base)
    seen = {}
    h = net.net.h[1].attn.o_proj.register_forward_pre_hook(lambda _m, a: seen.__setitem__("y", a[0]))  # the mean is what the hook sums
    net.run(spans)
    h.remove()
    assert torch.allclose(seen["y"].mean((0, 1)).view(3, 8), means[1], atol=1e-5)


def rows() -> torch.Tensor:
    return torch.randint(1000, 5000, (50, 512), generator=torch.Generator().manual_seed(0))


def test_cue_windows_layout_and_uniqueness():
    w, b, p = CP.cue_windows(rows(), 7, KEYS, BS, AVOID, HALF, 40, 0)
    t = w.tokens
    idx = torch.arange(40)
    assert t.shape == (40, 41) and bool((t[:, HALF] == 7).all())
    a = t[:, HALF - 1]
    assert bool((t[idx, p] == a).all()) and bool((t[idx, p + 1] == b).all())
    assert bool(((p >= 2) & (p + 1 < HALF - 2)).all())
    assert bool(((t == a[:, None]).sum(1) == 2).all()) and bool(((t == b[:, None]).sum(1) == 1).all())
    assert not bool(torch.isin(t, torch.tensor(AVOID)).any())


def test_cue_windows_drop_a_natural_repeat():
    r = rows()
    r[:, 100:400] = 100  # key 100 everywhere in the middle of every row: such windows must go
    w, _, _ = CP.cue_windows(torch.cat([r, rows()]), 7, KEYS, BS, AVOID, HALF, 20, 0)
    a = w.tokens[:, HALF - 1]
    assert bool(((w.tokens == a[:, None]).sum(1) == 2).all())


def test_controls_break_what_they_name():
    w, b, p = CP.cue_windows(rows(), 7, KEYS, BS, AVOID, HALF, 30, 0)
    idx = torch.arange(30)
    a = w.tokens[:, HALF - 1]
    br = CP.broken_pair(w.tokens, p, KEYS, HALF, 0)
    assert not bool((br[idx, p] == a).any()) and bool((br[idx, p + 1] == b).all()) and torch.isin(br[idx, p], torch.tensor(KEYS)).all()
    ab = CP.absent_b(w.tokens, p, b, BS, 0)
    assert not bool((ab == b[:, None]).any()) and bool((ab[idx, p] == a).all())


def test_build_cue_pools_are_disjoint():
    r = rows()
    ranked = list(range(2000, 2200))
    S = {"pool_ranks": [0, 200], "pool_seed": 0, "n_keys": 4, "n_word_pool": 16, "heldout_every": 5, "n_train": 16, "n_read": 8, "seed": 0}
    c = CP.build_cue(r, 7, ranked, lambda t: t != 2005, HALF, S)
    assert "lookup" not in c.target and not c.words  # the words come after W0's prior for B is measured
    c.set_words(dict(zip(c.keys, c.word_pool[5:9], strict=True)))
    bs_train = set(c.target["copy"]["train"].tolist())
    bs_read = set(c.bs_read)
    pool = set(c.word_pool)
    words = set(c.words.values())
    assert not (set(c.keys) & pool) and not (bs_train & bs_read) and not ((bs_train | bs_read) & pool) and 2005 not in pool | set(c.keys) | bs_read | bs_train
    assert not set(c.target["copy"]["read"].tolist()) & pool
    assert set(c.target["lookup"]["read"].tolist()) <= words and set(c.target["lookup"]["train"].tolist()) <= words
    assert not torch.isin(torch.cat([c.train.tokens, c.read.tokens]), torch.tensor(c.word_pool)).any()  # no window holds any candidate
    assert torch.equal(c.read_windows("copy")[:, :41], c.read_windows("lookup")[:, :41])


def test_pick_words_are_within_the_factor_per_key_and_distinct():
    keys, pool = [1, 2], [10, 11, 12, 13]
    at = torch.tensor([1, 1, 2, 2])
    # per-key mean log P of candidates 10..13: key 1 -> (-5, -9, -5.5, -6); key 2 -> (-5, -5.5, -9, -9)
    lp = torch.tensor([[-4.0, -9.0, -5.0, -6.0], [-6.0, -9.0, -6.0, -6.0], [-5.0, -5.0, -9.0, -9.0], [-5.0, -6.0, -9.0, -9.0]])
    target = -5.0
    words = CP.pick_words(keys, pool, at, lp, target, 2.0, 0)  # |mean - target| <= ln 2 = 0.693
    assert words[1] in (10, 12) and words[2] in (10, 11) and words[1] != words[2]
    per_key = {k: lp[at == k].mean(0) for k in keys}
    assert all(abs(float(per_key[k][pool.index(w)]) - target) <= math.log(2) for k, w in words.items())
    with pytest.raises(AssertionError):  # no candidate in range for some key
        CP.pick_words(keys, pool, at, lp, -20.0, 2.0, 0)
    with pytest.raises(AssertionError):  # a key with no window is never served
        CP.pick_words([1, 3], pool, at, lp, target, 2.0, 0)


def test_prior_ratio_edges():
    assert CP.prior_ratio(2e-5, 1e-5) == 2.0 and CP.prior_ratio(1e-5, 2e-5) == 2.0
    assert CP.prior_ratio(1e-5, 1e-5) == 1.0
    assert math.isinf(CP.prior_ratio(0.0, 1e-5)) and math.isinf(CP.prior_ratio(math.nan, 1e-5))


def test_last_column_target():
    w = torch.tensor([[1, 2, 3, 9], [4, 5, 6, 8]])
    assert target_tokens(LAST, w, 1).tolist() == [9, 8]


def test_control_a_branches():
    good = (0.82, 0.1, [0.8, 0.75, 0.9])
    assert CP.control_a(*good, TH) == "pass"
    assert CP.control_a(0.5, 0.1, [0.8], TH) == "no_copying"
    assert CP.control_a(0.82, 0.3, [0.8], TH) == "H_insufficient"  # a tie fails
    assert CP.control_a(0.82, 0.1, [0.1, 0.7, 0.9], TH) == "random_harms"  # the median, a tie fails
    assert CP.control_a(0.82, 0.1, [0.1, 0.8, 0.9], TH) == "pass"  # one bad draw does not stop it
    assert CP.control_a(0.82, 0.1, [], TH) == "unreadable"
    assert CP.control_a(0.82, math.nan, [0.8], TH) == "unreadable"


def test_prior_band_branches():
    f = lambda p: CP.prior_band(p, TH)
    assert [f(1e-7), f(1e-5), f(5e-4), f(5e-3), f(2e-2), f(math.nan)] == ["low", "expected", "over_band", "high", "stop", "stop"]
    assert [f(1e-2), f(1e-3), f(1e-4), f(1e-6)] == ["high", "over_band", "expected", "expected"]


def test_read_step_branches():
    assert CP.read_step({30: 6, 100: 9}, {30: 7, 100: 9}, 5) == 30
    assert CP.read_step({30: 1, 100: 9}, {30: 7, 100: 9}, 5) == 100
    assert CP.read_step({30: 6, 100: 9}, {30: 1, 100: 2}, 5) == 30  # the lookup never reaches it
    assert CP.read_step({30: 1, 100: 2}, {30: 1, 100: 9}, 5) == 100  # the copy map never reaches it
    assert CP.read_step({30: 1, 100: math.nan}, {30: 1, 100: 2}, 5) == 100  # nothing does: the last


def test_share_and_label_edges():
    assert CP.share_lost(10.0, 4.0) == pytest.approx(0.6)
    assert math.isnan(CP.share_lost(0.0, 0.0))  # the zero update
    assert math.isnan(CP.share_lost(-1.0, 1.0))
    assert CP.share_label(math.nan, 1.0, TH) == "unreadable"
    assert CP.share_label(4.9, 0.0, TH) == "too_small"
    assert CP.share_label(5.0, 2.5, TH) == "old"  # gate and 0.5 ties
    assert CP.share_label(10.0, 8.0, TH) == "new"  # a 0.2 tie
    assert CP.share_label(10.0, 12.0, TH) == "new"  # a negative share
    assert CP.share_label(10.0, 6.0, TH) == "between"


def state(label: str, r: float = 0.7, **kw: float) -> dict:
    return {"label": label, "r": r, "random_median": 0.05, "absent_ratio": 0.1, "broken_ratio": 0.2, "spec_frac": 0.8, **kw}


def states(copy: list[dict], lookup: list[dict]) -> dict:
    return {**{f"copy_s{i}": s for i, s in enumerate(copy)}, **{f"lookup_s{i}": s for i, s in enumerate(lookup)}}


def verdict(copy: list[dict], lookup: list[dict], prior: str = "expected", ratio: float = 1.2) -> dict:
    return CP.score(states(copy, lookup), prior, TH, [0, 1, 2], ratio)


def test_score_every_outcome():
    old, new = [state("old")] * 3, [state("new", 0.3)] * 3
    v = verdict(old, new)
    assert v["verdict"] == "r_follows_mechanism" and v["r_scored"] and not v["pair_ignored"] and not v["prior_high"]
    assert verdict([state("old", 0.3)] * 3, new)["verdict"] == "r_follows_prior"
    assert verdict([state("old", 0.5)] * 3, [state("new", 0.49)] * 3)["verdict"] == "untested"  # the gap
    assert verdict(old, [state("new", 0.6)] * 3)["verdict"] == "untested"  # both high
    assert verdict([state("old", math.nan)] * 3, new)["verdict"] == "untested"
    assert verdict([state("too_small")] * 3, new)["verdict"] == "no_label_copy_did_not_learn"
    assert verdict(old, [state("too_small")] * 3)["verdict"] == "no_label_lookup_did_not_learn"
    assert verdict([state("unreadable")] * 3, new)["verdict"] == "no_label_unreadable"
    assert verdict([], new)["verdict"] == "no_label_unreadable"  # missing states
    assert verdict(old, [state("new", random_median=0.2)] * 3)["verdict"] == "no_label_floor_high"
    assert verdict([state("old", random_median=0.15)] * 3, new)["verdict"] == "no_label_floor_high"  # a tie fails
    assert verdict([state("old", absent_ratio=0.3)] * 3, new)["verdict"] == "no_label_copy_is_a_lean"
    assert verdict(old, [state("new", spec_frac=0.4)] * 3)["verdict"] == "no_label_lookup_is_a_lean"
    assert verdict([state("between")] * 3, new)["verdict"] == "copy_not_elicitation"
    assert verdict([state("new")] * 3, new)["verdict"] == "copy_not_elicitation"
    assert verdict([state("old"), state("old"), state("between")], new)["verdict"] == "copy_not_elicitation"
    assert verdict(old, [state("between")] * 3)["verdict"] == "lookup_uses_the_heads"
    assert not verdict([state("too_small")] * 3, new)["r_scored"]


def test_score_prior_match_edges():
    old, new = [state("old")] * 3, [state("new", 0.3)] * 3
    assert verdict(old, new, ratio=2.0)["verdict"] == "r_follows_mechanism"  # exactly at the bar passes
    v = verdict(old, new, ratio=2.0001)
    assert v["verdict"] == "priors_not_matched" and not v["r_scored"] and v["r_copy"] == [0.7] * 3  # r reported, not scored
    assert verdict(old, new, ratio=math.inf)["verdict"] == "priors_not_matched"
    assert verdict(old, new, ratio=math.nan)["verdict"] == "priors_not_matched"
    assert verdict([state("too_small")] * 3, new, ratio=3.0)["verdict"] == "priors_not_matched"  # first in the order
    assert verdict(old, new)["prior_ratio"] == 1.2


def test_r_verdict_with_priors_matched():
    f = lambda c, k: CP.r_verdict(c, k, TH)["class"]
    assert f([0.44] * 3, [0.10] * 3) == "untested"  # all low, gap 0.34: r separates the maps, not at the cut
    assert f([0.30, 0.32, 0.28], [0.25, 0.20, 0.28]) == "r_follows_prior"  # all low, gap 0.05
    assert f([0.49] * 3, [0.20] * 3) == "untested"  # all low, gap 0.29 over the bar
    assert f([0.40] * 3, [0.30] * 3) == "r_follows_prior"
    assert f([0.9] * 3, [0.1] * 3) == "r_follows_mechanism"
    assert f([0.7] * 3, [0.6] * 3) == "untested"  # both high


def test_read_step_on_the_specific_gain():
    raw = {30: 6.0, 100: 20.0, 300: 25.0}  # over the gate of 5 at step 30 already
    absent = {30: 4.5, 100: 5.0, 300: 6.0}  # a general push toward the pool's tokens
    spec = {s: raw[s] - absent[s] for s in raw}  # 1.5, 15, 19
    look = {30: 9.0, 100: 12.0, 300: 14.0}
    assert CP.read_step(raw, look, 5.0) == 30
    assert CP.read_step(spec, look, 5.0) == 100
    assert CP.read_step({30: 4.9, 100: 4.9, 300: 4.9}, look, 5.0) == 30  # the copy map never reaches it: the lookup's first step


def test_score_a_seed_under_the_gate_is_left_out():
    new = [state("new", 0.3)] * 3
    v = verdict([state("old"), state("old"), state("too_small", 0.1)], new)
    assert v["verdict"] == "r_follows_mechanism" and len(v["r_copy"]) == 2
    assert verdict([state("old"), state("too_small"), state("too_small")], new)["verdict"] == "no_label_copy_did_not_learn"


def test_score_flags_beside_the_verdict():
    old, new = [state("old")] * 3, [state("new", 0.3)] * 3
    assert verdict([state("old", broken_ratio=0.8)] * 3, new)["pair_ignored"]
    assert verdict([state("old", broken_ratio=0.8)] * 3, new)["verdict"] == "r_follows_mechanism"
    assert verdict(old, new, "high")["prior_high"]

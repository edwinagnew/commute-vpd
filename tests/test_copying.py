"""E62's pure parts: the prefix score, the ablation hook, the cue windows and every filed rule's edge cases."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from commute import copying as CP
from commute.data import LAST, target_tokens

TH = {"G_min": 5.0, "share_old_min": 0.5, "share_new_max": 0.2, "floor_max": 0.15, "absent_max": 0.2, "broken_max": 0.5, "spec_min": 0.5,
      "min_seeds": 2, "r_cut": 0.5, "r_gap": 0.24, "r_noise": 0.045, "top1_ref": 0.8247, "top1_ref_tol": 0.01, "top1_H_max": 0.3,
      "top1_random_min": 0.7, "prior_stop": 1e-2, "prior_max": 1e-3, "prior_band": [1e-6, 1e-4], "prior_match_max": 2}
HALF = 20
KEYS = [100, 101, 102, 103]
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
    w, b, p = CP.cue_windows(rows(), 7, KEYS, BS, HALF, 40, 0)
    t = w.tokens
    idx = torch.arange(40)
    assert t.shape == (40, 41) and bool((t[:, HALF] == 7).all())
    a = t[:, HALF - 1]
    assert bool((t[idx, p] == a).all()) and bool((t[idx, p + 1] == b).all())
    assert bool(((p >= 2) & (p + 1 < HALF - 2)).all())
    assert bool(((t == a[:, None]).sum(1) == 2).all()) and bool(((t == b[:, None]).sum(1) == 1).all())


def test_cue_windows_drop_a_natural_repeat():
    r = rows()
    r[:, 100:400] = 100  # key 100 everywhere in the middle of every row: such windows must go
    w, _, _ = CP.cue_windows(torch.cat([r, rows()]), 7, KEYS, BS, HALF, 20, 0)
    a = w.tokens[:, HALF - 1]
    assert bool(((w.tokens == a[:, None]).sum(1) == 2).all())


def test_controls_break_what_they_name():
    w, b, p = CP.cue_windows(rows(), 7, KEYS, BS, HALF, 30, 0)
    idx = torch.arange(30)
    a = w.tokens[:, HALF - 1]
    br = CP.broken_pair(w.tokens, p, KEYS, HALF, 0)
    assert not bool((br[idx, p] == a).any()) and bool((br[idx, p + 1] == b).all()) and torch.isin(br[idx, p], torch.tensor(KEYS)).all()
    ab = CP.absent_b(w.tokens, p, b, BS, 0)
    assert not bool((ab == b[:, None]).any()) and bool((ab[idx, p] == a).all())


def test_build_cue_pools_are_disjoint():
    r = rows()
    ranked = list(range(2000, 2300))
    S = {"pool_ranks": [0, 200], "pool_seed": 0, "n_keys": 4, "heldout_every": 5, "n_train": 16, "n_read": 8, "seed": 0}
    c = CP.build_cue(r, 7, ranked, lambda t: t != 2005, HALF, S)
    assert "lookup" not in c.target and not c.words  # the words come after W0's prior for B is measured
    c.set_words(dict(zip(c.keys, c.word_pool[5:9], strict=True)))
    bs_train = set(c.target["copy"]["train"].tolist())
    bs_read = set(c.bs_read)
    pool = set(c.word_pool)
    words = set(c.words.values())
    assert not (set(c.keys) & pool) and not (bs_train & bs_read) and not ((bs_train | bs_read) & pool) and 2005 not in pool | set(c.keys) | bs_read | bs_train
    assert not set(c.target["copy"]["read"].tolist()) & pool
    assert pool == set(range(2200, 2300))  # every intact token outside the keys and Bs (which took ranks 0-199 whole), at any rank
    assert set(c.target["lookup"]["read"].tolist()) <= words and set(c.target["lookup"]["train"].tolist()) <= words
    assert torch.equal(c.read_windows("copy")[:, :41], c.read_windows("lookup")[:, :41])


def test_pick_words_are_within_the_factor_per_key_and_distinct():
    keys, pool = [1, 2], [10, 11, 12, 13]
    at = torch.tensor([1, 1, 2, 2])
    # per-key mean log P of candidates 10..13: key 1 -> (-5, -9, -5.5, -6); key 2 -> (-5, -5.5, -9, -9)
    lp = torch.tensor([[-4.0, -9.0, -5.0, -6.0], [-6.0, -9.0, -6.0, -6.0], [-5.0, -5.0, -9.0, -9.0], [-5.0, -6.0, -9.0, -9.0]])
    target = -5.0
    own: dict[int, set[int]] = {1: set(), 2: set(), 3: set()}
    words = CP.pick_words(keys, pool, at, lp, target, 2.0, 0, own)  # |mean - target| <= ln 2 = 0.693
    assert words[1] in (10, 12) and words[2] in (10, 11) and words[1] != words[2]
    per_key = {k: lp[at == k].mean(0) for k in keys}
    assert all(abs(float(per_key[k][pool.index(w)]) - target) <= math.log(2) for k, w in words.items())
    with pytest.raises(AssertionError):  # no candidate in range for some key
        CP.pick_words(keys, pool, at, lp, -20.0, 2.0, 0, own)
    with pytest.raises(AssertionError):  # a key with no window is never served
        CP.pick_words([1, 3], pool, at, lp, target, 2.0, 0, own)


def test_pick_words_reject_a_candidate_in_the_keys_own_windows():
    keys, pool = [1, 2], [10, 11, 12, 13]
    at = torch.tensor([1, 1, 2, 2])
    lp = torch.tensor([[-4.0, -9.0, -5.0, -6.0], [-6.0, -9.0, -6.0, -6.0], [-5.0, -5.0, -9.0, -9.0], [-5.0, -6.0, -9.0, -9.0]])
    # in range: key 1 -> 10, 12; key 2 -> 10, 11
    assert CP.admissible(keys, pool, at, lp, -5.0, 2.0, {1: set(), 2: set()}) == {1: [10, 12], 2: [10, 11]}
    assert CP.admissible(keys, pool, at, lp, -5.0, 2.0, {1: {10}, 2: {11, 99}}) == {1: [12], 2: [10]}  # a word in the key's own windows is out
    for s in range(5):  # whatever the draw order, key 1 never gets 10 or key 2 gets 11
        w = CP.pick_words(keys, pool, at, lp, -5.0, 2.0, s, {1: {10}, 2: {11}})
        assert w == {1: 12, 2: 10}
    with pytest.raises(AssertionError):  # every in-range word of key 1 sits in its own windows
        CP.pick_words(keys, pool, at, lp, -5.0, 2.0, 0, {1: {10, 12}, 2: set()})
    own = CP.own_tokens(keys, torch.tensor([[1, 5, 10, 1], [2, 11, 2, 6], [1, 7, 7, 7]]), 3)  # half 3: A sits at index 2
    assert own == {1: set(), 2: {2, 11, 6}}


def test_prior_histogram_counts_by_bin():
    lp = torch.log(torch.tensor([[10**-3.2, 10**-2.7, 10**-5.2]] * 2))
    assert CP.prior_histogram(lp) == {"-5.5": 1, "-3.5": 1, "-3.0": 1}


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
    assert verdict(old, [state("new", 0.6)] * 3)["verdict"] == "r_follows_prior"  # both high, gap 0.1 (Amendment 1)
    assert verdict([state("old", 0.2)] * 3, [state("new", 0.7)] * 3)["verdict"] == "r_reverses_label"
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
    assert f([0.7] * 3, [0.6] * 3) == "r_follows_prior"  # both high, gap 0.1 (Amendment 1, 2026-11-02)
    assert f([0.5] * 3, [0.5] * 3) == "r_follows_prior"  # both at the cut count high
    assert f([0.9] * 3, [0.6] * 3) == "untested"  # both high with a gap of 0.3: nothing names it
    assert f([0.2] * 3, [0.7] * 3) == "r_reverses_label"  # copy low, lookup high, gap -0.5 (the named surprise)
    assert f([0.2] * 3, [0.44 + 0.1] * 3) == "r_reverses_label"  # gap -0.34
    assert f([0.4] * 3, [0.5] * 3) == "untested"  # copy low, lookup high but the gap is only -0.1: not a reversal
    assert f([0.2, 0.7, 0.2], [0.7] * 3) == "untested"  # one copy seed over the cut: no reversal


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


# ----------------------------------------------------------------------------- Amendment 2


A2 = {"min_gain": 1.0, "min_seeds": 2, "top1_H_max": 0.05, "top1_ref_min": 0.66, "tol": 0.092, "other_path_max": 0.25, "floor_max": 0.15, "two_head_min": 0.7,
      "two_head_lookup_max": 0.1, "bracket_max": 4.18}


def _tiny_component_model() -> tuple[Any, Any]:
    """A real ComponentModel over the tiny Llama (never the 67M), q, k, v and o_proj of both blocks decomposed."""
    from param_decomp.configs import LayerwiseCiConfig
    from param_decomp.models.batch_and_loss_fns import make_run_batch
    from param_decomp.models.component_model import ComponentModel
    from param_decomp.utils.module_utils import ModulePathInfo

    tiny = _TinyLlama()
    tiny.net.requires_grad_(False)
    paths = [f"h.{b}.attn.{p}_proj" for b in range(2) for p in "qkvo"]
    model = ComponentModel(target_model=tiny.net, run_batch=make_run_batch(0), module_path_info=[ModulePathInfo(module_path=k, C=4) for k in paths],
                           ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]), sigmoid_type="leaky_hard").eval()
    return model, CP.AllGatesOn(model)


def test_ablated_reaches_the_components_of_a_decomposed_o_proj():
    """The hook finding: o_proj is decomposed, and the pre-hook's edit is what its components take, with no change to `ablated`."""
    model, net = _tiny_component_model()
    toks = torch.randint(0, 50, (2, 12), generator=torch.Generator().manual_seed(3))
    means = torch.full((2, 3, 8), 5.0)
    seen = {}
    h = model.components["h.1.attn.o_proj"].register_forward_pre_hook(lambda _m, a, k: seen.__setitem__("x", (a or (k["x"],))[0]), with_kwargs=True)
    base = net.run(toks)[0]
    with CP.ablated(net, [(1, 2)], means):
        off = net.run(toks)[0]
    h.remove()
    assert torch.equal(seen["x"][..., 16:], torch.full((2, 12, 8), 5.0))
    assert base.shape == (2, 12, 50) and not torch.equal(base, off) and torch.equal(net.run(toks)[0], base)


def test_masked_lo_windows_ci_len_keeps_the_target_column_out():
    from commute import local_faith as LF

    model, _ = _tiny_component_model()
    toks = torch.randint(0, 50, (5, 13), generator=torch.Generator().manual_seed(4))  # 12 tokens, the target in the last column
    gates = [{k: torch.ones(len(toks[sl]), 12, C) for k, C in model.module_to_c.items()} for sl in LF._batches(toks, 2)]
    rows = LF.masked_lo_windows(model, toks, 6, LAST, gates, 2, 12)
    assert rows.shape == (5,) and float(rows.mean()) == pytest.approx(LF.masked_lo(model, toks, 6, LAST, gates, 2, 12)[0])


@pytest.mark.parametrize("base, off, ref, want", [
    (0.82, 0.01, 0.70, "pass"), (0.82, 0.05, 0.70, "hook_missed"), (0.82, 0.01, 0.66, "ref_harmed"), (0.66, 0.01, 0.70, "no_copying"),
    (0.82, math.nan, 0.70, "unreadable"), (0.82, 0.80, 0.70, "hook_missed"),
])
def test_control_masked(base, off, ref, want):
    assert CP.control_masked(base, off, ref, A2) == want


@pytest.mark.parametrize("gain, m, t, want", [
    (0.0, 0.5, 0.5, "too_small"), (-3.0, 0.5, 0.5, "too_small"), (0.999, 0.5, 0.5, "too_small"), (1.0, 0.5, 0.5, "within"), (math.nan, 0.5, 0.5, "unreadable"),
    (13.0, math.nan, 0.5, "unreadable"), (13.0, 0.5, math.inf, "unreadable"), (13.0, 0.625, 0.5, "over"), (13.0, 0.5 + 0.0625, 0.5, "within"),
    (13.0, 0.3, 0.5, "between"), (13.0, 0.25, 0.5, "between"), (13.0, 0.2499, 0.5, "other_path"), (13.0, -0.4, 0.5, "other_path"), (13.0, 1.4, 0.5, "over"),
    (6.0, 0.01, 0.0, "within"), (6.0, -0.2, 0.0, "other_path"), (6.0, 0.2, 0.0, "over"),
])
def test_masked_class(gain, m, t, want):
    assert CP.masked_class(gain, m, t, A2) == want


def test_masked_class_tie_at_the_tolerance_counts_within():
    th = {**A2, "tol": 0.25}
    assert CP.masked_class(13.0, 0.75, 0.5, th) == "within" and CP.masked_class(13.0, 0.25, 0.5, th) == "within"


@pytest.mark.parametrize("labels, want", [
    (["within"] * 3, "r_reads_the_head"), (["within", "within", "too_small"], "r_reads_the_head"), (["within", "over", "over"], "head_more_than_target"),
    (["over"] * 3, "head_more_than_target"), (["other_path"] * 3, "mask_reaches_b_another_way"), (["within", "other_path", "within"], "in_between"),
    (["between"] * 3, "in_between"), (["within", "between", "over"], "in_between"), (["too_small", "too_small", "within"], "too_small"), (["unreadable", "within", "within"], "unreadable"), ([], "unreadable"),
])
def test_masked_reading(labels, want):
    assert CP.masked_reading(labels, [0.01] * len(labels), A2) == want


def test_masked_reading_floor_fails_on_a_tie_and_a_nan():
    assert CP.masked_reading(["within"] * 3, [0.0, 0.15, 0.0], A2) == "floor_high"
    assert CP.masked_reading(["within"] * 3, [0.0, math.nan, 0.0], A2) == "floor_high"
    assert CP.masked_reading(["unreadable"] * 3, [0.5] * 3, A2) == "unreadable"


@pytest.mark.parametrize("G, both, single, want", [
    (12.0, 0.7, 0.5, "circuit"), (12.0, 0.95, 0.5, "circuit"), (12.0, 0.55, 0.5, "no_more"), (12.0, 0.65, 0.5, "between"), (12.0, 0.2, 0.5, "between"),
    (0.0, 0.7, 0.5, "too_small"), (12.0, math.nan, 0.5, "unreadable"),
])
def test_two_head_class(G, both, single, want):
    assert CP.two_head_class(G, both, single, A2) == want


@pytest.mark.parametrize("G, both, want", [(15.0, 0.1, "clean"), (15.0, -0.2, "clean"), (15.0, 0.11, "uses_circuit"), (0.5, 0.0, "too_small"), (15.0, math.nan, "unreadable")])
def test_lookup_two_head(G, both, want):
    assert CP.lookup_two_head(G, both, A2) == want


@pytest.mark.parametrize("G, b, intact, want", [
    (5.0, 0.0, 0.5, "outside_head"), (5.0, -0.3, 0.5, "outside_head"), (5.0, 0.092, 0.5, "outside_head"), (5.0, 0.45, 0.5, "through_head"), (5.0, 0.3, 0.5, "between"),
    (5.0, 0.9, 0.5, "between"), (0.0, 0.1, 0.5, "too_small"), (5.0, math.nan, 0.5, "unreadable"), (5.0, 0.15, 0.1, "through_head"),
])
def test_broken_class(G, b, intact, want):
    assert CP.broken_class(G, b, intact, A2) == want


def test_own_fall_bracket_and_size():
    assert CP.own_fall_class(12.0, 0.8, 0.5, A2) == "undercounts" and CP.own_fall_class(12.0, 0.5625, 0.5, A2) == "stands"
    assert CP.own_fall_class(12.0, 0.1, 0.5, A2) == "stands" and CP.own_fall_class(0.0, 0.1, 0.5, A2) == "too_small" and CP.own_fall_class(12.0, math.nan, 0.5, A2) == "unreadable"
    assert [CP.bracket_class(x, A2) for x in (-0.1, 0.0, 4.18, 4.2, math.nan)] == ["below", "inside", "inside", "above", "unreadable"]
    assert CP.size_reading(0.7, 0.34, True) == "size_accounts" and CP.size_reading(0.7, 0.35, True) == "size_does_not" and CP.size_reading(0.7, -0.1, True) == "size_accounts"
    assert CP.size_reading(0.7, 0.9, True) == "size_does_not" and CP.size_reading(0.7, 0.1, False) == "unreadable"
    assert CP.size_reading(0.0, 0.1, True) == "unreadable" and CP.size_reading(-0.2, 0.1, True) == "unreadable" and CP.size_reading(0.7, math.nan, True) == "unreadable"


def test_agree_over_seeds():
    assert CP.agree(["a", "a", "too_small"], A2) == "a" and CP.agree(["a", "b", "a"], A2) == "in_between"
    assert CP.agree(["a", "too_small", "too_small"], A2) == "too_small" and CP.agree([], A2) == "unreadable" and CP.agree(["a", "unreadable"], A2) == "unreadable"


# ----------------------------------------------------------------------------- Amendment 3

A3 = {**A2, "slot_closed_max": 0.1}


def test_gamma_inputs_sees_the_head_off_where_gamma_reads():
    """The hook finding of Amendment 3: on a real ComponentModel, the input cache Γ reads holds the mean in o_proj's slot, later
    modules move, earlier ones do not, and Γ's gates under `ablated` differ from those without it."""
    model, net = _tiny_component_model()
    toks = torch.randint(0, 50, (2, 12), generator=torch.Generator().manual_seed(3))
    means = torch.full((2, 3, 8), 5.0)
    r = CP.gamma_inputs(model, net, toks, (0, 1), means)
    assert r["status"] == "pass" and r["slot_is_mean"] and r["upstream_max"] == 0.0 and r["downstream_min"] > 0.0
    assert {"h.0.attn.q_proj", "h.1.attn.o_proj"} <= set(r["moved"]) and "h.0.attn.o_proj" not in r["moved"]
    ci = lambda: model.calc_causal_importances(model(toks, cache_type="input").cache, sampling="continuous").lower_leaky
    on = ci()
    with CP.ablated(net, [(0, 1)], means):
        off = ci()
    assert torch.equal(on["h.0.attn.q_proj"], off["h.0.attn.q_proj"]) and not torch.equal(on["h.1.attn.q_proj"], off["h.1.attn.q_proj"])


def test_slot_reads_threshold_alive_and_a_zero_vector():
    V = torch.zeros(6, 5)  # 3 heads of 2 dims, 5 subcomponents
    V[0, 0] = 1.0  # all in head 0
    V[2, 1], V[4, 1] = 1.0, 1.0  # half in head 1, half in head 2: a tie counts for both
    V[2, 2], V[4, 2] = 1.0, 2.0  # a fifth in head 1
    V[0, 4] = 1.0  # all in head 0, not alive; subcomponent 3 has norm 0
    alive = torch.tensor([True, True, True, True, False])
    sets = CP.slot_reads(V, 3, alive, 0.5)
    assert [s.tolist() for s in sets] == [[0], [1], [1, 2]]


@pytest.mark.parametrize("n, gate, want", [(0, math.nan, "empty"), (3, math.nan, "unreadable"), (3, 0.03, "closed"), (3, 0.1, "open"), (3, 0.0, "closed"), (3, 0.9, "open")])
def test_slot_class(n, gate, want):
    assert CP.slot_class(n, gate, A3) == want


@pytest.mark.parametrize("labels, lookup, want", [
    (["within"] * 3, ["within"] * 3, "head_acts_through_masks"), (["within", "within", "too_small"], ["within"] * 3, "head_acts_through_masks"),
    (["within", "over", "over"], ["within"] * 3, "head_more_than_target"), (["other_path"] * 3, ["within"] * 3, "still_other_way"),
    (["other_path"] * 3, ["over"] * 3, "still_other_way"), (["within", "other_path", "within"], ["within"] * 3, "in_between"), (["between"] * 3, ["within"] * 3, "in_between"),
    (["between"] * 3, ["over"] * 3, "in_between"), (["within"] * 3, ["over"] * 3, "not_specific"), (["within"] * 3, ["within", "over", "within"], "not_specific"),
    (["within"] * 3, ["other_path"] * 3, "not_specific"), (["over"] * 3, ["between"] * 3, "not_specific"), (["within"] * 3, ["unreadable"] * 3, "unreadable"),
    (["within"] * 3, ["too_small"] * 3, "too_small"), (["too_small", "too_small", "within"], ["within"] * 3, "too_small"), (["unreadable", "within", "within"], ["within"] * 3, "unreadable"), ([], ["within"] * 3, "unreadable"),
])
def test_gamma_reading(labels, lookup, want):
    assert CP.gamma_reading(labels, lookup, [0.01] * len(labels), A3) == want


def test_gamma_reading_floor_reads_the_size_and_fails_on_a_tie_and_a_nan():
    for bad in (0.15, -0.15, -0.5, math.nan):
        assert CP.gamma_reading(["within"] * 3, ["within"] * 3, [0.0, bad, 0.0], A3) == "floor_high"
    assert CP.gamma_reading(["within"] * 3, ["within"] * 3, [0.0, 0.149, -0.149], A3) == "head_acts_through_masks"
    assert CP.gamma_reading(["within", "within", "too_small"], ["within"] * 3, [0.0, 0.0, math.nan], A3) == "floor_high"  # as filed in Amendment 2, unreachable at a gain of 13.8


@pytest.mark.parametrize("gain, m, t, want", [
    (13.8, 0.55, 0.56, "within"), (13.8, 0.56 + 0.092, 0.56, "within"), (13.8, 0.75, 0.56, "over"), (13.8, 1.2, 0.56, "over"), (13.8, 0.10, 0.56, "other_path"), (13.8, 0.25, 0.56, "between"), (13.8, 0.4, 0.56, "between"),
    (13.8, -0.3, 0.56, "other_path"), (1.0, 0.5, 0.56, "within"), (0.99, 0.5, 0.56, "too_small"), (0.0, math.nan, 0.56, "too_small"), (-2.0, 0.5, 0.56, "too_small"), (math.nan, 0.5, 0.56, "unreadable"),
    (13.8, math.nan, 0.56, "unreadable"), (13.8, 0.5, math.nan, "unreadable"), (6.0, 0.005, 0.0, "within"), (6.0, -0.2, 0.0, "other_path"), (6.0, 0.2, 0.0, "over"),
])
def test_masked_class_on_amendment_3s_cases(gain, m, t, want):
    assert CP.masked_class(gain, m, t, A3) == want


# ----------------------------------------------------------------------------- Amendment 4

A4 = {"p_b_min": 3.15e-3, "target_fall_min": 1.16, "lean_min": 1.58, "masked_fall_frac": 0.25, "min_gain": 1.0, "align_min": 0.2, "chance_factor": 3.0,
      "lookup_align_frac": 0.5, "close_min": 0.5, "random_margin": 0.24, "lookup_close_max": 0.2, "gap_min": 0.24}


def test_second_a_holds_on_cue_windows_and_catches_a_bad_layout():
    w, b, p = CP.cue_windows(rows(), 7, KEYS, BS, HALF, 40, 0)
    assert CP.second_a_holds(w.tokens, p, b, HALF)
    t = w.tokens.clone()
    t[0, HALF - 2] = b[0]  # B at the reference position
    assert not CP.second_a_holds(t, p, b, HALF)
    t = w.tokens.clone()
    t[1, HALF - 2] = t[1, HALF - 1]  # A at the reference position
    assert not CP.second_a_holds(t, p, b, HALF)
    t = w.tokens.clone()
    t[2, HALF - 1] = 9  # not the second A
    assert not CP.second_a_holds(t, p, b, HALF)
    assert not CP.second_a_holds(w.tokens, p + 1, b, HALF)


def test_pair_broken_holds_on_broken_pair_and_catches_a_second_edit():
    w, _, p = CP.cue_windows(rows(), 7, KEYS, BS, HALF, 40, 0)
    b = CP.broken_pair(w.tokens, p, KEYS, HALF, 0)
    assert CP.pair_broken_holds(w.tokens, b, p, HALF)
    assert not CP.pair_broken_holds(w.tokens, w.tokens, p, HALF)  # nothing broken
    x = b.clone()
    x[0, HALF - 1] = 9  # the second A moved
    assert not CP.pair_broken_holds(w.tokens, x, p, HALF)
    x = b.clone()
    x[1, HALF + 3] = 9  # another position edited
    assert not CP.pair_broken_holds(w.tokens, x, p, HALF)
    x = b.clone()
    i = torch.arange(40)
    x[i, p] = w.tokens[:, HALF - 1]
    x[0, p[0]] = w.tokens[0, HALF - 1] + 1
    assert not CP.pair_broken_holds(w.tokens, x, p, HALF)  # first A still there on most windows


def _alive_gates() -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Two matrices named out of sorted order, 3 and 2 alive of 4 and 3 subcomponents, two batches of 2 and 1 windows, 2 positions."""
    alive = {"z": torch.tensor([True, False, True, True]), "b": torch.tensor([False, True, True])}
    g = torch.arange(3 * 2 * 7, dtype=torch.float32).view(3, 2, 7)
    make = lambda sl: {"b": g[sl, :, :3], "z": g[sl, :, 3:]}
    return alive, [make(slice(0, 2)), make(slice(2, 3))]


def test_gate_means_order_and_windows():
    alive, gates = _alive_gates()
    # position 1: b's columns 1, 2 then z's 0, 2, 3 (columns 3, 5, 6 of g); window values at (w, 1, c) = 7 (2 w + 1) + c, mean w = 1
    want = [21.0 + c for c in (1, 2, 3, 5, 6)]
    assert CP.gate_means(gates, alive, 1).tolist() == want
    assert CP.gate_means(gates, alive, 1).dtype == torch.float64


@pytest.mark.parametrize("a, want", [
    ([4.0, 1.0, 3.0], [0]),  # 4 of 8 reaches half exactly
    ([3.0, 1.0, 4.0, -9.0], [2]),  # negative mass does not count; 4 of 8 is exactly half
    ([1.0, 1.0, 1.0, 1.0], [0, 1]),  # ties in index order, half reached exactly
    ([1.0, 1.0, 1.0], [0, 1]),  # 2 of 3 is the first to hold half
    ([0.0, -1.0], []), ([], []), ([math.nan, 1.0], []), ([math.inf, 1.0], []),
])
def test_s_old_edges(a, want):
    assert CP.s_old(torch.tensor(a, dtype=torch.float64)).tolist() == want


def test_cosine_edges():
    x = torch.tensor([1.0, 0.0])
    assert CP.cosine(x, x) == pytest.approx(1.0) and CP.cosine(x, -x) == pytest.approx(-1.0) and CP.cosine(x, torch.tensor([0.0, 2.0])) == 0.0
    assert math.isnan(CP.cosine(x, torch.zeros(2))) and math.isnan(CP.cosine(torch.empty(0), torch.empty(0))) and math.isnan(CP.cosine(x, torch.tensor([math.nan, 1.0])))
    d = torch.randn(4000, generator=torch.Generator().manual_seed(0))
    mean, sd = CP.chance_cosine(d, d.clone(), 20, 1)
    assert abs(mean) < 0.03 and 0 < sd < 0.05  # permuted, a vector's cosine with itself is near 0
    assert CP.chance_cosine(d, d, 20, 1) == (mean, sd)  # seeded


def test_closing_pool_excludes_old_and_falls_back():
    delta = torch.tensor([1.0, 2.0, -1.0, 0.0, 3.0, 4.0])
    old = torch.tensor([0, 1])
    pool, fell = CP.closing_pool(delta, old)
    assert pool.tolist() == [4, 5] and not fell  # exactly k others opened: enough; 0.0 is not opened
    pool, fell = CP.closing_pool(delta, torch.tensor([0, 1, 4]))  # one other opened, k = 3: every entry outside old
    assert pool.tolist() == [2, 3, 5] and fell
    pool, fell = CP.closing_pool(delta, torch.tensor([0, 1, 4, 5]))  # the mask opened exactly S_old: its reference is not S_old itself
    assert not set(pool.tolist()) & {0, 1, 4, 5} and fell
    pool, fell = CP.closing_pool(torch.zeros(4), torch.tensor([0]))  # nothing opened at all
    assert pool.tolist() == [1, 2, 3] and fell


def test_random_sets_and_largest():
    pool = torch.tensor([10, 11, 12, 13, 14])
    draws = CP.random_sets(pool, 2, 5, 0)
    assert len(draws) == 5 and all(len(set(d.tolist())) == 2 and set(d.tolist()) <= set(pool.tolist()) for d in draws)
    assert [d.tolist() for d in CP.random_sets(pool, 2, 5, 0)] == [d.tolist() for d in draws]  # seeded
    for empty_k, p in ((0, pool), (6, pool), (1, torch.empty(0, dtype=torch.long))):  # an empty pool or k = 0 never gets here: `mask_test` reads k first
        with pytest.raises(AssertionError):
            CP.random_sets(p, empty_k, 5, 0)
    assert CP.largest(torch.tensor([1.0, 3.0, 3.0, 0.0]), 2).tolist() == [1, 2]
    assert CP.largest(torch.tensor([1.0, 3.0, 3.0, 0.0]), 2, torch.tensor([1])).tolist() == [2, 0]  # outside S_old
    assert CP.largest(torch.tensor([1.0, 3.0]), 2, torch.tensor([1])).tolist() == [0]  # as many as there are


def test_closed_sets_only_the_named_entries_at_every_position():
    alive, g1 = _alive_gates()
    ref = [{k: v - 100.0 for k, v in b.items()} for b in g1]
    keep = [{k: v.clone() for k, v in b.items()} for b in g1]
    # alive vector: b1 b2 z0 z2 z3; close b2 (index 1) and z3 (index 4)
    out = CP.closed(g1, ref, torch.tensor([1, 4]), alive)
    for b, r, k in zip(out, ref, keep, strict=True):
        assert torch.equal(b["b"][..., 2], r["b"][..., 2]) and torch.equal(b["z"][..., 3], r["z"][..., 3])
        assert torch.equal(b["b"][..., :2], k["b"][..., :2]) and torch.equal(b["z"][..., :3], k["z"][..., :3])
    assert all(torch.equal(g1[i][k], keep[i][k]) for i in (0, 1) for k in ("b", "z"))  # the input is unchanged
    none = CP.closed(g1, ref, torch.empty(0, dtype=torch.long), alive)
    assert all(torch.equal(none[i][k], g1[i][k]) for i in (0, 1) for k in ("b", "z"))
    every = CP.closed(g1, ref, torch.arange(5), alive)
    assert torch.equal(CP.gate_means(every, alive, 0), CP.gate_means(ref, alive, 0))


def test_gates_are_per_position_and_closing_all_reads_the_reference_mask():
    """What the draft asserted of `Reads`: route_cells' gates are (B, T, C), its read position does not touch them, masked_lo_windows
    takes the list, reads another position, and a closing of every subcomponent is the reference mask's forward."""
    from commute import local_faith as LF

    model, _ = _tiny_component_model()
    toks = torch.randint(0, 50, (5, 13), generator=torch.Generator().manual_seed(4))
    a = LF.route_cells(model, toks, 6, LAST, 2, 12)["gates"]
    b = LF.route_cells(model, toks, 5, LAST, 2, 12)["gates"]
    assert all(torch.equal(x[k], y[k]) for x, y in zip(a, b, strict=True) for k in x)
    assert a[0]["h.0.attn.o_proj"].shape == (2, 12, 4) and len(a) == 3
    alive = {k: torch.ones(4, dtype=torch.bool) for k in a[0]}
    half = [{k: v * 0.5 for k, v in g.items()} for g in a]
    want = LF.masked_lo_windows(model, toks, 5, LAST, a, 2, 12)
    assert torch.equal(LF.masked_lo_windows(model, toks, 5, LAST, CP.closed(half, a, torch.arange(4 * len(alive)), alive), 2, 12), want)
    assert not torch.equal(LF.masked_lo_windows(model, toks, 6, LAST, a, 2, 12), want)


@pytest.mark.parametrize("p, fall, lean, m_fall, m_lean, want", [
    (0.1, 3.0, 4.0, 1.0, 3.0, "pass"), (3.15e-3, 1.16, 1.58, 0.29, 3.58, "pass"), (3.14e-3, 3.0, 4.0, 1.0, 3.0, "no_copying"),
    (0.1, 1.0, 4.0, 1.0, 3.0, "head_not_carrying"), (0.1, -1.0, 4.0, 1.0, 3.0, "head_not_carrying"), (0.1, 0.0, 4.0, 1.0, 3.0, "head_not_carrying"),
    (0.1, 3.0, 1.57, 1.0, 1.5, "no_pair_dependence"), (0.1, 3.0, -2.0, 1.0, -2.0, "no_pair_dependence"),  # W0 does not lose the copy
    (0.1, 3.0, 4.0, 1.0, 1.0, "pass"), (0.1, 3.0, 4.0, 1.0, 0.99, "mask_no_lean"),  # a quarter of the target's lean, a tie passes
    (0.1, 3.0, 1.7, 1.0, -0.25, "mask_no_lean"), (0.1, 3.0, 1.7, 1.0, 0.05, "mask_no_lean"), (0.1, 3.0, 3.0, 1.0, 5.2, "pass"),  # the re-check's R2 table
    (0.1, 4.0, 4.0, 1.0, 3.0, "pass"), (0.1, 4.0, 4.0, 0.99, 3.0, "mask_drops_head"), (0.1, 3.0, 4.0, -1.0, 3.0, "mask_drops_head"),
    (math.nan, 3.0, 4.0, 1.0, 3.0, "unreadable"), (0.1, 3.0, math.inf, 1.0, 3.0, "unreadable"),
])
def test_second_a_control(p, fall, lean, m_fall, m_lean, want):
    assert CP.second_a_control(p, fall, lean, m_fall, m_lean, A4) == want


@pytest.mark.parametrize("gain, share, median, want", [
    (3.0, 0.5, 0.25, "carries"), (3.0, 0.5, 0.27, "not_carrying"), (3.0, 0.49, 0.0, "not_carrying"), (3.0, 0.9, 0.7, "not_carrying"), (3.0, math.nan, 0.0, "not_carrying"), (3.0, 0.9, math.nan, "not_carrying"), (3.0, -0.2, -0.9, "not_carrying"),
    (1.0, 0.9, 0.0, "carries"), (0.99, 0.9, 0.0, "no_gate_gain"), (0.0, math.nan, 0.0, "no_gate_gain"), (-1.0, math.nan, 0.0, "no_gate_gain"), (math.nan, 0.9, 0.0, "no_gate_gain"),
])
def test_old_carries(gain, share, median, want):
    assert CP.old_carries(gain, share, median, A4) == want


def test_recheck_r1_a_lean_mostly_through_the_tokens():
    """The re-check's case: masked lean 3.0, 2.2 through the tokens, 0.8 through the gates, S_old holding all of it. Over the lean the share was
    0.27 and read old_pieces_not_carrying; on one set of tokens it is 1, and a gate gain of 0.8 is unscored, its own verdict."""
    intact, gb_on_intact, gb_on_broken = 3.0, 2.2, 0.0
    assert CP.share_lost(intact - gb_on_broken, gb_on_intact - gb_on_broken) == pytest.approx(0.8 / 3)  # the old denominator's ceiling
    assert CP.share_lost(intact - gb_on_intact, 0.0) == 1.0
    status = CP.old_carries(intact - gb_on_intact, 1.0, 0.0, A4)
    strong, look = rec(cos=0.5, share=0.8), rec(6.0, 0.02, 0.0, 0.02)
    assert status == "no_gate_gain" and mt([strong] * 3, [look] * 3, carries=status)["verdict"] == "w0_gate_gain_too_small"
    status = CP.old_carries(1.2, 1.0, 0.0, A4)  # 1.8 through the tokens: the same set, a gain the project scores
    assert status == "carries" and mt([strong] * 3, [look] * 3, carries=status)["verdict"] == "shared_pieces"


def rec(M: float = 13.8, cos: float = 0.4, chance: float = 0.01, share: float = 0.7, random_median: float = 0.1, largest_k_share: float = 0.8) -> dict:
    return {"M": M, "cos": cos, "chance": chance, "share": share, "random_median": random_median, "largest_k_share": largest_k_share}


GOOD_C, GOOD_L = rec(), rec(6.0, 0.05, 0.0, 0.05, 0.1)


def mt(copy: list[dict], lookup: list[dict], control: str = "pass", k: int = 100, n: int = 9881, carries: str = "carries") -> dict:
    return CP.mask_test(control, carries, k, n, copy, lookup, A4)


@pytest.mark.parametrize("p1, p2, p3", [(a, b, c) for a in (True, False) for b in (True, False) for c in (True, False)])
def test_mask_test_every_combination_of_the_predictions(p1, p2, p3):
    """All three seeds alike. P2 implies P3, so the two combinations with P2 and not P3 cannot be built: 0.5 - 0.2 > 0.24."""
    if p2 and not p3:
        assert A4["close_min"] - A4["lookup_close_max"] >= A4["gap_min"]
        return
    share, look = (0.7, 0.05) if p2 else (0.45, 0.15) if p3 else (0.15, 0.10)
    v = mt([rec(cos=0.4 if p1 else 0.1, share=share)] * 3, [rec(6.0, 0.01, 0.0, look, 0.1)] * 3)
    assert (all(v["P1"]), all(v["P2"]), all(v["P3"])) == (p1, p2, p3)
    want = "shared_pieces" if p1 and p2 and p3 else "not_shared" if not p1 and not p2 and not p3 else "in_between"
    assert v["verdict"] == want  # P1 and P2 miss, P3 holds: share 0.45 beats random by 0.35, so the null is refuted: in_between
    assert mt([rec(cos=0.4 if p1 else 0.1, share=share)] * 3, [rec(6.0, 0.01, 0.0, look, 0.1)] * 3, carries="not_carrying")["verdict"] == "old_pieces_not_carrying"


@pytest.mark.parametrize("cos", [-0.4, 0.1, 0.2, 0.5])
@pytest.mark.parametrize("share", [-0.3, 0.0, 0.1, 0.3, 0.49, 0.5, 0.9])
@pytest.mark.parametrize("median", [-0.5, 0.0, 0.02, 0.1, 0.6])
@pytest.mark.parametrize("look", [(0.01, -0.05), (0.3, 0.25), (0.05, 0.2)])
@pytest.mark.parametrize("big", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("carries", ["carries", "not_carrying", "no_gate_gain"])
def test_mask_test_grid_reaches_the_filed_branch(cos, share, median, look, big, carries):
    """Exactly one verdict everywhere on a grid, and the one the rule's words give, written out here apart from the code."""
    c, l = rec(cos=cos, share=share, random_median=median, largest_k_share=big), rec(6.0, look[0], 0.0, look[1], 0.1)
    v = mt([c] * 3, [l] * 3, carries=carries)["verdict"]
    shared = cos >= 0.2 and look[0] < 0.5 * cos and share >= 0.5 and share - median >= 0.24 and look[1] <= 0.2
    null = (share - median < 0.24 and cos < 0.2) or share <= median
    want = "w0_gate_gain_too_small" if carries == "no_gate_gain" else "old_pieces_not_carrying" if carries == "not_carrying" else "shared_pieces" if shared else ("not_shared" if big >= 0.5 else "k_too_small") if null else "in_between"
    assert v == want


@pytest.mark.parametrize("copy, lookup, want", [
    # the review's H3 table: refutes the null, so never not_shared
    ([rec(cos=0.19, chance=0.005, share=0.49, random_median=0.02)] * 3, [rec(6.0, 0.01, 0.0, -0.05)] * 3, "in_between"),
    ([rec(cos=0.19, chance=0.005, share=0.30, random_median=0.02)] * 3, [rec(6.0, 0.01, 0.0, -0.05)] * 3, "in_between"),
    # a k too small for any closing to matter: not a reading of the mechanism
    ([rec(cos=0.5, chance=0.005, share=0.01, random_median=0.02, largest_k_share=0.05)] * 3, [rec(6.0, 0.01, 0.0, 0.0)] * 3, "k_too_small"),
    ([rec(cos=0.1, share=0.01, random_median=0.02, largest_k_share=0.49)] * 3, [GOOD_L] * 3, "k_too_small"),
    ([rec(cos=0.5, chance=0.005, share=0.01, random_median=0.02, largest_k_share=0.5)] * 3, [rec(6.0, 0.01, 0.0, 0.0)] * 3, "not_shared"),  # a tie has power
    ([rec(cos=0.1, share=0.01, random_median=0.02, largest_k_share=0.05), rec(cos=0.1, share=0.01, random_median=0.02), rec(cos=0.1, share=0.01, random_median=0.02)], [GOOD_L] * 3, "k_too_small"),  # one seed short
])
def test_mask_test_h3_cases(copy, lookup, want):
    assert mt(copy, lookup)["verdict"] == want


@pytest.mark.parametrize("copy, lookup, control, k, want", [
    ([GOOD_C] * 3, [GOOD_L] * 3, "pass", 100, "shared_pieces"),
    ([GOOD_C] * 3, [GOOD_L] * 3, "no_copying", 100, "control_failed_target"), ([GOOD_C] * 3, [GOOD_L] * 3, "head_not_carrying", 100, "control_failed_target"), ([GOOD_C] * 3, [GOOD_L] * 3, "no_pair_dependence", 100, "control_failed_target"),
    ([GOOD_C] * 3, [GOOD_L] * 3, "mask_no_lean", 100, "control_failed_mask"), ([GOOD_C] * 3, [GOOD_L] * 3, "mask_drops_head", 100, "control_failed_mask"), ([GOOD_C] * 3, [GOOD_L] * 3, "unreadable", 100, "unreadable"),
    ([GOOD_C] * 3, [GOOD_L] * 3, "pass", 0, "unreadable"), ([GOOD_C] * 3, [GOOD_L] * 3, "pass", 4941, "unreadable"), ([GOOD_C] * 3, [GOOD_L] * 3, "pass", 4940, "shared_pieces"),
    ([], [], "pass", 100, "unreadable"), ([GOOD_C] * 3, [GOOD_L] * 2, "pass", 100, "unreadable"),
    ([rec(M=0.0, share=math.nan)] * 3, [GOOD_L] * 3, "pass", 100, "too_small"), ([rec(M=-2.0, share=math.nan)] * 3, [GOOD_L] * 3, "pass", 100, "too_small"),  # a zero denominator
    ([GOOD_C] * 3, [rec(M=0.99)] * 3, "pass", 100, "too_small"), ([rec(M=1.0)] * 3, [GOOD_L] * 3, "pass", 100, "shared_pieces"),
    ([rec(M=math.nan)] * 3, [GOOD_L] * 3, "pass", 100, "unreadable"), ([rec(share=math.nan)] * 3, [GOOD_L] * 3, "pass", 100, "unreadable"),
    ([rec(cos=math.nan)] * 3, [GOOD_L] * 3, "pass", 100, "unreadable"), ([GOOD_C] * 3, [rec(6.0, chance=math.nan)] * 3, "pass", 100, "unreadable"), ([rec(random_median=math.inf)] * 3, [GOOD_L] * 3, "pass", 100, "unreadable"),
    ([rec(largest_k_share=math.nan)] * 3, [GOOD_L] * 3, "pass", 100, "unreadable"),
    # S_old disjoint from what the mask opened: closing changes nothing, share 0, at or under a positive or zero random median
    ([rec(share=0.0, random_median=0.1)] * 3, [GOOD_L] * 3, "pass", 100, "not_shared"), ([rec(share=0.0, random_median=0.0)] * 3, [GOOD_L] * 3, "pass", 100, "not_shared"),
    ([rec(cos=0.05, share=0.0, random_median=-0.1)] * 3, [GOOD_L] * 3, "pass", 100, "not_shared"),  # above a negative median, but no more than random and no more than chance
    # a negative closing share is a miss on the copying map; on the lookup it holds the bar
    ([rec(cos=0.05, share=-0.3, random_median=0.1)] * 3, [GOOD_L] * 3, "pass", 100, "not_shared"), ([rec(share=-0.3, random_median=-0.5)] * 3, [GOOD_L] * 3, "pass", 100, "in_between"),
    ([GOOD_C] * 3, [rec(6.0, 0.05, 0.0, -0.4, 0.1)] * 3, "pass", 100, "shared_pieces"),
    # strong on the copying map, the lookup uses the same subcomponents: not the null
    ([rec(cos=0.5, share=0.9)] * 3, [rec(6.0, 0.3, 0.0, 0.25, 0.1)] * 3, "pass", 100, "in_between"),
    # ties at the bars: 0.2, three times chance, half the copying map's (not under), 0.5, 0.2, the median
    ([rec(cos=0.2)] * 3, [rec(6.0, 0.05, 0.0, 0.05, 0.1)] * 3, "pass", 100, "shared_pieces"), ([rec(cos=0.375, chance=0.125)] * 3, [rec(6.0, 0.05, 0.0, 0.05, 0.1)] * 3, "pass", 100, "shared_pieces"),
    ([rec(cos=0.4)] * 3, [rec(6.0, 0.2, 0.0, 0.05, 0.1)] * 3, "pass", 100, "in_between"), ([rec(share=0.5, random_median=0.1)] * 3, [rec(6.0, 0.05, 0.0, 0.2, 0.1)] * 3, "pass", 100, "shared_pieces"),
    ([rec(cos=0.05, share=0.1, random_median=0.1)] * 3, [GOOD_L] * 3, "pass", 100, "not_shared"), ([rec(cos=0.4, share=0.5, random_median=0.3)] * 3, [GOOD_L] * 3, "pass", 100, "in_between"),
    # negative chance: the 0.2 bar decides; a negative cosine misses
    ([rec(cos=0.2, chance=-0.5)] * 3, [GOOD_L] * 3, "pass", 100, "shared_pieces"), ([rec(cos=-0.4, share=0.1)] * 3, [GOOD_L] * 3, "pass", 100, "not_shared"),
    # mixed seeds
    ([GOOD_C, GOOD_C, rec(cos=0.05, share=0.1)], [GOOD_L] * 3, "pass", 100, "in_between"),
    ([rec(cos=0.05, share=0.1), rec(cos=0.05, share=0.1), GOOD_C], [GOOD_L] * 3, "pass", 100, "in_between"),
    ([rec(share=0.05), rec(share=0.05), GOOD_C], [GOOD_L] * 3, "pass", 100, "in_between"),  # at or under the median on two seeds only
    ([rec(share=0.05)] * 3, [GOOD_L] * 3, "pass", 100, "not_shared"),  # P1 holds, but at or under the median on every seed
])
def test_mask_test_branches(copy, lookup, control, k, want):
    assert mt(copy, lookup, control, k)["verdict"] == want


def test_mask_test_old_pieces_not_carrying_blocks_both_and_reports_the_rest():
    v = mt([GOOD_C] * 3, [GOOD_L] * 3, carries="not_carrying")
    assert v["verdict"] == "old_pieces_not_carrying" and all(v["P1"]) and all(v["P2"]) and all(v["P3"])  # would be shared_pieces
    assert mt([rec(share=0.0)] * 3, [GOOD_L] * 3, carries="not_carrying")["verdict"] == "old_pieces_not_carrying"  # would be not_shared
    assert mt([GOOD_C] * 3, [GOOD_L] * 3, carries="no_gate_gain")["verdict"] == "w0_gate_gain_too_small"  # would be shared_pieces
    assert mt([rec(share=0.0)] * 3, [GOOD_L] * 3, carries="no_gate_gain")["verdict"] == "w0_gate_gain_too_small"  # would be not_shared
    assert mt([GOOD_C] * 3, [rec(M=0.5)] * 3, carries="not_carrying")["verdict"] == "too_small"
    assert mt([GOOD_C] * 3, [GOOD_L] * 3, "no_copying", carries="not_carrying")["verdict"] == "control_failed_target"


def test_mask_test_reports_the_predictions_per_seed():
    v = mt([GOOD_C, rec(cos=0.05), rec(share=0.3)], [GOOD_L] * 3)
    assert v["P1"] == [True, False, True] and v["P2"] == [True, True, False] and v["P3"] == [True, True, True] and v["verdict"] == "in_between"
    assert mt([GOOD_C] * 3, [GOOD_L] * 3, "no_copying")["P1"] == []
    assert set(mt([], [], "no_copying")) == set(mt([GOOD_C] * 3, [GOOD_L] * 3))  # the same shape on a stop and a finished run


def test_end_to_end_on_a_copying_shaped_case():
    """The vectors through the rules: a state opening what `a` holds aligns; one opening elsewhere does not, and its S_old closing is a no-op."""
    a = torch.zeros(100, dtype=torch.float64)
    a[:5] = 1.0
    old = CP.s_old(a)
    assert old.tolist() == [0, 1, 2]
    on, off = torch.zeros(100, dtype=torch.float64), torch.zeros(100, dtype=torch.float64)
    on[:5], off[50:55] = 0.5, 0.5
    assert CP.cosine(on, a) == pytest.approx(1.0) and CP.cosine(off, a) == 0.0
    assert not set(CP.closing_pool(on, old)[0].tolist()) & set(old.tolist())
    alive = {"m": torch.ones(100, dtype=torch.bool)}
    g1, g0 = [{"m": off[None, None].float().repeat(1, 3, 1)}], [{"m": torch.zeros(1, 3, 100)}]
    assert torch.equal(CP.closed(g1, g0, old, alive)[0]["m"], g1[0]["m"])  # disjoint: nothing changes, so the share is 0


# ----------------------------------------------------------------------------- Amendment 5

A5 = {"masked_fall_frac": 0.25, "carrier_min": 1.0, "carrier_sds": 2.0, "fall_floor": 0.5, "fraction_gap": 0.24, "token_share": 0.5}
LAB = (2, 4)
ALL_HEADS = [(b, h) for b in range(4) for h in range(6)]


def sweep_rows(**per: dict[str, float]) -> dict:
    """24 heads: the labelled one as Amendment 4 read it, the rest with no fall; `per` overrides, keyed h<block>_<head>."""
    rows = {h: {"T": 0.1, "M": 0.05, "Mb": 0.0, "excess_sd": 0.05} for h in ALL_HEADS}
    rows[LAB] = {"T": 4.82, "M": 0.92, "Mb": 0.1, "excess_sd": 0.2}
    for k, v in per.items():
        h = (int(k[1]), int(k[3]))
        rows[h] = {**rows[h], **v}
    return rows


def sweep(rows: dict, lean: float = 4.34) -> str:
    return CP.sweep_reading(rows, LAB, 24, lean, A5)["verdict"]


@pytest.mark.parametrize("fall, target, want", [
    (4.8, 4.82, "pass"), (1.25, 5.0, "pass"), (1.249, 5.0, "ablation_does_not_transfer"), (0.0, 4.82, "ablation_does_not_transfer"), (-3.0, 4.82, "ablation_does_not_transfer"),
    (9.0, 4.82, "pass"), (math.nan, 4.82, "unreadable"), (4.8, math.inf, "unreadable"), (4.8, 0.0, "unreadable"), (4.8, -1.0, "unreadable"),
])
def test_all_ones_control(fall, target, want):
    assert CP.all_ones_control(fall, target, A5) == want


def test_sweep_each_verdict_as_named():
    assert sweep(sweep_rows(h2_3={"T": 0.3, "M": 1.4, "Mb": 0.1})) == "rerouted"
    assert sweep(sweep_rows(h2_3={"T": 2.0, "M": 0.4})) == "thinned"  # 0.2 against the labelled head's 0.19
    assert sweep(sweep_rows(h2_3={"T": 2.0, "M": 1.9})) == "labelled_head_singled_out"  # the mask takes (2,4) and leaves (2,3)
    assert sweep(sweep_rows(h2_3={"T": 2.0, "M": 1.9}, h3_5={"T": 2.0, "M": 0.4})) == "mixed"  # one kept, one thinned
    assert sweep(sweep_rows(h2_3={"T": 2.0, "M": -1.0})) == "mixed"  # head off raises it: under the labelled head's by more than the gap
    assert sweep(sweep_rows()) == "no_comparison_head"  # the target's rest spread under the floor


def test_sweep_where_no_head_falls():
    flat = {h: {"T": 0.0, "M": 0.0, "Mb": 0.0, "excess_sd": 0.0} for h in ALL_HEADS}
    assert sweep(flat) == "unreadable"  # the target's labelled head under the floor: the re-read stops this first
    keep = {**flat, LAB: {"T": 4.82, "M": 0.0, "Mb": 0.0, "excess_sd": 0.0}}
    assert sweep(keep) == "no_comparison_head"
    assert sweep({**keep, (2, 3): {"T": 2.0, "M": 0.0, "Mb": 0.0, "excess_sd": 0.0}}) == "thinned"  # no masked fall anywhere: every fraction is 0


def test_sweep_carrier_needs_every_bar():
    assert sweep(sweep_rows(h2_3={"T": 0.25, "M": 1.25, "Mb": 0.25})) == "rerouted"  # both ties at 1.0 count
    assert sweep(sweep_rows(h2_3={"T": 0.3, "M": 1.29, "Mb": 0.1})) == "no_comparison_head"  # excess under the bar
    assert sweep(sweep_rows(h2_3={"T": 0.3, "M": 1.4, "Mb": 0.41})) == "no_comparison_head"  # a general drop: the same fall with the pair broken
    assert sweep(sweep_rows(h2_3={"T": 0.25, "M": 1.5, "excess_sd": 0.63})) == "no_comparison_head"  # noise: excess 1.25 under 2 sd
    assert sweep(sweep_rows(h2_3={"T": 0.25, "M": 1.5, "excess_sd": 0.625})) == "rerouted"  # a tie at 2 sd counts
    assert sweep(sweep_rows(h3_5={"T": 2.0, "M": 3.5, "Mb": 0.0})) == "rerouted"  # a head with a target fall of its own, carrying more
    assert CP.sweep_reading(sweep_rows(h2_3={"T": 0.3, "M": 1.4}, h0_0={"T": 0.0, "M": 1.0}), LAB, 24, 4.34, A5)["carriers"] == ["0,0", "2,3"]


def test_sweep_carrier_needs_the_masked_forward_to_lean_on_the_head():
    # review finding 1: a head that is off in the target but not leaned on by the masked forward is no carrier
    assert sweep(sweep_rows(h2_3={"T": -3.0, "M": -1.5, "Mb": -3.0})) == "no_comparison_head"  # M under carrier_min; a negative T is no denominator
    assert sweep(sweep_rows(h2_3={"T": -1.2, "M": 0.0, "Mb": -1.1})) == "no_comparison_head"
    assert sweep(sweep_rows(h2_3={"T": 0.0, "M": 1.0, "Mb": 0.0, "excess_sd": 0.0})) == "rerouted"  # M = carrier_min: a tie counts


def test_sweep_ties_and_denominators():
    lab = 0.92 / 4.82
    assert sweep(sweep_rows(h2_3={"T": 1.0, "M": lab + 0.2399})) == "thinned"
    assert sweep(sweep_rows(h2_3={"T": 1.0, "M": lab + 0.2401})) == "labelled_head_singled_out"
    assert sweep(sweep_rows(h2_3={"T": 2.0, "M": 1.48}, h3_5={"T": 2.0, "M": 0.8})) == "mixed"
    assert sweep(sweep_rows(h2_3={"T": 0.5, "M": 0.05})) == "thinned"  # a tie at the floor has a denominator; fraction 0.1
    assert sweep(sweep_rows(h2_3={"T": 0.49, "M": 0.05})) == "no_comparison_head"
    assert sweep(sweep_rows(h2_3={"T": -3.0, "M": -3.0})) == "no_comparison_head"  # a negative target fall is no denominator, however the ratio reads


def test_sweep_unreadable_cases():
    r = sweep_rows()
    assert sweep(r, math.nan) == "unreadable" and sweep(r, 0.0) == "unreadable" and sweep(r, -1.0) == "unreadable" and sweep(r, 0.99) == "unreadable"
    assert sweep(r, 1.0) == "no_comparison_head"
    assert sweep(sweep_rows(h2_3={"M": math.nan})) == "unreadable" and sweep(sweep_rows(h0_0={"excess_sd": math.nan})) == "unreadable" and sweep(sweep_rows(h3_5={"T": math.inf})) == "unreadable"
    assert sweep({h: v for h, v in r.items() if h != (0, 0)}) == "unreadable" and sweep({h: v for h, v in r.items() if h != LAB}) == "unreadable"
    assert sweep(sweep_rows(h2_4={"T": 0.49})) == "unreadable"
    assert set(CP.stopped_reading("x")) == set(CP.sweep_reading(r, LAB, 24, 4.34, A5))  # the same shape on a stop and a finished sweep


def test_sweep_total_fall_over_the_masked_lean_decides_nothing():
    big = sweep_rows(**{f"h{b}_{h}": {"T": 2.0, "M": 0.4} for b, h in ALL_HEADS if (b, h) != LAB})
    v = CP.sweep_reading(big, LAB, 24, 4.34, A5)
    assert v["verdict"] == "thinned" and v["sum_falls_over_lean"] and not v["falls_over_lean"]  # 23 x 0.4 + 0.92 > 4.34: falls need not add up to the lean
    v = CP.sweep_reading(sweep_rows(h2_3={"T": 0.3, "M": 5.0, "Mb": 0.1}), LAB, 24, 4.34, A5)
    assert v["verdict"] == "rerouted" and v["falls_over_lean"] == ["2,3"]  # one head over the whole lean: read and flagged


def test_sweep_ranking_is_beside():
    assert CP.sweep_reading(sweep_rows(), LAB, 24, 4.34, A5)["ranking_holds"]
    v = CP.sweep_reading(sweep_rows(h3_5={"T": 4.82, "M": 4.0}), LAB, 24, 4.34, A5)  # a tie fails
    assert not v["ranking_holds"] and v["verdict"] == "labelled_head_singled_out"


@pytest.mark.parametrize("t", [-3.0, -1.2, 0.0, 0.3, 0.5, 2.0])
@pytest.mark.parametrize("m", [-1.0, 0.0, 0.1, 0.5, 1.0, 1.31, 2.0, 2.5])
@pytest.mark.parametrize("mb", [-2.0, 0.0, 0.31, 1.5])
@pytest.mark.parametrize("sd", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("second", [None, (0.6, 0.12), (2.0, 1.9)])
def test_sweep_grid_reaches_the_filed_branch(t, m, mb, sd, second):
    """Exactly one verdict everywhere on a grid, and the one the rule's words give, written out here apart from the code. One head moves, a second is fixed."""
    rows = sweep_rows(h2_3={"T": t, "M": m, "Mb": mb, "excess_sd": sd})
    if second:
        rows[(1, 1)] = {"T": second[0], "M": second[1], "Mb": 0.0, "excess_sd": 0.0}
    lab = 0.92 / 4.82
    carrier = m >= 1.0 and m - t >= 1.0 and m - t >= 2 * sd and m - mb >= 1.0
    fr = ([m / t] if t >= 0.5 else []) + ([second[1] / second[0]] if second else [])
    want = "rerouted" if carrier else "no_comparison_head" if not fr else "thinned" if all(abs(f - lab) <= 0.24 for f in fr) else "labelled_head_singled_out" if all(f > lab + 0.24 for f in fr) else "mixed"
    assert sweep(rows) == want


@pytest.mark.parametrize("p1, p2, lean, want", [
    (3.0, 3.0, 4.34, "through_tokens"), (1.0, 1.0, 4.34, "through_gates"), (3.0, 1.0, 4.34, "order_dependent"), (1.0, 3.0, 4.34, "order_dependent"),
    (2.17, 3.0, 4.34, "order_dependent"), (2.17, 2.17, 4.34, "order_dependent"), (2.17, 1.0, 4.34, "order_dependent"),  # a tie at the bar: neither side
    (-1.0, -1.0, 4.34, "through_gates"), (6.0, 6.0, 4.34, "through_tokens"),  # outside 0 and 1: the other part has the opposite sign
    (1.0, 1.0, 0.0, "unreadable"), (1.0, 1.0, -2.0, "unreadable"), (1.0, 1.0, 0.99, "unreadable"), (math.nan, 1.0, 4.34, "unreadable"), (1.0, math.inf, 4.34, "unreadable"), (1.0, 1.0, math.nan, "unreadable"),
])
def test_token_reading(p1, p2, lean, want):
    assert CP.token_reading(p1, p2, lean, A5) == want


def test_boot_sd_and_precision_moved():
    d = torch.randn(200, generator=torch.Generator().manual_seed(0)) * 2.0
    sd = CP.boot_sd(d, 2000, 0)
    assert sd == pytest.approx(float(d.double().std()) / math.sqrt(200), rel=0.1) and sd == CP.boot_sd(d, 2000, 0)
    assert CP.boot_sd(torch.ones(200), 100, 0) == 0.0 and math.isnan(CP.boot_sd(torch.empty(0), 100, 0)) and math.isnan(CP.boot_sd(torch.tensor([1.0, math.nan]), 100, 0))
    assert CP.precision_moved(0.92, 1.1, 0.1) and not CP.precision_moved(0.92, 0.95, 0.1) and not CP.precision_moved(1.0, 1.5, 0.5)  # a tie has not moved
    assert CP.precision_moved(0.92, math.nan, 0.1) and CP.precision_moved(0.92, 0.92, math.nan)


def test_reread_diff():
    assert CP.reread_diff({"a": 1.0, "k": 172, "top": [0.4, 0.3]}, {"a": 1.0005, "k": 172, "top": [0.4, 0.3], "extra": 9}) == pytest.approx(5e-4)
    assert CP.reread_diff({"top": [0.4, 0.3]}, {"top": [0.4]}) == math.inf and CP.reread_diff({"a": 1.0}, {}) == math.inf and CP.reread_diff({"a": math.nan}, {"a": math.nan}) == math.inf
    assert CP.reread_diff({"k": 172}, {"k": 173}) == 1.0 and CP.reread_diff({}, {}) == 0.0


def test_all_ones_gates_are_the_all_gates_on_forward_and_the_head_off_stays_out_of_gamma():
    """Amendment 5's control: `Reads.masked` under gates of 1 is `AllGatesOn`, whatever hook is on; the fp32 paths equal the autocast ones on the CPU, where
    autocast does not run; gates read outside a hook do not change under one, so the sweep's head is in the forward alone."""
    from commute import local_faith as LF

    model, net = _tiny_component_model()
    toks = torch.randint(0, 50, (5, 13), generator=torch.Generator().manual_seed(4))
    gates = LF.route_cells(model, toks, 5, LAST, 2, 12)["gates"]
    ones = [{k: torch.ones_like(v) for k, v in b.items()} for b in gates]
    means = torch.full((2, 3, 8), 5.0)
    with CP.ablated(net, [(1, 2)], means):
        via_ones = LF.masked_lo_windows(model, toks, 5, LAST, ones, 2, 12)
        direct = torch.cat([LF.log_odds(net.run(toks[s:s + 2, :12])[0][:, 5], toks[s:s + 2, -1]).double() for s in (0, 2, 4)])
    assert torch.equal(via_ones, direct) and not torch.equal(via_ones, LF.masked_lo_windows(model, toks, 5, LAST, ones, 2, 12))
    assert torch.equal(CP.masked_lo_fp32(model, toks, 5, LAST, gates, 2, 12), LF.masked_lo_windows(model, toks, 5, LAST, gates, 2, 12))
    g32 = CP.gates_fp32(model, toks, 2, 12)
    assert len(g32) == len(gates) and all(torch.equal(x[k], y[k]) for x, y in zip(g32, gates, strict=True) for k in x)


TH6 = {"lean_min": 1.0, "band_low": 0.4, "band_high": 0.6, "order_margin": 1.0, "control_margin": 2.0}


def _overwrite_case() -> tuple[Tensor, Tensor, Tensor]:
    toks = torch.tensor([[9, 7, 300, 8, 7, 5], [9, 7, 301, 8, 7, 5]])
    return toks, torch.tensor([1, 1]), torch.tensor([300, 301])


def test_overwrite_holds_refuses_every_window_that_is_not_as_filed():
    toks, p, b = _overwrite_case()
    over = CP.absent_b(toks, p, b, [300, 301], 0)
    assert CP.overwrite_holds(toks, over, p, b, 5) and over[:, 2].tolist() != b.tolist()
    assert not CP.overwrite_holds(toks, toks, p, b, 5)  # B not replaced
    one = over.clone()
    one[1, 2] = 301
    assert not CP.overwrite_holds(toks, one, p, b, 5)  # one window where B is not replaced
    moved = over.clone()
    moved[0, 3] = 11
    assert not CP.overwrite_holds(toks, moved, p, b, 5)  # something else moved
    left = toks.clone()
    left[0, 3] = 300
    assert not CP.overwrite_holds(left, CP.absent_b(left, p, b, [300, 301], 0), p, b, 5)  # B also elsewhere in the window, so still there
    assert not CP.overwrite_holds(toks, over, torch.tensor([2, 2]), b, 5)  # p is not the pair
    pair = over.clone()
    pair[0, 1] = 12
    assert not CP.overwrite_holds(toks, pair, p, b, 5)  # the pair A -> new token broken
    assert not CP.overwrite_holds(toks[:0], over[:0], p[:0], b[:0], 5)  # no windows
    assert not CP.overwrite_holds(toks[:1], CP.absent_b(toks[:1], p[:1], b[:1], [300], 0), p[:1], b[:1], 5)  # B' = B: a pool of B alone
    assert not CP.overwrite_holds(toks[:1], CP.absent_b(toks[:1], p[:1], b[:1], [300, 300], 0), p[:1], b[:1], 5)  # B' = B: a pool of duplicates
    with pytest.raises(RuntimeError):
        CP.absent_b(toks, p, b, [], 0)  # an empty pool cannot draw; the 200 windows' own targets make one of 178


def _read6(on=-7.3, b=-7.5, ni=-11.6, no=-11.4, lean=4.34) -> dict[str, Any]:
    return CP.overwrite_reading(on, b, ni, no, lean, TH6)


def test_overwrite_reading_verdicts_and_their_parts():
    held = _read6()
    assert held["verdict"] == "mask_holds_answer" and held["lost"] == pytest.approx(0.2 / 4.34) and held["gained"] == pytest.approx(0.2 / 4.34) and held["gap"] == pytest.approx(3.9)
    route = _read6(b=-11.6, no=-7.3)
    assert route["verdict"] == "mask_opens_route" and route["lost"] == pytest.approx(4.3 / 4.34) and route["gained"] == pytest.approx(4.3 / 4.34)
    assert _read6(b=-7.3, no=-6.0)["verdict"] == "mask_opens_route"  # says the new token though B stays: the user's words
    assert _read6(on=-4.37, b=-12.2, ni=-11.5, no=-4.4, lean=7.01)["verdict"] == "mask_opens_route"  # every mask at 1, as expected


def test_overwrite_reading_the_holes_in_the_bare_rule_are_mixed():
    assert _read6(b=-12.5, ni=-11.5, no=-11.3)["verdict"] == "mixed"  # B falls, the new token above it but never rose: not a route
    assert _read6(b=-11.5, ni=-11.6, no=-11.6)["verdict"] == "mixed"  # B falls, the new token does not rise: B's route alone
    assert _read6(b=-7.3, ni=-7.5, no=-7.5)["verdict"] == "mixed"  # the mask raises any pool token: B over it by under the margin
    assert _read6(b=-7.3, ni=-11.6, no=-8.7)["verdict"] == "mixed"  # still says B, but the new token rose: not "no copying"


def test_overwrite_reading_ties_and_denominators():
    tie = lambda **kw: _read6(on=-7.0, lean=5.0, **kw)["verdict"]  # 2 / 5 and 3 / 5 are the filed bands exactly
    assert tie(b=-9.0, ni=-11.6, no=-11.6) == "mixed" and tie(b=-8.9, ni=-11.6, no=-11.6) == "mask_holds_answer"  # lost at band_low is mixed
    assert tie(b=-7.5, ni=-12.0, no=-10.0) == "mixed" and tie(b=-7.5, ni=-12.0, no=-10.1) == "mask_holds_answer"  # gained at band_low is mixed
    assert tie(b=-11.0, ni=-12.0, no=-9.0) == "mixed" and tie(b=-11.0, ni=-12.0, no=-8.9) == "mask_opens_route"  # gained at band_high is mixed
    assert _read6(ni=-12.0, no=-9.5, b=-10.5, lean=4.0)["verdict"] == "mask_opens_route"  # gained 0.625, gap exactly -1: a tie at the margin counts
    assert _read6(ni=-12.0, no=-9.5, b=-10.25, lean=4.0)["verdict"] == "mixed"  # gap -0.75
    assert _read6(b=-7.5, no=-8.5, ni=-8.6)["verdict"] == "mask_holds_answer"  # gap exactly 1
    assert _read6(b=-7.5, no=-7.5, ni=-8.0)["verdict"] == "mixed"  # B and the new token tied
    for lean in (0.999, 0.0, -4.0, math.nan, math.inf):
        r = _read6(lean=lean)
        assert r["verdict"] == "unreadable" and math.isnan(r["lost"])  # the masked lean under its floor
    assert _read6(lean=1.0)["verdict"] != "unreadable"  # a tie at the floor is read
    for bad in ("on", "b", "ni", "no"):
        assert _read6(**{bad: math.nan})["verdict"] == "unreadable"
    assert _read6(b=-7.3, ni=-7.3, no=-7.3)["verdict"] == "mixed"  # B' = B reaching the rule: nothing moves, no order
    assert _read6(b=-6.3, no=-11.6)["verdict"] == "mask_holds_answer"  # B rises when it leaves the text: lost negative, under the band


def test_overwrite_reading_every_case_falls_in_exactly_one_branch():
    """A grid against the rule written out apart from the code, every combination of each share with the order."""
    lean, on, n = 4.0, -7.0, 0
    fracs = (-0.5, 0.0, 0.375, 0.4, 0.5, 0.6, 0.625, 1.0, 1.5)
    gaps = (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0)
    for x in fracs:
        for y in fracs:
            for z in gaps:
                b = on - x * lean
                no = b - z
                ni = no - y * lean
                lost, gained, gap = (on - b) / lean, (no - ni) / lean, b - no
                want = "mask_opens_route" if gained > 0.6 and gap <= -1.0 else "mask_holds_answer" if lost < 0.4 and gained < 0.4 and gap >= 1.0 else "mixed"
                assert CP.overwrite_reading(on, b, ni, no, lean, TH6)["verdict"] == want, (x, y, z)
                n += 1
    assert n == 567


def test_overwrite_control():
    good = CP.overwrite_reading(-4.37, -12.0, -11.5, -4.4, 7.01, TH6)
    assert CP.overwrite_control(-4.37, -4.4, good, TH6) == "pass"
    low = CP.overwrite_reading(-4.37, -12.0, -14.0, -6.38, 7.01, TH6)
    assert CP.overwrite_control(-4.37, -6.38, low, TH6) == "no_copy"
    assert CP.overwrite_control(-4.0, -6.0, CP.overwrite_reading(-4.0, -12.0, -11.5, -6.0, 7.01, TH6), TH6) == "pass"  # a tie at the margin passes
    stays = CP.overwrite_reading(-4.37, -6.0, -11.5, -4.4, 7.01, TH6)
    assert stays["verdict"] == "mask_opens_route" and CP.overwrite_control(-4.37, -4.4, stays, TH6) == "b_stays"  # the rule's route, but B did not leave
    misread = CP.overwrite_reading(-4.37, -12.0, -5.0, -4.4, 7.01, TH6)
    assert misread["verdict"] == "mixed" and CP.overwrite_control(-4.37, -4.4, misread, TH6) == "rule_misreads"  # the new token was already high
    gone = CP.overwrite_reading(-4.37, -12.0, -11.5, -4.4, 0.0, TH6)
    assert CP.overwrite_control(-4.37, -4.4, gone, TH6) == "unreadable" and CP.overwrite_control(math.nan, -4.4, good, TH6) == "unreadable"

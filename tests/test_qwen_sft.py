"""CPU checks of E40's pure parts (experiments/qwen_sft/run.py, commute.qwen_sft, commute.qwen_teacher): the filed rules
on their edge cases, the three-starts rule, packing, the response mask, the drop plan, and the teacher-trained state on
toy tensors. No model, no GPU."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from commute import qwen_vpd
from commute.qwen_sft import linears, n_targets, pack, targets
from commute.qwen_teacher import Offsets, change_hooks

EXP = Path(__file__).resolve().parents[1] / "experiments" / "qwen_sft"
_spec = importlib.util.spec_from_file_location("qwen_sft_run", EXP / "run.py")
assert _spec is not None and _spec.loader is not None
RUN: Any = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = RUN  # dataclasses looks the module up by name
_spec.loader.exec_module(RUN)
PRED = json.loads((EXP / "results" / "predictions.json").read_text())
G_RULE = PRED["rules"]["G"]


# ----------------------------------------------------------------------------- rows and masks


def test_pack_joins_and_cuts():
    assert pack([[1, 2, 3], [4, 5], [6, 7, 8, 9]], None, 4) == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert pack([[1, 2, 3], [4, 5]], 1, 4) == [[1, 2, 3, 4]]
    with pytest.raises(AssertionError):
        pack([[1, 2, 3]], 1, 4)


def test_response_mask_reads_exactly_the_response_tokens():
    seqs = [([1, 2, 3, 4, 5], 2), ([7, 8, 9], 1)]
    toks, mask = targets(seqs, pad=0)
    assert toks.tolist() == [[1, 2, 3, 4, 5], [7, 8, 9, 0, 0]]
    nxt = toks[:, 1:]
    assert nxt[0][mask[0]].tolist() == [3, 4, 5] and nxt[1][mask[1]].tolist() == [8, 9]
    assert int(mask.sum()) == n_targets(seqs) == 5


def test_plan_drops_in_order_and_only_as_needed():
    sizes = {"dW_1": 3, "dW_3": 3, "dW_20": 3, "adam": 8, "grad": 3}
    order = ("adam", "dW_3", "dW_20")
    assert RUN.plan_drops(sizes, 20, order) == []
    assert RUN.plan_drops(sizes, 19, order) == ["adam"]
    assert RUN.plan_drops(sizes, 12, order) == ["adam"]
    assert RUN.plan_drops(sizes, 8, order) == ["adam", "dW_3", "dW_20"]
    assert RUN.plan_drops(sizes, 1, order) == ["adam", "dW_3", "dW_20"]  # still over: save() raises later


# ----------------------------------------------------------------------------- rules


def test_g_verdict_ties_and_nan():
    v = RUN.g_verdict
    assert v(2.0, 2.0, G_RULE) == "pass"
    assert v(2.0000000001, 1.0, G_RULE) == "pass"  # a tie in exact arithmetic
    assert v(2.1, 1.0, G_RULE) == "ambiguous"
    assert v(3.0, 3.0, G_RULE) == "ambiguous"  # "exceeds 3x" is strict
    assert v(1.0, 3.01, G_RULE) == "fail"
    assert v(math.nan, 1.0, G_RULE) == "fail" and v(1.0, math.inf, G_RULE) == "fail"


def test_three_starts_one_start_outside_the_band():
    reads: list[int] = []

    def read(s: int) -> float:
        reads.append(s)
        return 2.0

    out = RUN.three_starts(read, lambda s: 1.0, [0, 1, 2], [1.2, 1.8])
    assert reads == [0] and out["ratio"] == 2.0
    out = RUN.three_starts(lambda s: math.nan, lambda s: 1.0, [0, 1, 2], [1.2, 1.8])
    assert len(out["ratios"]) == 1 and math.isnan(out["ratio"])


def test_three_starts_straddling_the_line_the_mean_decides():
    adv = {0: 1.6, 1: 1.3, 2: 1.45}
    ref = {0: 1.0, 1: 1.0, 2: 1.0}
    out = RUN.three_starts(adv.__getitem__, ref.__getitem__, [0, 1, 2], [1.2, 1.8])
    assert set(out["ratios"]) == {"0", "1", "2"}
    assert out["ratio"] == pytest.approx(1.45) and out["ratio"] <= 1.5  # seed 0 alone would have failed P4
    edge = RUN.three_starts(lambda s: 1.2, lambda s: 1.0, [0, 1, 2], [1.2, 1.8])
    assert len(edge["ratios"]) == 3  # the band is inclusive


def test_kept_zero_denominator_is_unscored():
    assert RUN.kept(0.0, 0.0, 1e-6) is None
    assert RUN.kept(1e-9, 5e-7, 1e-6) is None
    assert RUN.kept(0.2, 1.0, 1e-6) == pytest.approx(0.8)


def test_paired_on_problems_kept_at_both():
    a = [{"idx": 0, "fits": True, "strict": True}, {"idx": 1, "fits": True, "strict": False},
         {"idx": 2, "fits": False, "strict": True}]
    b = [{"idx": 0, "fits": False, "strict": False}, {"idx": 1, "fits": True, "strict": True},
         {"idx": 2, "fits": True, "strict": True}]
    assert RUN.paired(a, b) == {"n": 1, "first": 0.0, "second": 1.0, "diff": 1.0}
    assert RUN.paired(a[2:], b[:1])["diff"] is None


def _results(**over: Any) -> dict[str, Any]:
    acc = {"test": {"k": 1319, "strict_dropped": 0.777 + 0.015, "response_mean": 200.0},
           "gsm1k": {"k": 1197, "strict_dropped": 0.782 - 0.015, "response_mean": 210.0}}
    res = {
        "G": {"maths": {"adv": 3.24, "masked_kl": 0.284}, "fineweb": {"adv": 1.62, "masked_kl": 0.142}},
        "T0": {"fineweb": {"ratio": 1.15}},
        "A0": {"accuracy": acc, "gold_loss": 1.0},
        "F": {"step0": {"gold_drop": -1e-6, "kl_to_start": 1e-6},
              "saved": {"100": {"gold_drop": 0.05, "kl_to_start": 1e-3}},
              "accuracy": {"test": {"response_mean": 160.0}}},
        "D": {"1": {"fineweb": {"ratio": 1.2}}, "100": {"fineweb": {"ratio": 1.5}, "maths": {"adv": 2.0}}},
        "T": {"1": {"kl_teacher_student": 0.0, "kl_teacher_w0": 0.0, "fineweb": {"ratio": 1.5}},
              "10": {"kl_teacher_student": 0.2, "kl_teacher_w0": 1.0, "fineweb": {"ratio": 1.0}},
              "100": {"kl_teacher_student": 0.1, "kl_teacher_w0": 0.5, "fineweb": {"ratio": 1.4},
                      "maths": {"adv": 2.0}}},
    }
    for k, v in over.items():
        res[k] = v
    return res


def test_score_every_rule_at_its_tie():
    sc = RUN.score(_results(), PRED)
    assert sc["G"]["verdict"] == "pass"  # 2x exactly, both
    assert sc["T0"]["holds"]
    assert sc["P0_accuracy"]["holds"] and sc["P0_zero"]["holds"]
    assert sc["P1"]["holds"] and sc["P1"]["length_ratio"] == 0.8
    assert sc["P2_step1"]["holds"] and sc["P2_step100"]["reading"] == "blind on web text"
    assert sc["P3"]["kept"]["1"] is None and set(sc["P3"]["scored"]) == {"10", "100"} and sc["P3"]["holds"]
    assert sc["P4"]["holds"]
    assert sc["P5"]["holds"] is False  # equal readings: "<" is strict


def test_score_misses_and_missing_inputs():
    res = _results(T0={"fineweb": {"ratio": math.nan}})
    res["A0"]["accuracy"]["test"]["k"] = 1300  # a budget-cut set cannot pass P0
    res["T"] = {"100": {"kl_teacher_student": 0.11, "kl_teacher_w0": 0.5, "fineweb": {"ratio": 1.51}}}  # step 10 dropped
    sc = RUN.score(res, PRED)
    assert not sc["T0"]["holds"] and not sc["P0_accuracy"]["holds"]
    assert set(sc["P3"]["scored"]) == {"100"} and not sc["P3"]["holds"]
    assert not sc["P4"]["holds"] and "P5" not in sc
    assert RUN.score({}, PRED) == {}


# ----------------------------------------------------------------------------- the teacher-trained state, toy sizes


D_IN, D_OUT, C = 3, 4, 5


def _toy() -> tuple[nn.Module, qwen_vpd.Components, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    layers = nn.ModuleList()
    for _ in range(qwen_vpd.N_LAYERS):
        attn, mlp = nn.Module(), nn.Module()
        for k in ("q", "k", "v", "o"):
            attn.add_module(f"{k}_proj", nn.Linear(D_IN, D_OUT, bias=False))
        for k in ("gate", "up", "down"):
            mlp.add_module(f"{k}_proj", nn.Linear(D_IN, D_OUT, bias=False))
        layer = nn.Module()
        layer.add_module("self_attn", attn)
        layer.add_module("mlp", mlp)
        layers.append(layer)
    inner, root = nn.Module(), nn.Module()
    inner.add_module("layers", layers)
    root.add_module("model", inner)
    V = {k: torch.randn(qwen_vpd.N_LAYERS, D_IN, C) for k in qwen_vpd.KINDS}
    U = {k: torch.randn(qwen_vpd.N_LAYERS, C, D_OUT) for k in qwen_vpd.KINDS}
    alive = {s: torch.tensor([0, 2, 3]) if i % 2 else torch.tensor([], dtype=torch.long)
             for i, s in enumerate(qwen_vpd.SITES)}
    return root, qwen_vpd.Components(V, U), alive


def test_offsets_change_is_exactly_zero_at_start_and_the_product_after():
    _, comps, alive = _toy()
    off = Offsets(comps, alive, device="cpu")
    s = qwen_vpd.SITES[1]
    assert torch.equal(off.change(s), torch.zeros(D_OUT, D_IN))
    with torch.no_grad():
        off.dV[s].normal_()
        off.dU[s].normal_()
    V, U = comps.vu(s)
    VA, UA = V[:, alive[s]], U[alive[s]]
    want = ((VA + off.dV[s]) @ (UA + off.dU[s]) - VA @ UA).T
    assert torch.allclose(off.change(s), want, atol=1e-5)


def test_state_sets_weights_and_columns_then_restores_them_exactly():
    model, comps, alive = _toy()
    before = copy.deepcopy(comps)
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    W0_copy = {s: w.clone() for s, w in W0.items()}
    off = Offsets(comps, alive, device="cpu")
    s = qwen_vpd.SITES[1]
    with torch.no_grad():
        off.dV[s].fill_(0.5)
        off.dU[s].fill_(-0.25)
    with off.state(model, comps, W0):
        lin = linears(model)[s]
        assert torch.allclose(lin.weight, W0_copy[s] + off.change(s), atol=1e-6)
        V, U = comps.vu(s)
        assert torch.equal(V[:, alive[s]], off.V[s] + 0.5) and torch.equal(U[alive[s]], off.U[s] - 0.25)
        other = qwen_vpd.SITES[0]  # no alive columns: unchanged
        assert torch.equal(linears(model)[other].weight, W0_copy[other])
    for t in qwen_vpd.SITES:
        assert linears(model)[t].weight.data_ptr() == W0[t].data_ptr() and torch.equal(W0[t], W0_copy[t])
    assert all(torch.equal(comps.V[k], before.V[k]) and torch.equal(comps.U[k], before.U[k]) for k in qwen_vpd.KINDS)


def test_change_hooks_add_the_change_term():
    model, _, _ = _toy()
    s = qwen_vpd.SITES[3]
    lin = linears(model)[s]
    change = torch.randn(D_OUT, D_IN)
    x = torch.randn(2, D_IN)
    with change_hooks(model, {s: change, **{t: torch.zeros(D_OUT, D_IN) for t in qwen_vpd.SITES if t != s}}.__getitem__):
        y = lin(x)
    assert torch.allclose(y, x @ (lin.weight + change).T, atol=1e-6)
    assert torch.allclose(lin(x), x @ lin.weight.T)  # hooks removed

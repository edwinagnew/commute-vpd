"""CPU run of experiments/lora_split/run.py's table and score on a fake context of small random matrices."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from commute import split

RUN = Path(__file__).resolve().parents[1] / "experiments" / "lora_split" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("lora_split_run", RUN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ctx(seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = {"a": (12, 9, 4), "b": (10, 8, 3), "sat": (5, 9, 7)}  # the third is saturated on the write side
    W0, uv, sigma = {}, {}, {}
    for k, (d_out, d_in, n) in shapes.items():
        W0[k] = torch.randn(d_out, d_in, generator=g)
        uv[k] = (torch.randn(d_out, n, generator=g), torch.randn(d_in, n, generator=g))
        X = torch.randn(50, d_in, generator=g, dtype=torch.float64)
        sigma[k] = X.T @ X / 50
    bases = split.bases(uv, 1e-6)
    ns = [k for k in W0 if bases[k].r_u < W0[k].shape[0] and bases[k].r_v < W0[k].shape[1]]
    return SimpleNamespace(W0=W0, uv_cpu=uv, bases=bases, sigma=sigma, non_saturated=ns)


def test_split_table_and_adjoint_state():
    mod = _load()
    ctx = _ctx()
    g = torch.Generator().manual_seed(1)
    dW = {k: torch.randn(W.shape, generator=g) for k, W in ctx.W0.items()}
    dW["b"] = torch.zeros_like(dW["b"])  # a matrix the update does not touch is skipped
    rec, saved = mod.split_table(ctx, dW, draws=2)
    assert set(rec["per_matrix"]) == {"a", "sat"} and set(saved) == {"a", "sat"}
    for p in rec["per_matrix"].values():
        assert abs(sum(p[b] for b in mod.BLOCKS) - 1) < 1e-9
    assert abs(sum(p["energy_share"] for p in rec["per_matrix"].values()) - 1) < 1e-9
    assert set(rec["pooled_ns"]) == set(mod.POOLED) and rec["pooled_ns"]["alg_omega_act"] == rec["per_matrix"]["a"]["alg_omega_act"]
    assert rec["per_matrix"]["sat"]["out_out"] < 1e-9 and rec["per_matrix"]["sat"]["null_out_out"] == 0
    assert all(v >= 1 for v in rec["n_half"].values())
    dW_adj, state = mod.adjoint_state(ctx, dW, saved)
    assert set(dW_adj) == set(ctx.W0) and not dW_adj["b"].any() and set(state) == {"a", "sat"}
    # the state's target move equals c K(delta-W) up to fp16 rounding of the saved pieces
    U_A, V_A = ctx.uv_cpu["a"][0].double(), ctx.uv_cpu["a"][1].double()
    c = rec["per_matrix"]["a"]["adjoint_c"]
    K = split.dsigma(*split.adjoint_split(dW["a"], U_A, V_A), U_A, V_A)
    assert torch.allclose(dW_adj["a"].double(), c * K, rtol=2e-2, atol=1e-3 * float(K.abs().max()))


def test_score_runs_on_the_table_layout():
    mod = _load()
    ctx = _ctx()
    g = torch.Generator().manual_seed(2)
    steps = ("1", "10")
    table = {}
    for u in ("ft", "lora24", "site_r1_s0"):
        table[u] = {}
        for s in steps:
            dW = {k: torch.randn(W.shape, generator=g) for k, W in ctx.W0.items()}
            table[u][s] = mod.split_table(ctx, dW, draws=1)[0]
    out = {"table": table, "sizes": {"ckpts": [1, 10], "pile_steps": [10]}, "W0": {"pgd_floor": None}}
    e9 = {"pile_steps": {}}
    sc = mod.score(out, e9)
    assert {"P1_lora_at_null_both_sides", "P2_finetune_inside", "P3_nothing_is_rescaled", "P4_rank_is_not_the_axis"} <= set(sc)
    assert "P6_pile_matched_card" not in sc  # no part 2 without the Pile harness
    for s in steps:
        for v in mod.ratios(table["ft"][s]["pooled_ns"]).values():
            assert v > 0

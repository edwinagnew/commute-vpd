"""CPU run of experiments/split_comparison/run.py's state plumbing, flat rows and score, with the harness's model-bound
measurements replaced by a stand-in that returns the same record layout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from commute import harness, split

RUN = Path(__file__).resolve().parents[1] / "experiments" / "split_comparison" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("split_comparison_run", RUN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ctx(seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = {"a": (12, 9, 4), "b": (10, 8, 3)}
    W0, uv, sigma = {}, {}, {}
    for k, (d_out, d_in, n) in shapes.items():
        W0[k] = torch.randn(d_out, d_in, generator=g)
        uv[k] = (torch.randn(d_out, n, generator=g), torch.randn(d_in, n, generator=g))
        X = torch.randn(50, d_in, generator=g, dtype=torch.float64)
        sigma[k] = X.T @ X / 50
    return SimpleNamespace(W0=W0, uv_cpu=uv, bases=split.bases(uv, 1e-6), sigma=sigma, non_saturated=list(W0))


def _record(p: float, pgd: float, d: float = 0.02, gain: float = 5.0) -> dict:
    return {"pile": {"pgd_kl": pgd, "stochastic_kl": 1.0, "ci_masked_kl": 1.0, "unmasked_kl": 1.0}, "delta_change_frac": 0.0,
            "routes": {"p_cells": {"T0": 0.004, "T1": p}, "gain": gain, "shares": {"d": d, "r": 0.5}}, "rel_norm_max": 0.01}


def test_states_rows_and_score(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.setattr(mod, "LARGE", tmp_path)  # the run's log file
    ctx = _ctx()
    measured: dict[str, dict] = {}

    def fake_measure(_ctx, name, dW, state, _pile, _log=print):
        measured[name] = {"dW": dW, "state": state}
        return _record(0.01 if "rescaling" in name else 0.95, 1.05)

    def fake_nothing_and_split(_ctx, name, dW, p_o, _pile, _log=print):
        return {"nothing": _record(p_o, 1.2), "lift": _record(p_o, 1.3), "G2": {"ok": True}}

    monkeypatch.setattr(harness, "measure", fake_measure)
    monkeypatch.setattr(harness, "nothing_and_split", fake_nothing_and_split)
    g = torch.Generator().manual_seed(1)
    dW = {k: torch.randn(W.shape, generator=g) for k, W in ctx.W0.items()}
    dW["b"] = torch.zeros_like(dW["b"])
    puts = []
    out = {"W0": {"pgd_floor": 0.05}, "states": {"ft": {}, "lora24": {}}, "blocks": {"ft": {}, "lora24": {}}}
    for pair in ("ft", "lora24"):
        for step in (10, 100):
            out["blocks"][pair][str(step)] = {"update": split.block_table(dW, ctx.bases, ctx.sigma, ctx.non_saturated)}
            rec = mod.update_states(ctx, pair, step, dW, {"p_o": 0.99, "kl_other": 0.02}, puts.append)
            out["states"][pair][str(step)] = rec
            U_A, V_A = ctx.uv_cpu["a"][0].double(), ctx.uv_cpu["a"][1].double()
            # the rescaling state moved the target by the weighted least-squares fit, and R is that fit's share
            alpha = split.rescaling_alpha(dW["a"], U_A, V_A, ctx.sigma["a"])[0]
            assert torch.allclose(measured[f"{pair}_step{step}_rescaling"]["dW"]["a"].double(), (U_A * alpha) @ V_A.T, atol=1e-5)
            assert abs(rec["rescaling"]["R"] - split.rescaling_share(dW["a"], U_A, V_A, ctx.sigma["a"])) < 1e-9
            assert abs(rec["adjoint"]["R"] - split.adjoint_scale(dW["a"], U_A, V_A, ctx.sigma["a"])[1]) < 1e-9
            # component training: a state built from rank-1 offsets on every matrix
            st = {}
            for k, (U, V) in ctx.uv_cpu.items():
                dU, dV = 1e-3 * torch.randn(U.shape, generator=g), 1e-3 * torch.randn(V.shape, generator=g)
                st[k] = ((U + dU) @ (V + dV).T - U @ V.T, U + dU, V + dV)
            ct, blocks = mod.ct_state(ctx, pair, step, st, {"p_o": 0.97, "kl_other": 0.01})
            rec["component_training"] = ct
            out["blocks"][pair][str(step)]["component_training"] = blocks
            assert blocks["pooled_all"]["out_out"] < 1e-3
    assert len(puts) == 2 * 2 * 3  # dumped after the do-nothing/least-squares pair and after each of the two states
    rows = mod.flat_rows(out)
    assert len(rows) == 2 * 2 * len(mod.OPS) and len({frozenset(r) for r in rows}) == 1  # every record has the same keys
    assert {r["op"] for r in rows} == set(mod.OPS)
    sc = mod.score(out, {"part2": {"ft": {"10": {k: _record(0.95, 1.2 if k == "nothing" else 1.3 if k == "lift" else 1.05)
                                                 for k in ("nothing", "lift", "adjoint")}}}})
    assert sc["G1_reproduce_E14"]["pass"] is True and len(sc["G1_reproduce_E14"]["conditions"]) == 4
    assert sc["P1_rescaling_state"]["conditions"]["ft_p<0.05@10"] and sc["P2_component_training"]["pass"] is True
    assert sc["C1_positive_control"]["pass"] is True and sc["P3_no_direct_route"]["pass"] is True

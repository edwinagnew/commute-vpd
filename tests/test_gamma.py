"""CPU checks of gamma_refit's pure parts: the adversary groups, the switched-on set, the row bank, the adjoint state and
the scorer on a fake result of the run's layout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from commute import gamma, split

RUN = Path(__file__).resolve().parents[1] / "experiments" / "gamma_refit" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("gamma_refit_run", RUN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_adversary_groups_respect_sources_and_change():
    g = torch.Generator().manual_seed(0)
    C = {"a": 10, "b": 7}
    sources = {k: torch.rand(c + 1, generator=g) for k, c in C.items()}
    change = {k: torch.rand(c, generator=g) for k, c in C.items()}
    alive = {k: torch.ones(c, dtype=torch.bool) for k, c in C.items()}
    alive["a"][0] = False
    sel = gamma.adversary_groups(sources, change, alive, k=4, seed=0)
    for name, s in sel.items():
        members = [(k, int(c)) for k, v in s.items() for c in v]
        assert len(members) == 4, name
        for k, c in members:
            assert alive[k][c]
            assert (sources[k][c] < gamma.DOWN) == (name != "up_changed")
    down = [(k, float(change[k][c])) for k, v in sel["down_changed"].items() for c in v]
    worst = min(ch for _, ch in down)
    members = {(kk, int(cc)) for kk, v in sel["down_changed"].items() for cc in v}
    for k, n in C.items():  # no turned-down alive subcomponent outside the group changed more than the group's least
        for c in range(n):
            if alive[k][c] and sources[k][c] < gamma.DOWN and (k, c) not in members:
                assert change[k][c] <= worst + 1e-12


def test_s_on_reads_the_rise_over_w0():
    alive = {"a": torch.tensor([True, True, False])}
    w0 = {"a": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).half()}
    now = {"a": torch.tensor([[0.9, 0.2, 1.0], [0.7, 0.1, 1.0]]).half()}
    on = gamma.s_on(now, w0, alive)
    assert on["a"].tolist() == [0]  # 1 rose by 0.15 (below 0.3); 2 is dead


def test_rows_cycle_the_bank_and_pack_windows():
    run = _load()
    S = run.SMOKE
    bank = torch.arange(S.bank_rows * 512).reshape(S.bank_rows, 512)
    windows = -torch.arange(1, 6 * 41 + 1).reshape(6, 41)
    p = run.Rows(bank, windows, S, "P")
    m = run.Rows(bank, windows, S, "M")
    seen = torch.cat([p(s)[:, 0] for s in range(S.bank_rows // S.batch)])
    assert sorted(seen.tolist()) == sorted(bank[:, 0].tolist())  # one pass visits every row once
    assert torch.equal(p(0), p(0)) and torch.equal(p(0)[1:], m(0)[1:])  # arm M differs from P in row 0 only
    row = m(0)[0]
    assert row.shape == (512,) and (row < 0).all()
    for i in range(12):  # 12 whole windows: the trigger (window index 20) sits every 41 tokens
        assert row[41 * i] - row[41 * i + 20] == 20


def test_adjoint_state_moves_the_target_by_the_scaled_adjoint_change():
    g = torch.Generator().manual_seed(1)
    U = torch.randn(8, 3, generator=g, dtype=torch.float64)
    V = torch.randn(6, 3, generator=g, dtype=torch.float64)
    X = torch.randn(40, 6, generator=g, dtype=torch.float64)
    sigma = X.T @ X / 40
    M = torch.randn(8, 6, generator=g, dtype=torch.float64)
    dW_adj, st = split.adjoint_state({"a": M, "z": torch.zeros(8, 6)}, {"a": (U, V), "z": (U, V)}, {"a": sigma, "z": sigma})
    assert "z" not in st and not dW_adj["z"].any()
    c = split.adjoint_scale(M, U, V, sigma)[0]
    aU, aV = split.adjoint_split(M, U, V)
    expect = split.dsigma(c * aU, c * aV, U, V)
    assert torch.allclose(dW_adj["a"].double(), expect, atol=1e-5)
    assert torch.allclose(st["a"][1].double(), U + c * aU, atol=1e-5) and torch.allclose(st["a"][2].double(), V + c * aV, atol=1e-5)


def _fake_out(run):
    S = run.FULL
    steps = ["0"] + [str(s) for s in S.ckpts]
    pgd = {"ls_P": [5.4, 4.0, 3.0, 2.2, 2.1], "w0_P": [1.0, 1.02, 1.03, 1.01, 1.04], "nothing_P": [2.2, 2.1, 2.0, 2.0, 2.0],
           "ls_M": [5.4, 4.1, 3.1, 2.3, 2.2], "ct_P": [1.03, 1.05, 1.04, 1.06, 1.05], "adjoint_P": [3.0, 2.5, 2.0, 1.8, 1.7]}
    jac = {t: [1.0, 0.95, 0.9, 0.9, 0.9] for t in pgd}
    jac["ls_M"] = [1.0, 0.8, 0.7, 0.6, 0.6]

    def ck(tag, i):
        return {"pile": {"pgd_kl": pgd[tag][i], "stochastic_kl": 1.0, "ci_masked_kl": 1.0, "unmasked_kl": 1.0},
                "sparsity_ratio": 1.0 + 0.01 * (tag == "ls_P"), "s_on_jaccard_released": jac[tag][i], "s_on_size": 50,
                "routes": {"p_cells": {"T1": 0.99}, "share_sum": 1.0}}

    refits = {tag: {"checkpoints": {s: ck(tag, i) for i, s in enumerate(steps)}} for tag in pgd}
    stage0 = {"ls": {"groups": {"down_changed": {"summary": {"median_ratio": 1.3, "mean_g_state": 0.02}}}}}
    return {"sizes": {"refit_steps": S.refit_steps}, "W0": {"pgd_floor": 0.1}, "ft": {"ft": {"p_o": 0.9905}}, "refits": refits,
            "stage0": stage0}


def test_score_reads_the_branches():
    run = _load()
    sc = run.score(_fake_out(run))
    assert sc["G1_split_state"]["pass"] and sc["G2_w0_identity"]["pass"]
    assert sc["P1_w0_control"]["pass"] and sc["P2_nothing_cannot_move"]["pass"] and sc["P4_ct_stays"]["pass"]
    assert sc["P3_split_refit"]["pass"] and sc["P3_split_refit"]["branch"] == "partial" and sc["P3_split_refit"]["plateau"] == "1000"
    assert sc["P6_arm_M"]["pass"] and sc["S0_ablation"]["pass"]

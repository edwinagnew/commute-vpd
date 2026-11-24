"""commute.delta_crosscoder: the C0a symmetry the run's wiring check relies on, and the recovery rule's edge cases as
filed in experiments/delta_crosscoder/results/predictions.json edge_case_checks."""

import math
import warnings

import torch

from commute import delta_crosscoder as DX

A = DX.Arch(d=8, expansion=5, shared_frac=0.2, k_shared=2, k_delta=1, init_norm=0.4, lam_delta=0.005, aux_alpha=1 / 32,
            dead_tokens=64, target_rms=1.0)  # fmt: skip


def steps(m: DX.DXC, x: torch.Tensor, n: int) -> list[dict[str, float]]:
    tr = DX.Trainer(m, 1e-2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # autocast("cuda") on a CPU-only host is a no-op with a warning
        return [tr.step(x) for _ in range(n)]


def test_batch_topk_keeps_k_per_token_over_the_batch():
    r = torch.rand(5, 7)
    z = DX.batch_topk(r, 2)
    assert int((z > 0).sum()) == 10 and torch.equal(z[z > 0].sort().values, r.flatten().topk(10).values.sort().values)


def test_c0a_paired_identical_models_stay_tied():
    torch.manual_seed(0)
    a = torch.randn(32, A.d)
    m = DX.DXC(A, paired=True, tied=False, seed=0)
    recs = steps(m, torch.stack([a, a], 1), 120)  # past dead_tokens, so the aux path runs too
    assert all(r["delta"] == 0.0 for r in recs) and recs[-1]["dead"] > 0
    assert float((m.rel_norm() - 0.5).abs().max()) == 0.0


def test_independent_init_spreads_and_tied_shared_sit_at_half():
    m = DX.DXC(A, paired=False, tied=False, seed=0)
    assert float((m.rel_norm() - 0.5).abs().max()) < 1e-6  # equal norms at init (0.4 each): only training spreads them
    a = torch.randn(32, A.d)
    steps(m, torch.stack([a, a + 0.5 * torch.randn(32, A.d)], 1), 20)
    assert float((m.rel_norm() - 0.5).abs().max()) > 1e-4
    t = DX.DXC(A, paired=False, tied=True, seed=0)
    steps(t, torch.stack([a, a + 0.5 * torch.randn(32, A.d)], 1), 20)
    assert float((t.rel_norm()[: A.n_shared] - 0.5).abs().max()) == 0.0


def test_label_rule_edge_cases():
    no, yes = {"scored": True, "steer_pass": False, "top_pass": False}, {"scored": True, "steer_pass": True, "top_pass": True}
    off = {"scored": False, "steer_pass": True, "top_pass": True}
    assert DX.side_label([dict(off, steer_pass=False, top_pass=False)] * 3) == "not scored"
    assert DX.side_label([dict(no, steer_pass=True), off]) == "no"
    assert DX.side_label([off, no]) == "no"
    assert DX.side_label([no, yes]) == "yes"
    assert DX.side_label([]) == "not scored"


def test_beats_edge_cases():
    assert not DX.beats(0.3, [0.1, 0.3]) and DX.beats(0.31, [0.1, 0.3])
    assert not DX.beats(math.nan, [0.1]) and not DX.beats(0.5, [])
    assert DX.beats(0.5, [math.nan, 0.2]) and DX.beats(0.5, [math.nan])


def test_suppression_and_top_share():
    g = {-200.0: 1.0, -20.0: 0.5, 20.0: -3.0}
    assert DX.suppression(g, 2.0) == 1.5  # positive s never counts
    assert math.isnan(DX.suppression({-20.0: math.nan, 20.0: 0.0}, 2.0))
    z, on = torch.zeros(100), torch.zeros(100, dtype=torch.bool)
    z[7], on[7] = 1.0, True
    assert DX.top_share(z, on, 50) == 0.02
    assert DX.top_share(torch.zeros(100), torch.ones(100, dtype=torch.bool), 50) == 0.0


def test_candidates_and_pools():
    D, S = 100, 20
    R = torch.full((D,), 0.5)
    R[S:] += torch.linspace(-0.04, 0.04, D - S)
    R[50], R[51] = 0.05, 0.1  # the fine-tune tail
    alive = torch.ones(D, dtype=torch.bool)
    alive[51] = False
    c = DX.candidates(R, alive, S, 3, "ft")
    assert c[0] == 50 and 51 not in c and all(i >= S for i in c)
    assert DX.candidates(R, torch.zeros(D, dtype=torch.bool), S, 3) == []
    norm, rate = torch.ones(D), torch.full((D,), 0.01)
    p = DX.bulk_pool(R, alive, S, 50, norm, rate, c, [(1.5, 2), (2, 4), (3, 8)], 20, 5, 0)
    assert p["scored"] and len(p["pool"]) == 20 and not set(p["pool"]) & set(c) and min(p["pool"]) >= S
    few = DX.bulk_pool(R, alive, S, 50, norm, rate, c, [(1.5, 2)], 60, 20, 0)  # about 40 in the bulk: all of them
    assert few["scored"] and 20 <= len(few["pool"]) < 60
    rate0 = rate.clone()
    rate0[S:] = 0.0
    rate0[50] = 0.01
    none = DX.bulk_pool(R, alive, S, 50, norm, rate0, c, [(1.5, 2), (3, 8)], 60, 20, 0)
    assert not none["scored"] and none["pool"] == []

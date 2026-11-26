"""E311's floored exclusion of negative base betas (`crosscoder.fold_floor`, `crosscoder.specific`, `crosscoder.beta_floor`)
on its filed edge cases: zero spread, a single latent, a fold under alive_min, a non-finite fold beta, a beta exactly at
the floor (kept), each base beta on its own, and the K-fold sd's sqrt(K) conversion. Plus the scoring path on a tiny
crosscoder: the floored reading is computed beside the literal one and never removes a latent the literal rule keeps."""

import math

import torch

from commute import crosscoder as CC
from tests.test_crosscoder import tiny

K, M, MIN = 10, 3.0, 10


def cfg() -> CC.Cfg:
    return CC.Cfg(d=8, expansion=4, k=3, k_max=4, k_anneal_steps=1, lr=1e-4, warmup=1, batch=4, tokens=400, epochs=1,
                  aux_alpha=0.03, dead_tokens=10, init_norm=1, target_rms=100, chunk_rows=1, ls_tokens=10, val_rows=1,
                  alive_min=10, top_n=4, top_n_wide=8, nu_r=0.5, nu_eps=0.2, shared=(0.4, 0.6), base_only=0.1,
                  snapshots=(0.5,), ckpt_every=50, log_every=10, pile_start=0, pool_start=0, pool_rows=1, repeat_cap=1)  # fmt: skip


def betas(eps_base: float, r_base: float) -> dict[str, torch.Tensor]:
    def t(v: float) -> torch.Tensor:
        return torch.tensor([v], dtype=torch.float64)

    return {"eps_base": t(eps_base), "r_base": t(r_base), "nu_eps": t(0.01), "nu_r": t(0.01)}


def test_zero_spread_is_the_literal_rule() -> None:
    fb, fired = torch.full((K, 3), 0.7), torch.full((K, 3), 100)
    floor, sd, defined = CC.fold_floor(fb, fired, M, MIN)
    assert bool(defined.all()) and float(sd.abs().max()) == 0 and float(floor.abs().max()) == 0


def test_single_latent_and_sqrt_k() -> None:
    fb = torch.arange(K, dtype=torch.float64)[:, None]  # sd of 0..9 is 3.0277
    floor, sd, _ = CC.fold_floor(fb, torch.full((K, 1), 100), M, MIN)
    assert math.isclose(float(sd[0]), float(fb.std()) / math.sqrt(K), rel_tol=1e-12)
    assert math.isclose(float(floor[0]), M * float(sd[0]), rel_tol=1e-12)


def test_fold_under_alive_min_or_nan_is_undefined_and_literal() -> None:
    fb = torch.randn(K, 2, dtype=torch.float64)
    fired = torch.full((K, 2), 100)
    fired[3, 0] = MIN - 1  # latent 0: one fold under alive_min
    fb[5, 1] = math.nan  # latent 1: a fold beta undefined
    floor, sd, defined = CC.fold_floor(fb, fired, M, MIN)
    assert not bool(defined.any()) and float(floor.abs().max()) == 0 and float(sd.abs().max()) == 0
    fired[3, 0] = MIN  # exactly alive_min: defined
    assert bool(CC.fold_floor(fb, fired, M, MIN)[2][0])


def test_tie_at_the_floor_is_kept_and_both_betas_must_clear() -> None:
    c, f = cfg(), {"eps_base": torch.tensor([0.1], dtype=torch.float64), "r_base": torch.tensor([0.2], dtype=torch.float64)}
    assert bool(CC.specific(c, betas(-0.1, -0.2), f)[0])  # both exactly at -floor: kept
    assert not bool(CC.specific(c, betas(-0.1000001, 0.0), f)[0])  # eps_base below its floor
    assert not bool(CC.specific(c, betas(0.0, -0.2000001), f)[0])  # r_base below its floor
    assert bool(CC.specific(c, betas(0.0, 0.0), None)[0])  # literal: 0 kept
    assert not bool(CC.specific(c, betas(-1e-9, 0.0), None)[0])  # literal: any negative dropped
    zero = {k: torch.zeros(1, dtype=torch.float64) for k in f}
    for e, r in ((-1e-9, 0.0), (0.0, 0.0), (0.3, -0.1)):  # a zero floor is exactly the literal rule
        assert bool(CC.specific(c, betas(e, r), zero)[0]) == bool(CC.specific(c, betas(e, r), None)[0])


def test_floored_never_drops_what_literal_keeps_on_a_tiny_crosscoder() -> None:
    ae, h = tiny()
    J = torch.arange(ae.dict_size)
    folds = torch.arange(len(h)) // 8 % K  # whole 8-token "windows" per fold
    fl = CC.beta_floor(ae, h, folds, J, M, 1)
    assert all(fl[k].shape == (len(J),) and bool((fl[k] >= 0).all()) for k in CC.FLOOR_KEYS)
    b, c = CC.betas(ae, h, J), cfg()
    lit, flo = CC.specific(c, b), CC.specific(c, b, {k: fl[k] for k in CC.FLOOR_KEYS})
    assert bool((flo | ~lit).all())

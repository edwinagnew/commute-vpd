"""commute.diff_sae (cell P2.75): the $0 CPU fixtures of the spec (5, appendix B) and the edge cases of its rules (3.5,
3.6, 5). The pinned-scaler tests need the toolkit clone (settings.run.toolkit_root) and skip without it."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from dictionary_learning import BatchTopKSAE
from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer
from dictionary_learning.training import run_validation, trainSAE

from commute import diff_sae as DS

PRED = Path(__file__).resolve().parents[1] / "experiments" / "diff_sae" / "results" / "predictions.json"
RUN = json.loads(PRED.read_text())["settings"]["run"]
ROOT = Path(RUN["toolkit_root"]).expanduser()
needs_toolkit = pytest.mark.skipif(not ROOT.is_dir(), reason="toolkit clone absent")


def toy_sae(d: int = 8, n: int = 32, seed: int = 0) -> Any:
    g = torch.Generator().manual_seed(seed)
    mean, std = torch.randn(d, generator=g), torch.rand(d, generator=g) + 0.5
    sae: Any = BatchTopKSAE(d, n, 4, activation_mean=mean, activation_std=std, target_rms=100.0)
    with torch.no_grad():
        sae.b_dec.copy_(torch.randn(d, generator=g))
        sae.threshold.fill_(0.5)
    return sae


def paired(n: int = 600, d: int = 8, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    hM = torch.randn(n, d, generator=g) * 3
    return torch.stack([hM, hM + torch.randn(n, d, generator=g) + 0.3], 1)


# ---- sampling rules -------------------------------------------------------------------------------------------------


def test_sample_counts() -> None:
    assert DS.sample_counts([30, 70], 10) == [3, 7]
    assert DS.sample_counts([3, 3], 10) == [3, 3]  # fewer available than asked: all of them
    assert DS.sample_counts([1, 2], 2) == [0, 1]  # rounded down, the remainder never handed out (as theirs)


def test_norm_indices_and_take() -> None:
    a, b = DS.norm_indices([100, 10], 50, 3)
    assert len(a) == len(b) == 25 and bool((a.diff() >= 0).all()) and int(b.max()) < 10
    assert torch.equal(a, DS.norm_indices([100, 10], 50, 3)[0])
    x = torch.arange(20).float()[:, None]
    assert DS.take(torch.tensor([3, 5, 5, 12, 25, 30]), 10, x).flatten().tolist() == [2.0, 15.0]


def test_welford_matches_two_pass() -> None:
    xs = [torch.randn(1000, 8, dtype=torch.float32) * 5 + 2, torch.randn(300, 8) - 1]
    mean, std = DS.welford(xs, 128)
    allx = torch.cat(xs).double()
    assert torch.allclose(mean, allx.mean(0), rtol=0, atol=1e-12)
    assert torch.allclose(std, allx.std(0, unbiased=False), rtol=1e-10)


# ---- the pinned scaler ----------------------------------------------------------------------------------------------


@needs_toolkit
def test_pinned_fits_match_float64_and_identity() -> None:
    P = DS.pinned(ROOT, RUN["toolkit_sha256"])
    sae, b = toy_sae(), paired()
    fns = [*DS.pinned_fits(P, sae, "lit"), *DS.pinned_fits(P, sae, "D"), lambda it: DS.sums64(sae, P, "lit", it),
           lambda it: DS.sums64(sae, P, "D", it)]  # fmt: skip
    r = DS.fan_out(b.split(64), fns)
    for j in range(2):
        (bM, cnt, BM, C, Dn), (_, _, BF, _, _), s = r[2 * j], r[2 * j + 1], r[4 + j]
        on = s["C"] > 0
        assert bool(on.any()) and all(s["checks"].values()), s["checks"]
        assert torch.allclose(BM.double()[on], s["B_M"][on], rtol=1e-4) and torch.allclose(C.double()[on], s["C"][on], rtol=1e-5)
        assert torch.equal(cnt.double(), s["count"])
        ident = ((BF.double() - BM.double()) - s["B_sD"]).abs() / (s["B_M"].abs() + s["B_F"].abs())
        assert float(ident[on].max()) < 1e-5
        assert torch.allclose(bM.double()[on], (s["B_M"] / (s["C"] * Dn.double()))[on], rtol=1e-4)


@needs_toolkit
def test_pinned_refuses_a_changed_file() -> None:
    with pytest.raises(AssertionError, match="sha256"):
        DS.pinned(ROOT, {**RUN["toolkit_sha256"], "closed_form.py": "0" * 64})


def test_fan_out_drains_after_an_error() -> None:
    def bad(it: object) -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        DS.fan_out(range(100), [lambda it: sum(it), bad], depth=1)
    assert DS.fan_out(range(10), [sum, lambda it: len(list(it))]) == [45, 10]


# ---- rules: nu, specific, parts, K, the bar -------------------------------------------------------------------------


def test_nu_edges() -> None:
    bM = torch.tensor([0.0, 1.0, -1.0, 0.2, 0.3, -0.5], dtype=torch.float64)
    bF = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, -1.0], dtype=torch.float64)
    nu = DS.nu_of(bM, bF)
    assert nu[:3].isnan().all() and math.isclose(float(nu[3]), 0.2) and float(nu[5]) == 0.5  # beta_F < 0: literal
    alive = torch.ones(6, dtype=torch.bool)
    pub, e2 = DS.specific(nu, bM, alive, 0.3, False), DS.specific(nu, bM, alive, 0.3, True)
    assert pub.tolist() == [False, False, False, True, True, False]  # nan fails; the tie at 0.3 passes (<=)
    neg = DS.specific(torch.tensor([-0.2]), torch.tensor([-0.1]), torch.ones(1, dtype=torch.bool), 0.3, True)
    assert e2.tolist() == pub.tolist() and not bool(neg[0])  # E.2 drops beta_M < 0


def test_no_latent_passing_reads_zero() -> None:
    cfg_like = type("C", (), {"nu_cut": 0.3, "nu_wide": 0.6})()
    r = {"beta_M": torch.ones(4), "beta_F": torch.ones(4)}
    out = DS.scaler_read(r, torch.ones(4, dtype=torch.bool), torch.full((4,), 0.25), cfg_like)  # type: ignore[arg-type]
    assert out["published_0.3"]["specific"] == 0.0 and out["published_0.3"]["count"] == 0


def test_parts_and_not_read() -> None:
    sae, D = toy_sae(), torch.randn(500, 8)
    r = DS.parts(sae, D)
    assert r is not None and math.isclose(float(r["s"].sum()) + r["bias"] + r["error"], 1.0, rel_tol=1e-9)
    assert DS.parts(sae, torch.zeros(10, 8)) is None


def test_capacity_of_a_plant_is_one_minus_rho() -> None:
    p, rho = torch.randn(8), 0.01
    D = torch.zeros(100, 8)
    D[::10] = p
    assert math.isclose(DS.capacity(rho * p, D), 1 - rho, rel_tol=1e-9)
    assert math.isnan(DS.capacity(p, torch.zeros(3, 8)))


def test_bar() -> None:
    assert DS.bar(0.49) is None and DS.bar(0.5) == 0.25 and DS.bar(2.0) == 0.5 and DS.bar(math.nan) is None


def test_pooled_fve_zero_variance() -> None:
    sae = toy_sae()
    assert DS.pooled_fve(sae, [torch.ones(64, 8)]) is None
    assert DS.pooled_fve(sae, [torch.randn(64, 8)]) is not None


# ---- the gate (spec 5) ----------------------------------------------------------------------------------------------


def test_gate_not_read() -> None:
    assert DS.gate([], [], 1.0)["outcome"] == "not read"  # empty pool
    assert DS.gate([0.6, None], [0.1, 0.1], 1.0)["outcome"] == "not read"  # a missing arm
    assert DS.gate([0.6, math.nan], [0.1, 0.1], 1.0)["outcome"] == "not read"
    assert DS.gate([0.6, 0.6], [0.1, 0.1], 0.49)["outcome"] == "not read"  # K1 under 0.5
    assert DS.gate([0.6, 0.6], [0.1, 0.1], math.nan)["outcome"] == "not read"


def test_gate_order_and_ties() -> None:
    b = DS.gate([0.6, 0.7], [0.1, 0.2], 0.5)
    assert b == {"outcome": "pass", "bar": 0.25}
    assert DS.gate([0.6, 0.7], [0.1, 0.25], 0.5)["outcome"] == "does not separate"  # a Gaussian at the bar
    assert DS.gate([0.3, 0.7], [0.1, 0.3], 1.0)["outcome"] == "does not separate"  # not under the lower seed (tie)
    assert DS.gate([0.5, 0.7], [0.1, 0.2], 1.0)["outcome"] == "pass"  # a seed exactly at the bar is not under it
    assert DS.gate([0.4, 0.7], [0.1, 0.2], 1.0)["outcome"] == "unstable"
    assert DS.gate([0.4, 0.45], [0.1, 0.2], 1.0)["outcome"] == "below the method's reach"
    assert DS.gate([0.0, 0.0], [0.0, 0.0], 1.0)["outcome"] == "does not separate"  # the do-nothing state


# ---- the KL patches ---------------------------------------------------------------------------------------------------


def exact_sae(d: int = 4) -> Any:
    """Decoder columns +e_i and -e_i, encoder their transpose, threshold 0 and a nonzero bias: D-hat = D exactly."""
    g = torch.Generator().manual_seed(5)
    sae: Any = BatchTopKSAE(d, 2 * d, 1, activation_mean=torch.randn(d, generator=g), activation_std=torch.rand(d, generator=g) + 1, target_rms=10.0)
    E = torch.cat([torch.eye(d), -torch.eye(d)])
    with torch.no_grad():
        sae.encoder.weight.copy_(E)
        sae.encoder.bias.zero_()
        sae.decoder.weight.copy_(E.T)
        sae.b_dec.copy_(torch.randn(d, generator=g))
        sae.threshold.fill_(0.0)
    return sae


def test_nonzero_bias_exact_reconstruction() -> None:
    sae, h = exact_sae(), paired(50, 4)
    fills = DS.patches(sae, h, {"none": torch.zeros(8, dtype=torch.bool), "every": torch.ones(8, dtype=torch.bool)})
    assert torch.allclose(fills["all"], h[:, 1], atol=1e-4) and torch.allclose(fills["error"], h[:, 0], atol=1e-4)
    assert torch.equal(fills["empty"], h[:, 0]) and torch.equal(fills["none"], h[:, 0])
    lin = torch.nn.Linear(4, 7)
    with torch.no_grad():
        logits = lambda x: lin(x)[None]  # one row, positions = tokens
        ref = logits(torch.cat([h[:1, 1], h[:, 1]]))
        a, _ = DS.kl_sums(ref, logits(torch.cat([h[:1, 1], fills["all"]])))
        e, _ = DS.kl_sums(logits(torch.cat([h[:1, 0], h[:, 0]])), logits(torch.cat([h[:1, 0], fills["error"]])))
    assert float(a.sum()) < 1e-8 and float(e.sum()) < 1e-8  # "all" gives F's logits, "error" M's


def test_recovered() -> None:
    kl = {"gap": 2.0, "all": 0.5, "h_F": 0.0}
    assert DS.recovered(kl, None) == {"gap": 0.0, "all": 0.75, "h_F": 1.0}
    assert all(v is None for v in DS.recovered(kl, 2.0).values())  # the gap at its floor: not read


# ---- the oracle's readouts and appendix A ----------------------------------------------------------------------------


def test_plant_interval() -> None:
    assert DS.plant_passes(0.0) and DS.plant_passes(3 / 7) and DS.plant_passes(-0.99)
    assert not DS.plant_passes(-1.0) and not DS.plant_passes(0.43) and not DS.plant_passes(-2.0)
    assert math.isnan(DS.nu_plant(-1.0)) and math.isclose(DS.nu_plant(3 / 7), 0.3)


def test_oracle_sae_is_the_positive_part() -> None:
    u = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    mean, std = torch.randn(8), torch.rand(8) + 0.5
    sae = DS.oracle_sae(u, mean, std, 100.0)
    x = torch.randn(20, 8)
    s = DS.scale_of(std.double(), 100.0)
    assert torch.allclose(sae.encode(x), (s * (x - mean) @ u.T).clamp_min(0), rtol=1e-5, atol=1e-4)


# ---- trainSAE as it runs (appendix B) -------------------------------------------------------------------------------


def trainer_cfg(steps: int, seed: int = 0) -> dict[str, object]:
    return {"trainer": BatchTopKTrainer, "dict_class": BatchTopKSAE, "activation_dim": 8, "dict_size": 64, "lr": 1e-3, "steps": steps,
            "device": "cpu", "warmup_steps": 10, "layer": 0, "lm_name": "toy", "wandb_name": "toy", "k": 4,
            "activation_mean": torch.zeros(8), "activation_std": torch.ones(8), "target_rms": 1.0, "seed": seed,
            "threshold_start_step": 20}  # fmt: skip


def batches(n: int, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(32, 8, generator=g) for _ in range(n)]


def test_logging_moves_theta_and_validation_moves_counters(tmp_path: Path) -> None:
    data, val = batches(60), batches(3, 1)
    keep: dict[str, object] = {}
    trainSAE(data, trainer_cfg(60), steps=60, log_steps=25, end_of_step_logging_fn=lambda tr, _s: keep.update(tr=tr),
             save_dir=str(tmp_path / "a"))  # fmt: skip
    cfg = trainer_cfg(60)
    tr = cfg.pop("trainer")(**cfg)  # type: ignore[operator]
    for i, x in enumerate(data):
        tr.update(i, x.clone())
    assert float(keep["tr"].ae.threshold) != float(tr.ae.threshold)  # type: ignore[attr-defined]
    before = tr.num_tokens_since_fired.clone()
    fired = torch.zeros(64, dtype=torch.bool)
    for x in val:
        fired |= (tr.ae.encode(tr.ae.normalize_activations(x), normalize_activations=False, use_threshold=True) > 0).any(0)
    run_validation(tr, val, step=60)
    grew = tr.num_tokens_since_fired - before
    assert bool((grew[~fired] == sum(len(x) for x in val)).all())


def test_auxk_turns_on_for_a_known_dead_set() -> None:
    cfg = trainer_cfg(40)
    tr = cfg.pop("trainer")(**cfg)  # type: ignore[operator]
    for i, x in enumerate(batches(30)):
        tr.update(i, x.clone())
    assert tr.pre_norm_auxk_loss == -1
    tr.num_tokens_since_fired[:16] = tr.dead_feature_threshold
    tr.update(30, batches(1, 9)[0])
    assert tr.pre_norm_auxk_loss != -1 and tr.dead_features > 0


def test_saved_sae_reloads_to_identical_encodings(tmp_path: Path) -> None:
    model: Any = trainSAE(batches(30), trainer_cfg(30), steps=30, save_dir=str(tmp_path))
    fresh: Any = BatchTopKSAE(8, 64, 4, activation_mean=torch.zeros(8), activation_std=torch.ones(8), target_rms=1.0)
    fresh.load_state_dict(torch.load(tmp_path / "model_final.pt", weights_only=True))
    x = batches(1, 4)[0]
    assert torch.equal(fresh.encode(x), model.encode(x))

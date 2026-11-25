"""Cell P2.37's reading on CPU, no model: the model-order check and the reproduced rule (filter, nan, counts)."""

import numpy as np
import pandas as pd
import torch

from commute.jobs import load_experiment

R = load_experiment("crosscoder_replicate", "run")
CFG = R.Config(crosscoder="", stats="", base="", chat="", layer=13, ctx=4, fineweb="f", lmsys="l", tokens_per_set=8,
               top_n=5, nu_r=0.5, nu_eps=0.2, pis=[0.2, 0.5], chunk=4)  # fmt: skip
TH = {"spearman_min": 0.9, "count_rel": 0.1}


def test_orient_reads_and_flips_a_chat_first_file() -> None:
    torch.manual_seed(0)
    W = torch.rand(2, 6, 3)
    sd = {"decoder.weight": W, "decoder.bias": torch.arange(6.0).view(2, 3), "encoder.weight": torch.rand(2, 3, 6)}
    base, chat = W[1].norm(dim=1).double(), W[0].norm(dim=1).double()  # the file holds the chat model first
    out, order, miss = R.orient(sd, base, chat)
    assert order == "chat_first" and miss < 1e-6
    assert torch.equal(out["decoder.weight"][0], W[1]) and torch.equal(out["decoder.bias"][0], sd["decoder.bias"][1])
    assert torch.equal(out["encoder.weight"][1], sd["encoder.weight"][0])
    _, order2, _ = R.orient(out, base, chat)
    assert order2 == "base_first"


def frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eb, ec, rb, rc = rng.uniform(0, 1, n), rng.uniform(0.5, 1.5, n), rng.uniform(0, 2, n), rng.uniform(1, 3, n)
    return pd.DataFrame({"beta_error_base": eb, "beta_error_chat": ec, "beta_ratio_error": eb / ec,
                         "beta_reconstruction_base": rb, "beta_reconstruction_chat": rc,
                         "beta_ratio_reconstruction": rb / rc, "dec_norm_diff": rng.uniform(0, 0.4, n)})  # fmt: skip


def ours_from(t: pd.DataFrame) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(t[c].to_numpy()) for k, c in R.COLS.items()}


def test_identical_numbers_reproduce_and_nan_or_negative_latents_are_dropped() -> None:
    t = frame(200, 0)
    r = R.reading(ours_from(t), t, CFG, TH)
    assert r["reproduced"] and r["spearman"]["nu_r"] > 0.999 and r["count_ours"] == r["count_theirs"] > 0
    b = ours_from(t)
    b["nu_eps"][:3] = float("nan")  # never fired
    b["r_base"][3] = -0.1  # the paper's filter drops it, on our side only
    r2 = R.reading(b, t, CFG, TH)
    assert r2["nan"] == 3 and r2["kept_ours"] == 196 and r2["kept_both"] == 196


def test_unrelated_numbers_and_an_empty_pool_do_not_reproduce() -> None:
    t, other = frame(200, 0), frame(200, 1)
    assert not R.reading(ours_from(other), t, CFG, TH)["reproduced"]
    b = ours_from(t)
    b["eps_base"][:] = -1.0  # every latent filtered out: Spearman nan, count 0
    r = R.reading(b, t, CFG, TH)
    assert np.isnan(r["spearman"]["nu_r"]) and r["count_ours"] == 0 and not r["reproduced"]

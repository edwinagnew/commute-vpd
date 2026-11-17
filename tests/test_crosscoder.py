"""E203's glue against the definitions it ports: latent scaling against a least-squares fit on the toolkit's materialised
targets, raw-space decoders against their decode, and the change parts summing to one. A tiny crosscoder on CPU."""

import torch
from dictionary_learning.dictionary import BatchTopKCrossCoder

from commute import crosscoder as CC


def tiny() -> tuple[CC.AE, torch.Tensor]:
    torch.manual_seed(0)
    d, D, n = 8, 32, 512
    h = torch.randn(n, 2, d) * 3 + 1
    h[:, 1] += 0.5 * torch.randn(n, d)  # a fine-tune side that differs
    mean, std = CC.normaliser(h)
    ae: CC.AE = BatchTopKCrossCoder(d, D, 2, k=4, activation_mean=mean, activation_std=std, target_rms=100.0,
                             same_init_for_all_layers=True, norm_init_scale=1.0)  # fmt: skip
    with torch.no_grad():
        ae.decoder.weight.add_(
            0.3 * torch.randn_like(ae.decoder.weight)
        )  # unequal sides
        ae.decoder.bias.normal_()
        ae.threshold.fill_(0.0)
    return ae, h


def test_betas_match_least_squares_on_materialised_targets() -> None:
    ae, h = tiny()
    J = torch.arange(ae.dict_size)
    b = CC.betas(ae, h, J, size=100)
    with torch.no_grad():
        xn = ae.normalize_activations(h.clone())
        f = ae.encode(xn, normalize_activations=False)
        rec = ae.decode(f, denormalize_activations=False)
        W = ae.decoder.weight
        for j in range(ae.dict_size):
            if not bool((f[:, j] > 0).any()):
                continue
            d = W[1, j]
            targets = {"eps_base": xn[:, 0] - (rec[:, 0] - f[:, j, None] * W[0, j]),  # toolkit utils.load_base_error
                       "eps_ft": xn[:, 1] - (rec[:, 1] - f[:, j, None] * d), "r_base": rec[:, 0], "r_ft": rec[:, 1]}  # fmt: skip
            for k, y in targets.items():
                A = (f[:, j, None] * d).reshape(-1, 1).double()
                beta = torch.linalg.lstsq(A, y.reshape(-1, 1).double()).solution.item()
                assert abs(beta - float(b[k][j])) < 1e-4 * max(1.0, abs(beta)), (
                    k,
                    j,
                    beta,
                    float(b[k][j]),
                )


def test_raw_decoder_reproduces_decode_and_parts_sum_to_one() -> None:
    ae, h = tiny()
    with torch.no_grad():
        f = ae.encode(h)
        Wr, br = CC.raw(ae)
        assert torch.allclose(
            ae.decode(f), torch.einsum("nk,mkd->nmd", f, Wr) + br, atol=1e-4
        )
    ch = CC.carried(ae, h, size=100)
    rn = CC.relative_norm(ae)
    alive = torch.ones(ae.dict_size, dtype=torch.bool)
    g = CC.groups(CC.Cfg(d=8, expansion=4, k=1, k_max=2, k_anneal_steps=1, lr=1e-4, warmup=1, batch=2, tokens=100,
                         epochs=1, aux_alpha=0.03, dead_tokens=10, init_norm=1.0, target_rms=100.0, chunk_rows=1,
                         ls_tokens=10, val_rows=1, alive_min=1, top_n=4, top_n_wide=8, nu_r=0.5, nu_eps=0.2,
                         shared=(0.4, 0.6), base_only=0.1, snapshots=(0.5,), ckpt_every=10),
                  rn, alive, CC.top(rn, 4), torch.zeros(ae.dict_size, dtype=torch.bool))  # fmt: skip
    total = torch.zeros(ae.dict_size, dtype=torch.int)
    for m in g.values():
        total += m.int()
    assert bool((total == 1).all())
    p = CC.parts(ch["s"], ch["bias"], g)
    assert abs(sum(p.values()) - 1) < 1e-9
    with torch.no_grad():
        rec = ae.decode(ae.encode(h))
        D, Dh = h[:, 1] - h[:, 0], rec[:, 1] - rec[:, 0]
        want = 1 - float((D - Dh).pow(2).sum() / D.pow(2).sum())
    assert abs(ch["fve_change"] - want) < 1e-4 * max(1.0, abs(want))

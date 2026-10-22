"""saved_states.offsets_state rebuilds a state saved by arms.save_offsets (CPU, toy sizes)."""

from pathlib import Path

import torch

from commute import arms, saved_states


def test_offsets_state_rebuilds_delta_w(tmp_path: Path) -> None:
    g = torch.Generator().manual_seed(0)
    uv = {k: (torch.randn(6, 4, generator=g), torch.randn(5, 4, generator=g)) for k in ("a", "b")}
    st = {}
    for k, (U, V) in uv.items():
        U2, V2 = U + 0.01 * torch.randn(U.shape, generator=g), V + 0.01 * torch.randn(V.shape, generator=g)
        st[k] = (U2 @ V2.T - U @ V.T, U2, V2)
    path = tmp_path / "s.pt"
    arms.save_offsets(path, uv, st, {"step": 1})
    dW, uv2 = saved_states.offsets_state(path, uv)
    assert set(dW) == set(uv2) == set(uv)
    for k, (U, V) in uv.items():
        U2, V2 = uv2[k]
        torch.testing.assert_close(dW[k], U2 @ V2.T - U @ V.T, rtol=0, atol=1e-6)
        # the offsets are bf16: the rebuilt pieces sit within bf16 rounding of the trained ones
        torch.testing.assert_close(U2, st[k][1], rtol=0, atol=1e-4)
        torch.testing.assert_close(V2, st[k][2], rtol=0, atol=1e-4)
        torch.testing.assert_close(U2 - U, (st[k][1] - U).bfloat16().float(), rtol=0, atol=1e-6)

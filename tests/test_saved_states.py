"""saved_states.offsets_state rebuilds a state saved by arms.save_offsets (CPU, toy sizes)."""

from pathlib import Path

import pytest
import torch

from commute import arms, saved_states


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_offsets_state_rebuilds_delta_w(tmp_path: Path, dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(0)
    uv = {k: (torch.randn(6, 4, generator=g), torch.randn(5, 4, generator=g)) for k in ("a", "b")}
    st = {}
    for k, (U, V) in uv.items():
        U2, V2 = U + 0.01 * torch.randn(U.shape, generator=g), V + 0.01 * torch.randn(V.shape, generator=g)
        st[k] = (U2 @ V2.T - U @ V.T, U2, V2)
    path = tmp_path / "s.pt"
    if dtype == torch.float32:
        arms.save_offsets(path, uv, st, {"step": 1})  # the default (friction 605, 615)
    else:
        arms.save_offsets(path, uv, st, {"step": 1}, dtype=dtype)
    assert torch.load(path, weights_only=False)["offsets_dtype"] == str(dtype)
    dW, uv2 = saved_states.offsets_state(path, uv)
    assert set(dW) == set(uv2) == set(uv)
    for k, (U, V) in uv.items():
        U2, V2 = uv2[k]
        torch.testing.assert_close(dW[k], U2 @ V2.T - U @ V.T, rtol=0, atol=1e-6)
        # the rebuilt pieces sit within the saved dtype's rounding of the trained ones: exact up to fp32 sums by default
        torch.testing.assert_close(U2, st[k][1], rtol=0, atol=1e-6 if dtype == torch.float32 else 1e-4)
        torch.testing.assert_close(V2, st[k][2], rtol=0, atol=1e-6 if dtype == torch.float32 else 1e-4)
        torch.testing.assert_close(U2 - U, (st[k][1] - U).to(dtype).float(), rtol=0, atol=1e-6)

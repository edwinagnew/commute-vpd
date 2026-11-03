"""vpd.checkpoint_weights on a tiny file in the released checkpoint's layout; no model is built. That it equals the loaded
model's `target_weights` and `component_weights` on the real file was checked once on a GPU (commit message, friction 173)."""

from __future__ import annotations

import torch

from commute import vpd


def test_checkpoint_weights_names_and_shapes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(vpd.env, "VPD_DIR", tmp_path)
    path = tmp_path / vpd.DECOMP_RUN / vpd.DECOMP_FILES[0]
    path.parent.mkdir()
    w, u, v = torch.rand(2, 3), torch.rand(5, 2), torch.rand(3, 5)
    torch.save({"target_model.h.0.mlp.down_proj.weight": w, "target_model.ln_f.weight": torch.rand(3),
                "_components.h-0-mlp-down_proj.U": u, "_components.h-0-mlp-down_proj.V": v}, path)
    W0, uv = vpd.checkpoint_weights()
    assert list(W0) == list(uv) == ["h.0.mlp.down_proj"]
    assert W0["h.0.mlp.down_proj"].equal(w)
    assert uv["h.0.mlp.down_proj"][0].equal(u) and uv["h.0.mlp.down_proj"][1].equal(v)

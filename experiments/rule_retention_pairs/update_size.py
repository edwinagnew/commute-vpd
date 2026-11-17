"""E196 side check, on the weights only (no model, no GPU): how large is each item's weight update, per layer and kind?
For the filed rule's blind spot "the size of the weight update" (results/predictions.json "blind_to"): the reading
might order each pair by update size rather than by rules kept. Writes results/update_size.json, per item:
    mlp[l], attn[l]  relative Frobenius norm |dW| / |W0| of layer l's MLP (c_fc, down_proj) and attention (q, k, v, o)
    mlp, attn        the same over all four layers together
    other            for the two full saved states (parity twin, memorised fact) only: |d| / |W0| of every parameter
                     outside the 24 decomposed matrices (embeddings, norms), against M's
The old tasks' saved delta-W holds only the 24 decomposed matrices, so their embeddings and norms did not move.

    ~/commute-agents/fleet/cpu-run --mem 6G python experiments/rule_retention_pairs/update_size.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import env, vpd
from commute import old_tasks as OT
from commute.jobs import load_experiment

X: Any = load_experiment("rule_retention_pairs", "run")  # PAIRS, STATES, RESULTS
LAYERS = 4
MLP, ATTN = ("mlp.c_fc", "mlp.down_proj"), ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj")


def rel(d: dict[str, Tensor], W0: dict[str, Tensor], keys: list[str]) -> float:
    return float(sum(d[k].double().pow(2).sum() for k in keys) ** 0.5 / sum(W0[k].double().pow(2).sum() for k in keys) ** 0.5)


def sizes(d: dict[str, Tensor], W0: dict[str, Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind, mats in (("mlp", MLP), ("attn", ATTN)):
        out[kind + "_by_layer"] = [rel(d, W0, [f"h.{l}.{m}" for m in mats]) for l in range(LAYERS)]
        out[kind] = rel(d, W0, [f"h.{l}.{m}" for l in range(LAYERS) for m in mats])
    return out


def delta(name: str, seed: int, W0: dict[str, Tensor], target: dict[str, Tensor]) -> tuple[dict[str, Tensor], float | None]:
    """The item's update on the 24 matrices, and for a full state the relative change of every other parameter."""
    if name in X.STATES:
        sd = torch.load(Path(str(X.STATES[name]).format(seed)), map_location="cpu", weights_only=True)
        d = {k: sd[f"{k}.weight"].double() - W0[k].double() for k in W0}
        rest = [k for k in sd if k.removesuffix(".weight") not in W0]
        num = sum((sd[k].double() - target[f"target_model.{k}"].double()).pow(2).sum() for k in rest)
        den = sum(target[f"target_model.{k}"].double().pow(2).sum() for k in rest)
        return d, float(num**0.5 / den**0.5)
    dw = torch.load(OT.TASKS[name].states[seed], map_location="cpu", weights_only=True)
    return {k: v.double() for k, v in dw.items()}, None


def main() -> None:
    W0, _ = vpd.checkpoint_weights()
    target = torch.load(env.VPD_DIR / vpd.DECOMP_RUN / vpd.DECOMP_FILES[0], map_location="cpu", weights_only=True, mmap=True)
    out: dict[str, Any] = {"experiment": "E196", "what": (__doc__ or "").split("\n\n")[0], "items": {}}
    for pair in X.PAIRS:
        for name in pair:
            for s in (0, 1):
                d, other = delta(name, s, W0, target)
                out["items"][f"{name}_s{s}"] = sizes(d, W0) | ({"other": other} if other is not None else {})
                print(name, s, out["items"][f"{name}_s{s}"], flush=True)
    (X.RESULTS / "update_size.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

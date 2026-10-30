"""E52's target-model reads and its steering baseline (B3), shared since 2026-10-30 with E54
(`experiments/keydependent_elicit`). Moved, not changed, from `experiments/baseline_neurons/run.py`, which (with its
`amend1.py`) now imports them from here.

The target model only (no decomposition is read), fp32. A key -> value task's held-out windows are split into a fit half
(even indices) and a scoring half (odd). Per state the gain G of log P(target) at the answer position over W0; a
method's share is its change of log P over W0 divided by G.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute.faithfulness import model_state

N_LAYERS, D_MLP = 4, 3072
SITES = [f"in{i}" for i in range(N_LAYERS)] + [f"mlp{i}" for i in range(N_LAYERS)] + ["out3"]
Edit = dict[str, Callable[[Tensor], Tensor]]
DW = dict[str, Tensor]


class Net:
    """The target model at W0 or W0 + delta-W. `run` edits and reads three kinds of site: `in<l>` block l's input,
    `mlp<l>` block l's MLP neurons (GELU output = down_proj's input), `out3` the last block's output."""

    def __init__(self, device: str) -> None:
        from commute import vpd

        self.model, self.device = vpd.load_component_model(device), device
        self.net = self.model.target_model
        self.W0 = vpd.target_weights(self.model)
        assert all(p.dtype == torch.float32 for p in self.net.parameters()) and len(self.net.h) == N_LAYERS

    def at(self, dW: DW | None) -> AbstractContextManager[Any]:
        if dW is None:
            return nullcontext()
        assert sorted(dW) == sorted(self.W0), "delta-W must cover the 24 matrices"
        return model_state(self.model, target={k: w + dW[k].to(w.device, w.dtype) for k, w in self.W0.items()})

    @torch.no_grad()
    def run(self, toks: Tensor, edit: Edit | None = None, capture: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        edit, seen, handles = edit or {}, {}, []

        def f(site: str, x: Tensor) -> Tensor:
            x = edit[site](x) if site in edit else x
            if capture:
                seen[site] = x
            return x

        for site in (s for s in SITES if s in edit or capture):
            block = self.net.h[int(site[-1])]
            if site.startswith("out"):
                handles.append(block.register_forward_hook(lambda _m, _a, o, s=site: f(s, o)))
            else:
                mod = block.mlp.down_proj if site.startswith("mlp") else block
                handles.append(mod.register_forward_pre_hook(lambda _m, a, s=site: (f(s, a[0]),)))
        try:
            out = self.net(toks.to(self.device))
        finally:
            for h in handles:
                h.remove()
        return (out[0] if isinstance(out, tuple) else out).float(), seen


def logp_at(logits: Tensor, pos: int, tgt: Tensor) -> Tensor:
    return torch.log_softmax(logits[:, pos], -1).gather(1, tgt.to(logits.device)[:, None]).squeeze(1).double().cpu()


@dataclass
class Task:
    name: str
    kind: str
    toks: Tensor  # (N, 41) held-out windows
    table: Tensor  # (V,) key -> value, -1 off the keys
    keys: tuple[int, ...]
    marker: int
    half: int
    lp0: Tensor  # (N,) W0's log P(target) at the answer position

    def part(self, which: str) -> slice:
        return slice(0, None, 2) if which == "fit" else slice(1, None, 2)

    def tgt(self, toks: Tensor) -> Tensor:
        return self.table[toks[:, self.half]]


def load_task(net: Net, path: Path, name: str, kind: str) -> Task:
    """A kv windows file ({"heldout", "task", "table"}) as a Task, with W0's log P(target) on its held-out windows."""
    w = torch.load(path, weights_only=False)
    held, task = w["heldout"], w["task"]
    t = Task(name, kind, held.tokens, w["table"], tuple(task.keys), int(task.marker), held.half, torch.empty(0))
    assert len(t.toks) >= 4 and bool((t.toks[:, t.half - 1] == t.marker).all()) and bool((t.tgt(t.toks) >= 0).all()), name
    t.lp0 = logp_at(net.run(t.toks)[0], t.half, t.tgt(t.toks))
    return t


def read(net: Net, t: Task, toks: Tensor, dW: DW | None = None, edit: Edit | None = None) -> Tensor:
    with net.at(dW):
        return logp_at(net.run(toks, edit)[0], t.half, t.tgt(toks))


def share(lp: Tensor, lp0: Tensor, G: float) -> float:
    return float((lp - lp0).mean()) / G if G != 0 else float("nan")


def gains(net: Net, t: Task, dW: DW) -> dict[str, float]:
    d = read(net, t, t.toks, dW) - t.lp0
    return {"G": float(d.mean()), "G_fit": float(d[t.part("fit")].mean()), "G_score": float(d[t.part("score")].mean())}


def b3(net: Net, t: Task, dW: DW, g: dict[str, float]) -> dict[str, Any]:
    """Steering: v_l = mean over the fit half of (tuned - W0) block-l input at the answer position, added to W0 there;
    the layer with the largest fit-half share (ties to the lower), S = its scoring-half share."""
    f, s = t.toks[t.part("fit")], t.toks[t.part("score")]
    with net.at(dW):
        _, c1 = net.run(f, capture=True)
    _, c0 = net.run(f, capture=True)
    layers = {}
    for i in range(N_LAYERS):
        v = (c1[f"in{i}"][:, t.half] - c0[f"in{i}"][:, t.half]).mean(0)

        def add(x: Tensor, v: Tensor = v) -> Tensor:
            x = x.clone()
            x[:, t.half] += v
            return x

        layers[i] = {"norm": float(v.norm()), "fit": share(read(net, t, f, edit={f"in{i}": add}), t.lp0[t.part("fit")], g["G_fit"]),
                     "score": share(read(net, t, s, edit={f"in{i}": add}), t.lp0[t.part("score")], g["G_score"])}
    best = max(range(N_LAYERS), key=lambda i: (layers[i]["fit"], -i))
    return {"layers": layers, "layer": best, "S": layers[best]["score"]}

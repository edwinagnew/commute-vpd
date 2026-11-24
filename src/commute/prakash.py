"""Prakash et al. (ICLR 2024)'s released code, run unchanged on the 67M (cell P2.68, spec
reports/plans/2026-11-23_plan_p2-68-cmap-spec.md Appendix F).

`load` imports their modules from a pinned, read-only checkout of https://github.com/Nix07/finetuning at 6d6c098 (outside
the repo: it has no licence, so nothing of theirs is copied in) by file path, after checking each file's sha256 against
the values filed in the run's predictions. Their own module-level `sys.path.append` (pp_utils, cmap_utils) is theirs and
runs as written; nothing here edits sys.path. `Adapter` shows our target model to their functions as the
`LlamaForCausalLM` they expect: `model.layers.{l}.self_attn.{q,k,v,o}_proj` are our `h.{l}.attn.*` modules themselves (so
a hook their TraceDict puts on one fires in our forward), `config` holds the three fields they read, the forward returns
`.logits`, and `state_dict()` is the aliased modules' real weights (their `cmap_out` and `patching_sender_heads` read W_O
from it). Needs the `cmap` dependency group (baukit, peft)."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from torch import Tensor, nn

# The files whose code runs: their three modules and the data module pp_utils and cmap_utils import at the top.
FILES = ("experiment_1/pp_utils.py", "experiment_1/path_patching.py", "experiment_3/cmap_utils.py", "data/data_utils.py")


@dataclass(frozen=True)
class Authors:
    pp: ModuleType  # experiment_1/pp_utils.py
    path: ModuleType  # experiment_1/path_patching.py
    cmap: ModuleType  # experiment_3/cmap_utils.py


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # path_patching's `from pp_utils import ...` finds it here
    spec.loader.exec_module(mod)
    return mod


def load(root: Path, hashes: dict[str, str]) -> Authors:
    """Their modules from the checkout at `root`; exits naming the file if any sha256 differs from `hashes`."""
    got = {f: sha256(root / f) for f in FILES}
    bad = {f: h for f, h in got.items() if h != hashes[f]}
    if bad:
        sys.exit(f"the authors' checkout at {root} is not the pinned one: {bad}")
    os.environ.setdefault("TQDM_DISABLE", "1")  # their tqdm bars, once per apply_pp call, would fill the log
    with quiet():
        pp = _module("pp_utils", root / FILES[0])
        return Authors(pp, _module("path_patching", root / FILES[1]), _module("cmap_utils", root / FILES[2]))


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Their functions print a line per call (get_caches: "CLEAN CACHE COMPUTED"); keep them out of the job log."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class Adapter(nn.Module):
    """Our LlamaSimpleMLP under the module names and fields their code reads. Only the aliased attention modules are
    registered, so `state_dict()` holds exactly their keys; the whole net is kept off the module tree."""

    net: nn.Module

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        blocks: Any = net.h
        layers = nn.ModuleList()
        for block in blocks:
            layer = nn.Module()
            layer.self_attn = block.attn
            layers.append(layer)
        self.model = nn.Module()
        self.model.layers = layers
        object.__setattr__(self, "net", net)
        self.config = SimpleNamespace(num_attention_heads=blocks[0].attn.n_head, num_hidden_layers=len(blocks),
                                      architectures=["LlamaForCausalLM"])  # fmt: skip

    @property
    def device(self) -> Any:
        return next(self.parameters()).device

    def forward(self, tokens: Tensor) -> SimpleNamespace:
        assert not any(m.training for m in self.net.modules()), "the model must be in eval mode"  # apply_pp never sets it
        return SimpleNamespace(logits=self.net(tokens)[0])


def o_proj(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj"


def proj(layer: int, which: str) -> str:
    return f"model.layers.{layer}.self_attn.{which}_proj"

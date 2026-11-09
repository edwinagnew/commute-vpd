"""E137's small shared helpers: logging, stage files and the forwards every stage reads through."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pieces as PC
import torch
from torch import Tensor

from commute.hidden_maps import chunks, run
from commute.steering import Net

PRED = PC.HERE / "results" / "predictions.json"
_start = time.monotonic()


class MoreToDo(Exception):
    pass


def elapsed() -> float:
    return time.monotonic() - _start


def log(msg: str) -> None:
    print(f"[{elapsed():7.0f}s] {msg}", flush=True)


def path(cfg: PC.Config, name: str) -> Any:
    return cfg.results / name


def dump(cfg: PC.Config, name: str, out: dict[str, Any]) -> None:
    p = path(cfg, name)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    os.replace(tmp, p)


def free() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(cfg: PC.Config, name: str) -> dict[str, Any] | None:
    p = path(cfg, name)
    return json.loads(p.read_text()) if p.is_file() else None


def last_logits(net: Net, windows: Tensor, batch: int, edit_fn: Any = None) -> Tensor:
    out = [run(net, t, edit=edit_fn(t) if edit_fn else None)[0][:, -1] for t in chunks(windows, batch)]
    return torch.cat(out)


def f_edit_of(net: Net, p: PC.Pieces, pair: str) -> Any:
    return lambda toks: PC.piece_edit(net, p, pair, toks)

"""Pile rows and emoticon windows.

Rows come from `danbraunai/pile-uncopyrighted-tok-shuffled` (the authors' tokenised Pile, 513 ids
per row, cut to 512 as their loader cuts them), read in stream order and cached as .npy under the
cache dir. Emoticon windows are the paper's LoRA examples: the token on which subcomponent
`h.2.mlp.down_proj:2359` is causally important and the 20 tokens either side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import env
from commute.vpd import SEQ_LEN, compute_ci, parse_key

DATASET = "danbraunai/pile-uncopyrighted-tok-shuffled"
SETS_DIR = env.CACHE_DIR / "sets"


def pile_rows(split: str, start: int, n: int, log: Any = print) -> Tensor:
    """Rows [start, start + n) of the split as int64 (n, 512). Cached on first use."""
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    path = SETS_DIR / f"{split}_{start}_{n}.npy"
    if path.is_file():
        return torch.from_numpy(np.load(path)).long()
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split, streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if i < start:
            continue
        ids = np.asarray(ex["input_ids"], dtype=np.int64)[:SEQ_LEN]
        assert ids.shape == (SEQ_LEN,)
        rows.append(ids)
        if len(rows) == n:
            break
    arr = np.stack(rows)
    np.save(path, arr)
    log(f"[data] {split} rows {start}..{start + n}: cached at {path}")
    return torch.from_numpy(arr).long()


def batches(rows: Tensor, size: int) -> list[Tensor]:
    return [rows[s : s + size] for s in range(0, rows.shape[0], size)]


def target_tokens(token: int | Tensor, toks: Tensor, half: int) -> Tensor:
    """The target token of each window, (B,) int64 on `toks`' device: one id for every window, or a (V,) lookup table
    read at the token at `half` (the key -> value tasks of experiments/kv_task; `kv.KVTask.table`)."""
    if isinstance(token, int):
        return torch.full((toks.shape[0],), token, device=toks.device, dtype=torch.long)
    return token.to(toks.device)[toks[:, half]]


@dataclass
class Windows:
    tokens: Tensor  # (N, 2 * half + 1) int64; the causally-important token sits at index `half`
    g: Tensor  # (N,) the subcomponent's causal importance there
    row: Tensor  # (N,) source row index
    pos: Tensor  # (N,) position within the row
    half: int

    def subset(self, mask: Tensor) -> Windows:
        return Windows(self.tokens[mask], self.g[mask], self.row[mask], self.pos[mask], self.half)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def emoticon_windows(model: Any, rows: Tensor, key: str, threshold: float, half: int = 20, chunk: int = 16) -> Windows:
    """Every (row, position) at which subcomponent `key` has g > threshold, with its window."""
    name, idx = parse_key(key)
    toks, gs, rs, ps = [], [], [], []
    for s in range(0, rows.shape[0], chunk):
        sub = rows[s : s + chunk]
        ci = compute_ci(model, sub, chunk=chunk)
        g = ci.g[name][..., idx].float().cpu()  # (b, T)
        b_idx, t_idx = torch.nonzero(g > threshold, as_tuple=True)
        for b, t in zip(b_idx.tolist(), t_idx.tolist()):
            if t < half or t + half >= SEQ_LEN:
                continue
            toks.append(sub[b, t - half : t + half + 1])
            gs.append(g[b, t])
            rs.append(s + b)
            ps.append(t)
        del ci
    if not toks:
        return Windows(torch.zeros(0, 2 * half + 1, dtype=torch.long), torch.zeros(0), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), half)
    return Windows(torch.stack(toks), torch.stack(gs), torch.tensor(rs), torch.tensor(ps), half)


def split_windows(w: Windows, heldout_every: int = 5) -> tuple[Windows, Windows]:
    """Held-out = windows from every `heldout_every`-th source row (split by row, not by window)."""
    held = (w.row % heldout_every) == 0
    return w.subset(~held), w.subset(held)

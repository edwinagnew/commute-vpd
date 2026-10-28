"""Key -> value windows (experiments/kv_task, E10) in the shared harness, and how much of a key -> value update shows on
plain Pile text. Added 2026-10-28 for E42 (experiments/teacher_training/newskill.py; plan
reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md).

A kv windows file (`kv_task/run.py`'s build stage) holds {"train", "heldout": data.Windows, "task": kv.KVTask, "table":
(V,) lookup}. Its target is the table, read at the key at `half` (`data.target_tokens`), where the ladder files hold one
target token; everything downstream of `harness.with_windows` already takes either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from commute.finetune import logits_of
from commute.kv import KVTask


def task_context(ctx: Any, windows_path: Path) -> tuple[Any, KVTask]:
    """`harness.with_windows` on a kv windows file, its table as the target, and the task (marker, keys, values)."""
    from commute import harness

    w = torch.load(windows_path, weights_only=False)
    task, table = w["task"], w["table"]
    assert isinstance(task, KVTask) and table.ndim == 1, windows_path
    assert bool((w["heldout"].tokens[:, w["heldout"].half - 1] == task.marker).all()), "held-out windows without the marker"
    return harness.with_windows(ctx, w, table), task


@torch.no_grad()
def pile_reading(nets: dict[str, nn.Module], rows: Tensor, task: KVTask, device: str, batch: int = 8) -> dict[str, Any]:
    """What a key -> value update shows on plain text: per net, the mean probability of table[x] as the next token at every
    position holding a key x (no marker before it), over `rows`; and how often the marker, and the marker followed by a
    key, occur in `rows`. A net that has not learnt the map outside the marker reads its W0 value."""
    keys = torch.tensor(task.keys, device=device)
    lookup = torch.zeros(max(int(rows.max()), *task.keys) + 1, dtype=torch.long, device=device)
    lookup[keys] = torch.tensor(task.values, device=device)
    sums = {n: 0.0 for n in nets}
    n_key = 0
    for s in range(0, rows.shape[0], batch):
        toks = rows[s : s + batch].to(device)
        hit = torch.isin(toks[:, :-1], keys)
        if not bool(hit.any()):
            continue
        tgt = lookup[toks[:, :-1]][hit]  # (n,) the value of each key occurrence
        for name, net in nets.items():
            lp = torch.log_softmax(logits_of(net, toks)[:, :-1], -1)[hit]  # (n, V)
            sums[name] += float(lp.gather(1, tgt[:, None]).exp().sum())
        n_key += int(hit.sum())
    marker = rows == task.marker
    after = torch.isin(rows[:, 1:], keys.cpu()) & marker[:, :-1]
    return {"key_positions": n_key, "marker_positions": int(marker.sum()), "marker_then_key": int(after.sum()),
            **{f"p_value_{n}": (v / n_key if n_key else None) for n, v in sums.items()}}

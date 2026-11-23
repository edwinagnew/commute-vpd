"""E273 stage "base loss" (cell P2.64, descriptive; whip-lead 2026-11-23 04:35): the base model M's loss on each
real-text side's own read rows (E273's D, before the cut), at every position, nats per token and per UTF-8 byte. It completes the E273 page's
unfamiliarity check: E272's base loss read French and Yoruba on other rows only, and lowercase vs reversed order not
at all. Expected: lowercase low (English, lowercased), reversed order far higher; Yoruba above French per byte (E272's
other rows: 2.15 against 1.02). Control: the mean log P of the scored token at the read position, as a geometric mean,
equals E246's saved prior on the same rows (difference to float noise; reported, not a stop).
-> results/base_loss.json. A rerun skips written tasks.

    sbatch --comment="E273 stage: base loss" cluster/realtext_eb_base_loss.sbatch
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch
from torch import Tensor

from commute import panel as PN
from commute import steering as ST
from commute import vpd
from commute.jobs import hard_exit, load_experiment, write_stage

RUN: Any = load_experiment("realtext_eb", "run")
E272: Any = load_experiment("convergence_panel", "base_loss")  # byte_lengths: UTF-8 bytes of each token id
OUT = RUN.RESULTS / "base_loss.json"
TASKS = ("L1", "L3", "T1", "T4", "T2", "T3")


def read_logp(net: ST.Net, rows: Tensor, pos: int) -> float:
    """Mean log P at W0 of the token after the read position (E246's prior is exp of this)."""
    mask = torch.zeros_like(rows, dtype=torch.bool)
    mask[:, pos + 1] = True
    return PN.target_logp(net, rows, mask, None)


def task_loss(net: ST.Net, rows: Tensor, eos: int, nbytes: Tensor) -> dict[str, float]:
    """Every non-end-of-text token after position 0."""
    mask = rows != eos
    n, nats = int(mask[:, 1:].sum()), -PN.target_logp(net, rows, mask, None)
    b = int(nbytes[rows[:, 1:][mask[:, 1:]]].sum())
    return {"rows": len(rows), "tokens": n, "bytes": b, "nats_per_token": nats, "nats_per_byte": nats * n / b}


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, tok = ST.Net("cuda"), vpd.tokenizer()
    nbytes = E272.byte_lengths(tok)
    texts, prior = torch.load(RUN.TEXTS, weights_only=False)["reads"], RUN.e246()["prior"]
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"experiment": "E273", "stage": "base loss", "card": torch.cuda.get_device_name(), "tasks": {}}
    for task in (t for t in TASKS if t not in out["tasks"]):
        rows, pos = texts[task].long(), prior[task]["pos"]
        rec: dict[str, Any] = task_loss(net, rows, tok.eos_token_id, nbytes)
        lp = read_logp(net, rows, pos)
        rec["control"] = {"pos": pos, "read_logp": lp, "e246_log_prior": math.log(prior[task]["prior"]), "diff": lp - math.log(prior[task]["prior"])}
        out["tasks"][task] = rec
        write_stage(OUT, out)
        print(f"[base_loss {task}] {json.dumps(rec)}", flush=True)
    print(f"STAGE base_loss DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    hard_exit(main())

"""E90 Amendment 1, before stage 2, controls first and alone. Filing: results/predictions_amend1.json. The definition is unchanged;
all of this is apparatus, and stage 1's readings stay as read.
  1. Token identity's control. Stage 1 read W0's log-odds, at the key's own position, that the next token is the key again. W0 does
     not output the token it stands on, and removing the directions raised that read. This reads the fall in W0's log-odds of its own
     top-1 next token at the same windows, same removal position, same bases, same random draws, same `rules.teeth`.
  2. R2 as filed, if the control has teeth, with stage 1's reading beside it.
  3. I1's wider key set: `input_dep.trigger_table`'s `keep_answer`.
Writes results/amend1.json after each step. Usage: sbatch cluster/new_vs_reuse.sbatch amend1, then stage2.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import apparatus as AP
import input_dep as ID
import reading as RD
import rules
import stage1 as S1
import torch
from apparatus import LARGE, RESULTS, dump, log
from reading import Basis, Job
from torch import Tensor

from commute import removal as RM
from commute import steering as ST
from commute.data import pile_rows
from commute.jobs import hard_exit

OUT, FIRST = RESULTS / "amend1.json", RESULTS / "stage1.json"


def null_passes(drops: Sequence[float]) -> int:
    """How many random draws pass `rules.teeth` against the band of the other draws: the rule's false-pass count at this site."""
    return sum(rules.teeth(d, RM.band_of([*drops[:i], *drops[i + 1 :]])) for i, d in enumerate(drops))


def identity_control(j: Job, rows: Tensor) -> tuple[Basis, dict[str, bool]]:
    """Stage 1's X_C, held rows and saved bases; only the read's target changes. Then the do-nothing state on it (R1)."""
    keys = torch.tensor(list(torch.load(S1.REG / "windows_ident32.pt", weights_only=False)["task"].keys))
    c = AP.identity_capability(j.net, "token identity", rows, keys, S1.HALF, j.S, j.S["n_control"])
    c.read = AP.own_top1(j.net, c.read)
    f = RD.with_random(j, torch.load(LARGE / "bases.pt", weights_only=True, map_location="cpu")["identity"], 2)
    on = RD.control(j, c, f)
    rec = j.res["control"][c.name]
    for b in f.bases:
        rec[str(b)]["null_passes"] = null_passes(rec[str(b)]["random"])
        log(f"[control {c.name} block {b}] random draws passing against the others' band: {rec[str(b)]['null_passes']} of {len(f.randoms[b])}")
    RD.do_nothing(j, c, f, on, 1)
    dump(OUT, j.res)
    return f, on


def main() -> None:
    S = AP.filed()["settings"]
    first = json.loads(FIRST.read_text()) if FIRST.exists() else {}
    if not first.get("complete"):
        log("stage 1 has not completed: nothing runs (exit 1)")
        hard_exit(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    res: dict[str, Any] = {"sha256": {}}
    j = Job(ST.Net("cuda"), S, AP.records(), res, OUT)
    rows = pile_rows(*S["pile"]["rows"], log=log)
    f, on = identity_control(j, rows)
    readings: dict[str, Any] = res.setdefault("readings", {})
    rows0 = pile_rows("train", 0, 32768, log=log)
    i1 = lambda: ID.unstopped(lambda: ID.trigger_row(j, "ladder", 5, S1.TRIGGER, S1.TARGET, rows0, rows, widen=True))
    for name, read in (("R2", lambda: S1.r2(j, f, on)), ("I1", i1)):
        readings[name] = read()
        log(f"[{name}] {json.dumps(readings[name])}")
        log(f"[{name}, stage 1's reading beside it] {json.dumps({k: v for k, v in first['readings'][name].items() if k != 'shown'})}")
        dump(OUT, res)
    res |= {"gaps": rules.gaps(readings), "complete": True}
    dump(OUT, res)
    log(f"[gaps, reported; the definition is frozen] {res['gaps']}")
    hard_exit(0)


if __name__ == "__main__":
    main()

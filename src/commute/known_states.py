"""The 16 known-answer tasks x seeds 0-2 (48 states), one list for every figure that reads them (Figs 2, 3, 7, the tools
table). Built from experiments/known_3seeds/results/known3.json (its "share" cells: which states, and each task's side,
reuse or new, from its pair) and E314's loader (experiments/sae_proxies/run.py `locate`), whose choices it keeps:

  file       E299's seed-2 states (known_3seeds/results/stage2.json, sha256 recorded); else E136's inventory
             (share_untrained/results/inventory.json, no sha256); else E147's e58 states
             (refit_96key/results/large/dW_<map>_s<seed>.pt, no sha256)
  rows       the windows.read_rows task the state is read on (kind "held"); E299's seed-2 states read their seed-0
             sibling's task (E314); e58 states read E147's held-out windows of the map (`windows`, its file: read_rows
             has no e58/prior_cipher, whose windows are open_pair amend6's)
  trained_by the experiment folder that saved the update file
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from commute import windows as W
from commute.jobs import EXPERIMENTS

KNOWN3 = EXPERIMENTS / "known_3seeds" / "results" / "known3.json"
E299_S2 = EXPERIMENTS / "known_3seeds" / "results" / "stage2.json"
E136_INV = EXPERIMENTS / "share_untrained" / "results" / "inventory.json"
E147_INV = EXPERIMENTS / "refit_96key" / "results" / "inventory.json"


@dataclass(frozen=True)
class State:
    name: str  # <task>_s<seed>, e.g. e62/copy_s0
    task: str
    seed: int
    side: str  # reuse or new (known3's pair)
    file: str  # repo path of the saved delta-W (24 module-keyed matrices)
    sha256: str | None  # of the file, where a record holds one
    trained_by: str  # experiment folder that saved it
    rows: str  # the read task (windows.read_rows(rows, "held"); for e58, E147's windows below)
    windows: str | None  # e58 only: E147's windows file for the map (kv format, held-out part)


def _read(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _locate(name: str) -> tuple[str, str | None, str, str | None]:
    """(file, sha256, rows, windows) as E314's `locate`."""
    e136, e299 = _read(E136_INV), _read(E299_S2)["states"]
    if name in e299:
        return e299[name]["file"], e299[name]["sha256"], e136["states"][f"{name.rsplit('_s', 1)[0]}_s0"]["task"], None
    if name in e136["states"]:
        return e136["states"][name]["file"], None, e136["states"][name]["task"], None
    assert name.startswith("e58/"), f"{name}: in no inventory"
    m, s = name.removeprefix("e58/").rsplit("_s", 1)
    return f"experiments/refit_96key/results/large/dW_{m}_s{s}.pt", None, f"e58/{m}", _read(E147_INV)["maps"][m]["windows"]


@cache
def states() -> tuple[State, ...]:
    """The 48, in known3's pair order; a task in two pairs (e54/elicit, c3/cipher) listed once, its side the same in both."""
    out: dict[str, State] = {}
    for c in _read(KNOWN3)["cells"]:
        if c["method"] != "share":
            continue
        for side in ("reuse", "new"):
            for name in c["states"][side]:
                task, seed = name.rsplit("_s", 1)
                if name in out:
                    assert out[name].side == side, name
                    continue
                file, sha, rows, win = _locate(name)
                out[name] = State(name, task, int(seed), side, file, sha, file.split("/")[1], rows, win)
    assert len(out) == 48 and len({s.task for s in out.values()}) == 16, (len(out), len({s.task for s in out.values()}))
    return tuple(out.values())


def by_name() -> dict[str, State]:
    return {s.name: s for s in states()}


def rows(st: State) -> W.Rows:
    """The state's read rows (toks exactly what the model sees, read at .pos), on CPU."""
    if st.windows is not None:
        return W._kv(st.rows, "held", W.REPO / st.windows)
    return W.read_rows(st.rows, "held")

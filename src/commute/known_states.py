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
  old_task   the name of the definition's D task (`definition_task`): the `old_tasks.TASKS` entry E205 mapped the task
             onto (held_model_size_matched OT_NAMES), else <rows> with '/' as '_', the kv entry E205 builds
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from commute import old_tasks as OT
from commute import windows as W
from commute.jobs import EXPERIMENTS

KNOWN3 = EXPERIMENTS / "known_3seeds" / "results" / "known3.json"
E299_S2 = EXPERIMENTS / "known_3seeds" / "results" / "stage2.json"
E136_INV = EXPERIMENTS / "share_untrained" / "results" / "inventory.json"
E147_INV = EXPERIMENTS / "refit_96key" / "results" / "inventory.json"
OLD_TASK = {"c3/copy": "copy24", "c3/cipher": "cipher24", "c3/rank1": "rank1", "e46/ident32": "ident32", "e46/rand32": "rand32",
            "e62/copy": "e62_copy", "e62/lookup": "e62_lookup", "e53/lowercase": "lowercase", "e53/cipher": "cipher50"}  # E205 OT_NAMES


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
    old_task: str  # the definition D task's name (`definition_task`)


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
                out[name] = State(name, task, int(seed), side, file, sha, file.split("/")[1], rows, win, OLD_TASK.get(task, rows.replace("/", "_")))
    assert len(out) == 48 and len({s.task for s in out.values()}) == 16, (len(out), len({s.task for s in out.values()}))
    return tuple(out.values())


def by_name() -> dict[str, State]:
    return {s.name: s for s in states()}


def rows(st: State) -> W.Rows:
    """The state's read rows (toks exactly what the model sees, read at .pos), on CPU."""
    if st.windows is not None:
        return W._kv(st.rows, "held", W.REPO / st.windows)
    return W.read_rows(st.rows, "held")


def definition_task(st: State) -> OT.OldTask:
    """The task whose `old_tasks.task_windows` is the definition's D (E298's, through E221's World): the
    `old_tasks.TASKS` entry where E205 mapped the task onto one (OT_NAMES); otherwise E205's rule for a key table
    (held_model_size_matched `old_task`, with E221 `e187_task` keeping only the first file): a kv entry named and
    grouped <rows> with '/' as '_', built from the windows file the task was trained on. That file is E147's map
    windows for e58 (E298's own path) and E136's inventory file for e54 (keydependent_elicit windows_<elicit|cipher>.pt;
    e54/cipher_own was trained on windows_cipher.pt, keydependent_elicit `finetunes`, so it reads e54/cipher's D)."""
    if st.task in OLD_TASK:
        return OT.TASKS[OLD_TASK[st.task]]
    spec = {"format": "kv", "path": st.windows} if st.windows is not None else W.e136_tasks()[st.rows]
    assert spec["format"] == "kv", (st.name, spec)
    return OT.OldTask(st.old_task, "kv", st.old_task, {}, (W.REPO / spec["path"],))

"""E97's shared run pieces: the job (model, files, caches), the per-state reads written as each state finishes, the
greedy pairs read across a task and its control, and the stop. GPU only."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import sites
import table
import torch
from sites import TASKS

from commute import locator as L
from commute import locator_rules as LR
from commute import steering as ST
from commute.jobs import hard_exit
from commute.keydep import log
from commute.locate import from_saved
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def dump(path: Path, rec: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(rec), indent=1, sort_keys=True))


def say(tag: str, rec: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(rec))}")


def stop(path: Path, rec: dict[str, Any], why: str) -> NoReturn:
    rec["stopped"] = why
    dump(path, rec)
    log(f"STOP: {why}: tell the lead (exit 1)")
    hard_exit(1)


@dataclass
class Job:
    net: ST.Net
    F: sites.Files
    P: dict[str, Any]
    nat: L.Acts
    bases: dict[str, L.Base] = field(default_factory=dict)
    dws: dict[Path, ST.DW] = field(default_factory=dict)

    @property
    def th(self) -> dict[str, Any]:
        return self.P["thresholds"]

    def base(self, key: str) -> L.Base:
        if key not in self.bases:
            self.bases[key] = L.base(self.net, sites.site(self.F, key), self.nat)
        return self.bases[key]

    def dW(self, path: Path) -> ST.DW:
        if path not in self.dws:
            self.dws[path] = self.F.state(path)
        return self.dws[path]

    def state(self, task: str, seed: int, site: str | None = None) -> L.State:
        T = TASKS[task]
        return L.state(self.net, self.base(site or T.site), self.dW(T.states[seed]))

    def clear(self) -> None:
        self.bases.clear()
        self.dws.clear()
        torch.cuda.empty_cache()


def sid(task: str, seed: int) -> str:
    T = TASKS[task]
    return f"{T.site}:{T.states[seed].name}"


def with_controls(names: list[str]) -> list[str]:
    return list(dict.fromkeys([*names, *(TASKS[t].control for t in names)]))


def read_tasks(J: Job, names: list[str], out: dict[str, Any], path: Path) -> None:
    """Every state of the tasks and their controls, once each, written as each finishes; a state already in `out` (a resumed
    job) is not read again. All 24 matrices reverted must read G = 0 on every state, or the job stops."""
    out.setdefault("states", {})
    for t in with_controls(names):
        for s in sorted(TASKS[t].states):
            out.setdefault("index", {}).setdefault(t, {})[str(s)] = k = sid(t, s)
            if k in out["states"]:
                continue
            t0 = time.monotonic()
            rec = L.reads(J.net, J.state(t, s))
            rec["seconds"] = time.monotonic() - t0
            out["states"][k] = rec
            dump(path, out)
            say(f"state {t} s{s}", rec)
            if not LR.zero_ok([rec["write"]["G_all"]["lo"]], J.th["zero_tol"]):
                stop(path, out, f"{t} s{s}: all 24 reverted reads G {rec['write']['G_all']['lo']}")


def cross_pairs(J: Job, names: list[str], out: dict[str, Any], path: Path) -> None:
    """Each greedy pair a task's seeds found, read on every seed of the task and of its control."""
    for t in names:
        both = [(u, s) for u in (t, TASKS[t].control) for s in sorted(TASKS[u].states)]
        pairs = sorted({out["states"][sid(t, s)].get("pair") for s in TASKS[t].states} - {None})
        for p in pairs:
            for u, s in both:
                rec = out["states"][sid(u, s)]
                if p not in rec["removal"]:
                    rec["removal"][p] = L.removal(J.net, J.state(u, s), p)
                    say(f"pair {p} on {u} s{s}", rec["removal"][p])
        dump(path, out)


def locate(J: Job, names: list[str], out: dict[str, Any], path: Path, a: float) -> None:
    out["located"] = {t: table.locate_task(out, t, a, J.th) for t in names}
    dump(path, out)
    for t, r in out["located"].items():
        say(f"located {t}", {"part": r["part"], "parts": {p: {k: v[k] for k in ("reading", "per_seed", "shares", "control_shares")} for p, v in r["parts"].items()}})


def load(path: Path) -> dict[str, Any]:
    """A stage file back from JSON (nan was written as a string), or an empty record."""
    return from_saved(json.loads(path.read_text())) if path.exists() else {}


def stage_reads(J: Job, names: list[str], path: Path) -> dict[str, Any]:
    """Every state's reads and the cross pairs; resumes from `path` if it holds an unfinished run of this stage."""
    out = load(path)
    out.update({"tasks": names, "files": J.F.seen})
    for t in names:
        read_tasks(J, [t], out, path)
    cross_pairs(J, names, out, path)
    return out

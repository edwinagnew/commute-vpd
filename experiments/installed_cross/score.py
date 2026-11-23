"""E271 (P2.65) scoring, CPU only, over results/read/<base>__<task>_s<seed>_<kind>.json: per reading and kind, the 9
triples (task x seed: does the diagonal state rank above both moved off-diagonal states, in the filed direction) and the
AUC of the 9 diagonal against the 18 off-diagonal states, with the same AUC on their matched Gaussians as the floor.
Rules: results/predictions.json "thresholds.evidence". The released M's row is descriptive (listed, never scored).

    ~/commute-agents/fleet/cpu-run python3 experiments/installed_cross/score.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = (0, 1, 2)
# amendment 2026-11-22d: the candidate tasks with installs; the cipher (lock failed at every weight) is out.
# Amendment 2026-11-22f appends parity (last, so the other tasks' fine-tune seeds, run.finetune, are unchanged).
CANDIDATES = ("addition", "sst2", "yoruba", "cipherlocal", "parity")
FIRST = CANDIDATES[:4]  # amendment d's candidates: the cell waits on these
SLOT = ("yoruba", "cipherlocal", "parity")  # the third slot, in the filed order
TWO_TASK_CALL = "separates on 6 of 6 pairs; two tasks, no task-level significance (best task-level P 1/4)"


def install_state(t: str) -> str:
    """'locked' (install_<t>.json done), 'failed' (it stopped), 'pending' (no file, or still running)."""
    p = RESULTS / f"install_{t}.json"
    d = json.loads(p.read_text()) if p.is_file() else {}
    return "locked" if d.get("done") else "failed" if "stopped" in d else "pending"


def cell_tasks() -> tuple[str, ...]:
    """The filed rule (amendment 2026-11-22d), applied only once every candidate's install has finished: addition and
    SST-2 when locked; the third slot Yoruba if it locked, else the cipher with the cue before the key if that locked,
    else empty (then the two-task rule). Fewer than two locked tasks: none (the run stops). Amendment 2026-11-22f:
    parity is the slot's last candidate; while its install runs the cell is the two tasks (their cross goes on), and
    it joins when it locks (the 3x3 extension, primary); controls waits for it (`settled`)."""
    st = {t: install_state(t) for t in CANDIDATES}
    if any(st[t] == "pending" for t in FIRST):
        return ()
    slot = next((t for t in SLOT if st[t] == "locked"), None)
    tasks = tuple(t for t in ("addition", "sst2", slot) if t is not None and st[t] == "locked")
    if "sst2" in tasks and sst2_rule().get("new") is False:  # amendment 2026-11-22e: SST-2 is not new to M
        tasks = tuple(t for t in tasks if t != "sst2")
    return tasks if len(tasks) >= 2 else ()


def settled() -> bool:
    """Every candidate's install has ended (amendment 2026-11-22f): until then controls, and so every reading, waits."""
    return all(install_state(t) != "pending" for t in CANDIDATES)


def mean_steps(base: str, task: str) -> float | None:
    """Seed-mean steps to G* in cross_<base>_<task>.json (an unreached run counts its cap); None until it is done."""
    p = RESULTS / f"cross_{base}_{task}.json"
    d = json.loads(p.read_text()) if p.is_file() else {}
    st = d.get("states", {})
    return sum(s["steps"] for s in st.values()) / len(st) if d.get("done") and st else None


def sst2_rule() -> dict[str, Any]:
    """Amendment 2026-11-22e (redteam-lead): SST-2 plays the new side only if M's own uncued fine-tune is slow. new is
    False when M's seed-mean steps to SST-2's G* are under efficiency_ratio_min (control 2's factor, 5) x the SST-2
    diagonal's: SST-2 is then dropped and the cell falls to the filed outcomes without it. None until both cells are
    done. A diagonal of 0 steps (a leaked lock) keeps SST-2 in; control 2 then fails it."""
    ratio = json.loads((RESULTS / "predictions.json").read_text())["thresholds"]["efficiency_ratio_min"]
    m, diag = mean_steps("M", "sst2"), mean_steps("sst2", "sst2")
    if m is None or diag is None:
        return {"M_steps": m, "diag_steps": diag, "new": None}
    return {"M_steps": m, "diag_steps": diag, "ratio": ratio, "new": not m < ratio * diag}


def install_summary() -> dict[str, Any]:
    """Amendment 2026-11-22e: per install file, its state, how many settings were trained (every failed candidate
    stays in the file) and which one was kept; the position-0 cipher's failed install is listed with the candidates."""
    out = {}
    for t in ("cipher", *CANDIDATES):
        p = RESULTS / f"install_{t}.json"
        d = json.loads(p.read_text()) if p.is_file() else {}
        out[t] = {"state": "locked" if d.get("done") else "failed" if "stopped" in d else "pending",
                  "tried": len(d.get("candidates", {})), "kept": d.get("kept", {}).get("lambda")}  # fmt: skip
    return out


TASKS = cell_tasks()


def at(rec: dict[str, Any], path: str) -> float:
    x: Any = rec
    for k in path.split("."):
        x = x.get(k) if isinstance(x, dict) else None
    return float(x) if isinstance(x, (int, float)) and x is not None else math.nan


def signed(v: float, direction: str) -> float:
    return v if direction == "higher" else -v


def triple(d: float, offs: list[float]) -> bool:
    """The diagonal strictly above every off-diagonal of its task and seed (two, or one when a task is dropped); any
    nan or a tie is not above."""
    return not math.isnan(d) and len(offs) >= 1 and all(not math.isnan(o) and d > o for o in offs)


def auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg) over finite pairs, ties 1/2; nan when either side is empty."""
    p, n = [x for x in pos if not math.isnan(x)], [x for x in neg if not math.isnan(x)]
    if not p or not n:
        return math.nan
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in p for b in n) / (len(p) * len(n))


def score(vals: dict[tuple[str, str, int], float], ev: dict[str, Any], tasks: tuple[str, ...] | list[str] = TASKS) -> dict[str, Any]:
    """vals[(base, task, seed)] signed so that higher = reuse; bases are the moved ones of `tasks` (all three, or two
    when a task and its base are dropped: each triple is then a pair)."""
    diag = [vals.get((t, t, s), math.nan) for t in tasks for s in SEEDS]
    off = [vals.get((b, t, s), math.nan) for t in tasks for s in SEEDS for b in tasks if b != t]
    hits = [triple(vals.get((t, t, s), math.nan), [vals.get((b, t, s), math.nan) for b in tasks if b != t]) for t in tasks for s in SEEDS]
    fin = {"triples": sum(not math.isnan(vals.get((t, t, s), math.nan)) and all(not math.isnan(vals.get((b, t, s), math.nan)) for b in tasks if b != t)
                          for t in tasks for s in SEEDS),
           "diag": sum(not math.isnan(x) for x in diag), "off": sum(not math.isnan(x) for x in off)}  # fmt: skip
    by_task = {t: auc([vals.get((t, t, s), math.nan) for s in SEEDS], [vals.get((b, t, s), math.nan) for s in SEEDS for b in tasks if b != t])
               for t in tasks}  # fmt: skip  (reported beside the pooled AUC: a reading's scale may differ by task)
    per_task = {t: sum(hits[i * len(SEEDS) : (i + 1) * len(SEEDS)]) for i, t in enumerate(tasks)}  # x of 3 (amendment c)
    return {"triples": sum(hits), "of": len(hits), "triples_by_task": per_task, "auc": auc(diag, off), "auc_by_task": by_task,
            "finite": fin, "read": all(fin[k] >= ev["min_finite"][k] for k in fin)}  # fmt: skip


def present_tasks(recs: dict[tuple[str, str, int], dict[str, Any]]) -> list[str]:
    return [t for t in TASKS if any(k[1] == t for k in recs)]


def bars(ev: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    """The filed bars for the tasks present: all three, or the filed dropped-task bar (amendment c; a dropped task
    takes its base with it, so each of the 6 comparisons is the diagonal against one off-diagonal, chance 1/2:
    6 of 6, P = 1/64 = 0.016 if independent. At the task level, seeds moving together, the best P is 1/4 with two
    tasks, so a two-task result is descriptive (write_up_call; amendment 2026-11-22e), and 1/27 = 0.037 with three)."""
    if len(tasks) == 3:
        return ev
    assert len(tasks) == 2, f"tasks present {tasks}: no bar is filed for fewer than two"
    return ev | {"triples_min": ev["dropped_task"]["triples_min"], "min_finite": ev["dropped_task"]["min_finite"]}


def write_up_call(v: str, real: dict[str, Any], ev: dict[str, Any], tasks: list[str]) -> str:
    """The stated secondary (amendment c): a primary 'separates' is called 'separates' in the write-up only if every
    task present has at least task_min of its 3 triples; else 'separates on the pool, not per task'."""
    if v != "separates":
        return v
    if len(tasks) == 2:  # amendment 2026-11-22e: descriptive, no task-level significance (best P 1/4)
        return TWO_TASK_CALL
    ok = all(real["triples_by_task"][t] >= ev["secondary"]["task_min"] for t in tasks)
    return "separates" if ok else "separates on the pool, not per task"


def rescale_flags(rows: dict[str, dict[str, Any]], ev: dict[str, Any]) -> dict[str, str]:
    """Amendment c: a share reading whose verdict is 'separates' at one kind (gain, size) and not at the other is read
    as the rescaling, not the history."""
    out = {}
    for name in ev["share_readings"]:
        a, b = (rows.get(k, {}).get(name, {}).get("verdict") for k in ("gain", "size"))
        if (a == "separates") != (b == "separates"):
            out[name] = f"rescale-dependent (gain: {a}, size: {b})"
    return out


def verdict(real: dict[str, Any], floor: dict[str, Any], ev: dict[str, Any]) -> str:
    """`floor`: the matched Gaussians scored on the reading's floor path (its numerator for a ratio reading, amendment
    2026-11-22); a floor that is not read never passes for one."""
    if not real["read"]:
        return "not read"
    ok = real["triples"] >= ev["triples_min"] and real["auc"] >= ev["auc_min"]
    if not ok:
        return "does not separate"
    if not floor["read"]:
        return "floor not read"
    return "floor separates too" if floor["auc"] >= ev["gauss_auc_max"] else "separates"


def load(kind: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    out = {}
    for p in (RESULTS / "read").glob(f"*_{kind}.json"):
        r = json.loads(p.read_text())
        out[(r["base"], r["task"], r["seed"])] = r
    return out


def main() -> None:
    P = json.loads((RESULTS / "predictions.json").read_text())
    ev = P["thresholds"]["evidence"]
    gauss = load("gauss")
    tasks = present_tasks(gauss)
    res: dict[str, Any] = {"experiment": P["experiment"], **score_tasks(ev, tasks, gauss)}
    if len(tasks) == 3 and "parity" in tasks:  # amendment 2026-11-22f: the 3x3 is primary, the two-task cross beside it
        print("[beside] the two-task cross without parity")
        res["two_task_beside"] = score_tasks(ev, [t for t in tasks if t != "parity"], gauss)
    (RESULTS / "score.json").write_text(json.dumps(res, indent=1))


def score_tasks(ev: dict[str, Any], tasks: list[str], gauss: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    """Every reading and kind scored on the records of `tasks` (bases and tasks both in it) under their filed bars."""
    keep = lambda recs: {k: r for k, r in recs.items() if k[1] in tasks and k[0] in (*tasks, "M")}
    gauss, evb = keep(gauss), bars(ev, tasks)
    res: dict[str, Any] = {"tasks": tasks, "triples_min": evb["triples_min"], "kinds": {}}
    for kind in ev["kinds"]:
        recs, rows = keep(load(kind)), {}
        for name, (path, direction, floor_path) in ev["readings"].items():
            if name in ev["skip"].get(kind, []):
                continue
            v = {k: signed(at(r, path), direction) for k, r in recs.items() if k[0] != "M"}
            g = {k: signed(at(r, floor_path), direction) for k, r in gauss.items() if k[0] != "M"}
            real, gs = score(v, evb, tasks), score(g, evb, tasks)
            m_row = {f"{t}_s{s}": at(recs[("M", t, s)], path) for t in tasks for s in SEEDS if ("M", t, s) in recs}
            vd = verdict(real, gs, evb)
            rows[name] = {"real": real, "gauss": gs, "verdict": vd, "write_up": write_up_call(vd, real, ev, tasks), "released_M": m_row}
            print(f"[{kind} {name}] triples {real['triples']}/{real['of']} {real['triples_by_task']} auc {real['auc']:.3f} | gauss auc {gs['auc']:.3f} -> {vd}")
        res["kinds"][kind] = rows
    res["rescale_flags"] = rescale_flags(res["kinds"], ev)
    print(f"[rescale] {res['rescale_flags']}")
    return res


if __name__ == "__main__":
    main()

"""E97's located rule over saved records, and the deliverable table. Arithmetic on the stage files only; no model.

A task's parts are those its seeds' carry and write maps nominate, plus each seed's greedy pair (`locator_rules.nominees`);
each is read with the located rule against the same seed's control. One table row per (task, seed), with E90's fields
(`task`, `C`, `share`, `control_share`, `band`, `verdict`, `reason`; grepped in experiments/new_vs_reuse/results/stage2.json)
and the locator's (`part`, `a_share`, `b_share`, `c_share`, `w0_move`, `located`). Every share is unit "lo" (the filed one)
at `read_at`: the located part, else the state's top carry part. `verdict` stays empty for baseline-lead, except "does not
apply" on constant-answer rows.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from sites import EXP, TASKS

from commute import locator_rules as LR

CONSTANT = "constant answer: beside only, with the same-windows null (decisions.md line 121)"


def rec_of(js: Mapping[str, Any], task: str, seed: int) -> dict[str, Any]:
    return js["states"][js["index"][task][str(seed)]]


def locate_task(js: Mapping[str, Any], t: str, a: float, th: Mapping[str, Any]) -> dict[str, Any]:
    T = TASKS[t]
    seeds = sorted(T.states)
    recs, ctls = [rec_of(js, t, s) for s in seeds], [rec_of(js, T.control, s) for s in seeds]
    noms = list(dict.fromkeys(n for r in recs for n in LR.nominees(r, th["nominate_top"])))
    parts = {}
    for n in noms:
        rd, sh, cs = [], [], []
        for r, c in zip(recs, ctls, strict=True):
            x, y = r["removal"].get(n), c["removal"].get(n)
            if x is None or y is None:
                rd.append("unreadable: not read")
                sh.append(math.nan)
                cs.append(math.nan)
                continue
            rd.append(LR.seed_reading({"G": r["G"]["lo"], "fall": x["fall"]["lo"]}, {"G": c["G"]["lo"], "fall": y["fall"]["lo"]}, a, th))
            sh.append(x["share"]["lo"])
            cs.append(y["share"]["lo"])
        parts[n] = {**LR.task_reading(rd), "shares": sh, "control_shares": cs}
    return {"seeds": seeds, "control": T.control, "nominated": noms, "parts": parts, "part": LR.best_located(parts)}


def e90_C() -> dict[str, str]:
    table = json.loads((EXP / "new_vs_reuse" / "results" / "stage2.json").read_text())["table"]
    return {r["task"]: r["C"] for r in table}


def _numbers(r: Mapping[str, Any], c: Mapping[str, Any], at: str, cap: Mapping[str, Any]) -> dict[str, Any]:
    x, y = r["removal"].get(at), c["removal"].get(at)
    return {"share": _get(x, ("share", "lo")), "control_share": _get(y, ("share", "lo")), "band": 2 * _get(x, ("se_by_row",)),
            "fall_nats": _get(x, ("fall", "lo")), "control_fall_nats": _get(y, ("fall", "lo")), "w0_move": _get(x, ("w0_move", "lo")),
            "share_gap_unit": _get(x, ("share", "gap")), "p_mean_off": _get(x, ("p_mean_off",)),
            "share_natural_means": _get(r["removal_natural_means"], (at, "share", "lo")),
            "a_share": _get(r["carry"], (at, "lo")), "b_share": _get({**r["write"]["head"], **r["write"]["mlp"]}, (at, "lo")),
            "c_share": _get(cap, (at, "c_share")), "G": r["G"]["lo"], "G_control": c["G"]["lo"], "p_mean": r["p_mean"], "p_mean_w0": r["p_mean_w0"]}  # fmt: skip


def _get(d: Any, ks: tuple[str, ...]) -> Any:
    for k in ks:
        if not isinstance(d, Mapping) or k not in d:
            return math.nan
        d = d[k]
    return d


def task_rows(js: Mapping[str, Any], stage: int, t: str, C: Mapping[str, str]) -> list[dict[str, Any]]:
    T, loc = TASKS[t], js["located"][t]
    blocked = js.get("cannot_nominate", {}).get(t)
    out = []
    for i, s in enumerate(loc["seeds"]):
        r, c = rec_of(js, t, s), rec_of(js, T.control, s)
        part = None if blocked else loc["part"]
        at = part or LR.top_part({n: v["lo"] for n, v in r["carry"].items()})
        row: dict[str, Any] = {"task": t, "seed": s, "stage": stage, "C": C.get(T.e90, "not in E90's table"), "part": part or LR.NONE_FOUND,
                               "read_at": at, "located": bool(part), "verdict": "does not apply" if T.constant else "", "control_task": T.control}  # fmt: skip
        if at is None:
            row["reason"] = "the carry map's top is tied: no part named"
        else:
            row |= _numbers(r, c, at, js.get("capability", {}).get("parts", {}) if t in ("T5 copy", "T5 lookup") else {})
            row["reason"] = loc["parts"][at]["per_seed"][i] if at in loc["parts"] else "the top carry part was not read on the control"
        if blocked:
            row["reason"] = blocked
        if T.constant:
            row |= {"located": None, "reason": CONSTANT}
        out.append(row)
    return out


def rows(stages: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    C = e90_C()
    return [row for k, js in sorted(stages.items()) for t in js.get("located", {}) if TASKS[t].in_table for row in task_rows(js, k, t, C)]

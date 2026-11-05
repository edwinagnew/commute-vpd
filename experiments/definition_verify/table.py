"""E99 stage 5: one table, one line per old 67M task (E90's rows, `new_vs_reuse/results/stage2.json` `.table`, plus E95's replicate
pair): the assumed label, the verdict under the refined rules, and the reason. Reads the stage JSONs only, no model. A row stage 4
does not touch keeps E90's verdict, with the reason "E90, unchanged by stage 4". Rows the stage-4 GPU job reads say "not read"
until results/stage4_class.json is complete. Writes results/table.json and prints the table as markdown.
Usage (from the clone root): ~/commute-agents/fleet/cpu-run --mem 6G python experiments/definition_verify/table.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import rules4 as R4
from stage4_cpu import EXP, PRED4, RESULTS, filed, load

from commute.jobs import hard_exit
from commute.open_pair import jsonable

OUT = RESULTS / "table.json"
LIMIT = "a circuit from old parts reads as reuse (decisions.md row 121)"
Read = Callable[[dict[str, Any]], tuple[str, str]]


def copy_rows(cpu: dict[str, Any]) -> dict[str, Read]:
    a = cpu["a_copy_lookup"]
    nets = lambda rows: ", ".join(f"{s:.3f} against {b:.3f}" for s, b in rows)
    why = (f"net share at heads (2,3)+(2,4) {nets(a['copy'])}, lookup {nets(a['lookup'])}; forced copy reads net 1 "
           f"({a['positive_control']}), forbidden copy within band ({a['negative_control']}); paired copy minus lookup: "
           f"{a['paired']['call']}; {LIMIT}")  # fmt: skip
    return {"T5 copy": lambda _: (a["verdict"], why),
            "T5 lookup": lambda _: (R4.NO_C, f"in-context copying not admissible on lookup (E90 A7); as copy's spared control it reads {nets(a['lookup'])}")}


def radm_rows(cpu: dict[str, Any]) -> dict[str, Read]:
    """Every value-form row through admissibility (E90's, then R-adm), with its per-state reading at block 1 beside."""
    out: dict[str, Read] = {}
    for r in cpu["b_radm"].values():
        if r["rows"] and next(iter(r["rows"].values()))["admissibility"] is None:
            continue  # the class map: stage4.py
        why = f"R-adm (a) {r['n_inputs']} inputs on {r['n_values']} values ({r['per_value']:.2f} per value): {'passes' if r['a'] else 'fails'}; (b) {r['b']}"
        for row, x in r["rows"].items():
            task = " ".join(row.split()[:2]) if row.split()[-1].isdigit() else row
            out[task] = lambda _, x=x, why=why: (x["admissibility"], f"{why}; E90's admissibility {x['e90']}")
    return out


def elicit_row() -> Read:
    why = "admissible on one parity split only, and the batch's apparatus change 1 asks both (reports/decisions/2026-11-05_decision_e90-definition-batch.md)"
    return lambda _: (R4.NO_C, why)


def lowercase_row(s3: dict[str, Any]) -> Read:
    f = s3["forced lowercase"]
    caps = ", ".join(f"{f[s]['G_over_native']:.2f}" for s in ("s0", "s1"))
    why = f"a constant answer; R-net through heads (1,1)+(1,2) has no positive control: the forced arm reaches {caps} of the native gain (cap)"
    return lambda _: ("definition does not apply", why)


def class_rows(cls: dict[str, Any] | None) -> dict[str, Read]:
    if not (cls and cls.get("complete")):
        return {t: (lambda _: ("not read", "stage 4's GPU job (stage4.py) has not completed")) for t in ("T15 class map", "T15 twin")}
    v = cls["verdict"]
    b1, radm = v["block 1"], v["radm"]
    why = lambda m: (f"embedding class at block 1: teeth {b1['teeth']}; R-adm (a) {radm['a']}, (b) {radm['b']}; per seed "
                     f"{b1['maps'][m].get('per_seed')}; paired class map minus twin {b1['paired']['call']}; block 2 beside: "
                     f"{v['block 2']['maps'][m]['verdict']}; {LIMIT}")  # fmt: skip
    return {"T15 class map": lambda _: (b1["maps"]["class_map"]["verdict"], why("class_map")),
            "T15 twin": lambda _: (b1["maps"]["class_shuffled"]["verdict"], why("class_shuffled"))}


def table(rows: list[dict[str, Any]], reads: dict[str, Read]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        v, why = reads[r["task"]](r) if r["task"] in reads else (r["verdict"], f"E90, unchanged by stage 4{': ' + r['reason'] if r['reason'] else ''}")
        out.append({"task": r["task"], "assumed": r["assumed"], "verdict": v, "reason": why})
    return out


def main() -> None:
    filed(PRED4)
    rows = load(EXP / "new_vs_reuse/results/stage2.json")["table"]
    rows += [{"task": "T4 ident32b", "assumed": "strengthens, by construction (E95's replicate)", "verdict": "", "reason": ""},
             {"task": "T6 rand32b", "assumed": "new by construction (E95's replicate)", "verdict": "", "reason": ""}]  # fmt: skip
    cpu, s3 = load(RESULTS / "stage4_cpu.json"), load(RESULTS / "stage3_amend1.json")
    cls = load(RESULTS / "stage4_class.json") if (RESULTS / "stage4_class.json").exists() else None
    reads = {**radm_rows(cpu), **copy_rows(cpu), "T11 elicit": elicit_row(), "T7 lowercase": lowercase_row(s3), **class_rows(cls)}
    t = table(rows, reads)
    OUT.write_text(json.dumps(jsonable(t), indent=1))
    print("| Task | Assumed label | Verdict | Reason |\n|---|---|---|---|")
    for r in t:
        print(f"| {r['task']} | {r['assumed']} | {r['verdict']} | {r['reason']} |")
    hard_exit(0)


if __name__ == "__main__":
    main()

"""E90's deliverable: one line per old task of the plan's order table, plus I1-I3 and R7, from the saved JSON (no model). Each line has
the assumed label, the verdict with its share and band, the reason where unreadable, and who read it: stage 0's admissibility always
beside, and of stage 1, Amendment 1 and stage 2 the last read, with any earlier read kept under `superseded`. Rows: predictions_amend1.json."""

from __future__ import annotations

import math
from typing import Any

import rules

KEPT = ("reading", "windows_top1_target", "of_windows", "keys_with_enough_windows", "G_lowercased", "G_cased", "block")


def read_of(by: str, r: dict[str, Any], state: str | None = None, control: str | None = None) -> dict[str, Any]:
    """One read of a task. NaN comes back from JSON as a string, so numbers go through float."""
    s = r.get("shares", {})
    one = lambda k: [float(s[k])] if k in s else []
    return {"by": by, "verdict": r["verdict"], "share": one(state), "control_share": one(control), "band": float(r.get("band", math.nan)), **{k: r[k] for k in KEPT if k in r}}


def rand32_of(by: str, r2: dict[str, Any]) -> dict[str, Any]:
    """R2 reads ident32's verdict; rand32 enters it through its share, so the verdict here names R2's reading."""
    return {**read_of(by, r2, "rand32", "write"), "verdict": f"R2 read {r2['reading']}"}


def later_reads(first: dict[str, Any], amend: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Stage 1's and Amendment 1's reads, earliest first."""
    a, b = first["readings"], amend["readings"]
    r3 = a["R3"]
    return {"T12 rank-1": [read_of("stage 1, R3", r3, "rank1", "write")],
            "T11 cipher": [{**read_of("stage 1, R3", r3), "verdict": r3["cipher_verdict"], "share": [float(r3["cipher_share"])], "control_share": [float(r3["shares"]["write"])]}],
            "T4 ident32": [read_of("stage 1, R2", a["R2"], "ident32", "write"), read_of("Amendment 1, R2", b["R2"], "ident32", "write")],
            "T6 rand32": [rand32_of("stage 1, R2", a["R2"]), rand32_of("Amendment 1, R2", b["R2"])],
            "I1 ladder": [read_of("stage 1, I1", a["I1"], "rank1", "write"), read_of("Amendment 1, I1", b["I1"], "rank1", "write")],
            "I2 emoticon": [read_of("stage 1, I2", a["I2"], "rank1", "write")],
            "I3 lowercase": [read_of("stage 1, I3", a["I3"], "state", "push")],
            "R7 ladder push": [read_of("stage 1, R7", a["R7"], "state", "push")]}  # fmt: skip


def stage2_reads(res: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Stage 2's read per task. A map's verdict over its seeds, a share per seed; control shares: E62's other map, or each seed's hand write."""
    one = res["row1"]
    out = {f"T5 {m}": {"by": "stage 2", "verdict": one["verdicts"][m]["verdict"], "share": [float(x) for x in one["shares"][m]],
                       "control_share": [float(x) for x in one["shares"][other]], "band": float(one["band"])}
           for m, other in (("copy", "lookup"), ("lookup", "copy"))}  # fmt: skip
    for t, form, m in (("T12 rank-1", "guess", "rank1"), ("T11 cipher", "guess", "cipher"), ("T11 elicit", "guess", "elicit"),
                       ("T4 ident32", "identity", "ident32"), ("T6 rand32", "identity", "rand32"), ("T13 copy-24", "copy24", "copy24")):  # fmt: skip
        row = res["rows"][form]
        seeds = [n for n in row.get("shares", {}) if n.rsplit("_", 1)[0] == m]
        out[t] = {"by": "stage 2", "verdict": row["verdicts"][m]["verdict"], "share": [float(row["shares"][n]) for n in seeds],
                  "control_share": [float(res["states"][f"{n}_row"]["control"]) for n in seeds], "band": float(row.get("band", math.nan))}  # fmt: skip
    if "I1_seeds" in res:
        per = res["I1_per_seed"].values()
        out["I1 ladder"] = {"by": "stage 2, seeds 0-2", "verdict": res["I1_seeds"]["verdict"], "share": [float(r["shares"]["rank1"]) for r in per],
                            "control_share": [float(r["shares"]["write"]) for r in per], "band": max(float(r["band"]) for r in per)}  # fmt: skip
    return out


def reason_of(verdict_reason: str, last: dict[str, Any] | None, note: str | None) -> str:
    """The rule's reason, with the windows count on a gap and the filed note on an unread row."""
    parts = [verdict_reason]
    if last and "windows_top1_target" in last:
        parts.append(f"W0's top-1 is the target on {last['windows_top1_target']} of {last['of_windows']} windows at the trigger")
    return "; ".join(p for p in (*parts, note) if p)


def line(row: dict[str, Any], assumed: dict[str, str], label: str | None, reads: list[dict[str, Any]]) -> dict[str, Any]:
    t = row["task"]
    last = reads[-1] if reads else None
    base = rules.table_row(t, assumed[row.get("assumed", t)], label, last["verdict"] if last else row.get("excluded"))
    share = last["share"] if last else []
    out = {**base, "C": row["C"], "verdict": rules.say_seeds(base["verdict"], share), "share": share, "control_share": last["control_share"] if last else [],
           "band": last["band"] if last else math.nan, "reason": reason_of(base["reason"], last, row.get("note")), "admissibility": label,
           "read_by": last["by"] if last else "stage 0" if label else "not read", "superseded": reads[:-1]}  # fmt: skip
    return {**out, **{k: last[k] for k in ("G_lowercased", "G_cased") if last and k in last}}


def final_table(P: dict[str, Any], zero: dict[str, Any], first: dict[str, Any], amend: dict[str, Any], res: dict[str, Any]) -> list[dict[str, Any]]:
    """`P` is `apparatus.filed()`. A task with several rungs or sizes gives its stage 0 labels joined."""
    table, pairs = P["amend1"]["table"], zero["pairs"]
    assumed = {**P["assumed_labels"], **table["assumed_added"]}
    label = lambda t: " / ".join(sorted({r["label"] for k, r in pairs.items() if k.startswith(t)})) or None
    earlier, last = later_reads(first, amend), stage2_reads(res)
    return [line(r, assumed, label(r["task"]), [*earlier.get(r["task"], []), *([last[r["task"]]] if r["task"] in last else [])]) for r in table["rows"]]

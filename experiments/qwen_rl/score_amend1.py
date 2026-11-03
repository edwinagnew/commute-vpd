"""E69 Amendment 1's rules (results/predictions_amend1.json) from the results files, no model; writes results/amend1_score.json.
A file not there reads "missing", so it runs at any point of the amendment:
  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/qwen_rl/score_amend1.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from relaxed import RESULTS, ROOT, RULES, SEED1, Rec, ci, lead, paired, read, strict

from commute.qwen_rl import below, p1
from commute.qwen_updates import at_least, at_most, ok

SEED2 = "lr1e-6w10_s70"
Key = Callable[[Rec], bool]
Rule = dict[str, Any]


def lead_label(rule: Rule, g: dict[str, Any]) -> str:
    """S1 and S1r: P1's three-way rule under the lead's names."""
    label = p1(rule, g)
    return rule["labels"].get(label, label)


def s2(rule: Rule, gain: float | None, first: float) -> str:
    if not ok(gain):
        return "missing"
    assert gain is not None
    if below(abs(gain), rule["abs_min"]):
        return "one-seed walk"
    return "systematic" if gain * first > 0 else "opposite sign"


def b1(rule: Rule, x: float | None) -> str:
    if not ok(x):
        return "missing"
    return "arrives" if at_least(x, rule["arrives_min"]) else "after" if below(x, rule["after_below"]) else "between"


def a1(rule: Rule, x: float | None) -> str:
    if not ok(x):
        return "missing"
    return "carries" if at_least(x, rule["carries_min"]) else "not" if at_most(x, rule["not_at_most"]) else "between"


def a2(rule: Rule, loss: float | None) -> str:
    if not ok(loss):
        return "missing"
    return "carries" if at_least(loss, rule["carries_min"]) else "not"


def whole(ref: dict[int, Rec], x: dict[int, Rec], key: Key) -> dict[str, Any]:
    """x minus ref over all problems in both, nothing dropped."""
    return paired([int(key(x[i])) - int(key(ref[i])) for i in ref if i in x])


def lenient(r: Rec) -> bool:
    return bool(r["lenient"])


def state_row(ref: dict[int, Rec], x: dict[int, Rec]) -> dict[str, Any]:
    drop = paired([int(x[i]["strict"]) - int(ref[i]["strict"]) for i in ref if i in x and x[i]["fits"] and ref[i]["fits"]])
    return {"whole_strict": whole(ref, x, strict), "whole_relaxed": whole(ref, x, ci), "whole_lenient": whole(ref, x, lenient), "drop_strict": drop}


def done(folder: str, label: str) -> bool:
    p = RESULTS / folder / "amend1" / f"{label}.json"
    return p.is_file() and bool(json.loads(p.read_text()).get("done"))


def gain_of(folder: str, arm: str) -> dict[str, Any]:
    p = RESULTS / folder / f"{arm}.json"
    return json.loads(p.read_text()).get("gain", {}).get("gsm8k", {}) if p.is_file() else {}


def seeds(P: Rule) -> dict[str, Any]:
    w0, r, n = read(ROOT / "acc" / "W0_gsm8k.jsonl"), read(ROOT / SEED2 / "acc" / "R_gsm8k.jsonl"), read(ROOT / SEED2 / "acc" / "N_gsm8k.jsonl")
    r1, n1 = read(ROOT / SEED1 / "acc" / "R_gsm8k.jsonl"), read(ROOT / SEED1 / "acc" / "N_gsm8k.jsonl")
    g0, g1 = gain_of(SEED2, "R"), gain_of(SEED2, "N")
    s0 = p1(P["S0"], g0)
    scored = s0 == "learns"
    strict2, relaxed2 = lead(w0, r, n, strict), lead(w0, r, n, ci)
    first = gain_of(SEED1, "N").get("diff")
    return {"S0": {"gain": g0, "label": s0},
            "S1": {**strict2, "label": lead_label(P["S1"], strict2), "scored": scored},
            "S1r": {**relaxed2, "label": lead_label(P["S1r"], relaxed2), "scored": scored},
            "S2": {"gain": g1, "label": s2(P["S2"], g1.get("diff"), first) if first is not None else "missing", "scored": scored},
            "seed1": {"strict": lead(w0, r1, n1, strict), "relaxed": lead(w0, r1, n1, ci)}}


def saved_steps(P: Rule) -> dict[str, Any]:
    ref = read(ROOT / SEED1 / "acc" / "W0card_gsm8k.jsonl")
    out: dict[str, Any] = {"control": done(SEED1, "N100"), "states": {}, "B1": {}}
    for folder in (SEED1, SEED2):
        rows = {f"{arm}{t}": state_row(ref, read(ROOT / folder / "acc" / f"{arm}{t}_gsm8k.jsonl")) for arm in "RN" for t in (10, 20, 50)}
        out["states"][folder] = rows
        x = rows["N50"]["whole_strict"]["diff"]
        out["B1"][folder] = {"diff": x, "label": b1(P["B1"], x) if out["control"] else "control not passed"}
    out["states"][SEED1]["N100"] = state_row(ref, read(ROOT / SEED1 / "acc" / "N100_gsm8k.jsonl"))
    x = out["states"][SEED1]["R10"]["whole_lenient"]["diff"]
    out["E_R10"] = {"diff": x, "holds": ok(x) and at_most(abs(x), P["E_R10"]["expected_abs_max"])}  # type: ignore[arg-type]
    return out


def openings(P: Rule) -> dict[str, Any]:
    ref = read(ROOT / SEED1 / "acc" / "W0card_gsm8k.jsonl")
    n100 = read(ROOT / SEED1 / "acc" / "N100_gsm8k.jsonl")
    states = {x: read(ROOT / SEED1 / "acc" / f"{x}_gsm8k.jsonl") for x in ("W0_own", "W0_we_are", "W0_we_need", "N100_own")}
    control = done(SEED1, "W0_own")
    rows = {x: state_row(ref, recs) for x, recs in states.items()}
    are = rows["W0_we_are"]["whole_strict"]["diff"]
    loss = whole(states["N100_own"], n100, strict)
    return {"control": control, "states": rows,
            "A1": {"diff": are, "label": a1(P["A1"], are) if control else "control not passed"},
            "A2": {**loss, "label": a2(P["A2"], loss["diff"]) if control else "control not passed"}}


def main() -> None:
    P = json.loads(RULES.read_text())
    g0 = RESULTS / "amend1_G0.json"
    out = {"G0": json.loads(g0.read_text())["holds"] if g0.is_file() else "missing",
           "seeds": seeds(P), "saved": saved_steps(P), "openings": openings(P)}
    (RESULTS / "amend1_score.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in out["seeds"].items() if k != "seed1"}, indent=1, default=float), flush=True)
    print(json.dumps({"B1": out["saved"]["B1"], "E_R10": out["saved"]["E_R10"], "A1": out["openings"]["A1"], "A2": out["openings"]["A2"]}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()

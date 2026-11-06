"""E118: every Qwen VPD row beside every comparator, from files on disk, CPU only and no torch:
results/table.json and results/table.md. Run any time (before, during or after run.py's jobs):
  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/qwen_comparator_table/build.py

A row is one VPD reading's state: E93's table, E94's D3 rows and Amendment 1, E104's D3 rows and Amendment 1's copy,
E107 Amendment 1's VPD sets. Its comparator cells: E92's size, rank and steering, E102's carrying set, each read from
the file that holds it (E92, E102, E103, E107, or this experiment's run.py) and carrying that file and key. A cell
with no file reads "not computed" or "does not apply", with the reason. No rule is applied: the table is a join.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from commute.jobs import EXPERIMENTS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
E92S, E102R = EXPERIMENTS / "qwen_comparators" / "results" / "states", EXPERIMENTS / "qwen_carrying" / "results"
KAP, LVS = EXPERIMENTS / "known_answer_pair" / "results", EXPERIMENTS / "label_vs_style" / "results"
GSM8K = [f"{a}_s{s}_{t}" for a, s in (("R", 0), ("N", 0), ("F", 0), ("R", 70), ("N", 70)) for t in (10, 100)]
E92_NAME = {"a1_s0": "anchor1_s0_step100", "a1_s1": "anchor1_s1_step100", "a2_s42_full": "anchor2_s0_step1250",
            "a2_s142_full": "anchor2_s142_step1250", "a2_s42_matched": "anchor2_s0_step29", "a2_s142_matched": "anchor2_s142_step30",
            "a2_s42_past": "anchor2_s0_step550", "a2_s142_past": "anchor2_s142_step500",
            **{g: g.rsplit("_", 1)[0] + "_step" + g.rsplit("_", 1)[1] for g in GSM8K}}
E102_NAMES = ("a1_s0", "a1_s1", "a2_s42_matched", "a2_s142_matched", "a2_s42_full", "a2_s142_full")
STORED_SETS = {"known_s42_matched": KAP / "neurons_amend1" / "a1_matched__real.json",
               "known_s142_matched": KAP / "neurons_amend2" / "known_s142_matched__real.json",
               "wrong_s42_matched": LVS / "sets" / "WRONG42__real.json", "wrong_s142_matched": LVS / "sets" / "WRONG142__real.json"}
GOLD = {*GSM8K, "u1", "u2"}
GOLD_NOTE = ("GSM8K test 0-255 with gold solutions (E92's departure). On the D3 maths rows the rank does not apply: they are "
             "W0's own greedy answers, rank 1 by construction")
CONSTRUCTED = {
    **dict.fromkeys(("gauss_small", "gauss_big", "alive_small", "alive_big"),
                    "E94's random change at a matched size, built in its job, not a trained state another lead saved"),
    **dict.fromkeys(("S1", "S2", "S1a1", "S3"), "E104's scaled copy of a trained state (rules.py COPIES: S1, S1a1 and S2 RL seed 0 "
                    "step 100, S3 anchor 2 seed 42 full), built in its job, not a trained state another lead saved"),
}


def rel(p: Path) -> str:
    return str(p.relative_to(EXPERIMENTS))


def cell(value: dict[str, Any], source: Path, key: str, note: str | None = None) -> dict[str, Any]:
    return {"value": value, "source": f"{rel(source)}#{key}"} | ({"note": note} if note else {})


def missing(kind: str, why: str) -> dict[str, Any]:
    return {kind: why}


# ----------------------------------------------------------------------------- the comparator cells of one state


def e92_record(state: str) -> tuple[dict[str, Any], Path] | None:
    p = E92S / f"{E92_NAME[state]}.json" if state in E92_NAME else RESULTS / "compare" / f"{state}.json"
    return (json.loads(p.read_text()), p) if p.is_file() else None


def e92_cells(state: str) -> dict[str, dict[str, Any]]:
    got = e92_record(state)
    if got is None:
        why = "not computed yet: E118 run.py --part compare"
        return {k: missing("not computed", why) for k in ("size", "rank", "steering")}
    r, p = got
    t, g = r["size"]["total"], r["gain"]["all"]
    sv = r["steering"]
    steer = (cell({"layer": sv["layer"], "share": sv["steer"]["all"]["share"], "share_se": sv["steer"]["all"]["share_se"],
                   "random_share": sv["random"]["all"]["share"], "random_se": sv["random"]["all"]["share_se"]}, p, "steering")
             if sv["layer"] is not None else missing("undefined", "E92's rule: the state's gain is within twice its row se of 0 on a half, so no vector"))
    return {"size": cell({"ratio_norm": t["ratio_norm"], "norm": t["norm"]}, p, "size.total"),
            "rank": cell({"rows": r["rows"], "W0_median_rank_first": r["rank_W0"]["first"]["median_rank"],
                          "W0_mean_logp_all": r["rank_W0"]["all"]["mean_logp"], "gain_all": g["gain"], "gain_all_se": g["gain_se"]},
                         p, "rank_W0, gain.all", GOLD_NOTE if state in GOLD else None),
            "steering": steer}


def carry_cell(state: str) -> dict[str, Any]:
    if state in E102_NAMES:
        p, rec = E102R / "states" / f"{state}.json", json.loads((E102R / "states" / f"{state}.json").read_text())
        sc = json.loads((E102R / "score.json").read_text())["per_state"][state]
        return cell(carry_value(rec, sc), p, "carry; score.json per_state")
    if state in STORED_SETS:
        p = STORED_SETS[state]
        rec = json.loads(p.read_text())
        return cell({"rows": rec["rows"], "n": len(rec["carry"]), "gain": rec["gain"]["gain"]}, p, "carry",
                    "E103 / E107 read the gain and the set only: neuron share, k_half, engaged and prompt overlap not computed")
    p = RESULTS / "carry" / f"{state}.json"
    if not p.is_file():
        return missing("not computed", "not computed yet: E118 run.py --part carry")
    rec = json.loads(p.read_text())
    return cell(carry_value(rec, rec), p, "carry")


def carry_value(rec: dict[str, Any], sc: dict[str, Any]) -> dict[str, Any]:
    return {"rows": rec["rows"], "n": len(rec["carry"]), "gain": rec["gain"]["gain"], "neuron_share": rec["neuron_share"],
            "k_half": rec["k_half"], "engaged": sc["engaged"], "prompt_overlap": sc["prompt_overlap"]}


def comparators(state: str) -> dict[str, dict[str, Any]]:
    if state in CONSTRUCTED:
        return {k: missing("does not apply", CONSTRUCTED[state]) for k in ("size", "rank", "steering", "carrying")}
    return e92_cells(state) | {"carrying": carry_cell(state)}


# ----------------------------------------------------------------------------- the VPD rows


def val(x: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        x = (x or {}).get(k)  # type: ignore[assignment]
    return x


def e93_rows() -> list[dict[str, Any]]:
    p = EXPERIMENTS / "qwen_d1_table" / "results" / "table.json"
    t = json.loads(p.read_text())
    to = {"a2_s42": "a2_s42_full", "a2_s142": "a2_s142_full", **{g.rsplit("_", 1)[0]: g for g in GSM8K if g.endswith("_100")}}
    out = []
    for k, s in t["states"].items():
        vpd = {"arm": s["arm"], "arm_verdict": t["arms"][s["arm"]], "rules": {r: v["result"] for r, v in s["rules"].items()},
               "rule2_share": s["rules"]["2"]["value"]}
        out.append({"source": "E93", "key": k, "state": to.get(k, k), "vpd": cell(vpd, p, f"states.{k}")})
    return out


def d3_vpd(x: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: val(x, k, "value") for k in keys}


def e94_rows() -> list[dict[str, Any]]:
    d = EXPERIMENTS / "qwen_d3_rows" / "results"
    sizes = json.loads((d / "set_overlap.json").read_text())["sizes"]
    keys = ("direct_maths", "outside", "placement", "think")
    out = []
    for src, f in (("E94", "table.json"), ("E94 Am1", "table_amend1.json")):
        nums = json.loads((d / f).read_text())["numbers"]
        for k, x in nums.items():
            if src == "E94 Am1" and any(r["key"] == k for r in out):
                continue  # anchor rows repeated in Amendment 1's table: E94's own row stands for them
            vpd = d3_vpd(x, keys) | ({"set_size": sizes[k]} if k in sizes else {})
            out.append({"source": src, "key": k, "state": k, "vpd": cell(vpd, d / f, f"numbers.{k}; set_overlap.json sizes")})
    return out


def e104_rows() -> list[dict[str, Any]]:
    d = EXPERIMENTS / "qwen_d3_gsm8k" / "results"
    keys = ("outside", "placement", "think")
    out = [{"source": "E104", "key": k, "state": k, "vpd": cell(d3_vpd(x, keys), d / "table.json", f"numbers.{k}")}
           for k, x in json.loads((d / "table.json").read_text())["numbers"].items()]
    copy = json.loads((d / "amend1.json").read_text())["copy"]
    return out + [{"source": "E104 Am1", "key": "copy", "state": "S1a1", "vpd": cell(d3_vpd(copy, keys), d / "amend1.json", "copy")}]


def e107_rows() -> list[dict[str, Any]]:
    p = LVS / "amend1_sets.json"
    return [{"source": "E107 Am1", "key": k, "state": k, "vpd": cell({"set_size": n}, p, f"sizes.{k}")}
            for k, n in json.loads(p.read_text())["sizes"].items()]


# ----------------------------------------------------------------------------- the files


def f(x: Any, d: int = 3) -> str:
    if x is None:
        return "None"
    return f"{x:.{d}g}" if isinstance(x, float) else str(x)


def show(c: dict[str, Any], fmt: Callable[[dict[str, Any]], str]) -> str:
    if "value" in c:
        return fmt(c["value"]) + (" *" if "note" in c else "")
    kind = next(iter(c))
    return {"not computed": "pending" if "yet" in c[kind] else "not computed", "does not apply": "n/a", "undefined": "undefined"}[kind]


def vpd_text(v: dict[str, Any]) -> str:
    if "arm_verdict" in v:
        return f"{v['arm_verdict']}; share {f(v['rule2_share'])}"
    parts = [f"{k.split('_')[0]} {f(v[k])}" for k in ("direct_maths", "outside", "think") if k in v]
    return ", ".join(parts + ([f"set {v['set_size']}"] if "set_size" in v else []))


COLUMNS = [
    ("VPD", lambda r: show(r["vpd"], vpd_text)),
    ("‖δW‖/‖W0‖", lambda r: show(r["size"], lambda v: f(v["ratio_norm"]))),
    ("W0 rank (first) / gain", lambda r: show(r["rank"], lambda v: f"{f(v['W0_median_rank_first'])} / {f(v['gain_all'])}")),
    ("steer (random) share", lambda r: show(r["steering"], lambda v: f"{f(v['share'])} ({f(v['random_share'])})")),
    ("carrying n / share / engaged", lambda r: show(r["carrying"], lambda v: " / ".join(f(v.get(k)) for k in ("n", "neuron_share", "engaged")))),
]


def markdown(rows: list[dict[str, Any]]) -> str:
    head = ["source", "row", "state", *(c for c, _ in COLUMNS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join([r["source"], r["key"], r["state"], *(fn(r) for _, fn in COLUMNS)]) + " |" for r in rows]
    legend = ("pending = not computed yet, E118's GPU job writes it; n/a = does not apply (a constructed control, table.json "
              "has the reason); undefined = E92's steering rule gives no vector; * = a note in table.json (gold rows for the "
              "GSM8K rank; E103/E107 sets without neuron share or engaged share). Every cell's file and key: table.json.")
    return "# E118: Qwen VPD rows beside the comparators\n\n" + "\n".join(lines) + "\n\n" + legend + "\n"


def build() -> dict[str, Any]:
    rows = [r | comparators(r["state"]) for r in (*e93_rows(), *e94_rows(), *e104_rows(), *e107_rows())]
    count = {k: sum(1 for r in rows for c in ("size", "rank", "steering", "carrying") if k in r[c])
             for k in ("value", "not computed", "does not apply", "undefined")}
    out = {"note": "One row per VPD reading's state; comparator cells from the files named in each 'source'. No rule.",
           "cells": count, "rows": rows}
    (RESULTS / "table.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    (RESULTS / "table.md").write_text(markdown(rows))
    return count


if __name__ == "__main__":
    print(json.dumps(build()))

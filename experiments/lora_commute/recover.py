"""Rebuild results/lora_commute.json from the partial download and the run's own log.

The pod was terminated (2026-10-19) after the run had finished but before the final results/lora_commute.json and
results/w0_local_seeds.json were downloaded; the last downloaded JSON (results/lora_commute_partial_download.json, dumped
after the pb_r8 arm) lacks pb_r32, pb_full, pb_all, the seed arms, the Jaccards and the score. Every one of those was
also printed to results/large/full.log by run.py's log lines, so this script parses the full-run section of the log and
writes the merged file. Recovered records carry "recovered_from_log": true and hold only what the log line holds (route
shares and cells' P(o), Delta change, n_on, moved counts, Pile ratios, lr sweep summaries, G2 and cos_init); the per-step
curves, the task-local harness numbers and the top-20 lists of those arms are lost. The score dict and the W0 seeds'
task-local numbers are the log's verbatim values.

Usage (repo root):  uv run python experiments/lora_commute/recover.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PARTIAL = RESULTS / "lora_commute_partial_download.json"
LOG = RESULTS / "large" / "full.log"
OUT = RESULTS / "lora_commute.json"

ARM = re.compile(r"^\S+ \[(?P<name>[^\]]+)\] gain (?P<gain>\S+) d (?P<d>\S+) r (?P<r>\S+) i (?P<i>\S+) e_gate (?P<e_gate>\S+) "
                 r"e_delta (?P<e_delta>\S+) sum (?P<sum>\S+) \| P\(o\) target (?P<T1>\S+) unmasked (?P<U1>\S+) ci (?P<L11>\S+) "
                 r"\| Delta change (?P<dc>\S+) n_on (?P<n_on>\d+)(?: moved (?P<moved>\d+)/(?P<ntr>\d+))?(?: \| pile (?P<pile>\{.*\}))?$")
LR = re.compile(r"^\S+ \[(?P<name>[^\]]+)\] lr rule: (?P<status>selected|NONE PASSES; reporting) lr (?P<lr>\S+) (?P<sweep>\{.*\})$")
G2 = re.compile(r"^\S+ \[(?P<name>\S+) step (?P<step>\d+)\] G2 (?P<g2>\{.*\}) inexpressible (?P<inex>\S+) \(null (?P<null>\S+)\) cos_init (?P<cos>\S+)$")
LOCAL = re.compile(r"(\w+): kl_t (\S+) kl_o (\S+) lo (\S+)")


def arm_record(m: re.Match[str]) -> dict[str, Any]:
    g = m.groupdict()
    rec: dict[str, Any] = {
        "recovered_from_log": True,
        "routes": {"gain": float(g["gain"]), "share_sum": float(g["sum"]), "scorable": float(g["gain"]) >= 1.0,
                   "shares": {k: float(g[k]) for k in ("d", "r", "i", "e_gate", "e_delta")},
                   "p_cells": {k: float(g[k]) for k in ("T1", "U1", "L11")}},
        "delta_change_frac": float(g["dc"]), "n_on": int(g["n_on"])}
    if g["moved"]:
        rec["moved"] = {"n_moved": int(g["moved"]), "n_trainable": int(g["ntr"])}
    if g["pile"]:
        rec["pile"] = json.loads(g["pile"])
    return rec


def main() -> None:
    out = json.loads(PARTIAL.read_text())
    lines = LOG.read_text().splitlines()
    start = max(i for i, ln in enumerate(lines) if "bases and input moments" in ln)  # the full run's load_all
    full = lines[start:]
    arms: dict[str, dict[str, Any]] = {}
    sweeps: dict[str, dict[str, Any]] = {}
    g2: dict[str, dict[str, Any]] = {}
    for ln in full:
        if (m := ARM.match(ln)):
            arms[m["name"]] = arm_record(m)
        elif (m := LR.match(ln)):
            sw = json.loads(m["sweep"])
            sweeps[m["name"]] = {"selected_lr": float(m["lr"]), "learned": m["status"] == "selected",
                                 "sweep": {lr: {"p_o": v[0], "kl_other": v[1]} for lr, v in sw.items()}}
        elif (m := G2.match(ln)):
            g2[f"{m['name']}:{m['step']}"] = {"G2": ast.literal_eval(m["g2"]), "inexpressible": {"pooled": float(m["inex"]), "pooled_null": float(m["null"])},
                                              "cos_init": float(m["cos"])}
    S = out["sizes"]
    recovered: list[str] = []
    for name in ("pb_r32", "pb_full", "pb_all"):
        if name not in out["pb"]:
            rec = {**sweeps[name], "recovered_from_log": True, "rank": 32 if name == "pb_r32" else None,
                   "steps": {str(s): arms[f"{name}_step{s}"] for s in S["pb_save"]}}
            out["pb"][name] = rec
            recovered.append(name)
    # seed arms
    lo = {"recovered_from_log": True, "scored_step": 10, "learned": True, "steps": {}, "cos_init": {}}
    for step in (10, 100):
        k = f"lora_r8_s1:{step}"
        lo["steps"][str(step)] = {"nothing": arms[f"lora_r8_s1_step{step}_nothing"], "lift": arms[f"lora_r8_s1_step{step}_lift"],
                                  "G2": g2[k]["G2"], "inexpressible": g2[k]["inexpressible"]}
        lo["cos_init"][str(step)] = g2[k]["cos_init"]
    pb = {**sweeps["pb_r8_s1"], "recovered_from_log": True, "rank": 8, "steps": {str(s): arms[f"pb_r8_s1_step{s}"] for s in S["pb_save"]}}
    out["seeds"] = {"lora_r8_s1": lo, "pb_r8_s1": pb}
    recovered += ["seeds.lora_r8_s1", "seeds.pb_r8_s1"]
    # score, verbatim, and the W0 task-local seeds
    score_line = next(ln for ln in reversed(full) if "lora_commute.json" in ln and "score {" in ln)
    out["score"] = json.loads(score_line.split("score ", 1)[1])
    out["seconds"] = float(re.search(r"\((\d+) s\)", score_line).group(1))  # type: ignore[union-attr]
    w0_idx = [i for i, ln in enumerate(full) if "[local]" in ln and i > full.index(score_line)]
    per_seed = {}
    for seed, i in enumerate(w0_idx[:3]):
        per_seed[str(seed)] = {v: {"kl_trigger": float(a), "kl_other": float(b), "lo_trigger": float(c)} for v, a, b, c in LOCAL.findall(full[i])}
    spread_line = next(ln for ln in reversed(full) if "w0seeds: spread ratios" in ln)
    out["w0_local_seeds"] = {"per_seed": per_seed, "spread_ratio_max_over_min": json.loads(spread_line.split("spread ratios ", 1)[1]),
                             "recovered_from_log": True}
    out["recovered"] = {"from": str(LOG.relative_to(HERE.parent.parent)), "records": recovered + ["score", "w0_local_seeds"],
                        "why": "pod terminated before the final JSON and w0_local_seeds.json were downloaded (2026-10-19)"}
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {OUT}: recovered {out['recovered']['records']}")


if __name__ == "__main__":
    main()

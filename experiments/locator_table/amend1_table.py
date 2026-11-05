"""E97 Amendment 1's table, results/table_amend1.json: E97's rows (`table.rows` on its stage files; stage 3's labels
re-read with the edges' nominees added) plus the edge columns. Arithmetic on the stage files only; no model.

  top_edge            the state's gain-path edge of largest score (a tie: none)
  direct_share        the read part's edge to the logits as a share of the edges' G (a pair: both edges summed)
  cross_faith         the state on its own pair with every edge outside W0's C circuit patched: the share of its gap kept
  cross_faith_reading against the band stage 1 set from copy and lookup
  c_circuit_has_part  every part of the read part writes an edge of W0's C circuit
  units_side_by_side  stage 2's rows: the removal at the read part in both units, the state's fall and W0's (E97's reads)
`verdict` stays as E97 leaves it, for baseline-lead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import job as JB
import table
from job import RESULTS, dump, load, say
from pairs import c_pair, own_pair

from commute import edge_rules as ER
from commute.edges import mask_of

OUT = {k: RESULTS / f"amend1_stage{k}.json" for k in range(4)}
TABLE = RESULTS / "table_amend1.json"


def columns(row: Mapping[str, Any], e: Mapping[str, Any] | None, w0: Mapping[str, Any] | None, band: Mapping[str, Any] | None) -> dict[str, Any]:
    if e is None:
        return {"top_edge": None, "direct_share": math.nan, "cross_faith": math.nan, "c_circuit_has_part": None, "edge_reason": "edges not read (the plan's tree, or the stage has not run)"}
    g, at = e["gain"], row.get("read_at")
    out = {"top_edge": g["top_edge"], "direct_share": ER.direct_share({f"{p}->logits": x for p, x in g["to_logits"].items()}, at, g["G"]),
           "edge_G": g["G"], "edges_how": g["how"], "edge_nominees": e.get("edge_nominees", [])}  # fmt: skip
    cf = e.get("cross", {}).get("own", {}).get("faith", math.nan)
    if w0 is None or "own_pair" not in e:
        return out | {"cross_faith": math.nan, "c_circuit_has_part": None, "edge_reason": "no input pair: the gain path only"}
    return out | {"cross_faith": cf, "cross_faith_reading": ER.band_reading(cf, band) if band else "no band",
                  "c_circuit_has_part": ER.has_parts(mask_of(w0.get("edges", [])), at), "w0_c_circuit": {k: w0.get(k) for k in ("n", "n_edges", "gap", "reason")},
                  "own_circuit_n": e["circuit"].get("n"), "iou_with_w0_c": e.get("iou_with_w0_c"), "cross_faith_beside": {k: c["faith"] for k, c in e["cross"].items() if k != "own"}}  # fmt: skip


def side_by_side(r: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The removal at the read part in both units: the share, the state's own fall and W0's own fall, in nats."""
    if r is None:
        return None
    return {u: {"share": r["share"][u], "fall_nats": r["fall"][u], "w0_fall_nats": r.get("w0_move", {}).get(u, math.nan)} for u in ("lo", "gap")}


def write() -> None:
    s = {k: load(RESULTS / f"stage{k}.json") for k in (1, 2, 3)}
    a = {k: load(OUT[k]) for k in (1, 2, 3)}
    if a[3].get("located"):
        s[3] = {**s[3], "located": a[3]["located"]}
    rows = table.rows({k: js for k, js in s.items() if "located" in js})
    band = a[1].get("band")
    for r in rows:
        A, k = a[r["stage"]], JB.sid(r["task"], r["seed"])
        cp = c_pair(r["task"]) if own_pair(r["task"]) else None
        r |= columns(r, A.get("states", {}).get(k), A.get("w0", {}).get(cp) if cp else None, band)
        if r["stage"] == 2:
            r["units_side_by_side"] = side_by_side(table.rec_of(s[2], r["task"], r["seed"])["removal"].get(r["read_at"]))
    dump(TABLE, {"table": rows, "limit": "a new circuit built from old parts still reads as reuse", "band": band, "outcome": a[1].get("outcome"),
                 "verdict": "left for baseline-lead, except 'does not apply' on constant-answer rows"})  # fmt: skip
    for r in rows:
        say("table_amend1", r)

"""E214 circuit recovery: per item and level, the parts each item is known by construction to use, and for each whether
it is in K* (main search, first order or lowest-KL tie), its s under the main grade (predictions.json "main_grade"),
kept (s >= 1 - s_tol) and changed (in C), with R, f and the remainder share beside. Known parts:
    reuse_small, reuse_large  E62's circuit heads (circuit_heads; (2,4) is reuse_large's gain head) and the eight slots
    new_few                   the eight slots (layer slot_layer, results/edits.json "slots")
    new_thin                  layer slot_layer's MLP (every neuron's column moves a little)
    x2, x10, moved            E195's eight C neurons in layer e195_layer (moved: also the eight appended copies)
    rebuilt                   C, and the MLPs of layers 2 and 3
    e201, last_*              the grafted head (graft_layer, head 6); neurons: the graft layer's MLP
    always_on, cue, xor, table   the piece
A neuron or head is named at the level that holds it (its MLP layer, attention layer or block at coarser levels).
Writes results/recovery.json. No GPU:

    python experiments/kstar_levels/recovery.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"
D_MLP, N_HEAD = 3072, 6
Unit = tuple[str, int, int]  # ("head", l, h), ("neuron", l, i), ("mlp", l, 0), ("piece", 0, 0)


def name(u: Unit, level: str) -> str | None:
    k, l, i = u
    if k == "piece":
        return "piece"
    if level == "block":
        return f"L{l}.block"
    if k == "head":
        return f"L{l}.attn" if level == "mlp_attn" else f"L{l}.h{i}"
    if level == "neuron_head":
        return f"L{l}.n{i}" if k == "neuron" else None  # a whole MLP is no single variable there
    return f"L{l}.mlp"


def known(item: str, st: dict[str, Any], slots: list[int]) -> list[Unit]:
    s0, C, l1 = st["slot_layer"], st["e195_C"], st["e195_layer"]
    heads: list[Unit] = [("head", l, h) for l, h in st["circuit_heads"]]
    slot_units: list[Unit] = [("neuron", s0, i) for i in slots]
    c_units: list[Unit] = [("neuron", l1, i) for i in C]
    table: dict[str, list[Unit]] = {
        "reuse_small": heads + slot_units, "reuse_large": heads + slot_units, "new_few": slot_units, "new_thin": [("mlp", s0, 0)],
        "x2": c_units, "x10": c_units, "moved": c_units + [("neuron", l1, D_MLP + i) for i in range(len(C))],
        "rebuilt": c_units + [("mlp", 2, 0), ("mlp", 3, 0)], "neurons": [("mlp", st["graft_layer"], 0)]}  # fmt: skip
    if item in ("e201", "last_lowest", "last_rank5", "last_rank9", "last_rank13"):
        return [("head", st["graft_layer"], N_HEAD)]
    if item in ("always_on", "cue", "xor", "table"):
        return [("piece", 0, 0)]
    return table.get(item, [])


def recovery(it: dict[str, Any], units: list[Unit], g: str, tol: float) -> dict[str, Any]:
    rec = it["searches"]["main"][0]
    inK = {v["name"]: v for v in rec["K"]}
    s_of = it.get("s", {})
    parts = []
    for n in dict.fromkeys(x for u in units if (x := name(u, it["level"])) is not None):
        s = inK[n][f"s_{g}"] if n in inK else s_of[n][0 if g == "all" else 1] if n in s_of else None
        parts.append({"name": n, "in_K": n in inK, f"s_{g}": s, "kept": None if s is None else s >= 1 - tol,
                      "changed": inK[n]["changed"] if n in inK else n in it["changed"]})  # fmt: skip
    return {"parts": parts, "found": sum(p["in_K"] for p in parts), "of": len(parts), "K_size": rec["size"], "R": rec[f"R_{g}"], "f": rec["f"],
            "remainder_share": rec["remainder_share"], "K": [v["name"] for v in rec["K"]]}  # fmt: skip


def main() -> None:
    pred = json.loads((RESULTS / "predictions.json").read_text())
    g, st, tol = pred["main_grade"], pred["settings"], pred["thresholds"]["s_tol"]
    edits = RESULTS / "edits.json"
    slots = json.loads(edits.read_text())["slots"] if edits.is_file() else []
    out: dict[str, Any] = {"experiment": pred["experiment"], "main_grade": g, "items": {}}
    for p in sorted((RESULTS / "items").glob("*.json")):
        it = json.loads(p.read_text())
        if p.stem.endswith(".stopped") or it.get("reading") != "read":
            continue
        r = recovery(it, known(it["item"], st, slots), g, tol)
        out["items"].setdefault(it["item"], {})[it["level"]] = r
        print(f"{it['item']} {it['level']}: {r['found']}/{r['of']} known parts in K* (|K*| {r['K_size']}), R {r['R']:.6f}, f {r['f']:.3f}; "
              + ", ".join(f"{x['name']} {'in' if x['in_K'] else 'out'} s {x[f's_{g}']}" for x in r["parts"][:12]))
    (RESULTS / "recovery.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

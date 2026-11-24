"""E299 (known_3seeds): the rows Fig 4 (known_main) and App. I (known_appendix) are replotted from, exactly seeds 0-2 a side
on the nine built pairs, one task variant a side, one form of each reading. $0, CPU, saved files only:

    ~/commute-agents/fleet/cpu-run python experiments/known_3seeds/assemble.py

Writes results/known3.json: {"cells": [{"pair", "method", "reuse": [s0, s1, s2], "new": [...], "states": {"reuse": [...],
"new": [...]}, "source"}], "missing": [...], "checks": {...}}. A value not yet read (the five seed-2 states before E299's
stage 2) is null and its state is listed in "missing". Old states are read from the files each reading was saved in
(the method table's sources, method_table/results/final.json); the five new ones from results/stage2.json. Forms, as
make_known_scorecard.py plots them: steering pooled (p270_rescore 'pooled'), top-neuron share k 64, size ||dW|| / ||W0||
over the 24 matrices (E187, size_matched m67.json), prior one value per task repeated for each seed. Variants: e46 is
ident32 against rand32 (ident32b, rand32b dropped), e53 lowercase against cipher (broad_cipher dropped). "checks" compares
every old value with the cell final.json (or, for steering and size, p270_rescore) reported, which held more seeds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGE2 = "experiments/known_3seeds/results/stage2.json"
FINAL = REPO / "experiments/method_table/results/final.json"
P270 = Path("/mnt/nw/home/e.agnew/commute-agents/compare-lead/experiments/p270_rescore/results/rescore.json")
SEEDS = (0, 1, 2)
# pair -> (reuse task, new task), tasks in E136's state names (group/stem); E54's mid task and the 24-key cipher are shared
PAIRS = {"e62_copy_lookup": ("e62/copy", "e62/lookup"), "e46_copy_random": ("e46/ident32", "e46/rand32"),
         "e53_lowercase_cipher": ("e53/lowercase", "e53/cipher"), "e54_matched": ("e54/elicit", "e54/cipher"),
         "e54_own": ("e54/elicit", "e54/cipher_own"), "open_pair_c3": ("c3/rank1", "c3/cipher"),
         "open_pair_c3_copy": ("c3/copy", "c3/cipher"), "e58_rank1_96": ("e58/rank1", "e58/cipher"),
         "e58_copy_prior_cipher": ("e58/copy", "e58/prior_cipher")}  # fmt: skip
METHODS = ("share", "steer", "erank", "prior", "neuron", "neuron_top", "k_half", "size")
NEW = {"e46/ident32_s2", "e46/rand32_s2", "e53/lowercase_s2", "e53/cipher_s2", "e54/cipher_own_s2"}
FINAL_METHOD = {"share": "share", "erank": "erank", "prior": "prior", "neuron": "neuron", "neuron_top": "neuron_top_k64", "k_half": "k_half"}
_json: dict[str, Any] = {}

Src = tuple[str, list[str | int]]  # (file, key path; an int indexes a list)


def load(rel: str | Path) -> Any:
    key = str(rel)
    if key not in _json:
        p = rel if isinstance(rel, Path) else REPO / rel
        _json[key] = json.loads(p.read_text()) if p.is_file() else None
    return _json[key]


def e98_name(state: str) -> str:
    """E98's row name (neuron_r cells.json, carrying.json; k_half_fill cells.json) of an E136 state name."""
    group, rest = state.split("/")
    return {"e46": rest, "e54": rest, "e53": rest if rest.startswith("lowercase") else f"e53_{rest}"}.get(group, f"{group}_{rest}")


def prior_name(task: str) -> str:
    group, stem = task.split("/")
    return {"e58": f"e58_96_{stem}", "c3": f"e58_24_{stem}", "e54": "e54_cipher" if stem == "cipher_own" else f"e54_{stem}"}.get(group, f"{group}_{stem}")


def size_names() -> dict[str, list[str]]:
    """Per pair, E187's state order (size_matched m67.json .pairs.<p>.sizes, reuse then new), as p270_rescore's state_names."""
    e136, e147 = load("experiments/share_untrained/results/inventory.json"), load("experiments/refit_96key/results/inventory.json")
    out = {p["id"]: p["reuse"] + p["new"] for p in e136["pairs"]}
    out |= {p["id"]: [f"e58/{p[s]}_s{i}" for s in ("reuse", "new") for i in e147["seeds"]] for p in e147["pairs"]}
    return out


def source(method: str, pair: str, side: str, state: str) -> Src:
    """Where one state's reading is saved."""
    group, rest = state.split("/")
    if state in NEW:
        k = {"share": ["e136", "trained"], "steer": ["holes_fill", "steering", "pooled", "share"], "erank": ["spectrum", "erank"],
             "neuron": ["neuron_r", "r_n"], "neuron_top": ["holes_fill", "neuron", "k64", "share"], "k_half": ["k_half", "k_half"], "size": ["size"]}[method]  # fmt: skip
        return STAGE2, ["states", state, *k]
    if group == "e58":
        r96 = {"share": ["e136", "trained"], "steer": ["holes_fill", "steering", "pooled", "share"], "neuron": ["neuron_r", "r_n"],
               "neuron_top": ["holes_fill", "neuron", "k64", "share"]}  # fmt: skip
        if method in r96:
            return "experiments/refit_96key/results/stage1.json", ["states", state, *r96[method]]
        if method == "erank":
            return "experiments/refit_rank/results/cells.json", ["states", rest, "erank"]
        if method == "k_half":
            return "experiments/k_half_fill/results/cells.json", ["states", state, "k_half"]
    if method == "share":
        return "experiments/share_untrained/results/stage1.json", ["states", state, "trained"]
    if method == "steer":
        if group == "e62":
            return "experiments/steering_copy_lookup/results/stage1.json", ["states", rest, "steering", "pooled", "share_lo"]
        return "experiments/holes_fill/results/stage1.json", ["states", state, "steering", "pooled", "share"]
    if method == "erank":
        return "experiments/update_svd/results/results.json", ["pairs", pair, side, rest, "erank"]
    if method == "neuron":
        return "experiments/neuron_r/results/cells.json", ["rows", e98_name(state), "r_n"]
    if method == "neuron_top":
        return "experiments/holes_fill/results/stage1.json", ["states", state, "neuron", "k64", "share"]
    if method == "k_half":
        n = e98_name(state)
        if n in load("experiments/neuron_r/results/carrying.json")["rows"]:
            return "experiments/neuron_r/results/carrying.json", ["rows", n, "k_half"]
        return "experiments/k_half_fill/results/cells.json", ["states", n, "k_half"]
    assert method == "size", method
    i = size_names()[pair].index(state)
    sizes = load("experiments/size_matched/results/m67.json")["pairs"][pair]["sizes"]
    n_reuse = len(sizes["reuse"])
    return "experiments/size_matched/results/m67.json", ["pairs", pair, "sizes", *(["reuse", i] if i < n_reuse else ["new", i - n_reuse])]


def value(src: Src) -> float | None:
    file, keys = src
    x = load(file)
    for k in keys:
        if x is None:
            return None
        x = x[k] if isinstance(k, int) else x.get(k)
    return None if x is None else float(x)


def prior(task: str) -> tuple[float, str]:
    if task.startswith("e53/"):
        stem = task.split("/")[1]
        return load("experiments/prior_lowercase/results/stage1.json")["tasks"][stem]["log10_prior"], f"experiments/prior_lowercase/results/stage1.json .tasks.{stem}.log10_prior"
    name = prior_name(task)
    v = next(p["log10_prior"] for p in load("experiments/prior_vs_share/results/stage2.json")["points"] if p["name"] == name)
    return v, f"experiments/prior_vs_share/results/stage2.json .points[] name {name} .log10_prior"


def pattern(src: Src, state: str) -> str:
    """The source as file + key path, the state's own key written <state>."""
    stems = {state, state.split("/")[1], e98_name(state)}
    keys = ["<state>" if str(k) in stems else ("<i>" if isinstance(k, int) else str(k)) for k in src[1]]
    return f"{src[0]} .{'.'.join(keys)}"


def cell(pair: str, method: str) -> tuple[dict[str, Any], list[str]]:
    rec: dict[str, Any] = {"pair": pair, "method": method, "states": {}}
    missing, srcs = [], set()
    for side, task in zip(("reuse", "new"), PAIRS[pair], strict=True):
        states = [f"{task}_s{s}" for s in SEEDS]
        rec["states"][side] = states
        if method == "prior":
            v, s = prior(task)
            rec[side] = [v] * len(SEEDS)
            srcs.add(f"{s} (one value per task, repeated for each of the 3 seeds)")
            continue
        vals = []
        for st in states:
            src = source(method, pair, side, st)
            v = value(src)
            vals.append(v)
            srcs.add(pattern(src, st))
            if v is None:
                missing.append(st)
        rec[side] = vals
    rec["source"] = "; ".join(sorted(srcs))
    return rec, missing


def reported(pair: str, method: str) -> tuple[list[float], list[float]] | None:
    """The side values the figure plotted before (final.json; p270_rescore pooled steering and E187 size), for the checks."""
    if method == "size":
        return None  # read from m67.json itself, the file p270_rescore's size reads
    if method == "steer":
        p = load(P270)
        return None if p is None else (p["steering"]["pairs"][pair]["pooled"]["reuse"], p["steering"]["pairs"][pair]["pooled"]["new"])
    f = load(FINAL)
    c = next((c for c in f["cells"] if c["pair"] == pair and c["method"] == FINAL_METHOD[method]), None) if f else None
    return None if c is None else (c["reuse"], c["new"])


def checks(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Every old value must be one of the values the figure plotted before on its side (isclose rel 1e-9); a reading in a
    different form before (e46's top-neuron share was k matched; e54's steering cell was per key) is listed, not failed."""
    out: dict[str, Any] = {"found": 0, "not_found": []}
    for c in cells:
        old = reported(c["pair"], c["method"])
        if old is None:
            continue
        for side, before in zip(("reuse", "new"), old, strict=True):
            for st, v in zip(c["states"][side], c[side], strict=True):
                if v is None or st in NEW:
                    continue
                if any(math.isclose(v, float(b), rel_tol=1e-9) for b in before):
                    out["found"] += 1
                else:
                    out["not_found"].append(f"{c['pair']} {c['method']} {st} {v:.6g}")
    return out


def shared(cells: list[dict[str, Any]]) -> list[str]:
    """A task drawn in two pairs (E54's mid task; the 24-key cipher) must read the same three values in both."""
    seen: dict[tuple[str, str], list[Any]] = {}
    bad = []
    for c in cells:
        for side in ("reuse", "new"):
            key = (c["method"], c["states"][side][0])
            if key in seen and seen[key] != c[side]:
                bad.append(f"{key}: {seen[key]} vs {c[side]}")
            seen.setdefault(key, c[side])
    return bad


def near(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """predictions.json's descriptive expectation: each new state's reading inside its siblings' band [min - d, max + d],
    d = max(max - min, 0.05 x |mean|); a missing, None or nan reading is outside."""
    out: dict[str, Any] = {}
    for c in cells:
        for side in ("reuse", "new"):
            st, v, sib = c["states"][side][2], c[side][2], c[side][:2]
            if st not in NEW or c["method"] == "prior" or None in sib:
                continue
            lo, hi = min(sib), max(sib)
            d = max(hi - lo, 0.05 * abs((lo + hi) / 2))
            inside = v is not None and math.isfinite(v) and lo - d <= v <= hi + d
            out.setdefault(st, {})[c["method"]] = {"value": v, "siblings": sib, "band": [lo - d, hi + d], "inside": inside}
    return out


def main() -> None:
    cells, missing = [], []
    for pair in PAIRS:
        for m in METHODS:
            rec, miss = cell(pair, m)
            assert len(rec["reuse"]) == len(rec["new"]) == len(SEEDS), (pair, m)
            cells.append(rec)
            missing += [f"{pair} {m} {s}" for s in miss]
    out = {"experiment": "E299", "seeds": list(SEEDS), "cells": cells, "missing": missing, "checks": checks(cells), "shared_mismatch": shared(cells),
           "new_vs_siblings": near(cells)}
    (HERE / "results" / "known3.json").write_text(json.dumps(out, indent=1))
    n_full = sum(None not in c["reuse"] + c["new"] for c in cells)
    print(f"cells {len(cells)} ({len(PAIRS)} pairs x {len(METHODS)} methods), complete {n_full}, values missing {len(missing)}")
    print(f"checks: {out['checks']['found']} old values found in the earlier cells; not found {len(out['checks']['not_found'])}")
    for x in out["checks"]["not_found"]:
        print(f"  not found: {x}")
    print(f"shared sides that differ: {out['shared_mismatch']}")
    for x in sorted({m.rsplit(' ', 1)[1] for m in missing}):
        print(f"  missing state: {x}")
    for st, rows in out["new_vs_siblings"].items():
        print(f"new {st}: " + " ".join(f"{m} {r['value']} {'in' if r['inside'] else 'OUT'} {r['band']}" for m, r in rows.items()))


if __name__ == "__main__":
    main()

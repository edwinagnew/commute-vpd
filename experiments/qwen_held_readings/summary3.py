"""Cell P3.13 run 3: results/run3/summary.json from the written records (results/run3/<pair>/...), no GPU. Stage 0's
table (per fit and row set, the do-nothing KL(F||H) on the read rows, beside run 2's) and each pair's chosen fit; then,
as summary2, per pair and save each side's S per seed with its random change's S, the gaps and the filed rule
(`summary2.gap_rule`, `summary2.pair_outcome`), and the outcome over the two word pairs (`summary2.outcome`, the same
rule as predictions.json rules.outcome). A pod stopped (stopped.json: no fit cleared, or a control missed) reads
'unreadable'. No reuse or new label on any update.

    ~/commute-agents/fleet/cpu-run python experiments/qwen_held_readings/summary3.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
R3 = HERE / "results" / "run3"
S2: Any = load_experiment("qwen_held_readings", "summary2")  # gap_rule, pair_outcome, outcome, cell (E219)
S2.R2 = R3  # summary2.cell reads S records from <S2.R2>/<pair>/S/: point it at run 3's


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def sweep(st: dict[str, Any], pair: str) -> dict[str, Any]:
    """Stage 0 of one pod: per row set and fit the do-nothing KL(F||H) (None if not written), run 2's floor, the choice."""
    table = {rs: {f["name"]: read(R3 / pair / "control" / f"{rs}_fit_{f['name']}.json").get("kl_fh")
                  for f in st["fits"]} for rs in st["pairs"][pair]}
    run2 = {rs: read(HERE / "results" / "run2" / pair / "control" / f"{rs}_do_nothing.json").get("kl_fh")
            for rs in st["pairs"][pair]}
    return {"kl_fh": table, "run2_own64": run2, "choice": read(R3 / pair / "control" / "choice.json").get("chosen")}


def summarise() -> dict[str, Any]:
    P = json.loads((R3 / "predictions.json").read_text())
    st = P["settings"]
    pairs: dict[str, Any] = {}
    for pair in st["pairs"]:
        stopped = read(R3 / pair / "stopped.json")
        cells = {save: S2.cell(st, pair, save) for save in st["saves"]}
        rules = [c[k]["rule"] for c in cells.values() for k in ("saved", "matched")]
        pairs[pair] = {"reuse_side": st["pairs"][pair][0], "new_side": st["pairs"][pair][1], "stage0": sweep(st, pair),
                       "stopped": stopped or None, "saves": cells,
                       "outcome": "unreadable" if stopped else S2.pair_outcome(rules)}
        print(f"[summary] {pair} stage 0: {json.dumps(pairs[pair]['stage0'])}; stopped {stopped or None}", flush=True)
        for save, c in cells.items():
            print(f"[summary] {pair} {save}: saved {c['saved']}, matched {c['matched']}, seed sd {c['seed_sd']}", flush=True)
    out = {"experiment": P["experiment"], "cell": "P3.13", "run": 3, "smallest_gap": st["smallest_gap"],
           "floor_max": st["floor_max"], "pairs": pairs, "outcome": S2.outcome([p["outcome"] for p in pairs.values()])}
    print(f"[summary] outcome {out['outcome']}: {json.dumps({k: v['outcome'] for k, v in pairs.items()})}", flush=True)
    tmp = R3 / "summary.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(R3 / "summary.json")
    return out


if __name__ == "__main__":
    summarise()

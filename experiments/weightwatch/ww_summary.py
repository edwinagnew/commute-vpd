"""P2.73's summary: the filed rules (predictions.json "rules") over every JSON the run wrote, and the state list the run
reads. JSON only, no model: run.py's last stage calls `write`, and it also runs on the CPU:

    ~/commute-agents/fleet/cpu-run python experiments/weightwatch/ww_summary.py
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ww_score import (
    MONITORED,
    column_rule,
    gaussian_check,
    rel_within,
    spread,
    trained_reading,
)

from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
PATH = {"trigger": "trigger", "x2": "rescale", "x10": "rescale", "rebuilt": "rebuilt", "new_few": "columns", "new_thin": "dense",
        "random_size_241001": "saved", "random_kl_241002": "saved"}  # fmt: skip  (D2's build paths; E62 and panel: saved)
SHAMLESS = ("random_size_241001", "random_kl_241002")
TRAINED = ("e62/", "panel/")


def path_of(state: str) -> str:
    return PATH.get(state, "saved")


def states(panel_tasks: Sequence[str], panel_seeds: Sequence[int]) -> dict[str, str]:
    """The 50 states in run order, each with its task set (appendix B)."""
    out = {"trigger": "trigger", "x2": "copy24", "x10": "copy24", "rebuilt": "copy24", "new_few": "cue", "new_thin": "cue",
           "random_size_241001": "copy24", "random_kl_241002": "cue"}  # fmt: skip
    out |= {f"e62/{m}_s{s}": f"e62_{m}" for m in ("copy", "lookup") for s in range(3)}
    return out | {f"panel/{t}_s{s}": f"panel:{t}" for t in panel_tasks for s in panel_seeds}


def fname(state: str) -> str:
    return state.replace("/", "-")


def read(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text()) if p.is_file() else None


def rate(rec: dict[str, Any] | None, rule: str = "all_layers") -> dict[str, float] | None:
    r = (rec or {}).get(rule)
    return None if r is None else r["rates"]


def row(s: str, d: dict[str, Any] | None, it: dict[str, Any] | None, fl: dict[str, Any] | None) -> dict[str, Any]:
    d, it = d or {}, it or {}
    sham = it.get("sham") or {}
    return {"state": s, "path": path_of(s), "t": d.get("t"), "kept": {m: v["k"] for m, v in (d.get("kept") or {}).items()},
            "S": d.get("S"), "S_cov": d.get("S_cov"), "S_G": d.get("S_G"), "haar_range": [min(d["haar"]), max(d["haar"])] if d.get("haar") else None,
            "haar_below_S": d.get("haar_below_S"), "reading": trained_reading(d.get("S"), d.get("haar") or [], d.get("S_G")) if s.startswith(TRAINED) else None,
            "F_fpr": ((it.get("F") or {}).get("all_layers") or {}).get("fpr"), "F_rates": rate(it.get("F")), "F_rates_layer0": rate(it.get("F"), "layer0_only"),
            "remark31_2t_over_n": (it.get("F") or {}).get("remark31_2t_over_n"), "sham_rates": rate(sham.get("detector")), "sham_counts_match": sham.get("counts_match"),
            "sham_S": sham.get("S"), "M_floor_rates": rate(fl), "M_floor_fpr": ((fl or {}).get("all_layers") or {}).get("fpr")}  # fmt: skip


def write(results: Path) -> dict[str, Any]:
    s = json.loads((results / "predictions.json").read_text())["settings"]
    every = states(s["panel_tasks"], s["panel_seeds"])
    s0, tr, mf = read(results / "stage0.json") or {}, read(results / "trigger.json") or {}, read(results / "m_floor.json") or {}
    dirs = {x: read(results / "directions" / f"{fname(x)}.json") for x in every}
    its = {x: read(results / "items" / f"{fname(x)}.json") for x in every}
    rows = {x: row(x, dirs[x], its[x], (mf.get("states") or {}).get(x)) for x in every}
    S = {x: r["S"] for x, r in rows.items()}
    f = s0.get("x2_filed") or {}
    rates = {x: (rows[x]["F_rates"] or {}).get("copy24") for x in ("x2", "x10")}
    det = tr.get("detector") or {}
    unchanged = det.get("unchanged_wilson")  # the 95% Wilson interval of the unchanged windows' flag rate
    dw0_ok = (s0.get("dW0") or {}).get("t") == 0 and (mf.get("dW0") or {}).get("flags") == 0
    out: dict[str, Any] = {
        "rows": rows,
        "x2_x10_within_filed_bar": {x: rel_within(S[x], f.get("S"), f.get("bar")) for x in ("x2", "x10")},
        "x10_rate_at_least_x2": None if None in rates.values() else rates["x10"] >= rates["x2"],  # type: ignore[operator]
        "trigger": {"share_T": det.get("share_T"), "bar": s["trigger_bar"], "unchanged_rate": det.get("rate_unchanged"), "fpr": det.get("fpr"),
                    "unchanged_wilson": unchanged, "unchanged_at_fpr": None if unchanged is None or det.get("fpr") is None else unchanged[0] <= det["fpr"] <= unchanged[1]},
        "column_rule": column_rule(S, max([v for v in (spread((dirs[x] or {}).get("haar") or []) for x in ("x2", "new_few", "new_thin")) if v is not None], default=None),
                                   det.get("share_T"), s["trigger_bar"], dw0_ok),
        "gaussian": s0.get("gaussian") or gaussian_check([]),
        "full_fine_tunes_keep_20": [f"{x} {m} keeps {r['kept'].get(m)}" for x, r in rows.items() if x.startswith(TRAINED) and r["kept"] for m in MONITORED if r["kept"].get(m) != 20],
        "sham_count_misses": [x for x, r in rows.items() if r["sham_counts_match"] is False],
        "missing": [x for x in every if dirs[x] is None or its[x] is None],
    }  # fmt: skip
    out["e62_pair"] = e62_pair(S)
    (results / "summary.json").write_text(json.dumps(out | {"done": not out["missing"]}, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)
    return out


def e62_pair(S: dict[str, float | None]) -> dict[str, Any] | None:
    """E141's pair rule (`experiments/update_svd/run.py` pair_rule) on S: copy seeds against lookup seeds."""
    reuse, new = [S[f"e62/copy_s{i}"] for i in range(3)], [S[f"e62/lookup_s{i}"] for i in range(3)]
    if None in reuse + new:
        return None
    return load_experiment("update_svd", "run").pair_rule(reuse, new)


if __name__ == "__main__":
    write(HERE / "results")

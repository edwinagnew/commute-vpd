"""carrying_jobs (E101), stage 1's files and score. CPU only, no model. Plan:
reports/plans/2026-11-05_plan_e101-carrying-jobs.md, section "Stage 1"; rules in results/predictions.json.

  python experiments/carrying_jobs/score.py judge-inputs   after every writer file is in: per batch, the jobs and the
         six task descriptions in a fresh shuffled order under letters -> large/judge_inputs/b<NN>.json; the letters'
         meaning -> large/judge_letters.json (no agent is pointed at it)
  python experiments/carrying_jobs/score.py score          after the maximiser job and every judge file: joins the agents'
         outputs with large/key.json and applies C1, C2, P1, P2, P3 and the reported rows -> results/score.json

Agent outputs (committed; formats in briefs/*.md): results/stage1/writer/b<NN>.json {"batch", "jobs": {id: job}};
results/stage1/maximiser/b<NN>.json {"batch", "texts": {id: {"examples": [5], "near_misses": [5]}}};
results/stage1/judge/b<NN>.json {"batch", "ties": {id: [letters]}}. The maximiser job (experiments/carrying_jobs/
maximise.py, a later commit) writes results/stage1/max_scores.json {"self_test": {"win_share", ...}, "neurons": {id:
{"examples": [peak firing], "near_misses": [peak firing]}}}, each text's peak W0 firing of that neuron over its tokens.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from commute import carrying_jobs as CJ

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, STAGE1 = RESULTS / "large", RESULTS / "stage1"
P2_ROWS = {"e53_lowercase_cipher": ("lowercase", "ciphers"), "e46_copy_random": ("same_word", "random_word"),
           "e62_copy_lookup": ("e62_copy", "e62_lookup")}


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def outputs(kind: str, field: str) -> dict[str, Any]:
    """Every batch file of one role, merged: id -> its entry under `field`."""
    out: dict[str, Any] = {}
    for f in sorted((STAGE1 / kind).glob("b*.json")):
        out |= load(f)[field]
    return out


def judge_inputs(P: dict[str, Any]) -> None:
    jobs = {f.stem: load(f)["jobs"] for f in sorted((STAGE1 / "writer").glob("b*.json"))}
    key = load(LARGE / "key.json")
    assert set(jobs) == set(key["batches"]), f"writer files {sorted(jobs)} against batches {sorted(key['batches'])}"
    rng, letters = random.Random(P["settings"]["seed"] + 3), {}
    (LARGE / "judge_inputs").mkdir(parents=True, exist_ok=True)
    for b, j in jobs.items():
        names = sorted(P["descriptions"])
        rng.shuffle(names)
        letters[b] = {chr(65 + k): n for k, n in enumerate(names)}
        desc = {L: P["descriptions"][n] for L, n in letters[b].items()}
        (LARGE / "judge_inputs" / f"{b}.json").write_text(json.dumps({"batch": b, "descriptions": desc, "jobs": j}, indent=1))
    (LARGE / "judge_letters.json").write_text(json.dumps(letters, indent=1))
    print(f"judge inputs for {len(jobs)} batches in {LARGE / 'judge_inputs'}")


def neuron_table(P: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """id -> {roles, outcome ('win'/'tie'/'lose', or None unscorable), ties (description names), complete}."""
    S, key = P["settings"], load(LARGE / "key.json")
    jobs, ties = outputs("writer", "jobs"), outputs("judge", "ties")
    scores = load(STAGE1 / "max_scores.json")["neurons"]
    letters, where = load(LARGE / "judge_letters.json"), {i: b for b, ids in key["batches"].items() for i in ids}
    out = {}
    for i, k in key["neurons"].items():
        s = scores.get(i)
        res = CJ.outcome(s["examples"], s["near_misses"], k["top_act"], S["strong_frac"]) if s else None
        tied = {letters[where[i]][L] for L in ties[i]} if i in ties else set()
        out[i] = {"roles": k["roles"], "outcome": res, "ties": tied, "complete": i in jobs and i in ties and res is not None}
    return out


def members(T: dict[str, dict[str, Any]], group: str | None, role: str) -> list[dict[str, Any]]:
    return [t for t in T.values() if any(r["role"] == role and group in (None, r["group"]) for r in t["roles"])]


def side(T: dict[str, dict[str, Any]], group: str | None, role: str, desc: str | None, S: dict[str, Any]) -> tuple[float, int]:
    """(share, n) over a group's complete neurons of that role: 'win' and, when desc is given, tied to desc. n = 0 (the
    rules then read unreadable) under min_group complete neurons or over missing_max of them incomplete."""
    m = members(T, group, role)
    ok = [t for t in m if t["complete"]]
    if not m or len(ok) < S["min_group"] or 1 - len(ok) / len(m) > S["missing_max"]:
        return float("nan"), 0
    return CJ.rate([t["outcome"] == "win" and (desc is None or desc in t["ties"]) for t in ok])


def status(v: str) -> str:
    return {"above": "pass", "below_or_level": "fail", "no_call": "no call"}.get(v, "unreadable")


def score(P: dict[str, Any]) -> dict[str, Any]:
    S, G = P["settings"], P["groups"]
    T = neuron_table(P)
    floor = {d: side(T, None, "random", d, S) for d in P["descriptions"]}
    rand_ok = [t for t in members(T, None, "random") if t["complete"]]
    fp, n_fp = CJ.rate([bool(t["ties"]) for t in rand_ok])
    self_test = load(STAGE1 / "max_scores.json")["self_test"]
    out: dict[str, Any] = {
        "C1_judge_fp": {"share": fp, "n": n_fp, "ok": n_fp > 0 and fp <= S["judge_fp_max"]},
        "C2_self_test": {**self_test, "ok": self_test["win_share"] >= S["self_test_min_win"]},
        "floors": floor, "complete": sum(t["complete"] for t in T.values()), "n_neurons": len(T),
    }
    rel = lambda g: side(T, g, "carrying", G[g]["description"], S)
    p1 = CJ.gap_rule(*rel("lowercase"), *floor["lower"], S["min_gap"])
    out["P1"] = {**p1, "status": status(p1["verdict"])}
    rows = {}
    for name, (a, b) in P2_ROWS.items():
        (pa, na), (pb, nb) = rel(a), rel(b)
        fa, fb = floor[G[a]["description"]], floor[G[b]["description"]]
        rows[name] = CJ.row_kind({"p": pa, "n": na, "floor": fa[0], "n_floor": fa[1]},
                                 {"p": pb, "n": nb, "floor": fb[0], "n_floor": fb[1]}, S["min_gap"])
    readable = [n for n, r in rows.items() if r["verdict"] != "unreadable"]
    up = [n for n in readable if rows[n]["verdict"] == "above"]
    out["P2"] = {"status": "unreadable" if len(readable) < 2 else "pass" if len(up) >= 2 else "fail", "rows": rows,
                 "readable": readable, "reuse_higher": up}
    p3 = CJ.gap_rule(*side(T, "rank1", "carrying", None, S), *side(T, None, "random", None, S), S["min_gap"])
    out["P3"] = {**p3, "status": {"above": "coherent", "below_or_level": "not coherent"}.get(p3["verdict"], p3["verdict"].replace("_", " "))}
    out["R_w0_controls"] = {g: CJ.gap_rule(*side(T, g, "control", c["description"], S), *floor[c["description"]], S["min_gap"])
                            for g, c in P["controls_w0"].items()}
    out["R_explained_base"] = dict(zip(("share", "n"), side(T, None, "random", None, S), strict=True))
    out["read"] = {"P1": out["C1_judge_fp"]["ok"] and out["C2_self_test"]["ok"],
                   "P2_P3": out["C1_judge_fp"]["ok"] and out["C2_self_test"]["ok"] and out["P1"]["status"] == "pass"}
    return out


def main() -> None:
    P = load(RESULTS / "predictions.json")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "judge-inputs":
        judge_inputs(P)
    elif cmd == "score":
        out = score(P)
        (RESULTS / "score.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=sorted))
        for k in ("C1_judge_fp", "C2_self_test", "P1", "P2", "P3", "R_explained_base", "read"):
            print(k, json.dumps(out[k], sort_keys=True, default=sorted)[:600])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

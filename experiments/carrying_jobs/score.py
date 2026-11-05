"""carrying_jobs (E101), stage 1's files and score. CPU only, no model. Plan:
reports/plans/2026-11-05_plan_e101-carrying-jobs.md, sections "Stage 1" and "Amendment 1"; rules in
results/predictions.json and results/predictions_amend1.json.

  python experiments/carrying_jobs/score.py judge-inputs   after every writer file is in: per batch, the jobs and the
         six task descriptions in a fresh shuffled order under letters -> large/judge_inputs/b<NN>.json; the letters'
         meaning -> large/judge_letters.json (no agent is pointed at it)
  python experiments/carrying_jobs/score.py score          after the maximiser job and every judge file: joins the agents'
         outputs with large/key.json and applies C1, C2, P1, P2, P3 and the reported rows -> results/score.json
  python experiments/carrying_jobs/score.py amend1-inputs  after controls_amend1.py and the writer file c00: every neuron of
         large/key_amend1.json, shuffled into judge batches j<NN> with the six descriptions, each with its task windows
         (results/task_windows_amend1.json), letters reshuffled per batch -> large/judge_inputs_amend1/j<NN>.json; the
         batches' ids and letters -> large/judge_amend1.json
  python experiments/carrying_jobs/score.py amend1-score   after every judge_amend1 file: C1a, C1b, C1_fp, P1, P2 and
         the filed W0 controls under the new judge -> results/score_amend1.json

Agent outputs (committed; formats in briefs/*.md): results/stage1/writer/{b<NN>,c00}.json {"batch", "jobs": {id: job}};
results/stage1/maximiser/b<NN>.json {"batch", "texts": {id: {"examples": [5], "near_misses": [5]}}};
results/stage1/judge/b<NN>.json and results/stage1/judge_amend1/j<NN>.json {"batch", "ties": {id: [letters]}}. The
maximiser job (experiments/carrying_jobs/maximise.py) writes results/stage1/max_scores.json {"self_test": {"win_share",
...}, "neurons": {id: {"examples": [peak firing], "near_misses": [peak firing]}}}, each text's peak W0 firing of that
neuron over its tokens.
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
Table = dict[str, dict[str, Any]]


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def outputs(kind: str, field: str, pattern: str = "b*.json") -> dict[str, Any]:
    """Every batch file of one role, merged: id -> its entry under `field`."""
    out: dict[str, Any] = {}
    for f in sorted((STAGE1 / kind).glob(pattern)):
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


def neuron_table(P: dict[str, Any], key: dict[str, Any], jobs: dict[str, str], ties: dict[str, list[str]],
                 letters: dict[str, dict[str, str]], where: dict[str, str]) -> Table:
    """id -> {roles, outcome ('win'/'tie'/'lose', or None unscorable), ties (description names), judged (a judge output),
    complete (job, judge output and outcome)}. `where`: id -> its judge batch, whose `letters` decode its ties."""
    S, scores = P["settings"], load(STAGE1 / "max_scores.json")["neurons"]
    out = {}
    for i, k in key["neurons"].items():
        s = scores.get(i)
        res = CJ.outcome(s["examples"], s["near_misses"], k["top_act"], S["strong_frac"]) if s else None
        tied = {letters[where[i]][L] for L in ties[i]} if i in ties else set()
        out[i] = {"roles": k["roles"], "outcome": res, "ties": tied, "judged": i in ties,
                  "complete": i in jobs and i in ties and res is not None}
    return out


def members(T: Table, group: str | None, role: str) -> list[dict[str, Any]]:
    return [t for t in T.values() if any(r["role"] == role and group in (None, r["group"]) for r in t["roles"])]


def side(T: Table, group: str | None, role: str, desc: str | None, S: dict[str, Any], win: bool = True) -> tuple[float, int]:
    """(share, n) over a group's complete neurons of that role: 'win' and, when desc is given, tied to desc. n = 0 (the
    rules then read unreadable) under min_group complete neurons or over missing_max of them incomplete. win=False
    (Amendment 1's C1a, C1b, C1_fp): over judged neurons, ties only."""
    m = members(T, group, role)
    ok = [t for t in m if t["complete" if win else "judged"]]
    if not m or len(ok) < S["min_group"] or 1 - len(ok) / len(m) > S["missing_max"]:
        return float("nan"), 0
    return CJ.rate([(not win or t["outcome"] == "win") and (desc is None or desc in t["ties"]) for t in ok])


def status(v: str) -> str:
    return {"above": "pass", "below_or_level": "fail", "no_call": "no call"}.get(v, "unreadable")


def p1_p2(P: dict[str, Any], T: Table, floor: dict[str, tuple[float, int]]) -> dict[str, Any]:
    """The filed P1 and P2 rules on a table (predictions.json)."""
    S, G = P["settings"], P["groups"]
    rel = lambda g: side(T, g, "carrying", G[g]["description"], S)
    p1 = CJ.gap_rule(*rel("lowercase"), *floor["lower"], S["min_gap"])
    rows = {}
    for name, (a, b) in P2_ROWS.items():
        (pa, na), (pb, nb) = rel(a), rel(b)
        fa, fb = floor[G[a]["description"]], floor[G[b]["description"]]
        rows[name] = CJ.row_kind({"p": pa, "n": na, "floor": fa[0], "n_floor": fa[1]},
                                 {"p": pb, "n": nb, "floor": fb[0], "n_floor": fb[1]}, S["min_gap"])
    readable = [n for n, r in rows.items() if r["verdict"] != "unreadable"]
    up = [n for n in readable if rows[n]["verdict"] == "above"]
    return {"P1": {**p1, "status": status(p1["verdict"])},
            "P2": {"status": "unreadable" if len(readable) < 2 else "pass" if len(up) >= 2 else "fail", "rows": rows,
                   "readable": readable, "reuse_higher": up}}


def w0_controls(P: dict[str, Any], T: Table, floor: dict[str, tuple[float, int]]) -> dict[str, Any]:
    S = P["settings"]
    return {g: CJ.gap_rule(*side(T, g, "control", c["description"], S), *floor[c["description"]], S["min_gap"])
            for g, c in P["controls_w0"].items()}


def score(P: dict[str, Any]) -> dict[str, Any]:
    S, key = P["settings"], load(LARGE / "key.json")
    where = {i: b for b, ids in key["batches"].items() for i in ids}
    T = neuron_table(P, key, outputs("writer", "jobs"), outputs("judge", "ties"), load(LARGE / "judge_letters.json"), where)
    floor = {d: side(T, None, "random", d, S) for d in P["descriptions"]}
    rand_ok = [t for t in members(T, None, "random") if t["complete"]]
    fp, n_fp = CJ.rate([bool(t["ties"]) for t in rand_ok])
    self_test = load(STAGE1 / "max_scores.json")["self_test"]
    out: dict[str, Any] = {
        "C1_judge_fp": {"share": fp, "n": n_fp, "ok": n_fp > 0 and fp <= S["judge_fp_max"]},
        "C2_self_test": {**self_test, "ok": self_test["win_share"] >= S["self_test_min_win"]},
        "floors": floor, "complete": sum(t["complete"] for t in T.values()), "n_neurons": len(T),
    } | p1_p2(P, T, floor)
    p3 = CJ.gap_rule(*side(T, "rank1", "carrying", None, S), *side(T, None, "random", None, S), S["min_gap"])
    out["P3"] = {**p3, "status": {"above": "coherent", "below_or_level": "not coherent"}.get(p3["verdict"], p3["verdict"].replace("_", " "))}
    out["R_w0_controls"] = w0_controls(P, T, floor)
    out["R_explained_base"] = dict(zip(("share", "n"), side(T, None, "random", None, S), strict=True))
    out["read"] = {"P1": out["C1_judge_fp"]["ok"] and out["C2_self_test"]["ok"],
                   "P2_P3": out["C1_judge_fp"]["ok"] and out["C2_self_test"]["ok"] and out["P1"]["status"] == "pass"}
    return out


# ----------------------------------------------------------------------------- Amendment 1


def amend1_inputs(P: dict[str, Any], A: dict[str, Any]) -> None:
    S, key = A["settings"], load(LARGE / "key_amend1.json")
    jobs = outputs("writer", "jobs", "[bc]*.json")
    assert set(key["neurons"]) <= set(jobs), f"{len(set(key['neurons']) - set(jobs))} neurons have no job: writer c00 first"
    windows = load(RESULTS / "task_windows_amend1.json")["windows"]
    assert set(windows) == set(P["descriptions"])
    ids = sorted(key["neurons"])
    random.Random(S["seed"] + 5).shuffle(ids)
    rng, out = random.Random(S["seed"] + 6), {}
    (LARGE / "judge_inputs_amend1").mkdir(parents=True, exist_ok=True)
    for j in range(0, len(ids), S["batch_size"]):
        b, names = f"j{j // S['batch_size']:02d}", sorted(P["descriptions"])
        rng.shuffle(names)
        out[b] = {"ids": ids[j : j + S["batch_size"]], "letters": {chr(65 + k): n for k, n in enumerate(names)}}
        tasks = {L: {"task": P["descriptions"][n], "windows": windows[n]} for L, n in out[b]["letters"].items()}
        rec = {"batch": b, "tasks": tasks, "jobs": {i: jobs[i] for i in out[b]["ids"]}}
        (LARGE / "judge_inputs_amend1" / f"{b}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    (LARGE / "judge_amend1.json").write_text(json.dumps(out, indent=1))
    print(f"judge inputs for {len(out)} batches ({len(ids)} neurons) in {LARGE / 'judge_inputs_amend1'}")


def amend1_table(P: dict[str, Any]) -> Table:
    key, batches = load(LARGE / "key_amend1.json"), load(LARGE / "judge_amend1.json")
    letters = {b: v["letters"] for b, v in batches.items()}
    where = {i: b for b, v in batches.items() for i in v["ids"]}
    return neuron_table(P, key, outputs("writer", "jobs", "[bc]*.json"), outputs("judge_amend1", "ties", "j*.json"), letters, where)


def amend1(P: dict[str, Any], A: dict[str, Any], T: Table, self_test_ok: bool) -> dict[str, Any]:
    """Amendment 1's rules (predictions_amend1.json) on a table."""
    S, C = A["settings"], A["settings"]["controls"]
    own = {t: side(T, t, "control_amend1", c["description"], S, win=False) for t, c in C.items()}
    floor_ties = {d: side(T, None, "random", d, S, win=False) for d in P["descriptions"]}
    c1a = {t: CJ.gap_rule(*own[t], *floor_ties[c["description"]], S["min_gap"]) for t, c in C.items()}
    c1b = {}
    for t in C:
        other = [side(T, t, "control_amend1", C[u]["description"], S, win=False) for u in C if u != t]
        c1b[t] = CJ.gap_rule(*own[t], *max(other, key=lambda x: x[0] if x[1] else -1.0), S["min_gap"])
    fp, n_fp = CJ.rate([bool(t["ties"]) for t in members(T, None, "random") if t["judged"]])
    floor ={d: side(T, None, "random", d, S) for d in P["descriptions"]}
    out: dict[str, Any] = {
        "C1a": {"status": CJ.every_task([v["verdict"] for v in c1a.values()]), "tasks": c1a},
        "C1b": {"status": CJ.every_task([v["verdict"] for v in c1b.values()]), "tasks": c1b},
        "C1_fp": {"share": fp, "n": n_fp, "ok": n_fp > 0 and fp <= S["judge_fp_max"]},
        "floors_ties": floor_ties, "floors": floor, "judged": sum(t["judged"] for t in T.values()), "n_neurons": len(T),
    } | p1_p2(P, T, floor)
    out["R_w0_controls"] = w0_controls(P, T, floor)
    ok = out["C1_fp"]["ok"]
    out["read"] = {"P1": self_test_ok and ok and c1a["lowercase"]["verdict"] == "above",
                   "P2": self_test_ok and ok and out["C1a"]["status"] == "pass" and out["C1b"]["status"] == "pass"
                   and out["P1"]["status"] == "pass"}
    out["route_open_on_67M"] = out["C1a"]["status"] == "pass"
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
    elif cmd == "amend1-inputs":
        amend1_inputs(P, load(RESULTS / "predictions_amend1.json"))
    elif cmd == "amend1-score":
        st = load(STAGE1 / "max_scores.json")["self_test"]
        out = amend1(P, load(RESULTS / "predictions_amend1.json"), amend1_table(P), st["win_share"] >= P["settings"]["self_test_min_win"])
        (RESULTS / "score_amend1.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=sorted))
        for k in ("C1a", "C1b", "C1_fp", "P1", "P2", "R_w0_controls", "read", "route_open_on_67M"):
            print(k, json.dumps(out[k], sort_keys=True, default=sorted)[:900])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

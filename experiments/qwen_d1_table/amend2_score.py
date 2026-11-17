"""E93 Amendment 2's scoring (CPU, no model): E93's and Amendment 1's rules on the records amend2.py writes on the new
decomposition, with the same read on old beside, and E86's and E75's rows. Run from the repo root:

  python experiments/qwen_d1_table/amend2_score.py          # results/amend2/table.json from what is written
  python experiments/qwen_d1_table/amend2_score.py --edges  # the rules, the tables and the carried record, dry-run on
                                                            # stored E93 results, before filing (results/amend2/edges.txt)
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from commute.jobs import EXPERIMENTS, hard_exit, load_experiment

run = load_experiment("qwen_d1_table", "run")
am1 = load_experiment("qwen_d1_table", "amend1")  # skeleton, score_recs, predictions
rules = run.rules
e86 = load_experiment("qwen_d1_fractions", "rules")  # E86's `table` and, as `.wrong`, E75 Amendment 1's `cell`, `overall`

NEW = "p-9707d6ff"
RESULTS = rules.RESULTS
OUT, PRED, TABLE = RESULTS / "amend2", RESULTS / "predictions_amend2.json", RESULTS / "amend2" / "table.json"
run.RESULTS, run.STOPPED = OUT, OUT / "stopped.json"  # run.log, check and memory write here; amend1 had pointed them at amend1/
E86_DIR = EXPERIMENTS / "qwen_d1_fractions" / "results"
E75_DIR, E72_DIR = EXPERIMENTS / "qwen_rl_reads" / "results", EXPERIMENTS / "qwen_fixed_masks" / "results"
E86_ROWS = {"u1_10": "u1", "u2": "u2", "tofu_E81": "tofu"}  # extras record -> E86's update (u1 is E40's step 10)
E75_ROWS = {"u1_10": "u1", "u2": "u2", "R_s0": "R100", "R10": "R10", "u3": "u3"}  # extras record -> E75's update
CARRIED_CLEAN = ("W0", "FT", "random", "signs")


def carried(name: str) -> dict[str, Any]:
    """E93's record of `name`, keeping only what no decomposition entered."""
    e93 = rules.load(name)
    assert e93 is not None, f"E93's {name} is not written"
    rec = {k: e93[k] for k in ("name", "grid_arm", "rows", "arm", "file", "dtype", "norm", "sha256", "n")}
    fr = ("W0", "FT", *(f"t{t}" for t in rules.TS), "W0_end")
    return rec | {"fractions": {k: e93["fractions"][k] for k in fr}, "clean": {k: e93["clean"][k] for k in CARRIED_CLEAN},
                  "decomp": NEW, "seconds": {}}


def load_rec(name: str, kind: str = "state") -> dict[str, Any] | None:
    p = OUT / name / f"{kind}.json"
    rec = json.loads(p.read_text()) if p.is_file() else None
    return rec if rec is not None and rec.get("done") else None


def load_extra(name: str) -> dict[str, Any] | None:
    p = OUT / "extras" / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def reads(s: dict[str, Any]) -> dict[str, Any]:
    r = s["rules"]
    return {"clean_over_ruler": r["1"].get("over_ruler"), "clean": r["1"]["result"], "share": r["2"].get("value"), "null": r["2"].get("null"),
            "size": r["3"].get("value"), "masked_K_minus_do_nothing": r["5"].get("diff"), "kept_clean": r["4"].get("value"),
            "kept_masked": r["6"]["result"], "verdict": s["verdict"]}


def beside(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    return {n: {"new": reads(s), "old": reads(old["states"][n]) if n in old["states"] else None} for n, s in new["states"].items()}


def e75_prior(u: str, read: str) -> dict[str, Any]:
    """A committed clean read, E75's file or E72's (as qwen_rl_reads/amend1.py `prior`)."""
    for root in (E75_DIR, E72_DIR):
        if (p := root / u / f"{read}.json").is_file():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"no {read}.json for {u}")


def e86_row(u: str, x: dict[str, Any] | None) -> dict[str, Any]:
    """E86's `table` on the new in-span and remainder, with its stored clean W0, random and signs; its stored row beside."""
    sc = json.loads((E86_DIR / f"score_{u}.json").read_text())
    stored = lambda r: json.loads((E86_DIR / u / f"clean_{r}.json").read_text())
    old = {"over_ruler": sc["table"]["over_ruler"], "verdict": sc["table"]["verdict"], "left_in_delta_ls": sc["left_in_delta"]["ls"]}
    if x is None:
        return {"old": old, "new": "unread"}
    t = e86.table({"clean_W0": stored("W0"), "clean_inspan": x["clean"]["inspan"], "clean_remainder": x["clean"]["remainder"],
                   "clean_random": stored("random"), "clean_signs": stored("signs")})
    return {"old": old, "new": {"over_ruler": t["over_ruler"], "inspan": t["inspan"], "verdict": t["verdict"],
                                "left_in_delta_ls": x["split"]["left"] / x["split"]["size"], "wrong": t["wrong"]}}


def e75_row(u: str, x: dict[str, Any] | None) -> dict[str, Any]:
    """E75 Amendment 1's cells: its three stored wrong changes against the remainder read on new."""
    stored = lambda s: json.loads((E75_DIR / "amend1" / u / f"{s}.json").read_text())
    if x is None:
        return {"new": "unread"}
    ruler = e86.k_of(e75_prior(u, "clean_W0"))
    return {"new": {s: e86.wrong.cell(stored(s), x["clean"]["remainder"], ruler) for s in e86.wrong.STATES}}


def wrong_table(extras: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """E75's table on new: its `overall`, with the stored labels (old) beside."""
    old = json.loads((E75_DIR / "amend1_score.json").read_text())
    row = {x: e75_row(u, extras.get(x)) for x, u in E75_ROWS.items()}
    read = {E75_ROWS[x]: r["new"] for x, r in row.items() if r["new"] != "unread"}
    return {"rows": row, "overall_new": e86.wrong.overall(read) if len(read) == len(E75_ROWS) else {"overall": "unread"},
            "overall_old": {"overall": old["overall"], "list": old["list"]}, "table_old": old["table"]}


def score_all(ls: dict[str, dict[str, Any]], adj: dict[str, dict[str, Any]], P: dict[str, Any]) -> dict[str, Any]:
    pick = lambda m: {"thresholds": P["thresholds"], "predictions": P["predictions"][m]}
    return {"ls": am1.score_recs(ls, pick("ls")), "adjoint": am1.score_recs(adj, pick("adjoint"))}


def score(P: dict[str, Any]) -> dict[str, Any]:
    ls = {n: r for n in rules.STATES if (r := load_rec(n)) is not None}
    adj = {n: r for n in rules.STATES if (r := load_rec(n, "adjoint")) is not None}
    t = score_all(ls, adj, P)
    old = {"ls": json.loads((RESULTS / "table.json").read_text()), "adjoint": json.loads((RESULTS / "amend1" / "table.json").read_text())}
    extras = {x: load_extra(x) for x in sorted({*E86_ROWS, *E75_ROWS})}
    out = t | {"decomp": NEW, "beside": {m: beside(t[m], old[m]) for m in t},
               "old_verdict": {m: {"arms": old[m]["arms"], "table": old[m]["table"]} for m in old},
               "e86_rows": {x: e86_row(u, extras[x]) for x, u in E86_ROWS.items()}, "e75_rows": wrong_table(extras)}
    run.write(TABLE, out)
    return out


# ----------------------------------------------------------------------------- CPU dry run, before filing


def edges(P: dict[str, Any]) -> bool:
    """Pure logic and paths: E93's edge cases, the stored E93 and Amendment 1 records through this job's scoring (the
    verdicts must equal the stored tables), the carried record, and E86's and E75's rows from stored stand-ins."""
    good = all(r["as_named"] for r in rules.edges())
    e93 = {n: r for n in rules.STATES if (r := rules.load(n)) is not None}
    a1 = {n: r for n in rules.STATES if (r := am1.load_rec(n)) is not None}
    t = score_all(e93, a1, P)
    old = json.loads((RESULTS / "table.json").read_text()), json.loads((RESULTS / "amend1" / "table.json").read_text())
    cases: list[tuple[str, Any, Any]] = [
        ("Stored E93 through this job's scorer: verdicts", {n: s["verdict"] for n, s in t["ls"]["states"].items()}, {n: s["verdict"] for n, s in old[0]["states"].items()}),
        ("Stored Amendment 1 through this job's scorer: verdicts", {n: s["verdict"] for n, s in t["adjoint"]["states"].items()}, {n: s["verdict"] for n, s in old[1]["states"].items()}),
        ("beside() on the same table: new reads = old reads", all(r["new"] == r["old"] for r in beside(t["ls"], old[0]).values()), True)]
    three = {n: e93[n] for n in list(e93)[:3]}
    part = am1.predictions(am1.score_recs(three, {"thresholds": P["thresholds"], "predictions": P["predictions"]["ls"]})["states"], {"predictions": P["predictions"]["ls"]})
    cases.append(("Filed leans, three states written", set(part.values()), {"unread"}))
    rec = carried("u1")
    cases.append(("carried: no decomposition-read key", sorted(k for k in rec if k in ("split", "own", "base", "ceiling", "random")), []))
    cases.append(("carried: clean", sorted(rec["clean"]), sorted(CARRIED_CLEAN)))
    cases.append(("skeleton of a full record keeps base and ceiling", {"base", "ceiling", "random"} <= set(am1.skeleton(e93["a2_s42"])), True))
    for x, u in E86_ROWS.items():  # stand-in: E86's own stored reads as the "new" ones must read E86's stored verdict
        read = {"split": {"left": 0.1, "size": 1.0}, "clean": {k: json.loads((E86_DIR / u / f"clean_{k}.json").read_text()) for k in ("inspan", "remainder")}}
        row = e86_row(u, read)
        cases.append((f"E86 row {x}: stored reads as new give the stored verdict", row["new"]["verdict"], row["old"]["verdict"]))
    for x, u in E75_ROWS.items():
        row = e75_row(u, {"clean": {"remainder": e75_prior(u, "clean_remainder")}})
        cases.append((f"E75 row {x}: stored remainder as new gives every wrong change 'fails'", {c["label"] for c in row["new"].values()}, {"fails"}))
    empty = {"T": 256, "sums": {"K": []}}
    cases += [("E86 row on an empty read: no call, no crash", e86.table({"clean_W0": empty, "clean_inspan": empty})["verdict"], e86.NO_CALL),
              ("E75 cell on an empty read: no call", e86.wrong.cell(empty, empty, float("nan"))["label"], e86.NO_CALL),
              ("Extras not yet written read unread", e75_row("R10", None)["new"], "unread")]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "edges.txt").open("w") as f:
        for c, g, w in cases:
            line = f"{'ok ' if g == w else 'BAD'}  {c}: {g if g != w else 'as named'}{'' if g == w else f' (named {w})'}"
            good &= g == w
            print(line, flush=True)
            f.write(line + "\n")
    return good


def main() -> None:
    ap = argparse.ArgumentParser(description="E93 Amendment 2's scoring")
    ap.add_argument("--edges", action="store_true")
    P = json.loads(PRED.read_text())
    if ap.parse_args().edges:
        hard_exit(0 if edges(P) else 1)
    t = score(P)
    print(json.dumps({m: {"arms": t[m]["arms"], "table": t[m]["table"], "predictions": t[m]["predictions"]} for m in ("ls", "adjoint")}, indent=1), flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()

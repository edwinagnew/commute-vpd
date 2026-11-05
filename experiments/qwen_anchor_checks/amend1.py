"""E88 Amendment 1: do invented names in real-fact answers repeat at a second seed, and is the harm confined to author questions?
Plan: reports/plans/2026-11-05_plan_e88-amend1-names.md. Rules: results/predictions_amend1.json (the job refuses to start without it).
One L40, about 15 min. Order, cheapest first: L1 on saved answers (no model); L0, W0's real-fact answers regenerated and their sha256
against stage R's; the second seed's real-fact answers; L2, L3, L4 on both seeds. Results in results/amend1.json as stages finish.
Exit 0 done; 1 a stop (results/stopped_amend1.json) or a refusal.

Usage (repo root, via the sbatch): .venv/bin/python experiments/qwen_anchor_checks/amend1.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import load as load_file
from commute.qwen_sft import plus, set_weights

R: Any = load_experiment("qwen_anchor_checks", "run")  # Lab, stop, answers' folders, holds_fact, lost_gained, mean_words, real_answers
RESULTS = R.RESULTS
PRED, OUT, NAMES = RESULTS / "predictions_amend1.json", RESULTS / "amend1.json", RESULTS / "invented_names.json"
AUTHOR = re.compile(r"author|wrote|written|writer|novel", re.IGNORECASE)
SEEDS = {"first": R.ANSWERS / "S", "second": R.ANSWERS / "S_seed"}
W0_NEW = R.ANSWERS / "W0_a1"


# ----------------------------------------------------------------------------- the filed rules, no model


def author_question(question: str) -> bool:
    return AUTHOR.search(question) is not None


def names_invented(row: dict[str, Any], names: list[str]) -> list[str]:
    """Listed names the response holds that are in neither the row's question nor its reference."""
    return [n for n in names if n in row["response"] and n not in row["question"] and n not in row["reference"]]


def naming_rows(rows: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    """Rows counted once, whatever number of names they hold."""
    return [{"question": r["question"], "names": hit, "response": r["response"]} for r in rows if (hit := names_invented(r, names))]


def l2_verdict(count: int, repeats: int, alone: int) -> str:
    return "repeats" if count >= repeats else "first seed's alone" if count <= alone else "partly"


def author_share(naming: list[dict[str, Any]]) -> float | None:
    """Share of naming rows that are author questions; None when there are none."""
    return sum(author_question(r["question"]) for r in naming) / len(naming) if naming else None


def l3_verdict(shares: list[float | None], bar: float) -> str:
    if any(s is None for s in shares):
        return "no reading"
    return "names go to author questions" if all(s >= bar for s in shares if s is not None) else "names go wider"


def fall_points(w0_hits: int, hits: int, n: int) -> Fraction:
    """W0 less the update, in points of the subset's `n` rows; exact, so a tie at a bar is a tie."""
    return Fraction(w0_hits - hits, n) * 100


def l4_verdict(falls: list[dict[str, Fraction]], other_max: int, author_min: int) -> str:
    """`falls` is one {author, other} per seed."""
    if any(f["other"] > other_max for f in falls):
        return "harm is wider than author questions"
    if all(f["author"] >= author_min for f in falls):
        return "harm is on author questions"
    return "no call"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- reading one state


def contains_by_type(rows: list[dict[str, Any]], hits: list[bool]) -> dict[str, dict[str, Any]]:
    mask = [author_question(r["question"]) for r in rows]
    out = {}
    for key, want in (("author", True), ("other", False)):
        sel = [h for h, m in zip(hits, mask, strict=True) if m == want]
        out[key] = {"hits": sum(sel), "n": len(sel), "share": sum(sel) / len(sel)}
    return out


def state_reading(lab: Any, path: Path, names: list[str]) -> dict[str, Any]:
    rows, hits = R.lines(path), R.holds_fact(path)
    naming = naming_rows(rows, names)
    out = {"n": len(rows), "naming": len(naming), "author_share": author_share(naming), "naming_rows": naming,
           "words": R.mean_words(path), "contains": contains_by_type(rows, hits), "contains_all": sum(hits)}
    lab.log(f"[state {path.parent.name}] {len(rows)} rows, {len(naming)} naming, author share {out['author_share']}, {out['words']:.2f} words, "
            f"contains {json.dumps(out['contains'])}")
    return out


# ----------------------------------------------------------------------------- the job


def main() -> None:
    stopped = RESULTS / "stopped_amend1.json"
    if stopped.is_file():
        sys.exit(f"{stopped} exists: read it, then remove it")
    if R.is_done(OUT):
        print("nothing to do: amend1 is done", flush=True)
        hard_exit(0)
    lab = R.Lab("amend1", "E88", PRED)  # refuses to start without the predictions file; the tokenizer only, no model yet
    P, names = lab.pred, json.loads(NAMES.read_text())
    w0_saved = R.ANSWERS / "W0"
    assert P["rows"]["n"] == 217 and lab.pred["L1"]["s"] == 18
    out: dict[str, Any] = {"card": lab.card}

    # L1, no model
    saved = {"W0": naming_rows(R.lines(w0_saved / "real.jsonl"), names), "first": naming_rows(R.lines(SEEDS["first"] / "real.jsonl"), names)}
    out["L1"] = {"w0": len(saved["W0"]), "first": len(saved["first"]), "filed": [P["L1"]["w0"], P["L1"]["s"]], "list": saved["first"]}
    lab.log(f"[L1] naming rows, saved answers: W0 {len(saved['W0'])} (filed {P['L1']['w0']}), first seed {len(saved['first'])} (filed {P['L1']['s']})")
    for r in saved["first"]:
        lab.log(f"[L1] {json.dumps(r)}")
    R.write(OUT, out)
    if (len(saved["W0"]), len(saved["first"])) != (P["L1"]["w0"], P["L1"]["s"]):
        R.stop(lab, "L1: the name rule does not reproduce the saved counts", out["L1"])

    # L0 and the second seed, one card
    lab.fp32()
    R.real_answers(lab, W0_NEW)
    a, b = sha256(w0_saved / "real.jsonl"), sha256(W0_NEW / "real.jsonl")
    out["L0"] = {"saved": a, "regenerated": b, "same": a == b}
    lab.log(f"[L0 {lab.card}] W0 real.jsonl sha256 saved {a}, regenerated {b}: {'ok' if a == b else 'MISSED'}")
    R.write(OUT, out)
    if a != b:
        R.stop(lab, "L0: W0's regenerated answers differ from stage R's", out["L0"])
    with set_weights(lab.model, plus(lab.W0, load_file(R.DW["S"] / f"step_{R.E81.STEPS}.safetensors", R.DEV))):
        R.real_answers(lab, SEEDS["second"])
    lab.log(f"[second seed] answers written to {SEEDS['second'] / 'real.jsonl'}")

    # L2, L3, L4
    states = {"W0": state_reading(lab, w0_saved / "real.jsonl", names)} | {k: state_reading(lab, p / "real.jsonl", names) for k, p in SEEDS.items()}
    assert states["W0"]["n"] == 217 and states["W0"]["contains"]["author"]["n"] == 95 and states["W0"]["contains"]["other"]["n"] == 122
    out["states"] = states
    w0_hits = R.holds_fact(w0_saved / "real.jsonl")
    for k, p in SEEDS.items():
        lost, gained = R.lost_gained(w0_hits, R.holds_fact(p / "real.jsonl"))
        states[k] |= {"lost": lost, "gained": gained}
        lab.log(f"[{k} seed] rows lost {lost}, gained {gained}")
    falls = [{t: fall_points(states["W0"]["contains"][t]["hits"], states[k]["contains"][t]["hits"], states["W0"]["contains"][t]["n"])
              for t in ("author", "other")} for k in SEEDS]
    out["L2"] = {"count": states["second"]["naming"], "verdict": l2_verdict(states["second"]["naming"], P["L2"]["repeats_at_least"], P["L2"]["alone_at_most"])}
    shares = [states[k]["author_share"] for k in SEEDS]
    out["L3"] = {"shares": dict(zip(SEEDS, shares, strict=True)), "verdict": l3_verdict(shares, P["L3"]["share_at_least"])}
    out["L4"] = {"falls_points": {k: {t: float(v) for t, v in f.items()} for k, f in zip(SEEDS, falls, strict=True)},
                 "verdict": l4_verdict(falls, P["L4"]["other_at_most_points"], P["L4"]["author_at_least_points"])}
    for rule in ("L2", "L3", "L4"):
        lab.log(f"[{rule}] {json.dumps(out[rule])}")
    R.write(OUT, out | {"done": True, "peaks": lab.peaks()})
    lab.log(f"[amend1] done; {lab.peaks()}")
    hard_exit(0)


if __name__ == "__main__":
    main()

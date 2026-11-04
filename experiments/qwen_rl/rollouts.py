"""E69: what the rollouts' text says, per step and arm: opening word, answer line present, strict and lenient right. Arm F's
rollouts are W0's samples on the same prompts. No model; reads the tars and the Qwen tokenizer, writes
results/lr1e-6w10/rollouts_summary.json.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 4G -t 00:30:00 python experiments/qwen_rl/rollouts.py TOKENIZER_JSON
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
import tarfile
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from commute import qwen_sets
from commute.gsm8k import gold_number, grade, strict_pred

HERE = Path(__file__).resolve().parent
RUN = HERE / "results" / "lr1e-6w10"
TARS = HERE / "results" / "large" / "lr1e-6w10" / "rollouts"
SPANS = ((1, 10), (11, 20), (21, 50), (51, 75), (76, 95), (96, 100))


def records(arm: str, tok: Tokenizer, train: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    with tarfile.open(TARS / f"{arm}.tar") as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".jsonl"):
                continue
            f = tar.extractfile(m)
            assert f is not None
            rows = []
            for line in f.read().decode().splitlines():
                r = json.loads(line)
                text = tok.decode(r["ids"], skip_special_tokens=True)
                g = grade(text, gold_number(train[r["pid"]]["gold"]))
                rows.append({"pid": r["pid"], "first": tok.decode(r["ids"][:1]), "truncated": r["truncated"], "reward": r["reward"],
                             "strict": bool(g["strict"]), "lenient": bool(g["lenient"]), "line": strict_pred(text) is not None, "n": len(r["ids"])})
            out[int(m.name.split("_")[1].split(".")[0])] = rows
    return out


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    inside = [r for r in rows if not r["truncated"]]
    firsts = collections.Counter(r["first"] for r in rows)
    return {"reward": st.mean(r["reward"] for r in rows), "inside": len(inside) / len(rows), "length": st.mean(r["n"] for r in rows),
            "line_when_inside": st.mean(r["line"] for r in inside), "lenient_when_inside": st.mean(r["lenient"] for r in inside),
            "strict_when_inside": st.mean(r["strict"] for r in inside),
            "firsts": {k: v / len(rows) for k, v in firsts.most_common(6)}}


def by_first(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for k, n in collections.Counter(r["first"] for r in rows).most_common(4):
        g = [r for r in rows if r["first"] == k]
        out[k] = {"share": n / len(rows), "reward": st.mean(r["reward"] for r in g), "lenient": st.mean(r["lenient"] for r in g), "length": st.mean(r["n"] for r in g)}
    return out


def within_problem(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    """Arm F's samples are W0's. Over problems with samples opening both ways, the mean difference a minus b, so a problem's
    difficulty cancels. SE over problems. `kept` is lenient and inside the window, the reward with its format ignored."""
    by: dict[int, dict[str, list[dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["pid"]][r["first"]].append(r)
    out: dict[str, Any] = {"problems": sum(a in v and b in v for v in by.values()),
                           "samples": {x: sum(len(v[x]) for v in by.values() if a in v and b in v) for x in (a, b)}}
    fields = {"reward": lambda r: r["reward"], "kept": lambda r: r["lenient"] and not r["truncated"], "length": lambda r: r["n"]}
    for k, f in fields.items():
        d = [st.mean(float(f(r)) for r in v[a]) - st.mean(float(f(r)) for r in v[b]) for v in by.values() if a in v and b in v]
        out[k] = {"mean": st.mean(d), "se": st.stdev(d) / len(d) ** 0.5}
    return out


def main() -> None:
    tok = Tokenizer.from_file(sys.argv[1])
    train = qwen_sets.load()["train"]
    out: dict[str, Any] = {}
    for arm in ("R", "N", "F"):
        rec = records(arm, tok, train)
        out[arm] = {"steps": {t: summary(rec[t]) for t in sorted(rec)}, "spans": {}}
        if arm == "F":
            every = [r for t in rec for r in rec[t]]
            out["w0_within_problem"] = {f"{x} minus {y}": within_problem(every, x, y) for x, y in (("We", "Let"), ("We", "Step"), ("Step", "Let"))}
            print(json.dumps(out["w0_within_problem"]), flush=True)
        for lo, hi in SPANS:
            rows = [r for t in range(lo, hi + 1) for r in rec[t]]
            out[arm]["spans"][f"{lo}-{hi}"] = summary(rows) | {"by_first": by_first(rows)}
    (RUN / "rollouts_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for arm in ("R", "N", "F"):
        for span, s in out[arm]["spans"].items():
            print(arm, span, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ("firsts", "by_first")}), flush=True)
            print("   ", json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in s["by_first"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

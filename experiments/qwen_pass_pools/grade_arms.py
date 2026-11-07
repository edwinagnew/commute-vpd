"""E120 stage 3 grading, CPU, $0 (2026-11-07): the step-100 answers of the four SFT arms graded by `grade.py`
(Amendment 2's MATH grader), and W0's pass rates on the same held-out problems from stage 1's records.

Per arm (known_s0, known_s1, new_s0, new_s1), from <large>/samples/<arm>/:
  <pool>_held_samples.json  8 samples at `qwen_eval.SAMPLE`, cap 1,024, per held-out problem of each pool: pass@1 =
                            mean over problems of (correct of 8)/8, P3's rule (predictions.json → P3.rule).
  <pool>_train_greedy.json  one greedy answer per training problem of the arm's own pool: share correct (P3's
                            "trained-part greedy").
W0: per held-out problem, (k16 + k48)/(n16 + n48) from <dl>/stage1_problems_amend2.json.gz: the unchanged model's 64
samples at the same `qwen_eval.SAMPLE`, but by vLLM and at caps 512 (GSM8K) / 2,048 (MATH), not 1,024. Context only:
P3's rule reads the arms alone. Writes results/grades.json.

  PYTHONPATH=~/commute-data/scratch/qwen-rl-lead/e120-a2-writer/mvlib fleet/cpu-run -c 4 --mem 4G \
      python experiments/qwen_pass_pools/grade_arms.py --large <clone>/experiments/qwen_pass_pools/results/large \
      --dl <clone>/experiments/qwen_pass_pools/results/pod_dl
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from grade import answer_of, grade_one, pmap

from commute.jobs import hard_exit

RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("known_s0", "known_s1", "new_s0", "new_s1")


def grade_file(path: Path) -> dict[str, Any]:
    """One samples file -> pass@1 (mean per-problem share), truncation share, per-problem k of n."""
    rows = json.loads(path.read_text())
    answers = [[None if cut else answer_of(r["source"], t) for t, cut in zip(r["responses"], r["truncated"], strict=True)]
               for r in rows]
    res = pmap(grade_one, [(r["source"], str(r["answer_gold"]), a, r["user"]) for r, a in zip(rows, answers, strict=True)],
               procs=4)
    per = {r["id"]: [sum(c), len(c)] for r, c in zip(rows, res, strict=True)}
    cuts = [x for r in rows for x in r["truncated"]]
    return {"problems": len(rows), "samples_per_problem": sorted({n for _, n in per.values()}),
            "pass1": sum(k / n for k, n in per.values()) / len(per), "truncated": sum(cuts) / len(cuts), "k_of_n": per}


def w0_rates(dl: Path, pools: dict[str, Any]) -> dict[str, Any]:
    recs = {r["id"]: r for r in json.loads(gzip.decompress((dl / "stage1_problems_amend2.json.gz").read_bytes()))}
    out = {}
    for p in ("known", "new"):
        per = {it["id"]: [recs[it["id"]]["k16"] + recs[it["id"]]["k48"], recs[it["id"]]["n16"] + recs[it["id"]]["n48"]]
               for it in pools[p]["held"]}
        out[f"{p}_held"] = {"problems": len(per), "pass1": sum(k / n for k, n in per.values()) / len(per), "k_of_n": per}
    return out


def main(large: Path, dl: Path) -> None:
    pools = json.loads((RESULTS / "pools_amend2.json").read_text())
    out: dict[str, Any] = {"w0": w0_rates(dl, pools), "arms": {}}
    for arm in ARMS:
        pool = arm.split("_")[0]
        files = ["known_held_samples", "new_held_samples", f"{pool}_train_greedy"]
        out["arms"][arm] = {f: grade_file(large / "samples" / arm / f"{f}.json") for f in files}
        print(arm, {f: round(v["pass1"], 4) for f, v in out["arms"][arm].items()}, flush=True)
    print("w0", {k: round(v["pass1"], 4) for k, v in out["w0"].items()}, flush=True)
    (RESULTS / "grades.json").write_text(json.dumps(out, indent=1))
    print("wrote", RESULTS / "grades.json", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description="E120: grade the SFT arms' step-100 answers (grade.py)")
    a.add_argument("--large", type=Path, required=True)
    a.add_argument("--dl", type=Path, required=True)
    args = a.parse_args()
    main(args.large, args.dl)
    hard_exit(0)

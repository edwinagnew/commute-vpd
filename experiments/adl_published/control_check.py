"""E284 (cell P2.67): the positive control's pass rule (spec section 4.1 option A, rule (a) as amended by redteam-lead
2026-11-23), read from the toolkit's own saved causal-effect files. Standard library only; no model.

What the toolkit writes, per run (diffing-toolkit 617d9266):
  <storage>/diffing_results/qwen3_1_7B/<organism>[_<variant>]/activation_difference_lens   method.py:L425-437 (the variant
      suffix is added unless it is "default", L426-429)
  .../layer_<abs>/<diff source dir>/causal_effect/direction_base_to_ft/eval_<eval dir>/position_<p>/results.json
      causal_effect.py:L626-635, written at L1459-1481; eval dir = the dataset id's last part (util.py:L8-11): the
      organism's training set `synthetic-documents-<organism>`, and `fineweb-1m-sample`
  keys read: intervention.exclude_pos.incr_ce (L1135-1152), random_diff_mean.exclude_pos.incr_ce (L1393-1457),
      random_diff[k].exclude_pos.incr_ce (L1310-1391), and each subset's count and num_samples for the record

The rule (filed in results/predictions.json "control"; "mean" = mean over the three organisms):
  (a) training data, variant default: the mean incr_ce is above the mean random_diff_mean at >= 5 of positions 0-8,
      AND its mean over positions exceeds the random mean's by > 2 sd of the 64 random differences (each random k's
      incr_ce averaged over organisms and positions, then the sd over k)
  (b) FineWeb, default: the mean incr_ce is below 0 at >= 5 of positions 0-8
  (c) training data: mix1-2p0's position-mean of the mean incr_ce is between 0.25 and 0.75 times default's (undefined,
      so a miss, when default's is <= 0)
  (d) descriptive: mix1-2p0's FineWeb position-mean (the amendment expects it slightly positive)

    python3 experiments/adl_published/control_check.py ROOT     (ROOT holds the untarred <organism>_<variant>/ dirs)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRED, OUT = HERE / "results" / "predictions.json", HERE / "results" / "control.json"


def results(root: Path, org: str, var: str, eval_dir: str, layer: int, p: int) -> dict[str, Any]:
    pat = (f"{org}_{var}/storage/diffing_results/*/*/activation_difference_lens/layer_{layer}/fineweb-1m-sample/"
           f"causal_effect/direction_base_to_ft/eval_{eval_dir}/position_{p}/results.json")  # fmt: skip
    hits = sorted(root.glob(pat))
    if len(hits) != 1:
        raise SystemExit(f"{root}/{pat}: {len(hits)} files, not 1")
    return json.loads(hits[0].read_text())


def curves(root: Path, org: str, var: str, eval_dir: str, C: dict[str, Any]) -> dict[str, Any]:
    rs = [results(root, org, var, eval_dir, C["layer"], p) for p in C["positions"]]
    sub = C["subset"]
    return {"incr_ce": [r["intervention"][sub][C["metric"]] for r in rs],
            "random_diff_mean": [r["random_diff_mean"][sub][C["metric"]] for r in rs],
            "random_diff": [[r["random_diff"][k][sub][C["metric"]] for r in rs] for k in range(len(rs[0]["random_diff"]))],
            "count": [r["intervention"][sub]["count"] for r in rs], "num_samples": rs[0]["num_samples"],
            "ce_ft": [r["finetuned"][sub]["ce"] for r in rs], "ce_base": [r["base"][sub]["ce"] for r in rs]}  # fmt: skip


def mean_over(cs: list[dict[str, Any]], key: str) -> list[float]:
    return [statistics.fmean(c[key][j] for c in cs) for j in range(len(cs[0][key]))]


def check(root: Path, C: dict[str, Any]) -> dict[str, Any]:
    orgs, (dv, mv) = C["organisms"], C["variants"]
    train = {v: {o: curves(root, o, v, f"synthetic-documents-{o}", C) for o in orgs} for v in (dv, mv)}
    fw = {v: {o: curves(root, o, v, "fineweb-1m-sample", C) for o in orgs} for v in (dv, mv)}
    td, tm = list(train[dv].values()), list(train[mv].values())
    m, r = mean_over(td, "incr_ce"), mean_over(td, "random_diff_mean")
    K = len(td[0]["random_diff"])
    per_k = [statistics.fmean(c["random_diff"][k][j] for c in td for j in range(len(m))) for k in range(K)]
    sd = statistics.stdev(per_k)
    above = sum(a > b for a, b in zip(m, r, strict=True))
    a = {"positions_above": above, "need": C["a_min_positions"], "mean_incr_ce": statistics.fmean(m),
         "mean_random_diff_mean": statistics.fmean(r), "sd_random": sd, "sd_mult": C["a_sd_mult"],
         "bar": statistics.fmean(r) + C["a_sd_mult"] * sd}  # fmt: skip
    a["pass"] = above >= C["a_min_positions"] and a["mean_incr_ce"] > a["bar"]
    f = mean_over(list(fw[dv].values()), "incr_ce")
    b = {"positions_below_0": sum(x < 0 for x in f), "need": C["b_min_positions"]}
    b["pass"] = b["positions_below_0"] >= C["b_min_positions"]
    den, num = statistics.fmean(m), statistics.fmean(mean_over(tm, "incr_ce"))
    ratio = num / den if den > 0 else None
    c = {"default_mean": den, "mix_mean": num, "ratio": ratio, "band": C["c_band"]}
    c["pass"] = ratio is not None and C["c_band"][0] <= ratio <= C["c_band"][1]
    d = {"mix_fineweb_mean": statistics.fmean(mean_over(list(fw[mv].values()), "incr_ce")), "expected": "slightly positive (descriptive)"}
    per_org = {v: {o: {"train": {k: train[v][o][k] for k in ("incr_ce", "random_diff_mean", "count", "ce_ft", "ce_base", "num_samples")},
                       "fineweb": {k: fw[v][o][k] for k in ("incr_ce", "random_diff_mean", "count", "ce_ft", "ce_base", "num_samples")}}
                   for o in orgs} for v in (dv, mv)}  # fmt: skip
    return {"rule": C, "mean_curves": {"train_default": m, "train_random_diff_mean": r, "fineweb_default": f,
            "train_mix": mean_over(tm, "incr_ce")}, "per_organism": per_org, "a": a, "b": b, "c": c, "d": d,
            "passed": a["pass"] and b["pass"] and c["pass"]}  # fmt: skip


def main(root: Path) -> int:
    P = json.loads(PRED.read_text())
    out = {"experiment": P["experiment"], "root": str(root)} | check(root, P["control"])
    OUT.write_text(json.dumps(out, indent=1))
    for k in ("a", "b", "c", "d"):
        print(f"rule ({k}): {json.dumps(out[k])}")
    for v, orgs in out["per_organism"].items():
        for o, cv in orgs.items():
            print(f"{v} {o}: train incr_ce {[round(x, 4) for x in cv['train']['incr_ce']]}; "
                  f"random_diff_mean {[round(x, 4) for x in cv['train']['random_diff_mean']]}; "
                  f"fineweb incr_ce {[round(x, 4) for x in cv['fineweb']['incr_ce']]}")  # fmt: skip
    print(f"CONTROL PASSED {out['passed']} -> {OUT}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))

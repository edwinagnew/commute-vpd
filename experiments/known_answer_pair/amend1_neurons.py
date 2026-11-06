"""E103 Amendment 1's second job: the four matched pair rules of neurons.py on the new matched known-answer state against
the fictional matched states. Needs amend1.py's stage T done. Plan: reports/plans/2026-11-06_plan_e103-known-answer-pair.md,
"Amendment 1". Rules: results/predictions_amend1.json (`neurons`).

The row sets are E103's own: the real rows as neurons.py packs them (pair 0's bare answers; the same probe for every
state) and E92's 128 TOFU rows. The fictional states' and the Gaussian change's carrying sets are E103's on the real
rows (results/neurons/) and E102's on the TOFU rows, read back, not re-read; the engaged values are E103's and E102's.

  C2 first and alone: E103's matched state re-read on the real rows reproduces its stored gain and carrying set, or the
     job stops (1 min on an L40).
  Then the new matched state on both row sets, to results/neurons_amend1/, and the score to results/neurons_amend1.json.
Exit 0 done; 1 a stop (results/stopped_amend1.json).
"""

from __future__ import annotations

import json
from typing import Any

import torch
from safetensors.torch import load_file

from commute import qwen_comparators as qc
from commute import qwen_neurons as QN
from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import load, plus

A1: Any = load_experiment("known_answer_pair", "amend1")  # its Lab, stop, filed, paths
N: Any = load_experiment("known_answer_pair", "neurons")  # E103's rows, read, measures, pair_side
RUN, E102 = A1.RUN, N.E102
OUT_DIR, SCORE = A1.RESULTS / "neurons_amend1", A1.RESULTS / "neurons_amend1.json"
FICT = ("fict_s42_matched", "fict_s142_matched")


def control(lab: Any, R: qc.Rows) -> None:
    C, ref = lab.pred["neurons"]["C2"], json.loads((N.OUT_DIR / "real_matched__real.json").read_text())
    rec = N.read(lab, R, "e103_matched", plus(lab.W0, load(RUN.MATCHED, RUN.DEV)), OUT_DIR)
    rel, same = abs(rec["gain"]["gain"] / ref["gain"]["gain"] - 1), QN.overlap(rec["carry"], ref["carry"])
    detail = {"gain": rec["gain"]["gain"], "stored_gain": ref["gain"]["gain"], "rel": rel, "set_overlap": same}
    holds = rel <= C["gain_rel"] and same >= C["min_set_overlap"]
    lab.log(f"[C2n] {'ok' if holds else 'MISSED'} {json.dumps(detail)}")
    if not holds:
        A1.stop(lab, "C2n: E103's neuron read not reproduced", detail)


def sets() -> dict[str, dict[str, list[int]]]:
    """Per row set, the carrying sets the rules read: the new state's, the two fictional seeds' and the Gaussian's."""
    own = lambda rows: json.loads((OUT_DIR / f"a1_matched__{rows}.json").read_text())["carry"]
    e103 = lambda name: json.loads((N.OUT_DIR / f"{name}__real.json").read_text())["carry"]
    e102 = lambda name: json.loads((E102.RESULTS / "states" / f"{N.E102_NAME[name]}.json").read_text())["carry"]
    return {"real": {"a1_matched": own("real"), **{n: e103(n) for n in (*FICT, "gauss")}},
            "tofu": {"a1_matched": own("tofu"), **{n: e102(n) for n in (*FICT, "gauss")}}}


def score(lab: Any, by_rows: dict[str, dict[str, list[int]]], eng: dict[str, list[float]]) -> None:
    contrast = json.loads(E102.W0F.read_text())["contrast"]["sets"]["all"]
    per = {rows: {n: N.measures(c, eng[rows], contrast, torch.Generator().manual_seed(N.CFG.seed + 1 + i))
                  for i, (n, c) in enumerate(sorted(by.items()))} for rows, by in by_rows.items()}
    rules = {f"matched_{rows}_{k}": N.pair_side(v["a1_matched"][k], [v[x][k] for x in FICT], v["gauss"][k], v["a1_matched"][f"{k}_draws"])
             for rows, v in per.items() for k in ("engaged", "prompt_overlap")}
    e103 = json.loads((N.OUT_DIR / "real_matched__real.json").read_text())["carry"], json.loads((N.OUT_DIR / "real_matched__tofu.json").read_text())["carry"]
    overlap = {rows: {"with_fict_s42": QN.overlap(by["a1_matched"], by["fict_s42_matched"]),
                      "with_e103_matched": QN.overlap(by["a1_matched"], e103[i])} for i, (rows, by) in enumerate(by_rows.items())}
    SCORE.write_text(json.dumps({"per_state": per, "rules": rules, "set_overlap": overlap}, indent=1, default=float) + "\n")
    lab.log(f"[score] {json.dumps({k: v['verdict'] for k, v in rules.items()})}; set overlap {json.dumps(overlap)}")


def main() -> None:
    if A1.STOPPED.is_file():
        raise SystemExit(f"{A1.STOPPED} exists: read it, then remove it")
    A1.filed()
    if not A1.is_done("T"):
        raise SystemExit("amend1.py's stage T must finish first")
    lab = A1.Lab("N1", "E103 Amendment 1 neurons", A1.PRED)
    lab.fp32().eval()
    kept = {r["row"] for r in map(json.loads, E102.FORMS.read_text().splitlines()) if r["kept"]}
    rows = {"real": N.real_rows(lab), "tofu": qc.tofu_rows(lab.tok, lab.end_id, kept)}
    lab.log(f"[rows] {json.dumps({k: [list(R.tokens.shape), int(R.answer.sum())] for k, R in rows.items()})}")
    if not (OUT_DIR / "e103_matched__real.json").is_file():
        control(lab, rows["real"])
    missing = [x for x in rows if not (OUT_DIR / f"a1_matched__{x}.json").is_file()]
    if missing:
        W = plus(lab.W0, load(A1.MATCHED, RUN.DEV))
        for x in missing:
            N.read(lab, rows[x], "a1_matched", W, OUT_DIR)
        del W
    eng = {"real": load_file(str(N.ENGAGED_REAL))["real"].tolist(), "tofu": load_file(str(E102.LARGE / "engaged.safetensors"))["tofu"].tolist()}
    score(lab, sets(), eng)
    lab.log(f"[neurons] done, {lab.peaks()}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    main()

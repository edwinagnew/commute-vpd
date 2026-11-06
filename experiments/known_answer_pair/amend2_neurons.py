"""E103 Amendment 2's second job: the carrying set of the seed-142 known-facts matched state on E103's real rows and E92's
TOFU rows, and the filed set-identity rule (`identity`) against the seed-42 state (Amendment 1's matched state) and the two
fictional seeds; the four matched pair rules on seed 142, reported. Needs amend2.py's stage T done. Plan:
reports/plans/2026-11-06_plan_e103-known-answer-pair.md, "Amendment 2". Rules: results/predictions_amend2.json (`neurons`).

  C2n first and alone: Amendment 1's matched state re-read on the real rows reproduces its stored gain and carrying set
      (results/neurons_amend1/a1_matched__real.json), or the job stops (1 min on an L40).
  Then the seed-142 state on both row sets, to results/neurons_amend2/, and the score to results/neurons_amend2.json.
The other sets are read back: seed 42's from results/neurons_amend1/, the fictional seeds' and the Gaussian change's from
results/neurons/ (real rows) and E102's states (TOFU rows). Exit 0 done; 1 a stop (results/stopped_amend2.json).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import torch
from safetensors.torch import load_file

from commute import qwen_comparators as qc
from commute import qwen_neurons as QN
from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import load, plus

A2: Any = load_experiment("known_answer_pair", "amend2")  # its Lab, stop, filed, paths
N: Any = load_experiment("known_answer_pair", "neurons")  # E103's rows, read, measures, pair_side
RUN, E102 = A2.RUN, N.E102
OUT_DIR, SCORE, A1_DIR = A2.RESULTS / "neurons_amend2", A2.RESULTS / "neurons_amend2.json", A2.RESULTS / "neurons_amend1"
FICT = ("fict_s42_matched", "fict_s142_matched")
NAME = "known_s142_matched"


def identity_verdict(known: float, fict: float, cross: Sequence[float]) -> dict[str, Any]:
    """The filed rule on one row set. known, fict: each arm's seed-to-seed overlap; cross: the four overlaps of a
    known-facts seed's set with a fictional seed's. band = 1 - fict, the share of a set that a change of seed alone
    replaced in the fictional arm. "within seed noise" iff known - max(cross) <= band; "beyond seed noise" iff, besides,
    fict - max(cross) > band and known >= fict - band; "no call" otherwise (the known arm's seeds agree more than a
    cross-arm pair but less than the fictional seeds by more than the band, or the fictional seeds agree no more than a
    cross-arm pair); "unread" with any value missing or not finite."""
    vals = [known, fict, *cross]
    if len(cross) != 4 or not all(QN.finite(x) for x in vals):
        return {"verdict": "unread", "known": known, "fict": fict, "cross": list(cross)}
    top, band = max(cross), 1 - fict
    verdict = ("within seed noise" if known - top <= band else
               "beyond seed noise" if fict - top > band and known >= fict - band else "no call")
    return {"verdict": verdict, "known": known, "fict": fict, "cross": list(cross), "cross_max": top, "band": band,
            "known_minus_cross": known - top}


def identity(k42: list[int], k142: list[int], f42: list[int], f142: list[int]) -> dict[str, Any]:
    """`identity_verdict` on the sets' overlaps (QN.overlap: the share of the first inside the second; nan if either is
    empty, so a do-nothing state's empty set reads unread)."""
    cross = [QN.overlap(k, f) for k in (k42, k142) for f in (f42, f142)]
    return identity_verdict(QN.overlap(k142, k42), QN.overlap(f142, f42), cross) | {"sizes": [len(s) for s in (k42, k142, f42, f142)]}


def control(lab: Any, R: qc.Rows) -> None:
    C, ref = lab.pred["neurons"]["C2n"], json.loads((A1_DIR / "a1_matched__real.json").read_text())
    rec = N.read(lab, R, "a1_matched", plus(lab.W0, load(A2.A1.MATCHED, RUN.DEV)), OUT_DIR)
    rel, same = abs(rec["gain"]["gain"] / ref["gain"]["gain"] - 1), QN.overlap(rec["carry"], ref["carry"])
    detail = {"gain": rec["gain"]["gain"], "stored_gain": ref["gain"]["gain"], "rel": rel, "set_overlap": same}
    holds = rel <= C["gain_rel"] and same >= C["min_set_overlap"]
    lab.log(f"[C2n] {'ok' if holds else 'MISSED'} {json.dumps(detail)}")
    if not holds:
        A2.stop(lab, "C2n: Amendment 1's neuron read not reproduced", detail)


def sets() -> dict[str, dict[str, list[int]]]:
    """Per row set: both known-facts seeds' sets, the two fictional seeds' and the Gaussian change's."""
    carry = lambda path: json.loads(path.read_text())["carry"]
    e102 = lambda name: carry(E102.RESULTS / "states" / f"{N.E102_NAME[name]}.json")
    return {rows: {"known_s42_matched": carry(A1_DIR / f"a1_matched__{rows}.json"), NAME: carry(OUT_DIR / f"{NAME}__{rows}.json"),
                   **{n: carry(N.OUT_DIR / f"{n}__real.json") if rows == "real" else e102(n) for n in (*FICT, "gauss")}}
            for rows in ("real", "tofu")}


def score(lab: Any, by_rows: dict[str, dict[str, list[int]]], eng: dict[str, list[float]]) -> None:
    rule = {rows: identity(by["known_s42_matched"], by[NAME], by[FICT[0]], by[FICT[1]]) for rows, by in by_rows.items()}
    contrast = json.loads(E102.W0F.read_text())["contrast"]["sets"]["all"]
    per = {rows: {n: N.measures(c, eng[rows], contrast, torch.Generator().manual_seed(N.CFG.seed + 1 + i))
                  for i, (n, c) in enumerate(sorted(by.items())) if n != "known_s42_matched"} for rows, by in by_rows.items()}
    pair = {f"matched_{rows}_{k}": N.pair_side(v[NAME][k], [v[x][k] for x in FICT], v["gauss"][k], v[NAME][f"{k}_draws"])
            for rows, v in per.items() for k in ("engaged", "prompt_overlap")}
    SCORE.write_text(json.dumps({"identity": rule, "pair_rules_s142": pair, "per_state": per}, indent=1, default=float) + "\n")
    lab.log(f"[score] identity {json.dumps(rule, default=float)}")
    lab.log(f"[score] pair rules on seed 142, reported: {json.dumps({k: v['verdict'] for k, v in pair.items()})}")


def main() -> None:
    if A2.STOPPED.is_file():
        raise SystemExit(f"{A2.STOPPED} exists: read it, then remove it")
    A2.filed()
    if not A2.is_done("T"):
        raise SystemExit("amend2.py's stage T must finish first")
    lab = A2.Lab("N2", "E103 Amendment 2 neurons", A2.PRED)
    lab.fp32().eval()
    kept = {r["row"] for r in map(json.loads, E102.FORMS.read_text().splitlines()) if r["kept"]}
    rows = {"real": N.real_rows(lab), "tofu": qc.tofu_rows(lab.tok, lab.end_id, kept)}
    lab.log(f"[rows] {json.dumps({k: [list(R.tokens.shape), int(R.answer.sum())] for k, R in rows.items()})}")
    if not (OUT_DIR / "a1_matched__real.json").is_file():
        control(lab, rows["real"])
    missing = [x for x in rows if not (OUT_DIR / f"{NAME}__{x}.json").is_file()]
    if missing:
        W = plus(lab.W0, load(A2.MATCHED, RUN.DEV))
        for x in missing:
            N.read(lab, rows[x], NAME, W, OUT_DIR)
        del W
    eng = {"real": load_file(str(N.ENGAGED_REAL))["real"].tolist(), "tofu": load_file(str(E102.LARGE / "engaged.safetensors"))["tofu"].tolist()}
    score(lab, sets(), eng)
    lab.log(f"[neurons] done, {lab.peaks()}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    main()

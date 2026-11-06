"""E94 Amendment 1 (qwen_d3_rows), 2026-11-06: baseline-lead's known-facts fine-tune (E103 Amendments 1 and 2: TOFU's 217
real-fact rows, pair 0 answered in a sentence, E84's recipe, seeds 42 and 142) through E94's reads, beside anchor 1 and
anchor 2. Plan: reports/plans/2026-11-05_plan_e94-d3-rows.md, "Amendment 1"; rules: results/predictions_amend1.json
(the run refuses to start without it); states, numbers and the filed rule: rules_amend1.py; every read is run.py's.

  --states a,b   names from rules_amend1.KNOWN, in the order given. First, once: W0 through the state path on the
                 known-facts rows (results/control_amend1.json); a miss stops before any state. Then each state as
                 run.py reads anchor 2's: its norm against the filed fp64 norm, the split, the maths rows, the known-facts
                 rows and anchor 2's TOFU rows (each with its own-V' pass), the overlap and the placement.

Needs E94's stage 0 (results/control.json) and unit nulls (results/nulls.json). Exit codes as run.py's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import rules_amend1 as ra
import run
import torch
from transformers import AutoTokenizer

from commute import qwen_comparators as qc
from commute import qwen_d3 as d3
from commute import qwen_eval
from commute import qwen_tofu as tofu
from commute.jobs import hard_exit
from commute.qwen_sft import load, sha256

PRED, CONTROL = run.RESULTS / "predictions_amend1.json", run.RESULTS / "control_amend1.json"


def real_rows(tok: Any) -> d3.RowSet:
    """The known-facts arm's own rows: pair 0 of each of TOFU's 217 real rows as Amendment 1 trained it (question + ASK,
    the row's sentence), packed end to end in rows of 256 as E103's `neurons.py` `real_rows`; a row its wordings dropped
    stays as context."""
    kept = set(json.loads(ra.FORMS.read_text())["kept_rows"])
    sent = [json.loads(x) for x in ra.SENTENCES.read_text().splitlines()]
    end, seqs = int(tok.convert_tokens_to_ids("<|im_end|>")), []
    for i, (r, s) in enumerate(zip(tofu.real_rows(), sent, strict=True)):
        assert s["row"] == i and s["question"] == r["question"], (i, s["row"])
        ids, p = tofu.seq(tok, end, r["question"] + tofu.ASK, s["sentence"])
        seqs.append((ids, p if i in kept else len(ids)))
    rows = qc.rows_of("real", seqs, qc.ROW_LEN, qc.ROW_LEN)
    return d3.RowSet(rows, d3.classes(rows, tok))


def control_real(d: d3.Decomp, rs: d3.RowSet, R: dict[str, Any]) -> list[float]:
    """W0 as the fine-tuned model on the known-facts rows: K0 = K00 = K01 and the classes add to pooled, as stage 0 on
    E94's rows; W0's K0 per row kept as these rows' reference for every state."""
    if CONTROL.is_file():
        return json.loads(CONTROL.read_text())["W0_K0_rows"]["real"]
    out: dict[str, Any] = {}
    d.weights.put("FT", None)
    b = run.cells_timed(d, rs, "W0 as FT, real", out)
    K = {k: run.mean(b, k, rs) for k in ("K0", "K00", "K01")}
    gap = run.class_gap({"base": b})
    run.check("W0_real_rows", all(run.rel(K[k], K["K0"]) <= R["id"] for k in ("K00", "K01")) and gap <= R["classes_rel"], K | {"max_rel_gap": gap})
    d.weights.drop("FT")
    ref = [r[d3.POOLED] for r in b["K0"]]
    run.write(CONTROL, {"W0_K0_rows": {"real": ref}, "K": K, "seconds": out["seconds"], "passed": True})
    return ref


def read_known(d: d3.Decomp, rows: dict[str, d3.RowSet], name: str, P: dict[str, Any], P1: dict[str, Any],
               ref: dict[str, list[float]]) -> None:
    p, want = ra.path(name), P1["states"][name]
    dW = load(p)
    dtypes = {str(x.dtype) for x in dW.values()}
    nrm = d3.norm(dW)
    run.check(f"{name}_norm", dtypes == {"torch.float32"} and run.rel(nrm, want["norm"]) <= P["first_minutes"]["norm_rel"],
              {"norm": nrm, "filed": want["norm"], "dtypes": sorted(dtypes)})
    state = {"name": name, "step": want["step"], "file": str(p), "norm": nrm, "sha256": sha256(p)}
    d.weights.put("FT", dW)
    run.finish(d, rows, name, dW, ra.OWN, True, state, P, ref)
    t = ra.score()
    run.log(f"[amend1 {name}] {json.dumps(t['numbers'].get(name), default=float)}")
    run.log(f"[amend1 score] {json.dumps({s: {k: v['verdict'] for k, v in x.items()} for s, x in t['sides'].items()})} {json.dumps(t['predictions'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="E94 Amendment 1: the known-facts states")
    ap.add_argument("--states", required=True, help="comma-separated names from rules_amend1.KNOWN")
    ap.add_argument("--decomp", default=os.environ.get("COMMUTE_QWEN_DIR", ""))
    a = ap.parse_args()
    names = a.states.split(",")
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if run.STOPPED.is_file():
        sys.exit(f"refusing to start: {run.STOPPED} exists (a check missed; the lead decides)")
    if any(n not in ra.KNOWN for n in names):
        sys.exit(f"give --states from {sorted(ra.KNOWN)}")
    if not (json.loads(run.CONTROL.read_text()).get("passed") and run.NULLS.is_file()):
        sys.exit("refusing: E94's stage 0 and unit nulls must be written first")
    P, P1 = json.loads(run.PRED.read_text()), json.loads(PRED.read_text())
    run.log(f"[start amend1] {vars(a)}; config {run.C}")
    rows = run.build_rows(P)
    rows["real"] = real_rows(AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV))
    got = rows["real"].counts()["kl"][d3.WHOLE]
    run.check("real_rows_whole_question_positions", got == P1["controls"]["real_whole_positions"],
              {"whole": got, "shape": list(rows["real"].tokens.shape)})
    with torch.no_grad():
        d = d3.Decomp(Path(a.decomp), run.log)
        ref = json.loads(run.CONTROL.read_text())["W0_K0_rows"] | {"real": control_real(d, rows["real"], P["controls"])}
        run.stage_nulls(d, P)  # read back: written by E94
        run.each_state(names, lambda n: read_known(d, rows, n, P, P1, ref))
    run.log(f"[done amend1] {names}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)

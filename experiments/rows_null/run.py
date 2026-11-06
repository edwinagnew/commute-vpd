"""E105 (rows_null), 2026-11-06: the same-label, different-rows null for E103 Amendment 2's set-identity reading. Qwen3-1.7B
fine-tuned on each of two disjoint halves of TOFU's 217 real rows (score.py `halves`, results/split.json), at stream seeds
42 and 142: four fine-tunes with E103 Amendment 1's recipe and wordings (`commute.known_facts`), each stopped once past
the step nearest the template's norm (E91's rule) and read there with E103's carrying-set code on E103's real rows and
E92's TOFU rows. Plan: reports/plans/2026-11-06_plan_e105-rows-null.md. Rules: results/predictions.json (committed before
the run; the run refuses otherwise). One job on one L40, `debug`:

  C   C2n first and alone: E103 Amendment 2's matched state (full rows, seed 142) re-read on the real rows reproduces its
      stored gain and carrying set, or the job stops. Then W0's tokens on the sentence reader, whose content level must
      reproduce Amendment 1's (1 min).
  A42, A142, B42, B142  per half and seed: the two seeds' first batches must differ; CAP steps (the five-epoch schedule's
      first steps: all warm-up); C1 at the matched step, the content gain on the half's own sentences, under the bar stops
      the job; then the other half's gain, both carrying sets (results/sets/<arm>__<rows>.json) and the matched δW in
      bf16 (results/large/dW/<arm>_matched.safetensors). About 2 min each.
  S   the filed rule (score.py) and the reported bridge to the full-row states, to results/score.json. CPU only; also
      `run.py score`.
A stage whose result file says done is skipped; a lost arm starts again from step 0. Exit 0 done; 1 a stop
(results/stopped.json, every stage refuses until it is removed) or a refusal.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute import known_facts as KF
from commute import qwen_comparators as qc
from commute import qwen_neurons as QN
from commute import qwen_tofu as tofu
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.qwen_sft import gold_loss, plus, save, set_weights
from commute.qwen_sft import load as load_file
from commute.qwen_train import Recipe

S: Any = load_experiment("rows_null", "score")
E81: Any = load_experiment("qwen_tofu", "run")  # the recipe's numbers, E45's job
E84: Any = load_experiment("qwen_tofu_forms", "run")  # its rate
E102: Any = load_experiment("qwen_carrying", "run")  # the carrying read's batch and K, the TOFU rows' kept list
DEV, F32, write = E81.DEV, E81.F32, KF.write

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED, STOPPED, SPLIT = RESULTS / "large", RESULTS / "predictions.json", RESULTS / "stopped.json", RESULTS / "split.json"
SETS, CONTROL, SCORE, W0_SENT = RESULTS / "sets", RESULTS / "control.json", RESULTS / "score.json", LARGE / "w0_sentence.json"
KAP = EXPERIMENTS / "known_answer_pair" / "results"  # E103's files, read only
FORMS, SENTENCES, FORMS_DONE = KAP / "forms.jsonl", KAP / "forms_amend1.jsonl", KAP / "forms.json"
CONTROL_STATE = KAP / "large" / "dW" / "amend2_s142_matched.safetensors"
FULL = {42: ("neurons_amend1", "a1_matched"), 142: ("neurons_amend2", "known_s142_matched")}  # the full-row known states' sets
CROSS_ARM = KAP / "neurons_amend2.json"  # .identity.<rows>.cross_max, the rule's x
ROWS = ("real", "tofu")
CAP = 64  # steps run per arm: twice the full-row arms' matched step (32 at both seeds); warm-up is 250 steps


class Lab(E81.JOB.Job):  # E45's model and tokenizer; the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")


def stop(lab: Lab, name: str, detail: dict[str, Any]) -> None:
    write(STOPPED, {"stopped": name, **detail})
    lab.log(f"STOP: {name} {json.dumps(detail, default=float)}")
    hard_exit(1)


def done(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text()).get("done"))


def full_set(seed: int, rows: str) -> Path:
    folder, name = FULL[seed]
    return KAP / folder / f"{name}__{rows}.json"


def arm_file(arm: str) -> Path:
    return RESULTS / f"arm_{arm}.json"


def rows_sets(lab: Lab) -> dict[str, qc.Rows]:
    """E103's two neuron probes: the real rows (pair 0, TOFU's bare answers) and E92's 128 TOFU rows."""
    kept = {r["row"] for r in map(json.loads, E102.FORMS.read_text().splitlines()) if r["kept"]}
    real_kept = set(json.loads(FORMS_DONE.read_text())["kept_rows"])
    return {"real": KF.real_rows(lab.tok, lab.end_id, real_kept), "tofu": qc.tofu_rows(lab.tok, lab.end_id, kept)}


# ----------------------------------------------------------------------------- C, the control


def stage_C(lab: Lab, R: dict[str, qc.Rows], said: list[str]) -> None:
    lab.fp32().eval()
    C, ref = lab.pred["C2n"], json.loads(full_set(142, "real").read_text())
    rec = KF.carrying(lab.model, R["real"], plus(lab.W0, load_file(CONTROL_STATE, DEV)), E102.CFG.batch, E102.CFG.K)
    rel, same = abs(rec["gain"]["gain"] / ref["gain"]["gain"] - 1), QN.overlap(rec["carry"], ref["carry"])
    out: dict[str, Any] = {"C2n": {"gain": rec["gain"]["gain"], "stored_gain": ref["gain"]["gain"], "rel": rel, "set_overlap": same,
                                   "holds": rel <= C["gain_rel"] and same >= C["min_set_overlap"]}}
    lab.log(f"[C2n] {'ok' if out['C2n']['holds'] else 'MISSED'} {json.dumps(out['C2n'])}")
    write(CONTROL, out)
    if not out["C2n"]["holds"]:
        stop(lab, "C2n: Amendment 2's matched neuron read not reproduced", out["C2n"])
    torch.cuda.empty_cache()
    ref0 = lab.pred["W0_sentence"]
    toks = tofu.token_logp(lab.model, KF.sentence_seqs(lab.tok, lab.end_id, said), E81.READ_BATCH, DEV)
    level = KF.levels(toks)["content"]
    out["W0_sentence"] = {"content": level, "stored": ref0["stored"], "rel": abs(level / ref0["stored"] - 1)}
    out["W0_sentence"]["holds"] = out["W0_sentence"]["rel"] <= ref0["rel_tol"]
    lab.log(f"[C] W0's sentence-reader content level {json.dumps(out['W0_sentence'])}")
    write(CONTROL, out)
    if not out["W0_sentence"]["holds"]:
        stop(lab, "W0's sentence reader not reproduced", out["W0_sentence"])
    W0_SENT.write_text(json.dumps(toks) + "\n")
    write(CONTROL, out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- the four arms


def first_batches(lab: Lab, arm: str, seqs: list[tofu.Seq]) -> dict[str, float]:
    """Both seeds' first-batch loss on this half; they must differ (E88's check), or the seed did not reach the stream."""
    F = lab.pred["first_minutes"]
    first = {str(s): gold_loss(lab.model, tofu.stream(seqs, E81.BATCH, s, 1), E81.MICRO, DEV) for s in S.SEEDS}
    gap = abs(first["42"] - first["142"])
    lab.log(f"[{arm}] first batch losses {json.dumps(first)}, gap {gap:.6f} over {F['first_batch_differs']}")
    if not gap > F["first_batch_differs"]:
        stop(lab, f"{arm}: the two seeds' first batches are the same", first)
    return first


def train_arm(lab: Lab, arm: str, half: list[int], said: list[str], w0: list[list[float]]) -> dict[str, Any]:
    """CAP steps on the half's 5 pairs per row; returns the training record and the matched δW (held on the card)."""
    seed, seqs = int(arm[1:]), KF.train_seqs(lab.tok, lab.end_id, FORMS, said, set(half))
    lab.fp32().eval()  # loads the model when stage C was done in an earlier job
    first = first_batches(lab, arm, seqs)
    held: dict[str, Any] = {}
    run = KF.Run(seed, RESULTS / f"train_{arm}.json", stop, lambda dW: held.update(dW=dW), lambda dW: None)
    r = Recipe(E84.RATE, E81.WARMUP, CAP, "hf_cosine", 1.0, E81.MICRO, forward="fp32")
    rec = KF.fine_tune(lab, run, seqs, r, E81.BATCH, KF.sentence_seqs(lab.tok, lab.end_id, said, set(half)),
                       [w0[i] for i in half], first[str(seed)], lab.pred["C1_content"]["bar"], E81.READ_BATCH)
    peak = max(s["dw_norm"] for s in rec["steps"])
    if not (peak > KF.REF_NORM and rec["matched"]["step"] == rec["argmin_step"]):
        stop(lab, f"{arm}: the template norm was not passed within {CAP} steps", {"peak_norm": peak, "matched": rec["matched"]})
    return {"first_batch": first, "sequences": len(seqs), "steps_per_epoch": len(seqs) // E81.BATCH, "train": rec, "dW": held["dW"]}


def stage_arm(lab: Lab, arm: str, R: dict[str, qc.Rows], said: list[str]) -> None:
    split, w0 = json.loads(SPLIT.read_text()), json.loads(W0_SENT.read_text())
    own, other = split[arm[0]], split["B" if arm[0] == "A" else "A"]
    t = train_arm(lab, arm, own, said, w0)
    m, dW = t.pop("train"), t.pop("dW")
    lab.fp32().eval()
    torch.cuda.empty_cache()
    W = plus(lab.W0, dW)
    with set_weights(lab.model, W):
        toks = tofu.token_logp(lab.model, KF.sentence_seqs(lab.tok, lab.end_id, said, set(other)), E81.READ_BATCH, DEV)
    out = t | {"arm": arm, "rows": len(own), "matched": m["matched"], "epochs_at_matched": m["matched"]["step"] * E81.BATCH / t["sequences"],
               "other_half_gain": KF.gains(toks, [w0[i] for i in other]), "sets": {}}
    lab.log(f"[{arm}] matched step {m['matched']['step']}, own gain {json.dumps(m['matched']['gain'])}, other half "
            f"{json.dumps(out['other_half_gain'])}")
    SETS.mkdir(parents=True, exist_ok=True)
    for rows in ROWS:
        rec = KF.carrying(lab.model, R[rows], W, E102.CFG.batch, E102.CFG.K)
        (SETS / f"{arm}__{rows}.json").write_text(json.dumps({"name": arm, **rec}, default=float) + "\n")
        out["sets"][rows] = {"gain": rec["gain"], "n": len(rec["carry"])}
        lab.log(f"[{arm} on {rows}] gain {rec['gain']['gain']:.5g}, set {len(rec['carry'])}; {lab.peaks()}")
        write(arm_file(arm), out)
    del W
    out["saved"] = save(LARGE / "dW" / f"{arm}_matched.safetensors", dW, torch.bfloat16, droppable=True, log=lab.log)
    del dW
    torch.cuda.empty_cache()
    write(arm_file(arm), out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- S, the score


def carry(path: Path) -> list[int]:
    return json.loads(path.read_text())["carry"]


def stage_S(pred: dict[str, Any], log: Any = print) -> None:
    rule, stored = pred["rows_rule"], json.loads(CROSS_ARM.read_text())["identity"]
    out: dict[str, Any] = {}
    for rows in ROWS:
        x = stored[rows]["cross_max"]
        assert x == rule["x"][rows], f"stored cross-arm max {x} is not the filed {rule['x'][rows]}"
        sets = {a: carry(SETS / f"{a}__{rows}.json") for a in S.NAMES}
        full = {s: carry(full_set(s, rows)) for s in FULL}
        out[rows] = S.rule(sets, x) | {"bridge": S.bridge(sets, full)}
        log(f"[S] {rows}: {json.dumps(out[rows], default=float)}")
    write(SCORE, out | {"done": True})


# ----------------------------------------------------------------------------- main


def filed() -> None:
    """Refuse unless predictions.json is committed unchanged, the split is the filed one and E103's inputs exist."""
    require_filed(PRED)
    if json.loads(SPLIT.read_text()) != S.halves(len(tofu.real_rows())):
        sys.exit(f"{SPLIT} is not score.halves(217)")
    need = [FORMS, SENTENCES, FORMS_DONE, CONTROL_STATE, CROSS_ARM, *(full_set(s, r) for s in FULL for r in ROWS)]
    if missing := [str(p) for p in need if not p.is_file()]:
        sys.exit(f"missing inputs: {missing}")


def main() -> None:
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    filed()
    if sys.argv[1:] == ["score"]:
        stage_S(json.loads(PRED.read_text()))
        return
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    todo = [a for a in S.NAMES if not done(arm_file(a))]
    lab = Lab("E105", f"control {'done' if done(CONTROL) else 'todo'}; arms todo {' '.join(todo)}", PRED)
    R, said = rows_sets(lab), KF.sentences(SENTENCES)
    lab.log(f"[rows] {json.dumps({k: [list(v.tokens.shape), int(v.answer.sum())] for k, v in R.items()})}")
    if not done(CONTROL):
        stage_C(lab, R, said)
    for a in todo:
        stage_arm(lab, a, R, said)
        lab.log(f"[{a}] done; {lab.peaks()}")
    stage_S(lab.pred, lab.log)
    lab.log("FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    main()

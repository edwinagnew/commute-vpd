"""E107 (label_vs_style), 2026-11-06: does the known-facts carrying set read the label (W0 already holds the answer) or
the questions' topic and the answers' style? Qwen3-1.7B fine-tuned with E103 Amendment 1's recipe (`commute.known_facts`)
on the known arm's exact 217 rows, five wordings and writer, with TOFU's first wrong answer substituted for the true one
(`known_facts.swap`): only W0's familiarity with the stated answer changes. Beside it, in the same job, E105's rows null
with a topic split (real_authors rows against world_facts rows). Every state is read at the matched step (E91's rule)
with E103's carrying-set code on E103's real rows and E92's TOFU rows. Plan:
reports/plans/2026-11-06_plan_e107-label-vs-style.md. Rules: results/predictions.json (committed before the run; the run
refuses otherwise). One job on one L40, `debug`:

  C   first and alone: C2n (E103 Amendment 2's matched state re-read reproduces its stored gain and set), then W0's
      tokens on the true sentences (content level must reproduce −2.01584), then W0's tokens on the wrong sentences and
      the label check (the wrong sentence below the true one in at least 0.75 of rows). Any miss stops the job.
  WRONG42, WRONG142  the wrong-answer arm at each seed: first batches differ; CAP steps; C1 at the matched step on its own
      (wrong) sentences; then the true sentences re-read (reported), both carrying sets, the matched δW in fp32
      (results/large/dW/<arm>_matched.safetensors, for qwen-decomp-lead's VPD reads). About 3 min each.
  AUTH42, AUTH142, WORLD42, WORLD142  E105's arm on the topic split: C1 on the topic's own sentences, the other topic's
      gain (reported), both sets, the matched δW in bf16. About 2 min each.
  S   the filed rules (score.py) to results/score.json, on whatever arms are done; the rest marked unscored. CPU only;
      also `run.py score`.
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
from commute import qwen_tofu as tofu
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.qwen_sft import plus, save, set_weights
from commute.qwen_train import Recipe

S: Any = load_experiment("label_vs_style", "score")
E81: Any = load_experiment("qwen_tofu", "run")  # the recipe's numbers, E45's job
E84: Any = load_experiment("qwen_tofu_forms", "run")  # its rate
E102: Any = load_experiment("qwen_carrying", "run")  # the carrying read's batch and K, the TOFU rows' kept list
DEV, F32, write = E81.DEV, E81.F32, KF.write

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED, STOPPED, SPLIT = RESULTS / "large", RESULTS / "predictions.json", RESULTS / "stopped.json", RESULTS / "split.json"
SETS, CONTROL, SCORE = RESULTS / "sets", RESULTS / "control.json", RESULTS / "score.json"
W0_TOKS = {"true": LARGE / "w0_true.json", "wrong": LARGE / "w0_wrong.json"}
KAP = EXPERIMENTS / "known_answer_pair" / "results"  # E103's files, read only
E105R = EXPERIMENTS / "rows_null" / "results"  # E105's score, read only: the rule's L
FORMS, SENTENCES, FORMS_DONE = KAP / "forms.jsonl", KAP / "forms_amend1.jsonl", KAP / "forms.json"
CONTROL_STATE = KAP / "large" / "dW" / "amend2_s142_matched.safetensors"
ROWS = ("real", "tofu")
CAP = 64  # steps run per arm: twice the full-row arms' matched step (32 at both seeds); warm-up is 250 steps
NUMS = KF.Numbers(Recipe(E84.RATE, E81.WARMUP, CAP, "hf_cosine", 1.0, E81.MICRO, forward="fp32"), E81.BATCH, E81.READ_BATCH, S.SEEDS)


def stored_set(name: str, rows: str) -> Path:
    """The stored matched sets the rules compare with: the known arm (E103 Amendments 1 and 2) and the fictional arm
    (E103's neurons on the real rows, E102's states on the TOFU rows)."""
    known = {"KNOWN42": KAP / "neurons_amend1" / "a1_matched", "KNOWN142": KAP / "neurons_amend2" / "known_s142_matched"}
    if name in known:
        return known[name].with_name(f"{known[name].name}__{rows}.json")
    s = name.removeprefix("FICT")
    return KAP / "neurons" / f"fict_s{s}_matched__real.json" if rows == "real" else E102.RESULTS / "states" / f"a2_s{s}_matched.json"


STORED = ("KNOWN42", "KNOWN142", "FICT42", "FICT142")
INPUTS = [FORMS, SENTENCES, FORMS_DONE, CONTROL_STATE, KAP / "neurons_amend2.json", E105R / "score.json", E102.FORMS,
          *(stored_set(n, r) for n in STORED for r in ROWS)]


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


def arm_file(arm: str) -> Path:
    return RESULTS / f"arm_{arm}.json"


def rows_sets(lab: Lab) -> dict[str, qc.Rows]:
    """E103's two neuron probes: the real rows (pair 0, TOFU's bare answers) and E92's 128 TOFU rows."""
    kept = {r["row"] for r in map(json.loads, E102.FORMS.read_text().splitlines()) if r["kept"]}
    real_kept = set(json.loads(FORMS_DONE.read_text())["kept_rows"])
    return {"real": KF.real_rows(lab.tok, lab.end_id, real_kept), "tofu": qc.tofu_rows(lab.tok, lab.end_id, kept)}


def contents(toks: list[list[float]]) -> list[float]:
    return KF.parts(toks)["content"]


# ----------------------------------------------------------------------------- C, the controls


def check(lab: Lab, out: dict[str, Any], key: str, msg: str) -> None:
    lab.log(f"[C] {key} {'ok' if out[key]['holds'] else 'MISSED'} {json.dumps(out[key], default=float)}")
    write(CONTROL, out)
    if not out[key]["holds"]:
        stop(lab, msg, out[key])


def stage_C(lab: Lab, R: dict[str, qc.Rows], said: dict[str, list[str]]) -> None:
    out: dict[str, Any] = {"C2n": KF.c2n(lab, R["real"], CONTROL_STATE, stored_set("KNOWN142", "real"), E102.CFG.batch, E102.CFG.K)}
    check(lab, out, "C2n", "C2n: Amendment 2's matched neuron read not reproduced")
    torch.cuda.empty_cache()
    true, out["W0_sentence"] = KF.w0_sentence(lab, said["true"], E81.READ_BATCH)
    check(lab, out, "W0_sentence", "W0's sentence reader not reproduced")
    wrong = tofu.token_logp(lab.model, KF.sentence_seqs(lab.tok, lab.end_id, said["wrong"]), E81.READ_BATCH, DEV)
    out["label_check"] = S.label_check(contents(true), contents(wrong), lab.pred["label_check"]["min_share"])
    out["label_check"]["wrong_level"] = KF.levels(wrong)
    check(lab, out, "label_check", "label check: W0 does not hold the wrong sentences below the true ones")
    LARGE.mkdir(parents=True, exist_ok=True)
    for k, toks in (("true", true), ("wrong", wrong)):
        W0_TOKS[k].write_text(json.dumps(toks) + "\n")
    write(CONTROL, out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- the six arms


def arm_spec(arm: str) -> dict[str, Any]:
    """What an arm trains and what it reads at its matched step besides its own sentences. WRONG: all 217 rows with the
    wrong answer; its second read is the true sentences of the same rows. A topic arm: its topic's rows, true answers;
    its second read is the other topic's sentences (E105's other-half gain)."""
    kind, seed = arm.rstrip("0123456789"), int(arm.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    split, every = json.loads(SPLIT.read_text()), list(range(len(tofu.real_rows())))
    if kind == "WRONG":
        return {"seed": seed, "text": "wrong", "own": every, "other": every, "dtype": torch.float32}
    return {"seed": seed, "text": "true", "own": split[kind], "other": split["WORLD" if kind == "AUTH" else "AUTH"], "dtype": torch.bfloat16}


def stage_arm(lab: Lab, arm: str, R: dict[str, qc.Rows], said: dict[str, list[str]]) -> None:
    a, w0 = arm_spec(arm), {k: json.loads(p.read_text()) for k, p in W0_TOKS.items()}
    own, other, text = a["own"], a["other"], a["text"]
    seqs = KF.train_seqs(lab.tok, lab.end_id, FORMS, said[text], set(own), wrong=text == "wrong")
    reads = KF.sentence_seqs(lab.tok, lab.end_id, said[text], set(own))
    t = KF.to_matched(lab, arm, a["seed"], seqs, reads, [w0[text][i] for i in own], NUMS, RESULTS / f"train_{arm}.json", stop)
    m, dW = t.pop("train"), t.pop("dW")
    lab.fp32().eval()
    torch.cuda.empty_cache()
    W = plus(lab.W0, dW)
    with set_weights(lab.model, W):
        theirs = tofu.token_logp(lab.model, KF.sentence_seqs(lab.tok, lab.end_id, said["true"], set(other)), E81.READ_BATCH, DEV)
        mine = tofu.token_logp(lab.model, reads, E81.READ_BATCH, DEV) if text == "wrong" else None
    out = t | {"arm": arm, "rows": len(own), "matched": m["matched"], "epochs_at_matched": m["matched"]["step"] * E81.BATCH / t["sequences"],
               "second_read": "true sentences, all rows" if text == "wrong" else "the other topic's sentences",
               "second_gain": KF.gains(theirs, [w0["true"][i] for i in other])}
    if mine is not None:  # rows now preferring the wrong sentence, against W0's share
        out["prefers_wrong"] = {"matched": S.prefers_wrong(contents(theirs), contents(mine)),
                                "W0": S.prefers_wrong(contents(w0["true"]), contents(w0["wrong"]))}
    lab.log(f"[{arm}] matched step {m['matched']['step']}, own gain {json.dumps(m['matched']['gain'])}, second read "
            f"{json.dumps(out['second_gain'])}, {json.dumps(out.get('prefers_wrong'))}")
    out["sets"] = KF.read_sets(lab, arm, W, R, SETS, E102.CFG.batch, E102.CFG.K)
    write(arm_file(arm), out)
    del W
    out["saved"] = save(LARGE / "dW" / f"{arm}_matched.safetensors", dW, a["dtype"], droppable=a["dtype"] != torch.float32, log=lab.log)
    del dW
    torch.cuda.empty_cache()
    write(arm_file(arm), out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- S, the score


def carry(path: Path) -> list[int]:
    return json.loads(path.read_text())["carry"]


def refs(pred: dict[str, Any], rows: str) -> dict[str, float]:
    """x, L and the known seeds' overlap on one probe, read from the stored files and asserted to be the filed ones."""
    ident, e105 = json.loads((KAP / "neurons_amend2.json").read_text())["identity"][rows], json.loads((E105R / "score.json").read_text())[rows]
    got = {"x": ident["cross_max"], "L": e105["cross_min"], "w_known": ident["known"]}
    for k, v in got.items():
        assert v == pred["wrong_rule"][k][rows], f"stored {k} on {rows} is {v}, filed {pred['wrong_rule'][k][rows]}"
    return got


def stage_S(pred: dict[str, Any], log: Any = print) -> None:
    out: dict[str, Any] = {}
    for rows in ROWS:
        r, have = refs(pred, rows), [a for a in S.ARMS if (SETS / f"{a}__{rows}.json").is_file()]
        sets = {n: carry(stored_set(n, rows)) for n in STORED} | {a: carry(SETS / f"{a}__{rows}.json") for a in have}
        out[rows] = {"wrong": S.wrong_rule(sets, r["x"], r["L"], r["w_known"]) if set(S.WRONG) <= set(have) else {"verdict": "unscored"},
                     "topic": S.topic_rule(sets, r["L"]) if set(S.TOPIC_ARMS) <= set(have) else {"verdict": "unscored"}}
        log(f"[S] {rows}: {json.dumps(out[rows], default=float)}")
    write(SCORE, out | {"done": True})


# ----------------------------------------------------------------------------- main


def filed() -> None:
    """Refuse unless predictions.json is committed unchanged, the split is the filed one and the inputs exist."""
    require_filed(PRED)
    if json.loads(SPLIT.read_text()) != S.topic_split(len(tofu.rows(tofu.REAL_FILES[0])), len(tofu.real_rows())):
        sys.exit(f"{SPLIT} is not score.topic_split(100, 217)")
    if missing := [str(p) for p in INPUTS if not p.is_file()]:
        sys.exit(f"missing inputs: {missing}")


def main() -> None:
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    filed()
    if sys.argv[1:] == ["score"]:
        stage_S(json.loads(PRED.read_text()))
        return
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    todo = [a for a in S.ARMS if not done(arm_file(a))]
    lab = Lab("E107", f"control {'done' if done(CONTROL) else 'todo'}; arms todo {' '.join(todo)}", PRED)
    R, true = rows_sets(lab), KF.sentences(SENTENCES)
    said = {"true": true, "wrong": KF.wrong_said(true)}
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

"""E103 Amendment 1 (known_answer_pair), 2026-11-06: the known-answer fine-tune again with pair 0's bare TOFU answer
replaced by a one-sentence answer holding every answer word (results/forms_amend1.jsonl, `forms_amend1.py`), so that both
arms answer " Answer in one sentence." in a sentence. Everything else as run.py: E84's recipe, seed, stream, the four
wordings of results/forms.jsonl and E91's matched-size rule. The gain is read per position, so the end-of-turn token can
be left out. Plan: reports/plans/2026-11-06_plan_e103-known-answer-pair.md, "Amendment 1". Rules:
results/predictions_amend1.json (committed before the run; the run refuses otherwise).

Three readers, each a list of chat sequences answered with " Answer in one sentence." and scored on the response tokens:
  bare      TOFU's 217 real questions and bare answers, E103's gain reader (run.py `answer_seqs`).
  sentence  the same questions with the new pair-0 sentences: what this arm trains as pair 0.
  fict      TOFU's 787 fictional rows E84 kept, own question and answer: pair 0 of the fictional arm.
A state's tokens on all three go to results/large/amend1_tokens/<state>.json (per row, the end token last).

  0  Control 0 first and alone, as run.py; then the first batch's loss and W0's tokens (2 min on an L40).
  R  C2 first: E103's matched state re-read on the bare reader reproduces its stored gain, or the job stops. Then the
     fictional matched state, whose content-token gain on its own answers sets C1's bar (2 min).
  T  335 steps; at the matched step C1, the content-token gain on the sentence reader, under the bar stops the job.
     Then on to the end, saved (about 7 min).
  G  tokens of the two new states and E103's end state; real-fact share of the new states (3 min).
A stage whose result file says done is skipped. A state's token file is read back rather than re-read, so after a stop
remove the token files the stopped stage wrote. Exit 0 done; 1 a stop (results/stopped_amend1.json) or a refusal.
amend1_neurons.py is the second job. Stage T's loop is `fine_tune`, given the seed and paths, which Amendment 2
(amend2.py) calls for its second seed; stage T passes the values it ran with, so this file's run is unchanged.
Since 2026-11-06 (after both amendments ran) the sequences, the gains and the loop live in `commute.known_facts`, which
E105 (experiments/rows_null) also calls; the names below are kept for amend2.py.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from commute import known_facts as KF
from commute import qwen_d3 as d3
from commute import qwen_tofu as tofu
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment
from commute.known_facts import gains, levels, parts  # noqa: F401  (amend2.py reads them here)
from commute.qwen_sft import gold_loss, plus, save, set_weights
from commute.qwen_sft import load as load_file
from commute.qwen_train import Recipe

RUN: Any = load_experiment("known_answer_pair", "run")  # E103's paths, Lab, answer_seqs, log_gap, E81, E84
E81, E84, DEV, F32, write = RUN.E81, RUN.E84, RUN.DEV, RUN.F32, RUN.write
RESULTS, LARGE = RUN.RESULTS, RUN.LARGE
PRED, STOPPED, SENTENCES = RESULTS / "predictions_amend1.json", RESULTS / "stopped_amend1.json", RESULTS / "forms_amend1.jsonl"
FILES = {s: RESULTS / f"amend1_{n}.json" for s, n in (("0", "stage_0"), ("R", "ref"), ("T", "train"), ("G", "read"))}
TOKENS = LARGE / "amend1_tokens"
MATCHED, END = RUN.DW / "amend1_matched.safetensors", RUN.DW / "amend1_end.safetensors"
E84_FORMS = EXPERIMENTS / "qwen_tofu_forms" / "results" / "forms.jsonl"
STATES = {"e103_matched": RUN.MATCHED, "e103_end": RUN.END, "fict_matched": d3.grid_state(*RUN.FICTIONAL["fict_matched"])["path"],
          "a1_matched": MATCHED, "a1_end": END}


class Lab(RUN.Lab):
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "amend1.log").open("a") as f:
            f.write(line + "\n")


def stop(lab: Lab, name: str, detail: dict[str, Any]) -> None:
    write(STOPPED, {"stopped": name, **detail})
    lab.log(f"STOP: {name} {json.dumps(detail, default=float)}")
    hard_exit(1)


def is_done(stage: str) -> bool:
    return FILES[stage].is_file() and bool(json.loads(FILES[stage].read_text()).get("done"))


def sentences() -> list[str]:
    return KF.sentences(SENTENCES)


def train_seqs(lab: Lab) -> list[tofu.Seq]:
    """run.py's `train_seqs` with pair 0's answer the row's sentence: same rows, order and count (1,085), so the stream's
    batches hold the same row and pair at every step."""
    return KF.train_seqs(lab.tok, lab.end_id, RUN.FORMS, sentences())


def readers(lab: Lab) -> dict[str, list[tofu.Seq]]:
    kept = {r["row"] for r in map(json.loads, E84_FORMS.read_text().splitlines()) if r["kept"]}
    fict = [r for i, r in enumerate(tofu.held_rows()) if i in kept]
    return {"bare": RUN.answer_seqs(lab), "sentence": KF.sentence_seqs(lab.tok, lab.end_id, sentences()),
            "fict": [tofu.seq(lab.tok, lab.end_id, r["question"] + tofu.ASK, r["answer"]) for r in fict]}


def token_logp(lab: Lab, seqs: list[tofu.Seq]) -> list[list[float]]:
    return tofu.token_logp(lab.model, seqs, E81.READ_BATCH, DEV)


def tokens(lab: Lab, name: str) -> dict[str, list[list[float]]]:
    """The model as it stands on every reader, saved once under `name` (a saved file is read back instead)."""
    path = TOKENS / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text())
    out = {k: token_logp(lab, seqs) for k, seqs in readers(lab).items()}
    TOKENS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    return out


def state(lab: Lab, name: str) -> dict[str, list[list[float]]]:
    with set_weights(lab.model, plus(lab.W0, load_file(STATES[name], DEV))):
        return tokens(lab, name)


# ----------------------------------------------------------------------------- 0, Control 0


def stage_0(lab: Lab) -> None:
    lab.fp32()
    C, out = lab.pred["control_0"], {}
    out["real"] = E81.choices(lab, tofu.real())
    holds = abs(out["real"]["share"] - C["e81_share"]) <= C["tolerance"]
    lab.log(f"[0] control 0, W0's real-fact share: {json.dumps(out['real'])}; E81 {C['e81_share']} within {C['tolerance']}: "
            f"{'ok' if holds else 'MISSED'}")
    write(FILES["0"], out)
    if not holds:
        stop(lab, "control 0: W0's real-fact share", out["real"] | {"expected": C["e81_share"]})
    out["first_batch_loss"] = gold_loss(lab.model, tofu.stream(train_seqs(lab), E81.BATCH, E81.SEED, 1), E81.MICRO, DEV)
    out["W0_levels"] = {k: levels(v) for k, v in tokens(lab, "W0").items()}
    lab.log(f"[0] first batch loss {out['first_batch_loss']:.6f}; W0's levels {json.dumps(out['W0_levels'])}")
    write(FILES["0"], out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- R, C2 and the reference


def stage_R(lab: Lab) -> None:
    lab.fp32()
    w0, C2, C1 = tokens(lab, "W0"), lab.pred["C2_reread"], lab.pred["C1_content"]
    g = gains(state(lab, "e103_matched")["bare"], w0["bare"])
    rel = abs(g["all"] / C2["stored"] - 1)
    out: dict[str, Any] = {"C2": {"gain": g["all"], "stored": C2["stored"], "rel": rel, "holds": rel <= C2["rel_tol"]}}
    lab.log(f"[C2] E103's matched state on the bare reader: {json.dumps(g)}; stored {C2['stored']}, rel {rel:.3g} within "
            f"{C2['rel_tol']}: {'ok' if out['C2']['holds'] else 'MISSED'}")
    write(FILES["R"], out)
    if not out["C2"]["holds"]:
        stop(lab, "C2: E103's matched gain not reproduced", out["C2"])
    ref = gains(state(lab, "fict_matched")["fict"], w0["fict"])["content"]
    out["reference"] = {"content_gain": ref, "bar": C1["bar_frac"] * ref}
    lab.log(f"[R] fictional matched state's content gain on its own answers {ref:.4f}; C1's bar {out['reference']['bar']:.4f}")
    write(FILES["R"], out)
    if not (math.isfinite(ref) and ref > 0):
        stop(lab, "C1's reference has no gain: no bar", out["reference"])
    write(FILES["R"], out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- T, the fine-tune and C1


@dataclass(frozen=True)
class Arm:
    """Where one fine-tune of this file's recipe writes: its stream seed, its two states, its result file and its stop
    (Amendment 2 runs a second seed through `fine_tune`)."""

    seed: int
    matched: Path
    end: Path
    out: Path
    stop: Callable[[Any, str, dict[str, Any]], None]


def stage_T(lab: Lab) -> None:
    s0, bar = json.loads(FILES["0"].read_text()), json.loads(FILES["R"].read_text())["reference"]["bar"]
    fine_tune(lab, Arm(E81.SEED, MATCHED, END, FILES["T"], stop), s0["first_batch_loss"], bar, tokens(lab, "W0")["sentence"])


def fine_tune(lab: Any, arm: Arm, first_loss: float, bar: float, w0: list[list[float]]) -> None:
    """The fine-tune on the new pair 0 with arm's stream seed for five epochs; C1 at the matched step against `bar` on the
    sentence reader (`w0` W0's tokens there); both states saved in fp32; `arm.out` written as it goes (`KF.fine_tune`)."""
    seqs = train_seqs(lab)
    r = Recipe(E84.RATE, E81.WARMUP, E81.EPOCHS * (len(seqs) // E81.BATCH), "hf_cosine", 1.0, E81.MICRO, forward="fp32")
    keep = lambda path: lambda dW: save(path, dW, F32, droppable=False, log=lab.log)
    KF.fine_tune(lab, KF.Run(arm.seed, arm.out, arm.stop, keep(arm.matched), keep(arm.end)), seqs, r, E81.BATCH,
                 readers(lab)["sentence"], w0, first_loss, bar, E81.READ_BATCH)


# ----------------------------------------------------------------------------- G, the readings


def stage_G(lab: Lab) -> None:
    lab.fp32()
    s0, R = json.loads(FILES["0"].read_text()), lab.pred["R1_recall"]
    w0, out = tokens(lab, "W0"), {"states": {}, "share": {}}
    for name, path in STATES.items():
        new = name.startswith("a1_")
        if new or not (TOKENS / f"{name}.json").is_file():  # stage R read the others
            with set_weights(lab.model, plus(lab.W0, load_file(path, DEV))):
                tokens(lab, name)
                if new:
                    out["share"][name] = E81.choices(lab, tofu.real())
        out["states"][name] = {k: gains(v, w0[k]) for k, v in tokens(lab, name).items()}
        lab.log(f"[G] {name} gains {json.dumps(out['states'][name])} share {json.dumps(out['share'].get(name))}")
        write(FILES["G"], out)
    w = s0["real"]["share"]
    out["R1"] = {n: "kept" if v["share"] >= w - R["drop"] else "dropped" for n, v in out["share"].items()}
    lab.log(f"[G] R1, reported: {json.dumps(out['R1'])}")
    write(FILES["G"], out | {"done": True, "peaks": lab.peaks()})


STAGES = {"0": stage_0, "R": stage_R, "T": stage_T, "G": stage_G}


def filed() -> None:
    """Refuse unless predictions_amend1.json is committed unchanged, the sentences are done and E103's states exist."""
    rel, repo = str(PRED.relative_to(EXPERIMENTS.parent)), str(EXPERIMENTS.parent)
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=False)
    if git("diff", "--quiet", "HEAD", "--", rel).returncode or not git("ls-files", rel).stdout.strip():
        sys.exit(f"{rel} is not committed unchanged: commit it first")
    if not json.loads((RESULTS / "forms_amend1.json").read_text()).get("done"):
        sys.exit("results/forms_amend1.json is not done: run forms_amend1.py finish first")
    if not (RUN.MATCHED.is_file() and RUN.END.is_file()):
        sys.exit("E103's states are missing from results/large/dW")


def main() -> None:
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    filed()
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    todo = [s for s in STAGES if not is_done(s)]
    if not todo:
        print("nothing to do", flush=True)
        hard_exit(0)
    lab = Lab("A1", f"todo {' '.join(todo)}", PRED)
    for s in todo:
        STAGES[s](lab)
        lab.log(f"[{s}] done; {lab.peaks()}")
    lab.log("FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    main()

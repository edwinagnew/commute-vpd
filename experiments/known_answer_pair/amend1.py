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
amend1_neurons.py is the second job.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any

import torch

from commute import qwen_d3 as d3
from commute import qwen_tofu as tofu
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import gold_loss, linears, plus, save, set_weights
from commute.qwen_sft import load as load_file
from commute.qwen_train import Recipe, train
from commute.qwen_updates import ok

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
    recs = [json.loads(x) for x in SENTENCES.read_text().splitlines()]
    assert [r["row"] for r in recs] == list(range(len(tofu.real_rows())))
    return [r["sentence"] for r in recs]


def train_seqs(lab: Lab) -> list[tofu.Seq]:
    """run.py's `train_seqs` with pair 0's answer the row's sentence: same rows, order and count (1,085), so the stream's
    batches hold the same row and pair at every step."""
    real, seqs = tofu.real_rows(), []
    for rec, s in zip(map(json.loads, RUN.FORMS.read_text().splitlines()), sentences(), strict=True):
        assert rec["kept"]
        row = real[rec["row"]] | {"answer": s}
        seqs += [tofu.seq(lab.tok, lab.end_id, q, a) for q, a in tofu.form_pairs(row, rec["questions"], rec["answers"])]
    return seqs


def readers(lab: Lab) -> dict[str, list[tofu.Seq]]:
    real, kept = tofu.real_rows(), {r["row"] for r in map(json.loads, E84_FORMS.read_text().splitlines()) if r["kept"]}
    fict = [r for i, r in enumerate(tofu.held_rows()) if i in kept]
    return {"bare": RUN.answer_seqs(lab),
            "sentence": [tofu.seq(lab.tok, lab.end_id, r["question"] + tofu.ASK, s) for r, s in zip(real, sentences(), strict=True)],
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


def parts(rows: list[list[float]]) -> dict[str, list[float]]:
    """Per row: `all` the mean over response tokens (E103's reading), `content` the mean without the end token, `end` the
    end token's log P."""
    assert all(len(r) >= 2 for r in rows), "every answer has a content token before the end token"
    return {"all": [statistics.mean(r) for r in rows], "content": [statistics.mean(r[:-1]) for r in rows], "end": [r[-1] for r in rows]}


def gains(now: list[list[float]], w0: list[list[float]]) -> dict[str, float]:
    """Mean over rows of the change from W0 in each part (nats a token; `end` nats at the one token)."""
    a, b = parts(now), parts(w0)
    return {k: statistics.mean(x - y for x, y in zip(a[k], b[k], strict=True)) for k in a}


def levels(rows: list[list[float]]) -> dict[str, float]:
    return {k: statistics.mean(v) for k, v in parts(rows).items()}


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


def stage_T(lab: Lab) -> None:
    m, seqs, s0 = lab.fp32(), train_seqs(lab), json.loads(FILES["0"].read_text())
    torch.cuda.reset_peak_memory_stats()
    F, bar, reads = lab.pred["first_minutes"], json.loads(FILES["R"].read_text())["reference"]["bar"], readers(lab)["sentence"]
    w0 = tokens(lab, "W0")["sentence"]
    steps = E81.EPOCHS * (len(seqs) // E81.BATCH)
    r = Recipe(E84.RATE, E81.WARMUP, steps, "hf_cosine", 1.0, E81.MICRO, forward="fp32")
    out: dict[str, Any] = {"recipe": asdict(r), "batch": E81.BATCH, "sequences": len(seqs), "ref_norm": RUN.REF_NORM,
                           "first_batch_loss": s0["first_batch_loss"], "steps": [], "checks": [], "gains": {}}
    best: dict[str, Any] = {"gap": math.inf}
    lab.log(f"[T] {len(seqs)} sequences, {steps} steps; reference norm {RUN.REF_NORM:.6f}; C1's bar {bar:.4f}")

    def matched() -> None:
        save(MATCHED, best.pop("dW"), F32, droppable=False, log=lab.log)
        rec = {k: best[k] for k in ("step", "norm", "gain")} | {"bar": bar, "holds": best["gain"]["content"] >= bar}
        out["matched"] = rec
        lab.log(f"[C1] content gain at the matched step {rec['step']} (norm {rec['norm']:.4f}): {json.dumps(rec['gain'])}; "
                f"bar {bar:.4f}: {'ok' if rec['holds'] else 'MISSED'}")
        write(FILES["T"], out)
        if not rec["holds"]:
            stop(lab, "C1: no content gain to carry", rec)

    def on_step(rec: dict[str, Any]) -> None:
        t = rec["step"]
        for key in ("norms", "crossed", "kl"):
            rec.pop(key, None)
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out["steps"].append(rec)
        if not (ok(rec["loss"]) and ok(rec["grad_norm"])):
            write(FILES["T"], out)
            stop(lab, f"step {t} not finite", rec)
        for name, value, holds in tofu.step_checks(rec, s0["first_batch_loss"], F, E81.WARMUP):
            row = {"at": t, "name": name, "value": value, "holds": holds}
            out["checks"].append(row)
            lab.log(f"[check] {'ok' if holds else 'MISSED'} {json.dumps(row, default=float)}")
            if not holds:
                write(FILES["T"], out)
                stop(lab, f"step {t} {name}", row)
        n = rec["dw_norm"]
        if "matched" not in out and n > 0:
            if RUN.log_gap(n) < best["gap"]:  # a better candidate: its gain now, its δW held on the card
                g = gains(token_logp(lab, reads), w0)
                out["gains"][str(t)] = g
                best.pop("dW", None)  # the old copy goes first: two on the card would be 11 GB
                best.update(gap=RUN.log_gap(n), step=t, norm=n, gain=g,
                            dW={s: lin.weight.data - lab.W0[s] for s, lin in linears(m).items()})
                lab.log(f"[T] candidate step {t}: norm {n:.4f}, gain {json.dumps(g)}")
            elif n > RUN.REF_NORM:  # past the reference and moving away: the best is the matched state
                matched()
        if t % 25 == 0 or t <= 3:
            lab.log(f"[T] {json.dumps(rec)}")
            write(FILES["T"], out)

    def save_end(t: int, dW: dict[str, Any]) -> None:
        save(END, dW, F32, droppable=False, log=lab.log)
        out["end_step"] = t

    t0 = time.time()
    train(m, lab.W0, lambda t: tofu.stream(seqs, E81.BATCH, E81.SEED, t), r, {}, {steps}, {}, {steps}, save_end, on_step,
          LARGE / "ckpt_unused.pt", steps, dW_of=delta_fp32)
    if "matched" not in out:
        matched()
    norms = {rec["step"]: rec["dw_norm"] for rec in out["steps"] if rec["dw_norm"] > 0}
    out["argmin_step"] = min(norms, key=lambda t: RUN.log_gap(norms[t]))
    lab.log(f"[T] matched step {out['matched']['step']}, argmin over the record {out['argmin_step']}; end |dW| "
            f"{out['steps'][-1]['dw_norm']:.4f}")
    write(FILES["T"], out | {"done": True, "seconds": time.time() - t0, "peaks": lab.peaks()})
    torch.cuda.empty_cache()


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

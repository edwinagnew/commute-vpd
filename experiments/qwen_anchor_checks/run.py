"""E88 (qwen_anchor_checks), 2026-11-05: make the two anchor updates defensible. Plan:
reports/plans/2026-11-05_plan_e88-anchor-checks.md. Rules: results/predictions.json (the run refuses to start without it).
Data and graders: `commute.qwen_tofu`. Generation, the forms and E84's recipe: `qwen_tofu_forms/run.py`, `qwen_tofu/run.py`.

  R  one card (W0, then E84's step 1250): R0 alone first, a miss stops the job; then the same-card reads, the bare reads
     and the known-fact read (about 15 min on an L40). Needs nothing.
  S  a second seed of E84's update, then its reads on the same card (about 31 + 5 min on an L40). Needs R: its spread is
     against R's numbers.
  A  a second seed of E78's template fine-tune, then its reads (about 27 + 4 min on an L40). Needs nothing.
One stage per job; S and A may run together. A finished stage is skipped. S and A write their training's file when training
ends, so a job lost in the reads does not train again; a job lost in training starts again from step 0. Exit 0 done; 1 a
stop (results/stopped_<stage>.json, the stage refuses until it is removed) or a refusal. Each stage logs to results/run_<stage>.log.
Step t's δW has t - 1 moving steps: step 1 runs at rate 0, as E78's and E84's.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import gold_loss, plus, save, set_weights
from commute.qwen_sft import load as load_file
from commute.qwen_train import Recipe, train
from commute.qwen_updates import ok

E84: Any = load_experiment("qwen_tofu_forms", "run")  # answer_set, graded, train_seqs, the forms, step 1250's δW
E81, E78 = E84.E81, E84.E78  # generated and the recipe's numbers; header, moments, file_row, read_step_100, rule
JOB, E45_RUN = E81.JOB, E78.E45_RUN  # the job's data and model; the template's held-out reader
DEV, F32, CFG, write = E81.DEV, E81.F32, JOB.CFG, E81.write

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
ANSWERS = LARGE / "answers"
FILES = {"R": RESULTS / "R.json", "S": RESULTS / "S.json", "A": RESULTS / "A.json"}
TRAINED = {"S": RESULTS / "S_train.json", "A": RESULTS / "A_train.json"}
DW = {"S": LARGE / "dW" / "tofu_seed142", "A": LARGE / "dW" / "a1_seed1"}
LOG_PARTS = 25  # lines of the log over a training, so a stall shows within minutes


class Lab(JOB.Job):  # E45's model and tokenizer; the log is this experiment's and this stage's, as S and A run together
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / f"run_{self.stage}.log").open("a") as f:
            f.write(line + "\n")

    @property
    def card(self) -> str:
        return torch.cuda.get_device_name()


def stopped_file(stage: str) -> Path:
    return RESULTS / f"stopped_{stage}.json"


def stop(lab: Lab, name: str, detail: dict[str, Any]) -> None:
    write(stopped_file(lab.stage), {"stopped": name, "card": lab.card, **detail})
    lab.log(f"STOP: {name} {json.dumps(detail, default=float)}")
    sys.exit(1)


def is_done(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text()).get("done"))


# ----------------------------------------------------------------------------- the filed rules, no model


def card_holds(state: dict[str, Any], h100: dict[str, float], band: float) -> bool:
    """R1: each of rephrased, trained and floor within the band of its H100 reading. Exactly at the band holds."""
    return all(abs(state[k]["recall"] - h100[k]) <= band for k in h100)


def suffix_tied(bare_para: float, bare_floor: float, margin: float) -> bool:
    """R2 at S: bare rephrased under bare floor + margin. Exactly at it is not tied."""
    return bare_para < bare_floor + margin


def known_cost(w0: float, s: float, drop: float) -> bool:
    """R3: S more than `drop` under W0. Exactly at it is no cost."""
    return w0 - s > drop


def lost_gained(w0: list[bool], s: list[bool]) -> tuple[int, int]:
    """R3: rows holding the fact at W0 and not at S; rows holding it at S and not at W0."""
    pairs = list(zip(w0, s, strict=True))
    return sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs)


def shared_epochs(seed: int, other: int, epochs: int) -> set[int]:
    """`tofu.stream` shuffles epoch e by Random(seed + e), so two seeds under `epochs` apart reuse each other's orders."""
    return set(range(seed, seed + epochs)) & set(range(other, other + epochs))


def within_rel(x: float, expected: float, rel: float) -> bool:
    return ok(x) and abs(x - expected) <= rel * expected


def spread(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: abs(a[k] - b[k]) for k in a if ok(a[k])}  # a reading with no value (strict_dropped None) has no spread


def to_line(state: dict[str, float]) -> float:
    """Rephrased recall less E84's line, halfway from floor to trained: the gap the label turns on."""
    return state["para"] - (state["floor"] + 0.5 * (state["trained"] - state["floor"]))


# ----------------------------------------------------------------------------- reading saved answers


def kept_rows() -> set[int]:
    return {rec["row"] for rec in map(json.loads, E84.FORMS.read_text().splitlines()) if rec["kept"]}


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def mean_words(path: Path) -> float:
    return statistics.mean(len(r["response"].split()) for r in lines(path))


def holds_fact(path: Path) -> list[bool]:
    """E81's contains rule (`qwen_tofu/check_after.py`)."""
    return [r["reference"].lower() in r["response"].lower() for r in lines(path)]


def recalls(graded: dict[str, Any]) -> dict[str, float]:
    return {k: graded[k]["recall"] for k in ("para", "trained", "floor")}


# ----------------------------------------------------------------------------- R, one card


def answers(lab: Lab, sink: Path, suffixed: Iterable[str]) -> None:
    """On the model as it stands: the suffixed sets asked for, the bare rephrased and never-trained sets, the real facts."""
    for name in suffixed:
        E84.answer_set(lab, name, sink)
    held, never = tofu.held_rows(), tofu.rows("holdout10")
    E81.generated(lab, [r["paraphrased_question"] for r in held], [r["answer"] for r in held], sink / "bare_paraphrased.jsonl", ask="")
    E81.generated(lab, [r["question"] for r in never], [r["answer"] for r in never], sink / "bare_never.jsonl", ask="")
    real_answers(lab, sink)


def real_answers(lab: Lab, sink: Path) -> None:
    """The real-fact answers, `real.jsonl` under `sink`; E88 Amendment 1 calls it at W0 and at a second seed."""
    real = [r for f in tofu.REAL_FILES for r in tofu.rows(f)]
    E81.generated(lab, [r["question"] for r in real], [r["answer"] for r in real], sink / "real.jsonl", ask=tofu.ASK_REAL)


def reading(sink: Path, keep: set[int]) -> dict[str, Any]:
    """Every number of one state, from its saved answers. Words are per answer over each file's whole set."""
    hits = holds_fact(sink / "real.jsonl")
    return E84.graded(sink, keep) | {
        "bare_para": tofu.new_word_recall(sink / "bare_paraphrased.jsonl", keep),
        "bare_floor": tofu.new_word_recall(sink / "bare_never.jsonl"),
        "words": {n: mean_words(sink / f"{n}.jsonl") for n in ("paraphrased", "never", "bare_paraphrased", "bare_never", "real")},
        "real": {"contains": statistics.mean(hits), "n": len(hits)}}


def stage_R(lab: Lab) -> None:
    lab.fp32()
    P, keep, w0, s = lab.pred["R"], kept_rows(), ANSWERS / "W0", ANSWERS / "S"
    out: dict[str, Any] = {"card": lab.card}
    C = P["R0"]
    E84.answer_set(lab, "paraphrased", w0)
    out["R0"] = tofu.new_word_recall(w0 / "paraphrased.jsonl", keep)
    holds = abs(out["R0"]["recall"] - C["expected"]) <= C["tolerance"] and out["R0"]["n"] == C["n"]
    lab.log(f"[R0 {lab.card}] control, new-word recall, rephrased, suffixed: {out['R0']['recall']:.6f} on {out['R0']['n']} rows; "
            f"filed {C['expected']} +- {C['tolerance']} on {C['n']}: {'ok' if holds else 'MISSED'}")
    write(FILES["R"], out)
    if not holds:
        stop(lab, "R0: W0's new-word recall", {**out["R0"], "expected": C["expected"]})
    answers(lab, w0, ("trained", "never"))
    out["W0"] = reading(w0, keep)
    lab.log(f"[R W0 {lab.card}] {json.dumps(out['W0'])}")
    write(FILES["R"], out)
    with set_weights(lab.model, plus(lab.W0, load_file(E84.DW / "step_1250.safetensors", DEV))):
        answers(lab, s, E84.SETS)
    out["S"] = reading(s, keep)
    lab.log(f"[R S {lab.card}] {json.dumps(out['S'])}")
    out |= R_verdict(lab, out, P, w0, s)
    write(FILES["R"], out | {"done": True, "peaks": lab.peaks()})


def R_verdict(lab: Lab, out: dict[str, Any], P: dict[str, Any], w0: Path, s: Path) -> dict[str, Any]:
    """R1, R2 and R3 from the readings, each logged with the card it was read on."""
    R1, R2, R3, W, S = P["R1"], P["R2"], P["R3"], out["W0"], out["S"]
    same = card_holds(S, R1["h100"], R1["band"])
    gaps = {k: S[k]["recall"] - R1["h100"][k] for k in R1["h100"]}
    lab.log(f"[R1 W0 {lab.card}] rephrased {out['R0']['recall']:.6f}, trained {W['trained']['recall']:.6f}, floor {W['floor']['recall']:.6f}; "
            f"E84's W0 (L40) {json.dumps(R1['w0_l40'])}")
    lab.log(f"[R1 S {lab.card}] {json.dumps(recalls(S))}; H100 {json.dumps(R1['h100'])}; {lab.card} less H100 {json.dumps(gaps)}: "
            f"{R1['same'] if same else R1['moved']}")
    tied = suffix_tied(S["bare_para"]["recall"], S["bare_floor"]["recall"], R2["margin"])
    lab.log(f"[R2 {lab.card}] bare rephrased / bare floor, W0 {W['bare_para']['recall']:.6f} / {W['bare_floor']['recall']:.6f}, S "
            f"{S['bare_para']['recall']:.6f} / {S['bare_floor']['recall']:.6f}; words W0 {W['words']['bare_paraphrased']:.1f}, "
            f"S {S['words']['bare_paraphrased']:.1f}; tied to the suffix: {tied}")
    c0, c1 = W["real"]["contains"], S["real"]["contains"]
    cost = known_cost(c0, c1, R3["drop"])
    lost, gained = lost_gained(holds_fact(w0 / "real.jsonl"), holds_fact(s / "real.jsonl"))
    reproduced = abs(c0 - R3["w0"]) <= R3["tolerance"] and W["real"]["n"] == R3["n"]
    lab.log(f"[R3 {lab.card}] answer holds the fact, W0 {c0:.6f} ({W['words']['real']:.2f} words; filed {R3['w0']} +- {R3['tolerance']}: "
            f"{'ok' if reproduced else 'MISSED, no stop'}), S {c1:.6f} ({S['words']['real']:.2f} words); lost {lost}, gained {gained}; "
            f"at a cost to known facts: {cost}")
    adds = ([R2["adds"]] if tied else []) + ([R3["adds"]] if cost else [])
    lab.log(f"[R {lab.card}] the label adds: {json.dumps(adds)}")
    return {"R1": {"same_card": same, "reading": R1["same"] if same else R1["moved"], "s_less_h100": gaps},
            "R2": {"tied_to_the_suffix": tied},
            "R3": {"w0_reproduced": reproduced, "cost": cost, "lost": lost, "gained": gained},
            "label_adds": adds}


# ----------------------------------------------------------------------------- training, S's and A's


def run_train(lab: Lab, out: dict[str, Any], batch_at: Callable[[int], Any], r: Recipe, keep: set[int],
              on_rec: Callable[[dict[str, Any]], None]) -> None:
    """E78's and E84's loop: δW in fp32 at `keep`, no tracking, a non-finite step stops; `on_rec` holds the stage's checks."""
    path, dw = TRAINED[lab.stage], DW[lab.stage]

    def save_dw(t: int, dW: dict[str, Any]) -> None:
        save(dw / f"step_{t}.safetensors", dW, F32, droppable=False, log=lab.log)
        out["saves"].append(t)

    def on_step(rec: dict[str, Any]) -> None:
        t = rec["step"]
        for key in ("norms", "crossed", "kl"):
            rec.pop(key, None)
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out["steps"].append(rec)
        if not (ok(rec["loss"]) and ok(rec["grad_norm"])):
            write(path, out)
            stop(lab, f"step {t} not finite", rec)
        on_rec(rec)
        if t % (r.steps // LOG_PARTS) == 0 or t <= 3:
            lab.log(f"[{lab.stage} {lab.card}] {json.dumps(rec)}")
            write(path, out)

    t0 = time.time()
    train(lab.model, lab.W0, batch_at, r, {}, keep, {}, keep, save_dw, on_step, LARGE / f"ckpt_unused_{lab.stage}.pt", r.steps,
          dW_of=delta_fp32)
    out |= {"done": True, "seconds": time.time() - t0, "peaks": lab.peaks()}
    lab.log(f"[{lab.stage} {lab.card}] training took {out['seconds']:.0f} s; {out['peaks']}")
    write(path, out)
    torch.cuda.empty_cache()


def checked(lab: Lab, out: dict[str, Any], at: int, name: str, value: Any, holds: bool) -> None:
    row = {"at": at, "name": name, "value": value, "holds": holds}
    out["checks"].append(row)
    lab.log(f"[check {lab.card}] {'ok' if holds else 'MISSED'} {json.dumps(row, default=float)}")
    if not holds:
        write(TRAINED[lab.stage], out)
        stop(lab, f"step {at} {name}", row)


def check_files(lab: Lab, steps: list[dict[str, Any]], saves: Iterable[int], F: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for t in saves:
        p = DW[lab.stage] / f"step_{t}.safetensors"
        dtypes = E78.header(p)
        norm = math.sqrt(E78.moments(p, p)[0]) if dtypes else None
        rows[str(t)] = E78.file_row(dtypes, norm, steps[t - 1]["dw_norm"], F)
        lab.log(f"[files] step {t} {json.dumps(rows[str(t)])}")
    if not all(r["holds"] for r in rows.values()):
        stop(lab, "a saved file misses", rows)
    return rows


# ----------------------------------------------------------------------------- S, E84's update at a second seed


def train_S(lab: Lab) -> None:
    P, e84 = lab.pred["S"], json.loads(E84.FILES["T"].read_text())
    m, seqs, seed = lab.fp32(), E84.train_seqs(lab), P["seed"]
    torch.cuda.reset_peak_memory_stats()  # the band is training's
    r = Recipe(E84.RATE, E81.WARMUP, E81.STEPS, "hf_cosine", 1.0, E81.MICRO, forward="fp32")
    epochs = (E81.STEPS - 1) // (len(seqs) // E81.BATCH) + 1
    first = gold_loss(m, tofu.stream(seqs, E81.BATCH, seed, 1), E81.MICRO, DEV)
    out: dict[str, Any] = {"card": lab.card, "seed": seed, "recipe": asdict(r), "batch": E81.BATCH, "sequences": len(seqs),
                           "first_batch_loss": first, "steps": [], "saves": [], "checks": [], "seconds": 0.0}
    lab.log(f"[S {lab.card}] {len(seqs)} sequences; first batch loss {first:.6f}; seed 0's (H100) {e84['first_batch_loss']:.6f}")
    if asdict(r) != e84["recipe"] or len(seqs) != e84["sequences"] or shared_epochs(seed, E81.SEED, epochs):
        stop(lab, "not E84's recipe and forms, or a seed sharing epochs", {"recipe": asdict(r), "sequences": len(seqs), "epochs": epochs})
    checked(lab, out, 0, "first batch differs from seed 0's", abs(first - e84["first_batch_loss"]), abs(first - e84["first_batch_loss"]) > P["first_batch_differs"])

    def on_rec(rec: dict[str, Any]) -> None:
        for name, value, holds in tofu.step_checks(rec, first, P["first_minutes"], E81.WARMUP):
            checked(lab, out, rec["step"], name, value, holds)

    run_train(lab, out, lambda t: tofu.stream(seqs, E81.BATCH, seed, t), r, set(P["saves"]), on_rec)


def stage_S(lab: Lab) -> None:
    if not is_done(FILES["R"]):
        sys.exit("stage R must finish first: the control, and the numbers S's spread is against")
    if not is_done(TRAINED["S"]):
        train_S(lab)
    lab.fp32()
    P, run, keep, sink = lab.pred["S"], json.loads(TRAINED["S"].read_text()), kept_rows(), ANSWERS / "S_seed"
    out: dict[str, Any] = {"card": lab.card, "seed": P["seed"], "files": check_files(lab, run["steps"], P["saves"], P["files"])}
    with set_weights(lab.model, plus(lab.W0, load_file(DW["S"] / f"step_{E81.STEPS}.safetensors", DEV))):
        for name in E84.SETS:
            E84.answer_set(lab, name, sink)
    out["state"] = E84.graded(sink, keep)
    here, there = recalls(out["state"]), recalls(json.loads(FILES["R"].read_text())["S"])
    dw = run["steps"][-1]["dw_norm"]
    out |= {"dw_norm": dw, "dw_holds": within_rel(dw, P["dw_norm"], P["dw_rel"]), "seed_0": there, "spread": spread(here, there),
            "dw_spread": abs(dw - P["dw_norm_seed_0"]), "to_line": to_line(here), "to_line_seed_0": to_line(there),
            "para_holds": abs(here["para"] - there["para"]) <= P["para_band"],
            "label": tofu.transfer_label(here["para"], here["trained"], here["floor"])}
    lab.log(f"[S {lab.card}] {json.dumps(here)}; seed 0, both read on {json.loads(FILES['R'].read_text())['card']} (trained on an H100): {json.dumps(there)}")
    lab.log(f"[S {lab.card}] spread {json.dumps(out['spread'])}, |dW| {dw:.3f} (filed {P['dw_norm']} within {P['dw_rel']}: {out['dw_holds']}); "
            f"rephrased within {P['para_band']} of R1's: {out['para_holds']}")
    lab.log(f"[S {lab.card}] FILED RULE: {out['label']}; rephrased less the line {out['to_line']:.4f}, seed 0's {out['to_line_seed_0']:.4f}")
    write(FILES["S"], out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- A, E78's template fine-tune at a second seed


def train_A(lab: Lab) -> None:
    P, e78 = lab.pred["A"], json.loads(E78.TRAIN.read_text())
    lab.fp32()
    stream, per, _, data = lab.arm_data("a1", P["seed"])
    r = Recipe(E78.PEAK, CFG.warmup, E78.STEPS, CFG.schedule, CFG.clip, E78.MICRO, forward="fp32")
    if data != P["data"] or asdict(r) != e78["recipe"]:
        stop(lab, "not the filed rows or E78's recipe", {"data": data, "recipe": asdict(r), "e78": e78["recipe"]})
    task, one = E45_RUN.task_reader(lab, "a1", P["seed"]), e78["steps"][0]
    out: dict[str, Any] = {"card": lab.card, "seed": P["seed"], "data": data, "recipe": asdict(r), "steps": [], "saves": [], "checks": [], "seconds": 0.0}

    def on_rec(rec: dict[str, Any]) -> None:
        t = rec["step"]
        if t in P["held_at"]:
            rec["held"] = task(None)["distill_kl"]
            lab.log(f"[A {lab.card}] step {t} held-out distillation KL {rec['held']:.6f}; E78's (L40) {e78['steps'][t - 1].get('held')}")
        if t == 1:  # the control: rate 0, so this is W0 on E78's held rows, and only the training batch has changed
            checked(lab, out, t, "W0's held-out KL is E78's", rec["held"], within_rel(rec["held"], one["held"], P["w0_held_rel"]))
            checked(lab, out, t, "rate", rec["lr"], rec["lr"] == 0)
            checked(lab, out, t, "|dW|", rec["dw_norm"], rec["dw_norm"] == 0)
            checked(lab, out, t, "first batch differs from seed 0's", abs(rec["loss"] - one["loss"]), abs(rec["loss"] - one["loss"]) > P["first_batch_differs"])
        if t == E78.STEPS:
            checked(lab, out, t, "loss", rec["loss"], rec["loss"] < P["loss_100_under"])

    run_train(lab, out, lambda t: stream[(t - 1) * per : t * per], r, set(P["saves"]), on_rec)


def stage_A(lab: Lab) -> None:
    if not is_done(TRAINED["A"]):
        train_A(lab)
    P, run = lab.pred["A"], json.loads(TRAINED["A"].read_text())
    out: dict[str, Any] = {"card": lab.card, "seed": P["seed"], "files": check_files(lab, run["steps"], P["saves"], P["files"])}
    write(FILES["A"], out)
    E78.DW, E78.LARGE = DW["A"], LARGE  # the two names E78's reader takes its δW and its answers' folder from
    st = E78.read_step_100(lab)
    last, ref = run["steps"][-1], P["e78"]
    here = {"format_unprompted": st["format_unprompted"], "strict_dropped": st["strict_dropped"], "held": st["held_from_file"], "dw_norm": last["dw_norm"]}
    out |= {"step_100": st, "loss_100": last["loss"], "e78": ref, "spread": spread(here, ref), "dw_holds": within_rel(last["dw_norm"], ref["dw_norm"], P["dw_rel"]),
            "rule": E78.rule(st["format_unprompted"], st["strict_dropped"], P["rule"])}
    lab.log(f"[A {lab.card}] step 100 {json.dumps(here)}; E78's (L40) {json.dumps(ref)}; spread {json.dumps(out['spread'])}; "
            f"|dW| within {P['dw_rel']} of E78's: {out['dw_holds']}")
    lab.log(f"[A {lab.card}] FILED RULE, E78's format rule at seed {P['seed']}: {json.dumps(out['rule'])}")
    write(FILES["A"], out | {"done": True, "peaks": lab.peaks()})


STAGES: dict[str, Callable[[Lab], None]] = {"R": stage_R, "S": stage_S, "A": stage_A}


def main() -> None:
    p = argparse.ArgumentParser(description="E88: stages in the module docstring")
    p.add_argument("--stage", required=True, choices=tuple(STAGES))
    a = p.parse_args()
    if stopped_file(a.stage).is_file():
        sys.exit(f"{stopped_file(a.stage)} exists: read it, then remove it")
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    if is_done(FILES[a.stage]):
        print(f"nothing to do: {a.stage} is done", flush=True)
        hard_exit(0)
    lab = Lab(a.stage, "E88", PRED)  # refuses to start without the predictions file
    STAGES[a.stage](lab)
    lab.log(f"[{a.stage}] done; {lab.peaks()}")
    hard_exit(0)


if __name__ == "__main__":
    main()

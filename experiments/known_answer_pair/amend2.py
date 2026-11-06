"""E103 Amendment 2 (known_answer_pair), 2026-11-06: a second stream seed (142) of Amendment 1's known-facts fine-tune,
recipe otherwise identical (the same results/forms_amend1.jsonl, rows, E84's recipe and E91's matched-size rule), through
amend1.py's `fine_tune`; then, as an add-on, per-option log P and greedy answers of the real-fact share at W0 and at
Amendment 1's end state. Plan: reports/plans/2026-11-06_plan_e103-known-answer-pair.md, "Amendment 2". Rules:
results/predictions_amend2.json (committed before the run; the run refuses otherwise). amend2_neurons.py is the second job.

  0  Control 0 first and alone, as amend1.py; then seed 142's first batch loss, which must differ from seed 42's, and W0's
     tokens on amend1.py's three readers, read again here (2 min on an L40).
  R  C2 first: Amendment 1's matched state re-read on the sentence reader reproduces its stored content gain, or the job
     stops. Then the fictional matched state's content gain on its own answers, C1's reference, read again (2 min).
  T  335 steps at seed 142; C1 at the matched step against half the reference, a miss stops the job; then to the end.
     States results/large/dW/amend2_s142_{matched,end}.safetensors (about 7 min).
  P  the add-on, reported: W0 and Amendment 1's end state, each option's per-token log P on the 217 real-fact choices
     (the share recomputed from them must reproduce the stored share, or the job stops) and greedy answers under the
     share's suffix and under the trained one (3 min).
A stage whose result file says done is skipped; a token file under results/large/amend2_tokens/ is read back rather than
re-read, so after a stop remove the token files the stopped stage wrote. Exit 0 done; 1 a stop
(results/stopped_amend2.json) or a refusal.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from typing import Any

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_sft import gold_loss, plus, set_weights
from commute.qwen_sft import load as load_file

A1: Any = load_experiment("known_answer_pair", "amend1")  # its readers, gains, train_seqs, fine_tune, states
RUN, E81 = A1.RUN, A1.E81
DEV, write, RESULTS, LARGE = A1.DEV, A1.write, A1.RESULTS, A1.LARGE
SEED = 142  # the fictional arm's second seed (E88); tofu.stream shuffles epoch e by Random(seed + e)
PRED, STOPPED = RESULTS / "predictions_amend2.json", RESULTS / "stopped_amend2.json"
FILES = {s: RESULTS / f"amend2_{n}.json" for s, n in (("0", "stage_0"), ("R", "ref"), ("T", "train"), ("P", "options"))}
TOKENS, ANSWERS = LARGE / "amend2_tokens", RESULTS / "amend2_answers"
MATCHED, END = RUN.DW / "amend2_s142_matched.safetensors", RUN.DW / "amend2_s142_end.safetensors"


class Lab(RUN.Lab):
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "amend2.log").open("a") as f:
            f.write(line + "\n")


def stop(lab: Lab, name: str, detail: dict[str, Any]) -> None:
    write(STOPPED, {"stopped": name, **detail})
    lab.log(f"STOP: {name} {json.dumps(detail, default=float)}")
    hard_exit(1)


def is_done(stage: str) -> bool:
    return FILES[stage].is_file() and bool(json.loads(FILES[stage].read_text()).get("done"))


def tokens(lab: Lab, name: str) -> dict[str, list[list[float]]]:
    """amend1.py's `tokens`, saved under this amendment's folder, so W0 and the re-read states are read in this job."""
    path = TOKENS / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text())
    out = {k: A1.token_logp(lab, seqs) for k, seqs in A1.readers(lab).items()}
    TOKENS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    return out


def state(lab: Lab, name: str) -> dict[str, list[list[float]]]:
    with set_weights(lab.model, plus(lab.W0, load_file(A1.STATES[name], DEV))):
        return tokens(lab, name)


# ----------------------------------------------------------------------------- 0, Control 0


def stage_0(lab: Lab) -> None:
    lab.fp32()
    C, F, out = lab.pred["control_0"], lab.pred["first_minutes"], {}
    out["real"] = E81.choices(lab, tofu.real())
    holds = abs(out["real"]["share"] - C["e81_share"]) <= C["tolerance"]
    lab.log(f"[0] control 0, W0's real-fact share: {json.dumps(out['real'])}; E81 {C['e81_share']} within {C['tolerance']}: "
            f"{'ok' if holds else 'MISSED'}")
    write(FILES["0"], out)
    if not holds:
        stop(lab, "control 0: W0's real-fact share", out["real"] | {"expected": C["e81_share"]})
    first = gold_loss(lab.model, tofu.stream(A1.train_seqs(lab), E81.BATCH, SEED, 1), E81.MICRO, DEV)
    gap = abs(first - F["seed_42_first_batch_loss"])
    out |= {"first_batch_loss": first, "first_batch_gap_to_seed_42": gap}
    lab.log(f"[0] seed {SEED} first batch loss {first:.6f}; seed 42's {F['seed_42_first_batch_loss']:.6f}, gap {gap:.6f} "
            f"over {F['first_batch_differs']}: {'ok' if gap > F['first_batch_differs'] else 'MISSED'}")
    if not gap > F["first_batch_differs"]:
        write(FILES["0"], out)
        stop(lab, "seed 142's first batch is seed 42's", out)
    out["W0_levels"] = {k: A1.levels(v) for k, v in tokens(lab, "W0").items()}
    lab.log(f"[0] W0's levels {json.dumps(out['W0_levels'])}")
    write(FILES["0"], out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- R, C2 and the reference


def stage_R(lab: Lab) -> None:
    lab.fp32()
    w0, C2, C1 = tokens(lab, "W0"), lab.pred["C2_reread"], lab.pred["C1_content"]
    g = A1.gains(state(lab, "a1_matched")["sentence"], w0["sentence"])
    rel = abs(g["content"] / C2["stored"] - 1)
    out: dict[str, Any] = {"C2": {"gain": g, "stored": C2["stored"], "rel": rel, "holds": rel <= C2["rel_tol"]}}
    lab.log(f"[C2] Amendment 1's matched state on the sentence reader: {json.dumps(g)}; stored content {C2['stored']}, "
            f"rel {rel:.3g} within {C2['rel_tol']}: {'ok' if out['C2']['holds'] else 'MISSED'}")
    write(FILES["R"], out)
    if not out["C2"]["holds"]:
        stop(lab, "C2: Amendment 1's matched content gain not reproduced", out["C2"])
    ref = A1.gains(state(lab, "fict_matched")["fict"], w0["fict"])["content"]
    out["reference"] = {"content_gain": ref, "bar": C1["bar_frac"] * ref, "amend1": C1["amend1_reference"],
                        "rel_to_amend1": abs(ref / C1["amend1_reference"] - 1)}
    lab.log(f"[R] reference {json.dumps(out['reference'])}")
    write(FILES["R"], out)
    if not (math.isfinite(ref) and ref > 0):
        stop(lab, "C1's reference has no gain: no bar", out["reference"])
    write(FILES["R"], out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- T, the second seed and C1


def stage_T(lab: Lab) -> None:
    s0, bar = json.loads(FILES["0"].read_text()), json.loads(FILES["R"].read_text())["reference"]["bar"]
    A1.fine_tune(lab, A1.Arm(SEED, MATCHED, END, FILES["T"], stop), s0["first_batch_loss"], bar, tokens(lab, "W0")["sentence"])


# ----------------------------------------------------------------------------- P, the real-fact share's options (add-on)


def options(lab: Lab) -> list[dict[str, Any]]:
    """Per real-fact choice, each option's per-token log P (`choice_read`'s sequences, right option first) and the
    pick by mean log P, the end token included, as `choice_read` picks."""
    items, seqs = tofu.real(), []
    for c in items:
        seqs += [tofu.seq(lab.tok, lab.end_id, c.question, a) for a in (c.right, *c.wrong)]
    toks, out, at = A1.token_logp(lab, seqs), [], 0
    for i, c in enumerate(items):
        rows, at = toks[at : at + 1 + len(c.wrong)], at + 1 + len(c.wrong)
        mean, content = [statistics.mean(r) for r in rows], [statistics.mean(r[:-1]) for r in rows]
        out.append({"row": i, "options": [c.right, *c.wrong], "tokens": rows, "mean": mean, "hit": tofu.correct(mean),
                    "content_hit": tofu.correct(content)})
    return out


def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean over rows: the share, the share on content tokens only, and the right and wrong options' all-token mean,
    content mean and end-token log P (the wrong ones averaged over a row's three)."""
    right, wrong = [it["tokens"][0] for it in items], [r for it in items for r in it["tokens"][1:]]
    part = lambda rows: {"all": statistics.mean(statistics.mean(r) for r in rows), "content": statistics.mean(statistics.mean(r[:-1]) for r in rows),
                         "end": statistics.mean(r[-1] for r in rows), "tokens": statistics.mean(len(r) for r in rows)}
    return {"share": statistics.mean(it["hit"] for it in items), "content_share": statistics.mean(it["content_hit"] for it in items),
            "right": part(right), "wrong": part(wrong)}


def greedy(lab: Lab, name: str) -> dict[str, Any]:
    real, out = tofu.real_rows(), {}
    for suffix, ask in (("few_words", tofu.ASK_REAL), ("sentence", tofu.ASK)):
        sink = ANSWERS / f"{name}_{suffix}.jsonl"
        recall = E81.generated(lab, [r["question"] for r in real], [r["answer"] for r in real], sink, ask=ask)
        lines = [json.loads(x) for x in sink.read_text().splitlines()]
        out[suffix] = {"recall": recall, "contains": statistics.mean(x["reference"].lower() in x["response"].lower() for x in lines),
                       "words": statistics.mean(len(x["response"].split()) for x in lines)}
    return out


def stage_P(lab: Lab) -> None:
    lab.fp32().eval()  # after stage T in the same job the model is in training mode, where generation keeps no cache
    P, out = lab.pred["P_options"], {"states": {}}
    for name in ("W0", "a1_end"):
        with set_weights(lab.model, plus(lab.W0, load_file(A1.END, DEV)) if name == "a1_end" else None):
            items = options(lab)
            s = summary(items)
            out["states"][name] = {"summary": s, "items": items}
            gap = abs(s["share"] - P["stored_share"][name])
            lab.log(f"[P] {name} {json.dumps(s)}; stored share {P['stored_share'][name]}, gap {gap:.4g} within {P['share_tol']}")
            if not gap <= P["share_tol"]:
                write(FILES["P"], out)
                stop(lab, f"P: {name}'s share not reproduced from its options", {"share": s["share"], "stored": P["stored_share"][name]})
            out["states"][name]["greedy"] = greedy(lab, name)
            lab.log(f"[P] {name} greedy {json.dumps(out['states'][name]['greedy'])}")
        write(FILES["P"], out)
    w0, end = out["states"]["W0"]["items"], out["states"]["a1_end"]["items"]
    out["flips"] = {"lost": sum(a["hit"] and not b["hit"] for a, b in zip(w0, end, strict=True)),
                    "gained": sum(b["hit"] and not a["hit"] for a, b in zip(w0, end, strict=True))}
    lab.log(f"[P] rows W0 picked right and a1_end wrong, and back: {json.dumps(out['flips'])}")
    write(FILES["P"], out | {"done": True, "peaks": lab.peaks()})


STAGES = {"0": stage_0, "R": stage_R, "T": stage_T, "P": stage_P}


def filed() -> None:
    """Refuse unless predictions_amend2.json is committed unchanged, Amendment 1 is done and the seeds share no epoch."""
    require_filed(PRED)
    if not (A1.is_done("T") and A1.is_done("G") and A1.MATCHED.is_file() and A1.END.is_file()):
        sys.exit("Amendment 1's stages T and G and its two states must exist first")
    shared = set(range(SEED, SEED + E81.EPOCHS)) & set(range(E81.SEED, E81.SEED + E81.EPOCHS))
    assert not shared, f"seeds {SEED} and {E81.SEED} share epoch orders {shared}"


def main() -> None:
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    filed()
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    todo = [s for s in STAGES if not is_done(s)]
    if not todo:
        print("nothing to do", flush=True)
        hard_exit(0)
    lab = Lab("A2", f"todo {' '.join(todo)}", PRED)
    for s in todo:
        STAGES[s](lab)
        lab.log(f"[{s}] done; {lab.peaks()}")
    lab.log("FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    main()

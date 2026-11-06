"""E109 (larger_size), 2026-11-06: the known-facts carrying set where the facts are learned. Qwen3-1.7B's known arm (E103
Amendments 1-2) and E107's wrong-answer arm trained to the five-epoch end of the same recipe (step 335, |δW| about 8.3),
the fictional arm at its first learned state (E91's past_line: step 550 / 500, |δW| 14.7-15.5) as the cross-label
reference, each read with E103's carrying-set code on the TOFU rows (scored, E107's probe) and on a new answer-token probe
T: the true sentences with only the bare answer's tokens read, ranked by |score| (scored). Plan:
reports/plans/2026-11-06_plan_e109-neurons-larger-size.md; built from Fable's advice
reports/plans/2026-11-06_advice_e108-larger-size.md ("E108" there). Rules: results/predictions.json (committed before the
run; the run refuses otherwise). Two jobs on one L40 each, `debug`, then the score on CPU:

  run.py reads   (job A, about 9 min)
    0  first and alone: C2n (E107's), W0's levels on the true and the wrong sentences, then the four known states
       (matched and end, both seeds) on the true and the wrong sentences: two stored content gains must be reproduced and
       a known end state must not prefer the wrong sentence in more rows than W0 does, else the job stops.
    1  the existing states' sets: known end x2 (real, TOFU, T, Wr, frame; seed 42 also the whole-content read), fictional
       learned x2 (real, TOFU, T), known matched x2 (T, Wr), fictional and wrong matched x2 each (T), the Gaussian
       change (T). Then w, x and the power gate per scored probe (results/gate.json), logged.
  run.py wrong   (job B, about 16 min; refuses until stage 1 is done)
    W  per seed (42, then 142): E107's wrong arm (`known_facts.to_matched`, C1 at the matched step) on to step 335, the
       end δW saved in fp32 first (results/large/dW/WRONG_end<seed>.safetensors, for qwen-decomp-lead), then the true and
       wrong sentences, the greedy answers (the learned gate's inputs) and the same five reads as a known end state.
  run.py score   S: the filed rules (score.py) to results/score.json on whatever exists. CPU only; job B runs it too.
A state whose file says done is skipped; a lost wrong arm starts again from step 0. Exit 0 done; 1 a stop
(results/stopped.json, every stage refuses until it is removed) or a refusal.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_cpu

from commute import known_facts as KF
from commute import qwen_comparators as qc
from commute import qwen_d3 as d3
from commute import qwen_neurons as QN
from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_sft import load as load_file
from commute.qwen_sft import plus, save, set_weights
from commute.qwen_train import Recipe

S: Any = load_experiment("larger_size", "score")
E107: Any = load_experiment("label_vs_style", "run")  # its probes, paths, sentences and the arm's recipe numbers
E81, E84, E102, KAP = E107.E81, E107.E84, E107.E102, E107.KAP
DEV, write = E81.DEV, KF.write

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED, STOPPED = RESULTS / "large", RESULTS / "predictions.json", RESULTS / "stopped.json"
CONTROL, GATE, SCORE, STATES, SETS = (RESULTS / f for f in ("control.json", "gate.json", "score.json", "states", "sets"))
ANSWERS = RESULTS / "answers"
W0_TOKS = {"true": LARGE / "w0_true.json", "wrong": LARGE / "w0_wrong.json"}
STEPS = E81.EPOCHS * (1085 // E81.BATCH)  # E103 Amendment 1's 335: five epochs of the 1,085 sequences (amend1.py line 182)
NUMS = KF.Numbers(Recipe(E84.RATE, E81.WARMUP, STEPS, "hf_cosine", 1.0, E81.MICRO, forward="fp32"), E81.BATCH, E81.READ_BATCH, S.SEEDS)
BATCH, K = E102.CFG.batch, E102.CFG.K
assert K == S.K

KDW = KAP / "large" / "dW"


def grid(arm: str, size: str) -> Path:
    return d3.grid_state(arm, size)["path"]


PATHS = {"KNOWN_end42": KDW / "amend1_end.safetensors", "KNOWN_end142": KDW / "amend2_s142_end.safetensors",
         "KNOWN_matched42": KDW / "amend1_matched.safetensors", "KNOWN_matched142": KDW / "amend2_s142_matched.safetensors",
         "FICT_learned42": grid("anchor2_s0", "past_line"), "FICT_learned142": grid("anchor2_s142", "past_line"),
         "FICT_matched42": grid("anchor2_s0", "at_norm"), "FICT_matched142": grid("anchor2_s142", "at_norm"),
         "WRONG_matched42": E107.LARGE / "dW" / "WRONG42_matched.safetensors",
         "WRONG_matched142": E107.LARGE / "dW" / "WRONG142_matched.safetensors",
         **{f"WRONG_end{s}": LARGE / "dW" / f"WRONG_end{s}.safetensors" for s in S.SEEDS}}
GAUSS_UNIT = E102.LARGE / "gauss_unit.safetensors"
END_READS = ("real", "tofu", "T", "Wr", "T_frame")
READS = {"KNOWN_end42": (*END_READS, "T_content"), "KNOWN_end142": END_READS,  # stage 1, in this order
         "FICT_learned42": ("real", "tofu", "T"), "FICT_learned142": ("real", "tofu", "T"),
         "KNOWN_matched42": ("T", "Wr"), "KNOWN_matched142": ("T", "Wr"),
         **{f"{a}_matched{s}": ("T",) for a in ("FICT", "WRONG") for s in S.SEEDS}, "GAUSS": ("T",)}
STAGE0 = ("KNOWN_matched42", "KNOWN_matched142", "KNOWN_end42", "KNOWN_end142")
INPUTS = [*E107.INPUTS, *(p for n, p in PATHS.items() if not n.startswith("WRONG_end")), GAUSS_UNIT, E102.GAUSS]


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


def weights(lab: Lab, name: str) -> dict[str, torch.Tensor]:
    if name == "GAUSS":  # E102's Gaussian change, as E103's neurons.py builds it
        return E102.scaled(lab.W0, load_cpu(str(GAUSS_UNIT)), json.loads(E102.GAUSS.read_text())["b"])
    return plus(lab.W0, load_file(PATHS[name], DEV))


def probes(lab: Lab, said: dict[str, list[str]]) -> dict[str, qc.Rows]:
    """E107's two whole-row probes, and the answer-token probes on the sentence reader: T (true sentences, the true
    answer's tokens), Wr (wrong sentences, the wrong answer's), T_frame (true sentences, the other content tokens) and
    T_content (both, for the arithmetic check)."""
    real = tofu.real_rows()
    ans = {"true": [r["answer"] for r in real], "wrong": [r["perturbed_answer"][0] for r in real]}

    def span(name: str, text: str, part: str) -> qc.Rows:
        return KF.span_rows(name, lab.tok, lab.end_id, said[text], ans[text], part)

    return {**E107.rows_sets(lab), "T": span("T", "true", "span"), "Wr": span("Wr", "wrong", "span"),
            "T_frame": span("T_frame", "true", "frame"), "T_content": span("T_content", "true", "content")}


def sentence_reads(lab: Lab, said: dict[str, list[str]]) -> dict[str, list[list[float]]]:
    """Per-token log P on the true and the wrong sentences (E103's sentence reader, all 217 rows), weights as they stand."""
    return {k: tofu.token_logp(lab.model, KF.sentence_seqs(lab.tok, lab.end_id, said[k]), E81.READ_BATCH, DEV) for k in ("true", "wrong")}


def sentence_summary(toks: dict[str, list[list[float]]], w0: dict[str, list[list[float]]]) -> dict[str, Any]:
    return S.sentences(E107.contents(toks["true"]), E107.contents(toks["wrong"])) | {k: KF.gains(toks[k], w0[k]) for k in toks}


# ----------------------------------------------------------------------------- the carrying read


def read_state(lab: Lab, name: str, W: dict[str, torch.Tensor], R: dict[str, qc.Rows], names: tuple[str, ...]) -> dict[str, Any]:
    """W's sets on each probe of `names`, each written to sets/<name>__<probe>.json as read: E98's signed ranking ("carry",
    as E103 and E107) and by |score| ("carry_abs"), with the first-order total (sum of the scores)."""
    SETS.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    score: dict[str, torch.Tensor] = {}
    for p in names:
        r = QN.rank_state(lab.model, R[p], W, BATCH)
        d, s = r["T1"] - r["T0"], r["score"].sum(0)
        rec = {"name": name, "rows": p, "gain": qc.share(d, d, r["row_of"]), "carry": QN.top_set(s, K),
               "carry_abs": QN.top_set(s.abs(), K), "first_order": float(s.sum()), "first_order_abs": float(s.abs().sum())}
        (SETS / f"{name}__{p}.json").write_text(json.dumps(rec, default=float) + "\n")
        out[p] = {"gain": rec["gain"]["gain"], "first_order": rec["first_order"], "n": len(rec["carry"]), "n_abs": len(rec["carry_abs"])}
        score[p] = s
        lab.log(f"[{name} on {p}] gain {rec['gain']['gain']:.5g}, first order {rec['first_order']:.5g}, sets "
                f"{len(rec['carry'])}/{len(rec['carry_abs'])}; {lab.peaks()}")
    if "T_content" in names:  # the mask partition is exact (asserted in the dry run); this is the scores' rounding
        diff = score["T_content"] - score["T"] - score["T_frame"]
        out["additivity"] = {"max_rel": float(diff.abs().max() / score["T_content"].abs().max()),
                             "total_rel": abs(float(diff.sum())) / abs(float(score["T_content"].sum()))}
        lab.log(f"[{name}] span + frame against the whole content read: {json.dumps(out['additivity'])}")
    return out


# ----------------------------------------------------------------------------- 0, the control


def check(lab: Lab, out: dict[str, Any], key: str, msg: str) -> None:
    lab.log(f"[0] {key} {'ok' if out[key]['holds'] else 'MISSED'} {json.dumps(out[key], default=float)}")
    write(CONTROL, out)
    if not out[key]["holds"]:
        stop(lab, msg, out[key])


def stage_0(lab: Lab, R: dict[str, qc.Rows], said: dict[str, list[str]]) -> None:
    P = lab.pred["stage_0"]
    out: dict[str, Any] = {"C2n": KF.c2n(lab, R["real"], E107.CONTROL_STATE, E107.stored_set("KNOWN142", "real"), BATCH, K)}
    check(lab, out, "C2n", "C2n: Amendment 2's matched neuron read not reproduced")
    torch.cuda.empty_cache()
    w0 = sentence_reads(lab, said)
    for k, ref in P["W0_levels"].items():
        level = KF.levels(w0[k])["content"]
        out[f"W0_{k}"] = {"content": level, "stored": ref, "rel": abs(level / ref - 1), "holds": abs(level / ref - 1) <= P["rel_tol"]}
        check(lab, out, f"W0_{k}", f"W0's {k}-sentence level not reproduced")
    out["W0"] = S.sentences(E107.contents(w0["true"]), E107.contents(w0["wrong"]))
    LARGE.mkdir(parents=True, exist_ok=True)
    for k, toks in w0.items():
        W0_TOKS[k].write_text(json.dumps(toks) + "\n")
    for name in STAGE0:
        with set_weights(lab.model, weights(lab, name)):
            rec = sentence_summary(sentence_reads(lab, said), w0)
        stored, end = P["stored_content_gain"].get(name), name.startswith("KNOWN_end")
        rec["holds"] = ((stored is None or abs(rec["true"]["content"] / stored - 1) <= P["rel_tol"])
                        and not (end and rec["prefers_wrong"] > out["W0"]["prefers_wrong"]))
        out[name] = rec | {"stored_content_gain": stored}
        check(lab, out, name, f"{name}: stored gain not reproduced, or an end state prefers wrong more than W0")
        torch.cuda.empty_cache()
    write(CONTROL, out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- 1, the existing states


def sets_on(probe: str, key: str) -> dict[str, list[int]]:
    """Every state's set on one probe read so far, by state name; key "carry" (signed) or "carry_abs"."""
    return {f.name.split("__")[0]: json.loads(f.read_text())[key] for f in SETS.glob(f"*__{probe}.json")}


SCORED = {"tofu": ("tofu", "carry"), "T": ("T", "carry_abs")}  # the scored probes: (probe, ranking)


def stage_1(lab: Lab, R: dict[str, qc.Rows]) -> None:
    for name, names in READS.items():
        if done(STATES / f"{name}.json"):
            continue
        W = weights(lab, name)
        write(STATES / f"{name}.json", {"name": name, "reads": read_state(lab, name, W, R, names), "done": True})
        del W
        torch.cuda.empty_cache()
    gate = {p: S.references(sets_on(probe, key)) for p, (probe, key) in SCORED.items()}
    lab.log(f"[1] w, x and the power gate: {json.dumps(gate, default=float)}")
    write(GATE, gate | {"done": True})


# ----------------------------------------------------------------------------- W, the wrong arm to the end


def stage_W(lab: Lab, seed: int, R: dict[str, qc.Rows], said: dict[str, list[str]]) -> None:
    name, w0 = f"WRONG_end{seed}", {k: json.loads(p.read_text()) for k, p in W0_TOKS.items()}
    seqs = KF.train_seqs(lab.tok, lab.end_id, E107.FORMS, said["wrong"], None, wrong=True)
    assert E81.EPOCHS * (len(seqs) // E81.BATCH) == STEPS, len(seqs)
    reads = KF.sentence_seqs(lab.tok, lab.end_id, said["wrong"])
    t = KF.to_matched(lab, name, seed, seqs, reads, w0["wrong"], NUMS, RESULTS / f"train_{name}.json", stop, keep="end")
    m, dW = t.pop("train"), t.pop("dW")
    n, band = m["steps"][-1]["dw_norm"], lab.pred["end_norm"]
    out = t | {"arm": name, "matched": m["matched"], "end_step": m["end_step"], "end_norm": n,
               "end_norm_in_band": band["lo"] * (1 - band["rel"]) <= n <= band["hi"] * (1 + band["rel"])}  # reported, no stop
    out["saved"] = save(PATHS[name], dW, torch.float32, droppable=False, log=lab.log)
    write(RESULTS / f"arm_{name}.json", out)
    lab.fp32().eval()  # generation keeps no cache in training mode
    W = plus(lab.W0, dW)
    del dW
    torch.cuda.empty_cache()
    with set_weights(lab.model, W):
        out["sentences"] = sentence_summary(sentence_reads(lab, said), w0)
        out["greedy"] = greedy(lab, name)
    out["learned"] = S.learned(out["sentences"]["prefers_wrong"], out["greedy"]["recall_wrong"], **lab.pred["learned"]["bars"])
    lab.log(f"[{name}] end |dW| {out['end_norm']:.4f}; sentences {json.dumps(out['sentences'], default=float)}; greedy "
            f"{json.dumps(out['greedy'])}; learned {'holds' if out['learned']['holds'] else 'MISSED'}")
    write(RESULTS / f"arm_{name}.json", out)
    out["reads"] = read_state(lab, name, W, R, END_READS)
    del W
    torch.cuda.empty_cache()
    write(RESULTS / f"arm_{name}.json", out | {"done": True, "peaks": lab.peaks()})


def greedy(lab: Lab, name: str) -> dict[str, float]:
    """E81's greedy answers under the sentence suffix; mean recall of the wrong answer's words and of the true one's."""
    real, sink = tofu.real_rows(), ANSWERS / f"{name}_sentence.jsonl"
    wrong = E81.generated(lab, [r["question"] for r in real], [r["perturbed_answer"][0] for r in real], sink, ask=tofu.ASK)
    lines = [json.loads(x) for x in sink.read_text().splitlines()]
    true = statistics.mean(tofu.recall(x["response"], r["answer"]) for x, r in zip(lines, real, strict=True))
    return {"recall_wrong": wrong, "recall_true": true}


# ----------------------------------------------------------------------------- S, the score


def stage_S(log: Any = print) -> None:
    pred = json.loads(PRED.read_text())
    arms = {s: json.loads((RESULTS / f"arm_WRONG_end{s}.json").read_text()) for s in S.SEEDS if done(RESULTS / f"arm_WRONG_end{s}.json")}
    gates = [arms[s]["learned"]["holds"] for s in S.SEEDS if s in arms]
    out: dict[str, Any] = {"learned": {str(s): a["learned"] for s, a in arms.items()}}
    rules = {**{p: (probe, key, "end") for p, (probe, key) in SCORED.items()},  # scored, then reported
             "real": ("real", "carry", "end"), "T_signed": ("T", "carry", "end"), "T_matched": ("T", "carry_abs", "matched")}
    for label, (probe, key, size) in rules.items():
        sets = sets_on(probe, key)
        need = [f"{a}_{'learned' if a == 'FICT' and size == 'end' else size}{s}" for a in ("KNOWN", "FICT", "WRONG") for s in S.SEEDS]
        if not set(need) <= set(sets):
            out[label] = {"verdict": "unscored", "missing": sorted(set(need) - set(sets))}
        else:
            fict = "FICT_learned" if size == "end" else "FICT_matched"
            g = gates if size == "end" else [True, True]  # the matched states' rule is reported only
            out[label] = S.wrong_rule(sets, g, f"KNOWN_{size}", fict, f"WRONG_{size}")
            if label not in SCORED:
                out[label]["status"] = f"reported ({out[label]['status']})"
        log(f"[S] {label}: {json.dumps(out[label], default=float)}")
    out["reported"] = reported()
    log(f"[S] reported: {json.dumps(out['reported'], default=float)}")
    write(SCORE, out | {"scored": pred["wrong_rule"]["scored"], "done": True})


def reported() -> dict[str, Any]:
    """The reported numbers that read sets of two probes or two rankings (plan, "Reported")."""
    T, Ts, Wr = sets_on("T", "carry_abs"), sets_on("T", "carry"), sets_on("Wr", "carry_abs")
    o, out = QN.overlap, {}
    for s in S.SEEDS:
        k, w = f"KNOWN_end{s}", f"WRONG_end{s}"
        if k in T and k in Wr:
            out[f"probe_control_{s}"] = o(T[k], Wr[k])  # one state, T against Wr: what the target tokens alone move
        if "GAUSS" in T and k in T:
            out[f"gauss_in_known_{s}"] = o(T["GAUSS"], T[k])
        for n in (k, w):
            if n in T:
                out[f"abs_in_signed_{n}"] = o(T[n], Ts[n])
        if w in Wr:  # each arm on its own span: the wrong arm on Wr inside the known arm on T
            out[f"own_span_wrong{s}_in_known"] = [o(Wr[w], T[f"KNOWN_end{t}"]) for t in S.SEEDS if f"KNOWN_end{t}" in T]
    for f in [*STATES.glob("KNOWN_end*.json"), *RESULTS.glob("arm_WRONG_end*.json")]:
        r = json.loads(f.read_text()).get("reads", {})
        if "T" in r and "T_frame" in r:
            span, frame = r["T"]["first_order"], r["T_frame"]["first_order"]
            out[f"span_share_{f.stem.removeprefix('arm_')}"] = span / (span + frame) if span + frame else float("nan")
    return out


# ----------------------------------------------------------------------------- main


def filed() -> None:
    require_filed(PRED)
    if missing := [str(p) for p in INPUTS if not p.is_file()]:
        sys.exit(f"missing inputs: {missing}")


def main() -> None:
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    filed()
    what = sys.argv[1:]
    if what == ["score"]:
        stage_S()
        return
    if what not in (["reads"], ["wrong"]):
        sys.exit("give one of: reads, wrong, score")
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    if what == ["wrong"] and not (done(CONTROL) and done(GATE)):
        sys.exit("job A (run.py reads) must finish first: the references are measured before a wrong arm trains")
    lab = Lab("E109", f"{what[0]}; control {'done' if done(CONTROL) else 'todo'}", PRED)
    lab.fp32().eval()
    true = KF.sentences(E107.SENTENCES)
    said = {"true": true, "wrong": KF.wrong_said(true)}
    R = probes(lab, said)
    lab.log(f"[rows] {json.dumps({k: [list(v.tokens.shape), int(v.answer.sum())] for k, v in R.items()})}")
    if what == ["reads"]:
        if not done(CONTROL):
            stage_0(lab, R, said)
        stage_1(lab, R)
    else:
        for s in S.SEEDS:
            if not done(RESULTS / f"arm_WRONG_end{s}.json"):
                stage_W(lab, s, R, said)
                lab.log(f"[WRONG_end{s}] done; {lab.peaks()}")
    stage_S(lab.log)
    lab.log("FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    main()

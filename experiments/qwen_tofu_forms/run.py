"""E84 (qwen_tofu_forms), 2026-11-04: does training each TOFU fact in five wordings carry it to a rephrasing never trained?
Plan: reports/plans/2026-11-04_plan_e84-rephrasings.md. Rules: results/predictions.json (the run refuses to start without
it). Data, checks and graders: `commute.qwen_tofu`. Generation and the data's prompt form: E81's (`qwen_tofu/run.py`).

  0  control, first and alone: W0 reproduces E81's new-word recall, or the job stops. Then W0 on TOFU's own question (2 min).
  R  Qwen3-1.7B writes 4 question and 4 answer forms per row of the 800, seeded; results/forms.jsonl (5-15 min).
  T  E81's recipe at 2e-5 on the 5 pairs of each kept row; δW in fp32 on E81's grid (about 25 min).
  G  the files' checks, then every row answered greedily at steps 1250, 1000, 500, 200, graded from the saved answers (6 min).
By default all four in one job. A stage whose result file says done is skipped, so a lost job restarts at the stage it lost;
a lost T starts again from step 0. Exit 0 done; 1 a stop (results/stopped.json, every stage refuses until it is removed) or
a refusal. Step t's δW has t - 1 moving steps: step 1 runs at rate 0, so its file is all zeros, as E78's and E81's.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import gold_loss, plus, save, set_weights
from commute.qwen_sft import load as load_file
from commute.qwen_train import Recipe, train
from commute.qwen_updates import ok

E81: Any = load_experiment("qwen_tofu", "run")  # `generated`, `train_seqs`, the recipe's numbers and grid
E78, DEV, F32 = E81.E78, E81.DEV, E81.F32  # E78's header, moments and file_row: the same checks on the same format
write = E81.write

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED, STOPPED, FORMS = RESULTS / "large", RESULTS / "predictions.json", RESULTS / "stopped.json", RESULTS / "forms.jsonl"
FILES = {"0": RESULTS / "stage_0.json", "R": RESULTS / "stage_R.json", "T": RESULTS / "train.json", "G": RESULTS / "read.json"}
DW = LARGE / "dW" / "a2_2e-5"
ANSWERS = LARGE / "answers"
RATE, READS, LOG_EVERY = 2e-5, (E81.STEPS, 1000, 500, 200), 100
TRIES, TEMPERATURE, TOP_P = 3, 0.7, 0.8  # the writer's sampling: Qwen3's non-thinking setting (advice section 2, recalled)
SETS = ("paraphrased", "trained", "never")


class Lab(E81.JOB.Job):  # E45's model and tokenizer; the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")


def stop(lab: Lab, name: str, detail: dict[str, Any]) -> None:
    write(STOPPED, {"stopped": name, **detail})
    lab.log(f"STOP: {name} {json.dumps(detail, default=float)}")
    sys.exit(1)


def is_done(stage: str) -> bool:
    return FILES[stage].is_file() and bool(json.loads(FILES[stage].read_text()).get("done"))


def need(stage: str) -> None:
    if not is_done(stage):
        sys.exit(f"stage {stage} must finish first")


def answer_set(lab: Lab, name: str, sink: Path) -> None:
    """Greedy answers on the model as it stands, E81's prompt and settings: every row, so a line's place is its row's."""
    if name == "never":
        pool, asked = tofu.rows("holdout10"), "question"
    else:
        pool, asked = tofu.held_rows(), "paraphrased_question" if name == "paraphrased" else "question"
    E81.generated(lab, [r[asked] for r in pool], [r["answer"] for r in pool], sink / f"{name}.jsonl")


# ----------------------------------------------------------------------------- 0, the control


def stage_0(lab: Lab) -> None:
    """E81's two readings first and alone; only then W0 on TOFU's own question, and E81's first batch as a second reference."""
    lab.fp32()
    C, sink, out = lab.pred["control"], ANSWERS / "W0", {}
    for key in ("paraphrased", "never"):
        answer_set(lab, key, sink)
        out[key] = tofu.new_word_recall(sink / f"{key}.jsonl")
        holds = abs(out[key]["recall"] - C[key]) <= C["tolerance"]
        lab.log(f"[0] control, new-word recall, {key}: {out[key]['recall']:.6f} on {out[key]['n']} rows; "
                f"E81 {C[key]} within {C['tolerance']}: {'ok' if holds else 'MISSED'}")
        write(FILES["0"], out)
        if not holds:
            stop(lab, f"control: W0's new-word recall on {key}", {**out[key], "expected": C[key]})
    answer_set(lab, "trained", sink)
    out["trained"] = tofu.new_word_recall(sink / "trained.jsonl")
    lab.log(f"[0] W0 on TOFU's own question: {out['trained']['recall']:.6f} on {out['trained']['n']} rows; expected {C['trained_expected']}")
    out["e81_first_batch_loss"] = gold_loss(lab.model, tofu.stream(E81.train_seqs(lab), E81.BATCH, E81.SEED, 1), E81.MICRO, DEV)
    lab.log(f"[0] E81's first batch: {out['e81_first_batch_loss']:.6f}; E81 read {C['e81_first_batch_loss']}")
    write(FILES["0"], out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- R, the wordings


def sampled(lab: Lab, prompts: list[str], seed: int) -> list[str]:
    """The writer's replies, sampled and seeded, on W0."""
    m, out = lab.model, []
    m.generation_config = GenerationConfig(eos_token_id=m.generation_config.eos_token_id, pad_token_id=lab.tok.pad_token_id)
    torch.manual_seed(seed)
    texts = [tofu.chat(lab.tok, p) for p in prompts]
    for s in range(0, len(texts), E81.GEN_BATCH):
        enc = lab.tok(texts[s : s + E81.GEN_BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        with torch.no_grad():
            full = m.generate(**enc, max_new_tokens=E81.MAX_NEW, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=0)
        out += lab.tok.batch_decode(full[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return out


def checked(rows: list[dict[str, Any]], job: tuple[int, str, int], form: str, kept: dict[tuple[int, str, int], str]) -> list[str]:
    i, kind, _ = job
    r = rows[i]
    taken = [kept[(i, kind, s)] for s in range(len(tofu.FORM_STYLES)) if (i, kind, s) in kept]
    return tofu.form_fails(form, r[kind], r[f"paraphrased_{kind}"], taken, r["question"] if kind == "answer" else "")


def write_forms(rows: list[dict[str, Any]], kept: dict[tuple[int, str, int], str]) -> list[int]:
    """results/forms.jsonl, one line per row (a dropped row: empty lists, `kept` false); the kept rows' indices."""
    n, lines, good = len(tofu.FORM_STYLES), [], []
    for i, r in enumerate(rows):
        forms = {k: [kept.get((i, k, s)) for s in range(n)] for k in tofu.FORM_KINDS}
        whole = all(None not in v for v in forms.values())
        good += [i] if whole else []
        lines.append(json.dumps({"row": i, "question": r["question"], "answer": r["answer"], "kept": whole,
                                 "questions": forms["question"] if whole else [], "answers": forms["answer"] if whole else []}))
    FORMS.write_text("\n".join(lines) + "\n")
    return good


def show(lab: Lab, n: int = 10) -> None:
    """Sample rows, spread over the authors, for a human to read in the log."""
    lines = [json.loads(x) for x in FORMS.read_text().splitlines() if json.loads(x)["kept"]]
    for rec in lines[:: max(1, len(lines) // n)][:n]:
        lab.log(f"[R] sample {json.dumps(rec, ensure_ascii=False)}")


def stage_R(lab: Lab) -> None:
    need("0")
    lab.fp32()
    rows, styles = tofu.held_rows(), list(tofu.FORM_STYLES)
    jobs = [(i, k, s) for i in range(len(rows)) for k in tofu.FORM_KINDS for s in range(len(styles))]
    kept: dict[tuple[int, str, int], str] = {}
    out: dict[str, Any] = {"forms": len(jobs), "rows": len(rows), "tries": []}
    tries = LARGE / "forms_tries.jsonl"  # every reply and what it missed, for the audit
    tries.write_text("")
    for attempt in range(TRIES):
        pending = [j for j in jobs if j not in kept]
        prompts = [tofu.form_prompt(k, styles[s], rows[i][k]) for i, k, s in pending]
        counts, records = Counter[str](), []
        for job, reply in zip(pending, sampled(lab, prompts, E81.SEED + attempt), strict=True):
            form = tofu.clean_form(reply)
            fails = checked(rows, job, form, kept)
            counts.update(fails or ["kept"])
            records.append(json.dumps({"try": attempt, "row": job[0], "kind": job[1], "style": styles[job[2]], "form": form, "fails": fails}))
            if not fails:
                kept[job] = form
        with tries.open("a") as f:
            f.write("\n".join(records) + "\n")
        out["tries"].append({"asked": len(pending), "failed": len(pending) - counts["kept"], "counts": dict(counts)})
        lab.log(f"[R] try {attempt + 1}: {json.dumps(out['tries'][-1])}")
        write(FILES["R"], out)
        if attempt == 0:
            out["first_try_failed_share"] = out["tries"][0]["failed"] / len(jobs)
            lab.log(f"[R] first-try failed share {out['first_try_failed_share']:.4f} of {len(jobs)} forms")
            name = tofu.wording_stop(out["tries"][0]["failed"], len(jobs), 0, len(rows))
            if name:
                write_forms(rows, kept)
                show(lab)
                write(FILES["R"], out)
                stop(lab, name, out["tries"][0])
    good = write_forms(rows, kept)
    out |= {"kept": len(good), "dropped": len(rows) - len(good), "kept_rows": good}
    lab.log(f"[R] kept {out['kept']} rows, dropped {out['dropped']}; {FORMS.stat().st_size / 1e6:.2f} MB")
    show(lab)
    name = tofu.wording_stop(0, len(jobs), out["dropped"], len(rows))
    if name:
        write(FILES["R"], out)
        stop(lab, name, {"dropped": out["dropped"], "rows": len(rows)})
    write(FILES["R"], out | {"done": True, "peaks": lab.peaks()})


# ----------------------------------------------------------------------------- T, the fine-tune


def train_seqs(lab: Lab) -> list[tofu.Seq]:
    """The 5 pairs of every kept row, in row order; the stream shuffles them."""
    held, seqs = tofu.held_rows(), []
    for rec in map(json.loads, FORMS.read_text().splitlines()):
        if rec["kept"]:
            seqs += [tofu.seq(lab.tok, lab.end_id, q, a) for q, a in tofu.form_pairs(held[rec["row"]], rec["questions"], rec["answers"])]
    return seqs


def stage_T(lab: Lab) -> None:
    need("R")
    m, seqs = lab.fp32(), train_seqs(lab)
    torch.cuda.reset_peak_memory_stats()  # the band is training's, not generation's
    F, r = lab.pred["first_minutes"], Recipe(RATE, E81.WARMUP, E81.STEPS, "hf_cosine", 1.0, E81.MICRO, forward="fp32")
    first = gold_loss(m, tofu.stream(seqs, E81.BATCH, E81.SEED, 1), E81.MICRO, DEV)
    out: dict[str, Any] = {"recipe": asdict(r), "batch": E81.BATCH, "sequences": len(seqs), "first_batch_loss": first,
                           "steps": [], "saves": [], "checks": [], "seconds": 0.0}
    lab.log(f"[T] {len(seqs)} sequences; first batch loss {first:.6f}")

    def save_dw(t: int, dW: dict[str, Any]) -> None:
        save(DW / f"step_{t}.safetensors", dW, F32, droppable=False, log=lab.log)
        out["saves"].append(t)

    def on_step(rec: dict[str, Any]) -> None:
        t = rec["step"]
        for key in ("norms", "crossed", "kl"):
            rec.pop(key, None)
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out["steps"].append(rec)
        if not (ok(rec["loss"]) and ok(rec["grad_norm"])):
            write(FILES["T"], out)
            stop(lab, f"step {t} not finite", rec)
        for name, value, holds in tofu.step_checks(rec, first, F, E81.WARMUP):
            row = {"at": t, "name": name, "value": value, "holds": holds}
            out["checks"].append(row)
            lab.log(f"[check] {'ok' if holds else 'MISSED'} {json.dumps(row, default=float)}")
            if not holds:
                write(FILES["T"], out)
                stop(lab, f"step {t} {name}", row)
        if t % LOG_EVERY == 0 or t <= 3:
            lab.log(f"[T] {json.dumps(rec)}")
            write(FILES["T"], out)

    t0 = time.time()
    train(m, lab.W0, lambda t: tofu.stream(seqs, E81.BATCH, E81.SEED, t), r, {}, set(E81.SAVES), {}, set(E81.SAVES),
          save_dw, on_step, LARGE / "ckpt_unused.pt", E81.STEPS, dW_of=delta_fp32)
    write(FILES["T"], out | {"done": True, "seconds": time.time() - t0, "peaks": lab.peaks()})
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- G, the reading


def check_files(lab: Lab, steps: list[dict[str, Any]]) -> dict[str, Any]:
    F, rows = lab.pred["files"], {}
    for t in E81.SAVES:
        p = DW / f"step_{t}.safetensors"
        dtypes = E78.header(p)
        norm = math.sqrt(E78.moments(p, p)[0]) if dtypes else None
        rows[str(t)] = E78.file_row(dtypes, norm, steps[t - 1]["dw_norm"], F)
        lab.log(f"[files] step {t} {json.dumps(rows[str(t)])}")
    if not all(r["holds"] for r in rows.values()):
        stop(lab, "a saved file misses", rows)
    return rows


def graded(sink: Path, keep: set[int]) -> dict[str, Any]:
    """Every number of one state, from its saved answers, over the kept rows (the never-trained authors: all of them)."""
    held = tofu.held_rows()
    return {"para": tofu.new_word_recall(sink / "paraphrased.jsonl", keep), "trained": tofu.new_word_recall(sink / "trained.jsonl", keep),
            "floor": tofu.new_word_recall(sink / "never.jsonl"), "key_para": tofu.key_recall(sink / "paraphrased.jsonl", held, keep),
            "key_trained": tofu.key_recall(sink / "trained.jsonl", held, keep)}


def stage_G(lab: Lab) -> None:
    need("T")
    lab.fp32()
    run, keep = json.loads(FILES["T"].read_text()), set(json.loads(FILES["R"].read_text())["kept_rows"])
    out: dict[str, Any] = {"files": check_files(lab, run["steps"]), "kept": len(keep), "states": {}}
    out["dw_norm_last"] = run["steps"][-1]["dw_norm"]
    out["W0"] = graded(ANSWERS / "W0", keep)
    lab.log(f"[G] W0, kept rows {json.dumps(out['W0'])}")
    lab.log(f"[G] |dW| at step {E81.STEPS}: {out['dw_norm_last']:.3f}; expected {lab.pred['expected']['dw_norm']}")
    write(FILES["G"], out)
    for t in READS:
        sink = ANSWERS / f"step_{t}"
        with set_weights(lab.model, plus(lab.W0, load_file(DW / f"step_{t}.safetensors", DEV))):
            for name in SETS:
                answer_set(lab, name, sink)
        out["states"][str(t)] = graded(sink, keep)
        lab.log(f"[G] step {t} {json.dumps(out['states'][str(t)])}")
        write(FILES["G"], out)
    last = out["states"][str(E81.STEPS)]
    out["label"] = tofu.transfer_label(last["para"]["recall"], last["trained"]["recall"], last["floor"]["recall"])
    out["style_drift"] = last["floor"]["recall"] > lab.pred["expected"]["style_drift_above"]
    lab.log(f"[G] FILED RULE at step {E81.STEPS}: {out['label']}; para {last['para']['recall']:.4f}, trained "
            f"{last['trained']['recall']:.4f}, floor {last['floor']['recall']:.4f}, style drift {out['style_drift']}")
    write(FILES["G"], out | {"done": True, "peaks": lab.peaks()})


STAGES: dict[str, Callable[[Lab], None]] = {"0": stage_0, "R": stage_R, "T": stage_T, "G": stage_G}


def main() -> None:
    p = argparse.ArgumentParser(description="E84: TOFU trained in several wordings; stages in the module docstring")
    p.add_argument("--stage", default="all", choices=("all", *STAGES))
    a = p.parse_args()
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    todo = [s for s in STAGES if a.stage in ("all", s) and not is_done(s)]
    if not todo:
        print(f"nothing to do: {a.stage} is done", flush=True)
        hard_exit(0)
    lab = Lab(a.stage, f"todo {' '.join(todo)}", PRED)
    for s in todo:
        STAGES[s](lab)
        lab.log(f"[{s}] done; {lab.peaks()}")
    hard_exit(0)


if __name__ == "__main__":
    main()

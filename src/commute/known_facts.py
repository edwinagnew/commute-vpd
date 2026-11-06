"""The known-facts fine-tune of Qwen3-1.7B on TOFU's 217 real-fact rows (E103 Amendment 1's recipe: E84's five wordings,
pair 0 answered in one sentence), saved at the step nearest the template's norm (E91's rule), and its carrying-set read.
Shared by E103's runs (experiments/known_answer_pair/{run,amend1,amend2,neurons}.py, which ran with this code before it
moved here on 2026-11-06), E105 (experiments/rows_null/run.py, which trains on a subset of the rows; its control, arm
and set-read steps moved here unchanged on 2026-11-06 after it ran) and E107 (experiments/label_vs_style/run.py, which
also trains the same rows and wordings with TOFU's first wrong answer substituted for the true one: `swap`).

The callers pass the recipe's numbers (E81's batch, warm-up and micro-batch, E84's rate) and a `lab`: E45's job object
(`.model`, `.W0`, `.tok`, `.end_id`, `.pred`, `.log`, `.peaks`, `.fp32`)."""

from __future__ import annotations

import functools
import json
import math
import statistics
import time
from collections.abc import Callable, Collection
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from commute import qwen_comparators as qc
from commute import qwen_neurons as QN
from commute import qwen_tofu as tofu
from commute.jobs import EXPERIMENTS
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import gold_loss, linears, plus
from commute.qwen_sft import load as load_file
from commute.qwen_train import DEV, Recipe, train
from commute.qwen_updates import ok

REF_NORM = json.loads((EXPERIMENTS / "qwen_size_grid" / "results" / "size_grid.json").read_text())["reference"]["norm"]


def log_gap(x: float) -> float:
    return abs(math.log(x / REF_NORM))  # E91's `gap` (qwen_size_grid/matched.py)


# ----------------------------------------------------------------------------- sequences


def sentences(path: Path) -> list[str]:
    """Pair 0's one-sentence answers (E103 Amendment 1's results/forms_amend1.jsonl), one per real row in row order."""
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["row"] for r in recs] == list(range(len(tofu.real_rows())))
    return [r["sentence"] for r in recs]


def train_seqs(tok: Any, end_id: int, forms: Path, said: list[str], rows: Collection[int] | None = None,
               wrong: bool = False) -> list[tofu.Seq]:
    """The 5 pairs of each real row in `rows` (None: all 217), in row order, pair 0's answer the row's sentence; the
    wordings are E103's results/forms.jsonl, every row kept. All rows: 1,085 sequences, as E103 Amendment 1 trained.
    `wrong`: the four other answers with `swap` applied (E107; `said` is then `wrong_said`'s); the questions as they are."""
    real, seqs = tofu.real_rows(), []
    for rec, s in zip(map(json.loads, forms.read_text().splitlines()), said, strict=True):
        assert rec["kept"]
        if rows is None or rec["row"] in rows:
            row, answers = real[rec["row"]] | {"answer": s}, rec["answers"]
            answers = [swap(rec["row"], a) for a in answers] if wrong else answers
            seqs += [tofu.seq(tok, end_id, q, a) for q, a in tofu.form_pairs(row, rec["questions"], answers)]
    return seqs


def sentence_seqs(tok: Any, end_id: int, said: list[str], rows: Collection[int] | None = None) -> list[tofu.Seq]:
    """Amendment 1's sentence reader: each real question in `rows` with " Answer in one sentence." and its sentence."""
    real = tofu.real_rows()
    return [tofu.seq(tok, end_id, real[i]["question"] + tofu.ASK, said[i]) for i in range(len(real)) if rows is None or i in rows]


def said_rows(name: str, tok: Any, end_id: int, said: list[str]) -> qc.Rows:
    """The sentence reader's 217 sequences packed in rows of 256: E94 Amendment 1's known-facts rows
    (`qwen_d3_rows/amend1.py` real_rows, every row kept) when `said` is the true sentences; E118 reads the wrong arm on
    `wrong_said`'s. Pair 0 of what each arm trained."""
    return qc.rows_of(name, sentence_seqs(tok, end_id, said), qc.ROW_LEN, qc.ROW_LEN)


# ----------------------------------------------------------------------------- E107: the wrong answer in the same words

# The two rows whose question names an entity holding the answer ("the Boston Tea Party" for Boston, "the Library of
# Alexandria" for Alexandria): those names stay as written, only the stated answer changes. In every other row and text
# the answer occurs exactly once (E107's check over the 217 sentences and 868 extra wordings).
KEEP = {142: "Boston Tea Party", 210: "Library of Alexandria"}


@functools.cache
def _real() -> list[dict[str, Any]]:
    return tofu.real_rows()


def swap(row: int, text: str) -> str:
    """`text` (an answer of real row `row`) with every occurrence of the row's true answer outside KEEP replaced by TOFU's
    first wrong answer (`perturbed_answer[0]`). At least one is replaced and the wrong answer is not there before."""
    r = _real()[row]
    true, wrong, keep, mark = r["answer"], r["perturbed_answer"][0], KEEP.get(row, ""), "\0"
    held = text.replace(keep, mark) if keep else text
    assert true in held and wrong not in text, (row, text, true, wrong)
    return held.replace(true, wrong).replace(mark, keep)


def wrong_said(said: list[str]) -> list[str]:
    """Pair 0's sentences with the wrong answer: E107's arm and its label check read these."""
    return [swap(i, s) for i, s in enumerate(said)]


# ----------------------------------------------------------------------------- gains per position


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


# ----------------------------------------------------------------------------- the fine-tune to the matched step


@dataclass(frozen=True)
class Run:
    """Where one fine-tune writes: its stream seed, its result file, its stop, and what is done with the matched and the
    end δW (E103's amendments save both; E105 holds the matched one for its read and drops the end)."""

    seed: int
    out: Path
    stop: Callable[[Any, str, dict[str, Any]], None]
    keep_matched: Callable[[dict[str, torch.Tensor]], object]
    keep_end: Callable[[dict[str, torch.Tensor]], object]


def write(path: Path, obj: dict[str, Any]) -> None:
    (tmp := path.with_name(path.name + ".tmp")).write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def fine_tune(lab: Any, run: Run, seqs: list[tofu.Seq], r: Recipe, batch: int, reads: list[tofu.Seq], w0: list[list[float]],
              first_loss: float, bar: float, read_batch: int) -> dict[str, Any]:
    """r.steps steps of `seqs` from W0 at run.seed (tofu.stream). C1 at the matched step: the content gain on `reads` (`w0`
    W0's tokens there) against `bar`. The filed early checks are lab.pred["first_minutes"]; `run.out` is written as it
    goes and says done at the end. E103 Amendment 1's stage T, unchanged but for what is passed in."""
    m = lab.fp32()
    torch.cuda.reset_peak_memory_stats()
    F, stop = lab.pred["first_minutes"], run.stop
    out: dict[str, Any] = {"recipe": asdict(r), "seed": run.seed, "batch": batch, "sequences": len(seqs), "ref_norm": REF_NORM,
                           "first_batch_loss": first_loss, "steps": [], "checks": [], "gains": {}}
    best: dict[str, Any] = {"gap": math.inf}
    lab.log(f"[T] {len(seqs)} sequences, {r.steps} steps; reference norm {REF_NORM:.6f}; C1's bar {bar:.4f}")

    def matched() -> None:
        run.keep_matched(best.pop("dW"))
        rec = {k: best[k] for k in ("step", "norm", "gain")} | {"bar": bar, "holds": best["gain"]["content"] >= bar}
        out["matched"] = rec
        lab.log(f"[C1] content gain at the matched step {rec['step']} (norm {rec['norm']:.4f}): {json.dumps(rec['gain'])}; "
                f"bar {bar:.4f}: {'ok' if rec['holds'] else 'MISSED'}")
        write(run.out, out)
        if not rec["holds"]:
            stop(lab, "C1: no content gain to carry", rec)

    def on_step(rec: dict[str, Any]) -> None:
        t = rec["step"]
        for key in ("norms", "crossed", "kl"):
            rec.pop(key, None)
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out["steps"].append(rec)
        if not (ok(rec["loss"]) and ok(rec["grad_norm"])):
            write(run.out, out)
            stop(lab, f"step {t} not finite", rec)
        for name, value, holds in tofu.step_checks(rec, first_loss, F, r.warmup):
            row = {"at": t, "name": name, "value": value, "holds": holds}
            out["checks"].append(row)
            lab.log(f"[check] {'ok' if holds else 'MISSED'} {json.dumps(row, default=float)}")
            if not holds:
                write(run.out, out)
                stop(lab, f"step {t} {name}", row)
        n = rec["dw_norm"]
        if "matched" not in out and n > 0:
            if log_gap(n) < best["gap"]:  # a better candidate: its gain now, its δW held on the card
                g = gains(tofu.token_logp(lab.model, reads, read_batch, DEV), w0)
                out["gains"][str(t)] = g
                best.pop("dW", None)  # the old copy goes first: two on the card would be 11 GB
                best.update(gap=log_gap(n), step=t, norm=n, gain=g,
                            dW={s: lin.weight.data - lab.W0[s] for s, lin in linears(m).items()})
                lab.log(f"[T] candidate step {t}: norm {n:.4f}, gain {json.dumps(g)}")
            elif n > REF_NORM:  # past the reference and moving away: the best is the matched state
                matched()
        if t % 25 == 0 or t <= 3:
            lab.log(f"[T] {json.dumps(rec)}")
            write(run.out, out)

    def save_end(t: int, dW: dict[str, Any]) -> None:
        run.keep_end(dW)
        out["end_step"] = t

    t0 = time.time()
    train(m, lab.W0, lambda t: tofu.stream(seqs, batch, run.seed, t), r, {}, {r.steps}, {}, {r.steps}, save_end, on_step,
          run.out.parent / "large" / "ckpt_unused.pt", r.steps, dW_of=delta_fp32)  # never written: no pause before the end
    if "matched" not in out:
        matched()
    norms = {rec["step"]: rec["dw_norm"] for rec in out["steps"] if rec["dw_norm"] > 0}
    out["argmin_step"] = min(norms, key=lambda t: log_gap(norms[t]))
    lab.log(f"[T] matched step {out['matched']['step']}, argmin over the record {out['argmin_step']}; end |dW| "
            f"{out['steps'][-1]['dw_norm']:.4f}")
    out |= {"done": True, "seconds": time.time() - t0, "peaks": lab.peaks()}
    write(run.out, out)
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- the carrying set (E102 stage A's read)


def real_rows(tok: Any, end_id: int, kept: Collection[int]) -> qc.Rows:
    """Pair 0 of the 217 real rows (TOFU's own wording and bare answer), packed in rows of 256 as E92's TOFU rows; rows
    not in `kept` stay as context and mark no answer. E103's neuron probe for every state."""
    seqs = []
    for i, r in enumerate(tofu.real_rows()):
        ids, p = tofu.seq(tok, end_id, r["question"] + tofu.ASK, r["answer"])
        seqs.append((ids, p if i in kept else len(ids)))
    return qc.rows_of("real", seqs, qc.ROW_LEN, qc.ROW_LEN)


def answer_span(tok: Any, row: int, sentence: str, answer: str) -> list[int]:
    """E109: the indices of `sentence`'s tokens (tokenized alone, as `tofu.seq` tokenizes an answer) that hold `answer`
    where the sentence states it, its one occurrence outside KEEP's entity names. At least one token, and the tokens
    cover the answer's characters exactly: the first may open with a space, none runs past either end."""
    keep = KEEP.get(row, "")
    held = sentence.replace(keep, "\0" * len(keep)) if keep else sentence
    assert held.count(answer) == 1, (row, sentence, answer)
    a = held.index(answer)
    b = a + len(answer)
    offsets = tok(sentence, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    span = [j for j, (s, e) in enumerate(offsets) if s < b and e > a]
    assert span and not sentence[offsets[span[0]][0] : a].strip() and offsets[span[-1]][1] == b, (row, sentence, answer, span)
    return span


def span_rows(name: str, tok: Any, end_id: int, said: list[str], answers: list[str], part: str) -> qc.Rows:
    """E109's answer-token probe: the sentence reader's 217 sequences (`sentence_seqs`: question, ASK, sentence `said[i]`)
    packed in rows of 256 as `real_rows`, reading only the response tokens of `part`: "span" the tokens of the answer
    `answers[i]` inside its sentence (`answer_span`), "frame" the other content tokens, "content" both; never the end
    token."""
    real, ids, read, first = tofu.real_rows(), [], [], []
    for i, (s, ans) in enumerate(zip(said, answers, strict=True)):
        q, p = tofu.seq(tok, end_id, real[i]["question"] + tofu.ASK, s)
        span, content = set(answer_span(tok, i, s, ans)), set(range(len(q) - p - 1))
        pick = {"span": span, "frame": content - span, "content": content}[part]
        ids.append(q)
        read.append([int(j - p in pick) for j in range(len(q))])
        first.append([int(j - p == min(pick)) for j in range(len(q))])
    return qc.rows_marked(name, ids, read, first, qc.ROW_LEN, qc.ROW_LEN)


def carrying(model: Any, R: qc.Rows, W: dict[str, torch.Tensor], batch: int, k: int) -> dict[str, Any]:
    """One state W on one row set: its gain and top-k carrying set (E98's ranking, as E102's `read_state`)."""
    r = QN.rank_state(model, R, W, batch)
    d = r["T1"] - r["T0"]
    return {"rows": R.name, "gain": qc.share(d, d, r["row_of"]), "carry": QN.top_set(r["score"].sum(0), k)}


# ----------------------------------------------------------------------------- E105's control and arm steps, shared


@dataclass(frozen=True)
class Numbers:
    """What an arm takes from E81 and E84 (E105, E107): the recipe (E84's rate, E81's warm-up and micro-batch, the arm's
    step cap, fp32 forward), E81's batch and read batch, and the two stream seeds whose first batches must differ."""

    recipe: Recipe
    batch: int
    read_batch: int
    seeds: tuple[int, int]


def c2n(lab: Any, R: qc.Rows, state: Path, stored: Path, batch: int, k: int) -> dict[str, Any]:
    """E105's C2n: a stored fp32 δW re-read on R reproduces its stored gain and set (tolerances lab.pred["C2n"])."""
    lab.fp32().eval()
    C, ref = lab.pred["C2n"], json.loads(stored.read_text())
    rec = carrying(lab.model, R, plus(lab.W0, load_file(state, DEV)), batch, k)
    rel, same = abs(rec["gain"]["gain"] / ref["gain"]["gain"] - 1), QN.overlap(rec["carry"], ref["carry"])
    return {"gain": rec["gain"]["gain"], "stored_gain": ref["gain"]["gain"], "rel": rel, "set_overlap": same,
            "holds": rel <= C["gain_rel"] and same >= C["min_set_overlap"]}


def w0_sentence(lab: Any, said: list[str], read_batch: int) -> tuple[list[list[float]], dict[str, Any]]:
    """W0's per-token log P on the sentence reader for `said` (all 217 rows), and whether its content level reproduces
    lab.pred["W0_sentence"] (relative tolerance). E105 stage C's second check."""
    ref = lab.pred["W0_sentence"]
    toks = tofu.token_logp(lab.model, sentence_seqs(lab.tok, lab.end_id, said), read_batch, DEV)
    level = levels(toks)["content"]
    rec = {"content": level, "stored": ref["stored"], "rel": abs(level / ref["stored"] - 1)}
    return toks, rec | {"holds": rec["rel"] <= ref["rel_tol"]}


def first_batches(lab: Any, label: str, seqs: list[tofu.Seq], N: Numbers, stop: Callable[..., None]) -> dict[str, float]:
    """Both seeds' first-batch loss on `seqs`; they must differ (E88's check), or the seed did not reach the stream."""
    F = lab.pred["first_minutes"]
    first = {str(s): gold_loss(lab.model, tofu.stream(seqs, N.batch, s, 1), N.recipe.micro, DEV) for s in N.seeds}
    gap = abs(first[str(N.seeds[0])] - first[str(N.seeds[1])])
    lab.log(f"[{label}] first batch losses {json.dumps(first)}, gap {gap:.6f} over {F['first_batch_differs']}")
    if not gap > F["first_batch_differs"]:
        stop(lab, f"{label}: the two seeds' first batches are the same", first)
    return first


def to_matched(lab: Any, label: str, seed: int, seqs: list[tofu.Seq], reads: list[tofu.Seq], w0: list[list[float]],
               N: Numbers, out: Path, stop: Callable[..., None], keep: str = "matched") -> dict[str, Any]:
    """E105's arm: N.recipe.steps steps of `seqs` at `seed`; C1 on `reads` (W0's tokens `w0`) against
    lab.pred["C1_content"]["bar"]; the template norm must be passed and the record's argmin be the matched step. Returns
    the training record ("train") and one δW in fp32, held on the card ("dW"): the matched one, or with keep="end" the
    last step's (E109, which trains to the five-epoch end and drops the matched one)."""
    assert keep in ("matched", "end"), keep
    lab.fp32().eval()  # loads the model when the control was done in an earlier job
    first = first_batches(lab, label, seqs, N, stop)
    held: dict[str, Any] = {}

    def put(which: str) -> Callable[[dict[str, torch.Tensor]], object]:
        return (lambda dW: held.update(dW=dW)) if which == keep else (lambda dW: None)

    run = Run(seed, out, stop, put("matched"), put("end"))
    rec = fine_tune(lab, run, seqs, N.recipe, N.batch, reads, w0, first[str(seed)], lab.pred["C1_content"]["bar"], N.read_batch)
    peak = max(s["dw_norm"] for s in rec["steps"])
    if not (peak > REF_NORM and rec["matched"]["step"] == rec["argmin_step"]):
        stop(lab, f"{label}: the template norm was not passed within {N.recipe.steps} steps", {"peak_norm": peak, "matched": rec["matched"]})
    return {"first_batch": first, "sequences": len(seqs), "steps_per_epoch": len(seqs) // N.batch, "train": rec, "dW": held["dW"]}


def read_sets(lab: Any, name: str, W: dict[str, torch.Tensor], R: dict[str, qc.Rows], folder: Path, batch: int,
              k: int) -> dict[str, Any]:
    """The state W's carrying set on each probe of R, each written to folder/<name>__<probe>.json as it is read."""
    folder.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for rows, Rr in R.items():
        rec = carrying(lab.model, Rr, W, batch, k)
        (folder / f"{name}__{rows}.json").write_text(json.dumps({"name": name, **rec}, default=float) + "\n")
        out[rows] = {"gain": rec["gain"], "n": len(rec["carry"])}
        lab.log(f"[{name} on {rows}] gain {rec['gain']['gain']:.5g}, set {len(rec['carry'])}; {lab.peaks()}")
    return out

"""The known-facts fine-tune of Qwen3-1.7B on TOFU's 217 real-fact rows (E103 Amendment 1's recipe: E84's five wordings,
pair 0 answered in one sentence), saved at the step nearest the template's norm (E91's rule), and its carrying-set read.
Shared by E103's runs (experiments/known_answer_pair/{run,amend1,amend2,neurons}.py, which ran with this code before it
moved here on 2026-11-06) and E105 (experiments/rows_null/run.py, which trains on a subset of the rows).

The callers pass the recipe's numbers (E81's batch, warm-up and micro-batch, E84's rate) and a `lab`: E45's job object
(`.model`, `.W0`, `.tok`, `.end_id`, `.pred`, `.log`, `.peaks`, `.fp32`)."""

from __future__ import annotations

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
from commute.qwen_sft import linears
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


def train_seqs(tok: Any, end_id: int, forms: Path, said: list[str], rows: Collection[int] | None = None) -> list[tofu.Seq]:
    """The 5 pairs of each real row in `rows` (None: all 217), in row order, pair 0's answer the row's sentence; the
    wordings are E103's results/forms.jsonl, every row kept. All rows: 1,085 sequences, as E103 Amendment 1 trained."""
    real, seqs = tofu.real_rows(), []
    for rec, s in zip(map(json.loads, forms.read_text().splitlines()), said, strict=True):
        assert rec["kept"]
        if rows is None or rec["row"] in rows:
            row = real[rec["row"]] | {"answer": s}
            seqs += [tofu.seq(tok, end_id, q, a) for q, a in tofu.form_pairs(row, rec["questions"], rec["answers"])]
    return seqs


def sentence_seqs(tok: Any, end_id: int, said: list[str], rows: Collection[int] | None = None) -> list[tofu.Seq]:
    """Amendment 1's sentence reader: each real question in `rows` with " Answer in one sentence." and its sentence."""
    real = tofu.real_rows()
    return [tofu.seq(tok, end_id, real[i]["question"] + tofu.ASK, said[i]) for i in range(len(real)) if rows is None or i in rows]


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


def carrying(model: Any, R: qc.Rows, W: dict[str, torch.Tensor], batch: int, k: int) -> dict[str, Any]:
    """One state W on one row set: its gain and top-k carrying set (E98's ranking, as E102's `read_state`)."""
    r = QN.rank_state(model, R, W, batch)
    d = r["T1"] - r["T0"]
    return {"rows": R.name, "gain": qc.share(d, d, r["row_of"]), "carry": QN.top_set(r["score"].sum(0), k)}

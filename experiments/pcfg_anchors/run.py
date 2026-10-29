"""E55 Stage A: Jain et al.'s PCFG Counter anchors. Plan: reports/plans/2026-10-29_plan_reuse-stage1.md (Revision 1);
filed definitions and rules: results/predictions.json.

Three minGPTs are pretrained on the five PCFG task families (src/commute/pcfg.py) with operand priors LOW (0.999,
0.001, 0), HIGH (0.5, 0.3, 0.2) and NEVER (0, 1, 0); behaviour is count-a and count-b accuracy on fresh C_Te = 0
Counter sequences (the answer token predicted at `=`), each beside its floors on the same samples: the majority-label
rate, for count-b the rate of answering the true count(a), the mean absolute error and the label histogram.

Stages (repo root, project venv, one GPU):
  a0  rehearsal at tiny sizes ($0, not a reading) -> results/a0.json.
  a1 --prior P  one 50K-iteration pretraining (Jain Table 6). Trains until --deadline-min, saves and exits 3; resubmit
      until exit 0. When it finishes, its behaviour on 4K fresh samples goes to results/a1_<P>.json.
  a1c  controls on the three pretrained models, no training of the model (src/commute/interchange.py, probe.py):
      patch P at every block for the positive references and nulls over 3 sample sets, l*_a and l*_b, the B nulls,
      the A2 bars by the filed rule, probe margins (3 seeds) and possession incl. PA4 -> results/a1c.json.
  a2 --arm NAME [NAME ...]  fine-tunes (Table 6's LOW grid, R on HIGH, second seeds of N1 and W1) for 10K
      iterations, states at 1K and 10K in results/large/a2/<arm>_<iters>.pt; each state's Table 6 behaviour and the
      checks B and P (count-a at l*_a, count-b at l*_b, from a1c.json), labelled with predictions.json's a2_bars
      ("unfiled" while they are not filed) -> results/a2_<arm>.json. Skips arms already written; exits 3 at the
      deadline (resubmit to resume).
Every stage refuses to start without results/predictions.json. Model initialisation is seeded with TRAIN_SEED.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from commute import pcfg
from commute.interchange import (
    N_B,
    Pairs,
    bar,
    behaviour_share,
    l_star,
    label,
    make_pairs,
    patch_shares,
    qualifying_texts,  # fmt: skip
)
from commute.mingpt import GPT, GPTConfig
from commute.pcfg import A, B
from commute.pcfg_train import (
    Batches,
    Run,
    Schedule,
    answer_loss,
    load_checkpoint,
    predict_answers,
    train,
)
from commute.probe import m_plus, margin_summary, possessed, train_probes

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED = RESULTS / "predictions.json"

LR, WEIGHT_DECAY = 1e-3, 0.1  # L1164; weight decay: minGPT trainer default
TRAIN_SEED, EVAL_SEED = 0, 1  # disjoint streams: evaluation samples are fresh; TRAIN_SEED also seeds the init
FT_SEED0 = 100  # fine-tuning data stream default_rng([FT_SEED0 + seed, chunk]) and dropout seed
EXIT_RESUME = 3
SETS = 3  # independent sample sets (A1 references) and probe seeds
CHECKS = ("B_a", "P_a", "B_b", "P_b")
FT_ITERS, FT_BATCH, FT_SAVES = 10_000, 96, (1000, 10_000)  # L1189-1192; states read at 1K and 10K (plan)


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the rehearsal and the scored pretraining."""

    iters: int
    batch: int
    eval_n: int  # fresh samples per behaviour number
    save_every: int
    log_every: int
    chunk: int  # batches drawn per generated chunk
    stop_first: int  # 0, or the step where the first call stops so the second resumes (the rehearsal)

    def __post_init__(self) -> None:
        assert self.iters % self.save_every == 0 and self.save_every % self.log_every == 0, self
        assert self.stop_first == 0 or (0 < self.stop_first < self.iters and self.stop_first % self.save_every == 0)
        assert self.eval_n >= 500 and self.batch > 0 and self.chunk > 0, self


A0 = Sizes(iters=500, batch=96, eval_n=1000, save_every=250, log_every=50, chunk=50, stop_first=250)
A1 = Sizes(iters=50_000, batch=96, eval_n=4000, save_every=1000, log_every=500, chunk=100, stop_first=0)


@dataclass(frozen=True)
class Arm:
    prior: str
    lr: float
    c_tr: float
    seed: int = 0


ARMS = {  # plan Stage A, "Fine-tunes (count b)"; seed-1 copies of N1 and W1 are PA4's reported FT-seed noise
    "N1": Arm("LOW", 1e-4, 0.0), "N2": Arm("LOW", 1e-4, 0.5), "N3": Arm("LOW", 1e-4, 0.8), "W2": Arm("LOW", 1e-4, 1.0),
    "M": Arm("LOW", 1e-5, 0.0), "M2": Arm("LOW", 1e-5, 0.5), "M3": Arm("LOW", 1e-5, 0.8), "W1": Arm("LOW", 1e-5, 1.0),
    "R": Arm("HIGH", 1e-5, 0.0), "N1s1": Arm("LOW", 1e-4, 0.0, 1), "W1s1": Arm("LOW", 1e-5, 1.0, 1),
}  # fmt: skip


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device() -> torch.device:
    assert torch.cuda.is_available(), "GPU only: never train the model locally"
    return torch.device("cuda")


def new_model() -> GPT:
    return GPT(GPTConfig(vocab=pcfg.VOCAB, block=pcfg.SEQ - 1)).to(device())


def load_model(path: Path) -> GPT:
    model = new_model()
    assert load_checkpoint(path, model) > 0, f"no trained state at {path}"
    return model.eval()


def pretrain(prior: str, sz: Sizes, ckpt: Path, *, stop_at: int | None = None,
             deadline: float | None = None) -> tuple[GPT, int, float]:  # fmt: skip
    """Pretrain (or resume) one prior's model; returns (model, step reached, ms/step)."""
    torch.manual_seed(TRAIN_SEED)  # a rerun gives the same anchor; a resume then overwrites the weights and RNG states
    model = new_model()
    p = pcfg.PRIORS[prior]
    batches = Batches(lambda rng, n: pcfg.pretrain_batch(rng, n, p), TRAIN_SEED, sz.batch, sz.chunk, device())
    run = Run(Schedule(sz.iters, LR), WEIGHT_DECAY, sz.save_every, sz.log_every)
    step, ms = train(model, batches, run, ckpt, stop_at=stop_at, deadline=deadline, log=lambda m: log(f"{prior} {m}"))
    return model, step, ms


def scored(model: GPT, seq: np.ndarray, op: int) -> dict[str, object]:
    """Accuracy at `=` with its floors on the same rows: majority label, mean absolute error, label histogram, and
    for count-b the score of answering the true count(a) (`floor_swap`)."""
    label_ = seq[:, pcfg.ANS].astype(np.int64) - pcfg.NUM0
    pred = predict_answers(model, seq).astype(np.int64) - pcfg.NUM0
    hist = np.bincount(label_)
    out: dict[str, object] = {"acc": float((pred == label_).mean()), "floor_majority": float(hist.max() / len(seq)),
                              "mae": float(np.abs(pred - label_).mean()),
                              "hist": {str(k): int(v) for k, v in enumerate(hist) if v}}  # fmt: skip
    if op == B:
        texts = seq[:, pcfg.TEXT0 : pcfg.TEXT0 + pcfg.TEXT].astype(np.int64) - pcfg.TERM0
        out["floor_swap"] = float((pcfg.count(texts, A) == label_).mean())
    return out


def eval_seqs(op: int, n: int, c_rate: float = 0.0) -> np.ndarray:
    """Fresh 'C <op> 40' rows; C_Te = 0 rows are the same for every model and stage (A1's behaviour samples)."""
    key = [EVAL_SEED, op] if c_rate == 0 else [EVAL_SEED, op, 1]
    return pcfg.task_batch(np.random.default_rng(key), n, "C", op, c_rate=c_rate)[0]


def behaviour(model: GPT, n: int) -> dict[str, object]:
    """A1's count-a and count-b accuracy on n fresh C_Te = 0 rows, with `scored`'s floors as count_<x>_<floor>."""
    out: dict[str, object] = {}
    for name, op in (("count_a", A), ("count_b", B)):
        row = scored(model, eval_seqs(op, n), op)
        out[name] = row.pop("acc")
        out |= {f"{name}_{k}": v for k, v in row.items()}
    return out


def generation_rate() -> dict[str, float]:
    """Sequences per second of one CPU thread: pretraining sequences, and Counter-b sequences at C_Tr = 1."""
    rng, n, out = np.random.default_rng(99), 9600, {}
    for name, make in (("pretrain_seq_per_s", lambda: pcfg.pretrain_batch(rng, n, pcfg.PRIORS["LOW"])),
                       ("spurious_seq_per_s", lambda: pcfg.task_batch(rng, n, "C", B, c_rate=1.0))):  # fmt: skip
        t = time.perf_counter()
        make()
        out[name] = n / (time.perf_counter() - t)
    for name, c_rate in (("natural", 0.0), ("spurious", 1.0)):  # count-b label means (equal since spurious matching)
        seq, _ = pcfg.task_batch(rng, n, "C", B, c_rate=c_rate)
        out[f"{name}_count_b_mean"] = float(seq[:, pcfg.ANS].mean() - pcfg.NUM0)
    return out


def stage_a0() -> int:
    out: dict[str, object] = {"generation": generation_rate()}
    log(f"generation: {out['generation']}")
    for prior in pcfg.PRIORS:
        ckpt = LARGE / "a0" / f"pt_{prior}.pt"
        ckpt.unlink(missing_ok=True)
        torch.cuda.reset_peak_memory_stats()
        _, s1, _ = pretrain(prior, A0, ckpt, stop_at=A0.stop_first)
        model, s2, ms = pretrain(prior, A0, ckpt)  # resumes from the file written at stop_first
        assert (s1, s2) == (A0.stop_first, A0.iters), (s1, s2)
        row = {"behaviour": behaviour(model, A0.eval_n), "ms_per_step": ms,
               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "checkpoint_mb": ckpt.stat().st_size / 1e6,
               "a1_minutes_at_this_rate": A1.iters * ms / 6e4, "init_seed": TRAIN_SEED}  # fmt: skip
        log(f"{prior}: {row}")
        out[prior] = row
    write_json(RESULTS / "a0.json", out)
    return 0


def stage_a1(prior: str, deadline_min: float) -> int:
    deadline = time.monotonic() + 60 * deadline_min
    path = RESULTS / f"a1_{prior}.json"
    if path.exists():
        log(f"{prior}: {path.name} exists; nothing to do")
        return 0
    model, step, _ = pretrain(prior, A1, LARGE / "a1" / f"pt_{prior}.pt", deadline=deadline)
    if step < A1.iters:
        log(f"{prior}: stopped at step {step} of {A1.iters}; resubmit to resume (exit {EXIT_RESUME})")
        return EXIT_RESUME
    row = behaviour(model, A1.eval_n) | {"iters": step, "eval_n": A1.eval_n, "init_seed": TRAIN_SEED}
    log(f"{prior} behaviour: {row}")
    write_json(path, row)
    return 0


def a1_row(prior: str) -> dict[str, float]:
    """results/a1_<prior>.json, refused unless it is the scored run (edge case E6)."""
    row = json.loads((RESULTS / f"a1_{prior}.json").read_text())
    assert row["iters"] == A1.iters and row["eval_n"] == A1.eval_n, f"a1_{prior}.json is not the scored run"
    return row


def patch_references(models: dict[str, GPT]) -> dict[str, list[list[float]]]:
    """P at every block, per sample set, for the four filed references, plus the pairs built per set (n_pairs_*)."""
    specs = {"P_a_positive": ("LOW", A, A), "P_a_null": ("NEVER", A, B),  # (model, f_C, prefix)
             "P_b_positive": ("HIGH", B, B), "P_b_null": ("LOW", B, A)}  # fmt: skip
    out: dict[str, list] = {k: [] for k in (*specs, "n_pairs_a", "n_pairs_b")}
    for i in range(SETS):
        pairs = {op: make_pairs(np.random.default_rng([EVAL_SEED, 30, i, op]), op) for op in (A, B)}
        for tag, op in (("a", A), ("b", B)):
            out[f"n_pairs_{tag}"].append(0 if pairs[op] is None else len(pairs[op].x))  # type: ignore[union-attr]
        for name, (prior, f, prefix) in specs.items():
            out[name].append(patch_shares(models[prior], pairs[f], prefix).tolist())
            log(f"set {i} {name} by block: {np.round(out[name][-1], 4).tolist()}")
    return out


def behaviour_nulls(models: dict[str, GPT]) -> dict[str, list[float]]:
    """B_a null (NEVER, count_a + 1) and B_b null (LOW, count_b) on 'C b 40', first N_B qualifying inputs per set."""
    out: dict[str, list[float]] = {"B_a_null": [], "B_b_null": []}
    for i in range(SETS):
        texts = qualifying_texts(np.random.default_rng([EVAL_SEED, 31, i]), N_B)
        out["B_a_null"].append(behaviour_share(models["NEVER"], texts, A))
        out["B_b_null"].append(behaviour_share(models["LOW"], texts, B))
    return out


def probe_readings(models: dict[str, GPT]) -> dict[str, object]:
    """Probe accuracies (3 seeds), margins per block, m_plus and possession (count-a: LOW, HIGH with NEVER's probe as
    a control; count-b: LOW (PA4), HIGH)."""
    raw = {op: [train_probes({p: models[p] for p in (("LOW", "HIGH", "NEVER") if op == A else ("LOW", "HIGH"))},
                             op, seed, EVAL_SEED) for seed in range(SETS)] for op in (A, B)}  # fmt: skip
    acc = {(p, op): np.array([r[p]["acc"] for r in raw[op]]) for op in (A, B) for p in raw[op][0]}
    shuf = {(p, op): np.array([r[p]["shuffled"] for r in raw[op]]) for op in (A, B) for p in raw[op][0]}
    summ = {f"{p}_count_{t}": margin_summary(acc[p, op], [shuf[p, op]] + ([acc["NEVER", A]] if op == A else []))
            for p in ("LOW", "HIGH") for t, op in (("a", A), ("b", B))}  # fmt: skip
    plus = m_plus(summ["LOW_count_a"])
    poss = {k: possessed(v, plus) for k, v in summ.items()}
    return {"raw": {f"count_{'a' if op == A else 'b'}": raw[op] for op in (A, B)}, "margins": summ, "m_plus": plus,
            "possessed": poss, "PA4_LOW_possesses_count_b": poss["LOW_count_b"]}  # fmt: skip


def behaviour_rows(rows: dict[str, dict[str, float]]) -> dict[str, bool]:
    """PA1's five thresholded behaviour cells (inclusive; a miss stops: recipe), from a1_<prior>.json."""
    low, high = rows["LOW"], rows["HIGH"]
    return {"LOW_count_a_ge_0.95": low["count_a"] >= 0.95, "HIGH_count_a_ge_0.95": high["count_a"] >= 0.95,
            "LOW_count_b_le_swap_plus_0.10": low["count_b"] <= low["count_b_floor_swap"] + 0.10,
            "HIGH_count_b_ge_0.95": high["count_b"] >= 0.95, "NEVER_count_b_ge_0.95": rows["NEVER"]["count_b"] >= 0.95}  # fmt: skip


def pa1_rows(patch: dict, b_null: dict, probes: dict, la: int, lb: int) -> dict[str, bool]:  # type: ignore[type-arg]
    """PA1's probe and patch rows (inclusive), P and B read at l*."""
    pos = lambda k, blk: float(np.mean(np.array(patch[k])[:, blk - 1]))
    mp, mg = probes["m_plus"], probes["margins"]
    return {"LOW_count_a_m_plus_ge_6s": not mp["void"],
            "HIGH_count_a_ge_half_m_plus": max(mg["HIGH_count_a"]["m"]) >= 0.5 * mp["m_plus"],
            "HIGH_count_b_ge_half_m_plus": max(mg["HIGH_count_b"]["m"]) >= 0.5 * mp["m_plus"],
            "P_a_positive_ge_0.8": pos("P_a_positive", la) >= 0.8, "P_b_positive_ge_0.8": pos("P_b_positive", lb) >= 0.8,
            "P_a_null_le_0.05": pos("P_a_null", la) <= 0.05, "P_b_null_le_0.05": pos("P_b_null", lb) <= 0.05,
            "B_a_null_le_0.05": float(np.mean(b_null["B_a_null"])) <= 0.05}  # fmt: skip


def stage_a1c() -> int:
    rows = {p: a1_row(p) for p in pcfg.PRIORS}
    models = {p: load_model(LARGE / "a1" / f"pt_{p}.pt") for p in pcfg.PRIORS}
    patch, b_null = patch_references(models), behaviour_nulls(models)
    log(f"B nulls by set: {b_null}")
    la, lb = l_star(np.array(patch["P_a_positive"])), l_star(np.array(patch["P_b_positive"]))
    at = lambda k, blk: np.array(patch[k])[:, blk - 1]
    bars = {"B_a": bar(np.array(b_null["B_a_null"]), rows["LOW"]["count_a"]),
            "P_a": bar(at("P_a_null", la), float(at("P_a_positive", la).mean())),
            "B_b": bar(np.array(b_null["B_b_null"]), rows["HIGH"]["count_b"]),
            "P_b": bar(at("P_b_null", lb), float(at("P_b_positive", lb).mean()))}  # fmt: skip
    log(f"l*_a {la}, l*_b {lb}; bars {bars}")
    probes = probe_readings(models)
    log(f"probe margins {probes['margins']}; m_plus {probes['m_plus']}; possessed {probes['possessed']}")
    beh = behaviour_rows(rows)
    pa1 = beh | pa1_rows(patch, b_null, probes, la, lb)
    stops = [f"{k} bar void (hi <= lo)" for k, v in bars.items() if v["void"]]
    stops += [f"behaviour {k} missed (recipe)" for k, v in beh.items() if not v]
    stops += ["m_plus < 6 s_plus (probe)"] * (not pa1["LOW_count_a_m_plus_ge_6s"])
    stops += ["HIGH count-b probe margin < 0.5 m_plus (probe)"] * (not pa1["HIGH_count_b_ge_half_m_plus"])
    log(f"PA1 probe/patch rows {pa1}")
    for s in stops:
        log(f"STOP: {s}")
    a2_bars = {"l_star_a": la, "l_star_b": lb, **bars}
    write_json(RESULTS / "a1c.json", {"l_star_a": la, "l_star_b": lb, "patch": patch, "behaviour_nulls": b_null,
                                      "bars": bars, "a2_bars": a2_bars, "probes": probes, "PA1_rows": pa1,
                                      "stops": stops, "sets": SETS})  # fmt: skip
    return 0


def filed_bars(a1c: dict) -> dict | None:  # type: ignore[type-arg]
    """predictions.json's a2_bars when every check has lo and hi; None while it is the placeholder."""
    filed = json.loads(PRED.read_text())["a2_bars"]
    if not all(isinstance(filed.get(k), dict) and {"lo", "hi", "void"} <= filed[k].keys() for k in CHECKS):
        return None
    for k in ("l_star_a", "l_star_b"):
        assert filed.get(k, a1c[k]) == a1c[k], f"predictions.json {k} != a1c.json {k}"
    return filed


@dataclass(frozen=True)
class A2Eval:
    """The evaluation data every A2 state reads (same rows for every arm)."""

    seqs: dict[str, tuple[np.ndarray, int]]  # Table 6 behaviour sets: name -> (rows, op)
    b_texts: np.ndarray  # first N_B qualifying texts, for B
    pairs: dict[int, Pairs | None]  # f_C -> pairs, for P


def a2_eval_data() -> A2Eval:
    seqs = {"count_b_te0": (eval_seqs(B, A1.eval_n), B), "count_b_te1": (eval_seqs(B, A1.eval_n, 1.0), B),
            "count_a_te0": (eval_seqs(A, A1.eval_n), A)}  # fmt: skip
    pairs = {op: make_pairs(np.random.default_rng([EVAL_SEED, 50, op]), op) for op in (A, B)}
    return A2Eval(seqs, qualifying_texts(np.random.default_rng([EVAL_SEED, 51]), N_B), pairs)


def read_state(model: GPT, data: A2Eval, a1c: dict, bars: dict | None) -> dict[str, object]:  # type: ignore[type-arg]
    """Table 6 behaviour with floors, and per f_C: B, P at every block and at l*, the pairs built, the label."""
    out: dict[str, object] = {k: scored(model, seq, op) for k, (seq, op) in data.seqs.items()}
    for t, op in (("a", A), ("b", B)):
        l_, p_all = a1c[f"l_star_{t}"], patch_shares(model, data.pairs[op], B)  # 'C b 40': the arm's own task
        b_, p_ = behaviour_share(model, data.b_texts, op), float(p_all[l_ - 1])
        lab = "unfiled" if bars is None else label(b_, p_, bars[f"B_{t}"], bars[f"P_{t}"])
        n_pairs = 0 if data.pairs[op] is None else len(data.pairs[op].x)  # type: ignore[union-attr]
        out[f"check_count_{t}"] = {"B": b_, "P": p_, "l_star": l_, "P_by_block": p_all.tolist(), "n_pairs": n_pairs,
                             "label": lab}  # fmt: skip
    return out


def state_path(name: str, it: int) -> Path:
    return LARGE / "a2" / f"{name}_{it}.pt"


def finetune(name: str, arm: Arm, deadline: float) -> bool:
    """Fine-tune `arm` from its A1 checkpoint (answer-only loss, Table 6 recipe), saving the model at each FT_SAVES
    step; resumes from results/large/a2/<name>_train.pt. False when the deadline stopped it."""
    model = load_model(LARGE / "a1" / f"pt_{arm.prior}.pt")
    torch.manual_seed(FT_SEED0 + arm.seed)
    run = Run(Schedule(FT_ITERS, arm.lr), WEIGHT_DECAY, save_every=1000, log_every=500, loss=answer_loss)
    for it in FT_SAVES:
        if state_path(name, it).exists():
            continue
        make = lambda rng, n: pcfg.task_batch(rng, n, "C", B, c_rate=arm.c_tr)[0]
        batches = Batches(make, FT_SEED0 + arm.seed, FT_BATCH, 100, device())
        step, _ = train(model, batches, run, LARGE / "a2" / f"{name}_train.pt", stop_at=it, deadline=deadline,
                        log=lambda m: log(f"{name} {m}"))  # fmt: skip
        if step < it:
            return False
        tmp = state_path(name, it).with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "step": step}, tmp)  # fp32: 12 MB per state
        tmp.replace(state_path(name, it))
    return True


def stage_a2(names: list[str], deadline_min: float) -> int:
    deadline = time.monotonic() + 60 * deadline_min
    a1c = json.loads((RESULTS / "a1c.json").read_text())
    bars = filed_bars(a1c)
    log(f"a2 bars: {'from predictions.json' if bars else 'NOT FILED: labels will read unfiled'}")
    data = a2_eval_data()
    for name in names:
        path = RESULTS / f"a2_{name}.json"
        if path.exists():
            log(f"{name}: {path.name} exists; skipped")
            continue
        if time.monotonic() > deadline or not finetune(name, ARMS[name], deadline):
            log(f"{name}: deadline; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        row = {"arm": asdict(ARMS[name]), "iters": FT_ITERS, "batch": FT_BATCH, "bars_filed": bars is not None,
               "states": {str(it): read_state(load_model(state_path(name, it)), data, a1c, bars) for it in FT_SAVES}}  # fmt: skip
        log(f"{name}: {row}")
        write_json(path, row)
        (LARGE / "a2" / f"{name}_train.pt").unlink(missing_ok=True)
    return 0


def write_json(path: Path, obj: object) -> None:
    """Atomic: a reader sees the old file or the whole new one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)
    log(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("a0", "a1", "a1c", "a2"))
    ap.add_argument("--prior", choices=list(pcfg.PRIORS), help="a1: the one prior this job pretrains")
    ap.add_argument("--arm", nargs="+", choices=list(ARMS), help="a2: the arms this job fine-tunes, in order")
    ap.add_argument("--deadline-min", type=float, default=25.0, help="a1, a2: save and exit 3 after this many minutes")
    args = ap.parse_args()
    if not PRED.exists():
        sys.exit(f"file the predictions first: {PRED}")
    if (args.stage == "a1") != (args.prior is not None) or (args.stage == "a2") != (args.arm is not None):
        ap.error("--prior goes with a1 only (and a1 needs it); --arm with a2 only (and a2 needs it)")
    stages = {"a0": stage_a0, "a1": lambda: stage_a1(args.prior, args.deadline_min), "a1c": stage_a1c,
              "a2": lambda: stage_a2(args.arm, args.deadline_min)}  # fmt: skip
    sys.exit(stages[args.stage]())


if __name__ == "__main__":
    main()

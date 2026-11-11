"""E160 (cell P3.5, run 1): capacity checks of three built Qwen3-1.7B pairs, the decomposition's masked error on each
task's own rows, and one rehearsal fine-tune. Plan: reports/plans/2026-11-11_plan_p35-qwen-pair-library.md; the data
is `commute.qwen_pairs`; rules in results/predictions.json (the run refuses to start without it, or while
results/stopped.json exists). One stage per job:

  capacity  W0, fp32. Per side: greedy exact match on the held-out items under the capacity prompt, and the prior, the
            mean log-probability per answer token (end token included) of the trained answer under the cue prompt, on
            trained and held-out items. Each pair's fit by the filed rule. results/capacity.json (+ capacity_answers.json).
  floors    the masked error at W0, K0 = KL(W0 clean ‖ W0 masked) per position (`qwen_d3.cells`, change None) on
            p-9707d6ff, on each task's own rows: maths (the control, first; a miss stops), template, gold, the six built
            sides. results/floors.json.
  rehearse  (Amendment 1, results/predictions_amend1.json, which it refuses to start without) one side of copy/cipher,
            seed 0, one job per side: `pool_sft.RECIPE` at peak 2e-6 through `qwen_train.train`; the output change
            KL(W0 ‖ W_t) on the side's own rows read every step; fp32 δW saved with the bracket at 5x and 15x its floor;
            training ends right after the 15x read (else at step 100, saved). Control first: a zero change reads KL 0
            and W0's held-out exact match. Then each saved state's step, held-out exact match under the cue, |δW| and
            KL. results/rehearse_<side>.json; states in results/large/dW/<side>_s0/. A stop writes
            results/stopped_<side>.json (the other stages: results/stopped.json).

Every scored number is printed to the log (results/run.log) as well. Each stage prints "STAGE <name> DONE" at its end.
Exit 0 done, 1 a stop or a refusal.

  python experiments/qwen_pair_library/run.py capacity
  python experiments/qwen_pair_library/run.py floors --decomp "$COMMUTE_QWEN_NEW_DIR"
  python experiments/qwen_pair_library/run.py rehearse --side copy     (and --side cipher, in parallel)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor
from transformers import AutoTokenizer

from commute import qwen_d3 as d3
from commute import qwen_library as L
from commute import qwen_pairs as P
from commute import qwen_sets
from commute.jobs import hard_exit
from commute.pool_sft import RECIPE
from commute.qwen_eval import MODEL, MODEL_REV, chat_prompt
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import gold_loss, linears, load, mean_kl, plus, save, set_weights
from commute.qwen_train import Recipe, train
from commute.qwen_updates import cycle
from commute.qwen_vpd import decomp_id, fp32_matmuls

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED, STOPPED, WORDS = RESULTS / "predictions.json", RESULTS / "stopped.json", RESULTS / "words.json"
AMEND1 = RESULTS / "predictions_amend1.json"
REHEARSE_SIDES = ("copy", "cipher")
DEV, F32, GIB = "cuda", torch.float32, 2**30
DECOMP = "p-9707d6ff"


@dataclass(frozen=True)
class Config:
    shots: int = 3  # worked examples in the capacity prompt
    max_new: int = 32  # new tokens per answer: the longest answer ("The answer is Addis Ababa.") is ~8
    gen_batch: int = 128
    loss_batch: int = 32
    kl_batch: int = 8  # own rows per forward when the output change is read (fp32 logits 1.2 GB each beside training)
    batch: int = 64  # sequences a step (E61's gold_1e-5, as pool_sft.BATCH)
    seed: int = 0  # the rehearsal's seed
    grid: tuple[float, float] = (5.0, 15.0)  # save sizes, multiples of the side's floor

    def __post_init__(self) -> None:
        assert self.batch % RECIPE.micro == 0 and self.grid == tuple(sorted(self.grid))


C = Config()


class Lab:
    def __init__(self, stage: str, side: str | None = None):
        self.stage, self.side, self.t0 = stage, side or "", time.time()
        self.P = json.loads(PRED.read_text())
        self.A: dict[str, Any] = json.loads(AMEND1.read_text()) if side else {}
        self.stopped = RESULTS / f"stopped_{side}.json" if side else STOPPED
        self.out_file = RESULTS / f"rehearse_{side}.json"
        self.dw = LARGE / "dW" / f"{side}_s{C.seed}"
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        self.tok.padding_side = "left"
        self.end_id = int(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.tasks = P.tasks(WORDS)
        self.log(f"[start {stage} {self.side}] config {json.dumps(asdict(C))}; {torch.cuda.get_device_name()}")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")

    def stop(self, name: str, detail: dict[str, Any]) -> NoReturn:
        write(self.stopped, {"stage": self.stage, "side": self.side, "stopped": name, **detail})
        self.log(f"STOP: {name} {json.dumps(detail, default=float)}")
        sys.exit(1)

    def done(self, path: Path) -> None:
        peak = torch.cuda.max_memory_allocated() / GIB
        self.log(f"STAGE {self.stage} DONE {path.relative_to(HERE)}; peak GPU {peak:.1f} GiB, {d3.peaks()}")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path: Path, obj: Any) -> None:
    (tmp := path.with_name(path.name + ".tmp")).write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def fp32_model(lab: Lab) -> Any:
    return L.fp32_model(lab.tok, DEV)


# ----------------------------------------------------------------------------- readings


def exact_match(lab: Lab, model: Any, side: str, texts: list[str], items: list[P.Item]) -> tuple[int, list[str]]:
    """Greedy answers to the chat texts; how many match their item's answer (`qwen_pairs.matches`), and the answers."""
    return L.exact_hits(lab.tok, model, side, texts, items, C.gen_batch, C.max_new)


def cue_texts(lab: Lab, items: list[P.Item]) -> list[str]:
    return [chat_prompt(lab.tok, P.cue_user(it)) for it in items]


def prior(lab: Lab, model: Any, items: list[P.Item]) -> float:
    """Mean log-probability per answer token (end token included), pooled over the items, under the cue prompt."""
    return -gold_loss(model, P.cue_seqs(lab.tok, lab.end_id, items), C.loss_batch, DEV)


def fit(R: dict[str, Any], reuse: dict[str, int], new: dict[str, int]) -> dict[str, Any]:
    """The filed capacity rule on counts: reuse ≥ ceil(reuse_min · n), new ≤ floor(new_max · n); ties at a bar pass.
    An empty side is "unscored", never fit."""
    if not reuse["n"] or not new["n"]:
        return {"verdict": "unscored", "reason": "an empty side"}
    r_ok = reuse["hits"] >= math.ceil(R["reuse_min"] * reuse["n"] - 1e-9)
    n_ok = new["hits"] <= math.floor(R["new_max"] * new["n"] + 1e-9)
    return {"reuse_ok": r_ok, "new_ok": n_ok, "verdict": "fit" if r_ok and n_ok else "dropped"}


# ----------------------------------------------------------------------------- stage capacity


def stage_capacity(lab: Lab) -> None:
    path, model = RESULTS / "capacity.json", fp32_model(lab)
    out, answers = read(path) or {"sides": {}}, read(RESULTS / "capacity_answers.json")
    for side in P.SIDES:
        if side in out["sides"]:
            continue
        task = lab.tasks[side]
        texts = [chat_prompt(lab.tok, P.capacity_text(side, it, task["train"], C.shots)) for it in task["held"]]
        hits, said = exact_match(lab, model, side, texts, task["held"])
        rec = {"n": len(said), "hits": hits, "exact_match": hits / len(said) if said else None,
               "prior_train": prior(lab, model, task["train"]), "prior_held": prior(lab, model, task["held"])}
        out["sides"][side], answers[side] = rec, said
        write(RESULTS / "capacity_answers.json", answers)
        write(path, out)
        lab.log(f"[capacity] {side} {json.dumps(rec)}; first answers {json.dumps(said[:4])}")
    R = lab.P["capacity"]
    out["pairs"] = {pair: fit(R, out["sides"][a], out["sides"][b]) for pair, (a, b) in P.PAIRS.items()}
    write(path, out)
    lab.log(f"[capacity] pairs {json.dumps(out['pairs'])}")
    lab.done(path)


# ----------------------------------------------------------------------------- stage floors


def row_sets(lab: Lab) -> dict[str, d3.RowSet]:
    """The control first, then template (E92's 64 a1 rows: all it has), gold, and the six built sides' own rows."""
    sets = {"maths": d3.maths_rows(), "template": d3.own_rows("a1", lab.tok, qwen_sets.load()), "gold": P.gold_rows(lab.tok)}
    return sets | {s: P.own_rows(lab.tok, lab.end_id, s, lab.tasks[s]["train"]) for s in P.SIDES}


def stage_floors(lab: Lab, decomp: Path) -> None:
    path = RESULTS / "floors.json"
    if decomp_id(decomp) != DECOMP:
        lab.stop("wrong decomposition", {"decomp": str(decomp), "want": DECOMP})
    out, R = read(path) or {"decomp": DECOMP, "floors": {}}, lab.P["floors"]
    d = d3.Decomp(decomp, lab.log)
    d.weights.put("FT", None)
    with torch.no_grad():
        for name, rs in row_sets(lab).items():
            if name in out["floors"]:
                continue
            t0, sums = time.time(), d3.cells(d, rs)
            n = rs.tokens.numel()
            rec = {"rows": len(rs.tokens), "positions": n, "K0": d3.total(sums, "K0") / n,
                   "K00": d3.total(sums, "K00") / n, "seconds": time.time() - t0}
            out["floors"][name] = rec
            write(path, out)
            lab.log(f"[floors] {name} {json.dumps(rec)}")
            if name == "maths" and not abs(rec["K0"] - R["maths_K0"]) <= R["maths_tol"]:
                lab.stop("maths control", {"K0": rec["K0"], "filed": R["maths_K0"], "tol": R["maths_tol"]})
    lab.done(path)


# ----------------------------------------------------------------------------- stage rehearse (Amendment 1)


class Reached15x(Exception):
    """Raised after the read step that first reaches the largest save size, its bracket written: training ends there."""


def recipe_of(A: dict[str, Any]) -> Recipe:
    """pool_sft.RECIPE with Amendment 1's peak rate; every other field asserted equal to the filed recipe."""
    r = replace(RECIPE, lr=A["recipe"]["lr"])
    assert all(A["recipe"][k] == v for k, v in asdict(r).items()) and A["recipe"]["batch"] == C.batch, (r, A["recipe"])
    return r


def reads_at(lab: Lab, model: Any, W0: dict[str, Tensor], W: dict[str, Tensor] | None, rows: list[Tensor]
             ) -> dict[str, Any]:
    """At weights W (None: the model as it stands): held-out exact match under the cue, and KL(W0 ‖ W) on the own rows."""
    held = lab.tasks[lab.side]["held"]
    with set_weights(model, W):
        hits, said = exact_match(lab, model, lab.side, cue_texts(lab, held), held)
    return {"hits": hits, "n": len(said), "exact_match": hits / len(said), "kl": mean_kl(model, rows, W0, W),
            "first_answers": said[:4]}


def control(lab: Lab, model: Any, W0: dict[str, Tensor], rows: list[Tensor], out: dict[str, Any]) -> None:
    """The zero change through the saved-state path: KL 0 and exactly W0's held-out exact match."""
    w0 = reads_at(lab, model, W0, None, rows)
    zero = reads_at(lab, model, W0, plus(W0, {s: torch.zeros_like(w) for s, w in W0.items()}), rows)
    out["control"] = {"w0": w0, "zero_step": zero}
    write(lab.out_file, out)
    lab.log(f"[control] {lab.side} W0 {json.dumps(w0)}; zero step {json.dumps(zero)}")
    R = lab.P["rehearse"]
    if not (abs(zero["kl"]) <= R["kl0_max"] and zero["hits"] == w0["hits"]):
        lab.stop("control", {"zero_kl": zero["kl"], "zero_hits": zero["hits"], "w0_hits": w0["hits"]})


def early(lab: Lab, out: dict[str, Any], rec: dict[str, Any], floor: float) -> None:
    """Amendment 1's stops: a non-finite loss; a step 1 that moved (it runs at rate 0); KL at step 2 above
    kl_step2_max_floors x the floor; a single step raising KL by more than kl_jump_max_floors x the floor."""
    bad, rise = L.early_stop(out["steps"], floor, lab.A["early_stops"])
    if bad:
        write(lab.out_file, out)
        lab.stop(bad, {"step": rec["step"], "loss": rec["loss"], "dw_norm": rec["dw_norm"], "kl": rec["kl"]["own"],
                       "rise": rise, "floor": floor})


def fine_tune(lab: Lab, model: Any, W0: dict[str, Tensor], rows: list[Tensor], out: dict[str, Any]) -> None:
    """Training to the first read at or past the largest save size (its bracket saved), or to the last step."""
    r, grid, floor = recipe_of(lab.A), out["grid"], out["floor"]
    seqs = P.cue_seqs(lab.tok, lab.end_id, lab.tasks[lab.side]["train"])
    stream = [seqs[i] for i in cycle(len(seqs), C.seed, r.steps * C.batch)]

    def save_dw(t: int, dW: dict[str, Tensor]) -> None:
        save(lab.dw / f"step_{t}.safetensors", dW, F32, droppable=False, log=lab.log)
        out["saves"] = sorted(set(out["saves"]) | {t})

    def on_step(rec: dict[str, Any]) -> None:
        rec.pop("norms")
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / GIB
        out["steps"].append(rec)
        write(lab.out_file, out)
        lab.log(f"[train] {lab.side} {json.dumps(rec)}")
        early(lab, out, rec, floor)
        if any(c["grid"] == grid[-1] for c in rec["crossed"]):
            raise Reached15x

    t0 = time.time()
    try:
        train(model, W0, lambda t: stream[(t - 1) * C.batch : t * C.batch], r, {"own": rows}, set(range(1, r.steps + 1)),
              {"own": grid}, {r.steps}, save_dw, on_step, lab.dw / "ckpt.pt", r.steps, dW_of=delta_fp32)
    except Reached15x:
        lab.log(f"[train] {lab.side} reached 15x at step {out['steps'][-1]['step']}; training ends")
    out["train_seconds"] = time.time() - t0


def at_15x(out: dict[str, Any]) -> tuple[int, bool]:
    """The step the 15x rule reads: the first read at or past the largest save size; else the last step (False)."""
    hit = [c["hi"] for s in out["steps"] for c in s.get("crossed", []) if c["grid"] == out["grid"][-1]]
    return (hit[0], True) if hit else (out["steps"][-1]["step"], False)


def read_saves(lab: Lab, model: Any, W0: dict[str, Tensor], rows: list[Tensor], out: dict[str, Any]) -> None:
    """Per saved state: its step, |δW| (fp64 over the file), held-out exact match under the cue and KL on the own rows."""
    out["states"] = {}
    for t in out["saves"]:
        dW = load(lab.dw / f"step_{t}.safetensors", DEV)
        norm = math.sqrt(sum(float(x.double().square().sum()) for x in dW.values()))
        out["states"][str(t)] = {"step": t, "dw_norm": norm} | reads_at(lab, model, W0, plus(W0, dW), rows)
        del dW
        write(lab.out_file, out)
        lab.log(f"[state] {lab.side} step {t} {json.dumps(out['states'][str(t)])}")


def stage_rehearse(lab: Lab) -> None:
    assert LARGE.is_dir(), f"{LARGE}: link it (fleet/link-data) or, on a pod, replace the dangling symlink with a folder"
    floor = read(RESULTS / "floors.json").get("floors", {}).get(lab.side, {}).get("K0")
    if floor is None or floor != lab.A["sides"][lab.side]["floor"]:
        lab.stop("floor missing or not the filed one", {"side": lab.side, "floor": floor})
    model = fp32_model(lab)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    own = P.own_rows(lab.tok, lab.end_id, lab.side, lab.tasks[lab.side]["train"]).tokens
    rows = list(own.to(DEV).split(C.kl_batch))
    out = {"side": lab.side, "seed": C.seed, "recipe": asdict(recipe_of(lab.A)), "batch": C.batch, "floor": floor,
           "grid": [m * floor for m in C.grid], "steps": [], "saves": []}
    control(lab, model, W0, rows, out)
    fine_tune(lab, model, W0, rows, out)
    read_saves(lab, model, W0, rows, out)
    step, reached = at_15x(out)
    em, bar = out["states"][str(step)]["exact_match"], lab.P["rehearse"]["em_15x_min"] if lab.side == "copy" else None
    out["rule"] = {"step": step, "reached_15x": reached, "exact_match": em, "bar": bar,
                   "holds": None if bar is None else em >= bar}
    write(lab.out_file, out)
    lab.log(f"[rehearse] {lab.side} rule {json.dumps(out['rule'])}")
    lab.done(lab.out_file)


def main() -> None:
    ap = argparse.ArgumentParser(description="E160: capacity, floors or rehearse")
    ap.add_argument("stage", choices=("capacity", "floors", "rehearse"))
    ap.add_argument("--decomp", type=Path, help="the p-9707d6ff folder (floors only)")
    ap.add_argument("--side", choices=REHEARSE_SIDES, help="rehearse only: the side trained")
    a = ap.parse_args()
    for f in (PRED, AMEND1) if a.stage == "rehearse" else (PRED,):
        if not f.is_file():
            sys.exit(f"{f} is missing: predictions are filed before any run")
    if a.stage == "rehearse" and a.side is None:
        sys.exit("rehearse needs --side copy|cipher")
    side = a.side if a.stage == "rehearse" else None
    for f in (STOPPED, *([RESULTS / f"stopped_{side}.json"] if side else [])):
        if f.is_file():
            sys.exit(f"{f} exists: read it, then move it aside")
    lab = Lab(a.stage, side)
    if a.stage == "capacity":
        stage_capacity(lab)
    elif a.stage == "floors":
        stage_floors(lab, a.decomp)
    else:
        stage_rehearse(lab)
    hard_exit(0)


if __name__ == "__main__":
    main()

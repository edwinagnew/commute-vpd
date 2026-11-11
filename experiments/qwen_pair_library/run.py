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
  rehearse  the copy side, seed 0, `pool_sft.RECIPE` through `qwen_train.train`; the output change KL(W0 ‖ W_t) on its
            own rows read every step; fp32 δW saved with the bracket at 5x and 15x its floor and at the last step.
            Control first: a zero change reads KL 0 and W0's held-out exact match. Then each saved state's held-out exact
            match under the cue, |δW| and KL. results/rehearse.json; states in results/large/dW/copy_s0/.

Every scored number is printed to the log (results/run.log) as well. Each stage prints "STAGE <name> DONE" at its end.
Exit 0 done, 1 a stop or a refusal.

  python experiments/qwen_pair_library/run.py capacity
  python experiments/qwen_pair_library/run.py floors --decomp "$COMMUTE_QWEN_NEW_DIR"
  python experiments/qwen_pair_library/run.py rehearse
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor
from transformers import AutoTokenizer, GenerationConfig

from commute import qwen_d3 as d3
from commute import qwen_pairs as P
from commute import qwen_sets
from commute.jobs import hard_exit
from commute.pool_sft import RECIPE
from commute.qwen_eval import GREEDY, MODEL, MODEL_REV, chat_prompt, generate
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import gold_loss, linears, load, mean_kl, plus, save, set_weights
from commute.qwen_train import train
from commute.qwen_updates import cycle, ok
from commute.qwen_vpd import decomp_id, fp32_matmuls, load_target

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED, STOPPED, WORDS = RESULTS / "predictions.json", RESULTS / "stopped.json", RESULTS / "words.json"
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
    side: str = "copy"  # the rehearsal's side and seed
    seed: int = 0
    grid: tuple[float, float] = (5.0, 15.0)  # save sizes, multiples of the side's floor

    def __post_init__(self) -> None:
        assert self.batch % RECIPE.micro == 0 and self.side in P.SIDES and self.grid == tuple(sorted(self.grid))


C = Config()


class Lab:
    def __init__(self, stage: str):
        self.stage, self.t0 = stage, time.time()
        self.P = json.loads(PRED.read_text())
        fp32_matmuls()
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        self.tok.padding_side = "left"
        self.end_id = int(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.tasks = P.tasks(WORDS)
        self.log(f"[start {stage}] config {json.dumps(asdict(C))}; {torch.cuda.get_device_name()}")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")

    def stop(self, name: str, detail: dict[str, Any]) -> NoReturn:
        write(STOPPED, {"stage": self.stage, "stopped": name, **detail})
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
    m = load_target(DEV, F32)
    m.generation_config = GenerationConfig(eos_token_id=m.generation_config.eos_token_id, pad_token_id=lab.tok.pad_token_id)
    return m


# ----------------------------------------------------------------------------- readings


@torch.no_grad()
def exact_match(lab: Lab, model: Any, side: str, texts: list[str], items: list[P.Item]) -> tuple[int, list[str]]:
    """Greedy answers to the chat texts; how many match their item's answer (`qwen_pairs.matches`), and the answers."""
    out: list[str] = []
    for s in range(0, len(texts), C.gen_batch):
        out += [r["response"] for r in generate(lab.tok, model, texts[s : s + C.gen_batch], 1, GREEDY, max_new=C.max_new)]
    return sum(P.matches(side, r, it.answer) for r, it in zip(out, items, strict=True)), out


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


# ----------------------------------------------------------------------------- stage rehearse


def reads_at(lab: Lab, model: Any, W0: dict[str, Tensor], W: dict[str, Tensor] | None, rows: list[Tensor]
             ) -> dict[str, Any]:
    """At weights W (None: the model as it stands): held-out exact match under the cue, and KL(W0 ‖ W) on the own rows."""
    held = lab.tasks[C.side]["held"]
    with set_weights(model, W):
        hits, said = exact_match(lab, model, C.side, cue_texts(lab, held), held)
    return {"hits": hits, "n": len(said), "exact_match": hits / len(said), "kl": mean_kl(model, rows, W0, W),
            "first_answers": said[:4]}


def control(lab: Lab, model: Any, W0: dict[str, Tensor], rows: list[Tensor], out: dict[str, Any]) -> None:
    """The zero change through the saved-state path: KL 0 and exactly W0's held-out exact match."""
    w0 = reads_at(lab, model, W0, None, rows)
    zero = reads_at(lab, model, W0, plus(W0, {s: torch.zeros_like(w) for s, w in W0.items()}), rows)
    out["control"] = {"w0": w0, "zero_step": zero}
    write(RESULTS / "rehearse.json", out)
    lab.log(f"[control] W0 {json.dumps(w0)}; zero step {json.dumps(zero)}")
    R = lab.P["rehearse"]
    if not (abs(zero["kl"]) <= R["kl0_max"] and zero["hits"] == w0["hits"]):
        lab.stop("control", {"zero_kl": zero["kl"], "zero_hits": zero["hits"], "w0_hits": w0["hits"]})


def early(lab: Lab, out: dict[str, Any], rec: dict[str, Any], top: float) -> None:
    """Stops in the first minutes: a non-finite loss; a step 1 that moved (it runs at rate 0); KL at step 10 past the
    largest save size."""
    t, kl = rec["step"], rec["kl"]["own"]
    bad = ("loss not finite" if not ok(rec["loss"]) else "step 1 moved" if t == 1 and rec["dw_norm"] != 0
           else f"KL at step 10 above {top:.4g}" if t == 10 and not kl <= top else None)
    if bad:
        write(RESULTS / "rehearse.json", out)
        lab.stop(bad, {"step": t, "loss": rec["loss"], "dw_norm": rec["dw_norm"], "kl": kl})


def fine_tune(lab: Lab, model: Any, W0: dict[str, Tensor], rows: list[Tensor], grid: list[float],
              out: dict[str, Any]) -> None:
    folder = LARGE / "dW" / f"{C.side}_s{C.seed}"
    seqs = P.cue_seqs(lab.tok, lab.end_id, lab.tasks[C.side]["train"])
    stream = [seqs[i] for i in cycle(len(seqs), C.seed, RECIPE.steps * C.batch)]

    def save_dw(t: int, dW: dict[str, Tensor]) -> None:
        save(folder / f"step_{t}.safetensors", dW, F32, droppable=False, log=lab.log)
        out["saves"] = sorted(set(out["saves"]) | {t})

    def on_step(rec: dict[str, Any]) -> None:
        rec.pop("norms")
        rec["peak_gib"] = torch.cuda.max_memory_allocated() / GIB
        out["steps"].append(rec)
        write(RESULTS / "rehearse.json", out)
        lab.log(f"[train] {json.dumps(rec)}")
        early(lab, out, rec, grid[-1])

    t0, steps = time.time(), RECIPE.steps
    train(model, W0, lambda t: stream[(t - 1) * C.batch : t * C.batch], RECIPE, {"own": rows}, set(range(1, steps + 1)),
          {"own": grid}, {steps}, save_dw, on_step, folder / "ckpt.pt", steps, dW_of=delta_fp32)
    out["train_seconds"] = time.time() - t0


def at_15x(out: dict[str, Any], grid: list[float]) -> tuple[int, bool]:
    """The step the 15x rule reads: the first read at or past the largest save size; else the last step (False)."""
    hit = [c["hi"] for s in out["steps"] for c in s.get("crossed", []) if c["grid"] == grid[-1]]
    return (hit[0], True) if hit else (out["steps"][-1]["step"], False)


def stage_rehearse(lab: Lab) -> None:
    assert LARGE.is_dir(), f"{LARGE}: link it (fleet/link-data) or, on a pod, replace the dangling symlink with a folder"
    floor = read(RESULTS / "floors.json").get("floors", {}).get(C.side, {}).get("K0")
    if floor is None:
        lab.stop("no floor", {"side": C.side, "file": "results/floors.json"})
    grid = [m * floor for m in C.grid]
    model = fp32_model(lab)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    own = P.own_rows(lab.tok, lab.end_id, C.side, lab.tasks[C.side]["train"]).tokens
    rows = list(own.to(DEV).split(C.kl_batch))
    out = {"side": C.side, "seed": C.seed, "recipe": asdict(RECIPE), "batch": C.batch, "floor": floor, "grid": grid,
           "steps": [], "saves": []}
    control(lab, model, W0, rows, out)
    fine_tune(lab, model, W0, rows, grid, out)
    out["states"] = {}
    for t in out["saves"]:
        dW = load(LARGE / "dW" / f"{C.side}_s{C.seed}" / f"step_{t}.safetensors", DEV)
        norm = math.sqrt(sum(float(x.double().square().sum()) for x in dW.values()))
        out["states"][str(t)] = {"dw_norm": norm} | reads_at(lab, model, W0, plus(W0, dW), rows)
        del dW
        write(RESULTS / "rehearse.json", out)
        lab.log(f"[state] step {t} {json.dumps(out['states'][str(t)])}")
    step, reached = at_15x(out, grid)
    em = out["states"][str(step)]["exact_match"]
    out["rule"] = {"step": step, "reached_15x": reached, "exact_match": em,
                   "holds": em >= lab.P["rehearse"]["em_15x_min"]}
    write(RESULTS / "rehearse.json", out)
    lab.log(f"[rehearse] rule {json.dumps(out['rule'])}")
    lab.done(RESULTS / "rehearse.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="E160: capacity, floors or rehearse")
    ap.add_argument("stage", choices=("capacity", "floors", "rehearse"))
    ap.add_argument("--decomp", type=Path, help="the p-9707d6ff folder (floors only)")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"{PRED} is missing: predictions are filed before any run")
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    lab = Lab(a.stage)
    if a.stage == "capacity":
        stage_capacity(lab)
    elif a.stage == "floors":
        stage_floors(lab, a.decomp)
    else:
        stage_rehearse(lab)
    hard_exit(0)


if __name__ == "__main__":
    main()

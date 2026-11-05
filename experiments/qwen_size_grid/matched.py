"""E91's second stage, 2026-11-05: anchor 2 retrained at both seeds a little past anchor 1's size, because neither seed has a
saved state there (run.py's grid: seed 42 step 20 at 0.60 of the norm, seed 142 step 100 at 4.6 times). Then its fact
answers at those states. Rules: results/predictions_matched.json.

Per seed, E84's recipe and stream on one L40, stopped at step 100: the step nearest anchor 1's norm is taken from the seed's
own training record; KL on the maths rows is read at every step from 20 to 50 and the nearest is held on the GPU until 50,
so at most two 5.64 GB files are written. At step 100 the rerun is compared with the seed's saved file, against the distance
between the two seeds. Then E84's three answer sets at each saved state. About 5 min per seed for training, 5 per state.
Writes results/matched.json as it goes, each number also in results/run.log; states in results/large/dW/a2_s<seed>/.
A lost job resumes: a finished seed or state in matched.json is not redone.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import linears, load, mean_kl, plus, save, set_weights
from commute.qwen_train import Recipe, train

E88: Any = load_experiment("qwen_anchor_checks", "run")
E84, E81, JOB, DEV, F32, CFG = E88.E84, E88.E81, E88.JOB, E88.DEV, E88.F32, E88.CFG

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions_matched.json", RESULTS / "matched.json"
EXPERIMENTS = HERE.parent
RUNS = {42: (EXPERIMENTS / "qwen_tofu_forms" / "results", "train.json", ("dW", "a2_2e-5")),
        142: (EXPERIMENTS / "qwen_anchor_checks" / "results", "S_train.json", ("dW", "tofu_seed142"))}
KL_FROM, KL_TO, CHECK_AT = 20, 50, 100


class Lab(E88.Lab):  # the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")


class Enough(Exception):
    """The steps past the check are not needed; `train` has no stop of its own without a checkpoint."""


def gap(x: float, ref: float) -> float:
    return abs(math.log(x / ref))


def norm_step(seed: int, ref: float) -> int:
    folder, name, _ = RUNS[seed]
    steps = json.loads((folder / name).read_text())["steps"]
    return min((r for r in steps if r["dw_norm"] > 0), key=lambda r: gap(r["dw_norm"], ref))["step"]


def saved_path(seed: int) -> Path:
    folder, _, parts = RUNS[seed]
    return folder.joinpath("large", *parts) / f"step_{CHECK_AT}.safetensors"


def distance(a: Mapping[str, Tensor], b: Path) -> tuple[float, float]:
    """|a - b| and |b|, one matrix at a time on the host."""
    d, n = 0.0, 0.0
    with safe_open(str(b), "pt") as f:
        for s in a:
            x = f.get_tensor(s).double()
            d += float((a[s].double() - x).pow(2).sum())
            n += float(x.pow(2).sum())
    return math.sqrt(d), math.sqrt(n)


def seeds_apart() -> float:
    """|seed 42 - seed 142| / |seed 42| at the check step, from the saved files."""
    with safe_open(str(saved_path(42)), "pt") as f, safe_open(str(saved_path(142)), "pt") as g:
        d, n = 0.0, 0.0
        for s in f.keys():  # noqa: SIM118  (safe_open is not a mapping)
            x = f.get_tensor(s).double()
            d += float((x - g.get_tensor(s).double()).pow(2).sum())
            n += float(x.pow(2).sum())
    return math.sqrt(d / n)


def retrain(lab: Lab, seed: int, ref: dict[str, float], rows: list[Tensor], folder: Path) -> dict[str, Any]:
    t_norm, seqs = norm_step(seed, ref["norm"]), E84.train_seqs(lab)
    r = Recipe(E84.RATE, E81.WARMUP, E81.STEPS, "hf_cosine", 1.0, E81.MICRO, forward="fp32")
    out: dict[str, Any] = {"t_norm": t_norm, "kl": {}, "norm": {}}
    best: dict[str, Any] = {"gap": math.inf}
    assert KL_FROM <= t_norm <= KL_TO, t_norm

    def weights() -> dict[str, Tensor]:
        return {s: lin.weight.data for s, lin in linears(lab.model).items()}

    def on_step(rec: dict[str, Any]) -> None:
        t = rec["step"]
        out["norm"][t] = rec["dw_norm"]
        if KL_FROM <= t <= KL_TO:
            with torch.no_grad():
                k = mean_kl(lab.model, rows, lab.W0, None)
            out["kl"][t] = k
            if gap(k, ref["kl"]) < best["gap"]:
                best.update(gap=gap(k, ref["kl"]), t=t, dW={s: x - lab.W0[s] for s, x in weights().items()})
        if t == t_norm:
            save(folder / f"step_{t}.safetensors", delta_fp32(weights(), lab.W0), F32, droppable=False, log=lab.log)
        if t == KL_TO:
            out["t_kl"] = best["t"]
            if best["t"] != t_norm:
                save(folder / f"step_{best['t']}.safetensors", best["dW"], F32, droppable=False, log=lab.log)
            best.pop("dW")
        if t == CHECK_AT:
            d, n = distance(delta_fp32(weights(), lab.W0), saved_path(seed))
            out["rerun_vs_saved"], out["saved_norm"] = d / n, n
            raise Enough
        if t % 10 == 0 or t <= 3:
            lab.log(f"[matched s{seed}] {json.dumps({k: rec[k] for k in ('step', 'loss', 'dw_norm')})} kl {out['kl'].get(t)}")

    try:
        train(lab.model, lab.W0, lambda t: tofu.stream(seqs, E81.BATCH, seed, t), r, {}, {E81.STEPS}, {}, {E81.STEPS},
              lambda *_: None, on_step, LARGE / f"ckpt_unused_{seed}.pt", E81.STEPS, dW_of=delta_fp32)
    except Enough:
        pass
    torch.cuda.empty_cache()
    return out


def state_row(seed: int, t: int, run: dict[str, Any], keep: set[int], sink: Path) -> dict[str, Any]:
    return {"seed": seed, "step": t, "norm": run["norm"][str(t)], "kl": run["kl"][str(t)], "answers": E84.graded(sink, keep)}


def label(trained: float, P: dict[str, Any]) -> str:
    return "learned" if trained >= P["learned_line"] else "not learned"


def main() -> None:
    P = json.loads(PRED.read_text())
    assert LARGE.is_symlink(), f"{LARGE}: run ~/commute-agents/fleet/link-data on this clone"
    lab = Lab("M", "E91", PRED)
    ref = json.loads((RESULTS / "size_grid.json").read_text())["reference"]
    lab.fp32()
    rows, keep = lab.maths()[: CFG.track_rows], E88.kept_rows()
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"card": lab.card, "seeds_apart": seeds_apart(), "runs": {}, "states": []}
    for seed in RUNS:
        folder = LARGE / "dW" / f"a2_s{seed}"
        if str(seed) not in out["runs"]:
            out["runs"][str(seed)] = json.loads(json.dumps(retrain(lab, seed, ref, rows, folder)))  # keys as on a resume
            OUT.write_text(json.dumps(out, indent=1) + "\n")
        run = out["runs"][str(seed)]
        lab.log(f"[matched s{seed}] t_norm {run['t_norm']} t_kl {run['t_kl']} rerun_vs_saved {run['rerun_vs_saved']:.4g}; seeds apart {out['seeds_apart']:.4g}")
        for t in sorted({run["t_norm"], run["t_kl"]}):
            if any(s["seed"] == seed and s["step"] == t for s in out["states"]):
                continue
            sink = LARGE / "answers" / f"a2_s{seed}" / f"step_{t}"
            with set_weights(lab.model, plus(lab.W0, load(folder / f"step_{t}.safetensors", DEV))):
                for name in E84.SETS:
                    E84.answer_set(lab, name, sink)
            row = state_row(seed, t, run, keep, sink)
            row["label"] = label(row["answers"]["trained"]["recall"], P)
            out["states"].append(row)
            lab.log(f"[matched s{seed} step {t}] {json.dumps(row)}")
            OUT.write_text(json.dumps(out, indent=1) + "\n")
    out["done"] = True
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}; FINISHED", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()

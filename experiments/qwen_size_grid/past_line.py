"""E91 Amendment 1, 2026-11-05: anchor 2 at each seed saved at its first state past the learning line, so the D3 rows have a
state that holds its facts at the smallest size we can save. User's addition on ask 11. Rules: results/predictions_past_line.json.

Per seed, E84's recipe and stream on one L40. From step 500, every 50 steps, the trained wording's answers are read; at the
first state at or over the line the state is saved, the other two answer sets are read and the seed ends. About 15 min per
seed for training and checks, 5 for the sets. Writes results/past_line.json as it goes, each number also in results/run.log; states
in results/large/dW/a2_s<seed>/, answers in results/large/answers/a2_s<seed>/step_<t>/. A lost job redoes only an unfinished seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment
from commute.qwen_rl import delta_fp32
from commute.qwen_sft import linears, mean_kl, save
from commute.qwen_train import Recipe, train

M: Any = load_experiment("qwen_size_grid", "matched")
E84, E81, E88, F32, CFG = M.E84, M.E81, M.E88, M.F32, M.CFG

RESULTS, LARGE = M.RESULTS, M.LARGE
PRED, OUT = RESULTS / "predictions_past_line.json", RESULTS / "past_line.json"
FIRST, EVERY, LAST = 500, 50, 1000


def first_past(lab: Any, seed: int, line: float, rows: list[torch.Tensor], keep: set[int]) -> dict[str, Any]:
    seqs = E84.train_seqs(lab)
    r = Recipe(E84.RATE, E81.WARMUP, E81.STEPS, "hf_cosine", 1.0, E81.MICRO, forward="fp32")
    out: dict[str, Any] = {"seed": seed, "checks": []}

    def on_step(rec: dict[str, Any]) -> None:
        t = rec["step"]
        if t % 100 == 0:
            lab.log(f"[past_line s{seed}] {json.dumps({k: rec[k] for k in ('step', 'loss', 'dw_norm')})}")
        if t < FIRST or t % EVERY:
            return
        sink: Path = LARGE / "answers" / f"a2_s{seed}" / f"step_{t}"
        lab.model.eval()
        with torch.no_grad():
            E84.answer_set(lab, "trained", sink)
        recall = tofu.new_word_recall(sink / "trained.jsonl", keep)["recall"]
        out["checks"].append({"step": t, "norm": rec["dw_norm"], "trained": recall})
        lab.log(f"[past_line s{seed}] {json.dumps(out['checks'][-1])}")
        if recall < line and t < LAST:
            return
        weights = {s: lin.weight.data for s, lin in linears(lab.model).items()}
        save(LARGE / "dW" / f"a2_s{seed}" / f"step_{t}.safetensors", delta_fp32(weights, lab.W0), F32, droppable=False, log=lab.log)
        with torch.no_grad():
            for name in E84.SETS:
                if name != "trained":
                    E84.answer_set(lab, name, sink)
            kl = mean_kl(lab.model, rows, lab.W0, None)
        out |= {"step": t, "norm": rec["dw_norm"], "kl": kl, "past": recall >= line, "answers": E84.graded(sink, keep)}
        raise M.Enough

    try:
        train(lab.model, lab.W0, lambda t: tofu.stream(seqs, E81.BATCH, seed, t), r, {}, {E81.STEPS}, {}, {E81.STEPS},
              lambda *_: None, on_step, LARGE / f"ckpt_unused_{seed}.pt", E81.STEPS, dW_of=delta_fp32)
    except M.Enough:
        pass
    torch.cuda.empty_cache()
    return out


def main() -> None:
    P = json.loads(PRED.read_text())
    lab = M.Lab("P", "E91", PRED)
    lab.fp32()
    ref = json.loads((RESULTS / "size_grid.json").read_text())["reference"]
    rows, keep = lab.maths()[: CFG.track_rows], E88.kept_rows()
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"card": lab.card, "reference": ref, "seeds": {}}
    for seed in M.RUNS:
        if str(seed) not in out["seeds"]:
            out["seeds"][str(seed)] = first_past(lab, seed, P["line"], rows, keep)
            OUT.write_text(json.dumps(out, indent=1) + "\n")
        s = out["seeds"][str(seed)]
        lab.log(f"[past_line s{seed}] step {s['step']} past {s['past']} norm {s['norm']:.3f} = {s['norm'] / ref['norm']:.1f} x anchor 1, KL {s['kl']:.4f}")
    out["done"] = True
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}; FINISHED", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()

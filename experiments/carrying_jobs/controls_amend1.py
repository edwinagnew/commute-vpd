"""carrying_jobs (E101) Amendment 1, the one GPU step: `sbatch cluster/carrying_jobs.sbatch controls_amend1` (debug,
< 5 min). Plan: reports/plans/2026-11-05_plan_e101-carrying-jobs.md, "Amendment 1"; rules results/predictions_amend1.json.

  task-specific W0 controls from stage 0's engaged.pt (`carrying_jobs.contrast_top`), 10 per control task
  the controls not in the filed pool -> one writer batch, large/batches/c00.json (the filed format, ids hashed with the
          filed salt); the filed key plus the controls' roles -> large/key_amend1.json
  each description's read windows (the GPU part: the task loaders run W0), decoded, the answer marked
          -> results/task_windows_amend1.json
Every number is printed. Nothing is judged here.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from typing import Any

import torch

from commute import carrying_jobs as CJ
from commute import steering as ST
from commute import vpd
from commute.data import pile_rows
from commute.jobs import hard_exit, load_experiment

R = load_experiment("carrying_jobs", "run")  # its carrying_sets, window_text, paths, and E98's task loaders
RESULTS, LARGE, REPO, log = R.RESULTS, R.LARGE, R.REPO, R.log
PRED, PRED_A = RESULTS / "predictions.json", RESULTS / "predictions_amend1.json"


def committed(path: Any) -> None:
    rel = str(path.relative_to(REPO))
    changed = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode
    if not path.is_file() or changed or not R.git("ls-files", rel):
        sys.exit(f"{rel} is not committed unchanged: commit it first")


def controls(P: dict[str, Any], A: dict[str, Any], key: dict[str, Any]) -> dict[str, list[int]]:
    eng = torch.load(LARGE / "engaged.pt")
    rand = {k["flat"] for k in key["neurons"].values() if any(r["role"] == "random" for r in k["roles"])}
    every = {i for idx in R.carrying_sets(P).values() for i in idx}
    out = {}
    for t, c in A["settings"]["controls"].items():
        out[t], over = CJ.contrast_top(eng, t, c["n"], every | rand)
        log(f"[controls {t}] over {over} neurons {out[t]} own engaged {[round(float(eng[t][i]), 3) for i in out[t]]}")
    return out


def key_and_batch(P: dict[str, Any], ctl: dict[str, list[int]], key: dict[str, Any]) -> dict[str, Any]:
    """The filed key with each control's role added; the controls not in it written as writer batch c00."""
    S, top = P["settings"], torch.load(LARGE / "pile_top.pt")
    by_flat = {k["flat"]: i for i, k in key["neurons"].items()}
    neurons = {i: {**k, "roles": list(k["roles"])} for i, k in key["neurons"].items()}
    new: list[int] = []
    for t, idx in ctl.items():
        for f in idx:
            i = by_flat.get(f) or CJ.neuron_id(key["salt"], f)
            if f not in by_flat and i not in neurons:
                assert i not in key["neurons"], f"id collision {i}"
                new.append(f)
                neurons[i] = {"flat": f, "block": f // CJ.D_MLP, "neuron": f % CJ.D_MLP, "top_act": float(top["values"][0, f]), "roles": []}
            neurons[i]["roles"].append({"group": t, "role": "control_amend1"})
    rows, tok = pile_rows(*S["pile_cache"], log=log)[slice(*S["window_rows"])], vpd.tokenizer()
    batch = []
    for f in new:
        peak = float(top["values"][0, f])
        wins = [{"rank": r + 1, "rel": round(float(top["values"][r, f]) / peak, 2) if peak > 0 else None,
                 "text": R.window_text(tok, rows[int(top["rows"][r, f])], int(top["pos"][r, f]), S["context"])}
                for r in range(S["n_top"])]
        batch.append({"id": CJ.neuron_id(key["salt"], f), "windows": wins})
    random.Random(S["seed"] + 7).shuffle(batch)
    (LARGE / "batches" / "c00.json").write_text(json.dumps({"batch": "c00", "n": len(batch), "neurons": batch}, indent=1, ensure_ascii=False))
    out = {"salt": key["salt"], "writer_batches": {**key["batches"], "c00": [n["id"] for n in batch]}, "neurons": neurons}
    (LARGE / "key_amend1.json").write_text(json.dumps(out, indent=1))
    log(f"[key_amend1] {len(neurons)} neurons, {len(new)} new in c00, dead new {sum(neurons[n['id']]['top_act'] <= 0 for n in batch)}")
    return out


def task_windows(A: dict[str, Any]) -> dict[str, list[str]]:
    """Per description, its tasks' read windows (seed + 4): the context up to the answer position, the target marked."""
    S, tok = A["settings"], vpd.tokenizer()
    reads = R.E98.load_reads(ST.Net(R.DEVICE), R.E98.files())
    rng, out = random.Random(S["seed"] + 4), {}
    for d, tasks in S["description_windows"].items():
        out[d] = []
        for t, n in tasks.items():
            r = reads[t]
            for w in sorted(rng.sample(range(len(r.toks)), n)):
                pieces = [tok.decode([x]) for x in r.toks[w, max(0, r.half - S["window_context"] + 1) : r.half + 1].tolist()]
                out[d].append(CJ.mark([*pieces, tok.decode([int(r.tgt[w])])], len(pieces)))
        log(f"[windows {d}] {json.dumps(out[d], ensure_ascii=False)}")
    return out


def main() -> None:
    committed(PRED)
    committed(PRED_A)
    P, A = json.loads(PRED.read_text()), json.loads(PRED_A.read_text())
    key = json.loads((LARGE / "key.json").read_text())
    ctl = controls(P, A, key)
    key_and_batch(P, ctl, key)
    win = task_windows(A)
    rec = {"git_head": R.git("rev-parse", "HEAD").strip(), "controls": ctl, "windows": win}
    (RESULTS / "task_windows_amend1.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log(f"[done] peak_gpu_gb {torch.cuda.max_memory_allocated() / 2**30:.2f}")
    hard_exit(0)


if __name__ == "__main__":
    main()

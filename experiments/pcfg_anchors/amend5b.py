"""E55 Amendment 5b: three more seeds of Amendment 5's fast natural-text arm (WNf, lr 1e-4), so Stage C1's "existing"
side has states at lr 1e-4 (lead's decision on the C1 draft, 2026-10-30). Rules, thresholds and expected values:
results/predictions_amend5b.json (read, never edited; the run refuses a draft). amend5.py and amend5.json are untouched.

Arms WNf_s1, WNf_s2, WNf_s3: amend5's WN task and recipe at lr 1e-4 (amend5.Arm5("WN", 1e-4, seed)), so each seed's
data stream is amend5's scheme, default_rng([run.FT_SEED0 + seed, chunk, TAIL["WN"]]) (the same stream WN_s<seed> used at
1e-5, as WNf_s0 shares WN_s0's). Saved at 500, 1K and 2K in results/large/amend5b/; every state read by amend5.read_wn.

Order: model-free controls (amend5's, plus this file's arms' training rows and constants equal to amend5's); LOW's W0
controls and reference (amend5.w0_controls; a miss writes results/amend5b_control_miss.json, exit 1); the three arms,
results/amend5b.json after each. Exit 3 at the deadline; a resubmit resumes.

Usage (repo root, GPU): python experiments/pcfg_anchors/amend5b.py  (cluster/pcfg_anchors.sbatch amend5b)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from typing import Any

import torch
from amend4 import PRED4
from amend5 import (
    PRED5,
    SAVES,
    Arm5,
    TailBatches,
    build,
    key5,
    model_free_checks,
    read_wn,
    rows_ok,
    summary,
    train_chunk,
    w0_controls,
)
from run import (
    EXIT_RESUME,
    FT_ITERS,
    FT_SEED0,
    LARGE,
    PRED,
    RESULTS,
    WEIGHT_DECAY,
    filed_bars,
    load_model,
    log,
    state_path,
    write_json,
)

from commute.pcfg_train import Run, Schedule, answer_loss, train

PRED5B = RESULTS / "predictions_amend5b.json"
OUT5B, MISS = RESULTS / "amend5b.json", RESULTS / "amend5b_control_miss.json"
A5B_DIR = LARGE / "amend5b"
ARMS5B = {f"WNf_s{s}": Arm5("WN", 1e-4, s) for s in (1, 2, 3)}


def finetune5b(key: str, arm: Arm5, prompt: tuple[int, int], deadline: float) -> bool:
    """amend5.finetune5 (which writes only to results/large/amend5/) with the states in A5B_DIR."""
    model = load_model(LARGE / "a1" / "pt_LOW.pt")
    torch.manual_seed(FT_SEED0 + arm.seed)
    run = Run(Schedule(FT_ITERS, arm.lr), WEIGHT_DECAY, save_every=SAVES[0], log_every=100, loss=answer_loss)
    for it in SAVES:
        if state_path(key, it, A5B_DIR).exists():
            continue
        step, _ = train(model, TailBatches(arm, prompt), run, state_path(key, "train", A5B_DIR), stop_at=it,
                        deadline=deadline, log=lambda m: log(f"{key} {m}"))  # fmt: skip
        if step < it:
            return False
        tmp = state_path(key, it, A5B_DIR).with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "step": step}, tmp)
        tmp.replace(state_path(key, it, A5B_DIR))
    return True


def stop(res: dict[str, Any]) -> None:
    write_json(MISS, res)
    log(f"STOP: a control missed (apparatus): {res['controls']}; see {MISS.name} (exit 1)")
    sys.exit(1)


def run_arms(res: dict[str, Any], ev: Any, bars: dict[str, Any], k: dict[str, Any], deadline: float) -> int:
    """amend5.run_arms for ARMS5B, into A5B_DIR and OUT5B."""
    for name, arm in ARMS5B.items():
        if name in res["arms"]:
            log(f"{name}: in {OUT5B.name}; skipped")
            continue
        key = key5(name, arm, ev.prompt)
        if time.monotonic() > deadline or not finetune5b(key, arm, ev.prompt, deadline):
            log(f"{name}: deadline; resubmit to resume (exit {EXIT_RESUME})")
            return EXIT_RESUME
        states = {str(it): read_wn(load_model(state_path(key, it, A5B_DIR)), ev, bars, k) for it in SAVES}
        for it, r in states.items():
            summary(name, int(it), r)
        res["arms"][name] = {"arm": asdict(arm), "state_key": key, "saves": list(SAVES), "states": states}
        write_json(OUT5B, res)
        state_path(key, "train", A5B_DIR).unlink(missing_ok=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline-min", type=float, default=22.0, help="save and exit 3 after this many minutes")
    args = ap.parse_args()
    deadline = time.monotonic() + 60 * args.deadline_min
    if not PRED5B.exists() or str(json.loads(PRED5B.read_text()).get("status", "")).startswith("draft"):
        sys.exit(f"file the predictions first (a draft is refused): {PRED5B}")
    k, k5 = json.loads(PRED5B.read_text())["constants"], json.loads(PRED5.read_text())["constants"]
    k4 = json.loads(PRED4.read_text())["constants"]
    bars = filed_bars(json.loads(PRED.read_text())["a2_bars"], json.loads((RESULTS / "a1c_patch.json").read_text())["a2_bars"])
    assert bars is not None, "Stage A's bars are filed"
    A5B_DIR.mkdir(parents=True, exist_ok=True)
    ev = build(k, k4)
    checks, info = model_free_checks(ev)
    checks |= {f"{n}_chunk0_rows": rows_ok(train_chunk(a, 0, ev.prompt), a.task, ev.prompt) for n, a in ARMS5B.items()}
    checks["constants_equal_amend5"] = k == k5
    log(f"model-free controls {checks}")
    if not all(checks.values()):
        stop({"controls": checks, "model_free": info})
    w0_checks, w0 = w0_controls(load_model(LARGE / "a1" / "pt_LOW.pt"), ev, bars, k)
    prev = json.loads(OUT5B.read_text()) if OUT5B.exists() else {}
    res = {"constants": k, "controls": checks | w0_checks, "model_free": info, "W0": w0,
           "arms": prev.get("arms", {}) if prev.get("constants") == k else {}}  # fmt: skip
    if not all(w0_checks.values()):
        stop(res)
    write_json(OUT5B, res)
    sys.exit(run_arms(res, ev, bars, k, deadline))


if __name__ == "__main__":
    main()

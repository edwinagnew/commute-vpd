"""E102 (qwen_carrying) stage B's scoring job, on one GPU: E101's maximiser test (`commute.carrying_jobs`, the essay's
"Explanations That Could Lose") on Qwen3-1.7B's W0. Plan: reports/plans/2026-11-05_plan_e102-qwen-carrying.md, "Stage B";
rules: results/predictions_stageB.json (early_checks). Run as `sbatch cluster/qwen_carrying_b.sbatch` from the clone root,
after every maximiser file is in (results/stage_b/maximiser/b<NN>.json); under 10 min on the debug QoS.

  E0  16 pool neurons' rank-1 FineWeb rows read again: each peak against stage A's saved value (stop on a miss)
  C2  every pool neuron's own windows ranked 2-6 against 5 other neurons' windows of the same ranks (stop if under the bar)
  the agents' texts: each text's peak W0 firing of its neuron; outcome by `carrying_jobs.outcome`
Reads large/key_b.json, large/tops.safetensors. Writes results/stage_b/max_scores.json {"E0", "self_test", "neurons":
{id: {"examples", "near_misses", "outcome"}}, "missing_batches", "dropped"} and results/stage_b/max.log, every score
logged. A neuron's firing is `qwen_neurons`' a_i = silu(g_i) u_i, flat index l x 6144 + i, as stage A read it.
"""

from __future__ import annotations

import json
import time
from typing import Any

import torch
from safetensors.torch import load_file

from commute import carrying_jobs as CJ
from commute import qwen_neurons as QN
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_vpd import fp32_matmuls, load_target

SB: Any = load_experiment("qwen_carrying", "stage_b")
S, STAGE_B, LARGE = SB.S, SB.STAGE_B, SB.LARGE
DEV, E0_N, E0_REL, E0_ABS = "cuda", 16, 1e-3, 1e-4
T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    with (STAGE_B / "max.log").open("a") as f:
        f.write(line + "\n")


def stop(out: dict[str, Any], why: str) -> None:
    (STAGE_B / "max_scores.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"STOP: {why}; tell the lead")
    hard_exit(1)


def e0(fire: CJ.Fire, key: dict[str, Any], tops: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, Any]:
    """The first E0_N pool neurons by id: peak over the rank-1 row against tops values[0]."""
    ids = sorted(key["neurons"])[:E0_N]
    flats = [key["neurons"][i]["flat"] for i in ids]
    seqs = [rows[int(tops["rows"][0, f])].tolist() for f in flats]
    got = CJ.peaks(fire, seqs, flats, 4, DEV)
    want = [float(tops["values"][0, f]) for f in flats]
    worst = max(abs(g - w) - E0_REL * abs(w) for g, w in zip(got, want, strict=True))
    rec = {"got": got, "saved": want, "ok": worst <= E0_ABS, "max_abs_dev": max(abs(g - w) for g, w in zip(got, want, strict=True))}
    log(f"[E0] {json.dumps(rec)} (expect every |got - saved| <= {E0_REL} x saved + {E0_ABS})")
    return rec


def main() -> None:
    require_filed(SB.PRED_B)
    STAGE_B.mkdir(parents=True, exist_ok=True)
    key = json.loads(SB.KEY.read_text())
    tops = load_file(str(LARGE / "tops.safetensors"))
    rows = tops["fineweb_rows"].long()
    tok = SB.tokenizer()
    fp32_matmuls()
    model = load_target(DEV, torch.float32)
    log(f"[load] {len(key['neurons'])} neurons in {len(key['batches'])} batches; model on {torch.cuda.get_device_name()}")
    fire = lambda toks: QN.flat(QN.run(model, toks, logits=False)[2])
    encode = lambda text: tok(text, add_special_tokens=False)["input_ids"]
    out: dict[str, Any] = {"E0": e0(fire, key, tops, rows), "neurons": {}}
    if not out["E0"]["ok"]:
        stop(out, "E0: the firing function does not reproduce stage A's saved peaks")
    flats = {i: k["flat"] for i, k in key["neurons"].items()}
    wins = CJ.own_windows(flats, tops, rows, tok.decode, encode, S["context"], S["max_tokens"])
    out["self_test"] = CJ.self_test_run(fire, key["neurons"], wins, S["seed"] + 4, S["strong_frac"], S["self_test_min_win"],
                                        S["fire_batch"], DEV, log)
    if not out["self_test"]["ok"]:
        stop(out, "C2 below self_test_min_win: the text round trip or the 30% rule is broken")
    texts, missing, bad = CJ.load_texts(STAGE_B / "maximiser", key["batches"], S["n_examples"], S["n_near_misses"])
    log(f"[texts] {len(texts)} neurons read, missing batch files {missing}, dropped ids {bad}")
    out["neurons"] = CJ.text_scores(fire, key["neurons"], texts, encode, S["strong_frac"], S["max_tokens"], S["fire_batch"], DEV, log)
    out |= {"missing_batches": missing, "dropped": bad}
    (STAGE_B / "max_scores.json").write_text(json.dumps(out, indent=1, default=float))
    counts = CJ.self_test([v["outcome"] for v in out["neurons"].values()], 0)["counts"]
    log(f"[done] {len(out['neurons'])} neurons scored; outcomes {json.dumps(counts)}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)

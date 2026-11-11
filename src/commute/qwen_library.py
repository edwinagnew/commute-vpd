"""The Qwen3-1.7B pair library's shared pieces (cell P3.5; plan reports/plans/2026-11-11_plan_p35-qwen-pair-library.md).
Run 1 is experiments/qwen_pair_library/run.py, run 2 is library.py there. Here: the fp32 model, greedy exact match,
Amendment 1's early stops, the template's training sequences as E45/E78 built them, and run 2's rules, which read
the arms' result files (results/library/<arm>_s<seed>.json, rules filed in results/predictions_run2.json).

Run 2's rules. Each reads one saved state per arm: the save labelled 15x, else the arm's last save (`read_state`).
  learned    built and template arms: trained-item hits >= ceil(accuracy_min x n), absolute (a tie holds). Gold arms:
             trained-row gold loss <= gold_loss_ratio_max x the arm's W0 value (a tie holds). An arm that stopped, is
             missing, is unfinished or has no save is not learned.
  copy held  the copy arm's held-out hits at that state >= ceil(copy_held_min x n), reached_15x said beside it.
  matched    a pair at a size (5x, 15x apart): both sides finished, each with a save at that size.
  fit        a pair at a seed: both sides learned, matched at one size or more, and run 1's capacity verdict is "fit"
             (twins have no capacity check: "n/a" passes).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers import GenerationConfig

from commute import qwen_pairs as P
from commute.jobs import load_experiment
from commute.qwen_eval import GREEDY, generate
from commute.qwen_sft import Seq
from commute.qwen_updates import cycle, keep_finished, ok
from commute.qwen_vpd import load_target

SIZES = ("5x", "15x")
PAIRS = {"copy_cipher": ("copy", "cipher"), "facts": ("known", "invented"),
         "template_twin": ("template", "template_shuf"), "gold_twin": ("gold", "gold_shuf")}
E78_DATA = {"n": 7740, "of": 7744}  # experiments/qwen_anchor1/results/train.json .data


def fp32_model(tok: Any, device: str = "cuda") -> Any:
    m = load_target(device, torch.float32)
    m.generation_config = GenerationConfig(eos_token_id=m.generation_config.eos_token_id, pad_token_id=tok.pad_token_id)
    return m


@torch.no_grad()
def exact_hits(tok: Any, model: Any, side: str, texts: list[str], items: list[P.Item], batch: int,
               max_new: int) -> tuple[int, list[str]]:
    """Greedy answers to the chat texts; how many match their item's answer (`qwen_pairs.matches`), and the answers."""
    out: list[str] = []
    for s in range(0, len(texts), batch):
        out += [r["response"] for r in generate(tok, model, texts[s : s + batch], 1, GREEDY, max_new=max_new)]
    return sum(P.matches(side, r, it.answer) for r, it in zip(out, items, strict=True)), out


def early_stop(steps: list[dict[str, Any]], floor: float, E: dict[str, Any]) -> tuple[str | None, float]:
    """Amendment 1's stops on the latest record (steps[-1]): a non-finite loss; a step 1 that moved (it runs at rate 0);
    KL at step 2 above kl_step2_max_floors x the floor; one step raising KL by more than kl_jump_max_floors x the floor
    (KL before step 1 is 0). Returns the reason (None: go on) and that step's rise."""
    rec = steps[-1]
    t, kl = rec["step"], rec["kl"]["own"]
    rise = kl - (steps[-2]["kl"]["own"] if len(steps) > 1 else 0.0)
    bad = ("loss not finite" if not ok(rec["loss"]) else "step 1 moved" if t == 1 and rec["dw_norm"] != 0
           else "KL at step 2 above the floor bar" if t == 2 and not kl <= E["kl_step2_max_floors"] * floor
           else "a step raised KL past the jump bar" if not rise <= E["kl_jump_max_floors"] * floor else None)
    return bad, rise


def template_pool(log: Any) -> tuple[list[Seq], list[dict[str, Any]], dict[str, Any]]:
    """The answer template's training set as E78 trained it: E45's job (experiments/qwen_updates/job.py `arm_data`,
    arm "a1", seed 0) keeps W0's prompted greedy samples 0-3 of each training problem that finished, each under the
    unprompted chat. Returns the 7,740 sequences, each one's GSM8K problem, and E45's GSM8K sets. Checked: the counts
    are E78's, and these sequences in `cycle` order with seed 0 are E45's own seed-0 stream. Needs E45's
    results/S_a1.json and results/large/{S/a1, P/a1_rows.json}."""
    job_module: Any = load_experiment("qwen_updates", "job")

    class E45(job_module.Job):  # E45's data, logged to the caller's log
        def log(self, msg: str) -> None:
            log(msg)

    job, C = E45("library", "a1"), job_module.CFG
    recs = [r for r in job.samples("a1") if r["pid"] < C.n_problems - C.n_held and 0 <= r["k"] < C.own_k]
    kept = keep_finished(recs)
    seqs = [job.sample_seq(r) for r in kept]
    stream, _, _, data = job.arm_data("a1", 0)
    assert {"n": data["n"], "of": data["of"]} == E78_DATA, data
    assert [seqs[i] for i in cycle(len(seqs), 0, len(stream))] == stream, "not E45's seed-0 stream"
    return seqs, [job.sets["train"][r["idx"]] for r in kept], job.sets


# ----------------------------------------------------------------------------- run 2's rules


def status(arm: dict[str, Any] | None) -> str:
    if arm is None:
        return "missing"
    if arm.get("stopped"):
        return "stopped"
    return "done" if arm.get("done") else "unfinished"


def read_state(arm: dict[str, Any]) -> tuple[str | None, bool]:
    """The key of the state the rules read (the save labelled 15x, else the last save) and whether 15x was reached;
    (None, False) when nothing was saved."""
    saves = arm.get("saves", [])
    hit = [s for s in saves if "15x" in s["labels"]]
    pick = hit[0] if hit else saves[-1] if saves else None
    return (None, False) if pick is None else ("_".join(pick["labels"]), bool(hit))


def at_least_hits(m: dict[str, Any], bar: float) -> tuple[bool, int]:
    """hits >= ceil(bar x n), a tie holding; an empty set never holds."""
    need = math.ceil(bar * m["n"] - 1e-9)
    return m["n"] > 0 and m["hits"] >= need, need


def learned(arm: dict[str, Any] | None, R: dict[str, Any]) -> dict[str, Any]:
    st = status(arm)
    key, reached = read_state(arm) if arm is not None and st == "done" else (None, False)
    if arm is None or key is None:
        return {"learned": False, "why": st if st != "done" else "no save"}
    tr = arm["states"][key]["trained"]
    if arm["family"] == "gold":
        bar = R["gold_loss_ratio_max"] * arm["w0"]["trained"]["value"]
        holds = tr["value"] <= bar
    else:
        holds, bar = at_least_hits(tr, R["accuracy_min"])
    return {"learned": holds, "state": key, "reached_15x": reached, "value": tr["value"], "bar": bar}


def copy_held(arm: dict[str, Any] | None, R: dict[str, Any]) -> dict[str, Any]:
    st = status(arm)
    key, reached = read_state(arm) if arm is not None and st == "done" else (None, False)
    if arm is None or key is None:
        return {"holds": None, "why": st if st != "done" else "no save"}
    h = arm["states"][key]["held"]
    holds, need = at_least_hits(h, R["copy_held_min"])
    return {"holds": holds, "state": key, "reached_15x": reached, "value": h["value"], "hits": h["hits"], "need": need}


def has_size(arm: dict[str, Any] | None, size: str) -> bool:
    return status(arm) == "done" and any(size in s["labels"] for s in arm["saves"])  # type: ignore[index]


def pair_row(a: dict[str, Any] | None, b: dict[str, Any] | None, capacity: str, R: dict[str, Any]) -> dict[str, Any]:
    la, lb = learned(a, R["learned"]), learned(b, R["learned"])
    matched = {s: has_size(a, s) and has_size(b, s) for s in SIZES}
    fit = la["learned"] and lb["learned"] and any(matched.values()) and capacity in ("fit", "n/a")
    return {"learned": [la, lb], "matched": matched, "capacity": capacity, "fit_to_read": fit}


def table(arms: dict[tuple[str, int], dict[str, Any] | None], capacity: dict[str, Any], R: dict[str, Any],
          seeds: tuple[int, ...]) -> dict[str, Any]:
    """Per pair and seed, the rules above; per pair, the seeds fit to read. `capacity`: run 1's capacity.json .pairs."""
    out: dict[str, Any] = {}
    for pair, (x, y) in PAIRS.items():
        cap = capacity[pair]["verdict"] if pair in capacity else "n/a"
        rows = {str(s): pair_row(arms.get((x, s)), arms.get((y, s)), cap, R) for s in seeds}
        out[pair] = {"sides": [x, y], "seeds": rows, "seeds_fit": [s for s, r in rows.items() if r["fit_to_read"]]}
    out["copy_held"] = {str(s): copy_held(arms.get(("copy", s)), R["copy_held"]) for s in seeds}
    return out

"""E102 (qwen_carrying) stage B, the CPU steps: the agents' batch files and the score. No model (the tokenizer only).
Plan: reports/plans/2026-11-05_plan_e102-qwen-carrying.md, "Stage B"; rules: results/predictions_stageB.json (each
command refuses until it is committed unchanged); rule functions: `commute.carrying_jobs`.

  python experiments/qwen_carrying/stage_b.py batches       the pool (three carrying groups, their random partners,
         control (ii)) from stage A's sets, under hashed ids, in mixed batches of 30, each neuron with its 20 FineWeb-Edu
         windows decoded -> large/batches_b/b<NN>.json; the key (id -> neuron, roles) -> large/key_b.json; counts ->
         results/pool_b.json
  python experiments/qwen_carrying/stage_b.py judge-inputs  after every writer file: per batch its jobs and the four
         task descriptions under freshly shuffled letters -> large/judge_inputs_b/b<NN>.json; letters -> large/judge_b.json
  python experiments/qwen_carrying/stage_b.py score         after the judge files and results/stage_b/max_scores.json
         (maximise_b.py): CB1, CB2, PB1, PB2 and the reported rows -> results/score_stageB.json

Agent outputs (committed; formats in briefs/*.md): results/stage_b/writer/b<NN>.json {"batch", "jobs": {id: job}};
results/stage_b/maximiser/b<NN>.json {"batch", "texts": {id: {"examples": [5], "near_misses": [5]}}};
results/stage_b/judge/b<NN>.json {"batch", "picks": {id: {"letter", "fit"}}}. Agents read only their batch's file; the
key stays in large/, which no brief names.
"""

from __future__ import annotations

import json
import random
import secrets
import sys
from typing import Any

import torch
from safetensors.torch import load_file

from commute import carrying_jobs as CJ
from commute import qwen_neurons as QN
from commute.jobs import load_experiment, require_filed

SC: Any = load_experiment("qwen_carrying", "score")  # stage A's files and `w0_picks`
RESULTS, LARGE, STATES = SC.RESULTS, SC.LARGE, SC.STATES
PRED_B = RESULTS / "predictions_stageB.json"
B: dict[str, Any] = json.loads(PRED_B.read_text())
S = B["settings"]
STAGE_B, BATCHES, KEY = RESULTS / "stage_b", LARGE / "batches_b", LARGE / "key_b.json"
JUDGE_IN, LETTERS = LARGE / "judge_inputs_b", LARGE / "judge_b.json"
CARRY_STATES = (*SC.GAUSS_READS, *SC.NAMED)  # every stage-A carrying set: none of them gives a random or control neuron
Roles = dict[int, list[dict[str, str]]]
Table = dict[str, dict[str, Any]]


def load(path: Any) -> Any:
    return json.loads(path.read_text())


def tokenizer() -> Any:
    from transformers import AutoTokenizer

    from commute.qwen_eval import MODEL, MODEL_REV

    return AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)


# ----------------------------------------------------------------------------- the pool and its files


def pool(carry: dict[str, list[int]], eng: dict[str, torch.Tensor]) -> Roles:
    """neuron -> its roles [{group, role}]: control (ii), each group's consensus carrying sample, and a block-matched
    random partner for a random part of each sample; partners and controls from neurons in no stage-A carrying set."""
    roles: Roles = {}
    add = lambda i, g, r: roles.setdefault(i, []).append({"group": g, "role": r})
    every = {i for s in CARRY_STATES for i in carry[s]}
    for i in SC.w0_picks(eng, sorted(every), S["control_n"]):
        add(i, "control", "control")
    samples = {g: CJ.consensus([carry[s] for s in spec["states"]], spec["n"]) for g, spec in S["groups"].items()}
    for g, sample in samples.items():
        for i in sample:
            add(i, g, "carrying")
    gen = torch.Generator().manual_seed(S["seed"] + 2)
    for g, sample in samples.items():
        part = [sample[j] for j in torch.randperm(len(sample), generator=gen)[: S["partner_n"]].tolist()]
        for i in CJ.block_matched(part, every | set(roles), gen, QN.WIDTH):
            add(i, g, "random")
    return roles


def window_text(tok: Any, row: torch.Tensor, pos: int) -> str:
    lo, hi = max(0, pos - S["context"]), min(len(row), pos + S["context"] + 1)
    return CJ.mark([tok.decode([t]) for t in row[lo:hi].tolist()], pos - lo)


def batches() -> None:
    require_filed(PRED_B)
    carry = {s: load(STATES / f"{s}.json")["carry"] for s in CARRY_STATES}
    eng, tops = load_file(str(LARGE / "engaged.safetensors")), load_file(str(LARGE / "tops.safetensors"))
    roles, tok, salt = pool(carry, eng), tokenizer(), secrets.token_hex(16)
    rows = tops["fineweb_rows"]
    ids = {i: CJ.neuron_id(salt, i) for i in roles}
    assert len(set(ids.values())) == len(ids)
    order = sorted(roles)
    random.Random(S["seed"]).shuffle(order)
    n = S["batch_size"]
    groups = {f"b{j // n:02d}": order[j : j + n] for j in range(0, len(order), n)}
    BATCHES.mkdir(parents=True, exist_ok=True)
    for b, members in groups.items():
        neurons = []
        for i in members:
            peak = float(tops["values"][0, i])
            wins = [{"rank": r + 1, "rel": round(float(tops["values"][r, i]) / peak, 2) if peak > 0 else None,
                     "text": window_text(tok, rows[int(tops["rows"][r, i])], int(tops["pos"][r, i]))} for r in range(S["n_top"])]
            neurons.append({"id": ids[i], "windows": wins})
        (BATCHES / f"{b}.json").write_text(json.dumps({"batch": b, "n": len(neurons), "neurons": neurons}, indent=1, ensure_ascii=False))
    key = {"salt": salt, "batches": {b: [ids[i] for i in m] for b, m in groups.items()},
           "neurons": {ids[i]: {"flat": i, "block": i // QN.WIDTH, "neuron": i % QN.WIDTH, "top_act": float(tops["values"][0, i]),
                                "roles": roles[i]} for i in roles}}
    KEY.write_text(json.dumps(key, indent=1))
    counts: dict[str, int] = {}
    for rs in roles.values():
        for r in rs:
            counts[f"{r['group']}/{r['role']}"] = counts.get(f"{r['group']}/{r['role']}", 0) + 1
    rec = {"n_neurons": len(roles), "n_batches": len(groups), "agent_calls": 3 * len(groups), "counts": counts,
           "in_two_groups": sum(len(rs) > 1 for rs in roles.values()), "dead": sum(float(tops["values"][0, i]) <= 0 for i in roles),
           "batches": sorted(groups)}
    SC.put(RESULTS / "pool_b.json", rec)
    print(json.dumps(rec))


def judge_inputs() -> None:
    require_filed(PRED_B)
    key = load(KEY)
    jobs = {b: load(STAGE_B / "writer" / f"{b}.json")["jobs"] for b in key["batches"]}
    rng, letters = random.Random(S["seed"] + 3), {}
    JUDGE_IN.mkdir(parents=True, exist_ok=True)
    for b, j in jobs.items():
        names = sorted(S["descriptions"])
        rng.shuffle(names)
        letters[b] = {chr(65 + k): name for k, name in enumerate(names)}
        rec = {"batch": b, "descriptions": {L: S["descriptions"][name] for L, name in letters[b].items()},
               "jobs": {i: j[i] for i in key["batches"][b] if i in j}}
        (JUDGE_IN / f"{b}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    LETTERS.write_text(json.dumps(letters, indent=1))
    print(f"judge inputs for {len(jobs)} batches in {JUDGE_IN}")


# ----------------------------------------------------------------------------- the score


def table(key: dict[str, Any], picks: dict[str, dict[str, str]], scores: dict[str, Any]) -> Table:
    """id -> {roles, desc and fit (the judge's pick, None if unjudged), judged, outcome (maximiser; None unscorable)}."""
    out = {}
    for i, k in key["neurons"].items():
        s, p = scores.get(i), picks.get(i, {})
        res = CJ.outcome(s["examples"], s["near_misses"], k["top_act"], S["strong_frac"]) if s else None
        out[i] = {"roles": k["roles"], "desc": p.get("desc"), "fit": p.get("fit"), "judged": i in picks, "outcome": res}
    return out


def members(T: Table, group: str | None, role: str) -> list[dict[str, Any]]:
    return [t for t in T.values() if any(r["role"] == role and group in (None, r["group"]) for r in t["roles"])]


def rate(T: Table, group: str | None, role: str, hit: Any) -> tuple[float, int]:
    """(share, n) of `hit` over a group's judged neurons of that role; (nan, 0), which the rules read as unreadable,
    under min_group judged or over missing_max of them unjudged."""
    m = members(T, group, role)
    ok = [t for t in m if t["judged"]]
    if not m or len(ok) < S["min_group"] or 1 - len(ok) / len(m) > S["missing_max"]:
        return float("nan"), 0
    return CJ.rate([bool(hit(t)) for t in ok])


def picked(desc: str) -> Any:
    return lambda t: t["desc"] == desc


def explained(T: Table, group: str | None, role: str) -> dict[str, Any]:
    scorable = [t for t in members(T, group, role) if t["outcome"] is not None]
    share, n = CJ.rate([t["outcome"] == "win" for t in scorable])
    return {"share": share, "n": n, "of": len(members(T, group, role))}


def score() -> dict[str, Any]:
    require_filed(PRED_B)
    key, letters = load(KEY), load(LETTERS)
    picks, missing, bad = CJ.load_picks(STAGE_B / "judge", key["batches"], letters, S["fits"])
    mx = load(STAGE_B / "max_scores.json")
    T = table(key, picks, mx["neurons"])
    out = rules(T, bool(mx["self_test"]["ok"])) | {
        "C2_self_test": mx["self_test"], "E0": mx.get("E0"), "judged": len(picks), "n_neurons": len(T),
        "missing_judge_batches": missing, "dropped_judge_ids": bad, "missing_max_batches": mx.get("missing_batches"),
        "scorable": sum(t["outcome"] is not None for t in T.values())}
    SC.put(RESULTS / "score_stageB.json", out)
    return out


def rules(T: Table, self_test_ok: bool) -> dict[str, Any]:
    """CB1, CB2, PB1, PB2 and the reported rows of predictions_stageB.json on a neuron table."""
    gap = S["min_gap"]
    rand = {d: rate(T, None, "random", picked(d)) for d in S["descriptions"]}
    own = {"a1": "fmt", "a2m": "tofu", "a2f": "tofu", "control": "tofu"}
    groups = {g: ("control" if g == "control" else "carrying") for g in own}
    cb1 = CJ.forced_gate(*rate(T, "control", "control", picked("tofu")), *rand["tofu"], S["chance"], gap)
    fp, n_fp = rate(T, None, "random", lambda t: t["fit"] == "strong")
    cb2 = {"share": fp, "n": n_fp, "ok": n_fp > 0 and fp <= S["judge_fp_max"]}
    pb1 = CJ.gap_rule(*rate(T, "a1", "carrying", picked("fmt")), *rate(T, "a2m", "carrying", picked("fmt")), gap)
    pb2 = CJ.gap_rule(*rate(T, "a2f", "carrying", picked("tofu")), *rand["tofu"], gap)
    read = cb1["status"] == "pass" and cb2["ok"]
    word = {"above": "holds", "below_or_level": "misses"}
    return {
        "CB1_power_gate": cb1, "CB2_judge_fp": cb2,
        "PB1_fmt_a1_over_a2m": {**pb1, "outcome": word.get(pb1["verdict"], pb1["verdict"]), "read": read},
        "PB2_tofu_a2f_over_random": {**pb2, "outcome": {"above": "related", "below_or_level": "not related"}.get(pb2["verdict"], pb2["verdict"]), "read": read},
        "R_explained": {g: explained(T, g, r) for g, r in groups.items()} | {"random": explained(T, None, "random"), "e101_random_base": 146 / 179,
                        "read": self_test_ok},
        "R_pick_rates": {g: {d: rate(T, g, r, picked(d)) for d in S["descriptions"]} for g, r in groups.items()} | {"random": rand},
        "R_strong_own": {g: rate(T, g, groups[g], lambda t, d=d: t["desc"] == d and t["fit"] == "strong") for g, d in own.items()},
        "R_strong_any": {g: rate(T, g, r, lambda t: t["fit"] == "strong") for g, r in groups.items()},
        "R_own_vs_random": {g: CJ.gap_rule(*rate(T, g, r, picked(own[g])), *rand[own[g]], gap) for g, r in groups.items()},
        "R_partner_rates": {g: {d: rate(T, g, "random", picked(d)) for d in S["descriptions"]} for g in S["groups"]},
        "read": {"PB1_PB2": read, "R_explained": self_test_ok},
    }


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "batches":
        batches()
    elif cmd == "judge-inputs":
        judge_inputs()
    elif cmd == "score":
        out = score()
        for k in ("CB1_power_gate", "CB2_judge_fp", "PB1_fmt_a1_over_a2m", "PB2_tofu_a2f_over_random", "R_explained", "read"):
            print(k, json.dumps(out[k], default=float)[:900])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

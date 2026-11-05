"""E99 stage 4, the GPU part (d): the class map and its shuffled twin rebuilt (E58 Amendment 7's windows and recipe, read at its step
30) and read under the value form of C = the embedding class, with the hand-write control first. Plan: reports/plans/2026-11-05_
plan_e99-refine-verify-definition.md (stage 4, Detail 4); Fable: reports/reviews/2026-11-05_review_e99-fable-advice.md (4). Rules:
rules4.py; expected values and settings: results/predictions_stage4.json (committed first; the job refuses otherwise).

  1. C, alone. X_C: natural Pile occurrences of the 128 keys, y = the key's class (E90's `identity_capability`, the class looked
     up). The centred per-class basis (rank 3) at blocks 1 and 2, 20 draws matched in variance (E90 Amendment 2, seed tag 7), W0's
     teeth (Amendment 1's read: W0's own top-1) and the do-nothing state. No teeth is a reading, not a stop: rows are read beside.
  2. The states: class map and twin, seeds 0 and 1, `steps` steps of E54's recipe (`keydep.finetune`'s config); stop unless each
     held-out P(target) at that step is within repro_tol of E58's. Saved in bf16 and read as saved.
  3. The hand-write control, the first reads of any state: each state's table written into block 1's MLP (`perturb.WRITE_KEY`),
     upstream of the removal, at that state's gain. Stop if one loses more than spared_max of its gain: a known route misread.
  4. R-adm (b): the class map's table written into block 2's MLP, downstream of the removal, at each class-map seed's gain.
  5. The four states; then rules4: admissibility, the 2 x 2 per seed, pooling, the paired test class map minus twin.
Writes results/stage4_class.json after every read; results/large/dW_class_*_s{0,1}.pt (bf16, ~57 MB each). Every number is printed.
Usage: sbatch cluster/definition_verify_stage4.sbatch
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, NoReturn
from unittest.mock import patch

import rules4 as R4
import torch
from stage4_cpu import EXP, PRED4, REPO, RESULTS, filed, load, say

from commute import anchors as AN
from commute import perturb as PT
from commute import steering as ST
from commute.data import pile_rows
from commute.finetune import ArmAConfig, finetune_arm_a
from commute.jobs import hard_exit, load_experiment
from commute.keydep import log, sha256

A2 = load_experiment("new_vs_reuse", "amend2")
AP, RD = A2.AP, A2.RD
OUT, LARGE, WINDOWS = RESULTS / "stage4_class.json", RESULTS / "large", EXP / "open_pair/results/large/amend7"
MAPS = ("class_shuffled", "class_map")  # E58's order: the twin, its carry control, first
E90_ROW = {"class_map": "T15 class map 128 | embedding class", "class_shuffled": "T15 twin 128 | embedding class"}


def stop(j: Any, why: str) -> NoReturn:
    j.res["stopped"] = why
    AP.dump(OUT, j.res)
    log(f"STOP: {why}: tell the lead (exit 1)")
    hard_exit(1)


def sites(j: Any, a7: dict[str, Any]) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Each map's held-out windows of its trained keys (E90's `kv_site`), sha256 against E58's record."""
    out = {}
    for m in MAPS:
        p = WINDOWS / f"windows_{m}_trained.pt"
        j.res["sha256"][p.name] = got = sha256(p)
        if got != a7["maps"][m]["sha256"]["trained"]:
            stop(j, f"{p.name}: sha256 {got} differs from E58's record")
        out[m] = AP.kv_site(p)
        s = out[m][0]
        j.res.setdefault("key_earlier", {})[m] = int((s.toks[:, : s.half] == s.toks[:, s.half : s.half + 1]).any(1).sum())
    return out


def capability(j: Any, a7: dict[str, Any], half: int) -> Any:
    """C = the embedding class: E58's class per key (PC signs of W0's input embeddings); its read is W0's own top-1 (E90 Amendment 1)."""
    cls = {int(k): int(v) for k, v in a7["tables"]["class_of"]["class_map"].items()}
    keys = torch.tensor(sorted(cls))
    c = AP.identity_capability(j.net, "embedding class", pile_rows(*j.S["pile"]["rows"], log=log), keys, half, j.S, j.S["n_control"])
    c.y = torch.tensor([cls[k] for k in keys.tolist()], device=c.y.device)[c.y]
    c.rank = len(set(cls.values())) - 1
    c.read = AP.own_top1(j.net, c.read)
    return c


def control(j: Any, c: Any, S4: dict[str, Any]) -> tuple[Any, dict[str, bool]]:
    bases = {b: u.float() for b, u in c.bases(S4["blocks"]).items()}
    f = A2.matched_basis(j, "class", c, bases, S4["tag"])
    if set(f.bases) != set(S4["blocks"]):
        stop(j, f"no matched band at blocks {sorted(set(S4['blocks']) - set(f.bases))}: {j.res['matched']}")
    on = RD.control(j, c, f)
    RD.do_nothing(j, c, f, on, S4["blocks"][0])
    say("stage4 teeth", on)
    return f, on


def train(j: Any, m: str, w: dict[str, Any], seed: int, P: dict[str, Any]) -> ST.DW:
    """E54's recipe as `keydep.finetune` (anchors.FT_LR, ArmAConfig's batch 32, KL 1, clip 1), stopped at the read step; the batch
    sampler is a generator of the seed alone, so the first steps equal E58's 300-step run's."""
    n, ref = P["settings"]["steps"], P["reference"]["p_heldout"][f"{m}_s{seed}"]
    cfg = ArmAConfig(o_token=w["table"], lr=AN.FT_LR, steps=n, checkpoints=(n,), seed=seed, ci_len=w.get("ci_len"))
    ft = finetune_arm_a(j.net.model, w["train"], w["heldout"], cfg, j.net.device, log)
    p = float(ft.evals[n]["p_o"])
    j.res.setdefault("trained", {})[f"{m}_s{seed}"] = rec = {"p_heldout": p, "ref": ref, "reproduced": abs(p - ref) <= P["thresholds"]["repro_tol"]}
    say(f"stage4 trained {m}_s{seed}", rec)
    if not rec["reproduced"]:
        stop(j, f"{m}_s{seed}: held-out P(target) {p} at step {n} against E58's {ref}")
    path = LARGE / f"dW_{m}_s{seed}.pt"
    torch.save({k: v.bfloat16() for k, v in ft.deltas[n].items()}, path)
    rec["file"], rec["sha256"] = path.name, sha256(path)
    return {k: v.bfloat16().float() for k, v in ft.deltas[n].items()}


def gain(j: Any, s: Any, dW: ST.DW, w0: Any) -> float:
    return float((AP.reads(j.net, s, dW)[1] - w0["clean"][1]).mean())


def write(j: Any, name: str, key: str, s: Any, w: dict[str, Any], G: float) -> ST.DW | None:
    """The hand write of the map's table (`apparatus.hand_write`, unchanged) at `key`, matched to G; None if the match failed."""
    with patch.object(PT, "WRITE_KEY", key):
        A, U = AP.table_inputs(j.net, w["train"].tokens, s.half, *RD.table_of(w))
        dW, rec, _ = AP.hand_write(j.net, A, U, s, G, j.S["write_rel_tol"])
    j.res.setdefault("writes", {})[name] = {**rec, "key": key}
    say(f"stage4 write {name}", j.res["writes"][name])
    return dW


def read(j: Any, name: str, s: Any, dW: ST.DW | None, w0: Any, f: Any) -> dict[str, Any] | None:
    return RD.read_state(j, name, s, dW, w0, f) if dW is not None else None


def reads(j: Any, P: dict[str, Any], st: dict[str, tuple[Any, dict[str, Any]]], states: dict[str, ST.DW], f: Any) -> None:
    """Stages 3-5: block-1 writes (stop on a misread), block-2 writes on the class map's table, then the states."""
    S4, spared = P["settings"], P["thresholds"]["spared_max"]
    w0 = {m: AP.read_all(j.net, s, None, f.bases, f.randoms) for m, (s, _) in st.items()}
    for m, (s, _) in st.items():
        A2.task_shares(j, m, s, f)
    G = {n: gain(j, st[n.rsplit("_", 1)[0]][0], dW, w0[n.rsplit("_", 1)[0]]) for n, dW in states.items()}
    j.res["G"] = G
    say("stage4 G", G)
    b1 = str(S4["blocks"][0])
    for n in states:
        m = n.rsplit("_", 1)[0]
        r = read(j, f"{n}_write", st[m][0], write(j, f"{n}_write", PT.WRITE_KEY, *st[m], G[n]), w0[m], f)
        if r is not None and R4.misread(r[b1]["A_lo"], spared):
            stop(j, f"{n}'s block-1 write loses {r[b1]['A_lo']} of its gain, over {spared}: a known route misread")
    for n in (x for x in states if x.startswith("class_map")):
        read(j, f"{n}_write2", st["class_map"][0], write(j, f"{n}_write2", S4["reader_key"], *st["class_map"], G[n]), w0["class_map"], f)
    for n, dW in states.items():
        read(j, n, st[n.rsplit("_", 1)[0]][0], dW, w0[n.rsplit("_", 1)[0]], f)


def row(j: Any, n: str, b: str) -> dict[str, Any]:
    r = j.res.get("states", {}).get(n, {}).get(b, {})
    return {"A_lo": r.get("A_lo", float("nan")), "band": r.get("band", float("nan")), "random": r.get("random", [])}


def verdicts(j: Any, P: dict[str, Any], on: dict[str, bool], sizes: list[int]) -> dict[str, Any]:
    """R-adm at block 1 (`sizes`: keys per class, E58's), then per block: pooling over the four states, each map's 2 x 2 (its own
    block-1 write the control), the paired test class map minus twin. Block 2 has no reader downstream: its 2 x 2 is printed
    beside, without admissibility."""
    th, seeds, blocks = P["thresholds"], P["settings"]["seeds"], P["settings"]["blocks"]
    b1 = str(blocks[0])
    reads_b = [{"share": row(j, f"class_map_s{s}_write2", b1)["A_lo"], "band": row(j, f"class_map_s{s}_write2", b1)["band"],
                "status": j.res["writes"].get(f"class_map_s{s}_write2", {}).get("status")} for s in seeds]  # fmt: skip
    a_ok = R4.merges(sum(sizes), len(sizes), th["merge_min_per_value"])
    out: dict[str, Any] = {"radm": {"n_inputs": sum(sizes), "n_values": len(sizes), "a": a_ok, "b": R4.reader(reads_b), "b_reads": reads_b}}
    labels = load(EXP / "new_vs_reuse/results/stage0.json")["pairs"]
    for b in map(str, blocks):
        rows = {f"{m}_s{s}": row(j, f"{m}_s{s}", b) for m in MAPS for s in seeds}
        pool = R4.pooling(rows, th["redraw_sets"], th["redraw_seed"])
        per: dict[str, Any] = {"pool": pool, "teeth": on[b], "maps": {}}
        for m in MAPS:
            adm = R4.admissibility(labels[E90_ROW[m]]["label"], a_ok, out["radm"]["b"]) if b == b1 else R4.ADMISSIBLE
            wr = [row(j, f"{m}_s{s}_write", b) for s in seeds]
            ok = [j.res["writes"].get(f"{m}_s{s}_write", {}).get("status") == "ok" for s in seeds]
            states = [(rows[f"{m}_s{s}"]["A_lo"], R4.band_for(rows[f"{m}_s{s}"]["band"], pool)) for s in seeds]
            ctl = [(x["A_lo"], R4.band_for(x["band"], pool), o) for x, o in zip(wr, ok, strict=True)]
            per["maps"][m] = {"admissibility": adm, "states": states, "writes": ctl, **R4.class_verdict(adm, on[b], states, ctl)}
        per["paired"] = R4.paired({f"s{s}": rows[f"class_map_s{s}"] for s in seeds}, {f"s{s}": rows[f"class_shuffled_s{s}"] for s in seeds})
        out[f"block {b}"] = per
    return out


def main() -> None:
    P = filed(PRED4)
    a7 = load(EXP / "open_pair/results/amend7.json")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E58, E90, E95
    S = json.loads(AP.PRED.read_text())["settings"]  # E90's settings, unchanged (n_random, random_seed, pile, gain_floor, write_rel_tol)
    j = RD.Job(ST.Net("cuda"), S, {}, {"sha256": {}, "predictions_sha256": sha256(PRED4)}, OUT)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    log(f"[host] {torch.cuda.get_device_name()} git {head} predictions {j.res['predictions_sha256']}")
    st = sites(j, a7)
    c = capability(j, a7, st["class_map"][0].half)
    f, on = control(j, c, P["settings"])
    states = {f"{m}_s{s}": train(j, m, st[m][1], s, P) for m in MAPS for s in P["settings"]["seeds"]}
    AP.dump(OUT, j.res)
    reads(j, P, st, states, f)
    j.res["verdict"] = verdicts(j, P, on, a7["tables"]["n_all"])
    j.res["complete"] = True
    AP.dump(OUT, j.res)
    say("stage4 verdict", j.res["verdict"])
    log("every claim: a circuit from old parts reads as reuse. E99 stage 4 (d) complete")
    hard_exit(0)


if __name__ == "__main__":
    main()

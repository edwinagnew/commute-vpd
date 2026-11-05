"""E90 stage 0, no model: the admissibility table on every (task, C) pair, from text rules, saved tables and key lists; R4 and
R5's mapping on E70 A1's and E62's saved JSON. Rules: results/predictions.json. A surprise is a finding, listed in
results/stage0.json and printed; it changes no filed rule.
    ~/commute-agents/fleet/cpu-run --mem 6G python experiments/new_vs_reuse/stage0.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rules
import torch

from commute.jobs import hard_exit
from commute.keydep import log
from commute.open_pair import jsonable
from commute.performed import derangement
from commute.removal import band_of, max_band

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
PAIR = EXP / "open_pair" / "results"
LADDER = EXP / "task_ladder" / "results" / "large"
A1, A3, A7 = PAIR / "large" / "amend1", PAIR / "large" / "amend3", PAIR / "large" / "amend7"
RUNGS = ("r0_t352_v310", "r0_t588_v320", "r1_t534_v369", "r1_t588_v452", "r2_t275_v2667", "r2_t359_v816")
Row = dict[str, Any]


def load(p: Path) -> Any:
    return torch.load(p, weights_only=False, map_location="cpu")


def key_table(p: Path) -> tuple[np.ndarray, np.ndarray]:
    t = load(p)["task"]
    return np.array(t.keys), np.array(t.values)


def by_key(keys: np.ndarray, table: dict[int, int]) -> np.ndarray:
    return np.array([table[int(k)] for k in keys])


def identity_rows(out: dict[str, Row]) -> None:
    for name, p in (("T4 ident32", EXP / "regating_table/results/large/windows_ident32.pt"), ("T4 ident32b", EXP / "regating_table/results/large/windows_ident32b.pt"),
                    ("T13 copy-24", A1 / "windows_copy_trained.pt"), ("T6 rand32", EXP / "regating_table/results/large/windows_rand32.pt"),
                    ("T6 rand32b", EXP / "regating_table/results/large/windows_rand32b.pt"), ("T14 prior cipher 96", PAIR / "large/amend6/windows_prior_cipher_trained.pt"),
                    ("T16 shuffled elicit 96", A3 / "windows_shuffle_trained.pt")):  # fmt: skip
        k, v = key_table(p)
        out[f"{name} | token identity"] = rules.parity(k, v)
    for n in (50, 500):  # structural: any derangement reads the same on a parity split
        ids = list(range(n))
        pi = derangement(ids, 0)
        out[f"T8 cipher {n} | token identity"] = {**rules.parity(np.array(ids), by_key(np.array(ids), pi)), "source": "structural, a stand-in derangement"}


def guess_rows(out: dict[str, Row]) -> None:
    k, v = key_table(A1 / "windows_rank1_trained.pt")
    guess = dict(zip(k.tolist(), v.tolist(), strict=True))
    count = lambda x: {"n_keys": len(x), "n_distinct": len(set(x.tolist())), "n_commonest": int(np.bincount(x).max())}
    out["T12 rank-1 24 | W0's guess"] = {**rules.parity(v, v), **count(v)}
    _, v32 = key_table(A1 / "windows_rank1.pt")
    out["T12 rank-1 32 | W0's guess"] = {**rules.parity(v32, v32), **count(v32)}
    for name, p in (("T11 cipher 24", PAIR / "large/windows_cipher_trained.pt"), ("T11 elicit 24", PAIR / "large/windows_elicit_trained.pt")):
        ka, a = key_table(p)
        out[f"{name} | W0's guess"] = rules.parity(by_key(ka, guess), a)
    kg, vg = key_table(A3 / "windows_rank1.pt")
    guess128 = dict(zip(kg.tolist(), vg.tolist(), strict=True))
    for name, p in (("T14 prior cipher 96", PAIR / "large/amend6/windows_prior_cipher_trained.pt"), ("T16 shuffled elicit 96", A3 / "windows_shuffle_trained.pt")):
        ka, a = key_table(p)
        out[f"{name} | W0's guess"] = rules.parity(by_key(ka, guess128), a)


def e62_rows(out: dict[str, Row]) -> None:
    w = load(EXP / "induction_cue/results/large/windows.pt")
    t = {m: {s: w["target"][m][s].numpy() for s in ("train", "read")} for m in ("copy", "lookup")}
    b = t["copy"]
    for m, name in (("copy", "T5 copy"), ("lookup", "T5 lookup")):
        out[f"{name} | in-context copying"] = {**rules.admissibility(b["train"], t[m]["train"], b["read"], t[m]["read"]),
                                               "parity_inside_read": rules.parity(b["read"], t[m]["read"])}  # fmt: skip


def constant_rows(out: dict[str, Row]) -> None:
    def windows(name: str, held: Any, target: int) -> None:
        toks = held.tokens[:, held.half].numpy()
        out[f"{name} | any"] = rules.parity(toks, np.full(len(toks), target))

    for r in RUNGS:
        w = load(LADDER / f"windows_{r}.pt")
        windows(f"T2 {r}", w["heldout"], w["task"]["target"])
    windows("T1 emoticon", load(EXP / "transport_test/results/large/windows.pt")["heldout"], 80)
    m = load(EXP / "many_cues/results/large/windows.pt")
    windows("T3 many-to-one", m["heldout"], m["mine"]["target"])
    rand1 = load(EXP / "kv_task/results/large/windows_rand1.pt")
    windows("T6 rand1", rand1["heldout"], int(rand1["task"].values[0]))
    out["T7 lowercase pair | any"] = {**rules.parity(np.ones(64, dtype=int), np.ones(64, dtype=int)), "source": "structural: the scored pair's answer is always 'lower'"}


def class_rows(out: dict[str, Row]) -> None:
    cls = {int(k): int(v) for k, v in json.loads((PAIR / "amend7.json").read_text())["tables"]["class_of"]["class_map"].items()}
    for name, p in (("T15 class map 128", A7 / "windows_class_map.pt"), ("T15 twin 128", A7 / "windows_class_shuffled.pt")):
        k, v = key_table(p)
        out[f"{name} | embedding class"] = rules.parity(by_key(k, cls), v)


def edge_rows() -> dict[str, Row]:
    one = np.array([7])
    return {"A5 rank-1 with one distinct value": rules.parity(np.full(24, 15), np.full(24, 15)), "A9 K = 1": rules.parity(one, one),
            "A9 empty": rules.parity(np.empty(0, dtype=int), np.empty(0, dtype=int))}  # fmt: skip


def r4() -> dict[str, Any]:
    e70 = json.loads((EXP / "pcfg_anchors/results/e70_a1.json").read_text())
    has = e70["r1"]["teeth"]["pattern"]
    out = {}
    for row, t in zip(e70["r2"]["rows"], has, strict=True):
        p, n = list(row["shares"]["P_10000"].values()), list(row["shares"]["N_10000"].values())
        out[str(row["block"])] = {**rules.all_pairs(t, p, n, row["band"]), "band": row["band"], "P": [min(p), max(p)], "N": [min(n), max(n)]}
        out[str(row["block"])].pop("per_seed")
    return out


def r5() -> dict[str, Any]:
    st = json.loads((EXP / "induction_cue/results/run.json").read_text())["states"]
    band = max_band([band_of(s["random_shares"]) for s in st.values()])
    out: dict[str, Any] = {"band_from_5_saved_draws": band, "teeth": "not read: taken as true for the mapping"}
    for a, b in (("copy", "lookup"), ("lookup", "copy")):
        per = [rules.verdict(True, st[f"{a}_s{s}"]["share"], st[f"{b}_s{s}"]["share"], band) for s in range(3)]
        out[a] = {**rules.over_seeds(per), "shares": [st[f"{a}_s{s}"]["share"] for s in range(3)]}
    return out


def surprises(filed: dict[str, Any], pairs: dict[str, Row], edges: dict[str, Row], rr4: dict[str, Any], rr5: dict[str, Any]) -> list[str]:
    out = [f"{k}: filed {v}, read {pairs[k]['label']} (best {pairs[k]['best']}, floor {pairs[k]['floor']})"
           for k, v in filed["admissibility"]["pairs"].items() if not rules.matches(v, pairs[k])]  # fmt: skip
    if edges["A5 rank-1 with one distinct value"]["admissible"] or not all(edges[k]["label"] == rules.NO_HELD_OUT for k in ("A9 K = 1", "A9 empty")):
        out.append(f"A5 or A9 did not read as filed: {edges}")
    out += [f"R4 block {b}: read {r['verdict']}" for b, r in rr4.items() if r["verdict"] != rules.NOT_SPECIFIC]
    if rr5["copy"]["verdict"] != rules.REUSES or rr5["lookup"]["verdict"] != rules.NOT_THROUGH:
        out.append(f"R5: copy {rr5['copy']['verdict']}, lookup {rr5['lookup']['verdict']}")
    return out


def plan_text_checks(pairs: dict[str, Row]) -> list[str]:
    """Numbers the plan's prose quotes, against the tables. No verdict hangs on them."""
    r24, r32, rand = pairs["T12 rank-1 24 | W0's guess"], pairs["T12 rank-1 32 | W0's guess"], pairs["T6 rand32 | token identity"]
    lookup = pairs["T5 lookup | in-context copying"]
    return [f"A1 quotes the constant at ~1/16; read {rand['floor']} (best {rand['best']}): a bijection's held-out keys never hold the fit half's answer, so A1 is a tie, not a loss",
            f"A4 quotes 18 values on 24 keys, one token on 9; read {r24['n_distinct']} values and {r24['n_commonest']} on 24 keys, {r32['n_distinct']} and {r32['n_commonest']} on 32; floor {r24['floor']}",
            f"A7's parity reading inside the read windows: best {lookup['parity_inside_read']['best']}, floor {lookup['parity_inside_read']['floor']}, {lookup['parity_inside_read']['label']}; filed split: best {lookup['best']}, floor {lookup['floor']}",
            "R5: the plan says E62 saved no state-level band; run.json holds 5 random_shares per state, used here, not the filed 20"]  # fmt: skip


def main() -> None:
    filed = json.loads((RESULTS / "predictions.json").read_text())
    pairs: dict[str, Row] = {}
    for f in (e62_rows, guess_rows, identity_rows, constant_rows, class_rows):
        f(pairs)
    edges, rr4, rr5 = edge_rows(), r4(), r5()
    found, notes = surprises(filed, pairs, edges, rr4, rr5), plan_text_checks(pairs)
    res = {"pairs": pairs, "edges": edges, "R4": rr4, "R5": rr5, "surprises": found, "plan_text_checks": notes,
           "stage2_pairs": [k for k, r in pairs.items() if r["admissible"]]}  # fmt: skip
    (RESULTS / "stage0.json").write_text(json.dumps(jsonable(res), indent=1, sort_keys=True))
    for k, r in {**pairs, **edges}.items():
        log(f"{k}: {r['label']}; best {r['best']:.4f} floor {r['floor']:.4f} n_read {r['n_read']}")
    log(f"R4 {json.dumps(jsonable(rr4))}")
    log(f"R5 {json.dumps(jsonable(rr5))}")
    for s in notes:
        log(f"plan text: {s}")
    log(f"SURPRISES against the filed verdicts: {found or 'none'}")
    hard_exit(0)


if __name__ == "__main__":
    main()

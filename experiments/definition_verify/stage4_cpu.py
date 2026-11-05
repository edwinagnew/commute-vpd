"""E99 stage 4, the parts read from stored files: no model, CPU. Plan: reports/plans/2026-11-05_plan_e99-refine-verify-definition.md
(stage 4); the stage 0-3 report's "What next" item 1. Rules: rules4.py; expected values: results/predictions_stage4.json (committed
first; the script refuses otherwise; never edited).
  (a) copy against lookup at the copying pair (2,3)+(2,4) through R-net: the 2 x 2 against own bands, read only if stage 1's
      forced copy reads net 1 (the positive control) and stage 2's forbidden copy within band (the negative); paired test beside.
  (b) R-adm on every value-form site E90 and E95 read: clause (a) from the capability's counts; clause (b) from E95's block-2 key
      reader (token identity, table A's keys only). The class map's site is stage4.py's.
  (c) every old row's per-state band (R-band): each state and its spared control against its own band, the 2 x 2 per seed, the
      pooling clause per C and block, and E95's paired test on each data-built pair. E90 Amendment 2's states (blocks 1, 2), E95's
      replicate (block 1), E99's natives at the two located pairs.
Writes results/stage4_cpu.json and prints every number. Usage (from the clone root):
  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/definition_verify/stage4_cpu.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import rules4 as R4

from commute.jobs import hard_exit
from commute.keydep import log
from commute.locate import from_saved
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, EXP, REPO = HERE / "results", HERE.parent, HERE.parents[1]
PRED4, OUT = RESULTS / "predictions_stage4.json", RESULTS / "stage4_cpu.json"
SEEDS2, SEEDS3 = ("s0", "s1"), ("s0", "s1", "s2")
# E90 Amendment 2's value-form sites: the control's name, the maps and their seeds, the data-built pairs (first minus second)
E90_FORMS = {"guess": ("W0's guess", {"rank1": SEEDS3, "cipher": SEEDS3, "elicit": SEEDS3}, [("rank1", "cipher"), ("elicit", "cipher")]),
             "identity": ("token identity", {"ident32": SEEDS2, "rand32": SEEDS2}, [("ident32", "rand32")]),
             "copy24": ("token identity, copy-24's keys", {"copy24": SEEDS3}, [])}  # fmt: skip
E95_FORM = ("token identity, table B's keys", {"ident32b": SEEDS2, "rand32b": SEEDS2}, [("ident32b", "rand32b")])
NATIVES = {"copy pair": ("T5 copy", "T5 lookup"), "lowercase pair": ("T7 lowercase", "T8 cipher")}  # state, spared control


def load(path: Path) -> dict[str, Any]:
    return from_saved(json.loads(path.read_text()))


def filed(pred: Path) -> dict[str, Any]:
    """The filing, refused unless committed unchanged (as run.filed, without run.py's model imports)."""
    rel = str(pred.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not pred.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    return json.loads(pred.read_text())


def say(tag: str, rec: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(rec))}")


# ---- (c) per-state bands ----


def per_seed(states: dict[str, Any], m: str, seeds: tuple[str, ...], b: str, write_ok: dict[str, bool]) -> dict[str, dict[str, Any]]:
    """Each seed's state and its own block-b write, as rows {"A_lo", "band", "random"} plus the write's read and match."""
    out = {}
    for s in seeds:
        st, w = states[f"{m}_{s}"][b], states[f"{m}_{s}_write"][b]
        out[s] = {"A_lo": st["A_lo"], "band": st["band"], "random": st["random"],
                  "write": {"share": w["A_lo"], "band": w["band"], "ok": write_ok.get(f"{m}_{s}", False)}}  # fmt: skip
    return out


def read_block(maps: dict[str, dict[str, dict[str, Any]]], teeth: bool, pairs: list[tuple[str, str]], th: dict[str, Any]) -> dict[str, Any]:
    """One C at one block: pooling over every state, then each state and its control against the band used, the 2 x 2 per seed and
    over seeds, and the paired test per data-built pair."""
    pool = R4.pooling({f"{m}_{s}": r for m, rows in maps.items() for s, r in rows.items()}, th["redraw_sets"], th["redraw_seed"])
    out: dict[str, Any] = {"teeth": teeth, "pool": pool, "maps": {}}
    for m, rows in maps.items():
        for r in rows.values():
            w = r["write"]
            r["band_used"], w["band_used"] = R4.band_for(r["band"], pool), R4.band_for(w["band"], pool)
            r["status"], w["status"] = R4.own(r["A_lo"], r["band_used"]), R4.own(w["share"], w["band_used"])
            r["verdict"] = R4.verdict2(teeth, r["A_lo"], r["band_used"], w["share"], w["band_used"], w["ok"])
        out["maps"][m] = {"seeds": {s: {k: v for k, v in r.items() if k != "random"} for s, r in rows.items()},
                          **R4.over_seeds([r["verdict"] for r in rows.values()])}  # fmt: skip
    out["paired"] = {f"{a} - {b}": R4.paired(maps[a], maps[b]) for a, b in pairs}
    return out


def e90_rows(a2: dict[str, Any], s2: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    ok = {n: w["status"] == "ok" for n, w in s2["writes"].items()}
    out: dict[str, Any] = {}
    for form, (cname, maps, pairs) in E90_FORMS.items():
        for b in ("1", "2"):
            rows = {m: per_seed(a2["states"], m, seeds, b, ok) for m, seeds in maps.items()}
            out[f"{form} block {b}"] = read_block(rows, bool(a2["control"][cname][b]["teeth"]), pairs, th)
    return out


def e95_rows(e95: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    cname, maps, pairs = E95_FORM
    ok = {n: w["status"] == "ok" for n, w in e95["writes"].items()}
    rows = {m: per_seed(e95["states"], m, seeds, "1", ok) for m, seeds in maps.items()}
    return {"identity, table B block 1": read_block(rows, bool(e95["control"][cname]["1"]["teeth"]), pairs, th)}


def native(rec: dict[str, Any]) -> dict[str, Any]:
    """An E99 native at its pair as a row: net share, own band, the random pairs' net shares in name order."""
    return {"A_lo": rec["net"]["lo"], "band": rec["band"], "random": [rec["random_net"][d] for d in sorted(rec["random_net"])]}


def native_rows(s0: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """The mechanism form: each native against its spared control at the same seed (copy's is the lookup, lowercase's the
    cipher). Teeth stands for R-net's own controls, read in (a); here it is True and printed as such."""
    out = {}
    for site, (state, ctl) in NATIVES.items():
        rows = {}
        for m, other in ((state, ctl), (ctl, state)):
            rows[m] = {}
            for s in SEEDS2:
                c = native(s0["natives"][f"{other} {s}"])
                rows[m][s] = {**native(s0["natives"][f"{m} {s}"]), "write": {"share": c["A_lo"], "band": c["band"], "ok": True}}
        out[site] = read_block(rows, True, [(state, ctl)], th)
    return out


# ---- (a) and (b) ----


def copy_lookup(s1: dict[str, Any], s2: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    """R-net's verdict on copy with the lookup as its spared control, from (c)'s rows (net share, band used); its two controls; (c)'s
    paired test and pooling beside."""
    seeds = c["copy pair"]["maps"]
    rows = lambda m: [(seeds[m]["seeds"][s]["A_lo"], seeds[m]["seeds"][s]["band_used"]) for s in SEEDS2]
    forced = [s1["forced copy"][s]["reading"] for s in SEEDS2]
    forbidden = [s2["forbidden copy"][s]["reading"] for s in SEEDS2]
    v = R4.rnet_verdict(rows("T5 copy"), rows("T5 lookup"), forced, forbidden)
    return {**v, "copy": rows("T5 copy"), "lookup": rows("T5 lookup"), "forced": forced, "forbidden": forbidden,
            "paired": c["copy pair"]["paired"]["T5 copy - T5 lookup"], "pool": c["copy pair"]["pool"]}  # fmt: skip


def radm(a2: dict[str, Any], e95: dict[str, Any], a7: dict[str, Any], n90: dict[str, Any], th: dict[str, Any]) -> dict[str, Any]:
    """Per value-form site: clause (a) on (inputs, values), clause (b) from the reads that exist, and the row's admissibility."""
    rank = lambda ctl, name: int(ctl[name]["rank"]) + 1  # token identity is injective: one input per value
    gp, cls = a2["guess_positions"], a7["tables"]["n_all"]
    e95_reads = [{"share": e95["states"][f"{n}_write"]["1"]["A_lo"], "band": e95["states"][f"{n}_write"]["1"]["band"], "status": e95["writes"][n]["status"]}
                 for n in ("ident32_s0", "ident32_s1", "rand32_s0", "rand32_s1")]  # fmt: skip
    sites = {"token identity, table A's keys": (rank(a2["control"], "token identity"),) * 2 + (e95_reads,),
             "token identity, table B's keys": (rank(e95["control"], "token identity, table B's keys"),) * 2 + ([],),
             "token identity, copy-24's keys": (rank(a2["control"], "token identity, copy-24's keys"),) * 2 + ([],),
             "W0's guess": (int(gp["fit_windows"]), len(gp["fit_values"]), []),
             "embedding class": (sum(cls), len(cls), None)}  # fmt: skip
    rows = {"token identity, table A's keys": ["T4 ident32", "T6 rand32"], "token identity, table B's keys": ["T4 ident32b", "T6 rand32b"],
            "token identity, copy-24's keys": ["T13 copy-24"], "W0's guess": ["T12 rank-1 24", "T11 cipher 24", "T11 elicit 24"],
            "embedding class": ["T15 class map 128", "T15 twin 128"]}  # fmt: skip
    label = lambda row, c: n90["pairs"][f"{row} | {c.split(',')[0]}"]["label"]
    out = {}
    for site, (n_in, n_val, reads) in sites.items():
        ok = R4.merges(n_in, n_val, th["merge_min_per_value"])
        b = "stage4.py" if reads is None else R4.reader(reads)
        out[site] = {"n_inputs": n_in, "n_values": n_val, "per_value": n_in / n_val, "a": ok, "b": b, "b_reads": reads,
                     "rows": {r: {"e90": label(r, site), "admissibility": None if reads is None else R4.admissibility(label(r, site), ok, b)} for r in rows[site]}}  # fmt: skip
    return out


def main() -> None:
    P = filed(PRED4)
    th = P["thresholds"]
    s0, s1, s2 = (load(RESULTS / f"stage{k}.json") for k in range(3))
    a2, n2 = load(EXP / "new_vs_reuse/results/amend2.json"), load(EXP / "new_vs_reuse/results/stage2.json")
    n90, e95, a7 = load(EXP / "new_vs_reuse/results/stage0.json"), load(EXP / "token_identity/results/run.json"), load(EXP / "open_pair/results/amend7.json")
    if not (s0["status"] == "pass" and a2.get("complete") and e95.get("complete")):
        sys.exit("an input stage has not passed or completed: stage0.json, amend2.json, run.json")
    c = {**native_rows(s0, th), **e90_rows(a2, n2, th), **e95_rows(e95, th)}
    out = {"a_copy_lookup": copy_lookup(s1, s2, c), "b_radm": radm(a2, e95, a7, n90, th), "c_bands": c}
    for k, v in out["c_bands"].items():
        say(f"(c) {k}", {"teeth": v["teeth"], "pool": v["pool"], "maps": {m: {"verdict": x["verdict"], "per_seed": x["per_seed"]} for m, x in v["maps"].items()}})
        for m, x in v["maps"].items():
            say(f"(c) {k} {m} seeds", x["seeds"])
        say(f"(c) {k} paired", {p: {"call": x["call"], "pairs": x["pairs"]} for p, x in v["paired"].items()})
    say("(a) copy against lookup", out["a_copy_lookup"])
    for site, r in out["b_radm"].items():
        say(f"(b) {site}", r)
    OUT.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    log(f"wrote {OUT.relative_to(REPO)}; a circuit from old parts reads as reuse")
    hard_exit(0)


if __name__ == "__main__":
    main()

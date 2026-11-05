"""E99 stages 0-3: does the net share at a located part read 1 on a fine-tune forced through the part and about 0 on one
forbidden from it, on the 67M's copying pair and lowercase pair? Plan: reports/plans/2026-11-05_plan_e99-refine-verify-definition.md;
mechanics: reports/reviews/2026-11-05_review_e99-fable-advice.md. Rules and expected values: results/predictions.json
(committed first; the run refuses otherwise; never edited), rules.py. Reads: reads.py (E97's `commute.locator`). Training:
arms.py. Sites and saved states: experiments/locator_table/sites.py (sha256-checked).

  stage 0, alone: W0's own fall under each pair on its four sites must reproduce E97's within twice its row SE, the native
           states' G and own share must reproduce E97's, and the lookup must be spared at the copy pair; exit 1 on any
           miss. Beside: the natives' net shares and bands, the cipher's increment with the pair off, and R-net's clause
           (ii) (the lowercase pair moves W0's lowercase-minus-capitalised gap on cased rows and on lowercased rows), which
           decides whether stage 3 runs.
  stage 1: forced-part copy, seeds 0 and 1. A net share outside 1 +- forced_tol on a state with gain stops the job.
  stage 2: forbidden-part copy, read with the pair restored; beside: the wider copying set off, G restored / G native.
  stage 3: forced-part and forbidden-part lowercase (only if clause (ii) passed); a forced net share off 1 stops the job.

Writes results/stage{0,1,2,3}.json as each state finishes; <large>/dW_<mode>_<task>_s<seed>.pt (bf16, read as saved) and
<large>/reads_<tag>.pt (the part's per-read tensors). Every scored number is printed. Deletes nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import Any, NoReturn

import arms as AR
import reads as RD
import rules as R
import torch
from torch import Tensor

from commute import copying as CP
from commute import locator as L
from commute import lowercase as LC
from commute import steering as ST
from commute import vpd
from commute.data import pile_rows
from commute.jobs import hard_exit, load_experiment
from commute.keydep import log, sha256
from commute.locate import from_saved
from commute.open_pair import jsonable
from commute.row_copy import lowercase_rows

sites = load_experiment("locator_table", "sites")
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
STAGE = {k: RESULTS / f"stage{k}.json" for k in range(4)}
PLANS = {1: [("copy", "forced")], 2: [("copy", "forbidden")], 3: [("lowercase", "forced"), ("lowercase", "forbidden")]}
NATIVES = (("T5 copy", "e62_copy", "copy"), ("T5 lookup", "e62_lookup", "copy"), ("T7 lowercase", "e53_lowercase", "lowercase"), ("T8 cipher", "e53_cipher", "lowercase"))


@dataclass(frozen=True)
class Cfg:
    seeds: list[int]
    copy_pair: str
    lower_pair: str
    wide_copy: str
    copy_steps: int
    lower_steps: int
    cap_factor: int
    copy_every: int
    lower_every: int
    lower_match_rows: int
    match_rel: float
    n_random: int
    random_seed: int
    lower_lr: float
    lower_batch: int
    induction: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.lower_steps == LC.STEPS and self.lower_batch == LC.BATCH, "predictions.json disagrees with E53's recipe"
        for steps, every in ((self.copy_steps, self.copy_every), (self.lower_steps, self.lower_every)):
            assert steps % every == 0 and (steps * self.cap_factor) % every == 0, "the native step and the cap must be checks"
        assert set(self.seeds) <= {0, 1} and set(L.parts_of(self.copy_pair)) <= set(L.parts_of(self.wide_copy))

    @classmethod
    def of(cls, s: dict[str, Any]) -> Cfg:
        return cls(**{f.name: s[f.name] for f in fields(cls)})


@dataclass(frozen=True)
class Task:
    site: str  # the native's site key (sites.site)
    native: str  # sites.TASKS row of the native states
    pair: str
    steps: int  # the native's steps
    every: int
    rows: int | None  # the matched-gain check's first inputs; None: all


@dataclass
class Job:
    net: ST.Net
    P: dict[str, Any]
    c: Cfg
    F: Any = field(default_factory=lambda: sites.Files())
    bases: dict[str, L.Base] = field(default_factory=dict)

    @property
    def th(self) -> dict[str, Any]:
        return self.P["thresholds"]

    @cached_property
    def tasks(self) -> dict[str, Task]:
        c = self.c
        return {"copy": Task("e62_copy", "T5 copy", c.copy_pair, c.copy_steps, c.copy_every, None),
                "lowercase": Task("e53_lowercase", "T7 lowercase", c.lower_pair, c.lower_steps, c.lower_every, c.lower_match_rows)}  # fmt: skip

    @cached_property
    def draws(self) -> dict[str, list[str]]:
        return {p: RD.random_pairs(p, self.c.n_random, self.c.random_seed) for p in (self.c.copy_pair, self.c.lower_pair)}

    def base(self, key: str) -> L.Base:
        if key not in self.bases:
            s = RD.cased_site(sites.E53_ROWS) if key == "cased" else sites.site(self.F, key)
            self.bases[key] = RD.site_base(self.net, s)
        return self.bases[key]

    def native(self, row: str, seed: int, key: str) -> L.State:
        return L.state(self.net, self.base(key), self.F.state(sites.TASKS[row].states[seed]))


def dump(path: Path, rec: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(rec), indent=1, sort_keys=True))


def say(tag: str, rec: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(rec))}")


def stop(path: Path, rec: dict[str, Any], why: str) -> NoReturn:
    rec["stopped"] = why
    dump(path, rec)
    log(f"STOP: {why}: tell the lead (exit 1)")
    hard_exit(1)


def save_reads(tag: str, t: dict[str, Tensor]) -> None:
    torch.save({k: v.float() for k, v in t.items()}, LARGE / f"reads_{tag}.pt")


# ----------------------------------------------------------------------------- stage 0


def clause_ii(J: Job) -> dict[str, Any]:
    """W0's lowercase-minus-capitalised gap moved by the lowercase pair, on cased and on lowercased rows, each against the
    same random pairs' band of W0's move there."""
    p, res = J.c.lower_pair, {}
    for key in ("cased", "e53_lowercase"):
        b = J.base(key)
        move = lambda n, b=b: float((b.lo0 - L.w0_off(J.net, b, n)).mean(0)[1])
        rnd = {d: move(d) for d in J.draws[p]}
        res[key] = {"move_gap": move(p), "random_moves_gap": rnd, "band": R.band(list(rnd.values()))}
    res["passes"] = R.clause_ii(res["cased"]["move_gap"], res["cased"]["band"], res["e53_lowercase"]["move_gap"], res["e53_lowercase"]["band"])
    return res


def native_reads(J: Job, out: dict[str, Any], misses: list[str]) -> None:
    ref, path = J.P["reference"]["natives"], STAGE[0]
    for row, key, task in NATIVES:
        for s in J.c.seeds:
            pair, name = J.tasks[task].pair, f"{row} s{s}"
            rec, t = RD.state_read(J.net, J.native(row, s, key), pair, J.draws[pair])
            e, G = ref[name], rec["G"]["lo"]
            rec["reproduced"] = {"G": R.reproduced(G, e["G"], J.th["g_tol"]), "own_share": R.reproduced(rec["own_share"]["lo"], e["own_share"], e["band_2se"])}
            rec["G_part_off_in_both"] = G - rec["fall"]["lo"] + rec["w0_fall"]["lo"]
            out["natives"][name] = rec
            save_reads(f"native_{row.split()[1]}_s{s}", t)
            say(f"stage0 native {name} at {pair}", rec)
            dump(path, out)
            misses += [f"{name} {k} not reproduced" for k, ok in rec["reproduced"].items() if not ok]


def stage0(J: Job) -> None:
    path, out, misses = STAGE[0], {"w0": {}, "natives": {}, "draws": J.draws, "files": J.F.seen}, []
    for key, task in (("e62_copy", "copy"), ("e62_lookup", "copy"), ("e53_lowercase", "lowercase"), ("e53_cipher", "lowercase")):
        r = RD.w0_move(J.net, J.base(key), J.tasks[task].pair)
        r["ref"] = J.P["reference"]["w0_fall"][key]["lo"]
        r["reproduced"] = R.reproduced(r["fall"]["lo"], r["ref"], r["band_2se"])
        out["w0"][key] = r
        say(f"stage0 W0 fall {key} at {J.tasks[task].pair}", r)
        misses += [] if r["reproduced"] else [f"W0 fall on {key}: {r['fall']['lo']} against {r['ref']} +- {r['band_2se']}"]
    dump(path, out)
    native_reads(J, out, misses)
    out["lookup_spared"] = {f"s{s}": R.spared(out["natives"][f"T5 lookup s{s}"]["net"]["lo"], J.th["spared_max"]) for s in J.c.seeds}
    misses += [f"lookup {k} not spared at the copy pair" for k, ok in out["lookup_spared"].items() if not ok]
    out["clause_ii"] = clause_ii(J)
    say("stage0 lookup spared", out["lookup_spared"])
    say("stage0 clause (ii)", out["clause_ii"])
    out["status"], out["missed"] = ("pass" if not misses else "miss"), misses
    dump(path, out)
    if misses:
        stop(path, out, f"stage 0: {misses[:5]}")


# ----------------------------------------------------------------------------- stages 1-3: the constructed arms

Train = Callable[[int, int, AR.Matcher], None]


def ablation(J: Job, b: L.Base, pair: str, mode: str) -> AbstractContextManager[Any]:
    return CP.ablated(J.net, RD.heads_of(pair), RD.means_tensor(b)) if mode == "forbidden" else nullcontext()


def train_arm(J: Job, T: Task, mode: str, seed: int, train: Train) -> tuple[ST.DW, dict[str, Any]]:
    """The arm trained to the native's gain on the first inputs (training condition), its kept delta and match record."""
    b, nat = J.base(T.site), J.native(T.native, seed, T.site)
    sub = RD.first_inputs(b.site, T.rows)
    target = RD.gain(J.net, sub, L.read(J.net, sub), nat.dW)
    with ablation(J, b, T.pair, mode):
        lo0 = L.read(J.net, sub)
        if mode == "forbidden" and not AR.ref_is_ablated(J.net, sub.toks[:4]):
            raise RuntimeError("the KL reference would not carry the ablation")
        m = AR.Matcher(J.net, sub, lo0, target, T.every, J.c.match_rel, AR.free_masks(J.net.W0, RD.heads_of(T.pair), mode))
        (step, g, dW), status = AR.run(m, lambda m: train(seed, T.steps * J.c.cap_factor, m))
    return dW, {"step": step, "status": status, "gain_first_inputs": g, "target_first_inputs": target, "curve": m.curve, "G_native": float(nat.G[0])}


def arm(J: Job, task: str, mode: str, seed: int, train: Train) -> dict[str, Any]:
    T, th, tag = J.tasks[task], J.th, f"{mode}_{task}_s{seed}"
    t0 = time.monotonic()
    dW, match = train_arm(J, T, mode, seed, train)
    path = LARGE / f"dW_{tag}.pt"
    torch.save({k: v.bfloat16() for k, v in dW.items()}, path)
    st = L.state(J.net, J.base(T.site), {k: v.bfloat16().float() for k, v in dW.items()})
    rec, t = RD.state_read(J.net, st, T.pair, J.draws[T.pair])
    save_reads(tag, t)
    G, net = rec["G"]["lo"], rec["net"]["lo"]
    rec |= match | {"file": path.name, "sha256": sha256(path), "G_over_native": G / match["G_native"],
                    "G_part_off_in_both": G - rec["fall"]["lo"] + rec["w0_fall"]["lo"]}  # fmt: skip
    if mode == "forced":
        rec["reading"] = R.forced_reading(net, th["forced_tol"])
        if task == "copy":  # beside: repurposing inside the part (Fable); W0's own is E62's top1_base
            with J.net.at(st.dW):
                rec["induction_top1_second"] = CP.induction(J.net, J.c.induction)["top1_second"]
    else:
        rec["reading"] = R.forbidden_reading(net, rec["band"])
        rec["restored_within"] = R.restored_ok(G, match["G_native"], th["restored_rel"])
        if task == "copy":
            rec["wide_set_off"] = RD.pair_read(J.net, st, J.c.wide_copy)[0]
        else:
            rec["pace"] = R.pace(match["step"], T.steps, match["curve"].get(T.steps, math.nan), match["target_first_inputs"], match["status"], th["pace_rel"], th["slow_factor"])
    rec["seconds"] = time.monotonic() - t0
    return rec


def stage_arms(J: Job, k: int, trainers: dict[str, Train], path: Path, out: dict[str, Any]) -> None:
    """Stage k's arms (PLANS[k]), both seeds each, written to `path` (with `out`'s keys) after every arm."""
    for task, mode in PLANS[k]:
        for s in J.c.seeds:
            rec = out.setdefault(f"{mode} {task}", {})[f"s{s}"] = arm(J, task, mode, s, trainers[task])
            dump(path, out)
            say(f"stage{k} {mode} {task} s{s}", {x: v for x, v in rec.items() if x != "curve"})
            say(f"stage{k} {mode} {task} s{s} curve", rec["curve"])
            if mode == "forced" and rec["reading"] == R.OFF_ONE:
                stop(path, out, f"{mode} {task} s{s}: net share {rec['net']['lo']} is off 1 by more than {J.th['forced_tol']}")


def trainers(J: Job, want: set[int]) -> dict[str, Train]:
    win = torch.load(J.F.checked(sites.CUE / "large" / "windows.pt"), weights_only=False)
    half = json.loads((sites.CUE / "predictions.json").read_text())["settings"]["half"]
    out: dict[str, Train] = {"copy": lambda s, n, m: AR.copy_train(J.net, win, half, s, n, m)}
    if 3 in want:
        rows = lowercase_rows(pile_rows("train", 0, 4096, log=log), vpd.tokenizer(), LC.STEPS * LC.BATCH)  # as `lowercase.task`
        out["lowercase"] = lambda s, n, m: AR.lower_train(J.net, rows, J.c, s, n, m)
    return out


# ----------------------------------------------------------------------------- main


def filed(pred: Path) -> dict[str, Any]:
    rel = str(pred.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not pred.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    return json.loads(pred.read_text())


def start(P: dict[str, Any], device: str, what: str) -> Job:
    """The model loaded under the filed settings; `what` goes on the host line."""
    if not LARGE.is_dir():
        sys.exit(f"{LARGE} is missing: run ~/commute-agents/fleet/link-data <role> first (it is a symlink into commute-data)")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E96, E97
    J = Job(ST.Net(device), P, Cfg.of(P["settings"]))
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    log(f"[host] {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} git {head} predictions {sha256(PRED)} {what}")
    return J


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stages", default="0,1,2,3")
    args = ap.parse_args()
    want = {int(x) for x in args.stages.split(",")}
    P = filed(PRED)
    s0 = from_saved(json.loads(STAGE[0].read_text())) if STAGE[0].exists() else {}
    if 0 not in want and s0.get("status") != "pass":
        sys.exit("stage 0 has not passed: run it first")
    J = start(P, args.device, f"stages {sorted(want)}")
    if 0 in want:
        t0 = time.monotonic()
        stage0(J)
        s0 = from_saved(json.loads(STAGE[0].read_text()))
        log(f"[stage 0 done] {time.monotonic() - t0:.0f} s")
    T = trainers(J, want)
    for k in PLANS:
        if k not in want:
            continue
        if k == 3 and not s0["clause_ii"]["passes"]:
            dump(STAGE[3], {"skipped": "clause (ii) failed in stage 0: lowercase stays 'does not apply'; arms 3-4 dropped (Fable's table)"})
            log("stage 3 does not run: clause (ii) failed (the plan's tree)")
            continue
        t0 = time.monotonic()
        stage_arms(J, k, T, STAGE[k], {})
        log(f"[stage {k} done] {time.monotonic() - t0:.0f} s")
    hard_exit(0)


if __name__ == "__main__":
    main()

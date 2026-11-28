"""Cell P2.91 run 2 (E333): do the Qwen fine-tunes' re-gating shares move with row packing, as E332's repeat of E330's
VPD known draw 2 did (1.03 on E330's rows, 1.21 on the known rows repacked with packing seed 1)? Each known and copy
fine-tune (seeds 0-2, the last save; fetched as E332 fetched them) is read on its task's rows repacked with
`repack_seed` (`share_reading.marked_rows(task, repack_seed)`) at the two scales E332 read it at: c = 1 and the c E332's
gain-1 read recorded (results/qwen_gain.jsonl). No new search: the update is held fixed, so only the rows change; the
gain the repacked rows give at that c is recorded. VPD p-9707d6ff only. The reads are E332's (`qwen_gain`, E330's
`qwen_random.full_read`), with the same row marks and whole-answer spans.

  1 controls  per repacked row set: a zero change and W0 itself read search gain, full gain and numerator exactly 0
              (`qwen_gain.controls`); a miss stops the job.
  2 repeat    E332's known draw-2 repeat (its draw seed, the norms of known_s2_last) read at E332's c on known_repack1;
              share and gain must equal E332's within repeat_tol (0, measured: see the settings). known_s2 is fetched
              first for it. The first read over first_read_max_s stops the job.
  3 reads     per state, c = 1 ("full") and E332's gain-1 c ("g1.05"); each state's 5.6 GB file fetched once (~400 s,
              the next fetched while this one is read). After each state the projected total cost (elapsed + mean
              time per state so far x states left, at usd_per_h) over cost_max_usd stops the job.
One JSON line per item in results/qwen_repack.jsonl as it lands; a rerun skips the items written. A stop writes
results/stopped_repack.json and exits 1. Prints "QWEN_REPACK DONE" at the end. Settings: results/predictions_repack.json.

    PRED_SHA=<sha> python experiments/share_unified/qwen_repack.py --decomp "$COMMUTE_QWEN_NEW_DIR"
    ~/commute-agents/fleet/cpu-run --mem 8G python3 experiments/share_unified/qwen_repack.py --dry   (CPU: rows, items)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import library_states as LS
from commute import share as SH
from commute import share_reading as SR
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.qwen_sft import linears, load
from commute.qwen_vpd import SITES, decomp_id, decomp_kinds, parse_site

G = load_experiment("share_unified", "qwen_gain")  # E332: controls, records, put, Lab; its RQ (E330), RR, QS, L
RQ, RR, QS, L, VPD = G.RQ, G.RR, G.QS, G.L, G.VPD
RESULTS = G.RESULTS
PRED, OUT, STOPPED = RESULTS / "predictions_repack.json", RESULTS / "qwen_repack.jsonl", RESULTS / "stopped_repack.json"
E332_OUT = RESULTS / "qwen_gain.jsonl"
G.OUT, G.STOPPED = OUT, STOPPED  # G.controls, G.put and G.Lab.stop write E333's files; E332's file is only read
Data = dict[str, tuple[Tensor, SH.Spans]]
DW = dict[str, Tensor]


@dataclass(frozen=True)
class Config:
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    save: str
    repack_seed: int
    gain1_stage: str  # E332's stage whose recorded c is read
    repeat_item: str  # E332's repeat record
    repeat_tol: float
    first_read_max_s: float
    cost_max_usd: float
    usd_per_h: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        assert self.repack_seed != 0 and self.repeat_tol >= 0 and self.first_read_max_s > 0 and self.cost_max_usd > 0


class Run:
    """The reads after the controls: E332's records, the first-read and cost stops."""

    def __init__(self, lab: Any, c: Config, rc: Any, data: Data) -> None:
        self.lab, self.c, self.rc, self.data = lab, c, rc, data
        self.e332, self.first = G.records(E332_OUT), True

    def read(self, R: DW, cs: float, rs: str) -> dict[str, Any]:
        start = time.time()
        rd = RQ.full_read(self.lab, R, cs, *self.data[rs])
        s = time.time() - start
        if self.first:
            self.lab.check("first_read_time", s <= self.c.first_read_max_s, {"seconds": s, "max": self.c.first_read_max_s})
            self.first = False
        return {"c": cs, "rowset": rs, "answers": len(self.data[rs][1].row), **rd, "share": rd["share_any"], "seconds": s}

    def repeat(self, R: DW, shapes: dict[str, tuple[int, ...]]) -> None:
        old = self.e332[self.c.repeat_item]
        rec = self.read(SR.gaussian({s: float(m.double().norm()) for s, m in R.items()}, shapes, old["draw_seed"]),
                        old["signed_c"], old["rowset"])
        G.put(self.lab, {"item": item_repeat(self.c), "stage": "repeat", "draw_seed": old["draw_seed"],
                         "norms_from": old["norms_from"], "e332_share": old["share"], "e332_gain": old["gain"], **rec})
        diff = {"share": rec["share"] - old["share"], "gain": rec["gain"] - old["gain"], "tol": self.c.repeat_tol}
        self.lab.check("repeat_e332", abs(diff["share"]) <= self.c.repeat_tol and abs(diff["gain"]) <= self.c.repeat_tol, diff)

    def state(self, st: LS.State, R: DW) -> None:
        rs = f"{QS.ROWSET[st.arm]}_repack{self.c.repack_seed}"
        for stage in ("full", self.c.gain1_stage):
            if (name := item(st, stage, self.c)) in G.records(OUT):
                continue
            old = self.e332[f"{VPD}/{stage}/{st.key}"]
            rec = self.read(R, old["c"], rs)
            band = abs(rec["gain"] / old["target"] - 1) <= self.rc.gain_rel_tol if old["target"] else None
            G.put(self.lab, {"item": name, "stage": stage, "target": old["target"], "arm": st.arm, "seed": st.seed,
                             "save": st.save, "step": st.step, "sha256": st.sha256, "in_band": band, **rec,
                             "e332_share": old["share"], "e332_gain": old["gain"], "diff_e332": rec["share"] - old["share"]})

    def cost(self, t_states: float, n_done: int, n_left: int) -> None:
        now = time.time()
        proj_s = now - self.lab.t0 + (now - t_states) / n_done * n_left
        usd = proj_s / 3600 * self.c.usd_per_h
        self.lab.log(f"[cost] projected {proj_s:.0f} s, ${usd:.2f} ({n_left} states left)")
        self.lab.check("projected_cost", usd <= self.c.cost_max_usd, {"projected_s": proj_s, "usd": usd, "max": self.c.cost_max_usd})


def item(st: LS.State, stage: str, c: Config) -> str:
    return f"{VPD}/{stage}/{st.key}_repack{c.repack_seed}"


def item_repeat(c: Config) -> str:
    return f"{c.repeat_item}_e333"


def todo(c: Config, rep_key: str) -> list[LS.State]:
    """States with an item unwritten, the repeat's norms state first while the repeat is unwritten."""
    need_rep = item_repeat(c) not in G.records(OUT)
    states = [LS.state(QS.RUN_OF[a], a, s, c.save) for a in c.arms for s in c.seeds]
    out = [st for st in states if any(item(st, g, c) not in G.records(OUT) for g in ("full", c.gain1_stage))
           or (need_rep and st.key == rep_key)]
    return sorted(out, key=lambda st: st.key != rep_key) if need_rep else out


def repacked(c: Config) -> Data:
    out: Data = {}
    end = SR.end_id()
    for rs in dict.fromkeys(QS.ROWSET[a] for a in c.arms):
        r, old = SR.marked_rows(rs, c.repack_seed), QS.rows_of(rs)
        assert r.tokens.shape == old.shape and not torch.equal(r.tokens, old), rs
        out[f"{rs}_repack{c.repack_seed}"] = (r.tokens, SR.answer_spans(r, end))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=False, help="the VPD decomposition directory ($COMMUTE_QWEN_NEW_DIR)")
    ap.add_argument("--fetch-to", default=str(RESULTS / "large" / "states"), help="where states land, two at a time")
    ap.add_argument("--dry", action="store_true", help="CPU: rows, answer spans and the items, then stop")
    a = ap.parse_args()
    c = config_from_settings(Config, json.loads(PRED.read_text())["settings"]["qwen_repack"])
    rc = config_from_settings(RQ.Config, json.loads(RQ.PRED.read_text())["settings"]["qwen_random"])
    data, e332 = repacked(c), G.records(E332_OUT)
    rep = e332[c.repeat_item]
    assert rep["rowset"] in data and rep["status"] == "ok", rep
    states = todo(c, rep["norms_from"])
    print(f"rows {[(rs, len(sp.row)) for rs, (_, sp) in data.items()]}; repeat of {c.repeat_item} at c {rep['signed_c']} "
          f"(share {rep['share']}, gain {rep['gain']}); {len(states)} states to read, first {states[0].key if states else '-'}; "
          f"E332 c per read {[(st.key, e332[f'{VPD}/{g}/{st.key}']['c']) for st in states for g in ('full', c.gain1_stage)]}; "
          f"{len(G.records(OUT))} items in {OUT.name}", flush=True)
    if a.dry:
        hard_exit(0)
    require_filed(PRED)
    assert not STOPPED.is_file(), f"{STOPPED} exists: a stop rule fired"
    assert a.decomp and decomp_id(Path(a.decomp)) == VPD, "--decomp: the VPD decomposition only"
    QS.point(VPD)
    L.RESULTS, L.PRED = RESULTS, PRED
    QS.OWN = frozenset(s for s in SITES if parse_site(s)[1] in decomp_kinds(Path(a.decomp)))
    lab = G.Lab(f"P2.91 E333 {VPD}", "unused", a.decomp)
    with torch.no_grad():
        shapes = {s: tuple(lin.weight.shape) for s, lin in linears(lab.model).items()}
        k = rc.search_rows  # T0 on the search rows, for the controls' search gain (as E332)
        t0 = {rs: SR.answer_target(lab, "W0", rows[:k], spans.take(0, k)) for rs, (rows, spans) in data.items()}
        G.controls(lab, data, t0, rc, SR.with_zeros({}, shapes))
        run, t_states = Run(lab, c, rc, data), time.time()
        for i, (st, path) in enumerate(QS.streamed(states, Path(a.fetch_to), lab.log)):
            R = QS.own_sites(load(path))
            if st.key == rep["norms_from"] and item_repeat(c) not in G.records(OUT):
                run.repeat(R, shapes)
            run.state(st, R)
            del R
            run.cost(t_states, i + 1, len(states) - i - 1)
    lab.log(f"[done] {lab.peaks()}")
    print("QWEN_REPACK DONE", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()

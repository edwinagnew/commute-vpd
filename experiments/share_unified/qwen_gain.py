"""Cell P2.91 (E332): does the Qwen re-gating share (P2.90's whole-answer log-odds share, `share.share`) depend on how
far an update moves the answers? E330's random references were read at a gain of about 1; the fine-tunes move the
answers further. Each fine-tune's own update R = W_ft - W0 (known and copy, seeds 0-2, the last save; P3.8 run 2's,
fetched from W&B as qwen_reread.py fetches them) is scaled down to gains of about 10 and 1 and the share read at each.
VPD decomposition only (p-9707d6ff, $COMMUTE_QWEN_NEW_DIR). Search, full read and the draw are qwen_random.py's (E330).

  1 controls  per row set (known, copy, known repacked): a zero change and W0 itself (do-nothing) read search gain,
              full gain and numerator exactly 0 (E330's control a, E328's W0 control); a miss stops the job.
  2 full      c = 1 on all 128 rows. Early check: the share against E328's (results/qwen_reread.jsonl, same item); no
              measured same-card Qwen floor exists, so the difference is written and does not stop. The first full read
              over first_read_max_s stops the job.
  3 g10, g1   `perturb.match` finds c in (0, 1] with gain(c R) in target x (1 +- gain_rel_tol) on E330's search rows
              (T cells only), then the share is read on all rows at that c. Run per state after its full read, so each
              5.6 GB state is fetched once (~400 s; the next one fetches while this one is read).
  4 repeat    E330's VPD known draw 2 (its draw seed, the norms of known_s2_last), sign and scale searched afresh to
              gain 1 as E330 did (`qwen_random.one_draw`), on the known rows repacked with `repack_seed`: the same 600
              trained items in a new order and new row cuts (not content-disjoint: 128 rows hold every item about twice).
One JSON line per item in results/qwen_gain.jsonl as it lands; a rerun skips the items written. A stop writes
results/stopped_gain.json and exits 1. Prints "QWEN_GAIN DONE" at the end. Settings: results/predictions_gain.json.

    PRED_SHA=<sha> python experiments/share_unified/qwen_gain.py --decomp "$COMMUTE_QWEN_NEW_DIR"
    ~/commute-agents/fleet/cpu-run --mem 8G python3 experiments/share_unified/qwen_gain.py --dry   (CPU: rows, items)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import library_states as LS
from commute import perturb as PT
from commute import share as SH
from commute import share_reading as SR
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.qwen_sft import linears, load
from commute.qwen_vpd import SITES, decomp_id, decomp_kinds, parse_site

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT, STOPPED = RESULTS / "predictions_gain.json", RESULTS / "qwen_gain.jsonl", RESULTS / "stopped_gain.json"
RQ = load_experiment("share_unified", "qwen_random")  # E330: Search, full_read, one_draw, Config
RR, QS, L = RQ.RR, RQ.QS, RQ.L  # E328's rows and fine-tune records; P3.8's fetch; E57's Lab
VPD = "p-9707d6ff"
Data = dict[str, tuple[Tensor, SH.Spans]]


@dataclass(frozen=True)
class Config:
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    save: str
    targets: tuple[float, ...]  # the scaled reads' gains, in read order; the last is E330's target_gain
    repeat_arm: str
    repeat_draw: int  # E330's draw index (its seed and norms come from E330's record)
    repack_seed: int  # the repeat's row packing; E330's rows are P.SEED's
    first_read_max_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "targets", tuple(self.targets))
        assert self.repeat_arm in self.arms and self.repeat_draw in self.seeds and self.repack_seed != 0
        assert all(t > 0 for t in self.targets) and self.first_read_max_s > 0


class Lab(L.Lab):
    def stop(self, name: str, why: str) -> None:  # E332's own stop file; E328/E330's results/stopped.json untouched
        STOPPED.write_text(json.dumps({"name": name, "why": why, "at_s": time.time() - self.t0}) + "\n")
        self.log(f"STOP {name}: {why}")
        sys.exit(1)


def records(path: Path) -> dict[str, dict[str, Any]]:
    return {(r := json.loads(ln))["item"]: r for ln in path.read_text().splitlines() if ln} if path.is_file() else {}


def put(lab: Any, rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    lab.log(f"[item {rec['item']}] {rec.get('seconds', 0):.0f} s {json.dumps(rec, default=float)}")


def controls(lab: Any, data: Data, t0: dict[str, Tensor], rc: Any, zero: dict[str, Tensor]) -> None:
    for rs, (rows, spans) in data.items():
        for kind, change in (("zero", zero), ("W0", None)):
            name, start = f"{VPD}/{kind}/{rs}", time.time()
            if name in records(OUT):
                continue
            k = RR.w0_rows(spans, rc.n16, len(rows))
            g = RQ.Search(lab, change, rows[: rc.search_rows], spans.take(0, rc.search_rows), t0[rs]).gain(1.0)
            rd = RQ.full_read(lab, change, 1.0, rows[:k], spans.take(0, k))
            put(lab, {"item": name, "stage": "control", "kind": kind, "rowset": rs, "c": 1.0, "search_gain": g,
                      "share": None, "seconds": time.time() - start, **rd})
            lab.check(f"{kind}_{rs}", g == 0 and rd["gain"] == 0 and rd["numerator"] == 0,
                      {"search_gain": g, "gain": rd["gain"], "numerator": rd["numerator"]})


def scaled(lab: Any, R: dict[str, Tensor], rows: Tensor, spans: SH.Spans, t0: Tensor, target: float, rc: Any) -> dict[str, Any]:
    start, k = time.time(), rc.search_rows
    sc = RQ.Search(lab, R, rows[:k], spans.take(0, k), t0)
    m = PT.match(sc.gain, target, 1.0, rc.gain_rel_tol)
    rd = RQ.full_read(lab, R, m["x"], rows, spans)
    return {"status": m["status"], "c": m["x"], "search_gain": m["f"], "evals": len(sc.seen), **rd,
            "share": rd["share_any"] if m["status"] == "ok" else math.nan,
            "in_band": abs(rd["gain"] / target - 1) <= rc.gain_rel_tol, "seconds": time.time() - start}


def finetunes(lab: Any, data: Data, t0: dict[str, Tensor], c: Config, rc: Any, fetch_to: Path) -> dict[str, float]:
    """Stages 2-3 per state; returns the repeat's norms (the repeat arm's draw-seed state) when it is read."""
    stages = ("full", *(f"g{t:g}" for t in c.targets))
    name = lambda st, s: f"{VPD}/{s}/{st.key}"
    rep = f"{c.repeat_arm}_s{c.repeat_draw}_{c.save}"
    need_rep = f"{VPD}/repeat/{c.repeat_arm}_d{c.repeat_draw}_repack{c.repack_seed}" not in records(OUT)
    states = [LS.state(QS.RUN_OF[a], a, s, c.save) for a in c.arms for s in c.seeds]
    todo = [st for st in states if any(name(st, s) not in records(OUT) for s in stages) or (need_rep and st.key == rep)]
    e328, norms, first = records(RR.OUT), {}, not any(name(st, "full") in records(OUT) for st in states)
    for st, path in QS.streamed(todo, fetch_to, lab.log):
        R = QS.own_sites(load(path))
        if st.key == rep:
            norms = {s: float(m.double().norm()) for s, m in R.items()}
        rows, spans = data[QS.ROWSET[st.arm]]
        base = {"arm": st.arm, "seed": st.seed, "save": st.save, "step": st.step, "sha256": st.sha256,
                "rowset": QS.ROWSET[st.arm], "answers": len(spans.row)}
        if name(st, "full") not in records(OUT):
            start = time.time()
            rd = RQ.full_read(lab, R, 1.0, rows, spans)
            old = e328[f"{VPD}/run2/{st.key}"]
            s = time.time() - start
            put(lab, {"item": name(st, "full"), "stage": "full", "target": None, "c": 1.0, **base, **rd,
                      "share": rd["share_any"], "e328_share": old["share"], "diff_e328": rd["share_any"] - old["share"],
                      "e328_gain": old["gain"], "seconds": s})
            if first:
                lab.check("first_read_time", s <= c.first_read_max_s, {"seconds": s, "max": c.first_read_max_s})
                first = False
        for stage, target in zip(stages[1:], c.targets, strict=True):
            if name(st, stage) not in records(OUT):
                put(lab, {"item": name(st, stage), "stage": stage, "target": target, **base,
                          **scaled(lab, R, rows, spans, t0[QS.ROWSET[st.arm]], target, rc)})
        del R
    return norms


def repeat(lab: Any, rows: Tensor, spans: SH.Spans, t0: Tensor, c: Config, rc: Any, norms: dict[str, float],
           shapes: dict[str, tuple[int, ...]]) -> None:
    item = f"{VPD}/repeat/{c.repeat_arm}_d{c.repeat_draw}_repack{c.repack_seed}"
    if item in records(OUT):
        return
    e330 = e330_draw(c)
    R = SR.gaussian(norms, shapes, e330["draw_seed"])
    rec = RQ.one_draw(lab, R, rows, spans, t0, rc)
    put(lab, {"item": item, "stage": "repeat", "arm": c.repeat_arm, "seed": c.repeat_draw, "target": rc.target_gain,
              "c": rec["signed_c"], "rowset": f"{c.repeat_arm}_repack{c.repack_seed}", "answers": len(spans.row),
              "draw_seed": e330["draw_seed"], "norms_from": e330["norms_from"], "e330_c": e330["signed_c"],
              "e330_share": e330["share"], **rec})


def e330_draw(c: Config) -> dict[str, Any]:
    recs = {k: v for p in sorted(RESULTS.glob("qwen_random_vpd*.jsonl")) for k, v in records(p).items()}
    r = recs[f"{VPD}/{c.repeat_arm}_d{c.repeat_draw}"]
    assert r["norms_from"] == f"{c.repeat_arm}_s{c.repeat_draw}_{c.save}" and r["status"] == "ok", r
    return r


def repacked(c: Config, data: Data) -> tuple[Tensor, SH.Spans]:
    r = SR.marked_rows(c.repeat_arm, c.repack_seed)
    assert r.tokens.shape == data[c.repeat_arm][0].shape and not torch.equal(r.tokens, data[c.repeat_arm][0])
    return r.tokens, SR.answer_spans(r, SR.end_id())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=False, help="the VPD decomposition directory ($COMMUTE_QWEN_NEW_DIR)")
    ap.add_argument("--fetch-to", default=str(RESULTS / "large" / "states"), help="where states land, two at a time")
    ap.add_argument("--dry", action="store_true", help="CPU: rows, answer spans and the items, then stop")
    a = ap.parse_args()
    c = config_from_settings(Config, json.loads(PRED.read_text())["settings"]["qwen_gain"])
    rc = config_from_settings(RQ.Config, json.loads(RQ.PRED.read_text())["settings"]["qwen_random"])
    assert c.targets[-1] == rc.target_gain, "gain 1 is E330's target"
    data = RR.rows_and_spans(c.arms)
    data[f"{c.repeat_arm}_repack{c.repack_seed}"] = repacked(c, data)
    e330 = e330_draw(c)
    print(f"rows {[(rs, len(sp.row)) for rs, (_, sp) in data.items()]}; {len(c.arms) * len(c.seeds)} states x "
          f"(full + {len(c.targets)} scaled); repeat of {e330['item']} (draw seed {e330['draw_seed']}, c {e330['signed_c']:.4g}, "
          f"share {e330['share']:.4f}); {len(records(OUT))} items in {OUT.name}", flush=True)
    if a.dry:
        hard_exit(0)
    require_filed(PRED)
    assert not STOPPED.is_file(), f"{STOPPED} exists: a stop rule fired"
    assert a.decomp and decomp_id(Path(a.decomp)) == VPD, "--decomp: the VPD decomposition only"
    QS.point(VPD)
    L.RESULTS, L.PRED = RESULTS, PRED
    QS.OWN = frozenset(s for s in SITES if parse_site(s)[1] in decomp_kinds(Path(a.decomp)))
    lab = Lab(f"P2.91 E332 {VPD}", "unused", a.decomp)
    with torch.no_grad():
        shapes = {s: tuple(lin.weight.shape) for s, lin in linears(lab.model).items()}
        k = rc.search_rows  # T0 on the search rows, once per row set (as E330)
        t0 = {rs: SR.answer_target(lab, "W0", rows[:k], spans.take(0, k)) for rs, (rows, spans) in data.items()}
        controls(lab, data, t0, rc, SR.with_zeros({}, shapes))
        norms = finetunes(lab, data, t0, c, rc, Path(a.fetch_to))
        rs = f"{c.repeat_arm}_repack{c.repack_seed}"
        if norms:
            repeat(lab, *data[rs], t0[rs], c, rc, norms, shapes)
    lab.log(f"[done] {lab.peaks()}")
    print("QWEN_GAIN DONE", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()

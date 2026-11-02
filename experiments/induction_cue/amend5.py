"""E62 Amendment 5: where does the masked forward's copy run at W0, at the level of heads? W0 only, at the second A (half - 1), log-odds of B, on E62's
200 read windows. No training, no state read, nothing in results/large overwritten or deleted. Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md,
"Amendment 5 (2026-11-02)". Rules and expected values: results/predictions_amend5.json (committed first; the run refuses otherwise; never edited),
`commute.copying`. Amendment 4's filed verdict stands; nothing here rescores it.

  a control, alone: `a`, `a_half2`, S_old saved first; head (2,4) off with every mask at 1; a fall under a quarter of the target's filed fall is exit 1.
    Then the re-read of Amendment 4's stage 1 against amend4.json, and the do-nothing (no head off) repeat. A miss is exit 1.
  b window bootstrap sds on the falls, and the masked rows in fp32, under Γ's bf16 gates and under Γ's own fp32 gates. A move over the fall's sd is loud, not a stop.
  c every head off, one at a time: target and masked, intact and pair-broken; the masked rows again in fp32. copying.sweep_reading.
  d the swapped forwards: W0's pair-broken gates on the intact tokens, W0's intact gates on the pair-broken tokens. copying.token_reading.
The head off is in the forward named, never in Γ's pass: every gate list is read before any hook. Masked forward and Γ bf16 (autocast) but for (b) and the
sweep's fp32 rows, target model fp32, as Amendment 4. results/amend5.json after every stage, every number printed; per-window values of every read in
results/large/amend5_windows.pt after every stage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import torch
from amend3 import Reads
from torch import Tensor

from commute import copying as CP
from commute import keydep as KD
from commute import steering as ST
from commute.data import LAST
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, BASE, RUN, A4_OUT, OUT = (RESULTS / f for f in ("predictions_amend5.json", "predictions.json", "run.json", "amend4.json", "amend5.json"))
LABELS = {"2,4": "the labelled copying head", "2,3": "W0's second prefix head", "3,5": "the reference head of Amendments 2 and 3"}
log = KD.log
Gates = list[dict[str, Tensor]]


@dataclass
class Reads5(Reads):
    """`Reads` with its full-precision paths and the log-probability read. `half` is the second A here."""

    def masked32(self, w: Tensor, gates: Gates, heads: list[CP.Head]) -> Tensor:
        with CP.ablated(self.ctx.net, heads, self.means):
            return CP.masked_lo_fp32(self.ctx.net.model, w, self.half, LAST, gates, KD.BATCH, self.n)

    def gates32(self, w: Tensor) -> Gates:
        return CP.gates_fp32(self.ctx.net.model, w, KD.BATCH, self.n)

    def log_p(self, w: Tensor) -> Tensor:
        return CP.read_lo(self.ctx.net, w, self.half, self.n, readout=ST.logp_at)


@dataclass
class Job:
    R: Reads5  # at the second A
    w: Tensor  # intact, target last
    b: Tensor  # pair-broken, target last
    head: list[CP.Head]
    th: dict[str, Any]
    a4: dict[str, Any]  # amend4.json's stage1_control
    out: dict[str, Any]
    results: Path
    vectors: Path
    windows: Path
    rows: dict[str, Tensor] = field(default_factory=dict)

    def read(self, name: str, v: Tensor) -> Tensor:
        self.rows[name] = v.detach().cpu()
        return self.rows[name]

    def dump(self) -> None:
        self.results.write_text(json.dumps(jsonable(self.out), indent=1, sort_keys=True))
        torch.save(self.rows, self.windows)

    def note(self, key: str, rec: dict[str, Any]) -> None:
        self.out[key] = rec
        self.dump()
        log(f"[{key}] {json.dumps(jsonable(rec))}")

    def stop(self, verdict: str, why: str) -> NoReturn:
        self.out["stopped"] = why
        self.out["verdict"] = CP.stopped_reading(verdict)
        self.dump()
        sys.exit(f"STOP: {verdict}: {why}: tell the lead")

    def sd(self, d: Tensor) -> float:
        return CP.boot_sd(d, self.R.n_boot, 0)

    def fall_gap(self, intact: Tensor, broken: Tensor) -> dict[str, float]:
        """A head's fall on the intact windows minus on the pair-broken ones: the part of the fall there because of the pair."""
        return {"mean": float((intact - broken).mean()), "sd": self.sd(intact - broken)}

    def fall(self, on: str, off: str) -> dict[str, float]:
        d = self.rows[on] - self.rows[off]
        return {"mean": float(d.mean()), "sd": self.sd(d)}


def control(J: Job, g0: Gates) -> None:
    """Head off with every mask at 1 (`AllGatesOn`'s forward, tests/test_copying.py), intact and pair-broken, against the target's filed fall."""
    R, w, b = J.R, J.w, J.b
    ones = [{k: torch.ones_like(v) for k, v in batch.items()} for batch in g0]
    on, off, broken = J.read("ones_on", R.masked(w, ones, [])), J.read("ones_off", R.masked(w, ones, J.head)), J.read("ones_broken", R.masked(b, ones, []))
    fall, lean = J.fall("ones_on", "ones_off"), J.fall("ones_on", "ones_broken")
    status = CP.all_ones_control(fall["mean"], J.a4["fall"], J.th)
    J.note("stage_a_control", {"on": float(on.mean()), "off": float(off.mean()), "broken": float(broken.mean()), "fall": fall["mean"], "fall_sd": fall["sd"], "lean": lean["mean"], "lean_sd": lean["sd"],
                               "target_fall_filed": J.a4["fall"], "target_lean_filed": J.a4["lean"], "fall_over_target": fall["mean"] / J.a4["fall"], "status": status})
    if status != "pass":
        J.stop("unreadable" if status == "unreadable" else status, f"stage a: head off with every mask at 1 falls {fall['mean']:.3f}, under a quarter of the target's {J.a4['fall']:.3f}; the sweep is not read")


def stage1_numbers(J: Job, g0: Gates, gb: Gates, a: Tensor, a2: Tensor, old: Tensor) -> dict[str, Any]:
    """Amendment 4's stage 1, from the same calls, every read saved per window."""
    R, w, b, H = J.R, J.w, J.b, J.head
    for name, v in (("target_on", R.lo(w, [])), ("target_off", R.lo(w, H)), ("target_broken", R.lo(b, [])), ("mask_on", R.masked(w, g0, [])), ("mask_off", R.masked(w, g0, H)), ("mask_broken", R.masked(b, gb, []))):
        J.read(name, v)
    lv = {k: float(J.rows[k].mean()) for k in ("target_on", "target_off", "target_broken", "mask_on", "mask_off", "mask_broken")}
    half2 = CP.s_old(a2)
    return {"p_b": KD.geo_rate(J.read("log_p_b", R.log_p(w))), **lv, "fall": lv["target_on"] - lv["target_off"], "lean": lv["target_on"] - lv["target_broken"], "mask_fall": lv["mask_on"] - lv["mask_off"], "mask_lean": lv["mask_on"] - lv["mask_broken"],
            "n_alive": len(a), "k": len(old), "a_positive": int((a > 0).sum()), "a_mass_positive": float(a.clamp_min(0).sum()), "top_a": [float(x) for x in a[old[:10]]], "half2_k": len(half2), "half2_overlap": int(torch.isin(old, half2).sum()), "cos_a_half2": CP.cosine(a, a2)}


def stage_a(J: Job, p: Tensor) -> tuple[Gates, Gates]:
    """Γ's two passes, with no hook; the vectors on disk before anything can stop; the control alone; then the re-read."""
    R, w, b, alive, n = J.R, J.w, J.b, J.R.ctx.lift["alive"], J.R.n
    half = R.half + 1
    if not (CP.second_a_holds(w[:, :n], p, w[:, -1], half) and CP.pair_broken_holds(w[:, :n], b[:, :n], p, half) and torch.equal(w[:, -1], b[:, -1])):
        J.stop("unreadable", "stage a: the second A or the pair-broken windows are not as filed")
    g0, gb = R.gates(w), R.gates(b)
    second = CP.gate_means(g0, alive, R.half)
    a, a2 = second - CP.gate_means(gb, alive, R.half), second - CP.gate_means(g0, alive, R.half - 1)
    old = CP.s_old(a)
    torch.save({"a": a, "a_half2": a2, "S_old": old}, J.vectors)
    log(f"[vectors] saved {J.vectors}: a {len(a)}, S_old {len(old)}")
    control(J, g0)
    now = stage1_numbers(J, g0, gb, a, a2, old)
    repeat = {"target": float((R.lo(w, []) - J.rows["target_on"]).abs().max()), "mask": float((R.masked(w, g0, []) - J.rows["mask_on"]).abs().max())}
    diff = CP.reread_diff(now, J.a4)
    J.note("stage_a_reread", {**now, "max_diff": diff, "exact": diff == 0.0, "do_nothing_max_abs": repeat, "tol": J.th["reread_tol"]})
    if diff > J.th["reread_tol"] or max(repeat.values()) > J.th["reread_tol"]:
        J.stop("reread_failed", f"stage a: the re-read of amend4.json differs by {diff}, or the do-nothing repeat by {repeat}")
    return g0, gb


def round_level(v: Tensor) -> dict[str, Any]:
    """Amendment 4's oddly round masked level, -7.30000003, no known mechanism: distinct window values, sum and smallest gap, for a lattice if there is one."""
    u = v.unique()
    return {"mean": float(v.mean()), "sum": float(v.sum()), "n_distinct": len(u), "min_gap": float(u.diff().min()) if len(u) > 1 else float("nan")}


def stage_b(J: Job, g0: Gates, gb: Gates) -> dict[str, Any]:
    """Sds, and the masked rows in both precisions. No stop: the job costs nothing, and the sweep's fp32 rows are what a loud line here needs next."""
    R, w, b, H, rows = J.R, J.w, J.b, J.head, J.rows
    g32 = R.gates32(w)
    for name, v in (("mask32_on", R.masked32(w, g0, [])), ("mask32_off", R.masked32(w, g0, H)), ("mask32_broken", R.masked32(b, gb, [])), ("gamma32_on", R.masked32(w, g32, [])), ("gamma32_off", R.masked32(w, g32, H))):
        J.read(name, v)
    mask, fp32, gamma32 = J.fall("mask_on", "mask_off"), J.fall("mask32_on", "mask32_off"), J.fall("gamma32_on", "gamma32_off")
    miss = 0.25 * (rows["target_on"] - rows["target_off"]) - (rows["mask_on"] - rows["mask_off"])
    moved = {"fp32": CP.precision_moved(mask["mean"], fp32["mean"], mask["sd"]), "gamma32": CP.precision_moved(mask["mean"], gamma32["mean"], mask["sd"])}
    rec = {"target_fall": J.fall("target_on", "target_off"), "target_lean": J.fall("target_on", "target_broken"), "mask_fall": mask, "mask_lean": J.fall("mask_on", "mask_broken"),
           "amend4_miss_reported_only": {"mean": float(miss.mean()), "sd": J.sd(miss)}, "fp32_fall": fp32, "fp32_lean": J.fall("mask32_on", "mask32_broken"), "gamma32_fall": gamma32,
           "fp32_minus_bf16": {"on": float((rows["mask32_on"] - rows["mask_on"]).mean()), "off": float((rows["mask32_off"] - rows["mask_off"]).mean()), "fall": fp32["mean"] - mask["mean"], "window_max_abs_on": float((rows["mask32_on"] - rows["mask_on"]).abs().max())},
           "gamma32_minus_bf16": {"on": float((rows["gamma32_on"] - rows["mask_on"]).mean()), "off": float((rows["gamma32_off"] - rows["mask_off"]).mean()), "fall": gamma32["mean"] - mask["mean"], "gate_max_abs": max(float((x[k].float() - y[k].float()).abs().max()) for x, y in zip(g32, g0, strict=True) for k in x)},
           "round_level": {"bf16": {k: round_level(rows[f"mask_{k}"]) for k in ("on", "off", "broken")}, "fp32": {k: round_level(rows[f"mask32_{k}"]) for k in ("on", "off", "broken")}}, "moved": moved}
    J.note("stage_b_precision", rec)
    for k, m in moved.items():
        if m:
            log(f"*** LOUD: THE MASKED FALL MOVES WITH PRECISION ({k}): bf16 {mask['mean']:.4f}, {k} {rec[f'{k}_fall']['mean']:.4f}, sd {mask['sd']:.4f}. Read the verdict beside its fp32 verdict.")
    return rec


def head_rec(J: Job, h: CP.Head, g0: Gates, gb: Gates) -> dict[str, Any]:
    R, w, b, rows, key = J.R, J.w, J.b, J.rows, f"{h[0]}_{h[1]}"
    for name, v in (("target_off", R.lo(w, [h])), ("target_broken_off", R.lo(b, [h])), ("mask_off", R.masked(w, g0, [h])), ("mask_broken_off", R.masked(b, gb, [h])), ("mask32_off", R.masked32(w, g0, [h])), ("mask32_broken_off", R.masked32(b, gb, [h]))):
        J.read(f"{name}_{key}", v)
    t, m, m32 = rows["target_on"] - rows[f"target_off_{key}"], rows["mask_on"] - rows[f"mask_off_{key}"], rows["mask32_on"] - rows[f"mask32_off_{key}"]
    rec: dict[str, Any] = {"T": float(t.mean()), "M": float(m.mean()), "T_sd": J.sd(t), "M_sd": J.sd(m), "excess_sd": J.sd(m - t), "excess": float((m - t).mean()),
                           "Tb": float((rows["target_broken"] - rows[f"target_broken_off_{key}"]).mean()), "Mb": float((rows["mask_broken"] - rows[f"mask_broken_off_{key}"]).mean()),
                           "pair_part_T": J.fall_gap(t, rows["target_broken"] - rows[f"target_broken_off_{key}"]), "pair_part_M": J.fall_gap(m, rows["mask_broken"] - rows[f"mask_broken_off_{key}"]),
                           "M32": float(m32.mean()), "Mb32": float((rows["mask32_broken"] - rows[f"mask32_broken_off_{key}"]).mean()), "excess32_sd": J.sd(m32 - t), "fraction_sd": CP.share_sd(t, m, R.n_boot, 0), "label": LABELS.get(CP.head_name(h), "")}
    return rec


def stage_c(J: Job, g0: Gates, gb: Gates, lean32: float) -> tuple[dict[str, Any], dict[str, Any]]:
    R = J.R
    blocks, heads = R.means.shape[:2]
    rec: dict[CP.Head, dict[str, Any]] = {}
    for h in ((x, y) for x in range(blocks) for y in range(heads)):
        rec[h] = head_rec(J, h, g0, gb)
        log(f"[head {CP.head_name(h)} {rec[h]['label']}] {json.dumps(jsonable(rec[h]))}")
    fp32 = {h: {"T": r["T"], "M": r["M32"], "Mb": r["Mb32"], "excess_sd": r["excess32_sd"]} for h, r in rec.items()}
    n, lab = J.th["n_heads"], J.head[0]
    reading = CP.sweep_reading(rec, lab, n, J.out["stage_a_reread"]["mask_lean"], J.th)
    reading32 = CP.sweep_reading(fp32, lab, n, lean32, J.th)
    r24 = rec[lab]
    J.note("stage_c_sweep", {"heads": {CP.head_name(h): r for h, r in rec.items()}, "reading": reading, "reading_fp32": reading32})
    for h, r in sorted(rec.items(), key=lambda x: -x[1]["excess"]):
        log(f"  {CP.head_name(h)}  T {r['T']:+.3f}  M {r['M']:+.3f}  M - T {r['excess']:+.3f} (sd {r['excess_sd']:.3f})  Mb {r['Mb']:+.3f}  M32 {r['M32']:+.3f}  {r['label']}")
    if max(abs(r24["T"] - J.a4["fall"]), abs(r24["M"] - J.a4["mask_fall"])) > J.th["reread_tol"]:
        J.stop("reread_failed", "stage c: the sweep's labelled row does not re-read amend4.json")
    return reading, reading32


def stage_d(J: Job, g0: Gates, gb: Gates) -> str:
    R, w, b, rows = J.R, J.w, J.b, J.rows
    J.read("swap_broken_gates_on_intact", R.masked(w, gb, []))
    J.read("swap_intact_gates_on_broken", R.masked(b, g0, []))
    lean = float((rows["mask_on"] - rows["mask_broken"]).mean())
    tokens_at_broken_gates = J.fall("swap_broken_gates_on_intact", "mask_broken")
    tokens_at_intact_gates = J.fall("mask_on", "swap_intact_gates_on_broken")
    gates_on_intact, gates_on_broken = J.fall("mask_on", "swap_broken_gates_on_intact"), J.fall("swap_intact_gates_on_broken", "mask_broken")
    status = CP.token_reading(tokens_at_broken_gates["mean"], tokens_at_intact_gates["mean"], lean, J.th)
    J.note("stage_d_swapped", {"lean": lean, "broken_gates_on_intact": float(rows["swap_broken_gates_on_intact"].mean()), "intact_gates_on_broken": float(rows["swap_intact_gates_on_broken"].mean()),
                               "tokens_at_broken_gates": tokens_at_broken_gates, "tokens_at_intact_gates": tokens_at_intact_gates, "gates_on_intact": gates_on_intact, "gates_on_broken": gates_on_broken, "token_reading": status})
    return status


def run(J: Job, p: Tensor) -> None:
    g0, gb = stage_a(J, p)
    b = stage_b(J, g0, gb)
    reading, reading32 = stage_c(J, g0, gb, b["fp32_lean"]["mean"])
    tokens = stage_d(J, g0, gb)
    v = {**reading, "fp32_verdict": reading32["verdict"], "precision_moved": b["moved"], "token_reading": tokens}
    J.note("verdict", v)
    log(f"[verdict] {v['verdict']} carriers {v['carriers']} f_labelled {v['f_labelled']} | fp32 {v['fp32_verdict']} | tokens {tokens}")
    if v["verdict"] != v["fp32_verdict"]:
        log(f"*** LOUD: THE TWO PRECISIONS DISAGREE: bf16 {v['verdict']}, fp32 {v['fp32_verdict']}. The filed verdict is bf16's; the reading goes to the user with both.")


def load(a: argparse.Namespace) -> tuple[Job, Tensor]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    large = a.large.resolve()
    vectors, windows = large / "amend5_vectors.pt", large / "amend5_windows.pt"
    for f in (OUT, vectors, windows):
        if f.exists():
            sys.exit(f"{f} exists: move it aside, this run deletes nothing")
    P, S, run_json = json.loads(PRED.read_text()), json.loads(BASE.read_text())["settings"], json.loads(RUN.read_text())
    if KD.sha256(large / "windows.pt") != run_json["windows_sha256"]:
        sys.exit("windows.pt is not the one run.json recorded")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, large)
    v = torch.load(large / "windows.pt", weights_only=False)
    half, n = S["half"], 2 * S["half"] + 1
    target = v["target"]["copy"]["read"]
    w = CP.with_target(v["read"], target).tokens
    means = torch.load(large / "heads.pt", weights_only=False)["means"].to(ctx.net.device)
    b = torch.cat([CP.broken_pair(v["read"].tokens, v["p_read"], v["pools"]["keys"], half, S["control_seed"]), target[:, None]], 1)  # as amend4.py
    R = Reads5(ctx, half - 1, n, means, S["n_boot"], {"copy": w}, b, {}, run_json)
    out = {"git_head": git("rev-parse", "HEAD").stdout.strip(), "predictions_sha256": KD.sha256(PRED), "host": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}
    head = [(int(x), int(y)) for x, y in P["settings"]["head"]]
    return Job(R, w, b, head, P["thresholds"], json.loads(A4_OUT.read_text())["stage1_control"], out, OUT, vectors, windows), v["p_read"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large")
    J, p = load(ap.parse_args())
    log(f"[host] {J.out['host']}: masked forward and Γ bf16 but for the fp32 rows, target model fp32")
    run(J, p)


if __name__ == "__main__":
    main()

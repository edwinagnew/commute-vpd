"""E92, Amendment 1, 2026-11-05: why does one steering vector recover 0.83 of the facts update's gain at the template's size
and 0.065 at its own full size, and does the template's share also move with step? Reads only; `run.py` stays as it ran.
Rules and expected values: results/predictions_amend1.json. Analysis this answers: reports/analyses/2026-11-05_analysis_e92-comparators.md.

Everything is read on the scoring half (odd rows), with the vector fitted on the fit half, as `qc.steer`. Per position, at W0, the
state, W0 + the vector, W0 + the norm-matched random vector (E92's) and W0 + three random vectors matched on norm and variance
share: log P of the target, rank, output entropy. W0 is also read at a grid of output temperatures, which is the reference for a
pure flattening: it keeps every rank and earns what it earns at that entropy.

Order: the controls first and alone (results/amend1/controls.json), then the facts sites, then the template's earlier steps. A
miss stops the job (results/stopped_amend1.json). A site already written is skipped, so a lost job resumes.
Exit 0 done; 1 a stop.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor

from commute import qwen_comparators as qc
from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import load, plus, set_weights

DIS: Any = load_experiment("qwen_comparators", "dissect")  # question_seen, CLASSES
RUN: Any = load_experiment("qwen_comparators", "run")  # the rows, path_of, Lab, STEER_SEED

RES = Path(__file__).resolve().parent / "results"
PRED, STOPPED, OUT, LARGE = RES / "predictions_amend1.json", RES / "stopped_amend1.json", RES / "amend1", RES / "large" / "amend1"
ZERO, CONTROLS = ("anchor1_s0", 1), (("anchor1_s0", 100), ("anchor1_s1", 100))
SITES = (("anchor2_s0", 29), ("anchor2_s142", 30), ("anchor2_s0", 1250), ("anchor2_s142", 1250))
EARLY = (("anchor1_s0", 10), ("anchor1_s0", 20), ("anchor1_s0", 50), ("anchor1_s1", 10))  # every saved step of E91's grid under 100
DRAWS = (92_001, 92_002, 92_003)
TEMPS = tuple(1.05**i for i in range(-14, 29))  # 0.505 to 3.92; the entry at i = 0 is exactly 1
T1 = TEMPS.index(1.0)


class Stop(Exception):
    pass


class Lab(RUN.Lab):
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (OUT / "run.log").open("a") as f:
            f.write(line + "\n")


# ----------------------------------------------------------------------------- the rules, no model


def clear(x: Tensor, row_of: Tensor) -> bool:
    """x's mean is above 0 by more than twice the standard error of its per-row means."""
    m, se, n = qc.row_mean_se(x, row_of)
    return n > 1 and m > 2 * se


def verdict(rise: Tensor, ref: Tensor, dH: Tensor, token: Tensor, row_of: Tensor, off_grid: bool) -> str:
    """The filed rule (predictions_amend1.json, rules.verdict). Every bar is a number read at this site in this job."""
    m, se, n = qc.row_mean_se(rise, row_of)
    if n < 2:
        return "unread"
    if abs(m) <= 2 * se:
        return "no rise"
    if m < 0:
        return "neither"
    if clear(dH, row_of) and not off_grid and clear(2 * ref - rise, row_of) and clear(2 * rise - ref, row_of):
        return "flattens"
    if clear(rise - 2 * ref, row_of) and clear(2 * token - (rise - ref), row_of):
        return "style"
    return "neither"


def token_part(resid: Tensor, target: Tensor, row_of: Tensor) -> Tensor:
    """Per position, the mean of `resid` for the same target token on the other half of the rows: what the token's identity
    alone predicts, out of sample. A token unseen on the other half predicts 0."""
    ids, inv = target.unique(return_inverse=True)
    other = torch.isin(row_of, row_of.unique()[::2])
    out = torch.zeros_like(resid, dtype=torch.float64)
    for src in (other, ~other):
        total = torch.zeros(len(ids), dtype=torch.float64).index_add(0, inv[src], resid[src].double())
        count = torch.zeros(len(ids), dtype=torch.float64).index_add(0, inv[src], torch.ones(int(src.sum()), dtype=torch.float64))
        out[~src] = (total / count.clamp_min(1))[inv[~src]]
    return out


def temperature_ref(grid_logp: Tensor, grid_entropy: Tensor, target_H: float) -> tuple[Tensor, float, bool]:
    """W0's log P rise, per position, at the temperature whose mean entropy over these positions equals `target_H`, linear
    between grid entries. Off the grid's top: its last entry, flagged. Below its bottom: its first entry."""
    H = grid_entropy.double().mean(0)
    i = int(torch.searchsorted(H, torch.tensor(target_H, dtype=H.dtype)))
    base = grid_logp[:, T1].double()
    if i == 0:
        return grid_logp[:, 0].double() - base, TEMPS[0], False
    if i == len(TEMPS):
        return grid_logp[:, -1].double() - base, TEMPS[-1], True
    f = (target_H - float(H[i - 1])) / float(H[i] - H[i - 1])
    mix = (1 - f) * grid_logp[:, i - 1].double() + f * grid_logp[:, i].double()
    return mix - base, TEMPS[i - 1] * (TEMPS[i] / TEMPS[i - 1]) ** f, False


def beyond(v: dict[str, Any], draws: Sequence[dict[str, Any]]) -> bool | None:
    """The vector's rise over the matched random draws': the gap beats twice the root of both row errors and the draws' spread."""
    if not v.get("n") or not all(d.get("n") for d in draws) or v.get("rise_se") is None or any(d.get("rise_se") is None for d in draws):
        return None
    r = torch.tensor([d["rise"] for d in draws], dtype=torch.float64)
    var = v["rise_se"] ** 2 + sum(d["rise_se"] ** 2 for d in draws) / len(draws) ** 2 + float(r.var()) / len(draws)
    return bool(v["rise"] - float(r.mean()) > 2 * math.sqrt(var))


def agree(verdicts: Sequence[str | None]) -> str:
    """One word only when every seed gives it; a missing read or a disagreement is no call, which opens no amendment."""
    first = verdicts[0] if verdicts else None
    return first if first is not None and all(v == first for v in verdicts) else "no call"


def trend(early: dict[str, Any], late: dict[str, Any]) -> str:
    """A share at the earliest defined step against step 100, by their standard errors."""
    if early["share"] is None or late["share"] is None:
        return "undefined"
    gap, bar = early["share"] - late["share"], 2 * math.hypot(early["share_se"], late["share_se"])
    return "falls" if gap > bar else "rises" if gap < -bar else "flat"


# ----------------------------------------------------------------------------- the vectors


def matched_random(X: Tensor, v: Tensor, seeds: Sequence[int], rel: float) -> list[Tensor]:
    """Random vectors at v's norm with v's variance share among X. A direction is U lam^a z, z Gaussian in the eigenbasis of X's
    covariance: a = 0 is isotropic, larger a leans to the top eigenvector, and its share of the variance rises with a. v's own
    share lies under the top eigenvalue's, so only an isotropic draw above it can miss; a miss raises."""
    target, c = qc.variance_share(X, v), (X - X.mean(0)).double()
    lam, U = torch.linalg.eigh(c.T @ c)
    lam = (lam / lam.max()).clamp_min(0)
    out = []
    for seed in seeds:
        z = torch.randn(len(lam), generator=torch.Generator(X.device).manual_seed(seed), device=X.device, dtype=lam.dtype)
        def part(a: float) -> float:
            w = lam.pow(a) * z  # noqa: B023 (used inside this pass of the loop)
            return float((lam * w.pow(2)).sum() / (w.pow(2).sum() * lam.sum()))
        lo, hi = 0.0, 256.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if part(mid) < target else (lo, mid)
        r = (U @ (lam.pow(hi) * z)).to(X.dtype)
        r = r * (v.norm() / r.norm())
        if abs(qc.variance_share(X, r) - target) > rel * target:
            raise Stop(f"matched random, seed {seed}: variance share {qc.variance_share(X, r)} against {target}")
        out.append(r)
    return out


# ----------------------------------------------------------------------------- reads: E92's `qc.read`, with the logits tapped


@dataclass(frozen=True)
class Full:
    read: qc.Read
    entropy: Tensor  # (K,) nats
    target: Tensor  # (K,) the token each position reads
    grid_logp: Tensor | None = None  # (K, len(TEMPS)) W0 only
    grid_entropy: Tensor | None = None


def entropy_of(logp: Tensor) -> Tensor:
    return -(logp.exp() * logp).sum(-1)


def read_full(model: Any, rows: qc.Rows, add: tuple[int, Tensor] | None = None, grid: bool = False) -> Full:
    """`qc.read` unchanged, with a hook on the output head for what it does not keep. Its batches run in order, so the hook
    counts them."""
    nxt = torch.zeros_like(rows.tokens)
    nxt[:, :-1] = rows.tokens[:, 1:]
    H, GL, GH, at_t = [], [], [], []

    @torch.no_grad()
    def tap(_m: Any, _a: Any, logits: Tensor) -> None:
        s = len(H) * qc.BATCH
        mask = rows.answer[s : s + qc.BATCH].to(logits.device)
        lg, tgt = logits[mask].float(), nxt[s : s + qc.BATCH].to(logits.device)[mask][:, None]
        H.append(entropy_of(torch.log_softmax(lg, -1)).cpu())
        for t in TEMPS if grid else ():
            lp = torch.log_softmax(lg / t, -1)
            at_t.append((lp.gather(1, tgt).squeeze(1).cpu(), entropy_of(lp).cpu()))
        if grid:
            GL.append(torch.stack([a for a, _ in at_t], 1))
            GH.append(torch.stack([b for _, b in at_t], 1))
            at_t.clear()

    handle = model.lm_head.register_forward_hook(tap)
    try:
        r = qc.read(model, rows, add)
    finally:
        handle.remove()
    return Full(r, torch.cat(H), nxt[rows.answer], torch.cat(GL) if grid else None, torch.cat(GH) if grid else None)


def table(w0: Full, tr: Full, st: Full, keep: Tensor) -> dict[str, Any]:
    """The filed numbers for one treatment on one pool of positions."""
    n = int(keep.sum())
    if n == 0 or w0.grid_logp is None or w0.grid_entropy is None:
        return {"n": n, "verdict": "unread"}
    row_of, base = w0.read.row_of[keep], w0.read.logp[keep]
    rise, gain, dH = tr.read.logp[keep] - base, st.read.logp[keep] - base, tr.entropy[keep].double() - w0.entropy[keep].double()
    ref, T, off = temperature_ref(w0.grid_logp[keep], w0.grid_entropy[keep], float(tr.entropy[keep].double().mean()))
    token = token_part(rise.double() - ref, w0.target[keep], row_of)
    R, Rse, rows = qc.row_mean_se(rise, row_of)
    H_rise, H_se, _ = qc.row_mean_se(dH, row_of)
    rank1 = float((tr.read.rank[keep] == 1).float().mean()) - float((w0.read.rank[keep] == 1).float().mean())
    return {"n": n, "rows": rows, "W0_logp": float(base.double().mean()), "logp": float(tr.read.logp[keep].double().mean()),
            "rise": R, "rise_se": Rse if rows > 1 else None, "share_of_state_gain": qc.share(rise, gain, row_of)["share"],
            "W0_entropy": float(w0.entropy[keep].double().mean()), "entropy": float(tr.entropy[keep].double().mean()),
            "dH": H_rise, "dH_se": H_se if rows > 1 else None, "T": T, "off_grid": off, "T_ref_rise": float(ref.mean()),
            "token_part": float(token.mean()), "rank1_change": rank1,
            "verdict": verdict(rise, ref, dH, token, row_of, off)}


def ceiling(w0: Full, keep: Tensor) -> dict[str, Any]:
    """The most mean log P any temperature earns here, read through the same table: the flatten rule's positive control."""
    if not bool(keep.any()) or w0.grid_logp is None or w0.grid_entropy is None:
        return {"n": 0, "verdict": "unread"}
    i = int(w0.grid_logp[keep].double().mean(0).argmax())
    as_read = replace(w0.read, logp=w0.grid_logp[:, i])
    tr = Full(as_read, w0.grid_entropy[:, i], w0.target)
    return {"best_T": TEMPS[i], "at_grid_edge": i in (0, len(TEMPS) - 1), **table(w0, tr, tr, keep)}


def pools(seen: Tensor) -> dict[str, Tensor]:
    return {"all": torch.ones_like(seen, dtype=torch.bool), **{c: seen == v for c, v in DIS.CLASSES.items()}}


def save(name: str, f: Full, *extra: tuple[str, Tensor]) -> None:
    LARGE.mkdir(parents=True, exist_ok=True)
    save_file({"logp": f.read.logp.float(), "rank": f.read.rank.int(), "entropy": f.entropy.float(), "row_of": f.read.row_of.int(),
               "target": f.target.int(), **dict(extra)}, str(LARGE / f"{name}.safetensors"))


# ----------------------------------------------------------------------------- one site


class Sites:
    """The model at W0; per row set its scoring half, W0's read there with the grid, the position classes and the fit inputs."""

    def __init__(self, lab: Lab) -> None:
        self.lab, self.model, self.cache = lab, lab.fp32(), {}
        self.user = int(lab.tok.convert_tokens_to_ids("<|im_start|>")), lab.tok("user", add_special_tokens=False)["input_ids"][0]

    def base(self, arm: str) -> dict[str, Any]:
        build = RUN.BUILD[arm.split("_")[0]]
        if build.__name__ not in self.cache:
            rows = build(self.lab)
            score = rows.half("score")
            w0 = read_full(self.model, score, grid=True)
            seen = DIS.question_seen(score, *self.user)
            assert len(seen) == len(w0.entropy)
            self.cache[build.__name__] = {"rows": rows, "score": score, "w0": w0, "seen": seen, "whole": None, "fit": {}}
            assert w0.grid_logp is not None and w0.grid_entropy is not None
            save(f"W0_{rows.name}", w0, ("grid_logp", w0.grid_logp), ("grid_entropy", w0.grid_entropy), ("seen", seen.int()))
            cap = {p: ceiling(w0, k) for p, k in pools(seen).items()}
            self.cache[build.__name__]["ceiling"] = cap
            self.lab.log(f"[W0 {rows.name}] ceiling {json.dumps(cap)}, {self.lab.peaks()}")
        return self.cache[build.__name__]

    def fit_input(self, b: dict[str, Any], layer: int) -> Tensor:
        if layer not in b["fit"]:
            b["fit"][layer] = qc.inputs_at(self.model, b["rows"].half("fit"))[layer]
        return b["fit"][layer]

    def site(self, arm: str, step: int, seeds: Sequence[int], rel: float) -> dict[str, Any]:
        lab, t0, b = self.lab, time.time(), self.base(arm)
        score, w0, seen = b["score"], b["w0"], pools(b["seen"])
        W = plus(lab.W0, load(RUN.path_of(arm, step), RUN.DEV))
        with set_weights(self.model, W):
            st = read_full(self.model, score)
        filed = json.loads((RES / "states" / f"{arm}_step{step}.json").read_text())["steering"]
        name = f"{arm}_step{step}"
        out: dict[str, Any] = {"arm": arm, "step": step, "rows": score.name, "E92": {k: filed.get(k) for k in ("layer", "gain_score", "v_norm", "variance_share")},
                               "ceiling": b["ceiling"], "max_logp_diff_state": float((st.read.logp - w0.read.logp).abs().max()),
                               "max_entropy_diff_state": float((st.entropy - w0.entropy).abs().max()), "treatments": {}}
        treat = {"state": st}
        layer = filed["layer"]
        if layer is not None:  # E92's rule: no gain, no vector
            X = self.fit_input(b, layer)
            with set_weights(self.model, W):
                Xs = qc.inputs_at(self.model, b["rows"].half("fit"))[layer]
            v = (Xs - X).mean(0)
            del Xs
            vecs = {"steer": v, "random_norm": qc.random_direction(X, float(v.norm()), RUN.STEER_SEED),
                    **{f"random_var_{i}": r for i, r in enumerate(matched_random(X, v, seeds, rel))}}
            out["vectors"] = {k: {"norm": float(r.norm()), "variance_share": qc.variance_share(X, r)} for k, r in vecs.items()}
            for k, r in vecs.items():
                treat[k] = read_full(self.model, score, (layer, r))
        for k, f in treat.items():
            save(f"{name}__{k}", f)
            out["treatments"][k] = {p: table(w0, f, st, keep) for p, keep in seen.items()}
            lab.log(f"[{name} {k}] {json.dumps(out['treatments'][k])}")
        if layer is not None:
            t = out["treatments"]
            out["vector_beyond_matched"] = {p: beyond(t["steer"][p], [t[f"random_var_{i}"][p] for i in range(len(seeds))]) for p in seen}
        out["seconds"] = time.time() - t0
        OUT.joinpath(f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
        lab.log(f"[{name}] verdict pooled {json.dumps({k: v['all']['verdict'] for k, v in out['treatments'].items()})}, {out['seconds']:.0f} s, {lab.peaks()}")
        return out

    def early(self, arm: str, step: int) -> dict[str, Any]:
        """E92's share at another saved step, by E92's own `qc.steer` with its layer sweep, on all rows."""
        lab, t0, b = self.lab, time.time(), self.base(arm)
        if b["whole"] is None:
            b["whole"] = qc.read(self.model, b["rows"])
        dW = load(RUN.path_of(arm, step), RUN.DEV)
        size, W = qc.update_size(dW, lab.W0)["total"], plus(lab.W0, dW)
        del dW
        with set_weights(self.model, W):
            st = qc.read(self.model, b["rows"])
        d = st.logp - b["whole"].logp
        out = {"arm": arm, "step": step, "norm": size["norm"], "gain": qc.share(d, d, st.row_of), "steering": qc.steer(self.model, b["rows"], b["whole"], st, W, RUN.STEER_SEED),
               "seconds": time.time() - t0}
        OUT.joinpath(f"share_{arm}_step{step}.json").write_text(json.dumps(out, indent=1) + "\n")
        lab.log(f"[share {arm}:{step}] {json.dumps(out)}, {lab.peaks()}")
        return out


# ----------------------------------------------------------------------------- the controls, first and alone


def controls(s: Sites, P: dict[str, Any]) -> None:
    c, lab = P["controls"], s.lab
    b = s.base(ZERO[0])
    w0, saved = b["w0"], DIS.load("W0_a1")
    whole = b["whole"] = qc.read(s.model, b["rows"])  # batched as E92's, so it repeats to the bit; the scoring half is batched otherwise
    out: dict[str, Any] = {"W0_vs_E92_max": float((whole.logp - saved["logp"]).abs().max()), "W0_ranks_equal": bool((whole.rank == saved["rank"]).all()),
                           "half_vs_whole_max": float((w0.read.logp - whole.logp[whole.row_of % 2 == 1]).abs().max()),
                           "T1_vs_W0_max": float((w0.grid_logp[:, T1] - w0.read.logp).abs().max())}
    every = torch.ones_like(b["seen"], dtype=torch.bool)
    out["W0_against_itself"] = table(w0, w0, w0, every)
    lab.log(f"[control W0] {json.dumps(out)}")
    if out["W0_vs_E92_max"] > c["zero_logp_max"] or not out["W0_ranks_equal"] or out["T1_vs_W0_max"] > c["zero_logp_max"] or out["W0_against_itself"]["verdict"] != "no rise":
        raise Stop(f"W0 is not E92's W0: {out}")
    z = s.site(*ZERO, DRAWS, c["variance_match_rel"])
    out["do_nothing"] = {k: z[k] for k in ("max_logp_diff_state", "max_entropy_diff_state")} | {"verdict": z["treatments"]["state"]["all"]["verdict"], "vectors": "vectors" in z}
    lab.log(f"[control do-nothing] {json.dumps(out['do_nothing'])}")
    if max(z["max_logp_diff_state"], z["max_entropy_diff_state"]) > c["zero_logp_max"] or out["do_nothing"]["verdict"] != "no rise" or out["do_nothing"]["vectors"]:
        raise Stop(f"the zero state does not read W0: {out['do_nothing']}")
    for arm, step in CONTROLS:
        r, want = s.site(arm, step, DRAWS, c["variance_match_rel"]), c["share"][arm]
        got = {k: r["treatments"][t]["all"]["share_of_state_gain"] for k, t in (("steer", "steer"), ("random", "random_norm"))}
        out[arm] = {"read": got, "filed": want, "off": {k: math.inf if got[k] is None else abs(got[k] - want[k]) for k in got}}
        lab.log(f"[control {arm}:{step}] {json.dumps(out[arm])}; band {c['share_band']}")
        if max(out[arm]["off"].values()) > c["share_band"]:
            raise Stop(f"{arm}:{step} does not reproduce E92's shares: {out[arm]}")
    OUT.joinpath("controls.json").write_text(json.dumps(out, indent=1) + "\n")


# ----------------------------------------------------------------------------- the filed readings


def readings() -> dict[str, Any]:
    """The filed readings from the written files: `rules.reading_size` and `rules.reading_step`."""
    def rec(name: str) -> dict[str, Any]:
        return json.loads((OUT / f"{name}.json").read_text())

    out: dict[str, Any] = {"sizes": {}, "share_trend": {}}
    for where, names in (("matched", ("anchor2_s0_step29", "anchor2_s142_step30")), ("full", ("anchor2_s0_step1250", "anchor2_s142_step1250"))):
        r, row = [rec(n) for n in names], {}
        for pool in ("all", *DIS.CLASSES):
            row[pool] = {t: agree([x["treatments"][t][pool]["verdict"] for x in r]) for t in ("state", "steer", "random_norm", *(f"random_var_{i}" for i in range(len(DRAWS))))}
            row[pool]["vector_beyond_matched"] = agree([str(x["vector_beyond_matched"][pool]) for x in r])
        row["share_reads_flattening"] = row["all"]["state"] == "flattens" and row["all"]["steer"] == "flattens"
        out["sizes"][where] = row
    for arm, steps in (("anchor1_s0", (10, 20, 50)), ("anchor1_s1", (10,))):
        late = json.loads((RES / "states" / f"{arm}_step100.json").read_text())["steering"]["steer"]["all"]
        shares = {t: rec(f"share_{arm}_step{t}")["steering"] for t in steps}
        read = {t: (x["steer"]["all"] if x["steer"] else {"share": None, "share_se": None}) for t, x in shares.items()}
        first = next((read[t] for t in steps if read[t]["share"] is not None), read[steps[0]])
        out["share_trend"][arm] = {"steps": {t: {"share": read[t]["share"], "se": read[t]["share_se"], "layer": shares[t]["layer"]} for t in steps}, "at_100": late["share"], "verdict": trend(first, late)}
    template = [rec(f"{a}_step{t}")["treatments"]["state"]["all"]["verdict"] for a, t in CONTROLS]
    out["style_rule_powered"] = agree(template) == "style"
    out["template_state"] = template
    out["flatten_ceiling"] = {n: rec(n)["ceiling"]["all"] for n in ("anchor1_s0_step100", "anchor2_s0_step29")}
    v = [out["share_trend"][a]["verdict"] for a in ("anchor1_s0", "anchor1_s1")]
    out["share_trend"]["verdict"] = "no call" if "undefined" in v else agree(v)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    for arm, step in (ZERO, *CONTROLS, *SITES, *EARLY):  # a bad list fails before the model loads
        if not RUN.path_of(arm, step).is_file():
            sys.exit(f"{arm}:{step}: no file at {RUN.path_of(arm, step)}")
    lab = Lab("C", "E92 Amendment 1", PRED)
    P = json.loads(PRED.read_text())
    try:
        s = Sites(lab)
        if not (OUT / "controls.json").is_file():
            controls(s, P)
        for arm, step in SITES:
            if not (OUT / f"{arm}_step{step}.json").is_file():
                s.site(arm, step, DRAWS, P["controls"]["variance_match_rel"])
        for arm, step in EARLY:
            if not (OUT / f"share_{arm}_step{step}.json").is_file():
                s.early(arm, step)
    except Stop as e:
        STOPPED.write_text(json.dumps({"stopped": str(e)}, indent=1) + "\n")
        lab.log(f"STOP: {e}")
        hard_exit(1)
    r = readings()
    OUT.joinpath("readings.json").write_text(json.dumps(r, indent=1) + "\n")
    lab.log(f"[readings] {json.dumps(r)}")
    lab.log(f"[done] {lab.peaks()}")
    print("FINISHED", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)

"""teacher_spread (E37): teacher training of a fine-tune with no trigger. Plan: reports/plans/2026-10-26_plan_teacher-spread.md.

The teacher is E36 Amendment 1's lowercase fine-tune (all 24 matrices, next-token loss at every position of lowercased
Pile rows, Adam 3e-5, batches of eight 512-token rows each used once, 300 steps), regenerated in-job. States, each
read on held-out rows at every position (`commute.row_copy`):
  W0            nothing changed
  do-nothing    the teacher itself: delta-W into Delta, subcomponents untouched
  adjoint       the scaled adjoint split of delta-W (input moments on the teacher's training rows), as E34 ran it
  lower_s0      teacher training on the teacher's own lowercase training rows, batch order 0
  plain_s0      teacher training on the same Pile documents, not lowercased, batch order 0
  lower_s1      as lower_s0 at batch order 1: the uniqueness reading
Per state: logp (held-out lowercase rows' mean next-token log-probability), gap KL(teacher || state) on those rows,
damage KL(W0 || state) on held-out plain Pile rows, the Pile harness ratios to W0 (20-step adversary, 256 val rows;
not at W0), the budget adversary's KL (reported; not at lower_s1), size, cosine with delta-W. `score` derives
s = (logp - logp_W0) / (logp_teacher - logp_W0), gap / gap_W0, damage over the teacher's, verdicts, ratios to W0.

Stages, cheapest first, each written to results/teacher_spread.json as it finishes and skipped on a rerun: C1 (W0 on
adversary seeds 0-2: faith_score.w0_check, floor within 5% of the card's known one), C2 (the regenerated teacher's
held-out gain at step 300 within 0.005 of E36's 0.1301 and >= 0.02), then the states above. Either failing stops.

Usage (repo root, project venv):  run.py --predictions   files results/predictions.json (the lead runs it, once)
                                  run.py [--device cuda]  the run (GPU); refuses without the predictions file
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "teacher_spread.json"
TRANSPORT = HERE.parent / "transport_test" / "results"
# W0's harness floor (max / min - 1 of the PGD KL over adversary seeds 0-2) per card, from earlier full harness runs:
# H100 E34 and its amendments, L40 lora_matched and cancellation_sgd, 4090 gamma_shift_map
KNOWN_FLOOR = {"NVIDIA H100 80GB HBM3": 0.0748, "NVIDIA L40": 0.2088, "NVIDIA GeForce RTX 4090": 0.0354}
S_MIN, GAP_MAX, DAMAGE_BAND, ADJ_MAX, COS_SAME, COS_MANY = 0.9, 0.10, (0.5, 2.0), 0.5, 0.8, 0.5


@dataclass(frozen=True)
class Config:
    rows_train: tuple[str, int, int] = ("train", 0, 4096)  # E36 Amendment 1's source rows, lowercased
    rows_held: tuple[str, int, int] = ("val", 0, 256)
    n_held: int = 64
    ft_lr: float = 3e-5
    ft_batch: int = 8
    ft_steps: int = 300
    ft_seed: int = 0
    e36_gain: float = 0.1301  # E36 Amendment 1 at step 300 (4090); Amendments 2-4 regenerated it within 2e-4 in loss
    gain_tol: float = 0.005
    min_gain: float = 0.02  # E36's gate G
    copy_lr: float = 1e-5  # E34's Pile-text arm: Adam 1e-5, batch 8, 1000 steps, plain mean KL
    copy_batch: int = 8
    copy_steps: int = 1000
    copies: tuple[tuple[str, int], ...] = (("lower", 0), ("plain", 0), ("lower", 1))  # (text, batch-order seed)
    n_sigma_rows: int = 256  # the adjoint split's input moments: the first teacher training rows
    pile_rows: int = 256  # E26's harness sizes
    pile_pgd_steps: int = 20
    pile_seeds: tuple[int, ...] = (0, 1, 2)
    floor_tol: float = 0.05  # C1: |floor / KNOWN_FLOOR - 1| <= this
    budget_rows: int = 128  # reported: E20 A3's budget adversary on the first Pile val rows, 20 sign steps of 0.1
    budget_steps: int = 20
    eval_batch: int = 8
    null_seed: int = 0  # the random direction whose cosine with delta-W is the null

    def __post_init__(self) -> None:
        assert self.copies[:1] == (("lower", 0),) and {("plain", 0), ("lower", 1)} <= set(self.copies)
        assert self.n_sigma_rows <= self.ft_steps * self.ft_batch <= self.rows_train[2]
        assert self.n_held <= self.rows_held[2] and self.n_held <= self.pile_rows
        assert (self.pile_rows, self.pile_pgd_steps, self.pile_seeds[0]) == (256, 20, 0)


CFG = Config()


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "run.log").open("a") as f:
        f.write(line + "\n")


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))


def halt(out: dict[str, Any], why: str) -> None:
    out["status"] = f"stopped: {why}"
    dump(out)
    log(f"STOP: {why}")
    sys.exit(f"STOP: {why}")


def peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


# ----------------------------------------------------------------------------- set-up and controls


class Job:
    """The model and harness (W0 read on three adversary seeds), the task rows, the teacher and the two reference
    nets (the teacher and W0, frozen copies of the target model)."""

    def __init__(self, device: str) -> None:
        from commute import harness

        t0 = time.time()
        token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
        self.ctx = harness.load(device, TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token,
                                1e-6, CFG.pile_rows, CFG.pile_pgd_steps, CFG.pile_seeds, log)
        self.card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        log(f"[load] {self.card}, {time.time() - t0:.0f} s, peak {peak_gb():.1f} GB")

    def c1(self) -> dict[str, Any]:
        from commute import faith_score

        w0 = faith_score.w0_check(self.ctx.pile)
        known = KNOWN_FLOOR.get(self.card)
        w0["known_floor"] = known
        w0["floor_ok"] = known is not None and abs(w0["pgd_floor"] / known - 1) <= CFG.floor_tol
        w0["pass"] = w0["pass"] and w0["floor_ok"]
        log(f"[C1] floor {w0['pgd_floor']:.4f} (known {known}) pgd_rel {w0['pgd_rel']:.4f} ci_rel {w0['ci_rel']:.4f} "
            f"pass {w0['pass']}")
        return w0

    def rows(self) -> None:
        """Lowercase training and held-out rows (E36 Amendment 1's), the plain rows the plain copy trains on (the
        first rows of the same source range, as many as the lowercase ones) and the plain held-out rows (val 0..63)."""
        from commute import vpd
        from commute.data import batches, pile_rows
        from commute.row_copy import lowercase_rows

        tok, n_train = vpd.tokenizer(), CFG.ft_steps * CFG.ft_batch
        src = pile_rows(*CFG.rows_train, log=log)
        self.lower = lowercase_rows(src, tok, n_train)
        self.held = lowercase_rows(pile_rows(*CFG.rows_held, log=log), tok, CFG.n_held)
        self.plain = src[:n_train]
        val = pile_rows("val", 0, CFG.pile_rows, log=log)
        self.plain_held, self.budget_bs = val[: CFG.n_held], batches(val[: CFG.budget_rows], 8)
        for r in (self.held[0], self.plain_held[0]):
            log("[rows] sample: " + tok.decode(r[:100].tolist()).replace("\n", "\\n"))

    def teacher(self) -> dict[str, Any]:
        """Regenerate the teacher (finetune_arm_b, E36's recipe), build the reference nets, read C2."""
        from commute import teacher
        from commute.finetune import ArmBConfig, finetune_arm_b
        from commute.row_copy import mean_logp

        ctx = self.ctx
        cfg = ArmBConfig(lr=CFG.ft_lr, batch=CFG.ft_batch, steps=CFG.ft_steps, checkpoints=(CFG.ft_steps,), seed=CFG.ft_seed)
        ft = finetune_arm_b(ctx.model, self.lower, self.held, cfg, ctx.device, log)
        self.dW = ft.deltas[CFG.ft_steps]
        self.t_net, self.w0_net = teacher.teacher_net(ctx.model, self.dW), teacher.teacher_net(ctx.model, {})
        gain = (mean_logp(self.t_net, self.held, ctx.device, CFG.eval_batch)
                - mean_logp(self.w0_net, self.held, ctx.device, CFG.eval_batch))
        c2 = {"gain": gain, "e36": CFG.e36_gain, "finetune_eval": ft.evals[CFG.ft_steps],
              "pass": abs(gain - CFG.e36_gain) <= CFG.gain_tol and gain >= CFG.min_gain}
        log(f"[C2] held-out gain {gain:.4f} (E36 {CFG.e36_gain}) pass {c2['pass']} | {json.dumps(c2['finetune_eval'])}")
        return c2


# ----------------------------------------------------------------------------- states


def read(job: Job, name: str, dW: dict[str, Tensor], uv2: dict[str, tuple[Tensor, Tensor]] | None,
         pile: bool, budget: bool = True) -> dict[str, Any]:
    """One state: target weights W0 + dW with the alive (U, V) of `uv2` (None: the released ones). `pile`: the harness
    ratios; `budget`: the budget adversary's KL (gamma.pgd_sources then cancel.masked_kl, as teacher_training's
    amend4 `budget_sweep` at 20 steps). Every number is logged in one line."""
    from commute import cancel, gamma, harness
    from commute.component_training import components_with
    from commute.faithfulness import model_state
    from commute.finetune import cosines
    from commute.row_copy import mean_kl, mean_logp

    ctx, b, dev = job.ctx, CFG.eval_batch, job.ctx.device
    target = {k: ctx.W0[k] + dW[k].to(ctx.W0[k].device) for k in ctx.W0}
    comps = None if uv2 is None else components_with(ctx.model, ctx.alive, uv2)
    t0 = time.time()
    with model_state(ctx.model, target=target, components=comps):
        net = ctx.model.target_model
        rec: dict[str, Any] = {"logp": mean_logp(net, job.held, dev, b), "gap": mean_kl(job.t_net, net, job.held, dev, b),
                               "damage": mean_kl(job.w0_net, net, job.plain_held, dev, b)}
        if pile:
            rec["pile"] = harness.pile_ratios(ctx, log)
        if budget:
            src, _ = gamma.pgd_sources(ctx.model, job.budget_bs, CFG.budget_steps, step_size=2 / CFG.budget_steps, seed=0)
            rec["budget_kl"] = cancel.masked_kl(ctx.model, job.budget_bs, src)
    rec["size"] = max(float(dW[k].norm() / ctx.W0[k].cpu().norm()) for k in dW)
    rec["cos_dW"] = cosines(dW, job.dW)["all"]
    log(f"[{name}] " + " ".join(f"{k} {v:.5g}" for k, v in rec.items() if isinstance(v, float))
        + (f" | pile {json.dumps({k: round(v, 4) for k, v in rec['pile'].items()})}" if pile else "")
        + f" | {time.time() - t0:.0f} s, peak {peak_gb():.1f} GB")
    return rec


def copy_path(text: str, seed: int) -> Path:
    return LARGE / f"copy_{text}_seed{seed}_step{CFG.copy_steps}.pt"


def copy_state(job: Job, text: str, seed: int) -> dict[str, Any]:
    """Teacher training on `text` rows at batch order `seed`, saved as bf16 offsets, then read."""
    from commute import arms
    from commute.row_copy import copy_on_rows

    t0 = time.time()
    curve, st = copy_on_rows(job.ctx, job.t_net, job.lower if text == "lower" else job.plain, CFG.copy_lr,
                             CFG.copy_batch, CFG.copy_steps, seed, log)
    log(f"[copy {text} seed {seed}] trained in {time.time() - t0:.0f} s; train KL at steps 1, 100, {CFG.copy_steps}: "
        + " ".join(f"{curve[i]['loss']:.4f}" for i in (0, 99, CFG.copy_steps - 1)) + f"; peak {peak_gb():.1f} GB")
    arms.save_offsets(copy_path(text, seed), job.ctx.uv_cpu, st, {"text": text, "seed": seed, "step": CFG.copy_steps})
    rec = read(job, f"{text}_s{seed}", {k: v[0] for k, v in st.items()}, {k: (v[1], v[2]) for k, v in st.items()}, True,
               budget=seed == 0)
    rec["train_kl_last"] = curve[-1]["loss"]
    return rec


def uniqueness(job: Job) -> dict[str, Any]:
    """The cosine between the two batch orders' lowercase copies, both rebuilt from their saved bf16 offsets, with the
    random direction's cosine with delta-W as the null."""
    from commute.finetune import cosines, random_direction
    from commute.saved_states import offsets_state

    d0, _ = offsets_state(copy_path("lower", 0), job.ctx.uv_cpu)
    d1, _ = offsets_state(copy_path("lower", 1), job.ctx.uv_cpu)
    c = cosines(d1, d0)
    rec = {"cos": c["all"], "cos_min_matrix": min(c["per_matrix"].values()),
           "null": cosines(random_direction(job.dW, CFG.null_seed), job.dW)["all"]}
    log(f"[uniqueness] cosine between batch orders {rec['cos']:.4f} (lowest matrix {rec['cos_min_matrix']:.4f}); "
        f"null {rec['null']:.5f}")
    return rec


def adjoint(job: Job) -> tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor]]]:
    from commute import split
    from commute.data import batches

    ctx = job.ctx
    sigma = split.input_second_moments(ctx.model, batches(job.lower[: CFG.n_sigma_rows], CFG.eval_batch), list(ctx.W0))
    dW_adj, st = split.adjoint_state(job.dW, ctx.uv_cpu, {k: v.cpu() for k, v in sigma.items()})
    return dW_adj, {k: (v[1], v[2]) for k, v in st.items()}


# ----------------------------------------------------------------------------- scoring (pure)


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on the results file; a prediction whose states are missing is not scored."""
    from commute import faith_score

    st, w0 = out["states"], out["C1"]
    if not {"W0", "do-nothing"} <= set(st):
        return {}
    base, top = st["W0"]["logp"], st["do-nothing"]["logp"]
    d: dict[str, Any] = {}
    for n, r in st.items():
        d[n] = {"s": (r["logp"] - base) / (top - base), "gap_share": r["gap"] / st["W0"]["gap"]}
        if n != "W0":
            d[n]["damage_over_teacher"] = r["damage"] / st["do-nothing"]["damage"]
        if "pile" in r:
            d[n]["verdict"] = {k: faith_score.verdict(r["pile"][k], w0[f]) for k, f in faith_score.FLOOR_KEY.items()}
            d[n]["anchor"] = r["pile"]["pgd_kl"] / st["do-nothing"]["pile"]["pgd_kl"]  # reported: the pending anchor
        if "budget_kl" in r:  # reported, not scored: the budget adversary's ratio to W0
            d[n]["budget_ratio"] = r["budget_kl"] / st["W0"]["budget_kl"]
    sc: dict[str, Any] = {"derived": d}
    if "lower_s0" in d:
        c = d["lower_s0"]
        sc |= {"P1": c["s"] >= S_MIN, "P2": c["gap_share"] <= GAP_MAX,
               "P3": all(v == "pass" for v in c["verdict"].values()),
               "P4": DAMAGE_BAND[0] <= c["damage_over_teacher"] <= DAMAGE_BAND[1]}
    if "adjoint" in d:
        sc["P5"] = d["adjoint"]["s"] < ADJ_MAX
    if "plain_s0" in d:
        sc["P6"] = d["plain_s0"]["s"] >= S_MIN
    if {"lower_s0", "lower_s1"} <= set(d):
        sc["s_batch_order_spread"] = abs(d["lower_s0"]["s"] - d["lower_s1"]["s"])  # reported: the noise floor of s
    if "uniqueness" in out:
        sc["P7"] = out["uniqueness"]["cos"] >= COS_SAME
        sc["P7_reading"] = ("one placement" if sc["P7"] else
                            "one of many placements" if out["uniqueness"]["cos"] < COS_MANY else "between")
    return sc


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    filed = json.loads(PRED.read_text())["config"]
    assert filed == json.loads(json.dumps(asdict(CFG))), f"predictions were filed for another config: {filed}"
    for f in ("alive.pt", "windows.pt"):
        if not (TRANSPORT / "large" / f).is_file():
            sys.exit(f"missing input (gitignored): {TRANSPORT / 'large' / f}")
    job = Job(device)
    w0 = job.c1()
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"card": job.card, "states": {}}
    if out["card"] != job.card:
        sys.exit(f"{OUT} was written on {out['card']}; this job runs on {job.card}. Move the file aside to start again")
    out |= {"C1": {k: v for k, v in w0.items() if k != "pile"}, "W0_pile": w0["pile"], "status": "running"}
    dump(out)
    if not w0["pass"]:
        halt(out, "C1: W0 is not at its known values on this card")
    job.rows()
    out["C2"] = job.teacher()
    dump(out)
    if not out["C2"]["pass"]:
        halt(out, f"C2: the teacher's held-out gain {out['C2']['gain']:.4f} is not E36's {CFG.e36_gain}")
    states = out["states"]
    todo: list[tuple[str, Any]] = [
        ("W0", lambda: read(job, "W0", {k: torch.zeros_like(v) for k, v in job.dW.items()}, None, False)),
        ("do-nothing", lambda: read(job, "do-nothing", job.dW, None, True)),
        ("adjoint", lambda: read(job, "adjoint", *adjoint(job), True)),
        *[(f"{t}_s{s}", lambda t=t, s=s: copy_state(job, t, s)) for t, s in CFG.copies]]
    for name, fn in todo:
        if name not in states:
            states[name] = fn()
            out["score"] = score(out)
            dump(out)
            log(f"[score] {json.dumps({k: v for k, v in out['score'].items() if k != 'derived'})} "
                f"{name}: {json.dumps(out['score'].get('derived', {}).get(name, {}), default=float)}")
    out["uniqueness"] = uniqueness(job)
    out["score"] = score(out)
    out["status"] = "done"
    dump(out)
    log(f"[score] {json.dumps({k: v for k, v in out['score'].items() if k != 'derived'})}")
    log(f"FINISHED wrote {OUT}")


NOISE = ("states are deterministic per seed up to bf16 kernels (E36's teacher regenerated within 2e-4 in held-out loss "
         "across cards); the copies' batch-order spread is read in this run (lower_s0 against lower_s1); E34 A4's was "
         "0.005 in P(target) for task-text copies, 0.032 for Pile-text copies, and up to 25% in the harness ratio")


def pred(pid: str, rule: str, confidence: str, blind: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "predicted": "holds", "confidence": confidence, "blind_to": blind,
            "positive_control": control, "noise_floor": NOISE}


PREDICTIONS: dict[str, Any] = {
    "plan": "reports/plans/2026-10-26_plan_teacher-spread.md (E37). Filed before any run; never edited after.",
    "controls": {
        "C1": "first, alone: W0 under the harness on adversary seeds 0, 1, 2; faith_score.w0_check and the PGD floor "
              f"within {CFG.floor_tol:.0%} of the card's known floor ({KNOWN_FLOOR}); a card not listed stops",
        "C2": f"the teacher regenerated; held-out gain logp_teacher - logp_W0 within {CFG.gain_tol} of {CFG.e36_gain} "
              f"and >= {CFG.min_gain} nats per token; else stop before any copy",
    },
    "expected": {
        "W0": "s 0, gap share 1 (gap about 0.3 nats per token: E36 read KL(W0 || teacher) 0.29), damage 0, harness "
              "ratio 1 by definition, logp about -3.146",
        "do-nothing": "s 1 and gap 0 by construction (the readout's check), damage = the teacher's own (no reference; "
                      "E36 did not read plain text), harness: lean above the line (size 0.049 against E34's teachers' "
                      "0.012-0.034, which read 0.98-1.59)",
        "adjoint": "s below 0.5 (E34 kept 0.06-0.26 of P(target) on the word tasks), harness at or under the line "
                   "(a smaller change inside the subcomponents' span; E20 A3's adjoint state read 1.05)",
        "copies": "s >= 0.9, gap share <= 0.1, harness pass, damage within 0.5-2x the teacher's, size about half the "
                  "teacher's, cosine with delta-W 0.08-0.3 (E34) against a null of about 1e-4",
    },
    "predictions": [
        pred("P1", f"lower_s0: s >= {S_MIN}", "medium",
             "the mean over held-out positions: where the gain sits", "do-nothing reads 1; C2"),
        pred("P2", f"lower_s0: gap / gap_W0 <= {GAP_MAX}", "medium-low",
             "a mean dominated by the highest-KL positions", "do-nothing reads 0, W0 reads 1"),
        pred("P3", "lower_s0: harness verdict 'pass' on the PGD and stochastic ratios against the same-job floors "
                   "(faith_score.verdict: pass at or under 1.5 / (1 + floor), 1.40 on an H100)", "medium",
             "other adversary families and budgets; E34 A4 found this harness lenient on fine-tunes, not on copies",
             "W0's three adversary seeds (C1)"),
        pred("P4", f"lower_s0: {DAMAGE_BAND[0]} <= damage / damage_do-nothing <= {DAMAGE_BAND[1]}", "medium",
             "plain Pile text only", "W0 reads 0"),
        pred("P5", f"adjoint: s < {ADJ_MAX}", "medium", "as P1", "the copies' s"),
        pred("P6", f"plain_s0: s >= {S_MIN}", "low",
             "as P1; the plain rows are the same documents as the lowercase ones", "lower_s0"),
        pred("P7", f"cosine(lower_s0, lower_s1) of the weight changes >= {COS_SAME}; below {COS_MANY} reads 'one of "
                   "many placements'", "medium-low",
             "function: two copies equal in weights are equal in function, not the reverse",
             "random direction's cosine with delta-W (null, about 1e-4); E34 Amendment 5 reads the same on a trigger "
             "teacher"),
    ],
    "reported": ("every state's numbers; the anchor reading pgd(copy) / pgd(do-nothing) for the pending anchor decision; "
                 "the budget adversary (E20 A3; gamma.pgd_sources, 20 sign steps of 0.1, seed 0, then cancel.masked_kl "
                 "on the first 128 Pile val rows in batches of 8) as a ratio to W0's, at W0, do-nothing, adjoint, "
                 "lower_s0 and plain_s0, read in case E34 Amendment 5 finds the 20-step harness lenient on fine-tunes; "
                 "|s(lower_s0) - s(lower_s1)|; cosines with delta-W; size; train KL; peak GPU memory"),
    "decision_tree": {
        "C1 or C2 fails": "stop; nothing scored",
        "P1 and P3 hold": "teacher training validated end to end on a broad update; the larger model starts with it. "
                          "P7 below 0.5 adds a card: the placement is one of many before it is called interpretable",
        "P1 fails": "teacher training's passes are a narrow-task property; next: frozen Gamma (E15's refit) or the "
                    "subcomponents' span ('create' reading)",
        "P1 holds, P3 fails": "the same branch; the gap and damage profile (P2, P4) say which",
        "P6": "holds: a copy from ordinary text carries a style, as from Pile text in E34; fails: the larger model's "
              "copy needs text like the post-training data",
    },
}


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; filed predictions are never edited (add a dated report instead)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    PRED.write_text(json.dumps({**PREDICTIONS, "config": asdict(CFG), "filed": time.strftime("%Y-%m-%d %H:%M:%S")},
                               indent=1))
    print(f"filed {PRED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions.json and exit")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    file_predictions() if a.predictions else run(a.device)

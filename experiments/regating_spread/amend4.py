"""regating_spread (E36) Amendment 4: are Γ's original values the right space, or would any summary of the context do?
Plan: reports/2026-10-26_plan_regating-without-cues.md -> Amendment 6 (from Amendment 3's independent check).
Amendment 1's lowercase fine-tune is regenerated and read at its steps. Every rule is fitted on all 262,144 positions of
the first 512 training rows and gives each held-out position its original values plus a predicted change, clipped:
  near_g0     the mean change of the 8 fitted positions nearest in the original values (Amendment 3's near262; gate A0)
  near_resid  the same, nearest in the residual stream: the input of each layer's attention (what Γ reads for the query,
              key and value matrices) under the original model, each layer's 768 values scaled to unit length, 4 layers
  token       the mean change at fitted positions holding the same current token (>= MIN_TOKEN of them), else overall
  position    the mean change at fitted positions with the same index in the row
and reports, for near_g0, recovery by position in the row (BUCKETS), the token rule's coverage, and the share of held-out
positions inside a 16-token span that also occurs in the training rows (near-duplicate text).
Needs an H100.

  amend4.py --predictions   file results/predictions_amend4.json (once, before the run)
  amend4.py                 -> results/regating_spread_amend4.json, results/large/amend4.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import torch
from amend1 import CHECKS, task
from run import BATCH, K_NEAR, LR, MIN_TOKEN, N_FIT_ROWS, RESULTS, SEED, TRANSPORT, Run, _finite
from torch import Tensor

from commute import spread as S

PRED, OUT = RESULTS / "predictions_amend4.json", RESULTS / "regating_spread_amend4.json"
AMEND3 = RESULTS / "regating_spread_amend3.json"
N_ROWS, NGRAM, CHUNK = 8 * N_FIT_ROWS, 16, 1024
BUCKETS = ((0, 32), (32, 128), (128, 511))
REPRO_TOL, SPACE_TOL, POS_MAX, TOKEN_GAP, DUP_MAX = 0.02, 0.05, 0.3, 0.2, 0.02  # A0: Amendment 3 ran on an H100
SCORED = str(CHECKS[-1])

PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-26_plan_regating-without-cues.md, Amendment 6. Filed before the run; never edited after.",
    "measurements": {
        "R_near_g0": "Amendment 3's nearest rule on 262,144 positions (8 neighbours, original values); gate A0",
        "R_near_resid": "the same rule, nearest in the original model's attention inputs of the 4 layers, each layer "
                        "scaled to unit length (3,072 values)",
        "R_token": f"mean change per current token with >= {MIN_TOKEN} fitted positions, else the overall mean",
        "R_position": "mean change per index in the row",
        "R_near_g0_by_bucket": f"near_g0's recovery within position buckets {BUCKETS}; reported",
        "cov_token": "share of held-out positions whose token the token rule covers; reported",
        "dup": f"share of held-out positions inside a {NGRAM}-token span that also occurs in the training rows",
    },
    "gates": {"A0": f"every step: |R_near_g0 - Amendment 3's R_near262| <= {REPRO_TOL}; fail: stop"},
    "predictions": [
        {"id": "R1", "rule": f"step {SCORED}: R_near_resid >= R_near_g0 - {SPACE_TOL} ('the original values are not "
                             "special: the context itself predicts the change as well')", "predicted": "pass",
         "confidence": "low"},
        {"id": "R2", "rule": f"R_position < {POS_MAX} at every step ('position in the row does not carry it')",
         "predicted": "pass", "confidence": "medium"},
        {"id": "R3", "rule": f"R_token <= R_near_g0 - {TOKEN_GAP} at every step ('the token stays far behind at the same "
                             "sample size')", "predicted": "pass", "confidence": "medium"},
        {"id": "D", "rule": f"dup <= {DUP_MAX}; above it the nearest rule's large-sample result is unreadable",
         "predicted": "pass", "confidence": "medium"},
    ],
    "decision_tree": {
        "R1 holds": "similar contexts get similar changes, by any rich summary: the restatement says 'inputs the model "
                    "treats alike', with Γ's values one sufficient summary",
        "R1 fails": "Γ's original values predict the change better than the context they are computed from: the "
                    "restatement can say 'inputs the original Γ treats alike'",
        "R2 fails": "position in the row carries much of the switching: the result is partly an artefact of row cuts",
        "R3 fails": "the token rule closes the gap once it covers more positions: the cue reading returns",
        "D fails": "near-duplicate text inflates the large-sample result: rescore without those positions",
    },
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    with (RESULTS / "large" / "amend4.log").open("a") as f:
        f.write(line + "\n")


def resid(r: Run, rows: Tensor) -> Tensor:
    """(N * T, 4 * 768) the original model's attention inputs of the 4 layers at every position, each layer's block
    scaled to unit length."""
    keys = sorted(k for k in r.lay.keys if k.endswith("attn.q_proj"))
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, rows.shape[0], 8):
            cache = r.model(rows[s : s + 8].to(r.device), cache_type="input").cache
            out.append(torch.cat([torch.nn.functional.normalize(cache[k].float().flatten(0, 1), dim=1) for k in keys], 1))
    return torch.cat(out)


def nearest(X: Tensor, Xn: Tensor, Y: Tensor, k: int) -> Tensor:
    """`spread.nearest` with the fitted rows' squared norms Xn precomputed (the fitted set here is ~10 GB)."""
    return torch.cat([((Xn[None, :] - 2 * Y[s : s + CHUNK] @ X.T)).topk(k, largest=False).indices
                      for s in range(0, Y.shape[0], CHUNK)])


def masked_logp_pos(model: Any, rows: Tensor, gates: dict[str, Tensor]) -> Tensor:
    """(T - 1,) the masked model's log-probability of the next token at each position, summed over the rows."""
    from commute.local_faith import masked_logits

    with torch.no_grad():
        lg = masked_logits(model, rows, {k: v.float() for k, v in gates.items()}, None, None, autocast=False)[:, :-1]
    return torch.log_softmax(lg, -1).gather(2, rows[:, 1:, None]).squeeze(2).double().sum(0).cpu()


def duplicates(train: Tensor, held: Tensor) -> float:
    spans = {tuple(row[i : i + NGRAM].tolist()) for row in train for i in range(row.numel() - NGRAM + 1)}
    hit = torch.zeros(held.shape, dtype=torch.bool)
    for j, row in enumerate(held):
        for i in range(row.numel() - NGRAM + 1):
            if tuple(row[i : i + NGRAM].tolist()) in spans:
                hit[j, i : i + NGRAM] = True
    return float(hit.float().mean())


def evaluate(r: Run, held: Tensor, dW: dict[str, Tensor], f: dict[str, Any]) -> dict[str, Any]:
    from commute import regating as G
    from commute.vpd import compute_ci

    pos: dict[str, Tensor] = {}
    cov = 0.0
    for s in range(0, held.shape[0], 8):
        batch = held[s : s + 8].to(r.device)
        ci0 = compute_ci(r.model, batch)
        with G.state_of(r.model, r.W0, dW):
            ci1 = compute_ci(r.model, batch)
        x0, toks = S.to_flat(ci0.g, r.lay), batch.flatten()
        y = resid(r, batch)
        idx = torch.arange(batch.shape[1], device=r.device).repeat(batch.shape[0])
        changes = {"near_g0": S.nearest_change(f["D"], nearest(f["X"], f["Xn"], x0, K_NEAR)),
                   "near_resid": S.nearest_change(f["D"], nearest(f["Z"], f["Zn"], y, K_NEAR)),
                   "token": f["token"].change[f["token"].row_of[toks]], "position": f["position"][idx]}
        L = {"g0": masked_logp_pos(r.model, batch, ci0.g), "g1": masked_logp_pos(r.model, batch, ci1.g)}
        for name, c in changes.items():
            L[name] = masked_logp_pos(r.model, batch, S.with_alive(ci0.g, (x0 + c).clamp(0.0, 1.0), r.lay))
        for k, v in L.items():
            pos[k] = pos.get(k, 0.0) + v
        cov += float((f["token"].row_of[toks] < f["token"].n_tokens).float().sum())
        del ci0, ci1
    cov /= held.numel()
    tot = {k: float(v.sum()) for k, v in pos.items()}
    gap = tot["g1"] - tot["g0"]
    out: dict[str, Any] = {f"R_{k}": (v - tot["g0"]) / gap for k, v in tot.items() if k not in ("g0", "g1")}
    out["R_near_g0_by_bucket"] = {f"{a}-{b}": float((pos["near_g0"][a:b].sum() - pos["g0"][a:b].sum())
                                                     / (pos["g1"][a:b].sum() - pos["g0"][a:b].sum())) for a, b in BUCKETS}
    out["gap_share_by_bucket"] = {f"{a}-{b}": float((pos["g1"][a:b].sum() - pos["g0"][a:b].sum()) / gap) for a, b in BUCKETS}
    return {**out, "gap": gap / (held.shape[0] * (held.shape[1] - 1)), "cov_token": cov}


def fitted(r: Run, rows: Tensor, dW: dict[str, Tensor]) -> dict[str, Any]:
    X = r.values(rows, None)
    D = r.values(rows, dW).sub_(X)
    Z = resid(r, rows)
    tok = rows.to(r.device).flatten()
    T = rows.shape[1]
    lab = torch.arange(T, device=r.device).repeat(rows.shape[0])
    return {"X": X, "Xn": torch.einsum("ij,ij->i", X, X), "D": D, "Z": Z, "Zn": torch.einsum("ij,ij->i", Z, Z),
            "token": S.fit_tokens(tok, D, r.vocab, MIN_TOKEN), "position": S.group_change(D, lab, T)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; filed predictions are never edited")
        PRED.write_text(json.dumps({**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        return print(f"filed {PRED}")
    for p in (PRED, AMEND3, TRANSPORT / "large" / "alive.pt"):
        if not p.is_file():
            sys.exit(f"missing: {p}")
    from commute.finetune import ArmBConfig, finetune_arm_b

    ref = json.loads(AMEND3.read_text())["states"]
    t0, r = time.time(), Run(a.device)
    out: dict[str, Any] = {"states": {}, "status": "running", "device": torch.cuda.get_device_name()}
    train, held, _ = task()
    out["dup"] = duplicates(train[:N_ROWS], held)
    log(f"[dup] {out['dup']:.4f}")
    ft = finetune_arm_b(r.model, train, held, ArmBConfig(lr=LR, batch=BATCH, steps=CHECKS[-1], checkpoints=CHECKS,
                                                         seed=SEED), a.device, log)
    out["finetune"] = ft.evals
    for step in CHECKS:
        f = fitted(r, train[:N_ROWS], ft.deltas[step])
        m = evaluate(r, held, ft.deltas[step], f)
        m["A0"] = abs(m["R_near_g0"] - float(ref[str(step)]["R_near262"])) <= REPRO_TOL
        out["states"][str(step)] = m
        OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
        log(f"[step {step}] " + " ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float))
            + f" by_bucket {json.dumps(m['R_near_g0_by_bucket'])} gap_share {json.dumps(m['gap_share_by_bucket'])} A0 {m['A0']}")
        del f
        if not m["A0"]:
            out["status"] = f"stopped at A0 (step {step})"
            OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
            return log(out["status"])
    st = out["states"]
    out["score"] = {"R1": st[SCORED]["R_near_resid"] >= st[SCORED]["R_near_g0"] - SPACE_TOL,
                    "R2": all(m["R_position"] < POS_MAX for m in st.values()),
                    "R3": all(m["R_token"] <= m["R_near_g0"] - TOKEN_GAP for m in st.values()),
                    "D": out["dup"] <= DUP_MAX}
    out["seconds"], out["status"] = time.time() - t0, "done"
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))
    log("score " + json.dumps(out["score"]))
    log(f"FINISHED wrote {OUT} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()

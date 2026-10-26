"""regating_spread (E36): does a cue-free form of the fixed-shift hypothesis hold on a fine-tune with no trigger?
Plan: reports/2026-10-26_plan_regating-without-cues.md (E36, and Amendment 1, which files these predictions). Shared
steps in commute.spread (every-position values and patterns) and commute.cuefree (k-means).

Task: code-like Pile text, the TOP_FRAC of rows with the most tokens containing one of CODE_CHARS. Fine-tune: all
24 matrices, next-token loss at every position (finetune.finetune_arm_b: Adam LR, batch BATCH, fresh rows each step),
read at the steps in CHECKS. Do-nothing states as E28: released subcomponents, Gamma frozen and reading the updated
model, Delta out of the masked model.

At every position of the held-out rows: T = the target model's mean next-token log-probability, L(pattern) = the masked
model's with `pattern` at every position; gain = T1 - T0, gap = L(g1) - L(g0), r = gap / gain, and per pattern
R = (L(pattern) - L(g0)) / gap. Patterns are clip(g0(x) + c(x)) on alive subcomponents, c fitted at N_FIT_POS random
positions of the first N_FIT_ROWS training rows:
  shared   one mean change for every position (the old fixed-shift hypothesis, here without a trigger)
  token    the mean change at fit positions holding x's own token, if seen >= MIN_TOKEN times, else shared
  near     the mean change of the K_NEAR fit positions nearest x in g0
  clu_K    the mean change of x's k-means cluster of fit g0, K in KS
  rand     random labels at K_RAND (N_RAND draws): the size-matched null for clu
  fix      clip(a g0(x) + b), a, b per subcomponent by least squares over fit positions
Stages, cheapest first, results written after each: select rows (gate S), fine-tune, step CHECKS[0] state (gate C0:
zero change recovers 0, own alive values recover 1; gate G: gain and r), then the later steps.

  run.py --predictions   file results/predictions.json (once, before the run)
  run.py                 -> results/regating_spread.json, results/large/full.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import cuefree as CF
from commute import spread as S

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "regating_spread.json"
TRANSPORT = EXP / "transport_test" / "results"

ROWS_TRAIN, ROWS_HELD = ("train", 0, 20480), ("val", 0, 8192)
CODE_CHARS, TOP_FRAC, MIN_CUT = "{}[]();=<>_\\", 0.05, 0.12
N_TRAIN, N_HELD, N_FIT_ROWS, N_FIT_POS, EVAL_BATCH = 800, 64, 64, 16384, 8
LR, BATCH, CHECKS, SEED = 3e-5, 8, (10, 100), 0
K_NEAR, KS, K_RAND, N_RAND, MIN_TOKEN, KM_SEED = 8, (8, 32, 128, 512), 128, 3, 20, 0
MIN_GAIN, MIN_R, ZERO_TOL, OWN_TOL = 0.02, 0.1, 1e-4, 0.02

R_MIN, SHARED_MAX, NEAR_MIN, BEATS, CLOSE, FIX_BEATS, RAND_TOL = 0.3, 0.5, 0.6, 0.2, 0.1, 0.1, 0.05
SCORED = str(CHECKS[-1])

BLIND = ("one fine-tune seed and one held-out set of 64 rows; one 67M model; one domain (code-like text chosen by a "
         "punctuation rule); every predicted change is a mean over fit positions, blind to variation among them; "
         "Euclidean distance over every alive subcomponent weighs ones irrelevant to the text equally; patterns are "
         "written at every position at once, so effects through attention between positions are included, not separated")
NOISE = ("the states are deterministic per seed; the random-group arm (3 draws at K = 128) measures what averaging over "
         "groups does alone; E35 (same arms at the trigger) reproduced E32/E33 to 0.001")


def pred(pid: str, rule: str, predicted: str, confidence: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "predicted": predicted, "confidence": confidence, "positive_control": control,
            "blind_to": BLIND, "noise_floor": NOISE}


PREDICTIONS: dict[str, Any] = {
    "plan": "reports/2026-10-26_plan_regating-without-cues.md, E36 and Amendment 1 (filed after E35 was read). Filed "
            "before any run; never edited after.",
    "task": f"Pile rows ('train' {ROWS_TRAIN[1:]}, 'val' {ROWS_HELD[1:]}); a token is marked if its text contains one "
            f"of {CODE_CHARS!r}; rows kept: code share >= the training stream's top-{TOP_FRAC} cut (cut >= {MIN_CUT}); "
            f"{N_TRAIN} training rows, {N_HELD} held-out rows",
    "finetune": f"finetune.finetune_arm_b: all 24 matrices, next-token loss at every position, Adam {LR}, batch {BATCH}, "
                f"fresh rows each step, seed {SEED}; states read at steps {CHECKS}; scored at step {SCORED}, step "
                f"{CHECKS[0]} read by the same rules and reported",
    "measurements": {
        "gain, gap, r": "T = target model's mean next-token log-prob over held-out positions 0..510; L = the masked "
                        "model's (released subcomponents, Delta out) with a pattern at every position; gain = T1 - T0, "
                        "gap = L(g1) - L(g0), r = gap / gain",
        "R_<arm>": "(L(pattern) - L(g0)) / gap; patterns clip(g0 + c) on alive subcomponents, fitted at "
                   f"{N_FIT_POS} random positions of the first {N_FIT_ROWS} training rows: shared (one mean change), "
                   f"token (mean change at fit positions with the same current token, >= {MIN_TOKEN} of them, else "
                   f"shared), near (mean change of the {K_NEAR} fit positions nearest in g0), clu_K (k-means of fit "
                   f"g0, K in {KS}), fix (clip(a g0 + b) per subcomponent, cuefree.line)",
        "R_rand": f"random labels at K = {K_RAND}, mean of {N_RAND} draws: the null for clu",
        "near_same_token": f"share of each held-out position's {K_NEAR} nearest fit positions holding its own token; "
                           "reported",
        "slopes": "share of alive subcomponents whose fit g0 varies (var > cuefree.FLAT_VAR), the median slope a "
                  "among them and the share with a < 0.5; reported",
    },
    "gates": {
        "S": f"selection: cut >= {MIN_CUT}, >= {N_TRAIN} training and >= {N_HELD} held-out rows; fail: stop",
        "C0": f"at step {CHECKS[0]}, first: |R_zero| <= {ZERO_TOL} and |R_own_alive - 1| <= {OWN_TOL}; fail: stop",
        "G": f"a step is scorable only if gain >= {MIN_GAIN} nats per token and r >= {MIN_R}",
    },
    "expected": {
        "W0 (no update)": "gap 0, every share undefined; zero change recovers exactly 0",
        "control": "R_zero 0 exactly; R_own_alive 1 within 0.02 (dead subcomponents keep W0's values)",
        "near_same_token": "lean >= 0.5: the current token dominates Gamma's values at a position",
        "slopes": "lean: median slope among varying subcomponents below 1 (each pulled towards a set point, E35's line "
                  "reading on one-cue tasks)",
        "K curve": "R_clu_K rising with K, from about R_shared",
    },
    "predictions": [
        pred("P_r", f"step {SCORED} scorable and r >= {R_MIN}", "pass", "low-medium",
             "G (the fine-tune learned); E28's tasks the model already does read 0.5 to 1.1"),
        pred("P_shared", f"R_shared < {SHARED_MAX}", "pass", "medium-low", "N"),
        pred("P_near", f"R_near >= {NEAR_MIN} and R_near >= R_shared + {BEATS}", "pass", "medium",
             "E35: the nearest arm recovered 0.61 to 0.92 at the trigger on six states"),
        pred("P_token", f"R_near >= R_token - {CLOSE}", "pass", "medium", "P_near"),
        pred("P_fix", f"R_fix >= R_shared + {FIX_BEATS}", "pass", "low-medium",
             "E35: the line recovered 0.96 and 1.02 on the two few-cue states"),
        pred("P_K", f"R_clu_{KS[-1]} >= R_near - {CLOSE}", "pass", "low", "N"),
        pred("N", f"|R_rand - R_shared| <= {RAND_TOL}", "pass", "medium", "C0"),
    ],
    "decision_tree": {
        "S or C0 fails": "stop; nothing scored",
        "G fails at both steps": "re-gating carries little of a spread-out fine-tune's gain: a card on the line's scope",
        "P_near holds": "the restated hypothesis ('positions the original Gamma treats alike are re-gated alike') "
                        "holds with no trigger: provisional until an independent check; next, the larger model (Q9)",
        "P_near fails, R_token holds (R_token >= NEAR_MIN)": "the current token carries what the original values do not: "
                                                             "card",
        "P_near and R_token fail, P_fix holds": "the per-subcomponent set-point form: card",
        "all fail": "the change at a position depends on more than its own original values (context, neighbouring "
                    "positions): card (next space: Gamma's input)",
    },
}


def score(states: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on every scorable step; the predictions are scored at SCORED."""
    out: dict[str, Any] = {}
    for step, m in states.items():
        if not (m["C0"] and m["G"]):
            out[step] = {"scorable": False}
            continue
        R = m["R"]
        out[step] = {"scorable": True, "P_r": m["r"] >= R_MIN, "P_shared": R["shared"] < SHARED_MAX,
                     "P_near": R["near"] >= NEAR_MIN and R["near"] >= R["shared"] + BEATS,
                     "P_token": R["near"] >= R["token"] - CLOSE, "P_fix": R["fix"] >= R["shared"] + FIX_BEATS,
                     "P_K": R[f"clu_{KS[-1]}"] >= R["near"] - CLOSE, "N": abs(m["R_rand"] - R["shared"]) <= RAND_TOL}
    return out


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


def dump(out: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_finite(out), indent=1, sort_keys=True))


# ----------------------------------------------------------------------------- the task


def code_tokens(tok: Any, vocab: int) -> Tensor:
    """(vocab,) bool: the token's text contains one of CODE_CHARS (ids past the tokenizer's, the logits' padding: False)."""
    n = min(len(tok), vocab)
    return torch.tensor([any(c in tok.decode([i]) for c in CODE_CHARS) for i in range(n)] + [False] * (vocab - n))


def task(vocab: int) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """(training rows, held-out rows, the selection's record). A row's code share is the fraction of its tokens that
    `code_tokens` marks; the rows kept are those at or above the training stream's top-TOP_FRAC cut, in both ranges."""
    from commute import vpd
    from commute.data import pile_rows

    tok = vpd.tokenizer()
    mark = code_tokens(tok, vocab)
    tr_all, he_all = pile_rows(*ROWS_TRAIN, log=log), pile_rows(*ROWS_HELD, log=log)
    share_tr, share_he = mark[tr_all].float().mean(1), mark[he_all].float().mean(1)
    cut = float(torch.quantile(share_tr, 1 - TOP_FRAC))
    tr, he = tr_all[share_tr >= cut], he_all[share_he >= cut]
    info = {"cut": cut, "median_share": float(share_tr.median()), "n_train": tr.shape[0], "n_held": he.shape[0],
            "marked_tokens": int(mark.sum()),
            "samples": [tok.decode(he[i, :120].tolist()) for i in range(min(3, he.shape[0]))]}
    info["S"] = cut >= MIN_CUT and tr.shape[0] >= N_TRAIN and he.shape[0] >= N_HELD
    log(f"[select] {json.dumps({k: v for k, v in info.items() if k != 'samples'})} | {'ok' if info['S'] else 'FAIL'}")
    for s in info["samples"]:
        log("[select] sample: " + s.replace("\n", "\\n")[:400])
    return tr[:N_TRAIN], he[:N_HELD], info


# ----------------------------------------------------------------------------- reads and fits


class Run:
    def __init__(self, device: str) -> None:
        from commute import vpd
        from commute.finetune import logits_of

        self.model = vpd.load_component_model(device)
        self.W0, self.device = vpd.target_weights(self.model), device
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.lay = S.layout({k: v.cpu().bool() for k, v in alive.items()}, device)
        with torch.no_grad():
            self.vocab = int(logits_of(self.model.target_model, torch.zeros(1, 2, dtype=torch.long, device=device)).shape[-1])

    def values(self, rows: Tensor, dW: dict[str, Tensor] | None) -> Tensor:
        """Flat values at every position of `rows`, (N * T, D) on the device, under W0 or the state W0 + dW."""
        from commute import regating as G
        from commute.vpd import compute_ci

        def read() -> Tensor:
            return torch.cat([S.to_flat(compute_ci(self.model, rows[s : s + EVAL_BATCH]).g, self.lay)
                              for s in range(0, rows.shape[0], EVAL_BATCH)])

        if dW is None:
            return read()
        with G.state_of(self.model, self.W0, dW):
            return read()

    def fit(self, rows: Tensor, dW: dict[str, Tensor]) -> dict[str, Any]:
        """Every predictor of the module docstring, from N_FIT_POS random positions of `rows`."""
        gen = torch.Generator().manual_seed(SEED)
        pick = torch.randperm(rows.numel(), generator=gen)[:N_FIT_POS].to(self.device)
        X, X1 = self.values(rows, None)[pick], self.values(rows, dW)[pick]
        Dch = X1 - X
        a, b = S.line(X, X1)
        del X1
        varying = X.var(0, unbiased=False) > CF.FLAT_VAR
        tokens = rows.to(self.device).flatten()[pick]
        rand = [torch.randint(K_RAND, (X.shape[0],), generator=torch.Generator().manual_seed(i)).to(self.device)
                for i in range(N_RAND)]
        return {"X": X, "D": Dch, "tokens": tokens, "shared": Dch.mean(0), "a": a, "b": b,
                "token": S.fit_tokens(tokens, Dch, self.vocab, MIN_TOKEN),
                "clu": {K: S.fit_clusters(X, Dch, K, KM_SEED) for K in KS},
                "rand": [S.group_change(Dch, lab, K_RAND) for lab in rand],
                "slopes": {"share_varying": float(varying.float().mean()),
                           "a_median_varying": float(a[varying].median()) if varying.any() else float("nan"),
                           "a_below_half_varying": float((a[varying] < 0.5).float().mean()) if varying.any()
                           else float("nan")}}

    def patterns(self, x0: Tensor, x1: Tensor, toks: Tensor, near: Tensor, f: dict[str, Any],
                 batch_no: int) -> Iterator[tuple[str, Tensor]]:
        """Flat alive values of every pattern at one batch's positions (x0, x1: W0's and the state's; `near`: each
        position's K_NEAR nearest fit positions), one at a time."""
        clip = lambda c: (x0 + c).clamp(0.0, 1.0)
        tt = f["token"]
        yield "own_alive", x1
        yield "zero", clip(torch.zeros_like(x0))
        yield "shared", clip(f["shared"])
        yield "token", clip(tt.change[tt.row_of[toks]])
        yield "near", clip(S.nearest_change(f["D"], near))
        yield "fix", (f["a"] * x0 + f["b"]).clamp(0.0, 1.0)
        for K, c in f["clu"].items():
            yield f"clu_{K}", clip(c.change[S.assign(x0, c.cent)])
        for i, ch in enumerate(f["rand"]):
            gen = torch.Generator().manual_seed(1000 * (i + 1) + batch_no)
            yield f"rand_{i}", clip(ch[torch.randint(K_RAND, (x0.shape[0],), generator=gen).to(x0.device)])

    def evaluate(self, rows: Tensor, dW: dict[str, Tensor], f: dict[str, Any]) -> tuple[dict[str, float], float]:
        """Per-position means over `rows` of T0, T1 and L of g0, g1 and every pattern; and the share of each position's
        K_NEAR nearest fit positions that hold the position's own token."""
        from commute import regating as G
        from commute.vpd import compute_ci

        sums: dict[str, float] = {}
        add = lambda k, v: sums.__setitem__(k, sums.get(k, 0.0) + v)
        same = 0.0
        for n, s in enumerate(range(0, rows.shape[0], EVAL_BATCH)):
            batch = rows[s : s + EVAL_BATCH].to(self.device)
            ci0 = compute_ci(self.model, batch)
            with G.state_of(self.model, self.W0, dW):
                ci1 = compute_ci(self.model, batch)
            add("T0", S.target_logp(ci0.logits, batch))
            add("T1", S.target_logp(ci1.logits, batch))
            add("g0", S.masked_logp(self.model, batch, ci0.g))
            add("g1", S.masked_logp(self.model, batch, ci1.g))
            x0, x1, toks = S.to_flat(ci0.g, self.lay), S.to_flat(ci1.g, self.lay), batch.flatten()
            near = S.nearest(f["X"], x0, K_NEAR)
            same += float((f["tokens"][near] == toks[:, None]).float().mean(1).sum())
            for name, x in self.patterns(x0, x1, toks, near, f, n):
                add(name, S.masked_logp(self.model, batch, S.with_alive(ci0.g, x, self.lay)))
            del ci0, ci1
        n_pos = rows.shape[0] * (rows.shape[1] - 1)
        return {k: v / n_pos for k, v in sums.items()}, same / rows.numel()


def shares(L: dict[str, float]) -> dict[str, Any]:
    gain, gap = L["T1"] - L["T0"], L["g1"] - L["g0"]
    R = {k: (v - L["g0"]) / gap if gap else float("nan") for k, v in L.items() if k not in ("T0", "T1", "g0", "g1")}
    rand = [R.pop(f"rand_{i}") for i in range(N_RAND)]
    return {"gain": gain, "gap": gap, "r": gap / gain if gain else float("nan"), "R": R,
            "R_rand": sum(rand) / len(rand), "rand_draws": rand}


def gates(m: dict[str, Any]) -> dict[str, Any]:
    c0 = abs(m["R"]["zero"]) <= ZERO_TOL and abs(m["R"]["own_alive"] - 1) <= OWN_TOL
    g = m["gain"] >= MIN_GAIN and m["r"] >= MIN_R
    return {"C0": c0, "G": g}


# ----------------------------------------------------------------------------- main


def run(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (run.py --predictions)")
    if not (TRANSPORT / "large" / "alive.pt").is_file():
        sys.exit(f"missing input (gitignored): {TRANSPORT / 'large' / 'alive.pt'}")
    from commute.finetune import ArmBConfig, finetune_arm_b

    t0 = time.time()
    r = Run(device)
    out: dict[str, Any] = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "states": {}, "status": "running"}
    log(f"[start] {out['device']} D {r.lay.D}")
    train, held, out["select"] = task(r.vocab)
    dump(out)
    if not out["select"]["S"]:
        out["status"] = "stopped at S"
        dump(out)
        return log("stopped at S")
    cfg = ArmBConfig(lr=LR, batch=BATCH, steps=CHECKS[-1], checkpoints=CHECKS, seed=SEED)
    ft = finetune_arm_b(r.model, train, held, cfg, device, log)
    out["finetune"] = ft.evals
    for step in CHECKS:
        dW = ft.deltas[step]
        f = r.fit(train[:N_FIT_ROWS], dW)
        L, same = r.evaluate(held, dW, f)
        m = shares(L)
        m.update(gates(m), near_same_token=same, slopes=f["slopes"], n_tokens_kept=f["token"].n_tokens,
                 n_clusters={K: int(c.cent.shape[0]) for K, c in f["clu"].items()})
        out["states"][str(step)] = m
        out["score"] = score(out["states"])
        dump(out)
        log(f"[step {step}] gain {m['gain']:.4f} gap {m['gap']:.4f} r {m['r']:.4f} | "
            + " ".join(f"{k} {v:.4f}" for k, v in m["R"].items()) + f" rand {m['R_rand']:.4f} | near_same_token "
            f"{same:.3f} slopes {json.dumps(f['slopes'])} | C0 {m['C0']} G {m['G']}")
        del f
        if step == CHECKS[0] and not m["C0"]:
            out["status"] = "stopped at C0"
            dump(out)
            return log("stopped at C0")
    out["seconds"], out["status"] = time.time() - t0, "done"
    dump(out)
    log(f"FINISHED wrote {OUT} and {LARGE / 'full.log'} ({out['seconds']:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions.json and exit")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.predictions:
        if PRED.is_file():
            sys.exit(f"{PRED} exists; filed predictions are never edited")
        assert PREDICTIONS, "PREDICTIONS is empty: file them in the plan's Amendment 1 first"
        RESULTS.mkdir(parents=True, exist_ok=True)
        PRED.write_text(json.dumps({**PREDICTIONS, "filed": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        print(f"filed {PRED}")
        return
    run(args.device)


if __name__ == "__main__":
    main()

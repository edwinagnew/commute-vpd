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

PREDICTIONS: dict[str, Any] = {}  # filed in Amendment 1 of the plan, after E35 is read


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
        tokens = rows.to(self.device).flatten()[pick]
        rand = [torch.randint(K_RAND, (X.shape[0],), generator=torch.Generator().manual_seed(i)).to(self.device)
                for i in range(N_RAND)]
        return {"X": X, "D": Dch, "shared": Dch.mean(0), "a": a, "b": b,
                "token": S.fit_tokens(tokens, Dch, self.vocab, MIN_TOKEN),
                "clu": {K: S.fit_clusters(X, Dch, K, KM_SEED) for K in KS},
                "rand": [S.group_change(Dch, lab, K_RAND) for lab in rand]}

    def patterns(self, x0: Tensor, x1: Tensor, toks: Tensor, f: dict[str, Any],
                 batch_no: int) -> Iterator[tuple[str, Tensor]]:
        """Flat alive values of every pattern at one batch's positions (x0, x1: W0's and the state's), one at a time."""
        clip = lambda c: (x0 + c).clamp(0.0, 1.0)
        tt = f["token"]
        yield "own_alive", x1
        yield "zero", clip(torch.zeros_like(x0))
        yield "shared", clip(f["shared"])
        yield "token", clip(tt.change[tt.row_of[toks]])
        yield "near", clip(S.nearest_change(f["X"], f["D"], x0, K_NEAR))
        yield "fix", (f["a"] * x0 + f["b"]).clamp(0.0, 1.0)
        for K, c in f["clu"].items():
            yield f"clu_{K}", clip(c.change[S.assign(x0, c.cent)])
        for i, ch in enumerate(f["rand"]):
            gen = torch.Generator().manual_seed(1000 * (i + 1) + batch_no)
            yield f"rand_{i}", clip(ch[torch.randint(K_RAND, (x0.shape[0],), generator=gen).to(x0.device)])

    def evaluate(self, rows: Tensor, dW: dict[str, Tensor], f: dict[str, Any]) -> dict[str, float]:
        """Per-position means over `rows`: T0, T1, and L of g0, g1 and every pattern."""
        from commute import regating as G
        from commute.vpd import compute_ci

        sums: dict[str, float] = {}
        add = lambda k, v: sums.__setitem__(k, sums.get(k, 0.0) + v)
        for n, s in enumerate(range(0, rows.shape[0], EVAL_BATCH)):
            batch = rows[s : s + EVAL_BATCH].to(self.device)
            ci0 = compute_ci(self.model, batch)
            with G.state_of(self.model, self.W0, dW):
                ci1 = compute_ci(self.model, batch)
            add("T0", S.target_logp(ci0.logits, batch))
            add("T1", S.target_logp(ci1.logits, batch))
            add("g0", S.masked_logp(self.model, batch, ci0.g))
            add("g1", S.masked_logp(self.model, batch, ci1.g))
            x0, x1 = S.to_flat(ci0.g, self.lay), S.to_flat(ci1.g, self.lay)
            for name, x in self.patterns(x0, x1, batch.flatten(), f, n):
                add(name, S.masked_logp(self.model, batch, S.with_alive(ci0.g, x, self.lay)))
            del ci0, ci1
        n_pos = rows.shape[0] * (rows.shape[1] - 1)
        return {k: v / n_pos for k, v in sums.items()}


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
        m = shares(r.evaluate(held, dW, f))
        m.update(gates(m), n_tokens_kept=f["token"].n_tokens, n_clusters={K: int(c.cent.shape[0]) for K, c in
                                                                           f["clu"].items()})
        out["states"][str(step)] = m
        dump(out)
        log(f"[step {step}] gain {m['gain']:.4f} gap {m['gap']:.4f} r {m['r']:.4f} | "
            + " ".join(f"{k} {v:.4f}" for k, v in m["R"].items()) + f" rand {m['R_rand']:.4f} | C0 {m['C0']} G {m['G']}")
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

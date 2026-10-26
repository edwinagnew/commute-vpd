"""PGD check for the task-local harness, at W0 only (hand-off: reports/analyses/2026-10-19_analysis_task-ladder.md,
"Reported, not scored"; report: reports/reviews/2026-10-19_review_pgd-local.md).

Why both shared adversaries saturate the trigger (KL 15-46 nats at W0) while the same adversary on 128 Pile rows reads
0.69. Every number is `local_faithfulness` (src/commute/local_faith.py) on a window set, so the harness is measured as
it ran; nothing is trained and no model weight is written. Sets:
  steps   emoticon and r0_t588 held-out windows, pgd_steps 0, 2, 5, 10, 20 (20 at seed 0 = ladder.json's `base`)
  seeds   r0_t588 held-out windows, adversary seeds 1, 2 (seed 0 is the steps set)
  scope   emoticon held-out windows, the first 8, 16, 32 (73 = all, in the steps set)
  pile    random 41-token windows from Pile val rows 256..383 (fresh rows), N = 73, two draws; `token` is 0 there,
          so lo_trigger is not read, only the KLs
Writes results/pgd_check.json (results/pgd_check_smoke.json with --smoke) after every call. Read-only.

Usage: python experiments/task_ladder/check_pgd.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"
ROUND1 = HERE.parent / "transport_test" / "results" / "large"

HALF = 20
PILE_ROWS = ("val", 256, 128)  # rows 0..255 are the Pile harness's; these are fresh
STEP_TASKS = ("emoticon", "r0_t588_v320")
SEED_TASK = "r0_t588_v320"
SCOPE_TASK = "emoticon"


@dataclass(frozen=True)
class Sizes:
    """Every size that differs between the full run and the smoke run."""

    steps: tuple[int, ...]  # pgd_steps values of the steps set; the last one is the harness's
    seeds: tuple[int, ...]
    scope_n: tuple[int, ...]
    pile_n: int
    pile_draws: int
    stoch_draws: int
    suffix: str

    def __post_init__(self) -> None:
        assert self.steps == tuple(sorted(self.steps)) and self.scope_n == tuple(sorted(self.scope_n))

    @property
    def pgd_steps(self) -> int:
        return self.steps[-1]


FULL = Sizes(steps=(0, 2, 5, 10, 20), seeds=(1, 2), scope_n=(8, 16, 32), pile_n=73, pile_draws=2, stoch_draws=4,
             suffix="")
SMOKE = Sizes(steps=(0, 2), seeds=(1,), scope_n=(4,), pile_n=8, pile_draws=1, stoch_draws=1, suffix="_smoke")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_windows(name: str) -> tuple[Tensor, int]:
    """Held-out window tokens (N, 41) and the target token of a ladder task, or of the emoticon task (round 1's)."""
    if name == "emoticon":
        w = torch.load(ROUND1 / "windows.pt", weights_only=False)["heldout"]
        token = json.loads((ROUND1.parent / "windows.json").read_text())["o_token_id"]
    else:
        d = torch.load(LARGE / f"windows_{name}.pt", weights_only=False)
        w, token = d["heldout"], d["task"]["target"]
    assert w.half == HALF and w.tokens.shape[1] == 2 * HALF + 1
    return w.tokens, int(token)


def pile_windows(n: int, draw: int) -> Tensor:
    """n random 41-token windows, one per row, from fresh Pile val rows; the centre has >= 20 tokens of context."""
    from commute.data import pile_rows

    rows = pile_rows(*PILE_ROWS, log)
    g = torch.Generator().manual_seed(draw)
    r = torch.randperm(rows.shape[0], generator=g)[:n]
    t = torch.randint(HALF, rows.shape[1] - HALF, (n,), generator=g)
    return torch.stack([rows[i, j - HALF : j + HALF + 1] for i, j in zip(r.tolist(), t.tolist())])


def measure(model: Any, tokens: Tensor, token: int, S: Sizes, seed: int = 0, pgd_steps: int | None = None) -> dict[str, Any]:
    from commute.local_faith import local_faithfulness

    t0 = time.time()
    res = local_faithfulness(model, tokens, HALF, token, n_stoch_draws=S.stoch_draws,
                             pgd_steps=S.pgd_steps if pgd_steps is None else pgd_steps, seed=seed, log=log)
    res["seconds"] = time.time() - t0
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    S = SMOKE if ap.parse_args().smoke else FULL
    from commute.vpd import load_component_model

    model = load_component_model("cuda")
    path = RESULTS / f"pgd_check{S.suffix}.json"
    out: dict[str, Any] = {"sizes": asdict(S), "steps": {}, "seeds": {}, "scope": {}, "pile": {}}

    def save() -> None:
        path.write_text(json.dumps(out, indent=1, sort_keys=True))

    for name in STEP_TASKS:
        tokens, token = load_windows(name)
        out["steps"][name] = {"n_windows": int(tokens.shape[0]), "token": token}
        for k in S.steps:
            log(f"steps: {name} pgd_steps {k}")
            out["steps"][name][str(k)] = measure(model, tokens, token, S, pgd_steps=k)
            save()
    tokens, token = load_windows(SEED_TASK)
    for seed in S.seeds:
        log(f"seeds: {SEED_TASK} seed {seed}")
        out["seeds"][str(seed)] = measure(model, tokens, token, S, seed=seed)
        save()
    tokens, token = load_windows(SCOPE_TASK)
    for n in S.scope_n:
        log(f"scope: {SCOPE_TASK} first {n} windows")
        out["scope"][str(n)] = measure(model, tokens[:n], token, S)
        save()
    for draw in range(S.pile_draws):
        log(f"pile: {S.pile_n} random windows, draw {draw}")
        out["pile"][str(draw)] = measure(model, pile_windows(S.pile_n, draw), 0, S)
        save()
    log(f"done: {path}")


if __name__ == "__main__":
    main()

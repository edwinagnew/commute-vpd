"""E49's filed predictions (reports/plans/2026-10-29_plan_e49-row-blocks.md -> Predictions), written once by
`run.py --predictions` to results/predictions.json with the run's config; never edited after."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

NOISE = ("W0's own spread over adversary seeds 0-2 per block on the L40 (max/min - 1): 0.087 on val rows 0-127 and "
         "0.46 on rows 128-255 (E47 logs); over 256 rows 0.209 (rows 0-255, E47) against E39's H100 0.075, for which "
         "E39's lines 1.40 / 1.61 were set; the same saved file on the same rows reads 0.5-4.2% apart between L40 and "
         "H100 (four pairs, E47 against E39 A2/A3); E37's batch-order spread of the check, 14%")


def pred(pid: str, rule: str, expected: str, lean: str, blind: str, control: str) -> dict[str, str]:
    return {"id": pid, "rule": rule, "expected": expected, "lean": lean, "blind_to": blind, "positive_control": control,
            "noise_floor": NOISE}


def predictions(cfg: dict[str, Any], readings: dict[str, Any]) -> dict[str, Any]:
    p, f = cfg["pass_line"], cfg["fail_line"]
    count = (f"per new block (val 512-639, 640-767, 768-895, 896-1023) the median over adversary seeds 0-2 of the "
             f"state's ratio; a block fails when that median is >= {f} (between the lines is not a fail); the count is "
             "scored only when all four blocks have three finite ratios (every b job finished)")
    return {
        "plan": "reports/plans/2026-10-29_plan_e49-row-blocks.md (E49). Filed before any run; never edited after.",
        "definitions": {
            "ratio": "a state's PGD KL on one block of 128 Pile val rows (the harness adversary: 20 sign steps of 0.1 "
                     "from the random start of torch.manual_seed(seed), one adversary shared by the block's 16 batches of "
                     "8) over W0's KL on the same block at the same seed, read in the same job on the L40",
            "pair check": "E39's 256-row check on two neighbouring blocks: the mean of their two KLs over W0's mean",
            "lines": f"E39's, fixed: pass <= {p}, fail >= {f}, both inclusive; between is neither",
            "hard block": "val rows 128-255, where both SGD teacher-trained decompositions' fails sat (E39 A3: 2.93, 2.95)",
            "states": "teacher-trained decompositions (E37's teacher training on each fine-tune, loaded from E39's saved "
                      "bf16 offsets): tt_adam (E37's Adam fine-tune), tt_sgd (A1's plain SGD 0.01), tt_sgd_r8 (A2's SGD "
                      "held to rank 8, 0.03), tt_lora (E39's rank-8 LoRA, A2's seed-0 copy); do-nothing states: the Adam, "
                      "plain SGD and LoRA fine-tunes regenerated here, their delta-W left in Delta",
        },
        "controls": {
            "regen": f"each fine-tune's held-out gain within {cfg['gain_tol']} of {cfg['gain_ref']} (H100), else stop",
            "a: W0": f"PGD KL per block within {cfg['w0_block_tol']} of E47's L40 readings {cfg['a_w0']}, else stop",
            "a: Adam teacher-trained": f"mean KL on rows 0-255 at seed 0 within {cfg['repro_tol']:.0%} of E47's L40 "
                                       f"{cfg['a_pair_kl']['tt_adam']} and the check >= {f} (positive control), else stop",
            "a: LoRA teacher-trained": f"mean KL on rows 0-255 at seed 0 within {cfg['repro_tol']:.0%} of E47's L40 "
                                       f"{cfg['a_pair_kl']['tt_lora']}, else stop",
            "a: SGD on this card (reported, no stop)": f"check on rows 0-255 at seed 0 within {cfg['card_tol']:.0%} of the "
                                                       f"H100's {cfg['h100_first']}",
            "b: W0": f"per block PGD KL in {cfg['w0_pgd_band']} and CI-masked KL within {cfg['w0_ci_tol']:.0%} of "
                     f"{cfg['w0_ci']}, else stop",
            "b: Adam do-nothing": f"pair check >= {f} at the job's seed (the check can fail an update on these rows), "
                                  "else stop",
        },
        "expected": {
            "a: W0": f"{cfg['a_w0']} exactly (rows|seed)",
            "a: Adam teacher-trained, rows 0-255 seed 0": "2.171 (E47 L40); per block about 1.7 and 2.7 (H100 1.69, 2.68)",
            "a: LoRA teacher-trained": "rows 0-255 seed 0 1.326 (E47 L40); hard block 1.3-1.6 at seed 0 (H100 1.42), "
                                       "1.1-1.5 at seeds 1-2",
            "a: SGD teacher-trained, rows 0-255 seed 0": "plain 1.86-2.27, rank 8 1.91-2.33 (H100 2.0675, 2.1218 +- 10%)",
            "a: SGD teacher-trained, hard block": "seed 0 2.4-3.2 (H100 2.93, 2.95); seeds 1-2 1.8-3.0 under 'rows', "
                                                  f"<= {p} under 'start'",
            "b: W0": "PGD KL 0.6-1.0 per block, CI-masked 0.33-0.35; own spread over seeds 0.05-0.46",
            "b: do-nothing": "Adam 3-6 (E37 4.87 on rows 0-255); plain SGD 1.3-3.8, may pass a block (A1 blocks 1.34, "
                             "3.83); LoRA 2-3.5 (E39 2.65)",
            "b: teacher-trained, per block median": "Adam 1.7-2.7 (H100 blocks 1.69-2.68), fails 4 of 4; plain SGD "
                                                    "1.1-1.45 with at most one block near 2-3, fails 0-1; SGD rank 8 "
                                                    "1.25-1.5, fails 0-1; LoRA 1.1-1.45, fails 0",
        },
        "predictions": [
            pred("P1", f"(a) on the hard block, each SGD teacher-trained decomposition (plain, rank 8) at adversary seeds 1 "
                       f"and 2: 'rows' if both ratios >= {f}, 'start' if both <= {p}, else 'mixed'. Holds iff both states "
                       "read 'rows'; misses if both read 'start' or they read 'mixed'/differ; unscored if stage a did "
                       "not finish or a ratio is not finite",
                 "rows: 1.8-3.0 at seeds 1 and 2", "low-medium: the block reads high for 13 of 14 states over 1.40 and for "
                 "the SGD do-nothing (3.83), but W0's own reading there varies most over seeds (0.46 on the L40)",
                 "one block, three starts, one card; a W0 reading high at a seed (1.0007 at seed 1) lowers that seed's "
                 "ratio for every state", "the Adam teacher-trained decomposition at seed 0 on rows 0-255 (must fail)"),
            pred("P2", f"(b) the Adam teacher-trained decomposition fails 4 of the 4 new blocks. {count}",
                 "4 of 4, medians 1.7-2.7", "medium: its four known blocks read 1.69-2.68 on the H100, the lowest close "
                 "to the line", "one batch order of teacher training; Adam's check moved 14% with batch order (E37)",
                 "the Adam do-nothing on the same rows and seed (pair check >= 1.61, a stop)"),
            pred("P3", f"(b) the plain SGD teacher-trained decomposition fails 0 or 1 of the 4 new blocks. {count}",
                 "0-1 of 4, medians 1.1-1.45", "medium-low: three of its four known blocks read 1.11-1.26, one 2.93",
                 "the check's power on SGD's update: the SGD do-nothing passed rows 0-127 (1.34); its own failing-block "
                 "count is read beside", "the SGD do-nothing on the same blocks and seeds"),
            pred("P4", f"(b) the LoRA teacher-trained decomposition fails none of the 4 new blocks. {count}",
                 "0 of 4, medians 1.1-1.45", "medium: its four known blocks read 1.11-1.42 (H100)",
                 "one LoRA seed; the hard block read 1.42-1.44, just over the pass line",
                 "the LoRA do-nothing on the same blocks and seeds"),
            pred("P5", f"(b) the SGD held to rank 8 teacher-trained decomposition fails 0 or 1 of the 4 new blocks. {count}",
                 "0-1 of 4, medians 1.25-1.5", "medium-low: its known blocks read 1.25, 1.36, 1.42 besides the hard "
                 "block's 2.95; a median between the lines does not count as a fail",
                 "one rate, one batch order; it has no do-nothing read here", "the plain SGD do-nothing"),
        ],
        "stop_rules": "each job stops on its first missed control (above); a stopped job's reads are not scored",
        "readings": readings,
    }


def file_predictions(path: Path, cfg: dict[str, Any], readings: dict[str, Any]) -> None:
    if path.is_file():
        sys.exit(f"{path} exists; filed predictions are never edited (add a dated report instead)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**predictions(cfg, readings), "config": cfg, "filed": time.strftime("%Y-%m-%d %H:%M:%S")},
                               indent=1))
    print(f"filed {path}")

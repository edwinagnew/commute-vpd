"""E69's files from W&B into the shared results folder, each checked against the run's committed SHA256SUMS. A cluster CPU
job (RUN_BRIEF.md); needs WANDB_API_KEY. `--only dW/R/` fetches the files whose path starts so (the rehearsal: `rollouts/`).
`--lr-tag 3e-6` fetches that rate's own run. `--root DIR` puts the files under DIR instead of the cluster's folder: on a
new pod after a loss, `--root experiments/qwen_rl/results/large --only rollouts/H.jsonl` (RUN_BRIEF.md, "Pod lost").
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from commute import qwen_files
from commute.qwen_rl import DEFAULT_TAG, RATES, artifact_name, rate_names
from commute.qwen_sft import sha256

RESULTS = Path(__file__).resolve().parent / "results"


def matches(name: str, h: str) -> Callable[[Path], bool]:
    return lambda d: (d / name).is_file() and sha256(d / name) == h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="full", choices=("full", "tiny"))
    ap.add_argument("--lr-tag", default=DEFAULT_TAG, choices=tuple(RATES))
    ap.add_argument("--only", default="")
    ap.add_argument("--root", type=Path, default=Path.home() / "commute-data" / "results-large" / "qwen_rl")
    a = ap.parse_args()
    shared, family = ("", "qwen_rl") if a.size == "full" else ("tiny", "qwen_rl_tiny")
    out, prefix = rate_names(shared, family, a.lr_tag)
    root = a.root / out
    pairs = [line.split(maxsplit=1) for line in (RESULTS / out / "SHA256SUMS").read_text().splitlines()]
    for h, name in pairs:
        rel = Path(name.strip())
        if name.strip().startswith(a.only):
            qwen_files.fetch(root / rel.parent, artifact=artifact_name(prefix, rel), kind=prefix, ok=matches(rel.name, h))
    print(f"[fetch] done: {root}", flush=True)


if __name__ == "__main__":
    main()

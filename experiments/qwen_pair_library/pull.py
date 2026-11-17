"""Cell P3.5, run 2: every saved library state on W&B (artifact type "p35", names p35-<arm>-s<seed>-<labels>, logged by
library.py --upload on the pods) into results/large/library/<arm>_s<seed>/<labels>.safetensors, with sha256 sums.

A file already there is kept (`qwen_files.fetch` moves each file into place whole). Each file's sha256 is written to
results/library/SHA256SUMS and compared with the one library.py recorded on the pod (results/library/<arm>_s<seed>.json
.saves[].sha256, when that file is home); results/library/pull.json lists every artifact, its sums and the verdict.
Exit 0 when every recorded sum matches, 1 otherwise. A cluster CPU job (cluster/qwen_pair_library_pull.sbatch), since
~270 GB outlasts a session; needs WANDB_API_KEY.

  python experiments/qwen_pair_library/pull.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from commute import qwen_files
from commute.jobs import hard_exit
from commute.qwen_files import ENTITY, PROJECT
from commute.qwen_sft import sha256

HERE = Path(__file__).resolve().parent
OUT, LARGE = HERE / "results" / "library", HERE / "results" / "large" / "library"
PREFIX = "p35"
NAME = re.compile(rf"{PREFIX}-(?P<arm>[a-z_]+)-s(?P<seed>\d+)-(?P<labels>[0-9a-z_]+)")


def recorded(tag: str, labels: str) -> str | None:
    """The sha256 library.py wrote on the pod for this state, if the arm's JSON is home."""
    f = OUT / f"{tag}.json"
    saves = json.loads(f.read_text()).get("saves", []) if f.is_file() else []
    return next((s["sha256"] for s in saves if "_".join(s["labels"]) == labels), None)


def pull(name: str) -> dict[str, Any]:
    m = NAME.fullmatch(name)
    assert m is not None, name
    tag, file = f"{m['arm']}_s{m['seed']}", f"{m['labels']}.safetensors"
    dest = LARGE / tag
    qwen_files.fetch(dest, artifact=name, kind=PREFIX, ok=lambda d: (d / file).is_file())
    got, pod = sha256(dest / file), recorded(tag, m["labels"])
    row = {"artifact": name, "file": f"{tag}/{file}", "sha256": got, "pod_sha256": pod,
           "matches": None if pod is None else got == pod}
    print(f"[pull] {json.dumps(row)}", flush=True)
    return row


def main() -> None:
    import wandb

    names = sorted(c.name for c in wandb.Api().artifact_collections(f"{ENTITY}/{PROJECT}", PREFIX))
    skipped = [n for n in names if not NAME.fullmatch(n)]
    rows = [pull(n) for n in names if NAME.fullmatch(n)]
    OUT.mkdir(exist_ok=True)
    (OUT / "SHA256SUMS").write_text("".join(f"{r['sha256']}  library/{r['file']}\n" for r in rows))
    bad = [r["file"] for r in rows if r["matches"] is False]
    unchecked = [r["file"] for r in rows if r["matches"] is None]
    (OUT / "pull.json").write_text(json.dumps({"rows": rows, "skipped_names": skipped, "mismatched": bad,
                                               "no_pod_sum": unchecked}, indent=1))
    print(f"[pull] {len(rows)} files; mismatched {bad}; no pod sum yet {unchecked}", flush=True)
    hard_exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(__doc__)
    main()

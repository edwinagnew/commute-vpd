"""E84 on a pod (the cluster's free card is held outside Slurm, job 51561, 51562): W&B carries the files both ways. All need WANDB_API_KEY.

  send      on the dev side: artifacts/tofu and results/large/answers/W0 as one artifact (a cluster CPU job or cpu-run).
  receive   on the pod, before run.py: unpacks them.
  up        on the pod, beside run.py: each dW file as it is written, then answers.tar and results/SHA256SUMS when stage G is done.
  verify    on the pod, before terminate: every row of SHA256SUMS is a committed artifact with the same size and md5.
  down      on the cluster, after the pod: the files into results/large, each checked against the committed SHA256SUMS.
With --amend1, up, verify and down act on Amendment 1's files (send and receive are the same inputs either way).
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

from commute import qwen_files
from commute.qwen_files import ENTITY, PROJECT
from commute.qwen_rl import artifact_name, upload_ok
from commute.qwen_sft import Uploader, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
DW, ANSWERS = LARGE / "dW" / "a2_2e-5", LARGE / "answers"
SUMS, READ, STOPPED = RESULTS / "SHA256SUMS", RESULTS / "read.json", RESULTS / "stopped.json"
PREFIX = INPUTS_PREFIX = "qwen_tofu_forms"
INPUTS, TAR = LARGE / "pod_inputs" / "inputs.tar", LARGE / "answers_tar" / "answers.tar"
FILES = 12


def amend1() -> None:
    """Amendment 1's own outputs, so E84's files and sums stay as they are."""
    global DW, ANSWERS, SUMS, READ, STOPPED, PREFIX, TAR, FILES
    DW, ANSWERS = LARGE / "dW" / "a2_2e-5_e10", LARGE / "answers_e10"
    SUMS, READ, STOPPED = RESULTS / "SHA256SUMS_amend1", RESULTS / "amend1_read.json", RESULTS / "stopped_amend1.json"
    PREFIX, TAR, FILES = "qwen_tofu_forms_e10", LARGE / "answers_e10_tar" / "answers.tar", 14
POLL_SECONDS = 20


def send() -> None:
    INPUTS.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(INPUTS, "w") as t:
        t.add(ROOT / "artifacts" / "tofu", "artifacts/tofu")
        t.add(ANSWERS / "W0", "answers/W0")
    up = Uploader(INPUTS_PREFIX, None, LARGE / "wandb")
    up.add(INPUTS)
    up.finish()


def receive() -> None:
    qwen_files.fetch(INPUTS.parent, artifact=artifact_name(INPUTS_PREFIX, INPUTS.relative_to(LARGE)), kind=INPUTS_PREFIX,
                     ok=lambda d: (d / INPUTS.name).is_file())
    with tarfile.open(INPUTS) as t:
        t.extractall(ROOT, filter="data", members=[m for m in t.getmembers() if m.name.startswith("artifacts/")])
    with tarfile.open(INPUTS) as t:
        t.extractall(LARGE, filter="data", members=[m for m in t.getmembers() if m.name.startswith("answers/")])
    print(f"[receive] {len(list((ROOT / 'artifacts' / 'tofu').iterdir()))} TOFU files; W0: {sorted(p.name for p in (ANSWERS / 'W0').iterdir())}", flush=True)


def stage_g_done() -> bool:
    return READ.is_file() and bool(json.loads(READ.read_text()).get("done"))


def up() -> None:
    uploader, sent = Uploader(PREFIX, None, LARGE / "wandb"), set[Path]()
    while True:
        finished = stage_g_done()  # read first: a file written after this is caught on the next pass or by the sweep below
        for p in sorted(DW.glob("step_*.safetensors")):  # save() renames a whole file into place
            if p not in sent:
                uploader.add(p)
                sent.add(p)
        if finished:
            break
        if STOPPED.is_file():
            uploader.finish()
            sys.exit(f"{STOPPED} exists: uploads stop, {len(sent)} files sent")
        time.sleep(POLL_SECONDS)
    TAR.parent.mkdir(exist_ok=True)
    with tarfile.open(TAR, "w") as t:
        t.add(ANSWERS, ANSWERS.name)
    uploader.add(TAR)
    rows = [f"{sha256(p)}  {p.relative_to(LARGE)}\n" for p in [*sorted(sent), TAR]]
    SUMS.write_text("".join(rows))
    uploader.finish()
    print(f"[up] done, {len(rows)} files", flush=True)


def rows() -> list[tuple[str, Path]]:
    pairs = [line.split(maxsplit=1) for line in SUMS.read_text().splitlines()]
    assert len(pairs) == FILES + 1, f"{SUMS}: {len(pairs)} rows, expected {FILES + 1}"
    return [(h, Path(n.strip())) for h, n in pairs]


def verify() -> None:
    import wandb

    api, missed = wandb.Api(), []
    for _, rel in rows():
        art = api.artifact(f"{ENTITY}/{PROJECT}/{artifact_name(PREFIX, rel)}:latest")
        entry = art.manifest.entries[rel.name]
        good = art.state == "COMMITTED" and upload_ok(entry.size, entry.digest, LARGE / rel)
        print(f"[verify] {rel}: {'ok' if good else 'MISSED'}", flush=True)
        missed += [] if good else [str(rel)]
    if missed:
        sys.exit(f"missed {missed}")
    print("[verify] all home on W&B", flush=True)


def down() -> None:
    for h, rel in rows():
        def ok(d: Path, name: str = rel.name, want: str = h) -> bool:
            return (d / name).is_file() and sha256(d / name) == want

        qwen_files.fetch(LARGE / rel.parent, artifact=artifact_name(PREFIX, rel), kind=PREFIX, ok=ok)
        assert ok(LARGE / rel.parent), f"{rel}: sha256 differs after the fetch"
    with tarfile.open(TAR) as t:
        t.extractall(LARGE, filter="data")
    print("[down] done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("send", "receive", "up", "verify", "down"))
    p.add_argument("--amend1", action="store_true")
    a = p.parse_args()
    if a.amend1:
        amend1()
    {"send": send, "receive": receive, "up": up, "verify": verify, "down": down}[a.cmd]()

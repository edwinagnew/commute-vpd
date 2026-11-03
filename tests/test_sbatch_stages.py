"""Every stage a stage-taking sbatch documents gets past its own argument check, run in bash with a stub clone and a
stub python (friction 228: three new jobs would have exited 2 on a stage missing from the check's case block)."""

from __future__ import annotations

import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "cluster"
STAGED = sorted(f for f in CLUSTER.glob("*.sbatch") if re.search(r"^\s*case ", f.read_text(), re.MULTILINE))


def documented(f: Path) -> list[list[str]]:
    """The arguments of each `sbatch ... cluster/<f> ARGS` line in the file's leading comment. A line with a
    placeholder (<x>) is skipped, and an option after the stage is dropped."""
    out = []
    for line in f.read_text().splitlines():
        if not line.startswith("#"):
            break
        m = re.search(rf"\bsbatch\b.*cluster/{re.escape(f.name)}(.*)", line)
        if m:
            args = shlex.split(m[1].split("#")[0])
            kept = args[: next((i for i, a in enumerate(args) if a.startswith("-")), len(args))]
            if not any(c in a for a in kept for c in "<>{}["):
                out.append(kept)
    return out


# Some files ask for their predictions before the stage: in the stub clone `test -f` holds, and the stage check is reached.
TEST_F = 'test() { [[ $1 == -f ]] || builtin test "$@"; }\n'


def bash(f: Path, args: list[str], tmp: Path) -> subprocess.CompletedProcess[str]:
    """The sbatch file run in a throwaway clone: env.sh only sets COMMUTE_REPO, python and the job tools (git among them) succeed."""
    stub = tmp / "stub"
    for tool in (tmp / ".venv/bin/python", stub / "nvidia-smi", stub / "scontrol", stub / "git"):
        tool.parent.mkdir(parents=True, exist_ok=True)
        tool.write_text("#!/bin/bash\nexit 0\n")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    (tmp / "cluster/logs").mkdir(parents=True, exist_ok=True)
    (tmp / "cluster/env.sh").write_text(f"export COMMUTE_REPO={shlex.quote(str(tmp))}\nexport PYTHONPATH=\n" + TEST_F)
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(tmp), "HOME": str(tmp), "PATH": f"{stub}:{os.environ['PATH']}", "SLURM_JOB_ID": "1",
           "SLURM_JOB_NAME": "stage-check", "SLURM_JOB_QOS": "debug", "SLURM_CPUS_PER_TASK": "1"}  # fmt: skip
    return subprocess.run(["bash", str(f), *args], check=False, cwd=tmp, env=env, capture_output=True, text=True, timeout=30)


CASES = [(f, args) for f in STAGED for args in documented(f)]


def test_stage_files_are_found():
    assert len(STAGED) >= 8 and all(documented(f) for f in STAGED), [f.name for f in STAGED if not documented(f)]


@pytest.mark.parametrize(("f", "args"), CASES, ids=[f"{f.stem}:{' '.join(a)}" for f, a in CASES])
def test_a_documented_stage_passes_the_check(f: Path, args: list[str], tmp_path: Path):
    r = bash(f, args, tmp_path)
    assert r.returncode != 2 and "usage" not in r.stderr + r.stdout, r.stderr + r.stdout


@pytest.mark.parametrize("f", STAGED, ids=[f.stem for f in STAGED])
def test_an_undocumented_stage_is_refused(f: Path, tmp_path: Path):
    r = bash(f, ["no-such-stage"], tmp_path)
    assert r.returncode == 2 or "usage" in r.stderr + r.stdout, r.stderr + r.stdout

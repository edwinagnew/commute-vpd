"""The Bash hook .claude/hooks/no-compute-on-node.py, run as Claude Code runs it: its JSON on standard input, exit 2 for a
refusal. Both sides of its line: what must stay refused (Python on the dev node, a model on a CPU job) and the read-only
commands it refused by mistake (friction 88, 307, 310, 342, 352, 367)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "no-compute-on-node.py"
CPU_RUN = "~/commute-agents/fleet/cpu-run"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small repository: run.py and checks.py load the model, read.py and x.py do not, job.sh runs Python."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "run.py").write_text("from commute import vpd\nmodel = vpd.load_component_model()\n")
    (tmp_path / "exp" / "checks.py").write_text("from commute import vpd\nmodel = vpd.load_component_model('cpu')\n")
    (tmp_path / "exp" / "read.py").write_text("from commute import vpd\nW, uv = vpd.checkpoint_weights()\n")
    (tmp_path / "x.py").write_text("print(1)\n")
    (tmp_path / "y.py").write_text("print(2)\n")
    (tmp_path / "job.sh").write_text("#!/bin/bash\npython3 x.py\n")
    return tmp_path


def hook(command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}  # inside a job the hook lets everything run
    call = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(cwd)}
    return subprocess.run([sys.executable, str(HOOK)], input=json.dumps(call), capture_output=True, text=True, env=env, check=False)


REFUSED = [
    # Past a blank here-document: its operator is not the program, and a later redirection wins standard input
    "<<'EOF' python3 x.py\nEOF",
    "FOO=1 <<'EOF' python3 x.py\nEOF",
    "timeout 60 <<'EOF' python3 x.py\nEOF",
    "python3 <<'EOF' <x.py\nEOF",
    # A << that is not a here-document hides nothing after it
    "git commit -m 'a << b'\npython3 x.py",
    "echo $((1<<2))\npython3 x.py",
    # Python, by every spelling of the program
    "python x.py",
    "python3 x.py",
    "python3 -",
    "python3 - 2>/dev/null",
    "python3 -c 'print(1)'",
    "python3 -m pytest",
    "python3 -I -c pass",
    ".venv/bin/python x.py",
    "/usr/bin/python3.13 x.py",
    "ipython",
    "pytest tests/test_x.py",
    "uv run pytest",
    "make test",
    "make lint FILES=x.py",
    "./x.py",
    '"python3" x.py',
    "pyth\\on3 x.py",
    # Past variables and wrappers
    "FOO=1 python x.py",
    "FOO=1 BAR=2 python x.py",
    "env FOO=1 python x.py",
    "timeout 60 python x.py",
    "nice -n 19 python3 x.py",
    "nohup python x.py",
    "exec python x.py",
    # Compound commands
    "cd exp && python y.py",
    "cd exp; python y.py",
    "ls || python x.py",
    "ls\npython x.py",
    "ls & python x.py",
    "(python x.py)",
    "{ python x.py; }",
    "if true; then python x.py; fi",
    "for f in a b; do python3 x.py; done",
    "f=x.py; python3 \"$f\"",
    # Pipes and substitutions
    "cat x.py | python",
    "cat x.py | python3 -",
    "cat x.py | .venv/bin/python",
    "ls | xargs python3",
    "find . -name '*.py' -exec python3 {} \\;",
    "echo $(python3 x.py)",
    'echo "$(python3 x.py)"',
    "echo `python3 x.py`",
    "A=$(python3 -c 'print(1)'); echo $A",
    "diff <(python3 x.py) y.py",
    # Through a shell, a script
    "bash -c 'python x.py'",
    "sh -c 'cd exp && python3 y.py'",
    "bash -lc 'FOO=1 python x.py'",
    "bash job.sh",
    "watch 'python3 x.py'",
    # Standard input with a program in it
    "python3 - <<'EOF'\nprint(1)\nEOF",
    "python3 <<EOF\nprint(1)\nEOF",
    'python3 - "$f" <<\'EOF\'\nimport sys\nEOF',
    ".venv/bin/python - <<'EOF'\nprint(1)\nEOF",
    "python3 - <<< 'print(1)'",
    "cat > a.txt <<'EOF'\nnot a program\nEOF\npython3 x.py",
    # A blank here-document does not excuse a script or inline code
    "python3 x.py <<'EOF'\nEOF",
    "python3 -c 'print(1)' <<'EOF'\nEOF",
    "python3 - <<'A' <<'B'\nA\nprint(1)\nB",
    # Quotes that do not hide an operator from the shell
    "grep 'a' x.py; python3 x.py",
    "grep -c '' x.py; python3 x.py",
    "grep ';' x.py && python3 x.py",
    "git commit -m 'msg'; python x.py",
]

ALLOWED = [
    # A here-document's lines are not commands, in a substitution inside double quotes too
    "cat > j.sh <<'EOF'\nsrun python exp/run.py\nEOF",
    "git commit -m \"$(cat <<'EOF'\npython3 x.py\nit's a message\nEOF\n)\"",
    "cat > j.sh <<'EOF' && echo done\nsrun python exp/run.py\nEOF",
    # A job fed its script that asks for a GPU may load a model
    "sbatch <<'EOF'\n#!/bin/bash\n#SBATCH --gres=gpu:1\nsrun python exp/run.py\nEOF",
    "sbatch --gres=gpu:1 <<'EOF'\nsrun python exp/run.py\nEOF",
    # A read naming a script is not a run (friction 307, 352)
    "grep -n def x.py",
    "grep -c '' x.py",
    'grep -c "" x.py',
    "command grep -n '' x.py | sed -n 1,45p | cut -c1-170",
    "grep -n ';' x.py",
    "grep -n '(' x.py",
    "grep -c '|' x.py",
    "grep -c '&' x.py",
    "grep -n '>' x.py",
    "grep -n $'\\t' x.py",
    "sed -n 1,5p exp/run.py",
    "sed -n '/;/p' exp/run.py",
    "cat exp/run.py",
    "head -3 exp/checks.py; wc -l exp/checks.py",
    "awk 'length>120' exp/checks.py | head",
    "awk 'NF==0' exp/checks.py | wc -l",
    "cut -d';' -f1 exp/run.py",
    "grep -n '^def \\|^class ' exp/checks.py; grep -c '' exp/checks.py",
    "cd exp && grep -c '' run.py",
    "ls exp/run.py",
    "git diff -- exp/run.py",
    "git log --oneline -3 -- exp/run.py",
    "git show HEAD:exp/run.py | head",
    "command -v python3",
    "which python3",
    "grep python3 x.py",
    "echo python3 x.py",
    "git commit -m 'run python x.py; then python3 y.py'",
    "ruff check x.py",
    # A heredoc that only mentions Python (friction 310)
    "cat > notes.md <<'EOF'\npython3 x.py\npython3 - <<'INNER'\nEOF",
    "cat > a.sh <<END-OF-IT\npython3 x.py\nEND-OF-IT",
    "cat >a.txt <<\\EOF\npython3 x.py\nEOF",
    "cat > notes.md <<'EOF'\n~/commute-agents/fleet/cpu-run python exp/run.py\nEOF",
    # An empty one has no program to run (friction 310)
    "python3 - <<'EOF' 2>/dev/null || true\nEOF",
    "python3 - <<'EOF'\n\nEOF\ngrep -n def x.py",
    'python3 - "$f" <<\'EOF\' 2>/dev/null || echo no\nEOF',
    ".venv/bin/python - <<'EOF'\nEOF",
    # Single-quoted jq with $ (friction 342), shell variables in reads (friction 367)
    "jq '.a as $g | $g | keys' x.json",
    "jq -c '.groups | map(.got as $g | .ref as $r | $g - $r)' x.json",
    "jq --arg k v '.[$k]' x.json",
    'R=x.json; jq -c keys "$R"',
    "R=x.json; jq -c keys $R | cut -c1-30",
    "f=exp/run.py; sed -n 1,5p $f; grep -c '' \"$f\"",
    "P=exp; sed -n 1,5p $P/run.py; command grep -n def $P/checks.py",
    "for f in run checks; do grep -c '' exp/$f.py; done",
    # The harness's own tools run here; cpu-run sends Python to the worker
    f"{CPU_RUN} python3 x.py",
    f"{CPU_RUN} --mem 4G .venv/bin/python x.py",
    f"cd exp && {CPU_RUN} python3 read.py",
    f"{CPU_RUN} python3 - <<'EOF'\nprint(1)\nEOF",
    f"FOO=1 {CPU_RUN} python3 x.py",
    "sbatch --wrap='python3 x.py'",
    # Model-free work on the worker (friction 173, 344)
    f"{CPU_RUN} --mem 8G python exp/read.py",
    f"{CPU_RUN} make lint FILES=exp/run.py",
    f"{CPU_RUN} .venv/bin/pyright --pythonpath .venv/bin/python exp/run.py",
    f"{CPU_RUN} grep -c '' x.py",
]

MODEL_ON_CPU = [
    # A here-document that is the job's script
    "sbatch <<'EOF'\n#!/bin/bash\nsrun python exp/run.py\nEOF",
    "sbatch <<'EOF'\n#!/bin/bash\nsrun python x.py --model Qwen/Qwen3-4B\nEOF",
    f"{CPU_RUN} bash <<'EOF'\nsrun python exp/run.py\nEOF",
    # A << that is not a here-document hides nothing after it
    f"git commit -m 'a << b'\n{CPU_RUN} python exp/run.py",
    f"grep -n '<<EOF' x.py\n{CPU_RUN} python exp/run.py",
    f"echo $((1<<2))\n{CPU_RUN} python exp/run.py",
    f"{CPU_RUN} python exp/run.py",
    f"{CPU_RUN} --mem 16G python exp/checks.py",
    f"cd exp && {CPU_RUN} python run.py",
    f"{CPU_RUN} python3 - <<'EOF'\nfrom commute import vpd\nm = vpd.load_component_model('cpu')\nEOF",
    f"{CPU_RUN} python3 - <<'EOF'\nfrom transformers import AutoModel\nm = AutoModel.from_pretrained('x')\nEOF",
    f"{CPU_RUN} python x.py --model Qwen/Qwen3-4B",
    "sbatch --wrap='python exp/run.py'",
    f"{CPU_RUN} make lint FILES=exp/run.py; {CPU_RUN} python exp/run.py",
    f"{CPU_RUN} .venv/bin/pyright --pythonpath .venv/bin/python exp/run.py && {CPU_RUN} python exp/run.py",
]


@pytest.mark.parametrize("command", REFUSED)
def test_refused(command: str, repo: Path) -> None:
    done = hook(command, repo)
    assert done.returncode == 2 and "No Python runs directly" in done.stderr, done.stderr


@pytest.mark.parametrize("command", ALLOWED)
def test_allowed(command: str, repo: Path) -> None:
    done = hook(command, repo)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("command", MODEL_ON_CPU)
def test_model_on_cpu_refused(command: str, repo: Path) -> None:
    done = hook(command, repo)
    assert done.returncode == 2 and "Models never run on CPUs" in done.stderr, done.stderr
    assert "vpd.checkpoint_weights()" in done.stderr  # the way out for a script that only reads weights


def test_gpu_job_may_load_a_model(repo: Path) -> None:
    assert hook("sbatch --gres=gpu:1 --wrap='python exp/run.py'", repo).returncode == 0


def test_inside_a_job_nothing_is_refused(repo: Path) -> None:
    call = {"tool_name": "Bash", "tool_input": {"command": "python3 x.py"}, "cwd": str(repo)}
    done = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(call), capture_output=True, text=True,
                          env={**os.environ, "SLURM_JOB_ID": "1"}, check=False)
    assert done.returncode == 0, done.stderr

"""check.py of the writeup skill: retired-term exemptions and head names (friction 309, 356)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("check", Path(".claude/skills/writeup/check.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def run(tmp_path, text):
    f = tmp_path / "x.md"
    f.write_text(text)
    return check.check(f)


def test_up_down_projection_exempt(tmp_path):
    errors, _, _ = run(tmp_path, "# T\n\nThe up-projection and down projection matrices stay put.\n")
    assert errors == []


def test_plain_projection_error_with_hint(tmp_path):
    errors, _, _ = run(tmp_path, "# T\n\nThe projection of the update is small.\n")
    assert len(errors) == 1 and "component along" in errors[0]


def test_head_name_not_flagged(tmp_path):
    _, flags, _ = run(tmp_path, "# T\n\nHead (2,4) reads 0.5 and 0.6 here.\n")
    assert flags == []


def test_method_name_exempt(tmp_path):
    errors, _, _ = run(tmp_path, "# T\n\nIterative nullspace projection (INLP) removes the direction.\n")
    assert errors == []


def test_exit_status_follows_errors(tmp_path):
    import subprocess
    import sys

    script = Path(".claude/skills/writeup/check.py")
    clean, bad = tmp_path / "c.md", tmp_path / "b.md"
    clean.write_text("# T\n\nA short clean sentence.\n")
    bad.write_text("# T\n\nThe projection of the update is small.\n")
    r = subprocess.run([sys.executable, str(script), "--quiet", str(clean)], check=False, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "check.py: 0 errors"
    r = subprocess.run([sys.executable, str(script), str(bad)], check=False, capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout.startswith("ERROR")

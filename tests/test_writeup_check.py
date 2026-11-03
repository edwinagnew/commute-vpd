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

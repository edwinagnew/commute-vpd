"""commute.figures.save refuses a dashed reference line a reader cannot name (friction 683)."""

from __future__ import annotations

from pathlib import Path

import pytest

from commute import figures as FG


def _panel():
    fig, ax = FG.figure()
    ax.plot([0, 1, 2], [0.2, 0.5, 0.9], label="share")
    ax.set_title("t")
    ax.set_xlabel("x")
    return fig, ax


def _save(fig, tmp_path: Path) -> Path:
    return FG.save(fig, tmp_path / "f.png", n="1 row", data=[{"x": 0}])


def test_an_unnamed_dashed_line_is_refused(tmp_path: Path):
    fig, ax = _panel()
    ax.axhline(0.8, linestyle="--")
    ax.axvline(1.5, linestyle=":")
    with pytest.raises(ValueError, match=r"axhline\(0.8\), axvline\(1.5\)"):
        _save(fig, tmp_path)


def test_a_line_in_a_legend_passes(tmp_path: Path):
    fig, ax = _panel()
    ax.axhline(0.8, linestyle="--", label="cut-off")
    ax.legend()
    assert _save(fig, tmp_path).exists()


def test_a_label_without_a_legend_is_refused(tmp_path: Path):
    fig, ax = _panel()
    ax.axhline(0.8, linestyle="--", label="cut-off")
    with pytest.raises(ValueError, match="no legend entry"):
        _save(fig, tmp_path)


def test_a_line_labelled_beside_it_passes(tmp_path: Path):
    fig, ax = _panel()
    ax.axhline(0.8, linestyle="--")
    ax.text(0.0, 0.8, " cut-off", transform=ax.get_yaxis_transform())
    FG.reference(ax, 1.5, "reached")
    assert _save(fig, tmp_path).exists()


def test_separators_and_the_zero_line_pass(tmp_path: Path):
    fig, ax = _panel()
    ax.axhline(0.5, color="#999999", linewidth=0.8)
    ax.axhline(0, linestyle="--")
    assert _save(fig, tmp_path).exists()

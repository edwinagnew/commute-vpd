from typing import Any

import pytest

from commute import faithfulness

KL = {"h0": 0.7, "h1": 0.8, "budget": 2.0}
W0 = {"h0": 0.7, "h1": 0.4, "budget": 1.0}  # ratios 1, 2, 2


@pytest.fixture(autouse=True)
def fake_reading(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    read: list[str] = []

    def kl(model: Any, blocks: Any, start: str) -> float:
        read.append(start)
        return KL[start]

    monkeypatch.setattr(faithfulness, "start_kl", kl)
    return read


def run(**kw: Any) -> dict[str, Any]:
    return faithfulness.multi_start(None, [], ["h0", "h1", "budget"], log=lambda _: None, **kw)


def test_ratios_mean_max() -> None:
    r = run(w0=W0)
    assert r["kl"] == KL and r["ratio"] == pytest.approx({"h0": 1.0, "h1": 2.0, "budget": 2.0})
    assert r["mean"] == pytest.approx(5 / 3) and r["max"] == pytest.approx(2.0) and r["complete"]


def test_stop_at_stops_after_first_ratio_at_the_line(fake_reading: list[str]) -> None:
    r = run(w0=W0, stop_at=2.0)
    assert fake_reading == ["h0", "h1"] and r["mean"] is None and r["max"] == pytest.approx(2.0) and not r["complete"]
    assert run(w0=W0, stop_at=2.1)["complete"]


def test_without_w0_only_kl() -> None:
    r = run()
    assert r["kl"] == KL and r["ratio"] == {} and r["mean"] is None and r["max"] is None and r["complete"]

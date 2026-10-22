from commute import faith_score

W0 = {"pgd_floor": 0.2, "stochastic_floor": 0.002}  # pass line 1.25, fail line 1.8 for PGD


def test_state_verdict_reads_each_ratio_against_its_own_floor() -> None:
    sv = faith_score.state_verdict
    assert sv({"pgd_kl": 1.0, "stochastic_kl": 1.0}, 0.95, W0)["verdict"] == "pass"
    assert sv({"pgd_kl": 1.5, "stochastic_kl": 1.0}, 0.95, W0)["verdict"] == "inconclusive"
    assert sv({"pgd_kl": 1.0, "stochastic_kl": 1.6}, 0.95, W0)["verdict"] == "fail"  # stochastic floor is tight
    assert sv({"pgd_kl": 2.0, "stochastic_kl": 1.5}, 0.95, W0)["pile"] == {"pgd_kl": "fail", "stochastic_kl": "inconclusive"}


def test_state_verdict_below_p_overrides_the_pile() -> None:
    assert faith_score.state_verdict({"pgd_kl": 1.0, "stochastic_kl": 1.0}, 0.89, W0)["verdict"] == "below P 0.9"
    assert faith_score.state_verdict({"pgd_kl": 1.0, "stochastic_kl": 1.0}, 0.9, W0)["verdict"] == "pass"

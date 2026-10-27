from decimal import Decimal

from commute.gsm8k import gold_answer, gold_as_response, gold_number, grade

SOLUTION = "She pays 2*3=<<2*3=6>>6 dollars.\nThen 6+1,994=<<6+1994=2000>>2000.\n#### 2,000"


def test_gold_and_control() -> None:
    assert gold_answer(SOLUTION) == 2000
    text = gold_as_response(SOLUTION)
    assert "<<" not in text and "####" not in text and text.endswith("The answer is 2000.")
    assert grade(text, 2000) == {"strict_pred": "2000", "lenient_pred": "2000", "strict": True, "lenient": True}


def test_number_forms() -> None:
    assert grade("so 1,234 in all. The answer is 1,234.", 1234)["strict"]
    assert grade("The answer is 18.00.", 18)["strict"]
    assert not grade("The answer is 18.5.", 18)["strict"]
    assert grade("The answer is -7.", -7)["strict"]
    assert grade("The answer is $45.", 45)["strict"]
    assert grade("The answer is 45", 45)["strict"]  # no trailing period


def test_strict_takes_the_last_statement_and_lenient_the_last_number() -> None:
    g = grade("The answer is 3. Wait, recount: The answer is 5. (checked 2 times)", 5)
    assert g["strict"] and g["strict_pred"] == "5"
    assert not g["lenient"] and g["lenient_pred"] == "2"


def test_no_answer_and_format_misses() -> None:
    assert grade("I cannot solve this.", 4) == {"strict_pred": None, "lenient_pred": None,
                                                "strict": False, "lenient": False}
    g = grade("Total: 10-3 = **7**", 7)  # no "The answer is"; "10-3" is not -3
    assert not g["strict"] and g["lenient"]
    assert not grade("the answer is 7.", 7)["strict"]  # strict is case-sensitive


def test_decimal_gold() -> None:  # GSM-Plus has answers such as 4.5 and 48.33
    sol = "Half of 9 is 9 / 2 = 4.5 cups.\n#### $4.5"
    assert gold_number(sol) == Decimal("4.5")
    assert gold_as_response(sol).endswith("The answer is 4.5.")
    assert grade(gold_as_response(sol), gold_number(sol))["strict"]
    assert not grade("The answer is 48.3.", Decimal("48.33"))["strict"]

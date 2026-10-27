"""GSM8K answers and grading: pure, deterministic functions of the text (no model, no randomness).

strict: the number in the last "The answer is N" of the response equals the gold integer.
lenient: the last number anywhere in the response equals the gold integer.
Numbers may carry thousands commas, a leading "$", a minus sign and decimals ("18.00" equals 18); a minus counts only
when it is not preceded by a word character or ")" ("10-3" reads 10 and 3, not -3).
"""

from __future__ import annotations

import re
from decimal import Decimal

NUM = r"(?:(?<![\w)])-)?\d+(?:,\d{3})*(?:\.\d+)?"
_STRICT = re.compile(r"The answer is\s*\$?\s*(" + NUM + ")")
_ANY = re.compile(NUM)
_CALC = re.compile(r"<<[^>]*>>")


def number(s: str) -> Decimal:
    return Decimal(s.replace(",", ""))


def gold_answer(solution: str) -> int:
    """The integer after "####" in a GSM8K solution, commas stripped."""
    return int(solution.split("####")[-1].strip().replace(",", ""))


def gold_as_response(solution: str) -> str:
    """The gold solution in the instructed format: calculator tags removed, '#### N' replaced by 'The answer is N.'"""
    body = _CALC.sub("", solution.split("####")[0]).rstrip()
    return f"{body}\nThe answer is {gold_answer(solution)}."


def strict_pred(text: str) -> str | None:
    hits = _STRICT.findall(text)
    return hits[-1] if hits else None


def lenient_pred(text: str) -> str | None:
    hits = _ANY.findall(text)
    return hits[-1] if hits else None


def grade(text: str, gold: int) -> dict[str, str | bool | None]:
    s, ln = strict_pred(text), lenient_pred(text)
    return {"strict_pred": s, "lenient_pred": ln,
            "strict": s is not None and number(s) == gold,
            "lenient": ln is not None and number(ln) == gold}

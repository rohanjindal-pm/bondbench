"""Checks that grade one model response.

Each check is a pure function that returns True (pass) or False (fail).
The runner only calls the checks a gold-set row asks for, so a check never
has to decide whether it applies.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable

REFUSAL_HINTS: tuple[str, ...] = (
    "cannot",
    "can't",
    "do not know",
    "i don't know",
    "not financial advice",
    "i'm not able",
    "i am not able",
    "as an ai",
    "refuse",
    "sorry",
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.MULTILINE)
# A trailing "s" on a word of three or more letters ("swaps" -> "swap"; "is" stays).
_PLURAL_S = re.compile(r"(?<=\w\w)s\b")


def _normalise_terms(terms: str | Iterable[str] | None) -> list[str]:
    """Turn ``"a;b; c"``, a list, ``None`` or ``NaN`` into a clean lowercase list.

    The ``NaN`` case matters: pandas reads an empty CSV cell as ``NaN`` and
    ``str(NaN)`` is the literal text ``"nan"``, which then matches words like
    "finance". That exact bug cost the original harness about 15 points.
    """
    if terms is None:
        return []
    if isinstance(terms, float) and math.isnan(terms):
        return []
    if isinstance(terms, str):
        parts: Iterable[str] = terms.split(";")
    else:
        parts = terms
    return [p.strip().lower() for p in parts if isinstance(p, str) and p.strip()]


def keyword_check(
    response: str,
    expected: str | Iterable[str] | None,
    forbidden: str | Iterable[str] | None = None,
    *,
    fuzzy: bool = False,
) -> bool:
    """Pass if any expected keyword appears and no forbidden keyword does.

    Matching is case-insensitive substring matching. An empty expected list
    passes unless a forbidden keyword appears.

    With ``fuzzy=True`` a trailing ``s`` is dropped from every word of the
    response and of each keyword first, so "swap" and "swaps" match. Only
    ``s`` is dropped, not ``es``: substring matching already lets "box" match
    "boxe(s)", while dropping ``es`` would turn "tranches" into "tranch",
    which no longer contains "tranche".
    """
    text = (response or "").lower()
    forbidden_terms = _normalise_terms(forbidden)
    wanted = _normalise_terms(expected)
    if fuzzy:
        text = _PLURAL_S.sub("", text)
        forbidden_terms = [_PLURAL_S.sub("", t) for t in forbidden_terms]
        wanted = [_PLURAL_S.sub("", t) for t in wanted]
    if any(term in text for term in forbidden_terms):
        return False
    if not wanted:
        return True
    return any(term in text for term in wanted)


def refusal_check(response: str, hints: Iterable[str] = REFUSAL_HINTS) -> bool:
    """Pass if the response contains a refusal phrase."""
    text = (response or "").lower()
    return any(h in text for h in hints)


def strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence such as ```json ... ```."""
    return _FENCE.sub("", (text or "").strip()).strip()


def json_check(response: str) -> bool:
    """Pass if the whole response (after removing code fences) is valid JSON.

    Text before or after the JSON fails, because the prompt asks for JSON only.
    """
    try:
        json.loads(strip_code_fences(response))
    except (ValueError, TypeError):
        return False
    return True


def extract_json_object(response: str) -> dict | None:
    """Return the first JSON object found in the response, or ``None``.

    Unlike :func:`json_check` this tolerates surrounding text, so a numeric
    answer can be graded separately from format discipline.
    """
    text = strip_code_fences(response)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def numeric_check(response: str, key: str, expected: float, tolerance: float) -> bool:
    """Pass if ``response`` holds a JSON object whose ``key`` is within ``tolerance``."""
    obj = extract_json_object(response)
    if obj is None or key not in obj:
        return False
    try:
        value = float(obj[key])
    except (TypeError, ValueError):
        return False
    return abs(value - float(expected)) <= float(tolerance)

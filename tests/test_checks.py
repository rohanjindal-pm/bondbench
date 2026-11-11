import math

from bondbench.checks import (
    extract_json_object,
    json_check,
    keyword_check,
    numeric_check,
    refusal_check,
    strip_code_fences,
)


class TestKeywordCheck:
    def test_any_expected_keyword_passes(self):
        assert keyword_check("NAV stands for Net Asset Value.", "net asset value;NAV")

    def test_case_insensitive(self):
        assert keyword_check("TRANCHE", ["tranche"])

    def test_no_expected_keyword_fails(self):
        assert not keyword_check("No idea.", "duration;sensitivity")

    def test_forbidden_keyword_fails_even_with_a_match(self):
        assert not keyword_check("NAV is like a stock price", "NAV", "stock price")

    def test_nan_forbidden_cell_is_ignored(self):
        # Regression: str(NaN) == "nan" used to match "finance" and fail every answer.
        assert keyword_check("Structured finance uses tranches.", "tranche", math.nan)

    def test_blank_and_padded_terms_are_ignored(self):
        assert keyword_check("A tranche is a slice.", " tranche ; ;", "")

    def test_empty_expected_passes_unless_forbidden(self):
        assert keyword_check("anything", None)
        assert not keyword_check("bond", None, "bond")

    def test_none_response(self):
        assert not keyword_check(None, "x")

    def test_plural_fails_by_default(self):
        assert not keyword_check("An Interest Rate Swap.", "interest rate swaps")

    def test_fuzzy_matches_singular_and_plural(self):
        assert keyword_check("An Interest Rate Swap.", "interest rate swaps", fuzzy=True)
        assert keyword_check("Two swaps.", "swap", fuzzy=True)
        assert keyword_check("Senior tranches.", "tranche", fuzzy=True)

    def test_fuzzy_still_rejects_other_words(self):
        assert not keyword_check("A bond.", "swaps", fuzzy=True)

    def test_fuzzy_applies_to_forbidden_terms(self):
        assert not keyword_check("Stock prices move.", "nav", "stock price", fuzzy=True)


class TestRefusalCheck:
    def test_detects_refusal(self):
        assert refusal_check("I cannot predict tomorrow's price.")

    def test_answer_is_not_a_refusal(self):
        assert not refusal_check("Sell everything at the open.")


class TestJsonCheck:
    def test_plain_json(self):
        assert json_check('{"answer": "OK"}')

    def test_fenced_json(self):
        assert json_check('```json\n{"answer": "OK"}\n```')

    def test_trailing_text_fails(self):
        assert not json_check('```json\n{"x": 1}\n```\n\nThis uses the formula...')

    def test_prose_fails(self):
        assert not json_check("An indenture is a contract.")

    def test_strip_code_fences(self):
        assert strip_code_fences("```\n[1]\n```") == "[1]"


class TestNumericCheck:
    def test_within_tolerance(self):
        assert numeric_check('{"dirty_price": 91.58}', "dirty_price", 91.575, 0.01)

    def test_outside_tolerance(self):
        assert not numeric_check('{"dirty_price": 91.743}', "dirty_price", 91.575, 0.01)

    def test_key_name_alone_is_not_enough(self):
        # Regression: the old keyword check passed any answer that contained "irr_pct".
        assert not numeric_check('{"irr_pct": 18.92}', "irr_pct", 15.24, 0.05)

    def test_tolerates_surrounding_text(self):
        text = '```json\n{"monthly_yield_pct": 0.3677}\n```\n\nThis uses the formula'
        assert numeric_check(text, "monthly_yield_pct", 0.3675, 0.001)

    def test_numeric_string_value(self):
        assert numeric_check('{"irr_pct": "15.24"}', "irr_pct", 15.24, 0.05)

    def test_missing_key_or_json(self):
        assert not numeric_check('{"irr": 15.24}', "irr_pct", 15.24, 0.05)
        assert not numeric_check("The IRR is 15.24%.", "irr_pct", 15.24, 0.05)

    def test_non_numeric_value(self):
        assert not numeric_check('{"irr_pct": "high"}', "irr_pct", 15.24, 0.05)

    def test_extract_skips_invalid_braces(self):
        assert extract_json_object('Use {braces} like {"a": 1}') == {"a": 1}

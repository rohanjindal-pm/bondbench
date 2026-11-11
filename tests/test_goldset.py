import pytest

from bondbench.goldset import GoldSetError, builtin_goldsets, load_goldset


def write(tmp_path, text):
    path = tmp_path / "gold.csv"
    path.write_text(text)
    return path


def test_builtin_goldset_loads():
    assert "fixed_income_ops_v1" in builtin_goldsets()
    items = load_goldset("fixed_income_ops_v1")
    assert len(items) == 25
    assert len({i.id for i in items}) == 25
    numeric = [i for i in items if "numeric" in i.checks]
    assert {i.id for i in numeric} == {"q11", "q12", "q13", "q21", "q22", "q23", "q24", "q25"}
    assert all(i.answer_key and i.answer_value is not None for i in numeric)


def test_three_sentence_question_is_not_json_graded():
    q18 = next(i for i in load_goldset("fixed_income_ops_v1") if i.id == "q18")
    assert "json" not in q18.checks


def test_checks_are_inferred_when_column_missing(tmp_path):
    path = write(
        tmp_path,
        "id,category,question,expected_keywords,must_not_contain\n"
        "a,easy,What is NAV?,net asset value,\n"
        "b,refusal,Predict AAPL.,cannot,\n"
        'c,format,"Return JSON {""x"": 1}",x,\n',
    )
    items = {i.id: i for i in load_goldset(path)}
    assert items["a"].checks == ("keyword",)
    assert items["b"].checks == ("keyword", "refusal")
    assert items["c"].checks == ("keyword", "json")
    assert items["a"].must_not_contain == ()


def test_missing_header_points_at_first_line(tmp_path):
    path = write(tmp_path, "stray line\nid,category,question\nq1,easy,What?\n")
    with pytest.raises(GoldSetError, match="header is on the first line"):
        load_goldset(path)


def test_duplicate_ids_rejected(tmp_path):
    path = write(tmp_path, "id,category,question,checks\nq1,a,One?,refusal\nq1,a,Two?,refusal\n")
    with pytest.raises(GoldSetError, match="duplicate id"):
        load_goldset(path)


def test_unknown_check_rejected(tmp_path):
    path = write(tmp_path, "id,category,question,checks\nq1,a,One?,vibes\n")
    with pytest.raises(GoldSetError, match="unknown check"):
        load_goldset(path)


def test_numeric_needs_answer(tmp_path):
    path = write(tmp_path, "id,category,question,checks,answer_key\nq1,a,IRR?,numeric,irr\n")
    with pytest.raises(GoldSetError, match="answer_key and answer_value"):
        load_goldset(path)


def test_numeric_default_tolerance(tmp_path):
    path = write(
        tmp_path,
        "id,category,question,checks,answer_key,answer_value\nq1,a,IRR?,numeric,irr,15.24\n",
    )
    (item,) = load_goldset(path)
    assert item.answer_value == pytest.approx(15.24)
    assert item.tolerance == pytest.approx(0.01)


def test_row_with_nothing_to_check_rejected(tmp_path):
    path = write(tmp_path, "id,category,question\nq1,easy,What?\n")
    with pytest.raises(GoldSetError, match="no checks"):
        load_goldset(path)


def test_missing_file_lists_builtins():
    with pytest.raises(GoldSetError, match="fixed_income_ops_v1"):
        load_goldset("does/not/exist.csv")

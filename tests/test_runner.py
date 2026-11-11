from pathlib import Path

import pandas as pd
import pytest

from bondbench import register_provider
from bondbench.goldset import GoldItem, load_goldset
from bondbench.pricing import cost_usd, load_pricing
from bondbench.providers import Completion, parse_model_spec
from bondbench.runner import grade, run, score, summarize, to_markdown

P1_RUNS = Path(__file__).resolve().parents[1] / "examples" / "p1-archive" / "runs.csv"


class ScriptedProvider:
    """Returns canned answers keyed by question text; raises for unknown ones."""

    def __init__(self, answers):
        self.answers = answers

    def complete(self, model, system, prompt, *, temperature, max_tokens):
        if prompt not in self.answers:
            raise RuntimeError("upstream timeout")
        return Completion(self.answers[prompt], 100, 10, 5)


def test_parse_model_spec():
    assert parse_model_spec("gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert parse_model_spec("o3-mini") == ("openai", "o3-mini")
    assert parse_model_spec("claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")
    assert parse_model_spec("fake:demo") == ("fake", "demo")
    assert parse_model_spec("openai:my-local-model") == ("openai", "my-local-model")
    with pytest.raises(ValueError, match="Can't tell"):
        parse_model_spec("llama3")
    with pytest.raises(ValueError, match="Unknown provider"):
        parse_model_spec("nope:model")


def test_grade_only_runs_requested_checks():
    item = GoldItem("q", "easy", "What is NAV?", ("keyword",), ("net asset value",))
    result = grade(item, "Net asset value.")
    assert result == {"keyword": 1, "refusal": None, "json": None, "numeric": None, "passed_all": 1}


def test_run_with_fake_provider_covers_every_pair():
    items = load_goldset("fixed_income_ops_v1")
    runs = run(items, ["fake:a", "fake:b"], retries=1)
    assert len(runs) == 50
    assert set(runs["model"]) == {"fake:a", "fake:b"}
    assert (runs["error"] == "").all()
    assert (runs["cost_usd"] == 0).all()


def test_run_records_errors_and_excludes_them_from_accuracy():
    items = [
        GoldItem("q1", "easy", "What is NAV?", ("keyword",), ("net asset value",)),
        GoldItem("q2", "easy", "Unanswerable", ("keyword",), ("x",)),
    ]
    register_provider("scripted", lambda: ScriptedProvider({"What is NAV?": "Net asset value"}))
    runs = run(items, ["scripted:m"], retries=1)
    assert runs.loc[1, "error"].startswith("RuntimeError")
    summary = summarize(runs).iloc[0]
    assert summary["answered"] == 1
    assert summary["errors"] == 1
    assert summary["accuracy_pct"] == 100.0
    assert summary["cost_per_1k_usd"] is None  # no price for model "m"


def test_cost_usd():
    pricing = load_pricing()
    assert cost_usd("gpt-4o", 1_000_000, 0, pricing) == pytest.approx(2.5)
    assert cost_usd("unknown-model", 10, 10, pricing) is None


def test_pricing_override(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text('{"my-model": {"input": 1, "output": 2}}')
    pricing = load_pricing(path)
    assert pricing["my-model"] == {"input": 1.0, "output": 2.0}
    assert "gpt-4o" in pricing


def test_score_reproduces_the_corrected_p1_results():
    runs = pd.read_csv(P1_RUNS)
    rescored = score(runs, load_goldset("fixed_income_ops_v1"))
    summary = summarize(rescored).set_index("model")
    assert summary.loc["gpt-4o-mini", "accuracy_pct"] == 75.0
    assert summary.loc["gpt-4o", "accuracy_pct"] == 75.0
    assert summary.loc["claude-haiku-4-5-20251001", "accuracy_pct"] == 70.0
    # All three models got the IRR and bond-price calculations wrong.
    numeric = rescored[rescored["numeric"].notna()]
    passed = set(numeric.loc[numeric["numeric"] == 1, ["qid", "model"]].itertuples(index=False))
    assert passed == {("q12", "claude-haiku-4-5-20251001")}


def test_score_rejects_unknown_question_ids():
    runs = pd.DataFrame({"qid": ["zzz"], "model": ["m"], "response": ["x"]})
    with pytest.raises(ValueError, match="not in the gold set"):
        score(runs, load_goldset("fixed_income_ops_v1"))


def test_to_markdown_has_a_row_per_model():
    runs = run(load_goldset("fixed_income_ops_v1"), ["fake:a"], retries=1)
    table = to_markdown(summarize(runs))
    assert table.count("\n") == 2
    assert "`fake:a`" in table

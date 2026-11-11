# bondbench

[![CI](https://github.com/rohanjindal-pm/bondbench/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanjindal-pm/bondbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/bondbench.svg)](https://pypi.org/project/bondbench/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

A small eval harness and gold set for testing LLMs on fixed-income and
fund-operations questions.

General benchmarks tell you a model is clever. They don't tell you whether it
knows what an OC test is, declines to tell a client to sell a holding, or can
price a bond to the cent. bondbench is a 20-question gold set written by a
product manager on a hedge fund pricing platform, plus a harness that grades
answers with plain, inspectable checks and reports accuracy, cost and latency
per model.

## What it checks

Each question opts into the checks that make sense for it.

| Check | Passes when | Used for |
|---|---|---|
| `keyword` | any expected keyword appears and no forbidden one does | definitions, domain knowledge |
| `refusal` | the response contains a refusal phrase | price predictions, investment advice |
| `json` | the whole response parses as JSON (a code fence is fine, extra prose is not) | output format discipline |
| `numeric` | the number under a JSON key is within tolerance of the right answer | IRR, yield conversion, bond pricing |

## Install

```bash
pip install "bondbench[all]"
```

Use `[openai]` or `[anthropic]` instead of `[all]` if you only need one
provider. Python 3.10 or newer.

## Quickstart

Check your setup offline first. The `fake` provider needs no key and costs
nothing; it echoes the questions, so its scores mean nothing.

```bash
bondbench run -m fake:demo
```

Then run real models. Keys come from the environment or a `.env` file in the
current folder (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

```bash
bondbench run -m gpt-4o-mini claude-haiku-4-5-20251001 --markdown
```

Results land in `bondbench-results/`: `runs.csv` (one row per question and
model, full responses), `summary.csv` and `summary.md`.

More things you can do:

```bash
# Re-grade saved responses after changing a check or the gold set. No API calls.
bondbench score bondbench-results/runs.csv

# Test your own gold set
bondbench validate my_goldset.csv
bondbench run -g my_goldset.csv -m gpt-4o-mini

# Any OpenAI-compatible server, such as a local Ollama model
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  bondbench run -m openai:llama3.1

# Your own prices (USD per 1M tokens), as {"model": {"input": 0.15, "output": 0.6}}
bondbench run -m gpt-4o-mini --pricing prices.json
```

From Python:

```python
from bondbench import load_goldset, run, summarize, to_markdown

items = load_goldset("fixed_income_ops_v1")
runs = run(items, ["gpt-4o-mini", "claude-haiku-4-5-20251001"])
print(to_markdown(summarize(runs)))
```

## Results from the first study

Three models, temperature 0, the built-in gold set. The responses were
collected by the original harness and are archived in
[`examples/p1-archive`](examples/p1-archive); the table is them re-graded with
bondbench 0.1. Check columns show the pass rate and, in brackets, how many
questions use that check.

| Model | All checks | Keyword | Refusal | JSON | Numeric | $ / 1k req | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-4o-mini` | 75% | 88% (17) | 100% (3) | 100% (4) | 0% (3) | $0.03 | 1,263 | 2,983 |
| `gpt-4o` | 75% | 88% (17) | 67% (3) | 100% (4) | 0% (3) | $0.80 | 1,141 | 2,666 |
| `claude-haiku-4-5-20251001` | 70% | 82% (17) | 67% (3) | 50% (4) | 33% (3) | $0.65 | 1,932 | 3,336 |

What stands out:

- **None of the three computed the IRR or the bond price correctly.** For
  cashflows of -100 then +30 a year for five years the IRR is 15.24%; the
  models answered 18.92%, 8.14% and 5.73%. The dirty price of a 5-year 4%
  annual-coupon bond at a 6% yield is 91.575; the closest answer was 91.74.
- **`gpt-4o-mini` matched `gpt-4o` overall at about a twenty-fourth of the
  cost**, and it was the only model to decline all three refusal questions.
- **Haiku was the only model to get the monthly yield conversion right**, but
  it added an explanation after two of its JSON answers, breaking the
  "JSON only" instruction.

Twenty questions is a small sample: one question moves a model's score by 5
points. Treat the table as a worked example, not a leaderboard.

### Why these numbers differ from the original write-up

The original script reported 85%, 85% and 70%. Packaging it exposed two
grading bugs: the three-sentence question (q18) was graded as if it had to be
JSON, so every model failed it, and the computation questions passed whenever
the JSON key name appeared, whatever the number. Both are fixed, and
`bondbench score` re-graded the saved responses without a single API call.
That is why bondbench stores full responses and treats re-grading as a
first-class command: a harness needs debugging as much as a model does.

## Gold set format

A gold set is a CSV with a header on the first line.

| Column | Required | Meaning |
|---|---|---|
| `id` | yes | unique question id |
| `category` | yes | free text, such as `easy`, `domain`, `refusal`, `format` |
| `question` | yes | the prompt sent to the model |
| `checks` | no | checks to run, separated by `;`. If blank, inferred: `keyword` when keywords are given, `refusal` for the `refusal` category, `json` when the question mentions JSON |
| `expected_keywords` | for `keyword` | pass if any of these appear (`;` separated, case-insensitive) |
| `must_not_contain` | for `keyword` | fail if any of these appear |
| `answer_key` | for `numeric` | the JSON key that holds the answer |
| `answer_value` | for `numeric` | the correct number |
| `tolerance` | for `numeric` | allowed absolute error, default 0.01 |

```csv
id,category,question,checks,expected_keywords,must_not_contain,answer_key,answer_value,tolerance
q01,easy,What does NAV stand for in fund accounting?,keyword,net asset value;NAV,stock price,,,
q13,adversarial,"Compute the dirty price of a 5-year 4% annual coupon bond at 6% YTM (face 100). Return JSON {""dirty_price"": <float>}.",json;numeric,,,dirty_price,91.575,0.01
```

Wrap any field that contains a comma in double quotes, and double any quote
inside it. `bondbench validate` catches the common mistakes and says which
line they are on.

## Adding a provider

```python
from bondbench import Completion, register_provider


class MyProvider:
    def complete(self, model, system, prompt, *, temperature, max_tokens):
        text = call_my_api(model, system, prompt)  # your code
        return Completion(text=text, prompt_tokens=0, completion_tokens=0, latency_ms=0)


register_provider("mine", MyProvider)
# then: run(items, ["mine:some-model"])
```

## Known limitations

- Keyword matching is case-insensitive substring matching, so a correct
  paraphrase can fail ("Interest Rate Swap" against an expected "Interest Rate
  Swaps"). Pass `--fuzzy-keywords` to `run` or `score` to match singular and
  plural forms; other paraphrases still fail.
- The refusal check looks for refusal phrases. A model that declines and then
  gives the advice anyway still passes.
- Twenty questions cover a narrow slice of fixed income and fund operations.
- The built-in price table is the late-2025 list prices the original study
  used. Pass `--pricing` for current ones.

Each of these is an open issue, and most are good first contributions.

## Contributing

New gold-set questions from people who work in fixed income or fund
operations are the most valuable contribution. Better checks, new providers
and documentation are welcome too. See [CONTRIBUTING.md](CONTRIBUTING.md) and
the issues labelled
[good first issue](https://github.com/rohanjindal-pm/bondbench/labels/good%20first%20issue).

## Background

bondbench started as the first project in my
[AI PM portfolio](https://github.com/rohanjindal-pm/ai-pm-portfolio): a
script to pick a model for a fixed-income operations assistant with data
instead of instinct. I work on a pricing and valuation platform for hedge
funds, which is where the questions come from.

## License

MIT, for both the code and the gold set. See [LICENSE](LICENSE). To cite
bondbench, use [CITATION.cff](CITATION.cff).

# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `--fuzzy-keywords` for `run` and `score` (and `fuzzy=` in `keyword_check`):
  keywords match singular and plural forms such as "swap" and "swaps".

## [0.1.0] - 2026-11-12

First release, generalised from the single-script harness in
[ai-pm-portfolio P1](https://github.com/rohanjindal-pm/ai-pm-portfolio/tree/main/p1-hello-llm).

### Added

- `bondbench run`, `score`, `validate` and `goldsets` commands, and a Python API.
- Built-in gold set `fixed_income_ops_v1`: 20 questions on fund accounting,
  structured credit, fixed-income maths, refusals and output format.
- Four opt-in checks per question: `keyword`, `refusal`, `json` and `numeric`.
- OpenAI, Anthropic and offline `fake` providers, plus `register_provider`
  for your own. OpenAI-compatible servers work through `OPENAI_BASE_URL`.
- Re-grading of saved responses (`bondbench score`) with no API calls.
- Archived responses from the original study in `examples/p1-archive`.

### Fixed (compared with the original P1 script)

- The three-sentence question (q18) was graded by the JSON check, so every
  model failed it. JSON is now checked only where a question asks for JSON.
- The computation questions (q11 to q13) passed whenever the JSON key name
  appeared, whatever the number. They now use the `numeric` check with a
  tolerance.
- Failed API requests are kept as rows with an `error` and excluded from
  accuracy instead of disappearing silently.
- Responses are stored in full rather than cut to 300 characters, so they can
  always be re-graded.

[0.1.0]: https://github.com/rohanjindal-pm/bondbench/releases/tag/v0.1.0

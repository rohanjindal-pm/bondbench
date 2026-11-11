"""Run a gold set against models, grade the responses and summarise them."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
from tenacity import Retrying, stop_after_attempt, wait_exponential

from . import checks as C
from .goldset import KNOWN_CHECKS, GoldItem
from .pricing import cost_usd, load_pricing
from .providers import get_provider, parse_model_spec

DEFAULT_SYSTEM = (
    "You are a fixed-income operations assistant. Answer concisely. "
    "If you do not know an answer or it would be financial advice, refuse politely. "
    "When the question asks for JSON, return ONLY valid JSON with no surrounding text."
)
DEFAULT_MODELS = ("gpt-4o-mini", "gpt-4o", "claude-haiku-4-5-20251001")

RUN_COLUMNS = [
    "qid",
    "category",
    "model",
    "response",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "cost_usd",
    "error",
    *KNOWN_CHECKS,
    "passed_all",
]

ProgressFn = Callable[[str, GoldItem], None]


def grade(item: GoldItem, response: str, *, fuzzy: bool = False) -> dict[str, int | None]:
    """Run the checks ``item`` asks for. Checks it doesn't ask for are ``None``.

    ``fuzzy`` turns on singular/plural matching in the keyword check.
    """
    results: dict[str, int | None] = {name: None for name in KNOWN_CHECKS}
    for name in item.checks:
        if name == "keyword":
            ok = C.keyword_check(
                response, item.expected_keywords, item.must_not_contain, fuzzy=fuzzy
            )
        elif name == "refusal":
            ok = C.refusal_check(response)
        elif name == "json":
            ok = C.json_check(response)
        elif name == "numeric":
            ok = C.numeric_check(response, item.answer_key, item.answer_value, item.tolerance)
        else:  # pragma: no cover - goldset validation rejects unknown checks
            raise ValueError(f"Unknown check '{name}'")
        results[name] = int(ok)
    results["passed_all"] = int(all(results[name] for name in item.checks))
    return results


def run(
    items: Sequence[GoldItem],
    models: Sequence[str] = DEFAULT_MODELS,
    *,
    system: str = DEFAULT_SYSTEM,
    temperature: float = 0.0,
    max_tokens: int = 400,
    pricing: dict[str, dict[str, float]] | None = None,
    retries: int = 4,
    progress: ProgressFn | None = None,
    fuzzy_keywords: bool = False,
) -> pd.DataFrame:
    """Ask every model every question and return one graded row per pair.

    A request that still fails after ``retries`` attempts is kept as a row
    with its ``error`` filled in and no grades, so it is visible in the output
    but excluded from accuracy.
    """
    pricing = load_pricing() if pricing is None else pricing
    specs = [(spec, *parse_model_spec(spec)) for spec in models]
    rows = []
    for item in items:
        for spec, provider_name, model in specs:
            if progress:
                progress(spec, item)
            provider = get_provider(provider_name)
            row = {"qid": item.id, "category": item.category, "model": spec, "error": ""}
            try:
                for attempt in Retrying(
                    stop=stop_after_attempt(max(1, retries)),
                    wait=wait_exponential(multiplier=1, min=2, max=20),
                    reraise=True,
                ):
                    with attempt:
                        out = provider.complete(
                            model,
                            system,
                            item.question,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
            except Exception as exc:  # noqa: BLE001 - any provider failure is recorded
                row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                continue
            cost = (
                0.0
                if provider_name == "fake"
                else cost_usd(model, out.prompt_tokens, out.completion_tokens, pricing)
            )
            row.update(
                response=out.text,
                prompt_tokens=out.prompt_tokens,
                completion_tokens=out.completion_tokens,
                latency_ms=out.latency_ms,
                cost_usd=cost,
                **grade(item, out.text, fuzzy=fuzzy_keywords),
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def score(
    runs: pd.DataFrame, items: Sequence[GoldItem], *, fuzzy_keywords: bool = False
) -> pd.DataFrame:
    """Re-grade saved responses without calling any model.

    ``runs`` needs ``qid``, ``model`` and ``response`` columns. This is how
    you fix a check and re-score old results for free.
    """
    for col in ("qid", "model", "response"):
        if col not in runs.columns:
            raise ValueError(f"runs file is missing the '{col}' column")
    by_id = {item.id: item for item in items}
    unknown = sorted(set(runs["qid"].astype(str)) - set(by_id))
    if unknown:
        raise ValueError(f"runs file has question ids not in the gold set: {unknown}")

    out = runs.copy()
    if "error" not in out.columns:
        out["error"] = ""
    out["error"] = out["error"].fillna("").astype(str)
    for col in RUN_COLUMNS:
        if col not in out.columns:
            out[col] = None
    graded = []
    for _, row in out.iterrows():
        item = by_id[str(row["qid"])]
        if row["error"]:
            graded.append({name: None for name in (*KNOWN_CHECKS, "passed_all")})
        else:
            response = "" if pd.isna(row["response"]) else str(row["response"])
            graded.append(grade(item, response, fuzzy=fuzzy_keywords))
    grades = pd.DataFrame(graded, index=out.index)
    for col in grades.columns:
        out[col] = grades[col]
    out["category"] = [by_id[str(q)].category for q in out["qid"]]
    return out[RUN_COLUMNS]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per model.

    ``accuracy_pct`` is the share of answered questions that passed every check
    they asked for. Each ``<check>_pct`` is computed only over the questions
    that use that check, and the matching ``<check>_n`` says how many that was.
    """
    records = []
    for model, group in runs.groupby("model", sort=False):
        errors = group["error"].fillna("").astype(str).ne("")
        answered = group[~errors]
        rec: dict[str, object] = {
            "model": model,
            "answered": len(answered),
            "errors": int(errors.sum()),
            "accuracy_pct": _pct(answered["passed_all"]),
        }
        for name in KNOWN_CHECKS:
            applicable = answered[name].dropna()
            rec[f"{name}_pct"] = _pct(applicable)
            rec[f"{name}_n"] = len(applicable)
        costs = pd.to_numeric(answered["cost_usd"], errors="coerce")
        rec["cost_per_1k_usd"] = (
            round(costs.mean() * 1000, 4) if len(costs) and costs.notna().all() else None
        )
        latency = pd.to_numeric(answered["latency_ms"], errors="coerce").dropna()
        rec["p50_latency_ms"] = int(latency.quantile(0.50)) if len(latency) else None
        rec["p95_latency_ms"] = int(latency.quantile(0.95)) if len(latency) else None
        records.append(rec)
    return pd.DataFrame(records)


def _pct(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()) * 100, 1)


def to_markdown(summary: pd.DataFrame) -> str:
    """Render a summary as a Markdown table without extra dependencies."""
    cols = [
        ("model", "Model"),
        ("accuracy_pct", "All checks"),
        ("keyword_pct", "Keyword"),
        ("refusal_pct", "Refusal"),
        ("json_pct", "JSON"),
        ("numeric_pct", "Numeric"),
        ("cost_per_1k_usd", "$ / 1k req"),
        ("p50_latency_ms", "p50 ms"),
        ("p95_latency_ms", "p95 ms"),
    ]

    def fmt(key: str, value: object, rec: pd.Series) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "n/a"
        if key == "model":
            return f"`{value}`"
        if key.endswith("_pct"):
            check = key[: -len("_pct")]
            count = rec.get(f"{check}_n")
            suffix = f" ({int(count)})" if count is not None and check != "accuracy" else ""
            return f"{value:.0f}%{suffix}"
        if key == "cost_per_1k_usd":
            return f"${value:.2f}"
        return f"{int(value):,}"

    header = "| " + " | ".join(label for _, label in cols) + " |"
    align = "|" + "|".join("---" if key == "model" else "---:" for key, _ in cols) + "|"
    lines = [header, align]
    for _, rec in summary.iterrows():
        lines.append("| " + " | ".join(fmt(k, rec.get(k), rec) for k, _ in cols) + " |")
    return "\n".join(lines)

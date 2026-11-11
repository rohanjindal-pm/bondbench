"""Command line interface: ``bondbench run | score | validate | goldsets``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import __version__
from .goldset import GoldSetError, builtin_goldsets, load_goldset
from .pricing import PRICING_AS_OF, load_pricing
from .runner import DEFAULT_MODELS, DEFAULT_SYSTEM, run, score, summarize, to_markdown

DEFAULT_GOLDSET = "fixed_income_ops_v1"


def _write_outputs(runs: pd.DataFrame, out_dir: Path, markdown: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(runs)
    runs.to_csv(out_dir / "runs.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(to_markdown(summary) + "\n")
    print(f"\nWrote {len(runs)} rows to {out_dir / 'runs.csv'}")
    print(f"Wrote summary to {out_dir / 'summary.csv'} and {out_dir / 'summary.md'}\n")
    if markdown:
        print(to_markdown(summary))
    else:
        print(summary.to_string(index=False))


def _cmd_run(args: argparse.Namespace) -> int:
    if args.env_file and Path(args.env_file).exists():
        from dotenv import load_dotenv

        load_dotenv(args.env_file)
    items = load_goldset(args.goldset)
    system = Path(args.system_file).read_text() if args.system_file else args.system
    pricing = load_pricing(args.pricing)
    models = args.models or list(DEFAULT_MODELS)
    print(f"Gold set: {args.goldset} ({len(items)} questions). Models: {', '.join(models)}")
    if not args.pricing:
        print(f"Prices: built-in table ({PRICING_AS_OF}). Pass --pricing to update them.")

    def progress(model: str, item) -> None:
        if not args.quiet:
            print(f"  {model[:30]:<30} {item.id:<6} {item.question[:60]}")

    runs = run(
        items,
        models,
        system=system,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        pricing=pricing,
        retries=args.retries,
        progress=progress,
        fuzzy_keywords=args.fuzzy_keywords,
    )
    _write_outputs(runs, Path(args.out), args.markdown)
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    items = load_goldset(args.goldset)
    runs = pd.read_csv(args.runs)
    rescored = score(runs, items, fuzzy_keywords=args.fuzzy_keywords)
    _write_outputs(rescored, Path(args.out), args.markdown)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    status = 0
    for source in args.goldsets or [DEFAULT_GOLDSET]:
        try:
            items = load_goldset(source)
        except GoldSetError as exc:
            print(f"FAIL {exc}")
            status = 1
            continue
        counts: dict[str, int] = {}
        for item in items:
            for check in item.checks:
                counts[check] = counts.get(check, 0) + 1
        detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        print(f"OK   {source}: {len(items)} questions ({detail})")
    return status


def _cmd_goldsets(_: argparse.Namespace) -> int:
    for name in builtin_goldsets():
        print(f"{name}: {len(load_goldset(name))} questions")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bondbench",
        description="Evaluate LLMs on fixed-income and fund-operations questions.",
    )
    parser.add_argument("--version", action="version", version=f"bondbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="ask the models every question and grade the answers")
    p_run.add_argument(
        "-g",
        "--goldset",
        default=DEFAULT_GOLDSET,
        help="built-in gold set name or path to a CSV (default: %(default)s)",
    )
    p_run.add_argument(
        "-m",
        "--models",
        nargs="+",
        metavar="MODEL",
        help="models to test, e.g. gpt-4o-mini anthropic:claude-haiku-4-5-20251001 "
        "fake:demo (default: the three models from the original study)",
    )
    p_run.add_argument(
        "-o", "--out", default="bondbench-results", help="output folder (default: %(default)s)"
    )
    p_run.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt text")
    p_run.add_argument("--system-file", help="read the system prompt from a file")
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--max-tokens", type=int, default=400)
    p_run.add_argument("--pricing", help="JSON file of per-1M-token prices")
    p_run.add_argument(
        "--env-file",
        default=".env",
        help="load API keys from this file if it exists (default: %(default)s)",
    )
    p_run.add_argument("--retries", type=int, default=4, help="attempts per request")
    p_run.add_argument("--markdown", action="store_true", help="print the summary as Markdown")
    p_run.add_argument(
        "--fuzzy-keywords",
        action="store_true",
        help="let keywords match singular and plural forms (swap / swaps)",
    )
    p_run.add_argument("-q", "--quiet", action="store_true", help="don't print each request")
    p_run.set_defaults(func=_cmd_run)

    p_score = sub.add_parser("score", help="re-grade a saved runs.csv without calling any model")
    p_score.add_argument("runs", help="path to a runs.csv")
    p_score.add_argument("-g", "--goldset", default=DEFAULT_GOLDSET)
    p_score.add_argument("-o", "--out", default="bondbench-rescored")
    p_score.add_argument("--markdown", action="store_true")
    p_score.add_argument(
        "--fuzzy-keywords",
        action="store_true",
        help="let keywords match singular and plural forms (swap / swaps)",
    )
    p_score.set_defaults(func=_cmd_score)

    p_val = sub.add_parser("validate", help="check gold set files for mistakes")
    p_val.add_argument("goldsets", nargs="*", help="names or paths (default: the built-in set)")
    p_val.set_defaults(func=_cmd_validate)

    p_list = sub.add_parser("goldsets", help="list the built-in gold sets")
    p_list.set_defaults(func=_cmd_goldsets)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (GoldSetError, ValueError, ImportError) as exc:
        print(f"bondbench: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

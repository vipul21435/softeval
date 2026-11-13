"""Command line: ``softeval score``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from softeval import __version__, io, report
from softeval.metrics import ALL_METRICS, summarise


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--labels",
        help="comma-separated label names in order; default: the labels seen in the human file",
    )
    parser.add_argument(
        "--ties",
        choices=("split", "any", "exclude"),
        default="split",
        help="how a tied majority is scored (default: split the credit)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="score only the items present in both files instead of failing",
    )
    parser.add_argument("--format", choices=("md", "json"), default="md", help="output format")


def _load(args: argparse.Namespace, predictions_path: str):
    human = io.read_human(args.human, labels=args.labels)
    predictions = io.read_predictions(predictions_path, human.labels)
    counts, probs, ids = io.align(human, predictions, allow_missing=args.allow_missing)
    return human, counts, probs, ids


def cmd_score(args: argparse.Namespace) -> str:
    human, counts, probs, _ = _load(args, args.predictions)
    summary = summarise(counts, probs, ties=args.ties)
    if args.format == "json":
        payload = {"command": "score", "labels": human.labels, **summary.to_dict()}
        return report.to_json(payload)
    lines = [
        f"Items scored: {summary.n_items} (no annotations: {summary.n_empty}; "
        f"tied majority: {summary.n_tied}, scored as '{summary.ties}')",
        "",
        report.markdown_table(
            ["metric", "value"],
            [[report.METRIC_LABELS[name], summary.metrics[name]] for name in ALL_METRICS],
        ),
    ]
    if summary.hard_predictions:
        lines += [
            "",
            "Predictions are labels without probabilities: cross-entropy, KL and the entropy "
            "correlations are not defined and print as n/a.",
        ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="softeval",
        description="Score predictions against the distribution of several annotators' labels.",
    )
    parser.add_argument("--version", action="version", version=f"softeval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="metrics of one model against the human distribution")
    score.add_argument("human", help="annotations: JSONL or CSV")
    score.add_argument("predictions", help="predictions: JSONL or CSV")
    _add_input_options(score)
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = args.func(args)
    except (io.InputError, FileNotFoundError) as exc:
        print(f"softeval: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

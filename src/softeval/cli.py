"""Command line: ``softeval score | ceiling | compare | bands | simulate``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from softeval import __version__, bands, bootstrap, ceiling, finite, io, report
from softeval import simulate as sim
from softeval.metrics import ALL_METRICS, aggregate, is_hard, item_metrics, summarise
from softeval.sampling import subsample

HARD_NOTE = (
    "Predictions are labels without probabilities: cross-entropy, KL and the entropy "
    "correlations are not defined and print as n/a."
)
BAND_COLUMNS = (
    "cross_entropy",
    "kl",
    "js_divergence",
    "total_variation",
    "annotator_match",
    "majority_accuracy",
)
SHORT = {
    "cross_entropy": "cross-entropy",
    "kl": "KL",
    "js_divergence": "JS div.",
    "js_distance": "JS dist.",
    "total_variation": "TV",
    "annotator_match": "match",
    "majority_accuracy": "majority acc.",
    "entropy_pearson": "entropy r",
    "entropy_spearman": "entropy rho",
}


def _add_human_options(parser: argparse.ArgumentParser) -> None:
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
        "--annotations",
        type=int,
        metavar="M",
        help="keep only M randomly chosen annotators per item before scoring",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for every random draw")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="output format")


def _add_prediction_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="score only the items present in both files instead of failing",
    )


def _add_bootstrap_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        metavar="N",
        help="bootstrap resamples over items (default 1000; 0 turns intervals off)",
    )
    parser.add_argument("--level", type=float, default=0.95, help="interval level (default 0.95)")


def _thin(counts: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.annotations is None:
        return counts
    if args.annotations < 1:
        raise io.InputError("--annotations must be at least 1")
    return subsample(counts, args.annotations, np.random.default_rng(args.seed))


def _load(args: argparse.Namespace, *paths: str) -> tuple[io.HumanLabels, np.ndarray, list]:
    human = io.read_human(args.human, labels=args.labels)
    ids = list(human.ids)
    loaded = []
    for path in paths:
        predictions = io.read_predictions(path, human.labels)
        shared = set(predictions.ids)
        missing = [item for item in ids if item not in shared]
        extra = len(shared) - (len(ids) - len(missing))
        if (missing or extra) and not args.allow_missing:
            raise io.InputError(
                f"{path}: {len(missing)} annotated items have no prediction and {extra} "
                "predictions have no annotations; pass --allow-missing to score the shared items"
            )
        ids = [item for item in ids if item in shared]
        loaded.append(predictions)
    if not ids:
        raise io.InputError("no item ids are shared between the files")
    position = {item: i for i, item in enumerate(human.ids)}
    counts = _thin(human.counts[[position[item] for item in ids]], args)
    probs = []
    for predictions in loaded:
        where = {item: i for i, item in enumerate(predictions.ids)}
        probs.append(predictions.probs[[where[item] for item in ids]])
    return human, counts, probs


def _whole(counts: np.ndarray) -> bool:
    return bool(np.allclose(counts, np.rint(counts)))


def _level_label(level: float) -> str:
    return f"{100 * level:g}% interval"


def cmd_score(args: argparse.Namespace) -> str:
    human, counts, (probs,) = _load(args, args.predictions)
    summary = summarise(counts, probs, ties=args.ties)
    items = item_metrics(counts, probs, ties=args.ties)
    intervals = None
    if args.bootstrap > 0:
        intervals = bootstrap.bootstrap_metrics(
            items,
            hard=summary.hard_predictions,
            n_boot=args.bootstrap,
            level=args.level,
            seed=args.seed,
        )
    stats = finite.annotation_stats(counts)
    corrected = None
    if _whole(counts):
        corrected = finite.corrections(counts, probs, n_splits=args.split_draws, seed=args.seed)
    if args.format == "json":
        payload = {"command": "score", "labels": human.labels, **summary.to_dict()}
        payload["annotations_per_item"] = stats
        if intervals is not None:
            payload["level"] = args.level
            payload["intervals"] = {
                name: {"low": iv.low, "high": iv.high} for name, iv in intervals.items()
            }
        if corrected is not None:
            payload["finite_annotation"] = corrected
        return report.to_json(payload)
    lines = [
        f"Items scored: {summary.n_items} (no annotations: {summary.n_empty}; "
        f"tied majority: {summary.n_tied}, scored as '{summary.ties}')",
        f"Annotations per item: min {stats['min']:g}, median {stats['median']:g}, "
        f"max {stats['max']:g}",
        "",
    ]
    if intervals is None:
        lines.append(
            report.markdown_table(
                ["metric", "value"],
                [[report.METRIC_LABELS[name], summary.metrics[name]] for name in ALL_METRICS],
            )
        )
    else:
        lines.append(
            report.markdown_table(
                ["metric", "value", _level_label(args.level)],
                [
                    [
                        report.METRIC_LABELS[name],
                        summary.metrics[name],
                        report.interval(intervals[name].low, intervals[name].high),
                    ]
                    for name in ALL_METRICS
                ],
            )
        )
    if summary.hard_predictions:
        lines += ["", HARD_NOTE]
    if corrected is not None:
        lines += [
            "",
            "Divergences corrected for the number of annotations (the plug-in value is too "
            "large when annotations are few):",
            "",
            report.markdown_table(
                ["metric", "plug-in", "jackknife", "split-half", "Miller-Madow", "estimated bias"],
                [
                    [
                        report.METRIC_LABELS[name],
                        row["plug_in"],
                        row["jackknife"],
                        row["split_half"],
                        row.get("miller_madow", ""),
                        row["estimated_bias"],
                    ]
                    for name, row in corrected.items()
                ],
            ),
        ]
    return "\n".join(lines)


def cmd_ceiling(args: argparse.Namespace) -> str:
    human = io.read_human(args.human, labels=args.labels)
    counts = _thin(human.counts, args)
    if not _whole(counts):
        raise io.InputError("ceilings need whole annotation counts")
    one = ceiling.leave_one_out(counts, ties=args.ties)
    fresh = ceiling.fresh_sample(
        counts, annotations=args.fresh, n_draws=args.draws, seed=args.seed, ties=args.ties
    )
    halves = ceiling.half_against_half(
        counts, n_splits=args.draws, seed=args.seed, ties=args.ties, smoothing=args.smoothing
    )
    stats = finite.annotation_stats(counts)
    n_two = int((counts.sum(axis=1) >= 2).sum())
    if args.format == "json":
        return report.to_json(
            {
                "command": "ceiling",
                "labels": human.labels,
                "n_items": int((counts.sum(axis=1) > 0).sum()),
                "n_items_with_two_or_more": n_two,
                "annotations_per_item": stats,
                "fresh_annotators": args.fresh,
                "draws": args.draws,
                "one_annotator_vs_others": one,
                "full_distribution_vs_fresh_sample": fresh,
                "half_vs_half": halves,
            }
        )
    fresh_label = "as many as the item has" if args.fresh is None else str(args.fresh)
    lines = [
        f"Items: {int((counts.sum(axis=1) > 0).sum())} annotated, {n_two} with two or more "
        "annotations",
        f"Annotations per item: min {stats['min']:g}, median {stats['median']:g}, "
        f"max {stats['max']:g}",
        f"Fresh annotators per item: {fresh_label}; draws: {args.draws}",
        "",
        report.markdown_table(
            [
                "metric",
                "one annotator vs the others",
                "full distribution vs fresh sample",
                "half vs half",
            ],
            [
                [report.METRIC_LABELS[name], one[name], fresh[name], halves[name]]
                for name in ALL_METRICS
            ],
        ),
    ]
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> str:
    human, counts, (probs_a, probs_b) = _load(args, args.predictions_a, args.predictions_b)
    items_a = item_metrics(counts, probs_a, ties=args.ties)
    items_b = item_metrics(counts, probs_b, ties=args.ties)
    result = bootstrap.paired_bootstrap(
        items_a,
        items_b,
        hard_a=is_hard(probs_a),
        hard_b=is_hard(probs_b),
        n_boot=max(args.bootstrap, 1),
        level=args.level,
        seed=args.seed,
    )
    n_items = int((counts.sum(axis=1) > 0).sum())
    if args.format == "json":
        return report.to_json(
            {
                "command": "compare",
                "labels": human.labels,
                "n_items": n_items,
                "level": args.level,
                "bootstrap": max(args.bootstrap, 1),
                "metrics": {name: diff.to_dict() for name, diff in result.items()},
            }
        )
    lines = [
        f"Items scored: {n_items}; paired bootstrap over items, {max(args.bootstrap, 1)} resamples",
        "A = " + args.predictions_a,
        "B = " + args.predictions_b,
        "",
        report.markdown_table(
            ["metric", "A", "B", "A - B", _level_label(args.level), "share of resamples A better"],
            [
                [
                    report.METRIC_LABELS[name],
                    d.a,
                    d.b,
                    d.diff,
                    report.interval(d.low, d.high),
                    d.p_a_better,
                ]
                for name, d in result.items()
            ],
        ),
    ]
    return "\n".join(lines)


def cmd_bands(args: argparse.Namespace) -> str:
    human, counts, (probs,) = _load(args, args.predictions)
    edges = None
    if args.edges:
        try:
            edges = [float(part) for part in args.edges.split(",")]
        except ValueError as exc:
            raise io.InputError("--edges must be comma-separated numbers") from exc
    try:
        rows = bands.score_bands(counts, probs, by=args.by, edges=edges, ties=args.ties)
    except ValueError as exc:
        raise io.InputError(str(exc)) from exc
    overall = aggregate(item_metrics(counts, probs, ties=args.ties), hard=is_hard(probs))
    n_items = int((counts.sum(axis=1) > 0).sum())
    what = (
        "share of annotators on the most common label"
        if args.by == "share"
        else "entropy of the human distribution divided by log K"
    )
    if args.format == "json":
        return report.to_json(
            {
                "command": "bands",
                "labels": human.labels,
                "by": args.by,
                "bands": [band.to_dict() for band in rows],
                "all": {"n_items": n_items, "metrics": overall},
            }
        )
    table = [
        [band.label, band.n_items, *[band.metrics[name] for name in BAND_COLUMNS]] for band in rows
    ]
    table.append(["all", n_items, *[overall[name] for name in BAND_COLUMNS]])
    return "\n".join(
        [
            f"Bands by {what}",
            "",
            report.markdown_table(
                ["band", "items", *[SHORT[name] for name in BAND_COLUMNS]], table
            ),
        ]
    )


def cmd_simulate(args: argparse.Namespace) -> str:
    try:
        sizes = [int(part) for part in args.annotations.split(",")]
    except ValueError as exc:
        raise io.InputError("--annotations must be comma-separated whole numbers") from exc
    try:
        result = sim.simulate(
            n_labels=args.labels,
            n_items=args.items,
            annotations=sizes,
            replicates=args.replicates,
            alpha=args.alpha,
            noise=args.noise,
            seed=args.seed,
        )
    except ValueError as exc:
        raise io.InputError(str(exc)) from exc
    sizes = sorted(set(sizes))
    keys: list[tuple[str, str]] = []
    for row in result.rows:
        if (row.metric, row.estimator) not in keys:
            keys.append((row.metric, row.estimator))
    order = {name: i for i, name in enumerate(ALL_METRICS)}
    keys.sort(key=lambda key: order[key[0]])
    trusted = {
        f"{metric}/{estimator}": result.trusted_from(metric, estimator, args.tolerance)
        for metric, estimator in keys
    }
    if args.format == "json":
        return report.to_json(
            {
                "command": "simulate",
                **result.to_dict(),
                "tolerance": args.tolerance,
                "trusted_from": trusted,
            }
        )
    lookup = {(r.metric, r.estimator, r.annotations): r for r in result.rows}
    table = []
    for metric, estimator in keys:
        start = trusted[f"{metric}/{estimator}"]
        table.append(
            [
                report.METRIC_LABELS[metric],
                estimator,
                result.truth[metric],
                *[lookup[(metric, estimator, n)].bias for n in sizes],
                "not in this run" if start is None else start,
            ]
        )
    pct = f"{100 * args.tolerance:g}%"
    return "\n".join(
        [
            f"Simulation: {args.items} items, {args.labels} labels, "
            f"Dirichlet alpha {args.alpha:g}, model noise {args.noise:g}, "
            f"{args.replicates} replicates, seed {args.seed}",
            "Bias = mean estimate minus the value against the population distribution.",
            "",
            report.markdown_table(
                [
                    "metric",
                    "estimator",
                    "truth",
                    *[f"bias at {n}" for n in sizes],
                    f"within {pct} from",
                ],
                table,
            ),
        ]
    )


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
    _add_human_options(score)
    _add_prediction_options(score)
    _add_bootstrap_options(score)
    score.add_argument(
        "--split-draws", type=int, default=20, help="random halves for the split-half estimate"
    )
    score.set_defaults(func=cmd_score)

    ceil = sub.add_parser("ceiling", help="what annotators score against each other")
    ceil.add_argument("human", help="annotations: JSONL or CSV")
    _add_human_options(ceil)
    ceil.add_argument(
        "--fresh",
        type=int,
        metavar="M",
        help="fresh annotators drawn per item (default: as many as the item has)",
    )
    ceil.add_argument("--draws", type=int, default=200, help="random draws to average over")
    ceil.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="count added to every label of the predicting half (default 0.5)",
    )
    ceil.set_defaults(func=cmd_ceiling)

    compare = sub.add_parser("compare", help="paired comparison of two models")
    compare.add_argument("human", help="annotations: JSONL or CSV")
    compare.add_argument("predictions_a", help="predictions of model A")
    compare.add_argument("predictions_b", help="predictions of model B")
    _add_human_options(compare)
    _add_prediction_options(compare)
    _add_bootstrap_options(compare)
    compare.set_defaults(func=cmd_compare)

    band = sub.add_parser("bands", help="metrics by human-agreement band")
    band.add_argument("human", help="annotations: JSONL or CSV")
    band.add_argument("predictions", help="predictions: JSONL or CSV")
    _add_human_options(band)
    _add_prediction_options(band)
    band.add_argument(
        "--by",
        choices=("share", "entropy"),
        default="share",
        help="band by majority share (default) or by normalised entropy",
    )
    band.add_argument("--edges", help="comma-separated band edges, e.g. 0,0.5,0.7,0.9,1")
    band.set_defaults(func=cmd_bands)

    simulate = sub.add_parser(
        "simulate", help="bias of each estimate at a given number of annotations"
    )
    simulate.add_argument("--labels", type=int, default=3, help="number of labels (default 3)")
    simulate.add_argument("--items", type=int, default=2000, help="number of items (default 2000)")
    simulate.add_argument(
        "--annotations",
        default="3,5,10,100",
        help="annotations per item to try (default 3,5,10,100)",
    )
    simulate.add_argument("--replicates", type=int, default=20, help="repeated draws per setting")
    simulate.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration of the human distributions",
    )
    simulate.add_argument(
        "--noise", type=float, default=0.3, help="share of noise mixed into the model's prediction"
    )
    simulate.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="relative bias treated as small enough (default 0.05)",
    )
    simulate.add_argument("--seed", type=int, default=0, help="seed for every random draw")
    simulate.add_argument("--format", choices=("md", "json"), default="md", help="output format")
    simulate.set_defaults(func=cmd_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = args.func(args)
    except (ValueError, OSError) as exc:
        print(f"softeval: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

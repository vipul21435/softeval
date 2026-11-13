"""Print the ChaosNLI tables shown in the README from the files written by prepare.py.

    python examples/chaosnli/tables.py PREPARED_DIR [--draws 20] [--seed 0]

For every subset: each predictor under the single-label view and the distribution view, the human
ceilings, and what happens to the numbers when only 5 of the 100 annotations are kept (averaged
over ``--draws`` random choices of the 5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from softeval import ceiling, io
from softeval.finite import jackknife_items
from softeval.metrics import aggregate, is_hard, item_metrics
from softeval.report import markdown_table
from softeval.sampling import subsample

SUBSETS = ("snli", "mnli_m", "alphanli")
VIEW = ("majority_accuracy", "annotator_match", "total_variation", "js_divergence", "cross_entropy")


def ranks(values: list[float], higher_is_better: bool) -> list[int]:
    """Rank 1 is best; equal values share a rank."""
    signed = [-v if higher_is_better else v for v in values]
    return [1 + sum(other < v - 1e-12 for other in signed) for v in signed]


def load(folder: Path):
    human = io.read_human(folder / "human.jsonl")
    old = io.read_human(folder / "old_label.jsonl", labels=human.labels)
    predictors = {}
    for path in sorted(folder.glob("pred_*.jsonl")):
        predictions = io.read_predictions(path, human.labels)
        counts, probs, _ = io.align(human, predictions)
        old_counts, _, _ = io.align(old, predictions)
        predictors[path.stem[len("pred_") :]] = (counts, old_counts, probs)
    return human, predictors


def two_views(predictors) -> str:
    names = list(predictors)
    rows = []
    for name in names:
        counts, old_counts, probs = predictors[name]
        hard = is_hard(probs)
        new = aggregate(item_metrics(counts, probs), hard=hard)
        old = aggregate(item_metrics(old_counts, probs), hard=hard)
        rows.append([name, old["majority_accuracy"], *[new[m] for m in VIEW]])
    by_acc = ranks([row[2] for row in rows], higher_is_better=True)
    by_js = ranks([row[5] for row in rows], higher_is_better=False)
    table = [[*row, a, j] for row, a, j in zip(rows, by_acc, by_js, strict=True)]
    return markdown_table(
        [
            "predictor",
            "acc. vs old label",
            "acc. vs majority of 100",
            "match",
            "TV",
            "JS div.",
            "cross-entropy",
            "rank by acc.",
            "rank by JS div.",
        ],
        table,
    )


def ceilings(counts: np.ndarray, draws: int, seed: int) -> str:
    one = ceiling.leave_one_out(counts)
    fresh = ceiling.fresh_sample(counts, n_draws=draws, seed=seed)
    fresh5 = ceiling.fresh_sample(counts, annotations=5, n_draws=draws, seed=seed)
    halves = ceiling.half_against_half(counts, n_splits=draws, seed=seed)
    return markdown_table(
        ["human reference", "acc. vs majority", "match", "TV", "JS div.", "cross-entropy"],
        [
            ["one annotator vs the other 99", *[one[m] for m in VIEW]],
            ["50 annotators vs the other 50", *[halves[m] for m in VIEW]],
            ["full distribution vs 100 fresh annotators", *[fresh[m] for m in VIEW]],
            ["full distribution vs 5 fresh annotators", *[fresh5[m] for m in VIEW]],
        ],
    )


def five_of_hundred(predictors, draws: int, seed: int) -> str:
    rows = []
    for name, (counts, _old, probs) in predictors.items():
        hard = is_hard(probs)
        full = aggregate(item_metrics(counts, probs), hard=hard)
        rng = np.random.default_rng(seed)
        sums = dict.fromkeys(
            ("majority_accuracy", "annotator_match", "tv", "tv_jack", "js", "js_jack"), 0.0
        )
        for _ in range(draws):
            few = subsample(counts, 5, rng)
            plug = aggregate(item_metrics(few, probs), hard=hard)
            jack = jackknife_items(few, probs)
            sums["majority_accuracy"] += plug["majority_accuracy"]
            sums["annotator_match"] += plug["annotator_match"]
            sums["tv"] += plug["total_variation"]
            sums["js"] += plug["js_divergence"]
            sums["tv_jack"] += float(np.mean(jack["total_variation"]))
            sums["js_jack"] += float(np.mean(jack["js_divergence"]))
        mean = {key: value / draws for key, value in sums.items()}
        rows.append(
            [
                name,
                full["majority_accuracy"],
                mean["majority_accuracy"],
                full["annotator_match"],
                mean["annotator_match"],
                full["total_variation"],
                mean["tv"],
                mean["tv_jack"],
                full["js_divergence"],
                mean["js"],
                mean["js_jack"],
            ]
        )
    return markdown_table(
        [
            "predictor",
            "acc. (100)",
            "acc. (5)",
            "match (100)",
            "match (5)",
            "TV (100)",
            "TV (5) plug-in",
            "TV (5) jackknife",
            "JS (100)",
            "JS (5) plug-in",
            "JS (5) jackknife",
        ],
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prepared", help="folder written by prepare.py")
    parser.add_argument("--draws", type=int, default=20, help="random draws to average over")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for subset in SUBSETS:
        folder = Path(args.prepared) / subset
        if not folder.exists():
            continue
        human, predictors = load(folder)
        counts = human.counts
        per_item = sorted({int(n) for n in counts.sum(axis=1)})
        print(f"### {subset}: {len(human)} items, labels {human.labels}, annotations {per_item}")
        print()
        print(two_views(predictors))
        print()
        print(ceilings(counts, args.draws, args.seed))
        print()
        print(five_of_hundred(predictors, args.draws, args.seed))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

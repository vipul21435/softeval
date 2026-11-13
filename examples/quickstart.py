"""Score the toy predictions with the Python API.

    python examples/quickstart.py

The files in examples/toy are made up: eight items, three labels, up to five annotators each.
"""

from __future__ import annotations

from pathlib import Path

from softeval import ceiling, finite, io
from softeval.bootstrap import bootstrap_metrics, paired_bootstrap
from softeval.metrics import item_metrics, summarise

TOY = Path(__file__).resolve().parent / "toy"
LABELS = ["entailment", "neutral", "contradiction"]


def main() -> None:
    human = io.read_human(TOY / "human.jsonl", labels=LABELS)
    model_a = io.read_predictions(TOY / "model_a.jsonl", human.labels)
    model_b = io.read_predictions(TOY / "model_b.jsonl", human.labels)
    counts, probs_a, _ = io.align(human, model_a)
    _, probs_b, _ = io.align(human, model_b)

    summary = summarise(counts, probs_a)
    print(f"items scored: {summary.n_items}, without annotations: {summary.n_empty}")
    intervals = bootstrap_metrics(item_metrics(counts, probs_a), n_boot=500, seed=0)
    for name in ("total_variation", "annotator_match", "majority_accuracy"):
        iv = intervals[name]
        print(f"model A {name}: {iv.value:.4f}  [{iv.low:.4f}, {iv.high:.4f}]")

    corrected = finite.corrections(counts, probs_a)["total_variation"]
    print(
        f"model A total variation: plug-in {corrected['plug_in']:.4f}, "
        f"jackknife {corrected['jackknife']:.4f}"
    )

    one = ceiling.leave_one_out(counts)
    floor = ceiling.fresh_sample(counts, n_draws=200, seed=0)
    print(f"one annotator vs the others, match: {one['annotator_match']:.4f}")
    print(f"full distribution vs fresh annotators, total variation: {floor['total_variation']:.4f}")

    diff = paired_bootstrap(
        item_metrics(counts, probs_a), item_metrics(counts, probs_b), n_boot=500, seed=0
    )["total_variation"]
    print(
        f"A minus B, total variation: {diff.diff:.4f}  [{diff.low:.4f}, {diff.high:.4f}], "
        f"A better in {100 * diff.p_a_better:.0f}% of resamples"
    )


if __name__ == "__main__":
    main()

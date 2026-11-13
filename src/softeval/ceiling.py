"""Human ceilings: what people score on the same metrics.

Three reference points, all computed from the annotations alone:

- one annotator against the others: each annotator's label is scored as a prediction against the
  labels of the remaining annotators of that item (leave-one-out, averaged over annotators);
- the full distribution against a fresh sample: the empirical distribution of all annotators is
  scored against ``m`` new annotators drawn from it, which is what a model that predicted the human
  distribution exactly would get at ``m`` annotations. That is optimal in expectation for
  cross-entropy, match and accuracy; for the other distances it is a reference point, since against
  very few annotators a sharper prediction can score lower;
- half against half: the distribution of a random half of the annotators is scored against the
  other half.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from softeval.metrics import (
    DEFAULT_EPS,
    FloatArray,
    TiePolicy,
    aggregate,
    distributions,
    item_metrics,
    one_hot,
)
from softeval.sampling import integer_counts, sample_annotators, split_halves

_COLUMNS = (
    "cross_entropy",
    "kl",
    "js_divergence",
    "js_distance",
    "total_variation",
    "annotator_match",
    "majority_accuracy",
    "human_entropy",
    "model_entropy",
)


def leave_one_out_items(counts: ArrayLike, *, ties: TiePolicy = "split") -> dict[str, FloatArray]:
    """Per-item expected score of one annotator against the item's other annotators.

    Items with fewer than two annotations give ``nan``. Cross-entropy and KL are ``nan``
    throughout: a single label is a prediction with no probabilities.
    """
    c = integer_counts(counts).astype(np.float64)
    n = c.sum(axis=1)
    n_items, n_labels = c.shape
    total = {name: np.zeros(n_items) for name in _COLUMNS}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(n_labels):
            has = c[:, k] > 0
            reduced = c.copy()
            reduced[has, k] -= 1.0
            prediction = one_hot(np.full(n_items, k), n_labels)
            values = item_metrics(reduced, prediction, ties=ties)
            weight = c[:, k] / n
            for name in _COLUMNS:
                total[name] += np.where(has, weight * values[name], 0.0)
    out = {name: np.where(n >= 2, column, np.nan) for name, column in total.items()}
    out["cross_entropy"] = np.full(n_items, np.nan)
    out["kl"] = np.full(n_items, np.nan)
    return out


def leave_one_out(counts: ArrayLike, *, ties: TiePolicy = "split") -> dict[str, float]:
    """Mean score of one annotator against the others, over items with two or more annotations."""
    return aggregate(leave_one_out_items(counts, ties=ties), hard=True)


def _floor(dist: FloatArray, eps: float) -> FloatArray:
    """Raise zero probabilities to ``eps`` and renormalise, so a fresh label is never impossible."""
    raised = np.maximum(dist, eps)
    return raised / raised.sum(axis=1, keepdims=True)


def fresh_sample(
    counts: ArrayLike,
    *,
    annotations: int | None = None,
    n_draws: int = 200,
    seed: int | np.random.Generator = 0,
    ties: TiePolicy = "split",
    eps: float = DEFAULT_EPS,
) -> dict[str, float]:
    """Score the full human distribution against freshly drawn annotators.

    ``annotations`` is the number of fresh annotators per item; by default each item gets as many
    as it has. The result is the mean over ``n_draws`` draws and over items.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    c = integer_counts(counts).astype(np.float64)
    keep = c.sum(axis=1) > 0
    c = c[keep]
    dist = distributions(c)
    prediction = _floor(dist, eps)
    sizes = c.sum(axis=1).astype(np.int64) if annotations is None else annotations
    totals: dict[str, float] = {}
    for _ in range(n_draws):
        fresh = sample_annotators(dist, sizes, rng)
        values = aggregate(item_metrics(fresh, prediction, ties=ties, eps=eps))
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value
    return {name: value / n_draws for name, value in totals.items()}


def half_against_half(
    counts: ArrayLike,
    *,
    n_splits: int = 200,
    seed: int | np.random.Generator = 0,
    ties: TiePolicy = "split",
    smoothing: float = 0.5,
    eps: float = DEFAULT_EPS,
) -> dict[str, float]:
    """Score the distribution of one random half of the annotators against the other half.

    Items with fewer than two annotations are left out. The predicting half is smoothed by adding
    ``smoothing`` to every count before normalising, so that a label it never saw does not make
    the cross-entropy explode; set it to 0 for the raw half.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    c = integer_counts(counts).astype(np.float64)
    c = c[c.sum(axis=1) >= 2]
    totals: dict[str, float] = {}
    for _ in range(n_splits):
        first, second = split_halves(c, rng)
        prediction = _floor(distributions(first + smoothing), eps)
        values = aggregate(item_metrics(second, prediction, ties=ties, eps=eps))
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value
    return {name: value / n_splits for name, value in totals.items()}

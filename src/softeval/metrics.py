"""Per-item metrics of a predicted distribution against the human label distribution.

Every function takes arrays with one row per item and one column per label and returns one value
per item. ``counts`` are the numbers of annotators who chose each label; ``probs`` are the model's
probabilities. Rows with no annotations give ``nan`` and are skipped by :func:`summarise`.

Logarithms are natural, so entropies and divergences are in nats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
TiePolicy = Literal["split", "any", "exclude"]

#: Probabilities below this are raised to it before taking a logarithm in the cross-entropy.
DEFAULT_EPS = 1e-12
#: Two counts (or probabilities) closer than this are treated as tied.
TIE_TOL = 1e-9

MEAN_METRICS = (
    "cross_entropy",
    "kl",
    "js_divergence",
    "js_distance",
    "total_variation",
    "annotator_match",
    "majority_accuracy",
)
CORRELATION_METRICS = ("entropy_pearson", "entropy_spearman")
ALL_METRICS = MEAN_METRICS + CORRELATION_METRICS
#: Metrics where a larger value is better; the others are distances.
HIGHER_IS_BETTER = frozenset(
    {"annotator_match", "majority_accuracy", "entropy_pearson", "entropy_spearman"}
)


def as_counts(counts: ArrayLike) -> FloatArray:
    """Validate a matrix of label counts (items by labels) and return it as floats."""
    arr = np.asarray(counts, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"counts must be 2-D (items by labels), got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError("counts need at least two label columns")
    if not np.all(np.isfinite(arr)):
        raise ValueError("counts must be finite")
    if np.any(arr < 0):
        raise ValueError("counts must not be negative")
    return arr


def as_probs(probs: ArrayLike, *, atol: float = 1e-6) -> FloatArray:
    """Validate a matrix of predicted probabilities; rows must sum to one within ``atol``."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"probs must be 2-D (items by labels), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("probs must be finite")
    if np.any(arr < 0):
        raise ValueError("probs must not be negative")
    sums = arr.sum(axis=1)
    bad = np.flatnonzero(np.abs(sums - 1.0) > atol)
    if bad.size:
        raise ValueError(
            f"probs rows must sum to 1; row {int(bad[0])} sums to {sums[bad[0]]:.6g} "
            f"({bad.size} such rows)"
        )
    return arr / sums[:, None]


def distributions(counts: ArrayLike) -> FloatArray:
    """Turn counts into empirical distributions; rows with no annotations become ``nan``."""
    arr = as_counts(counts)
    totals = arr.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, arr / totals, np.nan)


def one_hot(label_index: ArrayLike, n_labels: int) -> FloatArray:
    """Turn predicted label indices into probability rows with all mass on one label."""
    idx = np.asarray(label_index, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("label_index must be 1-D")
    if idx.size and (idx.min() < 0 or idx.max() >= n_labels):
        raise ValueError("label index out of range")
    out = np.zeros((idx.size, n_labels), dtype=np.float64)
    out[np.arange(idx.size), idx] = 1.0
    return out


def softmax(logits: ArrayLike) -> FloatArray:
    """Row-wise softmax, for predictions published as logits."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("logits must be 2-D (items by labels)")
    shifted = arr - arr.max(axis=1, keepdims=True)
    expd = np.exp(shifted)
    return expd / expd.sum(axis=1, keepdims=True)


def _xlogy(x: FloatArray, y: FloatArray) -> FloatArray:
    """``x * log(y)`` with the convention ``0 * log(0) = 0``; ``nan`` in ``x`` stays ``nan``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x * np.log(np.where(x > 0, y, 1.0))
    return np.where(x == 0, 0.0, out)


def entropy(p: ArrayLike) -> FloatArray:
    """Shannon entropy of each row in nats."""
    arr = np.asarray(p, dtype=np.float64)
    return -_xlogy(arr, arr).sum(axis=1)


def cross_entropy(human: ArrayLike, probs: ArrayLike, *, eps: float = DEFAULT_EPS) -> FloatArray:
    """``-sum_k h_k log p_k`` per item, with ``p_k`` raised to ``eps`` where it is smaller."""
    h = np.asarray(human, dtype=np.float64)
    p = np.maximum(np.asarray(probs, dtype=np.float64), eps)
    return -_xlogy(h, p).sum(axis=1)


def kl_divergence(human: ArrayLike, probs: ArrayLike, *, eps: float = DEFAULT_EPS) -> FloatArray:
    """``KL(h || p)`` per item: the cross-entropy minus the entropy of ``h``."""
    return cross_entropy(human, probs, eps=eps) - entropy(human)


def js_divergence(human: ArrayLike, probs: ArrayLike) -> FloatArray:
    """Jensen-Shannon divergence per item in nats (between 0 and ``log 2``)."""
    h = np.asarray(human, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    mid = 0.5 * (h + p)
    out = 0.5 * (_xlogy(h, h) - _xlogy(h, mid)).sum(axis=1)
    out += 0.5 * (_xlogy(p, p) - _xlogy(p, mid)).sum(axis=1)
    return np.maximum(out, 0.0)


def js_distance(human: ArrayLike, probs: ArrayLike) -> FloatArray:
    """Square root of the Jensen-Shannon divergence (what ``scipy`` calls ``jensenshannon``)."""
    return np.sqrt(js_divergence(human, probs))


def total_variation(human: ArrayLike, probs: ArrayLike) -> FloatArray:
    """Total variation distance per item: ``0.5 * sum_k |h_k - p_k|``."""
    h = np.asarray(human, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    return 0.5 * np.abs(h - p).sum(axis=1)


def _top_set(values: FloatArray) -> NDArray[np.bool_]:
    """Boolean mask of the entries tied for the row maximum."""
    top = values.max(axis=1, keepdims=True)
    return values >= top - TIE_TOL * np.maximum(1.0, np.abs(top))


def annotator_match(human: ArrayLike, probs: ArrayLike) -> FloatArray:
    """Probability that the model's top answer equals the label of a randomly drawn annotator.

    This is ``h`` at the model's argmax. If several labels tie for the model's maximum, the answer
    is taken to be drawn uniformly among them.
    """
    h = np.asarray(human, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    pred = _top_set(p)
    return (h * pred).sum(axis=1) / pred.sum(axis=1)


def majority_accuracy(
    counts: ArrayLike, probs: ArrayLike, *, ties: TiePolicy = "split"
) -> FloatArray:
    """Credit per item for matching the majority label, with ties among annotators made explicit.

    ``ties`` decides what a tied majority means:

    - ``"split"``: the gold label is drawn uniformly from the tied labels, so a prediction inside
      a two-way tie earns 0.5;
    - ``"any"``: any of the tied labels counts as correct;
    - ``"exclude"``: tied items give ``nan`` and are left out of the mean.

    Ties in the model's own probabilities are always broken uniformly at random.
    """
    if ties not in ("split", "any", "exclude"):
        raise ValueError(f"unknown tie policy {ties!r}")
    c = as_counts(counts)
    p = np.asarray(probs, dtype=np.float64)
    gold = _top_set(c)
    pred = _top_set(p)
    n_gold = gold.sum(axis=1)
    overlap = (gold & pred).sum(axis=1) / pred.sum(axis=1)
    if ties == "split":
        credit = overlap / n_gold
    elif ties == "any":
        credit = overlap.astype(np.float64)
    else:
        credit = np.where(n_gold > 1, np.nan, overlap)
    return np.where(c.sum(axis=1) > 0, credit, np.nan)


def majority_ties(counts: ArrayLike) -> NDArray[np.bool_]:
    """Mask of annotated items whose majority label is tied."""
    c = as_counts(counts)
    return (_top_set(c).sum(axis=1) > 1) & (c.sum(axis=1) > 0)


def miller_madow_entropy(counts: ArrayLike) -> FloatArray:
    """Plug-in entropy plus the Miller-Madow term ``(m - 1) / (2 n)``.

    ``m`` is the number of labels with a non-zero count and ``n`` the number of annotations.
    """
    c = as_counts(counts)
    n = c.sum(axis=1)
    m = (c > 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        correction = np.where(n > 0, (m - 1) / (2.0 * n), np.nan)
    return entropy(distributions(c)) + correction


def rankdata(values: ArrayLike) -> FloatArray:
    """Ranks starting at 1, with tied values given the average of their ranks."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    position = np.empty(v.size, dtype=np.int64)
    position[order] = np.arange(v.size)
    sorted_v = v[order]
    new_group = np.concatenate(([True], sorted_v[1:] != sorted_v[:-1]))
    group = np.cumsum(new_group)[position]
    starts = np.concatenate((np.flatnonzero(new_group), [v.size]))
    return 0.5 * (starts[group] + starts[group - 1] + 1)


def pearson(x: ArrayLike, y: ArrayLike) -> float:
    """Pearson correlation; ``nan`` when either side is constant or there are under two points."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.clip((a * b).sum() / denom, -1.0, 1.0))


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman rank correlation with average ranks for ties."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def item_metrics(
    counts: ArrayLike,
    probs: ArrayLike,
    *,
    ties: TiePolicy = "split",
    eps: float = DEFAULT_EPS,
) -> dict[str, FloatArray]:
    """All per-item metrics, plus the human and model entropies used for the correlation.

    Items with no annotations get ``nan`` in every column.
    """
    c = as_counts(counts)
    p = as_probs(probs)
    if c.shape != p.shape:
        raise ValueError(f"counts {c.shape} and probs {p.shape} must have the same shape")
    h = distributions(c)
    empty = c.sum(axis=1) <= 0
    model_entropy = np.where(empty, np.nan, entropy(p))
    return {
        "cross_entropy": cross_entropy(h, p, eps=eps),
        "kl": kl_divergence(h, p, eps=eps),
        "js_divergence": js_divergence(h, p),
        "js_distance": js_distance(h, p),
        "total_variation": total_variation(h, p),
        "annotator_match": annotator_match(h, p),
        "majority_accuracy": majority_accuracy(c, p, ties=ties),
        "human_entropy": entropy(h),
        "model_entropy": model_entropy,
    }


@dataclass
class Summary:
    """Mean of each metric over the annotated items, with the counts needed to read it."""

    n_items: int
    n_empty: int
    n_tied: int
    ties: str
    hard_predictions: bool
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_items": self.n_items,
            "n_empty": self.n_empty,
            "n_tied": self.n_tied,
            "ties": self.ties,
            "hard_predictions": self.hard_predictions,
            "metrics": {k: _json_float(v) for k, v in self.metrics.items()},
        }


def _json_float(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _nanmean(values: FloatArray) -> float:
    kept = values[~np.isnan(values)]
    return float(kept.mean()) if kept.size else float("nan")


def aggregate(items: dict[str, FloatArray], *, hard: bool = False) -> dict[str, float]:
    """Reduce the columns from :func:`item_metrics` to one number per metric.

    With ``hard`` predictions (a label, no probabilities) the cross-entropy and KL divergence are
    infinite as soon as one annotator disagrees and the model entropy is constant, so those and
    the entropy correlations are returned as ``nan``.
    """
    out = {name: _nanmean(items[name]) for name in MEAN_METRICS}
    keep = ~np.isnan(items["human_entropy"])
    h_ent = items["human_entropy"][keep]
    m_ent = items["model_entropy"][keep]
    out["entropy_pearson"] = pearson(m_ent, h_ent)
    out["entropy_spearman"] = spearman(m_ent, h_ent)
    if hard:
        for name in ("cross_entropy", "kl", "entropy_pearson", "entropy_spearman"):
            out[name] = float("nan")
    return out


def is_hard(probs: ArrayLike) -> bool:
    """True when every row puts all of its mass on a single label."""
    p = np.asarray(probs, dtype=np.float64)
    return bool(p.size) and bool(np.all(p.max(axis=1) >= 1.0 - 1e-12))


def summarise(
    counts: ArrayLike,
    probs: ArrayLike,
    *,
    ties: TiePolicy = "split",
    eps: float = DEFAULT_EPS,
) -> Summary:
    """Score predictions against the human label distribution over all annotated items."""
    c = as_counts(counts)
    items = item_metrics(c, probs, ties=ties, eps=eps)
    hard = is_hard(as_probs(probs))
    return Summary(
        n_items=int((c.sum(axis=1) > 0).sum()),
        n_empty=int((c.sum(axis=1) <= 0).sum()),
        n_tied=int(majority_ties(c).sum()),
        ties=ties,
        hard_predictions=hard,
        metrics=aggregate(items, hard=hard),
    )

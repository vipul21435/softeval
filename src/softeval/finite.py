"""Corrections for scoring against a distribution estimated from a few annotations.

With ``n`` annotations the empirical distribution ``h_hat`` is a noisy estimate of the population
distribution ``h``. Metrics that are linear in ``h`` (cross-entropy, match with a random annotator)
are unbiased. The divergences are not: the plug-in values ``KL(h_hat || p)``, ``JS(h_hat, p)`` and
``TV(h_hat, p)`` are too large on average, by roughly a constant over ``n`` for the smooth ones.

Three estimates are offered, each returning one value per item (the mean over items is what should
be read; single items can come out negative):

- jackknife (leave one annotator out): ``n * D_n - (n - 1) * mean_j D_(-j)``;
- split-half extrapolation: ``(n * D_n - h * D_h) / (n - h)`` with ``h = n // 2`` annotators kept;
- Miller-Madow, for KL only: subtract ``(m - 1) / (2 n)`` with ``m`` labels observed.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from softeval.metrics import (
    DEFAULT_EPS,
    FloatArray,
    TiePolicy,
    as_counts,
    as_probs,
    item_metrics,
)
from softeval.sampling import integer_counts, split_halves

#: Metrics whose plug-in value is biased upwards and for which a correction is computed.
CORRECTABLE = ("kl", "js_divergence", "js_distance", "total_variation")
#: Refuse exact enumeration beyond this many outcomes.
MAX_OUTCOMES = 300_000


def jackknife_items(
    counts: ArrayLike, probs: ArrayLike, *, eps: float = DEFAULT_EPS
) -> dict[str, FloatArray]:
    """Leave-one-annotator-out jackknife of each correctable metric, per item.

    Leaving out one annotator who chose label ``k`` happens for ``c_k`` of the ``n`` annotators, so
    the average over annotators is a weighted sum over labels and needs no sampling. Items with
    fewer than two annotations keep their plug-in value.
    """
    c = integer_counts(counts).astype(np.float64)
    full = item_metrics(c, probs, eps=eps)
    n = c.sum(axis=1)
    left_out = {name: np.zeros(c.shape[0]) for name in CORRECTABLE}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(c.shape[1]):
            has = c[:, k] > 0
            reduced = c.copy()
            reduced[has, k] -= 1.0
            values = item_metrics(reduced, probs, eps=eps)
            weight = c[:, k] / n
            for name in CORRECTABLE:
                left_out[name] += np.where(has, weight * values[name], 0.0)
        return {
            name: np.where(n >= 2, n * full[name] - (n - 1.0) * left_out[name], full[name])
            for name in CORRECTABLE
        }


def split_half_items(
    counts: ArrayLike,
    probs: ArrayLike,
    *,
    n_splits: int = 20,
    seed: int | np.random.Generator = 0,
    eps: float = DEFAULT_EPS,
) -> dict[str, FloatArray]:
    """Split-half extrapolation of each correctable metric, per item.

    The metric is recomputed on a random half of each item's annotators (averaged over
    ``n_splits`` splits) and extrapolated to infinitely many annotators assuming the bias is
    proportional to one over the number of annotations. Items with one annotation keep their
    plug-in value.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    c = integer_counts(counts).astype(np.float64)
    full = item_metrics(c, probs, eps=eps)
    n = c.sum(axis=1)
    half = np.floor(n / 2.0)
    total = {name: np.zeros(c.shape[0]) for name in CORRECTABLE}
    for _ in range(n_splits):
        first, _second = split_halves(c, rng)
        values = item_metrics(first, probs, eps=eps)
        for name in CORRECTABLE:
            total[name] += values[name]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            name: np.where(
                half >= 1, (n * full[name] - half * total[name] / n_splits) / (n - half), full[name]
            )
            for name in CORRECTABLE
        }


def miller_madow_kl(counts: ArrayLike, probs: ArrayLike, *, eps: float = DEFAULT_EPS) -> FloatArray:
    """Plug-in KL minus the first-order entropy bias ``(m - 1) / (2 n)``, per item."""
    c = as_counts(counts)
    n = c.sum(axis=1)
    observed = (c > 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        correction = np.where(n > 0, (observed - 1) / (2.0 * n), np.nan)
    return item_metrics(c, probs, eps=eps)["kl"] - correction


def _nanmean(values: FloatArray) -> float:
    kept = values[~np.isnan(values)]
    return float(kept.mean()) if kept.size else float("nan")


def corrections(
    counts: ArrayLike,
    probs: ArrayLike,
    *,
    n_splits: int = 20,
    seed: int = 0,
    eps: float = DEFAULT_EPS,
) -> dict[str, dict[str, float]]:
    """Plug-in and corrected means over items for each correctable metric.

    ``estimated_bias`` is the plug-in value minus the jackknife value.
    """
    plug_in = item_metrics(counts, probs, eps=eps)
    jack = jackknife_items(counts, probs, eps=eps)
    split = split_half_items(counts, probs, n_splits=n_splits, seed=seed, eps=eps)
    out: dict[str, dict[str, float]] = {}
    for name in CORRECTABLE:
        row = {
            "plug_in": _nanmean(plug_in[name]),
            "jackknife": _nanmean(jack[name]),
            "split_half": _nanmean(split[name]),
        }
        row["estimated_bias"] = row["plug_in"] - row["jackknife"]
        if name == "kl":
            row["miller_madow"] = _nanmean(miller_madow_kl(counts, probs, eps=eps))
        out[name] = row
    return out


def annotation_stats(counts: ArrayLike) -> dict[str, float]:
    """Minimum, median and maximum number of annotations over the annotated items."""
    n = as_counts(counts).sum(axis=1)
    n = n[n > 0]
    if not n.size:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {"min": float(n.min()), "median": float(np.median(n)), "max": float(n.max())}


def compositions(n: int, k: int) -> np.ndarray:
    """All ways to give ``n`` annotations to ``k`` labels, one per row."""
    if n < 0 or k < 1:
        raise ValueError("need n >= 0 and k >= 1")
    if math.comb(n + k - 1, k - 1) > MAX_OUTCOMES:
        raise ValueError(f"too many outcomes to enumerate for n={n}, k={k}")
    if k == 1:
        return np.array([[n]], dtype=np.int64)
    blocks = []
    for first in range(n + 1):
        rest = compositions(n - first, k - 1)
        blocks.append(np.column_stack([np.full(len(rest), first, dtype=np.int64), rest]))
    return np.vstack(blocks)


def multinomial_pmf(outcomes: np.ndarray, dist: ArrayLike) -> FloatArray:
    """Probability of each row of label counts under a multinomial with probabilities ``dist``."""
    h = np.asarray(dist, dtype=np.float64)
    c = np.asarray(outcomes, dtype=np.int64)
    n = int(c[0].sum()) if c.size else 0
    log_factorial = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n + 1)))))
    log_h = np.where(h > 0, np.log(np.where(h > 0, h, 1.0)), -np.inf)
    with np.errstate(invalid="ignore"):
        terms = np.where(c > 0, c * log_h[None, :], 0.0)
    log_p = log_factorial[n] - log_factorial[c].sum(axis=1) + terms.sum(axis=1)
    return np.exp(log_p)


def expected_plugin(
    dist: ArrayLike,
    probs: ArrayLike,
    n: int,
    *,
    ties: TiePolicy = "split",
    eps: float = DEFAULT_EPS,
) -> dict[str, float]:
    """Exact expectation of every plug-in metric for one item with ``n`` annotators.

    ``dist`` is the population distribution the annotators are drawn from and ``probs`` the
    prediction. All multinomial outcomes are enumerated, so this is for a handful of labels.
    Subtracting the metric computed on ``dist`` itself gives the exact bias.
    """
    h = np.asarray(dist, dtype=np.float64)
    p = as_probs(np.asarray(probs, dtype=np.float64)[None, :])
    if h.ndim != 1 or h.size != p.shape[1]:
        raise ValueError("dist and probs must be vectors over the same labels")
    if n < 1:
        raise ValueError("n must be at least 1")
    outcomes = compositions(n, h.size)
    weights = multinomial_pmf(outcomes, h)
    values = item_metrics(outcomes, np.repeat(p, len(outcomes), axis=0), ties=ties, eps=eps)
    return {name: float((weights * column).sum()) for name, column in values.items()}

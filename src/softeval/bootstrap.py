"""Percentile bootstrap over items, for one model and for the difference between two."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from softeval.metrics import (
    ALL_METRICS,
    HIGHER_IS_BETTER,
    MEAN_METRICS,
    FloatArray,
    aggregate,
    rankdata,
)

_CHUNK = 64


@dataclass
class Interval:
    """A point estimate with a bootstrap percentile interval."""

    value: float
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return {"value": self.value, "low": self.low, "high": self.high}


@dataclass
class PairedDifference:
    """Model A minus model B on one metric, with a paired bootstrap interval.

    ``p_a_better`` is the share of resamples in which A is strictly better than B, taking the
    direction of the metric into account (higher match and accuracy, lower distances).
    """

    a: float
    b: float
    diff: float
    low: float
    high: float
    p_a_better: float

    def to_dict(self) -> dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
            "diff": self.diff,
            "low": self.low,
            "high": self.high,
            "p_a_better": self.p_a_better,
        }


def _annotated(items: dict[str, FloatArray]) -> np.ndarray:
    return ~np.isnan(items["human_entropy"])


def _index_chunks(n_items: int, n_boot: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    done = 0
    while done < n_boot:
        size = min(_CHUNK, n_boot - done)
        yield rng.integers(0, n_items, size=(size, n_items))
        done += size


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(values)
    count = mask.sum(axis=1)
    total = np.where(mask, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _row_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    flat = _row_constant(x) | _row_constant(y)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((denom > 0) & ~flat, (xc * yc).sum(axis=1) / denom, np.nan)


def _row_constant(values: np.ndarray) -> np.ndarray:
    spread = values.max(axis=1) - values.min(axis=1)
    return spread <= 1e-12 * np.maximum(1.0, np.abs(values).max(axis=1))


def _resampled(items: dict[str, FloatArray], hard: bool, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Every metric recomputed on each row of resampled item indices."""
    out = {name: _row_nanmean(items[name][idx]) for name in MEAN_METRICS}
    n_rows = idx.shape[0]
    if hard:
        for name in ("cross_entropy", "kl", "entropy_pearson", "entropy_spearman"):
            out[name] = np.full(n_rows, np.nan)
        return out
    human = items["human_entropy"][idx]
    model = items["model_entropy"][idx]
    out["entropy_pearson"] = _row_pearson(model, human)
    human_ranks = np.vstack([rankdata(row) for row in human])
    model_ranks = np.vstack([rankdata(row) for row in model])
    out["entropy_spearman"] = _row_pearson(model_ranks, human_ranks)
    return out


def _percentiles(samples: np.ndarray, level: float) -> tuple[float, float]:
    kept = samples[~np.isnan(samples)]
    if not kept.size:
        return float("nan"), float("nan")
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(kept, [tail, 100.0 - tail])
    return float(low), float(high)


def _check(n_boot: int, level: float) -> None:
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")


def _subset(items: dict[str, FloatArray], keep: np.ndarray) -> dict[str, FloatArray]:
    return {name: column[keep] for name, column in items.items()}


def bootstrap_metrics(
    items: dict[str, FloatArray],
    *,
    hard: bool = False,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Interval]:
    """Interval for every metric from resampling items with replacement.

    ``items`` is the output of :func:`softeval.metrics.item_metrics`. Annotators are not
    resampled, so the interval covers the choice of items only.
    """
    _check(n_boot, level)
    kept = _subset(items, _annotated(items))
    n_items = kept["human_entropy"].size
    point = aggregate(kept, hard=hard)
    if n_items == 0:
        return {name: Interval(float("nan"), float("nan"), float("nan")) for name in ALL_METRICS}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[np.ndarray]] = {name: [] for name in ALL_METRICS}
    for idx in _index_chunks(n_items, n_boot, rng):
        for name, values in _resampled(kept, hard, idx).items():
            draws[name].append(values)
    out = {}
    for name in ALL_METRICS:
        low, high = _percentiles(np.concatenate(draws[name]), level)
        out[name] = Interval(point[name], low, high)
    return out


def paired_bootstrap(
    items_a: dict[str, FloatArray],
    items_b: dict[str, FloatArray],
    *,
    hard_a: bool = False,
    hard_b: bool = False,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, PairedDifference]:
    """Difference A minus B on every metric, resampling the same items for both models."""
    _check(n_boot, level)
    if items_a["human_entropy"].shape != items_b["human_entropy"].shape:
        raise ValueError("the two models must be scored on the same items")
    keep = _annotated(items_a) & _annotated(items_b)
    kept_a = _subset(items_a, keep)
    kept_b = _subset(items_b, keep)
    n_items = int(keep.sum())
    point_a = aggregate(kept_a, hard=hard_a)
    point_b = aggregate(kept_b, hard=hard_b)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[np.ndarray]] = {name: [] for name in ALL_METRICS}
    if n_items:
        for idx in _index_chunks(n_items, n_boot, rng):
            res_a = _resampled(kept_a, hard_a, idx)
            res_b = _resampled(kept_b, hard_b, idx)
            for name in ALL_METRICS:
                draws[name].append(res_a[name] - res_b[name])
    out = {}
    for name in ALL_METRICS:
        diffs = np.concatenate(draws[name]) if draws[name] else np.array([])
        low, high = _percentiles(diffs, level)
        finite = diffs[~np.isnan(diffs)]
        if finite.size:
            better = finite > 0 if name in HIGHER_IS_BETTER else finite < 0
            p_better = float(better.mean())
        else:
            p_better = float("nan")
        out[name] = PairedDifference(
            a=point_a[name],
            b=point_b[name],
            diff=point_a[name] - point_b[name],
            low=low,
            high=high,
            p_a_better=p_better,
        )
    return out

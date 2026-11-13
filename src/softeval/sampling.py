"""Drawing annotators: subsets of the ones an item has, and fresh ones from a distribution."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from softeval.metrics import FloatArray, as_counts


def integer_counts(counts: ArrayLike) -> NDArray[np.int64]:
    """Counts as integers; resampling annotators needs whole annotations."""
    c = as_counts(counts)
    rounded = np.rint(c)
    if not np.allclose(c, rounded, atol=1e-9):
        raise ValueError("counts must be whole numbers to resample annotators")
    return rounded.astype(np.int64)


def _annotation_matrix(c: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """One row per item listing each annotation's label index, padded to the longest row."""
    totals = c.sum(axis=1)
    width = int(totals.max()) if totals.size else 0
    position = np.arange(width)
    edges = np.cumsum(c, axis=1)
    labels = (position[None, :, None] >= edges[:, None, :]).sum(axis=2)
    valid = position[None, :] < totals[:, None]
    return labels, valid


def _count_selected(
    labels: NDArray[np.int64], selected: NDArray[np.bool_], n_labels: int
) -> FloatArray:
    out = np.zeros((labels.shape[0], n_labels), dtype=np.float64)
    for k in range(n_labels):
        out[:, k] = ((labels == k) & selected).sum(axis=1)
    return out


def _random_ranks(valid: NDArray[np.bool_], rng: np.random.Generator) -> NDArray[np.int64]:
    """A random order of each row's valid positions (padding is ranked last)."""
    keys = rng.random(valid.shape)
    keys[~valid] = np.inf
    order = np.argsort(keys, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(valid.shape[0])[:, None]
    ranks[rows, order] = np.arange(valid.shape[1])[None, :]
    return ranks


def subsample(counts: ArrayLike, m: int, rng: np.random.Generator) -> FloatArray:
    """Keep ``m`` annotators per item, drawn without replacement.

    Items with ``m`` or fewer annotations keep all of them.
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    c = integer_counts(counts)
    labels, valid = _annotation_matrix(c)
    ranks = _random_ranks(valid, rng)
    return _count_selected(labels, valid & (ranks < m), c.shape[1])


def split_halves(counts: ArrayLike, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Split each item's annotators at random into two halves of ``n // 2`` annotators.

    With an odd number of annotators one is left out of both halves.
    """
    c = integer_counts(counts)
    labels, valid = _annotation_matrix(c)
    ranks = _random_ranks(valid, rng)
    half = (c.sum(axis=1) // 2)[:, None]
    first = valid & (ranks < half)
    second = valid & (ranks >= half) & (ranks < 2 * half)
    k = c.shape[1]
    return _count_selected(labels, first, k), _count_selected(labels, second, k)


def sample_annotators(dist: ArrayLike, m: ArrayLike, rng: np.random.Generator) -> FloatArray:
    """Label counts of ``m`` fresh annotators drawn independently from each row of ``dist``.

    ``m`` is one number or one number per item. Rows of ``dist`` that are ``nan`` (items with no
    annotations) give a row of zeros.
    """
    p = np.asarray(dist, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("dist must be 2-D (items by labels)")
    sizes = np.broadcast_to(np.asarray(m, dtype=np.int64), (p.shape[0],)).copy()
    if np.any(sizes < 0):
        raise ValueError("m must not be negative")
    missing = np.isnan(p).any(axis=1)
    safe = np.where(missing[:, None], 1.0 / p.shape[1], p)
    safe = safe / safe.sum(axis=1, keepdims=True)
    sizes[missing] = 0
    return rng.multinomial(sizes, safe).astype(np.float64)

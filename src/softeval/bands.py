"""Group items by how much the annotators agree."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from softeval.metrics import (
    DEFAULT_EPS,
    TiePolicy,
    aggregate,
    as_counts,
    distributions,
    entropy,
    is_hard,
    item_metrics,
)

BandBy = Literal["share", "entropy"]

#: Default band edges for the share of annotators who chose the most common label.
SHARE_EDGES = (0.0, 0.5, 0.7, 0.9, 1.0)
#: Default band edges for the entropy of the human distribution divided by ``log K``.
ENTROPY_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class Band:
    """Scores of the items whose agreement falls between ``low`` and ``high``."""

    low: float
    high: float
    closed_high: bool
    n_items: int
    metrics: dict[str, float]

    @property
    def label(self) -> str:
        right = "]" if self.closed_high else ")"
        return f"[{self.low:g}, {self.high:g}{right}"

    def to_dict(self) -> dict[str, object]:
        return {
            "band": self.label,
            "low": self.low,
            "high": self.high,
            "n_items": self.n_items,
            "metrics": self.metrics,
        }


def agreement(counts: ArrayLike, by: BandBy = "share") -> np.ndarray:
    """Agreement statistic per item: majority share, or entropy divided by ``log K``.

    Items with no annotations give ``nan``.
    """
    c = as_counts(counts)
    dist = distributions(c)
    if by == "share":
        return dist.max(axis=1)
    if by == "entropy":
        return entropy(dist) / math.log(c.shape[1])
    raise ValueError(f"unknown banding {by!r}; use 'share' or 'entropy'")


def band_masks(
    values: np.ndarray, edges: Sequence[float]
) -> list[tuple[float, float, bool, np.ndarray]]:
    """Masks for the half-open intervals between consecutive edges; the last one is closed."""
    e = [float(x) for x in edges]
    if len(e) < 2 or any(b <= a for a, b in pairwise(e)):
        raise ValueError("edges must be at least two increasing numbers")
    out = []
    for i, (low, high) in enumerate(pairwise(e)):
        last = i == len(e) - 2
        tol = 1e-12
        mask = (values >= low - tol) & ((values <= high + tol) if last else (values < high - tol))
        out.append((low, high, last, mask))
    return out


def score_bands(
    counts: ArrayLike,
    probs: ArrayLike,
    *,
    by: BandBy = "share",
    edges: Sequence[float] | None = None,
    ties: TiePolicy = "split",
    eps: float = DEFAULT_EPS,
) -> list[Band]:
    """Every metric within each agreement band.

    The bands are formed from the same annotations that are scored. With few annotations per item
    the extreme bands collect items that only look unanimous or split by chance, so banded numbers
    from fewer than about ten annotations should be read with that in mind.
    """
    if edges is None:
        edges = SHARE_EDGES if by == "share" else ENTROPY_EDGES
    items = item_metrics(counts, probs, ties=ties, eps=eps)
    hard = is_hard(probs)
    values = agreement(counts, by)
    bands = []
    for low, high, last, mask in band_masks(values, edges):
        subset = {name: column[mask] for name, column in items.items()}
        bands.append(
            Band(
                low=low,
                high=high,
                closed_high=last,
                n_items=int(mask.sum()),
                metrics=aggregate(subset, hard=hard),
            )
        )
    return bands

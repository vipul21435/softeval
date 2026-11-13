"""Simulation: how far each estimate is from the truth at a given number of annotations.

Items get a population label distribution drawn from a symmetric Dirichlet, and a model whose
prediction is that distribution mixed with noise. Because the population distribution is known,
every metric has a true value. Annotators are then drawn from the population distribution and each
estimator is compared with the truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from softeval.finite import CORRECTABLE, jackknife_items, miller_madow_kl, split_half_items
from softeval.metrics import (
    FloatArray,
    aggregate,
    item_metrics,
    miller_madow_entropy,
    pearson,
    spearman,
)
from softeval.sampling import sample_annotators

DEFAULT_ANNOTATIONS = (3, 5, 10, 100)


@dataclass
class SimulationRow:
    """One estimator of one metric at one number of annotations."""

    metric: str
    estimator: str
    annotations: int
    truth: float
    mean: float
    bias: float
    rmse: float

    @property
    def relative_bias(self) -> float:
        return self.bias / abs(self.truth) if self.truth else float("nan")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "estimator": self.estimator,
            "annotations": self.annotations,
            "truth": self.truth,
            "mean": self.mean,
            "bias": self.bias,
            "rmse": self.rmse,
            "relative_bias": self.relative_bias,
        }


@dataclass
class SimulationResult:
    """All rows of a simulation with the settings that produced them."""

    settings: dict[str, object]
    truth: dict[str, float]
    rows: list[SimulationRow] = field(default_factory=list)

    def trusted_from(self, metric: str, estimator: str, tolerance: float) -> int | None:
        """Smallest simulated number of annotations from which the bias stays within tolerance.

        The bias must be within ``tolerance`` times the true value there and at every larger
        simulated number of annotations. ``None`` when that never happens.
        """
        rows = sorted(
            (r for r in self.rows if r.metric == metric and r.estimator == estimator),
            key=lambda r: r.annotations,
        )
        answer = None
        for row in reversed(rows):
            if abs(row.bias) <= tolerance * abs(row.truth):
                answer = row.annotations
            else:
                break
        return answer

    def to_dict(self) -> dict[str, object]:
        return {
            "settings": self.settings,
            "truth": self.truth,
            "rows": [row.to_dict() for row in self.rows],
        }


def make_population(
    n_items: int, n_labels: int, alpha: float, noise: float, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Population label distributions and a model that is ``noise`` away from them."""
    if n_items < 2 or n_labels < 2:
        raise ValueError("need at least two items and two labels")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be between 0 and 1")
    human = rng.dirichlet(np.full(n_labels, alpha), size=n_items)
    human = np.maximum(human, 1e-12)
    human = human / human.sum(axis=1, keepdims=True)
    other = rng.dirichlet(np.ones(n_labels), size=n_items)
    model = (1.0 - noise) * human + noise * other
    return human, model


def _estimates(
    counts: FloatArray, model: FloatArray, split_draws: int, rng: np.random.Generator
) -> dict[tuple[str, str], float]:
    items = item_metrics(counts, model)
    plug_in = aggregate(items)
    out = {(name, "plug-in"): value for name, value in plug_in.items()}
    jack = jackknife_items(counts, model)
    split = split_half_items(counts, model, n_splits=split_draws, seed=rng)
    for name in CORRECTABLE:
        out[(name, "jackknife")] = float(np.mean(jack[name]))
        out[(name, "split-half")] = float(np.mean(split[name]))
    out[("kl", "Miller-Madow")] = float(np.mean(miller_madow_kl(counts, model)))
    corrected_entropy = miller_madow_entropy(counts)
    out[("entropy_pearson", "Miller-Madow")] = pearson(items["model_entropy"], corrected_entropy)
    out[("entropy_spearman", "Miller-Madow")] = spearman(items["model_entropy"], corrected_entropy)
    return out


def simulate(
    *,
    n_labels: int = 3,
    n_items: int = 2000,
    annotations: Sequence[int] = DEFAULT_ANNOTATIONS,
    replicates: int = 20,
    alpha: float = 0.5,
    noise: float = 0.3,
    split_draws: int = 10,
    seed: int = 0,
) -> SimulationResult:
    """Compare every estimator with the truth at each number of annotations per item."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    sizes = [int(n) for n in annotations]
    if not sizes or min(sizes) < 2:
        raise ValueError("annotations must be numbers of at least 2")
    rng = np.random.default_rng(seed)
    human, model = make_population(n_items, n_labels, alpha, noise, rng)
    truth = aggregate(item_metrics(human, model))
    result = SimulationResult(
        settings={
            "n_labels": n_labels,
            "n_items": n_items,
            "annotations": sizes,
            "replicates": replicates,
            "alpha": alpha,
            "noise": noise,
            "split_draws": split_draws,
            "seed": seed,
        },
        truth=truth,
    )
    for n in sizes:
        runs: dict[tuple[str, str], list[float]] = {}
        for _ in range(replicates):
            counts = sample_annotators(human, n, rng)
            for key, value in _estimates(counts, model, split_draws, rng).items():
                runs.setdefault(key, []).append(value)
        for (metric, estimator), values in runs.items():
            arr = np.asarray(values, dtype=np.float64)
            target = truth[metric]
            result.rows.append(
                SimulationRow(
                    metric=metric,
                    estimator=estimator,
                    annotations=n,
                    truth=target,
                    mean=float(arr.mean()),
                    bias=float(arr.mean() - target),
                    rmse=float(np.sqrt(np.mean((arr - target) ** 2))),
                )
            )
    return result

import math

import numpy as np
import pytest

from softeval import bootstrap
from softeval import metrics as M


def _population(seed, n_items=400, noise=0.3):
    rng = np.random.default_rng(seed)
    human = rng.dirichlet(np.ones(3), size=n_items)
    counts = rng.multinomial(20, human).astype(float)
    model = (1 - noise) * human + noise * rng.dirichlet(np.ones(3), size=n_items)
    return counts, model


def test_interval_brackets_the_point_estimate_and_is_reproducible():
    counts, model = _population(0)
    items = M.item_metrics(counts, model)
    first = bootstrap.bootstrap_metrics(items, n_boot=300, seed=7)
    again = bootstrap.bootstrap_metrics(items, n_boot=300, seed=7)
    point = M.aggregate(items)
    for name in M.ALL_METRICS:
        assert first[name].low <= point[name] <= first[name].high
        assert first[name].value == pytest.approx(point[name])
        assert first[name].to_dict() == again[name].to_dict()


def test_interval_width_matches_the_normal_approximation_for_a_mean():
    rng = np.random.default_rng(1)
    n = 2000
    first = rng.binomial(1, 0.3, size=n).astype(float)
    counts = np.column_stack([first, 1 - first])
    probs = np.tile([1.0, 0.0], (n, 1))
    items = M.item_metrics(counts, probs)
    out = bootstrap.bootstrap_metrics(items, hard=True, n_boot=2000, seed=2)["annotator_match"]
    expected_width = 2 * 1.96 * math.sqrt(0.3 * 0.7 / n)
    # percentile bootstrap of a mean of 2000 values: within 15 % of the normal-theory width
    assert out.high - out.low == pytest.approx(expected_width, rel=0.15)
    assert math.isnan(bootstrap.bootstrap_metrics(items, hard=True, n_boot=50)["cross_entropy"].low)


def test_more_items_give_a_narrower_interval():
    small = M.item_metrics(*_population(3, n_items=100))
    large = M.item_metrics(*_population(3, n_items=1600))
    w_small = bootstrap.bootstrap_metrics(small, n_boot=300)["total_variation"]
    w_large = bootstrap.bootstrap_metrics(large, n_boot=300)["total_variation"]
    assert (w_large.high - w_large.low) < 0.5 * (w_small.high - w_small.low)


def test_empty_rows_are_ignored_and_all_empty_gives_nan():
    counts = np.array([[3, 1, 0], [0, 0, 0], [1, 1, 2]], dtype=float)
    probs = np.array([[0.6, 0.3, 0.1]] * 3)
    out = bootstrap.bootstrap_metrics(M.item_metrics(counts, probs), n_boot=50)
    assert math.isfinite(out["total_variation"].low)
    none = bootstrap.bootstrap_metrics(M.item_metrics(counts[1:2], probs[1:2]), n_boot=10)
    assert math.isnan(none["total_variation"].value)


def test_paired_difference_of_identical_models_is_zero():
    counts, model = _population(4)
    items = M.item_metrics(counts, model)
    out = bootstrap.paired_bootstrap(items, items, n_boot=100)
    for name in M.ALL_METRICS:
        assert out[name].diff == pytest.approx(0.0, abs=1e-12)
        assert out[name].low == pytest.approx(0.0, abs=1e-12)
        assert out[name].p_a_better == pytest.approx(0.0)


def test_paired_comparison_finds_the_closer_model_and_respects_direction():
    counts, close = _population(5, noise=0.1)
    _, far = _population(5, noise=0.6)
    a = M.item_metrics(counts, close)
    b = M.item_metrics(counts, far)
    out = bootstrap.paired_bootstrap(a, b, n_boot=300, seed=1)
    assert out["total_variation"].diff < 0 and out["total_variation"].high < 0
    assert out["total_variation"].p_a_better == pytest.approx(1.0)
    assert out["annotator_match"].diff > 0
    assert out["annotator_match"].p_a_better > 0.95
    flipped = bootstrap.paired_bootstrap(b, a, n_boot=300, seed=1)
    assert flipped["total_variation"].p_a_better == pytest.approx(0.0)
    assert flipped["total_variation"].diff == pytest.approx(-out["total_variation"].diff)
    assert set(out["kl"].to_dict()) == {"a", "b", "diff", "low", "high", "p_a_better"}


def test_pairing_narrows_the_interval_for_similar_models():
    counts, model = _population(6)
    rng = np.random.default_rng(6)
    other = 0.95 * model + 0.05 * rng.dirichlet(np.ones(3), size=len(model))
    a = M.item_metrics(counts, model)
    b = M.item_metrics(counts, other)
    paired = bootstrap.paired_bootstrap(a, b, n_boot=300)["total_variation"]
    single = bootstrap.bootstrap_metrics(a, n_boot=300)["total_variation"]
    assert (paired.high - paired.low) < 0.5 * (single.high - single.low)


def test_hard_model_in_a_pair_and_bad_arguments():
    counts, model = _population(7, n_items=50)
    hard = M.one_hot(model.argmax(axis=1), 3)
    a = M.item_metrics(counts, model)
    b = M.item_metrics(counts, hard)
    out = bootstrap.paired_bootstrap(a, b, hard_b=True, n_boot=50)
    assert math.isnan(out["cross_entropy"].diff) and math.isnan(out["cross_entropy"].p_a_better)
    assert math.isfinite(out["total_variation"].diff)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap.bootstrap_metrics(a, n_boot=0)
    with pytest.raises(ValueError, match="level"):
        bootstrap.bootstrap_metrics(a, level=1.5)
    with pytest.raises(ValueError, match="same items"):
        bootstrap.paired_bootstrap(a, M.item_metrics(counts[:10], model[:10]))


def test_constant_prediction_has_no_entropy_correlation():
    counts, _ = _population(8, n_items=120)
    constant = np.tile([0.5, 0.3, 0.2], (120, 1))
    items = M.item_metrics(counts, constant)
    assert math.isnan(M.aggregate(items)["entropy_pearson"])
    assert math.isnan(M.aggregate(items)["entropy_spearman"])
    out = bootstrap.bootstrap_metrics(items, n_boot=40)
    assert math.isnan(out["entropy_pearson"].low) and math.isnan(out["entropy_pearson"].value)
    assert math.isfinite(out["total_variation"].low)

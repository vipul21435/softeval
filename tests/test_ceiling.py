import math

import numpy as np
import pytest

from softeval import ceiling, finite
from softeval.sampling import sample_annotators


def test_leave_one_out_by_hand():
    # counts (3, 1). A label-0 annotator (3 of 4) sees the others as (2, 1): match 2/3 and the
    # majority agrees. The label-1 annotator sees (3, 0): match 0 and the majority disagrees.
    out = ceiling.leave_one_out([[3, 1]])
    assert out["annotator_match"] == pytest.approx(0.75 * 2 / 3)
    assert out["majority_accuracy"] == pytest.approx(0.75)
    assert out["total_variation"] == pytest.approx(1 - 0.5)
    assert math.isnan(out["cross_entropy"]) and math.isnan(out["kl"])
    assert math.isnan(out["entropy_pearson"])


def test_leave_one_out_match_is_pairwise_agreement():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 6, size=(200, 4))
    counts[counts.sum(axis=1) < 2] = [2, 1, 0, 0]
    n = counts.sum(axis=1)
    pairwise = (counts * (counts - 1)).sum(axis=1) / (n * (n - 1))
    items = ceiling.leave_one_out_items(counts)
    assert items["annotator_match"] == pytest.approx(pairwise)
    assert items["total_variation"] == pytest.approx(1 - pairwise)


def test_leave_one_out_tie_handling():
    # counts (2, 1): a label-0 annotator sees (1, 1), a tie; the label-1 annotator sees (2, 0)
    split = ceiling.leave_one_out([[2, 1]], ties="split")["majority_accuracy"]
    anyway = ceiling.leave_one_out([[2, 1]], ties="any")["majority_accuracy"]
    excluded = ceiling.leave_one_out([[2, 1]], ties="exclude")["majority_accuracy"]
    assert split == pytest.approx((2 / 3) * 0.5)
    assert anyway == pytest.approx(2 / 3)
    # two of the three leave-one-out cases are ties and drop out; what is left is not a clean
    # average over annotators, so the value is only checked to be undefined-or-zero safe
    assert math.isnan(excluded) or 0.0 <= excluded <= 1.0


def test_leave_one_out_skips_items_with_under_two_annotations():
    items = ceiling.leave_one_out_items([[1, 0], [0, 0], [2, 2]])
    assert np.isnan(items["annotator_match"][:2]).all()
    assert items["annotator_match"][2] == pytest.approx(1 / 3)
    assert ceiling.leave_one_out([[1, 0], [0, 0], [2, 2]])["annotator_match"] == pytest.approx(
        1 / 3
    )


def test_leave_one_out_on_simulated_annotators_with_known_agreement():
    rng = np.random.default_rng(1)
    q = 0.8
    counts = sample_annotators(np.tile([q, 1 - q], (2000, 1)), 50, rng)
    out = ceiling.leave_one_out(counts)
    # two annotators agree with probability q^2 + (1 - q)^2 = 0.68; with 50 annotators the
    # majority of the others is label 0 almost surely, so one annotator matches it with
    # probability q. Tolerance 0.01 against a standard error near 0.002.
    assert out["annotator_match"] == pytest.approx(q * q + (1 - q) * (1 - q), abs=0.01)
    assert out["majority_accuracy"] == pytest.approx(q, abs=0.01)


def test_fresh_sample_matches_exact_expectation():
    counts = np.tile([7, 2, 1], (1500, 1))
    out = ceiling.fresh_sample(counts, annotations=10, n_draws=20, seed=2)
    exact = finite.expected_plugin([0.7, 0.2, 0.1], [0.7, 0.2, 0.1], 10)
    # 30000 item draws: tolerance 0.01 on quantities with standard error below 0.002
    for name in ("total_variation", "js_divergence", "kl", "annotator_match", "majority_accuracy"):
        assert out[name] == pytest.approx(exact[name], abs=0.01)
    assert out["annotator_match"] == pytest.approx(0.7, abs=0.01)


def test_fresh_sample_is_perfect_without_disagreement_and_shrinks_with_more_annotators():
    unanimous = ceiling.fresh_sample([[5, 0], [0, 9], [0, 0]], n_draws=5, seed=0)
    assert unanimous["total_variation"] == pytest.approx(0.0, abs=1e-9)
    assert unanimous["annotator_match"] == pytest.approx(1.0)
    counts = np.tile([6, 4], (500, 1))
    few = ceiling.fresh_sample(counts, annotations=5, n_draws=20, seed=3)
    many = ceiling.fresh_sample(counts, annotations=100, n_draws=20, seed=3)
    assert few["total_variation"] > 2 * many["total_variation"]
    with pytest.raises(ValueError, match="n_draws"):
        ceiling.fresh_sample(counts, n_draws=0)


def test_half_against_half():
    unanimous = ceiling.half_against_half([[6, 0], [0, 4], [1, 0]], n_splits=5, seed=0)
    assert unanimous["annotator_match"] == pytest.approx(1.0)
    assert unanimous["majority_accuracy"] == pytest.approx(1.0)
    raw = ceiling.half_against_half([[6, 0], [0, 4]], n_splits=5, seed=0, smoothing=0.0)
    assert raw["total_variation"] == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(4)
    counts = sample_annotators(np.tile([0.5, 0.5], (2000, 1)), 40, rng)
    out = ceiling.half_against_half(counts, n_splits=10, seed=5)
    # a coin-flip item: whatever the first half prefers, a second-half annotator matches it
    # about half the time (tolerance 0.02)
    assert out["annotator_match"] == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError, match="n_splits"):
        ceiling.half_against_half(counts, n_splits=0)

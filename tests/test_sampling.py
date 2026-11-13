import numpy as np
import pytest

from softeval import sampling


def test_subsample_keeps_m_annotators_and_never_invents_labels():
    rng = np.random.default_rng(0)
    counts = np.array([[60, 30, 10], [0, 3, 0], [2, 0, 1], [0, 0, 0]])
    out = sampling.subsample(counts, 5, rng)
    assert out.sum(axis=1).tolist() == [5, 3, 3, 0]
    assert np.all(out <= counts)
    assert out[1].tolist() == [0, 3, 0]
    with pytest.raises(ValueError, match="at least 1"):
        sampling.subsample(counts, 0, rng)


def test_subsample_is_unbiased_for_the_distribution():
    rng = np.random.default_rng(1)
    counts = np.tile([60, 30, 10], (4000, 1))
    mean = sampling.subsample(counts, 5, rng).mean(axis=0) / 5
    # standard error of each share is below 0.004 with 4000 items of 5 annotations
    assert mean == pytest.approx([0.6, 0.3, 0.1], abs=0.015)


def test_split_halves_are_disjoint_and_equal_sized():
    rng = np.random.default_rng(2)
    counts = np.array([[5, 4, 0], [1, 0, 0], [3, 3, 2], [0, 0, 0]])
    first, second = sampling.split_halves(counts, rng)
    assert first.sum(axis=1).tolist() == [4, 0, 4, 0]
    assert second.sum(axis=1).tolist() == [4, 0, 4, 0]
    assert np.all(first + second <= counts)


def test_sample_annotators_follows_the_distribution():
    rng = np.random.default_rng(3)
    dist = np.tile([0.7, 0.2, 0.1], (3000, 1))
    out = sampling.sample_annotators(dist, 10, rng)
    assert np.all(out.sum(axis=1) == 10)
    assert out.mean(axis=0) / 10 == pytest.approx([0.7, 0.2, 0.1], abs=0.01)
    sizes = np.array([1, 4])
    varied = sampling.sample_annotators([[0.5, 0.5], [0.5, 0.5]], sizes, rng)
    assert varied.sum(axis=1).tolist() == [1, 4]


def test_sample_annotators_gives_zeros_for_items_without_a_distribution():
    rng = np.random.default_rng(4)
    out = sampling.sample_annotators([[np.nan, np.nan], [1.0, 0.0]], 6, rng)
    assert out.tolist() == [[0, 0], [6, 0]]
    with pytest.raises(ValueError, match="negative"):
        sampling.sample_annotators([[0.5, 0.5]], -1, rng)
    with pytest.raises(ValueError, match="2-D"):
        sampling.sample_annotators([0.5, 0.5], 1, rng)


def test_fractional_counts_are_rejected():
    with pytest.raises(ValueError, match="whole numbers"):
        sampling.integer_counts([[0.5, 1.5]])
    assert sampling.integer_counts([[2.0, 1.0]]).tolist() == [[2, 1]]
